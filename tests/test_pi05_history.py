"""Causal RGB/state windows, real processor/model adapters, and checkpoint contracts."""
import copy
import json
import shutil
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

SUBMODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUBMODULE))
from excavator_vlm.data.contract import CAMERAS
from excavator_vlm.policies.pi05.history import (
    MASK, ObservationHistory, bind_history, history_spec, make_history, observation_batch, window_mask,
)
from excavator_vlm.policies.pi05.config import build_policy_config, resolve_settings
from excavator_vlm.policies.pi05.processors import make_processors
from excavator_vlm.policies.pi05.adapter import policy_class, training_batch
from test_pi05 import cpu_settings, toy_tokenizer

torch.set_num_threads(2)


def current(step):
    return {'observation.state': torch.full((1, 28), step / 20), 'task': ['Dig five scoops.'],
            **{key: torch.full((1, 3, 256, 256), (step + camera) / 20)
               for camera, key in enumerate(CAMERAS)}}


def window(step=6):
    buffer = ObservationHistory(make_history([-6, -3, 0]))
    for index in range(step + 1):
        buffer.append(current(index), index)
    return buffer.batch()


def config_and_processors(directory):
    settings = cpu_settings(directory)
    settings['policy'].update(n_obs_steps=3, tokenizer_max_length=512, gradient_checkpointing=False)
    settings['observation_history'] = {'delta_indices': [-6, -3, 0]}
    toy_tokenizer(settings['assets']['tokenizer_path'])
    config = build_policy_config(settings)
    stats = {key: {'q01': [0.] * size, 'q99': [1.] * size}
             for key, size in [('observation.state', 28), ('action', 4)]}
    pre, post = make_processors(config, stats, settings['assets']['tokenizer_path'])
    return config, pre, post, settings, stats


class WindowTests(unittest.TestCase):
    def test_offsets_and_masks_match_offline_episode_boundaries(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        spec = make_history([-6, -3, 0])
        buffer = ObservationHistory(spec)
        # Exercise LeRobot's real index/padding logic at a nonzero parent offset.
        offline = object.__new__(LeRobotDataset)
        offline.meta = SimpleNamespace(episodes=[{'dataset_from_index': 100, 'dataset_to_index': 110}])
        offline.delta_indices = {key: spec['delta_indices'] for key in ('observation.state', *CAMERAS)}
        for step in range(10):
            original = current(step)
            buffer.append(original, step)
            original['observation.state'].fill_(999) # The capture must own its tensor storage.
            batch = buffer.batch()
            indices, pads = offline._get_query_indices(100 + step, 0)
            expected_steps = [i - 100 for i in indices['observation.state']]
            torch.testing.assert_close(batch['observation.state'][0, :, 0], torch.tensor(expected_steps) / 20)
            torch.testing.assert_close(batch[MASK][0], pads['observation.state_is_pad'])
            for camera, key in enumerate(CAMERAS):
                torch.testing.assert_close(batch[key][0, :, 0, 0, 0], (torch.tensor(expected_steps) + camera) / 20)
        buffer.reset()
        buffer.append(current(0), 0)
        self.assertEqual(buffer.batch()[MASK].tolist(), [[True, True, False]])

    def test_no_skipped_steps_future_offsets_or_cross_camera_padding(self):
        buffer = ObservationHistory(make_history([-6, -3, 0]))
        with self.assertRaises(ValueError):
            buffer.append(current(0), 3)
        for offsets in ([0, 1], [-3, -3, 0], [0, -3], [-1], [-101, 0]):
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                make_history(offsets)
        batch = window(0)
        batch[f'{CAMERAS[0]}_is_pad'] = torch.zeros(1, 3, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, 'agree'):
            window_mask(batch, make_history([-6, -3, 0]))

    def test_dataset_queries_only_past_observations_and_future_actions(self):
        from excavator_vlm.policies.pi05.dataset import make_pi05_dataset
        with patch('lerobot.datasets.lerobot_dataset.LeRobotDataset') as factory, \
                patch('excavator_vlm.policies.pi05.dataset.load_info', return_value={'fps': 10}):
            make_pi05_dataset('local/test', '/unused', [2], 10, 'pyav', make_history([-6, -3, 0]))
        delta = factory.call_args.kwargs['delta_timestamps']
        for key in ('observation.state', *CAMERAS):
            self.assertEqual(delta[key], [-.6, -.3, 0.])
        self.assertEqual(delta['action'], [i / 10 for i in range(10)])

    def test_c_config_is_opt_in_and_requires_explicit_matching_offsets(self):
        settings = resolve_settings(SUBMODULE / 'configs/pi05/history_c.yaml')
        self.assertEqual(settings['policy']['n_obs_steps'], 3)
        self.assertEqual(settings['training']['mode'], 'vlm_lora_expert_full')
        for overrides in ({'policy': {'n_obs_steps': 3}},
                          {'observation_history': {'delta_indices': [-6, 0]}},
                          {'observation_history': {'delta_indices': [-6, -3, 0]},
                           'policy': {'n_obs_steps': 3, 'tokenizer_max_length': 200}}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                resolve_settings(SUBMODULE / 'configs/pi05/full.yaml', overrides)

    def test_local_video_dataset_matches_online_windows_without_parent_leakage(self):
        from excavator_vlm.policies.pi05.dataset import make_pi05_dataset
        import datasets.config
        source = SUBMODULE.parent / 'data/pi05_singlecycle_physx_pilot_v2'
        if not (source / 'meta/info.json').is_file():
            self.skipTest('Optional local video integration dataset is not installed')
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(datasets.config, 'HF_DATASETS_CACHE', Path(directory) / 'cache'):
            root = Path(directory) / 'dataset'
            for name in ('meta', 'data'):
                shutil.copytree(source / name, root / name)
            # Decode original media read-only; metadata, tables and caches are temporary.
            (root / 'videos').symlink_to((source / 'videos').resolve(), target_is_directory=True)
            spec = make_history([-6, -3, 0])
            single = make_pi05_dataset('local/excavator', root, [20], 10)
            history = make_pi05_dataset('local/excavator', root, [20], 10, 'pyav', spec)
            buffer = ObservationHistory(spec)
            for step in range(7):
                frame = single[step]
                self.assertEqual(int(frame['episode_index']), 20)
                buffer.append({key: frame[key].unsqueeze(0) for key in ('observation.state', *CAMERAS)}
                              | {'task': [frame['task']]}, step)
                offline, online = history[step], buffer.batch()
                for key in ('observation.state', *CAMERAS):
                    torch.testing.assert_close(offline[key], online[key][0], rtol=0, atol=0)
                    torch.testing.assert_close(offline[f'{key}_is_pad'], online[MASK][0])


class TinyCore(torch.nn.Module):
    """Synthetic action head exercising the real PI05 image adapter, not policy quality."""
    def __init__(self, config, **kwargs):
        super().__init__()
        self.projection = torch.nn.Linear(1, 32)
        self.config = config

    def sample_actions(self, images, masks, tokens, token_masks, **kwargs):
        features = sum((i + 1) * image.mean((1, 2, 3)) * mask for i, (image, mask) in
                       enumerate(zip(images, masks)))[:, None]
        return self.projection(features)[:, None].expand(-1, self.config.chunk_size, -1)

    def forward(self, images, masks, tokens, token_masks, actions, **kwargs):
        return (self.sample_actions(images, masks, tokens, token_masks) - actions).square()


class TinyWindowDataset(torch.utils.data.Dataset):
    def __init__(self, episodes):
        self.episode = episodes[0]

    def __len__(self):
        return 8

    def __getitem__(self, index):
        batch = window(index)
        item = {key: value[0] if torch.is_tensor(value) else value[0] for key, value in batch.items()}
        item.update(action=torch.full((10, 4), .1), action_is_pad=torch.zeros(10, dtype=torch.bool),
                    action_loss_weight=torch.ones(10), **{'observation.recovery_active': torch.zeros(10)},
                    episode_index=torch.tensor(self.episode), frame_index=torch.tensor(index))
        return item


class ProcessorModelTests(unittest.TestCase):
    def test_history_prompt_uses_normalized_past_states_and_padding_without_subtasks(self):
        from lerobot.processor.core import TransitionKey
        from excavator_vlm.policies.pi05.history_processor import HistoryPromptProcessorStep
        transition = {TransitionKey.OBSERVATION: {'observation.state': torch.zeros(1, 3, 28)},
                      TransitionKey.COMPLEMENTARY_DATA: {'task': ['Dig five scoops.'],
                                                        MASK: torch.tensor([[True, False, False]])}}
        prompt = HistoryPromptProcessorStep()(transition)[TransitionKey.COMPLEMENTARY_DATA]['task'][0]
        self.assertIn('t=-0.6s: padding', prompt)
        self.assertIn('t=-0.3s: ' + ' '.join(['128'] * 32), prompt)
        self.assertNotIn('Subtask', prompt)
        self.assertNotEqual(prompt, HistoryPromptProcessorStep()({**transition,
            TransitionKey.OBSERVATION: {'observation.state': torch.ones(1, 3, 28)}})
            [TransitionKey.COMPLEMENTARY_DATA]['task'][0])

    def test_training_inference_and_saved_processor_agree_and_keep_action_timestamps(self):
        from lerobot.processor import PolicyProcessorPipeline
        with tempfile.TemporaryDirectory() as directory:
            config, pre, post, _, _ = config_and_processors(directory)
            batch = window(3)
            batch.update(action=torch.full((1, 10, 4), .5),
                         action_is_pad=torch.zeros(1, 10, dtype=torch.bool),
                         action_loss_weight=torch.ones(1, 10))
            training = training_batch(pre, copy.deepcopy(batch))
            inference = pre(observation_batch(copy.deepcopy(batch)))
            for key in ('observation.state', *CAMERAS, MASK, 'observation.language.tokens'):
                torch.testing.assert_close(training[key], inference[key])
            torch.testing.assert_close(training['action'], torch.zeros(1, 10, 4))
            torch.testing.assert_close(post(training['action'][:, 0]), torch.full((1, 4), .5))
            pre.save_pretrained(Path(directory) / 'processor')
            restored = PolicyProcessorPipeline.from_pretrained(Path(directory) / 'processor',
                                                               config_filename='policy_preprocessor.json')
            torch.testing.assert_close(restored(copy.deepcopy(batch))['observation.language.tokens'],
                                       training['observation.language.tokens'])
            with self.assertRaises(ValueError):
                pre({**window(), 'task': ['long ' * 1000]})

    def test_real_pi05_adapter_uses_all_nine_images_and_masks_initial_padding(self):
        from lerobot.policies.pi05 import modeling_pi05
        with tempfile.TemporaryDirectory() as directory, patch.object(modeling_pi05, 'PI05Pytorch', TinyCore):
            config, pre, _, _, _ = config_and_processors(directory)
            policy = policy_class()(config)
            batch = window()
            images, masks = policy._preprocess_images(pre(copy.deepcopy(batch)))
            self.assertEqual(len(images), 9)
            self.assertEqual([round(float(image.mean()), 3) for image in images],
                             [-1., -.9, -.8, -.7, -.6, -.5, -.4, -.3, -.2])
            result = policy.predict_action_chunk(pre(copy.deepcopy(batch)))
            changed = copy.deepcopy(batch)
            changed[CAMERAS[0]][:, 0].fill_(1.)
            self.assertFalse(torch.equal(result, policy.predict_action_chunk(pre(changed))))
            padded = window(0)
            _, masks = policy._preprocess_images(pre(copy.deepcopy(padded)))
            self.assertEqual([bool(m[0]) for m in masks], [False] * 6 + [True] * 3)
            changed = copy.deepcopy(padded)
            for key in CAMERAS:
                changed[key][:, :2].fill_(1.)
            torch.testing.assert_close(policy.predict_action_chunk(pre(padded)),
                                       policy.predict_action_chunk(pre(changed)))
            batch.update(action=torch.zeros(1, 10, 4), action_is_pad=torch.zeros(1, 10, dtype=torch.bool),
                         action_loss_weight=torch.ones(1, 10))
            loss, _ = policy(training_batch(pre, batch))
            loss.backward()
            self.assertGreater(float(policy.model.projection.weight.grad.abs().sum()), 0.)

    def test_checkpoint_reload_preserves_window_and_rejects_mismatched_stride(self):
        from lerobot.policies.pi05 import modeling_pi05
        from excavator_vlm.policies.pi05 import checkpoints, inference
        from excavator_vlm.policies.pi05.precision import configure_trainable_precision
        with tempfile.TemporaryDirectory() as directory, patch.object(modeling_pi05, 'PI05Pytorch', TinyCore):
            config, pre, post, settings, stats = config_and_processors(directory)
            policy = policy_class()(config)
            configure_trainable_precision(policy)
            checkpoint = checkpoints.save_checkpoint(Path(directory) / 'checkpoint', policy, pre, post,
                stats, settings['assets'], {}, {'step': 1})
            with patch.object(inference, 'check_environment'), patch.object(inference, 'load_base'):
                restored, loaded_pre, _ = inference.load_pi05(checkpoint, 'cpu')
            self.assertEqual(history_spec(restored.config), make_history([-6, -3, 0]))
            torch.testing.assert_close(policy.predict_action_chunk(pre(window())),
                                       restored.predict_action_chunk(loaded_pre(window())))
            bind_history(policy.config, make_history([-4, -2, 0]))
            with self.assertRaisesRegex(ValueError, 'history'):
                checkpoints.apply_delta(policy, checkpoint)
            payload = json.loads((checkpoint / 'checkpoint.json').read_text())
            payload.pop('observation_history')
            (checkpoint / 'checkpoint.json').write_text(json.dumps(payload))
            with patch.object(inference, 'check_environment'), self.assertRaisesRegex(ValueError, 'explicit'):
                inference.load_pi05(checkpoint, 'cpu')

    def test_real_trainer_validation_reload_and_exact_resume_with_history(self):
        import pandas as pd
        from safetensors.torch import load_file
        from lerobot.policies.pi05 import modeling_pi05
        from excavator_vlm.policies.pi05 import checkpoints, inference, train
        frames = pd.DataFrame({'episode_index': [1, 1],
            'observation.state': [np.zeros(28), np.ones(28)], 'action': [np.zeros(4), np.ones(4)]})
        fixture = ({'fingerprint': 'synthetic-window-fixture'},
                   {'train': [1], 'validation': [2], 'test': [3]}, frames, [1])
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(modeling_pi05, 'PI05Pytorch', TinyCore), \
                patch.object(inference, 'check_environment'), \
                patch.object(checkpoints, 'load_base', return_value={'synthetic': True}), \
                patch.object(train, 'prepare', return_value=fixture), \
                patch.object(train, 'make_pi05_dataset',
                             side_effect=lambda repo, root, episodes, *args: TinyWindowDataset(episodes)):
            runs = []
            for name, stop in [('continuous', None), ('interrupted', 1)]:
                config, _, _, settings, _ = config_and_processors(Path(directory) / name)
                train.run(settings, config, SUBMODULE / 'configs/pi05/smoke.yaml', stop_after=stop)
                runs.append((settings, config))
            settings, config = runs[1]
            settings['training']['resume'] = str(Path(settings['output_dir']) / 'checkpoints/000001/pretrained_model')
            train.run(settings, config, SUBMODULE / 'configs/pi05/smoke.yaml')
            paths = [Path(settings['output_dir']) / 'checkpoints/000002/pretrained_model'
                     for settings, _ in runs]
            expected, actual = [load_file(str(path / 'model.safetensors')) for path in paths]
            for key in expected:
                torch.testing.assert_close(expected[key], actual[key], atol=0, rtol=0)
            self.assertEqual(json.loads((paths[1] / 'checkpoint.json').read_text())['observation_history'],
                             make_history([-6, -3, 0]))
            self.assertEqual(json.loads((paths[1].parents[2] / 'reload_check.json').read_text())['status'], 'passed')


if __name__ == '__main__':
    unittest.main()
