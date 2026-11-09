"""PI05 state-history prompts without phase labels or new pretrained parameters."""
from dataclasses import dataclass

import numpy as np
from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.processor.core import TransitionKey

from .history import MASK


@ProcessorStepRegistry.register(name='excavator_pi05_history_prompt_v1')
@dataclass
class HistoryPromptProcessorStep(ProcessorStep):
    delta_indices: tuple[int, ...] = (-6, -3, 0)
    fps: int = 10
    max_state_dim: int = 32

    def __call__(self, transition):
        transition = transition.copy()
        complementary = dict(transition[TransitionKey.COMPLEMENTARY_DATA])
        state = transition[TransitionKey.OBSERVATION]['observation.state'].detach().cpu().numpy()
        mask = complementary[MASK].detach().cpu().numpy()
        bins = np.digitize(state, np.linspace(-1, 1, 257)[:-1]) - 1
        prompts = []
        for row, task in enumerate(complementary['task']):
            task = task.strip().replace('_', ' ').replace('\n', ' ')
            history = []
            for time, offset in enumerate(self.delta_indices):
                if mask[row, time]:
                    history.append(f't={offset / self.fps:g}s: padding')
                else:
                    # Match upstream padding BEFORE quantization: padded zeros map to bin 128.
                    values = np.pad(bins[row, time], (0, self.max_state_dim - state.shape[-1]),
                                    constant_values=128)
                    history.append(f't={offset / self.fps:g}s: ' + ' '.join(map(str, values)))
            prompts.append(f'Task: {task}, Images: oldest to newest; cameras 0 1 2 per time. '
                           + 'State history: ' + '; '.join(history) + ';\nAction: ')
        complementary['task'] = prompts
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    def transform_features(self, features):
        return features
