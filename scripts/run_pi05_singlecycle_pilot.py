"""Verified single-backend 20-episode PI05 pilot with 2k checkpoint retention."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from run_pi05_lora_ab import Experiment, PROJECT, SUBMODULE, read


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def source_manifest(config):
    paths={Path(__file__).resolve(), SUBMODULE/'scripts/train_pi05.py',
           SUBMODULE/'scripts/eval_pi05_success.py',config,
           PROJECT/'scripts/eval_smolvla_success.py',PROJECT/'scripts/excavator_diagnostics.py',
           PROJECT/'scripts/success_eval_runtime.py',PROJECT/'scripts/eval_video.py',
           PROJECT/'ExcavatorVLA-smolvla/run_simulation.py',
           PROJECT/'ExcavatorVLA-smolvla/scripts/excavator_app/sand_site_runtime.py',
           PROJECT/'ExcavatorVLA-smolvla/scripts/excavator_app/newton_mpm_soil.py',
           PROJECT/'ExcavatorVLA-isaac/assets/usd/excavator_scene.usd'}
    for folder in ['policies/pi05','common','data','evaluation']:
        paths.update((SUBMODULE/'excavator_vlm'/folder).glob('*.py'))
    paths.update((PROJECT/'ExcavatorVLA-smolvla/excavator_common').glob('*.py'))
    paths.update(SUBMODULE/'configs/pi05'/name for name in
                 ['full.yaml','overfit.yaml','expert_ab_overfit.yaml','lora_overfit.yaml'])
    return {str(p.relative_to(PROJECT)):digest(p) for p in sorted(paths)}


class SingleCyclePilot(Experiment):
    def __init__(self,args):
        super().__init__(args)
        self.hashes=source_manifest(args.config)
        self.state.update(phase='preflight',soil_backend='physx',training_mode='vlm_lora_expert_full',
            training_source='immutable pi05_base; new LoRA/optimizer and training-subset quantiles',
            requested_steps=10000,checkpoint_steps=[2000,4000,6000,8000,10000],source_hashes=self.hashes)
        self.save()

    def check_sources(self):
        if source_manifest(self.args.config)!=self.hashes:
            raise RuntimeError('Pilot processing/simulator/config source changed; preserve results and inspect')

    def run(self,name,commands):
        try:
            super().run(name,commands)
        except BaseException as exc:
            if self.state['stages'] and self.state['stages'][-1]['name']==name:
                self.state['stages'][-1].update(status='failed',error=f'{type(exc).__name__}: {exc}')
            self.save()
            raise

    def eval_command(self,checkpoint,reference,output,gpu,replay=False):
        command=[sys.executable,'-u',str(SUBMODULE/'scripts/eval_pi05_success.py'),
            '--checkpoint',str(checkpoint),'--dataset',str(self.dataset),
            '--reference-episode',str(reference),'--episodes','1','--steps','350',
            '--min-particles','100','--seed','2','--replan-interval','3','--gpus',str(gpu),
            '--workers','1','--cpu-threads','8','--headless','--fresh-simulator',
            '--render-every-physics-step','--video','--soil-backend','physx',
            '--sand-walls-during-settle','--output',str(output)]
        if replay:command.append('--replay-reference')
        return command,{}

    def execute(self):
        from excavator_vlm.policies.pi05.config import resolve_settings
        from excavator_vlm.policies.pi05.train import prepare
        settings=resolve_settings(self.args.config)
        report,split,frames,episodes=prepare(settings)
        self.dataset=Path(settings['dataset']['root'])
        if (report['collection_backend']['soil_backend']!='physx' or len(episodes)!=20
                or len(split['validation'])!=4 or len(split['test'])!=4
                or settings['steps']!=10000 or settings['save_freq']!=2000
                or settings['benchmark']['validation_freq']!=2000
                or settings['benchmark']['validation_max_batches']!=0
                or settings['training']['keep_checkpoints']<5
                or settings['training']['mode']!='vlm_lora_expert_full'):
            raise ValueError('Pilot data/backend/retention contract differs from requested 20/4/4 and 10k/2k')
        train_dir=self.root/'train'
        self.state.update(dataset=str(self.dataset),dataset_audit=report,split_manifest=split,
            training_episodes=episodes,training_frames=int(frames.episode_index.isin(episodes).sum()),
            validation_frames=int(frames.episode_index.isin(split['validation']).sum()),
            optimizer_steps_each=10000,effective_global_batch_each=8,
            protected_checkpoints=[str(train_dir/'checkpoints'/f'{s:06d}'/'pretrained_model')
                                   for s in [2000,4000,6000,8000,10000]])
        self.save()
        references={'train':self.dataset/f'references/episode_{episodes[0]:04d}.json',
                    'validation':self.dataset/f'references/episode_{split["validation"][0]:04d}.json'}
        gates={label:self.root/f'expert_gate_{label}' for label in references}
        self.run('physx_expert_replay_gates',[
            self.eval_command(self.args.gate_checkpoint,references[label],gates[label],gpu,True)
            for label,gpu in zip(references,self.args.gpus)])
        for label,path in gates.items():
            status=read(path/'episodes/episode_0001/episode_status.json')
            if status['status']!='completed' or not status['success'] or status.get('error'):
                raise RuntimeError(f'PhysX {label} source expert is not reproducible; no training started')
        self.state['expert_gates_passed']=True
        self.save()
        command=[sys.executable,'-u',str(SUBMODULE/'scripts/train_pi05.py'),
            '--config',str(self.args.config),'--gpus',','.join(map(str,self.args.gpus)),
            '--output',str(train_dir)]
        self.run('lora_full_expert_ddp_10000',[(command,{})])
        latest=read(train_dir/'last_checkpoint.json')
        if latest['step']!=10000 or read(train_dir/'reload_check.json')['status']!='passed':
            raise RuntimeError('Pilot did not finish 10k/reload validation')
        for path in self.state['protected_checkpoints']:
            if not (Path(path)/'model.safetensors').is_file():
                raise RuntimeError('A requested 2k-interval checkpoint is missing')
        provenance=read(train_dir/'provenance.json')
        if (provenance['batch_contract']!={'world_size':2,'batch_size':1,'accumulation':4}
                or provenance['collection_backend']['soil_backend']!='physx'):
            raise RuntimeError('Actual training GPU/batch/backend contract differs')
        self.state.update(training_completed=True,checkpoint=latest['checkpoint'])
        self.save()
        final={label:self.root/f'model10k_{label}_video' for label in references}
        self.run('final_physx_closed_loop_videos',[
            self.eval_command(Path(latest['checkpoint']),references[label],final[label],gpu)
            for label,gpu in zip(references,self.args.gpus)])
        results={label:read(path/'episodes/episode_0001/episode_status.json') for label,path in final.items()}
        (self.root/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
        self.state.update(status='completed',phase='completed',completed_utc=datetime.now(timezone.utc).isoformat())
        self.save()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=SUBMODULE/'configs/pi05/lora_singlecycle20_physx_10k.yaml')
    parser.add_argument('--gate-checkpoint',type=Path,default=PROJECT/
        'outputs/experiments/pi05_lora_ab_20261028_01/lora_overfit/checkpoints/005000/pretrained_model')
    args=parser.parse_args()
    args.output=args.output.resolve();args.config=args.config.resolve();args.gate_checkpoint=args.gate_checkpoint.resolve()
    args.gpus=[0,1];args.steps=10000
    job=SingleCyclePilot(args)
    try:
        job.execute()
    except BaseException as exc:
        job.state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        job.save()
        raise


if __name__=='__main__':main()
