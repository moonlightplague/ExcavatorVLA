"""Build 20/4/4 single-cycle episodes with explicit collection-backend evidence.

Uses only the first successful cycle of each distinct repeated parent. Source
video containers are immutable hard links; metadata exposes only the selected
cycle's timestamps. No action is resampled, shifted or synthesized.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

SUBMODULE = Path(__file__).resolve().parents[1]
ROOT = SUBMODULE.parent
sys.path.insert(0, str(SUBMODULE))
from excavator_vlm.data.contract import CAMERAS, audit_dataset, file_hash, load_info, validate_split
from excavator_vlm.policies.pi05.dataset import training_stats

TASK = 'Excavate one scoop from the visible sand pile, carry the material, and dump it into the truck bed.'


def controller_parity(states, actions):
    """Check recorded setpoints under the benchmark's bounded velocity integrator.

    A source may reset its target after dumping without encoding that reset in
    velocity actions. Assess through the first open-bucket frame after curling,
    before that terminal reset; never repair labels by inventing velocities.
    """
    states=np.asarray(states,dtype=np.float32);actions=np.asarray(actions,dtype=np.float32)
    limits=np.array([[-np.pi,np.pi],[-1.3089969158,1.3089969158],
                     [-1.6580628157,1.6580628157],[-2.0943951607,1.5707963705]],dtype=np.float32)
    target=states[:,:4]+states[:,8:12]
    current=target[0].copy();errors=[]
    for i in range(len(states)-1):
        for _ in range(6):current=np.clip(current+actions[i]/60,limits[:,0],limits[:,1])
        errors.append(np.abs(current-target[i+1]))
    curled=np.flatnonzero(states[:,3]<=-1.8)
    opened=np.flatnonzero((np.arange(len(states))>(int(curled[0]) if len(curled) else len(states)))
                         &(states[:,3]>=-.7))
    end=int(opened[0])+1 if len(opened) else len(states)-1
    return float(np.max(np.asarray(errors)[:end])),end


def read(path):
    return json.loads(path.read_text())


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def diverse(records, count, seed):
    if len(records) < count:
        raise ValueError(f'Need {count} independent parents, only {len(records)} eligible')
    values = np.asarray([r['selection_features'] for r in records], dtype=float)
    scale = np.quantile(values, .95, axis=0)-np.quantile(values, .05, axis=0)
    values = (values-np.median(values, axis=0))/np.where(scale>1e-8, scale, 1.)
    selected = [int(np.random.default_rng(seed).integers(len(records)))]
    distance = np.full(len(records), np.inf)
    while len(selected)<count:
        distance = np.minimum(distance, ((values-values[selected[-1]])**2).sum(1))
        distance[selected] = -np.inf
        selected.append(int(np.argmax(distance)))
    return [records[i] for i in selected]


def candidates(source, raw, backend):
    metadata = pd.concat([pd.read_parquet(p) for p in sorted(source.glob('meta/episodes/**/*.parquet'))],
                         ignore_index=True).set_index('episode_index')
    result, rejected = [], []
    seen = set()
    for run in sorted(raw.glob('run_*')):
        rows = {r['episode_id']:r for r in (json.loads(line) for line in
                (run/'episodes.jsonl').read_text().splitlines() if line)}
        for path in sorted(run.glob('episode_*/meta.json')):
            meta = read(path)
            export_path = path.parent/'export.json'
            if not meta.get('success') or not export_path.is_file():
                continue
            try:
                if meta.get('repeat_baseline', {}).get('soil_backend') != backend:
                    raise ValueError('collection_backend_unknown_or_mixed')
                if not meta.get('source_sampling_audit', {}).get('ok'):
                    raise ValueError('source_sampling_audit_failed')
                exported = read(export_path)
                if not exported.get('validated') or Path(exported['export_dir']).resolve()!=source:
                    raise ValueError('export_mapping_unverified')
                eid = int(exported['export_episode_index'])
                if eid in seen:
                    raise ValueError('duplicate_source_export_episode')
                row = metadata.loc[eid]
                frame_path = source/f"data/chunk-{int(row['data/chunk_index']):03d}/file-{int(row['data/file_index']):03d}.parquet"
                full = pd.read_parquet(frame_path)
                full = full[full.episode_index==eid].sort_values('frame_index').reset_index(drop=True)
                if len(full)!=meta['source_sampling_audit']['sample_count'] or full.frame_index.tolist()!=list(range(len(full))):
                    raise ValueError('raw_export_frame_mapping_differs')
                cycle = meta['repeat_cycles'][0]
                cut = int(cycle['return_start'])
                if (cycle['scoop_index']!=0 or cycle['sample_start']!=0
                        or not cycle['metrics']['success'] or cycle['metrics']['payload_particles']<100
                        or not 80<=cut<len(full)):
                    raise ValueError('first_cycle_not_a_complete_success')
                clip = full.iloc[:cut].copy()
                state = np.stack(clip['observation.state'])
                phase = set(map(int, clip['observation.stage_current_id']))
                if not {0,1,2,3,4,5,6,7,9}.issubset(phase):
                    raise ValueError('missing_dig_curl_secure_lift_or_dump_phase')
                loaded = state[:,26]>=100/6400
                if (clip['observation.scoop_index'].max()!=0 or clip['observation.return_to_pile'].any()
                        or (clip.action_is_expert!=1).any() or (clip.action_loss_weight<=0).any()
                        or state[:,3].min()>-1 or state[-1,3]<-.8 or state[-1,26]>.02
                        or loaded.sum()<3 or np.ptp(np.unwrap(state[:,0])[loaded])<.3):
                    raise ValueError('cycle_geometry_labels_or_terminal_dump_incomplete')
                parity_error,parity_end=controller_parity(state,np.stack(clip.action))
                if parity_error>.02:
                    raise ValueError(f'velocity_integrator_setpoint_mismatch:{parity_error:.6f}rad')
                np.testing.assert_allclose(clip.timestamp, np.arange(cut)/10, atol=1e-5)
                index = rows[meta['episode_id']]
                applied = index['scene_randomization']['applied']
                yaw = math.radians(applied['truck_yaw_deg'])
                features = [*applied['sand_center'][:2], *applied['truck_translation_xyz'][:2],
                            math.sin(yaw), math.cos(yaw), *state[0,:4],
                            float(state[:,26].max()), cycle['metrics']['payload_particles']/1000]
                result.append(dict(source_id=eid, parent_id=f'{run.name}/{path.parent.name}',worker=run.name[-2:],
                    meta_path=path, export_path=export_path, index=index, meta=meta, row=row,
                    frame_path=frame_path, frames=clip, cut=cut, selection_features=features,
                    controller_parity_max_error_rad=parity_error,controller_parity_until_frame=parity_end))
                seen.add(eid)
            except (ValueError, KeyError, AssertionError) as exc:
                rejected.append(dict(path=str(path), reason=str(exc)))
    return result, rejected


def build(args):
    info = load_info(args.source)
    records, rejected = candidates(args.source, args.raw, args.backend)
    groups = sorted({r['worker'] for r in records})
    if len(groups)!=2:
        raise ValueError('Expected two independently collected worker pools')
    selected = {k:[] for k in ('train','validation','test')}
    pool_ids = {}
    for worker_index, worker in enumerate(groups):
        group = sorted([r for r in records if r['worker']==worker],key=lambda r:r['source_id'])
        rng = np.random.default_rng(args.seed+worker_index)
        order = rng.permutation(len(group))
        n = len(group); held = n//10
        if held<2 or n-2*held<10:
            raise ValueError('Too few verified parents for independent 20/4/4 selection')
        pools = {'train':order[:n-2*held], 'validation':order[n-2*held:n-held], 'test':order[n-held:]}
        for split, count in [('train',10),('validation',2),('test',2)]:
            pool = [group[int(i)] for i in pools[split]]
            pool_ids[f'{worker}/{split}'] = [r['source_id'] for r in pool]
            selected[split].extend(diverse(pool,count,args.seed+worker_index*10+len(selected[split])))
    parents = [r['parent_id'] for values in selected.values() for r in values]
    if len(parents)!=len(set(parents)):
        raise ValueError('Source parent leakage between selected episodes/splits')
    staging = args.output.with_name(args.output.name+'.building')
    if args.output.exists() or staging.exists() or args.split_file.exists():
        raise FileExistsError('Refusing to overwrite an existing dataset, build or frozen split')
    staging.mkdir(parents=True)
    episodes, frames, evidence, references = [], [], [], []
    split = dict(schema_version=1,seed=args.seed,method='source_parent_split_first_then_diverse_subset_20_4_4',
                 train=[],validation=[],test=[])
    offset = 0
    for subset, values in selected.items():
        for record in values:
            eid=len(episodes); df=record['frames'].copy(); length=len(df)
            original_frames=df.frame_index.tolist()
            df['index']=np.arange(offset,offset+length,dtype=np.int64)
            df['episode_index']=eid;df['frame_index']=np.arange(length,dtype=np.int64)
            df['task_index']=0
            for key in ['observation.scoops_target','observation.scoops_min','observation.scoops_max']:
                df[key]=1
            target=staging/f'data/chunk-000/file-{eid:03d}.parquet'
            target.parent.mkdir(parents=True,exist_ok=True);df.to_parquet(target,index=False)
            row=record['row'].to_dict()
            row.update(episode_index=eid,tasks=[TASK],length=length,dataset_from_index=offset,
                       dataset_to_index=offset+length)
            row.update({'meta/episodes/chunk_index':0,'meta/episodes/file_index':0,
                        'data/chunk_index':0,'data/file_index':eid})
            for camera in CAMERAS:
                relative=info['video_path'].format(video_key=camera,
                    chunk_index=int(row[f'videos/{camera}/chunk_index']),file_index=int(row[f'videos/{camera}/file_index']))
                video=staging/relative;video.parent.mkdir(parents=True,exist_ok=True)
                if not video.exists():os.link(args.source/relative,video)
                row[f'videos/{camera}/to_timestamp']=float(row[f'videos/{camera}/from_timestamp'])+(length-1)/10
            source_dir=staging/f'provenance/episode_{eid:04d}'
            source_dir.mkdir(parents=True)
            shutil.copy2(record['meta_path'],source_dir/'meta.json')
            shutil.copy2(record['export_path'],source_dir/'export.json')
            dump(source_dir/'episode_index_record.json',record['index'])
            meta_relative=str((source_dir/'meta.json').relative_to(staging))
            entry=dict(episode_index=eid,split=subset,parent_id=record['parent_id'],
                source_export_episode=record['source_id'],source_frame_range=[0,length],
                source_cycle_index=0,soil_backend=args.backend,
                controller_parity_max_error_rad=record['controller_parity_max_error_rad'],
                controller_parity_until_frame=record['controller_parity_until_frame'],
                payload_particles=record['meta']['repeat_cycles'][0]['metrics']['payload_particles'],
                phases=sorted(map(int,df['observation.stage_current_id'].unique())),
                evidence=dict(path=meta_relative,sha256=file_hash(source_dir/'meta.json'),field='repeat_baseline.soil_backend'))
            evidence.append(entry)
            applied=record['index']['scene_randomization']['applied']
            first=record['meta']['repeat_cycles'][0]
            state=np.stack(df['observation.state'])
            setup=dict(schema_version='excavator_reference_setup_v1',fps=10,action_units='rad/s',
                state_names=info['features']['observation.state']['names'],action_names=info['features']['action']['names'],
                profile=dict(sand_xy=applied['sand_center'][:2],truck_translation_xyz=applied['truck_translation_xyz'],
                    truck_yaw_deg=applied['truck_yaw_deg'],unload_landing_xyz=first['landing']['landing']),
                task=TASK,initial_joints_deg=np.degrees(state[0,:4]).tolist(),
                dig_target_xyz=record['index']['target_xyz'],sand_amount=applied['sand_amount_multiplier'],
                expected_initial_state=state[0].tolist(),actions=np.stack(df.action).tolist(),
                provenance=dict(kind='singlecycle_from_explicit_backend_collection',soil_backend=args.backend,
                    source_parent=record['parent_id'],source_export_episode=record['source_id'],
                    source_frame_range=[0,length],source_meta_sha256=entry['evidence']['sha256']))
            dump(staging/f'references/episode_{eid:04d}.json',setup)
            references.append(dict(episode_index=eid,split=subset,path=f'references/episode_{eid:04d}.json'))
            split[subset].append(eid);episodes.append(row);frames.append(df);offset+=length
    merged=pd.concat(frames,ignore_index=True)
    metadata=staging/'meta/episodes/chunk-000/file-000.parquet';metadata.parent.mkdir(parents=True)
    pd.DataFrame(episodes).to_parquet(metadata,index=False)
    pd.DataFrame({'task_index':[0]},index=pd.Index([TASK])).to_parquet(staging/'meta/tasks.parquet')
    info=copy.deepcopy(info)
    info.update(total_episodes=len(episodes),total_frames=offset,total_tasks=1,splits={'train':f'0:{len(episodes)}'},
        collection_soil_backend=args.backend,
        extraction=dict(source_dataset=str(args.source),source_cycle=0,return_removed=True,
                        language=TASK,metadata_stats_scope='training episodes only; no visual normalization',
                        video_storage='immutable complete source containers; only clipped timestamps are queried'))
    dump(staging/'meta/info.json',info)
    dump(staging/'meta/stats.json',training_stats(merged,split['train']))
    dump(staging/'backend_provenance.json',dict(schema_version='excavator_backend_provenance_v1',
        soil_backend=args.backend,episodes=evidence))
    dump(staging/'selection.json',dict(seed=args.seed,source_dataset=str(args.source),
        source_meta_info_sha256=file_hash(args.source/'meta/info.json'),eligible_parents=len(records),
        rejected=rejected,split_first_source_pools=pool_ids,selected=evidence,references=references))
    report,_=audit_dataset(staging,10)
    split['dataset_fingerprint']=report['fingerprint']
    validate_split(split,range(len(episodes)),report['fingerprint'])
    dump(staging/'audit.json',report)
    dump(staging/'split_manifest.json',split)
    staging.rename(args.output)
    args.split_file.parent.mkdir(parents=True,exist_ok=True)
    with args.split_file.open('x') as f:json.dump(split,f,indent=2);f.write('\n')
    print(json.dumps(dict(dataset=str(args.output),audit=report,split_counts={k:len(split[k]) for k in selected},
                         selected_source_ids={k:[r['source_id'] for r in selected[k]] for k in selected}),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'data/repeat/export_repeat')
    parser.add_argument('--raw',type=Path,default=ROOT/'data/repeat/raw')
    parser.add_argument('--output',type=Path,default=ROOT/'data/pi05_singlecycle_physx_pilot_v2')
    parser.add_argument('--split-file',type=Path,default=SUBMODULE/'configs/splits/pi05_singlecycle20_physx_v2_seed1000.json')
    parser.add_argument('--backend',choices=('physx','newton_mpm'),default='physx')
    parser.add_argument('--seed',type=int,default=1000)
    args=parser.parse_args()
    for name in ['source','raw','output','split_file']:setattr(args,name,getattr(args,name).resolve())
    build(args)


if __name__=='__main__':main()
