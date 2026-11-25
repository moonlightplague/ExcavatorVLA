"""Raw RGB sequence training for persistent ACT without changing deployment code.

Keep the native ACT core in eval mode (zero prior latent, no dropout), while
allowing gradients through explicitly selected visual/decoder parameters.
The frozen GRU uses its training kernel only for input gradients; it has no
recurrent dropout. Optimizer updates occur only at parent-episode boundaries.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import av
import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.policies.act.temporal_extension import TemporalACT


class VisualSequence:
    def __init__(self, model, gradients=False):
        if getattr(model.config,'action_parameterization','velocity')!='velocity':
            raise ValueError('Visual fine-tuning requires the established velocity model')
        self.model=model;self.gradients=gradients
        self.history=deque(maxlen=model.config.temporal_horizon)
        self.hidden=None
        if model.memory_adapter.gru.dropout!=0:raise ValueError('Frozen recurrent dropout must be zero')

    def step(self,batch):
        self.model.memory_adapter.gru.train(self.gradients)
        state=batch['observation.state'];images=[batch[k] for k in self.model.config.image_features]
        spatial=[self.model.backbone(image)['feature_map'] for image in images]
        pooled=torch.stack([value.mean((-2,-1)) for value in spatial],dim=1)
        self.history.append((state.detach(),pooled.detach()))
        missing=self.model.config.temporal_horizon-len(self.history)
        entries=[self.history[0]]*missing+list(self.history)
        states=torch.stack([entry[0] for entry in entries],1)
        visuals=torch.stack([entry[1] for entry in entries[:-1]]+[pooled],1)
        padding=torch.tensor([[True]*missing+[False]*len(self.history)],device=state.device,dtype=torch.bool).expand(state.shape[0],-1)
        current={'observation.state':state,'observation.images':images,'_act_cached_backbone':spatial}
        # Deliberately bypass PersistentACT's inference-only no_grad wrapper.
        base,_=TemporalACT._current_forward(self.model,current,states,visuals,padding)
        features=torch.cat((state,pooled.flatten(1),base.flatten(1)),dim=-1)
        delta,hidden=self.model.memory_adapter(features[:,None],self.hidden)
        self.hidden=hidden.detach()
        return base+delta[:,0].reshape_as(base)


def enable_visual_gradients(policy):
    policy.eval()
    for name,parameter in policy.named_parameters():
        parameter.requires_grad_(name.startswith(('model.backbone.layer4.','model.decoder.','model.action_head.')))
    return [name for name,p in policy.named_parameters() if p.requires_grad]


def photometric(images,brightness,contrast,noise_sigma,generator=None):
    """Appearance only: same episode lighting factors for all three cameras."""
    result={}
    for key,image in images.items():
        center=image.mean((-2,-1),keepdim=True)
        value=((image-center)*contrast+center)*brightness
        if noise_sigma:
            value=value+torch.randn(value.shape,dtype=value.dtype,device=value.device,generator=generator)*(noise_sigma/255.)
        result[key]=value.clamp(0.,1.)
    return result


def selected_video(path,frames,offset,fps):
    output=np.empty((len(frames),3,256,256),dtype=np.uint8);i=0
    with av.open(str(path)) as container:
        stream=container.streams.video[0];stream.thread_count=1;stream.thread_type='SLICE'
        for frame in container.decode(video=0):
            if i>=len(frames):break
            stamp=float(frame.pts*frame.time_base);target=offset+int(frames[i])/fps
            if stamp<target-.5/fps:continue
            if abs(stamp-target)>.02:raise ValueError('Unsynchronized RGB timestamp: '+str(path))
            image=frame.to_ndarray(format='rgb24')
            if image.shape!=(256,256,3):raise ValueError('Unexpected canonical RGB size')
            output[i]=image.transpose(2,0,1);i+=1
    if i!=len(frames):raise ValueError('Incomplete episode video: '+str(path))
    return output


class RGBEpisodes:
    def __init__(self,root):
        self.root=Path(root);self.info=json.loads((self.root/'meta/info.json').read_text())
        self.rows={r['episode_index']:r for p in sorted((self.root/'meta/episodes').rglob('*.parquet')) for r in pq.read_table(p).to_pylist()}
        self.cameras=[f'observation.images.{i}' for i in range(3)]

    def load(self,episode,max_decisions=0):
        r=self.rows[episode]
        table=pq.read_table(self.root/f'data/chunk-{r["data/chunk_index"]:03d}/file-{r["data/file_index"]:03d}.parquet')
        ids=np.asarray(table['episode_index'].to_pylist());table=table.filter(__import__('pyarrow').compute.equal(table['episode_index'],episode))
        states=np.asarray(table['observation.state'].to_pylist(),dtype=np.float32)
        actions=np.asarray(table['action'].to_pylist(),dtype=np.float32)
        frames=np.arange(0,len(states),3)
        if max_decisions:frames=frames[:max_decisions]
        future=frames[:,None]+np.arange(10);pad=future>=len(states)
        stage=np.asarray(table['observation.stage_current_id'].to_pylist()).reshape(-1)[frames]
        ret=np.asarray(table['observation.return_to_pile'].to_pylist()).reshape(-1)[frames]>0
        group=np.stack((np.isin(stage,[2,3,4,5])&~ret,(stage>=6)&~ret,ret),axis=1)
        def decode(camera):
            path=self.root/f'videos/{camera}/chunk-{r[f"videos/{camera}/chunk_index"]:03d}/file-{r[f"videos/{camera}/file_index"]:03d}.mp4'
            return selected_video(path,frames,r[f'videos/{camera}/from_timestamp'],self.info['fps'])
        with ThreadPoolExecutor(max_workers=3) as pool:
            images=dict(zip(self.cameras,pool.map(decode,self.cameras),strict=True))
        return dict(episode=episode,state=states[frames],target=actions[np.minimum(future,len(states)-1)],padding=pad,groups=group,frames=frames,images=images)

    def group(self,episodes,max_decisions=0):return [self.load(episode,max_decisions) for episode in episodes]


def batch_at(episodes,index):
    valid=np.array([index<len(ep['frames']) for ep in episodes]);idx=[min(index,len(ep['frames'])-1) for ep in episodes]
    raw={'observation.state':torch.from_numpy(np.stack([ep['state'][j] for ep,j in zip(episodes,idx,strict=True)])),
         'action':torch.from_numpy(np.stack([ep['target'][j] for ep,j in zip(episodes,idx,strict=True)]))}
    for camera in episodes[0]['images']:
        raw[camera]=torch.from_numpy(np.stack([ep['images'][camera][j] for ep,j in zip(episodes,idx,strict=True)])).float()/255.
    pad=np.stack([ep['padding'][j] for ep,j in zip(episodes,idx,strict=True)])|~valid[:,None]
    groups=np.stack([ep['groups'][j] for ep,j in zip(episodes,idx,strict=True)])&valid[:,None]
    return raw,pad,groups,valid
