from pathlib import Path
import time
base=Path('/root/gpufree-data/ExcavatorVLA_new/ExcavatorVLA-isaac/experiments/bed_deposition_validation/same_scoop_v195v34c');previous=base/'baseline_repeat'
end=time.monotonic()+600
def running():
 proc=Path('/proc')/(previous/'pid').read_text().strip()/'cmdline'
 try:return b'baseline_repeat' in proc.read_bytes()
 except FileNotFoundError:return False
while running() or not (previous/'result.json').exists():
 if time.monotonic()>end:raise RuntimeError('baseline repeat did not finish')
 time.sleep(2)
from pathlib import Path
import os,json,subprocess,hashlib
root=Path('/root/gpufree-data/ExcavatorVLA_new');repo=root/'ExcavatorVLA-isaac';base=Path('/root/gpufree-data/ExcavatorVLA_new/ExcavatorVLA-isaac/experiments/bed_deposition_validation/same_scoop_v195v34c')
label='deposition_final'
out=base/label;out.mkdir()
src=(Path('/root/gpufree-data/ExcavatorVLA_new/ExcavatorVLA-isaac/experiments/bed_floor_validation/run_20261118')/'record.py').read_text()
src=src.replace("project = output.parents[3]","project = Path("+repr(str(root))+")")
src=src.replace("import json, os, sys, traceback","import asyncio, hashlib, json, os, sys, time, traceback")
src=src.replace("            stage = soil.stage","""            if 'model' not in st:
                from excavator_app import newton_mpm_soil
                code=Path(newton_mpm_soil.__file__)
                (output/'model.json').write_text(json.dumps(dict(sha=hashlib.sha256(code.read_bytes()).hexdigest(), diagnostics=soil.diagnostics()), default=str))
                st['model']=True
            stage = soil.stage""")
src=src.replace("t=soil.sim_time,step=soil.steps,","t=soil.sim_time,step=soil.steps,wall=time.perf_counter(),physics=soil.physics_wall_seconds,render=soil.render_wall_seconds,deposit=getattr(getattr(soil,'bed_deposition',None),'elapsed',0.),graph=soil.graph_replays,phase=rt.STATE.get('last_action_mode'),")
marker="        except Exception:"
src=src.replace(marker,"""            baseline=getattr(rt,'repeat_baseline',None)
            if baseline and baseline.cycles and baseline.cycles[0].get('returned') and not st.get('done'):
                st['done']=True
                (output/'first_return.json').write_text(json.dumps(dict(cycle=baseline.cycles[0],time=soil.sim_time,frame=frame,diagnostics=soil.diagnostics()),default=str))
                asyncio.get_event_loop().call_soon(rt.stop_auto_collect)
"""+marker)
compile(src,'observe','exec');(out/'observe.py').write_text(src)
env=os.environ.copy(); env.update(EXCAVATOR_AUTO_RUN_ID_SUFFIX='bedlocalaudit',EXCAVATOR_AUTO_COLLECT_RESULT_FILE=str(out/'result.json'),EXCAVATOR_KIT_LOG_FILE=str(out/'kit.log'),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
cmd=[str(root/'run_isaac_python.sh'),str(out/'observe.py'),'--active-gpu','1','--physics-gpu','1','--success-count','1','--max-attempts','1','--dataset-root',str(out/'raw'),'--scoops-per-episode','2','--sand-amount','3','--scene-seed-start','17720593130087219389','--acceptance-profile','coarse_mass','--min-dig-height','.12','--min-curl-height','1.4']
with (out/'run.log').open('w') as log:
 proc=subprocess.Popen(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(out/'pid').write_text(str(proc.pid));print(proc.pid)
