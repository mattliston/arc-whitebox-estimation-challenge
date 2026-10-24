"""Qualify only completed, passing TRAIN90 fits under the original guard."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;RUNS=Path('/home/arcworker/arc-runs')
A=HERE.parents[1]/'fable_oct19_kc_v1/agent_astra_joint_v1'
sys.path.insert(0,str(A/'cloud_linux_capacity_v7'));from common_linux_v7 import thread_env
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
script=HERE/'qualify_passed_v1.py'
assert sha(script)=='8f718352699d093a09d782ffb23a99b072b2bb6dc9522f604b55e1612c548174'
guard=A/'cloud_linux_capacity_v7/guard_linux_v7.py'
assert sha(guard)=='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
deadline=time.monotonic()+21600
while not (HERE/'FIT_PIPELINE_FINAL_v1.json').exists():
    assert time.monotonic()<deadline,'Fit wait timed out'
    time.sleep(15)
finish=read(HERE/'FIT_PIPELINE_FINAL_v1.json');assert finish['exit_code']==0,finish
env=os.environ.copy();env.update(thread_env(1));env['ARC_BLAS_THREADS']='1'
receipt=RUNS/'local_innovation90_qualification_v1_GUARD.json';log=RUNS/'local_innovation90_qualification_v1.log'
assert not receipt.exists() and not log.exists()
cmd=['/home/arcworker/arc-envs/judge/bin/python','-B',str(guard),'--receipt',str(receipt),
    '--state-dir',str(RUNS/'admission'),'--max-active','36','--cgroup-root','/sys/fs/cgroup/arc-estimator-v5',
    '--','/home/arcworker/arc-envs/server/bin/python','-B',str(script)]
with log.open('x') as stream:
    p=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
    write(HERE/'QUALIFY_LAUNCH_v1.json',dict(utc=utc(),pid=p.pid,command=cmd,script_sha256=sha(script),
        fit_pipeline_final_sha256=sha(HERE/'FIT_PIPELINE_FINAL_v1.json'),pipeline_sha256=sha(__file__)))
    code=p.wait()
result=HERE/'native_v1/QUALIFIED_v1.json'
rec=dict(utc=utc(),exit_code=code,guard=read(receipt) if receipt.exists() else None,
    qualified_sha256=sha(result) if result.exists() else None,
    qualified=[r['candidate'] for r in read(result)['qualified']] if result.exists() else None)
write(HERE/'QUALIFY_PIPELINE_FINAL_v1.json',rec);print(json.dumps(rec),flush=True)
assert code==0 and result.exists(),rec

