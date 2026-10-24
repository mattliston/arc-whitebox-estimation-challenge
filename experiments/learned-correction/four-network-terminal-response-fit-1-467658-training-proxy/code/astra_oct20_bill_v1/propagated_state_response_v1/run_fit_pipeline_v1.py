"""Wait for immutable capture completion, then run one guarded TRAIN4 fit."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
RUNS=Path('/home/arcworker/arc-runs')
A=HERE.parents[1]/'fable_oct19_kc_v1/agent_astra_joint_v1'
sys.path.insert(0,str(A/'cloud_linux_capacity_v7'))
from common_linux_v7 import thread_env
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
fit=HERE/'fit_terminal_response_v2.py'
assert sha(fit)=='2a0091f351f1bcb04de3cf6263a89cc43595e279a201aefe2bc4bd99e6635978'
guard=A/'cloud_linux_capacity_v7/guard_linux_v7.py'
assert sha(guard)=='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
capture=RUNS/'full_state_response_train4_v1'
deadline=time.monotonic()+3600
while not (capture/'COMPLETE.json').exists():
 assert time.monotonic()<deadline,'capture wait timeout'
 time.sleep(15)
completion=json.loads((capture/'COMPLETE.json').read_text())
assert completion['successful'],completion
receipt=RUNS/'propagated_state_response_fit_v1_GUARD.json'
log=RUNS/'propagated_state_response_fit_v1.log'
assert not receipt.exists() and not log.exists()
env=os.environ.copy();te=thread_env(1);env.update(te);env['ARC_BLAS_THREADS']='1'
py='/home/arcworker/arc-envs/judge/bin/python'
cmd=[py,'-B',str(guard),'--receipt',str(receipt),'--state-dir',str(RUNS/'admission'),
 '--max-active','36','--cgroup-root','/sys/fs/cgroup/arc-estimator-v5','--',py,'-B',str(fit)]
with log.open('x') as f:
 child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
 launch=dict(utc=utc(),guard_pid=child.pid,command=cmd,fit_sha256=sha(fit),guard_sha256=sha(guard),
  pipeline_sha256=sha(__file__),capture_complete_sha256=sha(capture/'COMPLETE.json'),
  guard_receipt=str(receipt),log=str(log),blas_threads=1,no_native_launch_inside_fit=True)
 write(HERE/'FIT_LAUNCH_v1.json',launch);print(json.dumps(launch),flush=True)
 code=child.wait()
out=RUNS/'propagated_state_response_fit_v1/FITTED_v1.json'
finish=dict(utc=utc(),exit_code=code,guard=json.loads(receipt.read_text()) if receipt.exists() else None,
 fit_sha256=sha(out) if out.exists() else None,fit_status=json.loads(out.read_text())['status'] if out.exists() else None)
write(HERE/'PIPELINE_FINAL_v1.json',finish);print(json.dumps(finish),flush=True)
assert code==0 and out.exists(),finish

