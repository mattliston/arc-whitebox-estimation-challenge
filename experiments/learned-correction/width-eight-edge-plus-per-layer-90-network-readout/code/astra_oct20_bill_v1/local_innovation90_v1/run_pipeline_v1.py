"""Original guarded fit after exact TRAIN90 capture completion; no native launch."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;RUNS=Path('/home/arcworker/arc-runs')
A=HERE.parents[1]/'fable_oct19_kc_v1/agent_astra_joint_v1'
sys.path.insert(0,str(A/'cloud_linux_capacity_v7'));from common_linux_v7 import thread_env
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
fit=HERE/'fit_local90_v1.py';assert sha(fit)=='e0c431f98a071e240706f07560955ad64ada870c436d0b4da4d19efab6ad9f49'
prereg=HERE/'PREREG_v1.json';assert sha(prereg)=='970f361d7df4d93afac33dff551080ebe6a36257341057cde0ade59382eba143'
guard=A/'cloud_linux_capacity_v7/guard_linux_v7.py'
assert sha(guard)=='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
capture=RUNS/'full_state_response_train90_v1';deadline=time.monotonic()+21600
while not (capture/'COMPLETE.json').exists():
    assert time.monotonic()<deadline,'Capture wait timeout'
    time.sleep(15)
done=json.loads((capture/'COMPLETE.json').read_text());assert done['successful'],done
log=RUNS/'local_innovation90_fit_v1.log';receipt=RUNS/'local_innovation90_fit_v1_GUARD.json'
assert not log.exists() and not receipt.exists()
env=os.environ.copy();env.update(thread_env(1));env['ARC_BLAS_THREADS']='1'
py='/home/arcworker/arc-envs/judge/bin/python'
cmd=[py,'-B',str(guard),'--receipt',str(receipt),'--state-dir',str(RUNS/'admission'),'--max-active','36',
    '--cgroup-root','/sys/fs/cgroup/arc-estimator-v5','--',py,'-B',str(fit)]
with log.open('x') as stream:
    proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
    launch=dict(utc=utc(),guard_pid=proc.pid,command=cmd,fit_sha256=sha(fit),prereg_sha256=sha(prereg),
        guard_sha256=sha(guard),capture_completion_sha256=sha(capture/'COMPLETE.json'),
        pipeline_sha256=sha(__file__),log=str(log),guard_receipt=str(receipt),blas_threads=1)
    write(HERE/'FIT_LAUNCH_v1.json',launch);print(json.dumps(launch),flush=True);code=proc.wait()
result=RUNS/'local_innovation90_fit_v1/FITTED_v1.json'
finish=dict(utc=utc(),exit_code=code,guard=json.loads(receipt.read_text()) if receipt.exists() else None,
    fit_sha256=sha(result) if result.exists() else None,fit_status=json.loads(result.read_text())['status'] if result.exists() else None)
write(HERE/'FIT_PIPELINE_FINAL_v1.json',finish);print(json.dumps(finish),flush=True)
assert code==0 and result.exists(),finish

