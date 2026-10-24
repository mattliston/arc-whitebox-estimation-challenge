"""Wait read-only for all16 frozen FD arms, then one guarded fit; no native follow-up."""
import hashlib,json,os,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
V=HERE.parent
RUNS=Path('/home/arcworker/arc-runs')
GUARD=V.parent/'fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/guard_linux_v7.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def main():
    build=json.loads((HERE/'BUILD_v1.json').read_text())
    ready=json.loads((HERE/'READY_v1.json').read_text())
    assert sha(HERE/'fit_v1.py')==ready['fit_sha256']=='8e09bbb6d3314c406c089dd15b16af4223dc63ab4a97ce5bd7922cc683530bc1'
    assert sha(GUARD)=='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
    assert not (HERE/'FIT_GUARD_v1.json').exists() and not (HERE/'FROZEN_TRAIN_FIT_v1.json').exists()
    previous=None
    while True:
        pending=[]
        for r in build['records']:
            p=RUNS/r['name']/'COMPLETE.json'
            if not p.exists():pending.append(r['name']);continue
            complete=json.loads(p.read_text())
            assert complete['successful'] and not complete['pending'] and all(v==0 for v in complete['results'].values()),(r['name'],complete)
        if not pending:break
        if len(pending)!=previous:
            print(json.dumps(dict(utc=utc(),pending_fd_arms=len(pending))),flush=True);previous=len(pending)
        time.sleep(20)
    env=dict(os.environ)
    for k in ('ARC_BLAS_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
    cmd=['/home/arcworker/arc-envs/judge/bin/python','-B',str(GUARD),
         '--receipt',str(HERE/'FIT_GUARD_v1.json'),'--state-dir',str(RUNS/'admission'),
         '--max-active','36','--cgroup-root','/sys/fs/cgroup/arc-estimator-v5','--',
         '/home/arcworker/arc-envs/judge/bin/python','-B',str(HERE/'fit_v1.py')]
    print(json.dumps(dict(utc=utc(),event='all64_completed_launch_fixed_fit',command=cmd)),flush=True)
    done=subprocess.run(cmd,env=env)
    assert done.returncode==0,done.returncode
    guard=json.loads((HERE/'FIT_GUARD_v1.json').read_text())
    assert guard['successful'] and guard['whole_group_retired']
    fit=json.loads((HERE/'FROZEN_TRAIN_FIT_v1.json').read_text())
    out=dict(utc=utc(),successful=True,status=fit['status'],train=fit['train_linear_proxy'],
        loo=fit['loo_proxy'],fit_sha256=sha(HERE/'FROZEN_TRAIN_FIT_v1.json'),
        guard_receipt_sha256=sha(HERE/'FIT_GUARD_v1.json'),no_native_followup_launched=True)
    with (HERE/'FIT_PIPELINE_COMPLETE_v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out),flush=True)
if __name__=='__main__':main()

