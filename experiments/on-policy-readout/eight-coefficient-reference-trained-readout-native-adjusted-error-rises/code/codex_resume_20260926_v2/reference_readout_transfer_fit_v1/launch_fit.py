import subprocess
from common import *

LIGHT=ROOT/'.venv/bin/python'
NATIVE=P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
GUARD=P2/'codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py'
verify(); guard('assemble')
assert (HERE/'features_frozen.json').exists()
codes=['common.py','prepare_targets.py','fit.py','verify_fit.py','native_train.py','assess_train.py','launch_fit.py']
once(HERE/'fit_seal.json',dict(sources=[pin(HERE/x) for x in codes+['PROTOCOL.md','features_frozen.json','features.npz']],
    features_frozen=True,final_targets_decoded=False))
def run(name,script,args,runtime):
    print(json.dumps(dict(event='launch',name=name)),flush=True)
    with (HERE/(name+'.log')).open('x') as handle:
        proc=subprocess.run([str(LIGHT),str(GUARD),'--record',str(HERE/(name+'_guard.json')),
            '--seconds','300',str(runtime),str(HERE/script),*args],stdout=handle,stderr=subprocess.STDOUT)
    g=guard(name); assert proc.returncode==0
    print(json.dumps(dict(event='retired',name=name,seconds=g['elapsed_s'],peak_rss=g['max_rss_bytes'])),flush=True)
run('targets','prepare_targets.py',[],LIGHT)
run('fit','fit.py',[],LIGHT)
run('verify_fit','verify_fit.py',[],LIGHT)
assert read(HERE/'fit_verification.json')['passed']
once(HERE/'native_seal.json',dict(sources=[pin(HERE/x) for x in codes+[
    'fit_seal.json','targets.json','targets.npz','fit_result.json','fit_verification.json','readout.npz','candidate.py']],
    candidate_frozen_before_native_score=True))
run('native_parent','native_train.py',['parent'],NATIVE)
run('native_fit','native_train.py',['fit'],NATIVE)
run('assess_train','assess_train.py',[],LIGHT)
print(json.dumps(dict(event='ACTUAL_FAST_TRAIN_COMPLETE',result=read(HERE/'native_train_result.json'))),flush=True)
