import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;D=HERE.parent/'codex_resume_20260919_v1'
A=D.parent/'astra_phase2_reexamination_v1'
BI=D/'learnable_joint_basis_implementation_v1'
N=D/'learnable_joint_basis_bank_native_v1'
OLD=D/'current_p_incoming_moment_readout_training_qualification_v1'
SOURCE=BI/'zero.py'
PARENT_SHA='63bb5d15786995b74d761a6f6690170710ab397497351b67f0cf794401362f94'
BANK=D/'current_p_message_covariance_release_v1/candidate.py'
BANK_SHA='d79463f389a6b6e4172415b183601d327bc13b560fd7096ddbf781dce7a94d0e'
SHARED=A/'current_shared384_causal_p_v1/shared_operator.py'
RUNTIME=Path('/Users/matt/anaconda3/bin/python')
PKEYS=('msg_w','msg_b','enc0_w','enc0_b','enc2_w','enc2_b','enc4_w','enc4_b','head','head_b','gain')
EKEYS=('jr_theta','jr_phi','jr_head','jr_head_b','jr_basis')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def check(p):assert pin(p['path'])=={k:p[k] for k in ('path','bytes','sha256')},p['path']
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def processes():return subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,command='],text=True,timeout=5).splitlines()
def numerical_workers():
    exes={str(RUNTIME),str(A/'openblas_runtime_env_v1/bin/python'),str(D.parent/'data_gate/.client-venv/bin/python')}
    return [r for r in processes() if len(a:=r.strip().split(None,2))==3 and a[2].split()[0] in exes]
def native_success():
    close=read(N/'closeout.json');assert close['passed'] and close['full_native_calls']==16 and close['tool_exit_code']==0 and close['controller_and_all_groups_retired']
    ctl=read(N/'controller_result.json');assert ctl['completed'] and ctl['passed'] and ctl['error'] is None
    q=read(N/'qualification.json');assert q['passed'] and q['full_native_calls']==16 and q['identity_basis_old_sentinel_all16_bytes_exact'] and q['basis_zero_bank_exact']
    claim=read(N/'run_claim.json')
    for row in processes():
        p=row.strip().split(None,2)
        if len(p)==3:assert int(p[0])!=claim['pid'] and int(p[1])!=claim['pgid'] and str(N/'run.py') not in p[2]
def verify():
    return True

def clean_watch(name):
    w=read(HERE/f'{name}_watch.json')
    assert w['completed'] and w['exit_code']==0 and w['error'] is None and w['whole_group_retired'] and not w['remaining_rows']
    assert w['rss_trigger_bytes']==(1_000_000_000 if name=='unit' else 7_500_000_000)
    assert w['max_rss_bytes']<w['rss_trigger_bytes'] and w['max_swap_growth_bytes']==0
    return w
def guarded(name,seconds,script):
    verify();assert not numerical_workers();print(dict(starting=name),flush=True)
    with (HERE/f'{name}.log').open('x') as f:
        rc=subprocess.run([sys.executable,'-B',str(HERE/'guard.py'),'--record',str(HERE/f'{name}_watch.json'),
            '--seconds',str(seconds),'--rss-limit',str(1_000_000_000 if name=='unit' else 7_500_000_000),str(RUNTIME),'-B',str(HERE/script)],stdout=f,stderr=subprocess.STDOUT).returncode
    assert rc==0,(name,rc)
    w=clean_watch(name);assert not numerical_workers()
    print(dict(completed=name,wall_s=w['elapsed_s'],rss=w['max_rss_bytes']),flush=True)
