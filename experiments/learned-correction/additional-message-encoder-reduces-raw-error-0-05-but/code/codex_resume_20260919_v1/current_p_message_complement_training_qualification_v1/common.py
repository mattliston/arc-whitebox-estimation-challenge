import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
D=HERE.parent
A=D.parent/'astra_phase2_reexamination_v1'
N=D/'current_p_message_complement_native_qualification_v1'
SMALL=D/'current_p_message_complement_design_v1'
SOURCE=SMALL/'zero.py'
PARENT_SHA='1bc6c43a49ed5858144d6e90c5bb4cbd4e5797b1abfb66648468b3a536680a90'
BANK=D/'current_p_message_covariance_release_v1/candidate.py'
BANK_SHA='d79463f389a6b6e4172415b183601d327bc13b560fd7096ddbf781dce7a94d0e'
SHARED=A/'current_shared384_causal_p_v1/shared_operator.py'
RUNTIME=Path('/Users/matt/anaconda3/bin/python')
PKEYS=('msg_w','msg_b','enc0_w','enc0_b','enc2_w','enc2_b','enc4_w','enc4_b','head','head_b','gain')
EKEYS=('complement_enc_w',)
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
    r=read(N/'result.json');assert r['completed'] and r['passed'] and r['error'] is None and r['full_native_calls']==10
    check(r['qualification']);check(r['sources'])
    q=read(N/'qualification.json');assert q['passed'] and q['zero_all16_outputs_domains_selectors_states_exact']
    assert q['direct_recurrent_state_ownership_verified'] and q['all_full_calls_18_object_retirement_qualified']
    check(q['unit_result']);assert q['unit_result']==pin(SMALL/'unit_result.json')
    for name in ('native_disabled','native_zero','native_zero_cold100','native_sentinel','qualify'):
        w=read(N/f'{name}_watch.json')
        assert w['completed'] and w['exit_code']==0 and w['error'] is None and w['whole_group_retired'] and not w['remaining_rows']
        assert w['max_swap_growth_bytes']==0 and w['max_rss_bytes']<7_500_000_000
    receipt=read(N/'orchestration_receipt.json')
    assert receipt['tool_terminal_observed'] and receipt['tool_terminal_exit_code']==0 and receipt['controller_and_worker_groups_retired']
    claim=read(N/'run_claim.json')
    for row in processes():
        p=row.strip().split(None,2)
        if len(p)==3:assert int(p[0])!=claim['pid'] and int(p[1])!=claim['pgid'] and str(N/'run.py') not in p[2]
def verify():
    for p in read(HERE/'sources_sealed.json')['pins']:check(p)
    assert sha(SOURCE)==PARENT_SHA and sha(BANK)==BANK_SHA
    native_success()
    assert shutil.disk_usage(HERE).free>=470_000_000_000
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<100_000_000
def clean_watch(name):
    w=read(HERE/f'{name}_watch.json')
    assert w['completed'] and w['exit_code']==0 and w['error'] is None and w['whole_group_retired'] and not w['remaining_rows']
    assert w['max_rss_bytes']<7_500_000_000 and w['max_swap_growth_bytes']==0
    return w
def guarded(name,seconds,script):
    verify();assert not numerical_workers();print(dict(starting=name),flush=True)
    with (HERE/f'{name}.log').open('x') as f:
        rc=subprocess.run([sys.executable,'-B',str(HERE/'guard.py'),'--record',str(HERE/f'{name}_watch.json'),
            '--seconds',str(seconds),str(RUNTIME),'-B',str(HERE/script)],stdout=f,stderr=subprocess.STDOUT).returncode
    assert rc==0,(name,rc)
    w=clean_watch(name);assert not numerical_workers()
    print(dict(completed=name,wall_s=w['elapsed_s'],rss=w['max_rss_bytes']),flush=True)
