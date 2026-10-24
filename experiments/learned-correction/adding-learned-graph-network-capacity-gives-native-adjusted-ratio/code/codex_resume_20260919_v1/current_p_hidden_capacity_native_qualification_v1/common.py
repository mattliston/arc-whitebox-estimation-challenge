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
PARENT=D/'current_p_release_v1/candidate.py'
PARENT_SHA='8260ddb1f4966cc6acc4c318a53835981fde1b839d946ac10604b18325d2a040'
RUNTIME=A/'openblas_runtime_env_v1/bin/python'

def read(p):return json.loads(Path(p).read_text())
def pin(p):
    p=Path(p).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def check(p):assert pin(p['path'])=={k:p[k] for k in ('path','bytes','sha256')},p['path']
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def verify(data=False):
    s=read(HERE/'sources_sealed.json')
    for p in s['pins']+(s['data_pins'] if data else []):check(p)
    assert pin(PARENT)['sha256']==PARENT_SHA
    assert shutil.disk_usage(HERE).free>=470_000_000_000
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<100_000_000
def process_rows():
    return subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,pgid=,command='],text=True,timeout=5).splitlines()
def numerical_workers():
    exes={str(RUNTIME),'/Users/matt/anaconda3/bin/python',str(D.parent/'data_gate/.client-venv/bin/python')}
    return [r for r in process_rows() if len(a:=r.strip().split(None,3))==4 and a[3].split()[0] in exes]
def admission():
    verify();activation();b=read(HERE/'build.json')
    assert b['completed'] and b['whole_source_inverses_exact'] and b['old_packed_entries_unchanged']
    assert b['parent']==pin(PARENT) and b['new_parameters']==12352 and not b['fitted']
    for k in ('disabled','zero','sentinel'):check(b[k])
    return b
def done(name):
    w=read(HERE/f'{name}_watch.json')
    assert w['completed'] and w['exit_code']==0 and w['error'] is None
    assert w['whole_group_retired'] and not w['remaining_rows']
    assert w['max_rss_bytes']<7_500_000_000 and w['max_swap_growth_bytes']==0
    return w
def stage_ready(stage):
    assert stage=='control';admission();done('unit');assert read(HERE/'unit_result.json')['passed']
def guarded(name,seconds,script,*args):
    verify();assert not numerical_workers()
    print(dict(starting=name),flush=True)
    with (HERE/f'{name}.log').open('x') as log:
        rc=subprocess.run([sys.executable,'-B',str(HERE/'guard.py'),'--record',str(HERE/f'{name}_watch.json'),
            '--seconds',str(seconds),str(RUNTIME),'-B',str(HERE/script),*map(str,args)],stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0,(name,rc)
    w=done(name);assert not numerical_workers()
    print(dict(completed=name,wall_s=w['elapsed_s'],rss=w['max_rss_bytes']),flush=True)

def activation():
    # Deliberately absent during source preparation. A separately audited
    # current-pipeline closeout is required before the first numerical call.
    a=read(HERE/'activation.json')
    assert a['qualification_sources']==pin(HERE/'sources_sealed.json')
    assert a['parent']==pin(PARENT) and a['upstream_pipeline_finalized']
    assert not a['targets_permitted'] and not a['fit_or_upload_permitted']
    check(a['upstream_closeout'])
    assert a['upstream_terminal_receipts'] and a['upstream_claims']
    for p in a['upstream_terminal_receipts']:check(p)
    rows=process_rows()
    parsed=[r.strip().split(None,3) for r in rows]
    parsed=[r for r in parsed if len(r)==4]
    for p in a['upstream_claims']:
        check(p);claim=read(p['path'])
        assert 'pid' in claim and 'pgid' in claim
        assert all(int(r[0])!=claim['pid'] and int(r[2])!=claim['pgid'] for r in parsed)
    forbidden=['current_p_message_covariance_'+s+'/' for s in
        ['fit_v1','native_v1','native_bridge_v1','downstream_v1','downstream_bridge_v1','release_v1']]
    assert not any(any(s in r[3] for s in forbidden) for r in parsed)
    return a
