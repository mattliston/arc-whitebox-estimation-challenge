"""Bounded, serial native objective experiment metadata and execution helpers."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
D = HERE.parent
P2 = D.parent
A = P2/'astra_phase2_reexamination_v1'
MODEL_DIR = D/'joint_incoming_moment_head_v1'
SOURCE = D/'current_p_release_v1/candidate.py'
PARENT_SHA = '8260ddb1f4966cc6acc4c318a53835981fde1b839d946ac10604b18325d2a040'
RUNTIME = A/'openblas_runtime_env_v1/bin/python'

def read(p):
    return json.loads(Path(p).read_text())

def pin(p):
    p = Path(p).resolve()
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1 << 20), b''):
            h.update(block)
    return dict(path=str(p),sha256=h.hexdigest(),bytes=p.stat().st_size)

def sha(p):
    return pin(p)['sha256']

def check(p):
    assert pin(p['path']) == {k:p[k] for k in ('path','sha256','bytes')},p['path']

def write(p,d):
    with Path(p).open('x') as f:
        json.dump(d,f,indent=2,allow_nan=False)
        f.write('\n')

def verify():
    for p in read(HERE/'sources_sealed.json')['pins']:
        check(p)
    assert shutil.disk_usage(HERE).free >= 470_000_000_000
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()) < 100_000_000

def clean_watch(name):
    w=read(HERE/f'{name}_watch.json')
    assert w['completed'] and w['exit_code']==0 and w['whole_group_retired'] and not w['remaining_rows']
    assert w['max_swap_growth_bytes']==0 and w['max_rss_bytes']<7_500_000_000
    return w

def admission():
    verify()
    clean_watch('build')
    b=read(HERE/'build.json')
    assert b['completed'] and b['exact_source_inverse'] and b['all_parameter_bytes_exact']
    assert b['parent']==pin(SOURCE)
    for k in ('parent','candidate','null_candidate','model'):
        check(b[k])
    return b

def stage_ready(stage):
    assert stage=='development'
    admission()

def guarded(name,seconds,script,*args):
    verify()
    print(dict(starting=name),flush=True)
    with (HERE/f'{name}.log').open('x') as f:
        rc=subprocess.run([sys.executable,'-B',str(HERE/'guard.py'),
            '--record',str(HERE/f'{name}_watch.json'),'--seconds',str(seconds),
            str(RUNTIME),'-B',str(HERE/script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT).returncode
    assert rc==0,(name,rc)
    w=clean_watch(name)
    print(dict(completed=name,elapsed_s=w['elapsed_s'],peak_rss_bytes=w['max_rss_bytes']),flush=True)
    return w
