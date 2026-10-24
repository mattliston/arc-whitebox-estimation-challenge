import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
D=HERE.parent
A=D.parent/'astra_phase2_reexamination_v1'
SOURCE=D/'current_p_release_v1/candidate.py'
PARENT_SHA='8260ddb1f4966cc6acc4c318a53835981fde1b839d946ac10604b18325d2a040'
SHARED=A/'current_shared384_causal_p_v1/shared_operator.py'
RUNTIME=Path('/Users/matt/anaconda3/bin/python')
PKEYS=('msg_w','msg_b','enc0_w','enc0_b','enc2_w','enc2_b','enc4_w','enc4_b','head','head_b','gain')
INACTIVE=('head','head_b','gain')
EKEYS=('left','right','edge','bias','output')

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
def verify():
    for p in read(HERE/'sources_sealed.json')['pins']:check(p)
    assert sha(SOURCE)==PARENT_SHA
    assert shutil.disk_usage(HERE).free>=470_000_000_000
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<100_000_000
def clean_watch(name):
    w=read(HERE/f'{name}_watch.json')
    assert w['completed'] and w['exit_code']==0 and w['whole_group_retired'] and not w['remaining_rows']
    assert w['max_rss_bytes']<7_500_000_000 and w['max_swap_growth_bytes']==0
    return w
def guarded(name,seconds,script):
    verify();print(dict(starting=name),flush=True)
    with (HERE/f'{name}.log').open('x') as f:
        rc=subprocess.run([sys.executable,'-B',str(HERE/'guard.py'),'--record',str(HERE/f'{name}_watch.json'),
            '--seconds',str(seconds),str(RUNTIME),'-B',str(HERE/script)],stdout=f,stderr=subprocess.STDOUT).returncode
    assert rc==0,(name,rc)
    w=clean_watch(name);print(dict(completed=name,wall_s=w['elapsed_s'],rss=w['max_rss_bytes']),flush=True)
