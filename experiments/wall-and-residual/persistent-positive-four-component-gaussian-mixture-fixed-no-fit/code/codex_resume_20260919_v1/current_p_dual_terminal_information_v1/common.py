import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
D=HERE.parent
P2=D.parent
A=P2/'astra_phase2_reexamination_v1'
SOURCE=D/'current_p_release_v1/candidate.py'
RUNTIME=A/'openblas_runtime_env_v1/bin/python'
TRAIN=[r for r in range(35,59) if r not in (37,39,42,48)]
PARENT_NEW=[36,38,56,57,58]
PACKET=('hplus','hminus','sigma','host','delta','minus_delta','minus_host','mu')

def read(p):return json.loads(Path(p).read_text())
def pin(p):
    p=Path(p).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def check(p):assert pin(p['path'])=={k:p[k] for k in ('path','bytes','sha256')},p['path']
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def verify():
    for p in read(HERE/'sources_sealed.json')['pins']:check(p)
    assert shutil.disk_usage(HERE).free>=470_000_000_000
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<100_000_000
def clean_watch(name):
    w=read(HERE/f'{name}_watch.json')
    assert w['completed'] and w['exit_code']==0 and w['whole_group_retired'] and not w['remaining_rows']
    assert w['max_swap_growth_bytes']==0 and w['max_rss_bytes']<7_500_000_000
    return w
def guarded(name,seconds,script,*args):
    verify();print(dict(starting=name),flush=True)
    with (HERE/f'{name}.log').open('x') as f:
        rc=subprocess.run([sys.executable,'-B',str(HERE/'guard.py'),'--record',str(HERE/f'{name}_watch.json'),
             '--seconds',str(seconds),str(RUNTIME),'-B',str(HERE/script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT).returncode
    assert rc==0,(name,rc)
    w=clean_watch(name)
    print(dict(completed=name,wall_s=w['elapsed_s'],rss=w['max_rss_bytes']),flush=True)
def baseline(rank):
    meta=read(HERE/'inputs.json')
    if str(rank) in meta['baselines']:return meta['baselines'][str(rank)]
    q=read(HERE/f'parent_{rank}/result.json')
    assert q['source']==pin(SOURCE) and len(q['records'])==1
    return dict(record=q['records'][0],source=q['source'],receipt=pin(HERE/f'parent_{rank}/result.json'))
