import hashlib,json,subprocess,sys,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;D=HERE.parent;P2=D.parent;A=P2/'astra_phase2_reexamination_v1';OLD=D/'incoming_third_pair_head_v1'
RUNTIME=A/'openblas_runtime_env_v1/bin/python';SOURCE=D/'full_lower_covariance_edge_packed_v1/candidate.py'
PARENT_SHA='facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0'
TRAIN=(57,68,45,59,3,12);VALID=(14,71);LAYERS=tuple(range(8,15));KEYS=('left','right','edge','bias','output')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def check(p):assert pin(p['path'])=={k:p[k] for k in ('path','sha256','bytes')},p['path']
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def verify(name='sources_sealed.json'):
    for p in read(HERE/name)['pins']:check(p)
def resource_floor():
    assert shutil.disk_usage(HERE).free>=470_000_000_000
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<=2_500_000_000
def guarded(name,seconds,script,*args):
    resource_floor();print(dict(starting=name),flush=True)
    with (HERE/f'{name}.log').open('x') as f:
        rc=subprocess.run([sys.executable,'-B',str(D/'run_guarded.py'),'--record',str(HERE/f'{name}_watch.json'),'--seconds',str(seconds),str(RUNTIME),'-B','-W','ignore',str(HERE/script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT).returncode
    r=read(HERE/f'{name}_watch.json');assert rc==0 and r['completed'] and r['whole_group_retired'] and r['max_swap_growth_bytes']==0,(name,r)
    print(dict(completed=name,wall_s=r['elapsed_s'],peak_rss=r['max_rss_bytes']),flush=True);resource_floor()
