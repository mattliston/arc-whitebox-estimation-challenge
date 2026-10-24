import hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;D=HERE.parent;P2=D.parent
PARENT=D/'full_lower_covariance_edge_packed_v1/candidate.py'
PARENT_SHA='facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0'
RUNTIME=P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
EPS=1/16;TRAIN=[36,38,40,41,43,44]
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def verify(p):assert pin(p['path'])==p,p['path']
def seal_check():
    for p in read(HERE/'sources_sealed.json')['pins']:verify(p)
    assert sha(PARENT)==PARENT_SHA
def guarded(name,seconds,args):
    print(dict(starting=name),flush=True);watch=HERE/f'{name}_watch.json'
    cmd=[sys.executable,'-B',str(D/'run_guarded.py'),'--record',str(watch),'--seconds',str(seconds),*map(str,args)]
    with (HERE/f'{name}.log').open('x') as log:r=subprocess.run(cmd,cwd=D,stdout=log,stderr=subprocess.STDOUT)
    g=read(watch);assert r.returncode==0 and g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0,(name,g)
    print(dict(completed=name,peak_rss_bytes=g['max_rss_bytes'],elapsed_s=g['elapsed_s']),flush=True)
    return g
