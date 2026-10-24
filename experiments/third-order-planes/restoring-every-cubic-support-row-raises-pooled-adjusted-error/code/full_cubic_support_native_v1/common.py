import hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;D=HERE.parent;P2=D.parent;A=P2/'astra_phase2_reexamination_v1'
PARENT=D/'full_lower_covariance_edge_packed_v1/candidate.py'
PARENT_SHA='facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0'
RUNTIME=A/'openblas_runtime_env_v1/bin/python'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def check(p):assert pin(p['path'])==p,p['path']
def seal_check():
    for p in read(HERE/'sources_sealed.json')['pins']:check(p)
def done(name):
    g=read(HERE/f'{name}_watch.json')
    assert g['completed'] and g['exit_code']==0 and g['whole_group_retired'] and g['max_swap_growth_bytes']==0,g
    return g
def guarded(name,seconds,script,*args):
    with (HERE/f'{name}.log').open('x') as f:
        code=subprocess.run([sys.executable,'-B',str(D/'run_guarded.py'),'--record',str(HERE/f'{name}_watch.json'),'--seconds',str(seconds),str(RUNTIME),'-B','-W','ignore',str(HERE/script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT).returncode
    assert code==0,(name,code)
    g=done(name);print(dict(stage=name,wall_s=g['elapsed_s'],rss=g['max_rss_bytes']),flush=True)
