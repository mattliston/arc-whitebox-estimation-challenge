import hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;D=HERE.parent;P2=D.parent
ROOT=Path('/Users/matt/arc-white-box-estimation')
PARENT=D/'full_lower_covariance_edge_packed_v1/candidate.py'
DONOR=D/'readout_residual_runtime_v1/candidate.py'
RUNTIME=P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
CONFIRM=[37,39,42,48,59,63,66,67]
READOUT_TRAIN=[35,36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58]
EDGE_TRAIN=[35,100,150,200,250,300];EDGE_VALID=[400,500]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def verify(p):assert pin(p['path'])=={k:p[k] for k in ('path','sha256','bytes')},p['path']
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def seal_check():
    for p in read(HERE/'sources_sealed.json')['pins']:verify(p)
def done(name):
    r=read(HERE/f'{name}_watch.json')
    assert r['completed'] and r['exit_code']==0 and r['whole_group_retired'] and r['max_swap_growth_bytes']==0,r
    return r
def guarded(name,seconds,script,*args):
    print(dict(starting=name),flush=True)
    with (HERE/f'{name}.log').open('x') as f:
        c=subprocess.run([sys.executable,'-B',str(D/'run_guarded.py'),'--record',str(HERE/f'{name}_watch.json'),'--seconds',str(seconds),str(RUNTIME),'-B','-W','ignore',str(HERE/script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT).returncode
    assert c==0,(name,c)
    r=done(name);print(dict(completed=name,elapsed_s=r['elapsed_s'],peak_group_rss=r['max_rss_bytes']),flush=True)
def relative(a,b):
    import numpy as np
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
