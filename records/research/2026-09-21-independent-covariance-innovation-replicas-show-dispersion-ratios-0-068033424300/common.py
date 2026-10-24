import hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;D=HERE.parent;P2=D.parent;A=P2/'astra_phase2_reexamination_v1'
FIT=D/'current_edge_nonlinear_transfer_v1';OLD=D/'full_lower_covariance_innovation_supervision_v1'
PARENT=D/'full_lower_covariance_edge_packed_v1/candidate.py'
RUNTIME=A/'openblas_runtime_env_v1/bin/python'
KEYS=('left','right','edge','bias','output');ARMS=('output_only','nonlinear')
def read(p):return json.loads(Path(p).read_text())
def pin(p):
    p=Path(p).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return dict(path=str(p),sha256=h.hexdigest(),bytes=p.stat().st_size)
def check(p):assert pin(p['path'])=={k:p[k] for k in ('path','sha256','bytes')},p['path']
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def seal_check():
    for p in read(HERE/'sources_sealed.json')['pins']:check(p)
def done(name):
    g=read(HERE/f'{name}_watch.json');assert g['completed'] and g['exit_code']==0 and g['whole_group_retired'] and g['max_swap_growth_bytes']==0,g
    return g
def guarded(name,seconds,script,*args):
    with (HERE/f'{name}.log').open('x') as f:
        code=subprocess.run([sys.executable,'-B',str(D/'run_guarded.py'),'--record',str(HERE/f'{name}_watch.json'),'--seconds',str(seconds),str(RUNTIME),'-B',str(HERE/script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT).returncode
    assert code==0,(name,code)
    g=done(name);print(dict(stage=name,wall_s=g['elapsed_s'],rss=g['max_rss_bytes']),flush=True)
def relative(a,b):
    import numpy as np
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
