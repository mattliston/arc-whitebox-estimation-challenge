import hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;D=HERE.parent;P2=D.parent;A=P2/'astra_phase2_reexamination_v1'
OLD=D/'full_lower_covariance_innovation_supervision_v1'
PARENT=D/'full_lower_covariance_edge_packed_v1/candidate.py'
RUNTIME=A/'openblas_runtime_env_v1/bin/python'
TRAIN=(35,100,150,200,250,300);VALID=(400,500)
KEYS=('left','right','edge','bias','output')
STORE=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def check(p):assert pin(p['path'])=={k:p[k] for k in ('path','sha256','bytes')},p['path']
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def verify(name='sources_sealed.json'):
    s=read(HERE/name)
    for p in s['pins']:check(p)
    return s
def weights(rank):
    rows=read(STORE/'manifest.json')['roles']['observer_fit']['row_manifests']
    return (STORE/rows[rank]['path']).parent/'weights.npy'
def truth(rank):return P2/f'public_g19_attribution_v1/mps_cov2_rank{rank:04d}.npz'
def done(name):
    g=read(HERE/f'{name}_watch.json')
    assert g['completed'] and g['exit_code']==0 and g['whole_group_retired'] and g['max_swap_growth_bytes']==0,g
    return g
def guarded(name,seconds,script,*args):
    with (HERE/f'{name}.log').open('x') as f:
        c=subprocess.run([sys.executable,'-B',str(D/'run_guarded.py'),'--record',str(HERE/f'{name}_watch.json'),'--seconds',str(seconds),str(RUNTIME),'-B',str(HERE/script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT).returncode
    assert c==0,(name,c)
    g=done(name);print(dict(stage=name,wall_s=g['elapsed_s'],rss=g['max_rss_bytes']),flush=True)
def tangent_cov(pre,e,E,W):
    import numpy as np
    u=W.T@e;V=W.T@E@W;z=np.diag(V)
    s=pre['sigma'];g=pre['gate'];p=pre['pdf'];a=pre['alpha'];mg=pre['mg']
    dg=p*u/s-a*p*z/(2*s*s)
    T=(g[:,None]*g[None,:])*V+pre['C_pre']*(dg[:,None]*g[None,:]+g[:,None]*dg[None,:])
    np.fill_diagonal(T,(g-mg*p/s)*z+2*mg*(1-g)*u)
    return (T+T.T)*.5
