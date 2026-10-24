import hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
D=HERE.parent; P2=D.parent; A=P2/'astra_phase2_reexamination_v1'
SOURCE=D/'full_lower_covariance_edge_packed_v1/candidate.py'
PARENT_SHA='facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0'
RUNTIME=A/'openblas_runtime_env_v1/bin/python'
STORE=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
TRAIN=(35,100,150,200,250,300); VALID=(400,500)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def write(p,obj):
    with Path(p).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
def read(p):return json.loads(Path(p).read_text())
def verify(name='sources_sealed.json'):
    s=read(HERE/name)
    for p in s['pins']:assert sha(p['path'])==p['sha256'],p['path']
    return s
def weights(rank):
    m=read(STORE/'manifest.json')['roles']['observer_fit']['row_manifests']
    return (STORE/m[rank]['path']).parent/'weights.npy'
def truth(rank):return P2/f'public_g19_attribution_v1/mps_cov2_rank{rank:04d}.npz'
def guarded(name,seconds,args):
    with (HERE/f'{name}.log').open('x') as f:
        p=subprocess.run([sys.executable,str(D/'run_guarded.py'),'--record',str(HERE/f'{name}_watch.json'),'--seconds',str(seconds),*map(str,args)],stdout=f,stderr=subprocess.STDOUT)
    r=read(HERE/f'{name}_watch.json')
    assert p.returncode==0 and r['completed'] and r['whole_group_retired'] and r['max_swap_growth_bytes']==0,(name,r)
    print(json.dumps(dict(stage=name,completed=True,rss=r['max_rss_bytes'],wall=r['elapsed_s'])),flush=True)
def tangent_cov(pre,e,E,W):
    import numpy as np
    u=W.T@e;V=W.T@E@W;z=np.diag(V)
    s=pre['sigma'];g=pre['gate'];p=pre['pdf'];a=pre['alpha'];mg=pre['mg']
    dg=p*u/s-a*p*z/(2*s*s)
    T=(g[:,None]*g[None,:])*V+pre['C_pre']*(dg[:,None]*g[None,:]+g[:,None]*dg[None,:])
    np.fill_diagonal(T,(g-mg*p/s)*z+2*mg*(1-g)*u)
    return (T+T.T)*.5
def basis(d,t):
    import numpy as np
    F=d['F'];R=d['R'];S=d['S']
    h=(F@t['left'])[:,None,:]+(F@t['right'])[None,:,:]+t['bias']
    h=h+R[:,:,None]*t['edge'][0]+S[:,:,None]*t['edge'][1]+S.T[:,:,None]*t['edge'][2]
    h=np.tanh(h).astype(np.float64)
    raw=h*d['pf'].astype(np.float64)[:,None,None]*d['gf'].astype(np.float64)[None,:,None]
    B=.5*(raw+raw.transpose(1,0,2))
    B*=d['sf'].astype(np.float64)[:,None,None]*d['sf'].astype(np.float64)[None,:,None]
    B*=float(t['cov_factor'])
    return B
