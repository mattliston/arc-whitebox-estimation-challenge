"""Scalar metadata helpers and the fixed Gaussian-gain tangent."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
D=HERE.parent; P2=D.parent; A=P2/'astra_phase2_reexamination_v1'
SOURCE=D/'lazy_rotation_objective_v1/candidate.py'
PARENT_SHA='b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c'
STORE=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def write(p,obj):
    with Path(p).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
def verify_seal(name='sources_sealed.json'):
    s=json.loads((HERE/name).read_text())
    for v in s['pins']:assert sha(v['path'])==v['sha256'],v['path']
    return s
def weight_path(rank):
    rows=json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
    return (STORE/rows[rank]['path']).parent/'weights.npy'
def tangent(pre,e,E,W,final=False):
    import numpy as np
    u=W.T@e;V=W.T@E@W;z=np.diag(V)
    s=pre['sigma'];g=pre['gate'];p=pre['pdf'];a=pre['alpha'];mg=pre['mg']
    en=g*u+p*z/(2*s)
    if final:return en,None
    dg=p*u/s-a*p*z/(2*s*s)
    En=(g[:,None]*g[None,:])*V+pre['C_pre']*(dg[:,None]*g[None,:]+g[:,None]*dg[None,:])
    np.fill_diagonal(En,(g-mg*p/s)*z+2*mg*(1-g)*u)
    return en,(En+En.T)*.5
