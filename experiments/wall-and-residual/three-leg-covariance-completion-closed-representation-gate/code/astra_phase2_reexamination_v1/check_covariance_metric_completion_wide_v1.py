"""One frozen-bank cell, target-free CP completion query fidelity."""
from pathlib import Path
import hashlib
import json
import resource
import time
import numpy as np
from scipy.special import ndtr
from threadpoolctl import threadpool_limits
import compact_fullpair_final_reference_v1 as parent
from radial_features import features
from radial_gnn_numpy import gnn_step
from covariance_metric_completion_v1 import (
    operator,factors,plane,push_plane,star_from_plane,star_query,complete)

HERE=Path(__file__).resolve().parent
OUT=HERE/'covariance_metric_completion_wide_r35_l12_v1.json'
ART=OUT.with_suffix('.npz')
assert not OUT.exists() and not ART.exists()
assert json.loads((HERE/'covariance_metric_completion_certificate_v1.json').read_text())['passed']
W=np.load(HERE/'rows_current_transport_shadow_v2/row_0035/weights.npy')
P=dict(np.load(HERE/'payload_compact_feed_fullpair_v1.npz'))
state=np.zeros((1024,64),np.float32)
start=time.time();captured={}
class Done(Exception):pass


def hook(l,m,C,K21,aux,w):
    global state
    dm,state=gnn_step(l,features(l,m,C,K21,aux,w),aux['sigma'],w,state,P)
    return dm


def post_probe(l,t):
    if l!=12:return
    for name in ('p','lam','C','m','star','b','Bpost','alive'):
        captured[name]=np.array(t[name],copy=True)
    raise Done


source=parent.source
before='        previous_alive = alive'
assert source.count(before)==1
source=source.replace(before,'        post_probe(l,locals())\n'+before)
env=dict(parent.env,post_probe=post_probe);exec(source,env)
records=[];saves={}
with threadpool_limits(limits=4):
    try:env['predict'](W,hook=hook,gate_floor=.01,cov32=True,metric='covariance')
    except Done:pass
    assert captured
    alive=captured['alive'];idx=np.flatnonzero(alive);n=len(idx)
    C=captured['C'].astype(np.float64);C=(C+C.T)/2
    block=C[np.ix_(idx,idx)]
    sigma=np.sqrt(np.diag(block));R=block/np.outer(sigma,sigma)
    eigenvalues=np.linalg.eigvalsh(R)
    assert np.isfinite(R).all() and eigenvalues.min()>0,('not SPD',float(eigenvalues.min()))
    p=captured['p'].astype(np.float64);lam=captured['lam'].astype(np.float64)
    assert np.max(np.abs(p[~alive]))==0
    smallp=p[idx]
    K=(smallp*smallp*lam)@smallp.T
    scales=sigma[:,None]**2*sigma[None,:]
    Kstandard=K/scales
    H,checkpoints,convergence=complete(R,Kstandard,steps=30,tolerance=1e-10,checkpoints=(2,))
    assert 2 in checkpoints
    wn=W[13].astype(np.float64);rows=np.arange(0,1024,32)
    pp=wn.T@p
    cp_query=(pp[rows]*pp[rows]*lam)@pp.T
    del pp
    old_star=captured['star'].astype(np.float64)
    Squery=star_query(old_star,wn,rows)
    metric=wn.T@captured['Bpost'].astype(np.float64)@wn
    bpre=wn.T@captured['b']
    trace=np.diag(metric)[rows,None]*bpre[None,:]+2*metric[rows]*bpre[rows,None]
    baseline=cp_query+Squery+trace
    mu=wn.T@captured['m'];Cpre=wn.T@C@wn
    sd=np.sqrt(np.maximum(np.diag(Cpre),1e-10));a=mu/sd
    response=.5*np.exp(-a[rows]**2/2)/(np.sqrt(2*np.pi)*sd[rows])
    weight=response[:,None]*ndtr(a)[None,:]
    weighted_norm=float(np.linalg.norm(baseline*weight))
    for name,control in (('practical_two_steps',checkpoints[2]),('reference_30_max',H)):
        u,v=factors(R,control,sigma)
        repeated=plane(u,v);residual=K-repeated
        correction=star_from_plane(residual)
        zero=np.zeros((n,n));dummy=np.eye(n)
        cert=float(np.linalg.norm(repeated+star_query(correction,dummy,np.arange(n))-K)/np.linalg.norm(K))
        assert cert<1e-10,cert
        U=np.zeros((1024,n));V=np.zeros_like(U)
        U[idx]=u;V[idx]=v
        fullstar=np.zeros((1024,1024));fullstar[np.ix_(idx,idx)]=correction
        new=push_plane(U,V,wn,rows)+star_query(fullstar,wn,rows)+Squery+trace
        delta=new-baseline
        weighted=float(np.linalg.norm(delta*weight)/weighted_norm)
        raw=float(np.linalg.norm(delta)/np.linalg.norm(baseline))
        cp_relative=float(np.linalg.norm(delta*weight)/np.linalg.norm(cp_query*weight))
        equation=float(np.linalg.norm(operator(R,control)-Kstandard)/np.linalg.norm(Kstandard))
        row=dict(arm=name,linear_equation_residual=equation,
                 repeated_plane_restoration_relative=cert,
                 next_plane_relative=raw,weighted_next_plane_relative=weighted,
                 weighted_error_relative_to_cp_query=cp_relative,
                 representation_gate=weighted<=.03)
        records.append(row);saves[name]=new
        print(json.dumps(dict(**row,wall_s=time.time()-start)),flush=True)
result=dict(rank=35,post_layer=12,next_layer=13,active=n,cp_columns=p.shape[1],
            correlation_eigen_min=float(eigenvalues.min()),
            correlation_eigen_max=float(eigenvalues.max()),
            pcg_iterations=len(convergence),pcg_convergence=convergence,records=records,
            passed=records[0]['representation_gate'],
            reference_converged=convergence[-1]['residual_relative']<1e-10,
            wall_s=time.time()-start,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            qualification='Target-free rank35 frozen current P-bank post12 CP replacement, original separate star/trace. All current repeated CP entries restored exactly.32 fixed next-W13 rows and Gaussian-pair response weight. TwoPCG sole practical arm;30-max reference is diagnostic. No MC truths, fit, raw score, metered bill, rollout or candidate.')
np.savez(ART,baseline=baseline,cp_query=cp_query,consumer_weight=weight,indices=rows,**saves)
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='pcg_convergence'},indent=2),flush=True)
