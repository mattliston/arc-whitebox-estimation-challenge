"""Freeze current-source q4 geometry and parent third slices before new MC."""
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from scipy.linalg import eigh
from scipy.special import ndtr
from threadpoolctl import threadpool_limits
import radial_compact_feed_fullpair_v1 as parent

HERE=Path(__file__).resolve().parent
OUT=HERE/'residual_one_high_inputs_r35_l12_v1.npz'
RECEIPT=OUT.with_suffix('.json')
assert not OUT.exists() and not RECEIPT.exists()
assert json.loads((HERE/'residual_one_high_oracle_certificate_v1.json').read_text())['passed']
W=np.load(HERE/'rows_current_transport_shadow_v2/row_0035/weights.npy')
truth=np.load(HERE.parent/'public_g19_attribution_v1/mps_cov2_rank0035.npz')
assert int(truth['P'])==8000000
mt=truth['mc_post'].astype(np.float64);ct=truth['mc_post_cov'].astype(np.float64)
old=np.load(HERE/'rows_covariance_innovation_probe_v2/row_0035/data.npz')
indices=np.arange(0,1024,32)
saved={};stats={};start=time.time()
class Done(Exception):pass


def hook(l,m,C,K21,aux,w):
    if l==12:
        e=ct[l]-C;e=(e+e.T)/2
        assert np.array_equal(e.astype(np.float32),old['target'][4])
        host=(C+C.T)/2
        ev,U=eigh(host,subset_by_index=(1020,1023))
        assert ev.min()>0
        U*=np.where(U[np.argmax(np.abs(U),axis=0),np.arange(4)]<0,-1.,1.)
        saved.update(U=U,host_covariance=host,eigenvalues=ev)
        stats['host_covariance_top4_trace_fraction']=float(ev.sum()/np.trace(host))
    return mt[l]-m if l>=1 else np.zeros_like(m)


def cov_hook(l,C):return ct[l].copy()


def post_probe(l,t):
    if l!=12:return
    p=t['p'].astype(np.float64);lam=t['lam'].astype(np.float64)
    star=t['star'].astype(np.float64);b=t['b'];B=t['Bpost']
    U=saved['U'];slices=[]
    for u in U.T:
        M=(p*((p.T@u)*lam))@p.T
        M+=(star*u[None,:]+u[:,None]*star.T+np.diag(star.T@u))/3
        bu=B@u
        M+=(b@u)*B+np.outer(b,bu)+np.outer(bu,b)
        slices.append(M)
    wn=W[l+1].astype(np.float64);pp=wn.T@p
    K=(pp[indices]**2*lam)@pp.T
    S=wn.T@star;V=wn.T
    K+=((2*S[indices]*V[indices])@V.T+(V[indices]**2)@S.T)/3
    Bn=wn.T@B@wn;bn=wn.T@b
    K+=np.diag(Bn)[indices,None]*bn[None,:]+2*Bn[indices]*bn[indices,None]
    reference=np.load(HERE/'third_covariance_edge_r35_l12_v1.npz')['baseline']
    assert np.allclose(K,reference,rtol=0,atol=2e-12)
    mu=wn.T@mt[l];cov=wn.T@ct[l]@wn;s=np.sqrt(np.diag(cov));alpha=mu/s
    response=.5*np.exp(-alpha[indices]**2/2)/(np.sqrt(2*np.pi)*s[indices])
    consumer=response[:,None]*ndtr(alpha)[None,:]
    saved.update(model_slices=np.stack(slices),model_next_K21=K,indices=indices,
                 next_weight=wn,consumer_weight=consumer,
                 post_shift=mt[l],next_pre_shift=truth['mc_pre_m'][l+1].astype(np.float64),
                 post_covariance=ct[l],next_pre_covariance=truth['mc_pre_cov'][l+1].astype(np.float64))
    stats.update(rank=35,layer=l,q=4,post_cp_columns=p.shape[1],
                 original_covariance_target_byte_exact=True,
                 parent_plane_max_difference=float(np.max(np.abs(K-reference))))
    raise Done


source=parent.PREDICT_SOURCE
for a,b in (
    ('gate_floor=.001):','gate_floor=.001, cov_hook=None):'),
    ('            m = m+dm','            m = m+dm\n        if cov_hook is not None:\n            C = cov_hook(l,C)'),
    ('        radial = radial_new','        radial = radial_new\n        if cov_hook is not None:\n            C = cov_hook(l,C)'),
    ('        previous_alive = alive','        post_probe(l,locals())\n        previous_alive = alive')):
    assert source.count(a)==1;source=source.replace(a,b)
env=dict(parent.env,post_probe=post_probe);exec(source,env)
with threadpool_limits(limits=4):
    try:env['predict'](W,hook=hook,cov_hook=cov_hook,gate_floor=.01,cov32=True,metric='covariance')
    except Done:pass
assert 'model_slices' in saved
np.savez(OUT,**saved)
stats.update(wall_s=time.time()-start,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
             output_sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),
             source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             qualification='Current source-v2 oracle previous m/C. Host covariance eigenspace frozen before any new target. Full retained parent tensor, not compressed. No raw score.')
RECEIPT.write_text(json.dumps(stats,indent=2)+'\n');print(json.dumps(stats,indent=2),flush=True)
