"""One fixed 32-row full-edge information gate, current coherent source state."""
import argparse
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
from third_covariance_edge_v1 import ThirdCore,row

HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--rank',type=int,choices=(35,100),required=True);args=ap.parse_args()
OUT=HERE/f'third_covariance_edge_r{args.rank}_l12_v1.json'
ART=HERE/f'third_covariance_edge_r{args.rank}_l12_v1.npz'
PROGRESS=HERE/f'third_covariance_edge_r{args.rank}_l12_progress_v1'
assert not OUT.exists() and not ART.exists() and not PROGRESS.exists()
assert json.loads((HERE/'third_covariance_edge_certificate_v1.json').read_text())['passed']
if args.rank==100:assert json.loads((HERE/'third_covariance_edge_r35_l12_v1.json').read_text())['information_gate']
PROGRESS.mkdir()
W=np.load(HERE/'rows_current_transport_shadow_v2'/f'row_{args.rank:04d}'/'weights.npy')
truth=np.load(HERE.parent/'public_g19_attribution_v1'/f'mps_cov2_rank{args.rank:04d}.npz')
assert int(truth['P'])==8000000
mt=truth['mc_post'].astype(np.float64);ct=truth['mc_post_cov'].astype(np.float64)
old=np.load(HERE/'rows_covariance_innovation_probe_v2'/f'row_{args.rank:04d}'/'data.npz')
indices=np.arange(0,1024,32);pre={};answer={}
start=time.time()
class Done(Exception):pass


def pre_probe(l,t):
    if l!=12:return
    w=t['w'];star=t['star'].astype(np.float64)
    pre.update(p=t['pp'].astype(np.float64),lam=t['lam'].astype(np.float64),U=w.T@star,V=w.T.copy(),
               b=t['bpre'].copy(),B=t['Bpre'].copy(),K=t['K21'].copy(),C=t['Cpre'].copy(),alive=t['alive'].copy())


def hook(l,m,C,K21,aux,w):
    if l==12:
        e=ct[l]-C;e=(e+e.T)/2
        assert np.array_equal(e.astype(np.float32),old['target'][4])
        pre['g']=aux['gate'].copy()*pre['alive'];pre['d']=aux['pdf']/aux['sigma']*pre['alive']
    return mt[l]-m if l>=1 else np.zeros_like(m)


def cov_hook(l,C):return ct[l].copy()


def post_probe(l,t):
    if l!=12:return
    wn=W[l+1].astype(np.float64)
    g,d,C=pre['g'],pre['d'],pre['C']
    ev,vec=eigh((C+C.T)/2,subset_by_index=(1020,1023))
    assert ev.min()>0
    modes=vec*np.sqrt(ev)[None,:]
    core=ThirdCore(**{k:pre[k] for k in ('p','lam','U','V','b','B','K')})
    # This full current post tensor is the baseline being corrected. All
    # source inputs and queried output indices are frozen before targets.
    pp=wn.T@t['p'].astype(np.float64);lam=t['lam'].astype(np.float64)
    K=(pp[indices]**2*lam)@pp.T
    U=wn.T@t['star'].astype(np.float64);V=wn.T
    K+=((2*U[indices]*V[indices])@V.T+(V[indices]**2)@U.T)/3
    B=wn.T@t['Bpost']@wn;b=wn.T@t['b']
    K+=np.diag(B)[indices,None]*b[None,:]+2*B[indices]*b[indices,None]
    frozen={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in pre.items()}
    corrections=[];approximations=[]
    for j,index in enumerate(indices):
        tick=time.time()
        full,low=row(core,C,g,d,wn,int(index),modes)
        assert np.isfinite(full).all() and np.isfinite(low).all()
        corrections.append(full);approximations.append(low)
        np.savez(PROGRESS/f'query_{int(index):04d}.npz',full=full,top4=low)
        print(json.dumps(dict(rank=args.rank,layer=l,query=j+1,total=len(indices),index=int(index),wall_s=time.time()-tick)),flush=True)
    correction=np.stack(corrections);approximation=np.stack(approximations)
    planes_path=HERE.parent/'public_g19_attribution_v1'/f'mps_planes_rank{args.rank:04d}.npz'
    target=np.load(planes_path)[f'K21_{l+1}'].astype(np.float64)[indices]
    target_diagonal=truth['mc_pre_k3'][l+1]
    diagonal_discrepancy=float(np.linalg.norm(target[np.arange(len(indices)),indices]-target_diagonal[indices])/np.linalg.norm(target_diagonal[indices]))
    assert diagonal_discrepancy<.002
    mu=wn.T@mt[l];cov=wn.T@ct[l]@wn;s=np.sqrt(np.diag(cov));a=mu/s
    response=.5*np.exp(-a[indices]**2/2)/(np.sqrt(2*np.pi)*s[indices])
    consumer_weight=response[:,None]*ndtr(a)[None,:]
    error=target-K
    def metrics(weight):
        e=error*weight;f=correction*weight;q=approximation*weight
        return dict(natural_error_ratio=float(np.sum((e-f)**2)/np.sum(e*e)),
                    correction_error_cosine=float(np.sum(e*f)/np.sqrt(np.sum(e*e)*np.sum(f*f))),
                    correction_relative_norm=float(np.linalg.norm(f)/np.linalg.norm(e)),
                    fixed_top4_relative_error=float(np.linalg.norm(q-f)/np.linalg.norm(f)),
                    baseline_error_mse=float(np.mean(e*e)),correction_mse=float(np.mean(f*f)))
    unweighted=metrics(1.);weighted=metrics(consumer_weight)
    info=weighted['natural_error_ratio']<=.95 and weighted['correction_error_cosine']>=.4 and unweighted['natural_error_ratio']<=1
    cheap=max(unweighted['fixed_top4_relative_error'],weighted['fixed_top4_relative_error'])<=.15
    answer.update(rank=args.rank,layer=l,queried_indices=indices.tolist(),pre_cp_columns=pre['p'].shape[1],post_cp_columns=pp.shape[1],
                  source_inputs_frozen_before_target=frozen,original_covariance_target_byte_exact=True,
                  true_plane_sha256=hashlib.sha256(planes_path.read_bytes()).hexdigest(),true_K21_diagonal_relative_discrepancy=diagonal_discrepancy,
                  top4_covariance_trace_fraction=float(ev.sum()/np.trace(C)),unweighted=unweighted,leading_consumer_weighted=weighted,
                  information_gate=info,top4_approximation_gate=cheap)
    np.savez(ART,indices=indices,baseline=K,target=target,correction=correction,top4=approximation,consumer_weight=consumer_weight)
    raise Done


source=parent.PREDICT_SOURCE
for a,b in (
    ('gate_floor=.001):','gate_floor=.001, cov_hook=None):'),
    ('        m, C, target, radial_new, aux = nonlinear','        pre_probe(l,locals())\n        m, C, target, radial_new, aux = nonlinear'),
    ('            m = m+dm','            m = m+dm\n        if cov_hook is not None:\n            C = cov_hook(l,C)'),
    ('        radial = radial_new','        radial = radial_new\n        if cov_hook is not None:\n            C = cov_hook(l,C)'),
    ('        previous_alive = alive','        post_probe(l,locals())\n        previous_alive = alive')):
    assert source.count(a)==1;source=source.replace(a,b)
env=dict(parent.env,pre_probe=pre_probe,post_probe=post_probe);exec(source,env)
with threadpool_limits(limits=4):
    try:env['predict'](W,hook=hook,cov_hook=cov_hook,gate_floor=.01,cov32=True,metric='covariance')
    except Done:pass
assert answer
answer.update(wall_s=time.time()-start,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
              qualification='One fixed layer, 32 predetermined next-K21 rows, natural coefficient1. Full-edge offline queries include exact current CP/star/trace and distinct removal. Oracle previous mean/covariance; same-source8M plane labels; no fitted coefficient, deployable cost, rollout or final score.')
OUT.write_text(json.dumps(answer,indent=2)+'\n');print(json.dumps(answer,indent=2),flush=True)
