"""First late-layer target-free current-host truncation/precision round.

Fixed cells35/100 L12/14. Stop on first late failure. This is not an
estimator accuracy test, fit or metering run. Exact full-C reference is
offline only. No truth or teacher labels are loaded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
import radial_compact_feed_fullpair_v1 as parent
from radial_features import features
from radial_gnn_numpy import gnn_step
from fourth_trace_full_queries_v1 import fused_query,complete_trace_query
from fourth_trace_fused_cp_queries_v1 import FusedCPQueries
from fourth_trace_third_queries_v1 import ThirdContraction,DistinctQueries
from fourth_trace_full_edge_reference_v1 import FullEdgeThird,full_edge_query,zero_repeated_source_diagonal
from fourth_repeated_compiled_v1 import repeated
from fourth_trace_core_projection_v1 import project
from thin_fourth_wide_helpers_v1 import PAIR,mixed,modes,radial_query,core_query

HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--rank',type=int,required=True,choices=(35,100));args=ap.parse_args()
OUT=HERE/f'thin_fourth_wide_r{args.rank}_v1.json'
assert not OUT.exists()
assert json.loads((HERE/'fourth_full_edge_reference_certificate_v2.json').read_text())['passed']
if args.rank==100:
    assert json.loads((HERE/'thin_fourth_wide_r35_v1.json').read_text())['passed']
W=np.load(HERE/'rows_current_transport_shadow_v2'/f'row_{args.rank:04d}'/'weights.npy')
P=dict(np.load(HERE/'payload_compact_feed_fullpair_v1.npz'))
hstate=np.zeros((1024,P['head'].shape[1]),np.float32)
records=[];started=time.time()


class FailedGate(Exception):pass


def rel(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))


def probe(l,tape):
    if l not in (12,14):return
    t0=time.time();n=1024
    C=tape['Cpre'].copy();pp=tape['pp'].astype(np.float64);lam=tape['lam'].astype(np.float64)
    star=tape['star'].astype(np.float64);ws=tape['w'].copy();b=tape['bpre'].copy();B=tape['Bpre'].copy()
    cp=(pp*pp)@(pp*lam).T
    wp=ws.T;wa=wp@star
    sp=(2*(wa*wp)@wp.T+(wp*wp)@wa.T)/3
    K=cp+sp+np.diag(B)[:,None]*b[None,:]+2*B*b[:,None]
    radial=tape['radial'];v=np.diag(C);A=radial*C
    K22=radial*(np.outer(v,v)+2*C*C)/3
    K31=radial*v[:,None]*C
    m,Cpost,_,rpost,aux=PAIR(tape['mu'],C,K,radial*v*v,K22,'covariance')
    h,wick=aux['h'],aux['wick']
    _,V=modes(C);U,_=modes(Cpost)
    rep=repeated(C,K,K22,K31,A,C,h,wick)
    other=DistinctQueries(ThirdContraction(np.zeros((n,0)),np.zeros(0),star,ws,b,B),K-cp,V)
    lowthird=FusedCPQueries(pp,lam,cp,V,other)
    f=fused_query(C,K,K22,A,C,h,wick,V,lowthird,rep,U)
    RU=complete_trace_query(f,Cpost,aux['post31'],aux['post22'],U)
    reference=FullEdgeThird(pp,lam,star,ws,b,B,K,C)
    ff=full_edge_query(C,K,K22,A,C,h,wick,V,reference,rep,U)
    Rfull=complete_trace_query(ff,Cpost,aux['post31'],aux['post22'],U)
    radial_U=rpost*radial_query(Cpost,U)
    increment_full=Rfull-radial_U;increment_low=RU-radial_U
    query_truncation=rel(increment_low,increment_full)
    trace_R=float(aux['post22'].sum())
    core=project(Cpost,U,RU,trace_R)
    Ac=core['U']@core['V'].T+core['V']@core['U'].T
    projected=core_query(Ac,Cpost,U)+core['alpha']*radial_query(Cpost,U)
    poly_error=rel(projected,RU)
    # Source at the following nonlinearity: push the new anisotropic core
    # and its full covariance metric by the actual next weight matrix.
    wn=W[l+1].astype(np.float64)
    Ap=wn.T@Ac@wn;Cp=wn.T@Cpost@wn
    next_mu=wn.T@m;sv=np.sqrt(np.maximum(np.diag(Cp),1e-10))
    from scipy.special import ndtr
    alpha=next_mu/sv;g=ndtr(alpha);d=np.exp(-alpha*alpha/2)/(np.sqrt(2*np.pi)*sv)
    _,Vp=modes(Cp);Clow=Vp@Vp.T;np.fill_diagonal(Clow,np.diag(Cp))
    # New L12 core is pushed to L13, generates source there, read at L14.
    # L14 core has only finalL15 scalar4 use; no later source is generated.
    if l+2<len(W):
        Wread=W[l+2].astype(np.float64)
        exact_source=zero_repeated_source_diagonal(Ap,Cp,g,d,Wread)
        low_source=zero_repeated_source_diagonal(Ap,Clow,g,d,Wread)
        source_truncation=rel(low_source,exact_source)
    else:
        exact_source=low_source=np.zeros(n)
        source_truncation=None
    # Mixed large GEMMs only, all input values and query directions fixed
    # to isolate arithmetic fidelity before any moving-subspace question.
    cast=mixed
    Cx,Kx,K22x,Ax,Vx,Ux=map(cast,(C,K,K22,A,V,U))
    hx,wx=list(map(cast,h)),list(map(cast,wick))
    px=cast(pp);lx=cast(lam);cpx=(px*px)@(px*lx).T
    wpx=cast(ws).T;wax=wpx@cast(star)
    spx=(2*(wax*wpx)@wpx.T+(wpx*wpx)@wax.T)/3
    Kx=cpx+spx+np.diag(cast(B))[:,None]*cast(b)[None,:]+2*cast(B)*cast(b)[:,None]
    _,Cpostx,_,_,auxx=PAIR(cast(tape['mu']),Cx,Kx,radial*cast(v)*cast(v),K22x,'covariance')
    hx,wx=auxx['h'],auxx['wick']
    mx=ThirdContraction(cast(np.zeros((n,0))),cast(np.zeros(0)),cast(star),cast(ws),cast(b),cast(B))
    tq=FusedCPQueries(px,lx,cpx,Vx,DistinctQueries(mx,Kx-cpx,Vx))
    rx=repeated(Cx,Kx,K22x,cast(K31),Ax,Cx,hx,wx)
    fx=fused_query(Cx,Kx,K22x,Ax,Cx,hx,wx,Vx,tq,rx,Ux)
    Rmix=complete_trace_query(fx,Cpostx,auxx['post31'],auxx['post22'],Ux)
    mixed_query=rel(np.asarray(Rmix)-radial_U,increment_low)
    # Project with mixed products; small linalg and orthogonality dots stay64.
    cm=project(cast(Cpostx),Ux,Rmix,float(auxx['post22'].sum()))
    Am=cm['U']@cm['V'].T+cm['V']@cm['U'].T
    mixed_core=rel(np.asarray(Am),Ac)
    row=dict(rank=args.rank,layer=l,cp_columns=len(lam),active=int(tape['alive'].sum()),
             query_edge_relative_error=query_truncation,source_edge_relative_error=source_truncation,
             polynomial_query_relative_error=poly_error,mixed_query_relative_error=mixed_query,
             mixed_core_relative_error=mixed_core,core_info=core['info'],
             increment_full_norm=float(np.linalg.norm(increment_full)),
             anisotropic_core_norm=float(np.linalg.norm(Ac)),source_diagonal_norm=float(np.linalg.norm(exact_source)),
             wall_s=time.time()-t0)
    row['passed']=query_truncation<=.15 and (source_truncation is None or source_truncation<=.15) and poly_error<=1e-6 and mixed_query<=.01 and mixed_core<=.01
    records.append(row)
    artifact=HERE/f'thin_fourth_wide_cell_r{args.rank}_l{l}_v1.npz';assert not artifact.exists()
    np.savez(artifact,Cpre=C,Cpost=Cpost,core=Ac,U=U,modes=V,full_RU=Rfull,thin_RU=RU,mixed_RU=np.asarray(Rmix),
             exact_source=exact_source,low_source=low_source,core_alpha=core['alpha'])
    print(json.dumps(row),flush=True)
    if not row['passed']:raise FailedGate


def hook(l,m,C,K21,aux,w):
    global hstate
    delta,hstate=gnn_step(l,features(l,m,C,K21,aux,w),aux['sigma'],w,hstate,P)
    return delta


src=parent.PREDICT_SOURCE
needle='        host = m.copy()'
assert src.count(needle)==1
src=src.replace(needle,"        probe(l,locals())\n"+needle)
env=dict(parent.env,probe=probe);exec(src,env)
with threadpool_limits(limits=3):
    try:env['predict'](W,hook=hook,gate_floor=.01)
    except FailedGate:pass
passed=len(records)==2 and all(r['passed'] for r in records)
result=dict(rank=args.rank,records=records,passed=passed,stopped_on_first_failure=bool(records and not records[-1]['passed']),
            wall_s=time.time()-started,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            qualification='Target-free frozen current-host diagnostic, exact retained-tensor full-C reference NOT MC truth. No model update, score measurement, training or submission.')
OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
