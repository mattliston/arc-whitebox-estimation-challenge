"""One fixed same-rank firstjoin selector comparison on historical saved operands."""
from pathlib import Path
import os,json,hashlib,time,subprocess,re,resource
import numpy as np
D=Path(__file__).resolve().parent;O=Path("/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1")
P=O/'engineer/a35_join/state/s39_rank0651.npz'
START=time.monotonic();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'))
assert os.getpriority(os.PRIO_PROCESS,0)==10
def free():
 return int(re.search(r'System-wide memory free percentage: (\d+)%',subprocess.check_output(['/usr/bin/memory_pressure'],text=True)).group(1))
entry=free();assert entry>=60
assert sha(P)=='f8d64aec21a167a21e7afd330f82d2acd3c2b508422e6158acb35a295647787d'
with np.load(P,allow_pickle=False)as z:J=z['J_07'];lam=z['lam_07'];rank=int(z['rank_07'])
assert J.shape==(960,6144)and lam.shape==(6144,)and rank==384
w=lam*lam;d=(J*J)@w
baseline=np.sort(np.argsort(-d.astype('d'),kind='stable')[:rank])
anchor=int(np.argsort(-d.astype('d'),kind='stable')[0])
c=np.sum(J,axis=0,dtype='f4');y=J@(w*c);y64=y.astype('d');s=float(np.sum(y64))
fallback=not(np.isfinite(s)and s>0 and np.all(np.isfinite(y64)))
if fallback:proposed=baseline.copy();score=d.astype('d')
else:
 score=d.astype('d')-y64*y64/s;score[anchor]=-np.inf
 proposed=np.sort(np.concatenate(([anchor],np.argsort(-score,kind='stable')[:rank-1])))
assert len(set(proposed))==rank and anchor in proposed
# F64 reference Gram and QR isolate selector geometry from small F32 QR errors.
A=J.astype('d')*lam.astype('d')[None,:];G=A@A.T
g2=float(np.sum(G*G));trace=float(np.trace(G));results={}
for name,ix in [('baseline',baseline),('commonmode',proposed)]:
 Q,R=np.linalg.qr(G[:,ix],mode='reduced');GQ=G@Q;core_exact=Q.T@GQ
 core=np.linalg.solve(Q[ix],R.T);core=(core+core.T)*.5
 projection=Q@GQ.T
 err=G-projection
 err2=G-Q@core_exact@Q.T
 nys=G-Q@core@Q.T
 weighted_residual=max(0.,trace-float(np.trace(core_exact)))
 results[name]=dict(weighted_factor_residual_energy=weighted_residual,weighted_factor_residual_fraction=weighted_residual/trace,gram_left_residual_fro2=float(np.sum(err*err)),gram_left_residual_fraction=float(np.sum(err*err))/g2,gram_two_sided_residual_fro2=float(np.sum(err2*err2)),nystrom_gram_residual_fro2=float(np.sum(nys*nys)),core_trace_deficit=float(np.trace(core_exact)-np.trace(core)),selected_Q_condition=float(np.linalg.cond(Q[ix])),captured_fraction=1-weighted_residual/trace,rows=ix.tolist())
assert free()>=50
b=results['baseline'];p=results['commonmode']
ratios={key:p[key]/b[key]for key in ('weighted_factor_residual_energy','gram_left_residual_fro2','gram_two_sided_residual_fro2','nystrom_gram_residual_fro2','core_trace_deficit')}
r=dict(status='SAVED_GEOMETRY_ONLY',snapshot_path=str(P),snapshot_sha256=sha(P),source_sha256=sha(__file__),scope='Historical S39 network651 firstjoin; same rank384 and same Gram/QR/core formulas. No current-model or final-error inference.',selector='Original maxdiag anchor + stable rank-minus-one largest d-y^2/s; y=M1, s=1Ty; no grids.',selector_dtypes='F32 sumJ/matvec/paid d, F64s/score; F64 reference Gram/QR for geometry metrics',rank=rank,shape=list(J.shape),anchor=anchor,rows_changed=int(rank-np.intersect1d(baseline,proposed).size),s=s,fallback=fallback,commonmode_trace_fraction=float(np.dot(y64,y64)/s/trace)if not fallback else None,baseline=results['baseline'],proposed=results['commonmode'],ratios=ratios,elapsed_seconds=time.monotonic()-START,entry_free_pct=entry,exit_free_pct=free(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,no_truths=True,no_models=True,no_captures=True)
out=D/'SAVED_SELECTOR_GEOMETRY_v1.json'
with out.open('x')as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in r.items()if k not in('baseline','proposed')}),flush=True)

