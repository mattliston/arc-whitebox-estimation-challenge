"""Fixed protected common256 plus difference64 in a nested128 residual span."""
from pathlib import Path
import hashlib,json,os,time,resource
import numpy as np
assert os.getpriority(os.PRIO_PROCESS,0)>=10
assert all(os.environ[k]=='1'for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'))
D=Path(__file__).resolve().parent;O=D.parents[2];P=O/'engineer/a35_join/state/s39_rank0651.npz';start=time.monotonic()
with np.load(P)as z:
 J=z['J_10'].astype(float);lam=z['lam_10'].astype(float);Qp=z['Qp_10'].astype(float);core=z['core_10'].astype(float);old=z['old_09'][z['prev_10']].astype(float);lo=z['lamold_09'].astype(float)
full=np.concatenate([old,J],axis=1);coeff=np.r_[lo,lam];sign=np.sign(coeff);ix=np.r_[0,np.flatnonzero(sign[1:]!=sign[:-1])+1,len(coeff)];blocks=[]
for j in range(0,len(ix)-1,2):
 a,b,c=map(int,ix[j:j+3]);assert b-a==c-b and np.array_equal(coeff[a:b],-coeff[b:c]);blocks.append((a,b,c))
def split(F):return np.concatenate([.5*(F[:,a:b]+F[:,b:c])for a,b,c in blocks],axis=1),np.concatenate([F[:,a:b]-F[:,b:c]for a,b,c in blocks],axis=1)
u,v=split(full);w=np.concatenate([coeff[a:b]**2 for a,b,c in blocks]);diag=(J*J)@(lam*lam)+np.sum((Qp@core)*Qp,axis=1)
order=np.argsort(-diag,kind='stable');base=np.sort(order[:320]);extra=np.sort(order[320:384]);selected=np.r_[base,extra]
Y=J@((lam*lam)[:,None]*J[selected].T)+Qp@(core@Qp[selected].T)
Q=np.linalg.qr(Y,mode='reduced')[0];Qbase=np.linalg.qr(Y[:,:320],mode='reduced')[0]
assert np.max(np.abs(Q[:,:320]@Q[:,:320].T-Qbase@Qbase.T))<1e-10
# Protect the established common subspace. The candidate changes only the64
# directions beyond it, selected once from the128 available residual modes.
fu,fv=split(Q.T@full);R=Q[:,256:];B=fv[256:]
G=(B*w)@B.T;values,V=np.linalg.eigh((G+G.T)/2);T=V[:,-64:]
uh=Q[:,:256]@fu[:256];vbase=Q[:,:320]@fv[:320];vhat=Q[:,:256]@fv[:256]+(R@T)@(T.T@B)
vfull384=Q@fv
eu=float(np.sum((u-uh)**2*w));denv=float(np.sum(v*v*w))
evbase=float(np.sum((v-vbase)**2*w));evhat=float(np.sum((v-vhat)**2*w));ev384=float(np.sum((v-vfull384)**2*w))
saved=json.loads((O/'lead/runs/astra_full384_exact_rescue_audit_v1/SAVED_COMMON192_PROXY_v2.json').read_text())
baseline=saved['rows'][0]
assert abs(eu-baseline['common_weighted_residual'])<1e-8
assert abs(evbase-baseline['difference_weighted_residual'])<1e-8
# Compare one same-input cubic read while keeping its common leg fixed.
weights=np.concatenate([coeff[a:b]for a,b,c in blocks])
def plane(vv):return ((uh*uh+.25*vv*vv)*weights)@vv.T+((2*uh*vv)*weights)@uh.T
target=plane(v);oldplane=plane(vbase);newplane=plane(vhat);fullplane=plane(vfull384)
def metric(p):
 residual=p-target;return dict(relative_error=float(np.sum(residual**2)/np.sum(target**2)),error_energy=float(np.sum(residual**2)))
out=dict(status='FIXED_PROTECTED_DIFFERENCE64_SAVED_FACTOR_PROXY',model_calls=0,captures=0,truth_reads=0,
 shape=dict(active_rows=int(J.shape[0]),joining_columns=int(J.shape[1]),old_columns=int(old.shape[1]),pairs=int(v.shape[1]),old_rank=int(Qp.shape[1])),
 rule='Keep original common256 subspace; select64 weighted-difference directions within the nested128 residual span; retained totalrank320.',
 common_weighted_error_unchanged=eu,difference_error_baseline320=evbase,difference_error_candidate320=evhat,difference_error_nested384=ev384,
 candidate_difference_error_reduction_pct=100*(1-evhat/evbase),fraction_nested384_improvement_recovered=(evbase-evhat)/(evbase-ev384),
 same_common_cubic_read=dict(baseline320=metric(oldplane),candidate320=metric(newplane),nested384=metric(fullplane)),
 common_subspace_projector_max_difference=float(np.max(np.abs(Q[:,:256]@Q[:,:256].T-Qbase[:,:256]@Qbase[:,:256].T))),
 archive_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),seconds=time.monotonic()-start,rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
 limitations=['Historical S39 network651 L10 saved factors only; not final mean-MSE or nativeFNP cost.','Cubic target fixes common256 and restores all difference factors; its target is not the true cumulant plane.','Selection derives from factors/coefficients only; no outcome fitting, rank or coefficient sweep.'])
assert out['rss_bytes']<2_000_000_000
(D/'SAVED_RESULT_v1.json').open('x').write(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
