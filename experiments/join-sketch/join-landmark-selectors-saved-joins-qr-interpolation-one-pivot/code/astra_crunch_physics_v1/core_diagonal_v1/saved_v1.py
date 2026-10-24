"""Fixed exact projected-operator diagonal replacement on six historical joins."""
from pathlib import Path
import hashlib, json, os, resource, time
import numpy as np
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    assert os.environ[key] == '1'
D=Path(__file__).resolve().parent
C=D.parent.parent/'opus_sketchgb_v1/cap'
files=sorted(C.glob('join_*.npz')); assert len(files)==6
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prereg=dict(source_sha256=sha(Path(__file__)),rank=384,
 formula='diag=sum(Fj^2*w,axis1)+sum((overlap@oldcore)*overlap,axis1)',
 scope='Fixed six archived joins, unchanged static topdiag rows, no model or tuning; only middle join is novel and has a consumed stored core.',
 inputs={str(p):sha(p) for p in files})
(D/'PREREG_v1.json').open('x').write(json.dumps(prereg,indent=2)+'\n')
rows=[]; t0=time.monotonic()
for p in files:
    z=np.load(p); J=z['J'].astype(float); w=z['w'].astype(float)
    Qp=z['Qp'].astype(float); oldcore=z['core'].astype(float); d=z['d'].astype(float)
    n,k=J.shape; rank=384; oldrank=Qp.shape[1]
    G=(J*w[None,:])@J.T
    if oldrank: G+=Qp@oldcore@Qp.T
    G=(G+G.T)*.5
    pick=np.sort(np.argsort(-d,kind='stable')[:rank])
    Q,R=np.linalg.qr(G[:,pick],mode='reduced')
    core=np.linalg.solve(Q[pick],R.T); core=(core+core.T)*.5
    exact=Q.T@G@Q; exact=(exact+exact.T)*.5
    Fj=Q.T@J; diagonal=np.sum(Fj*Fj*w[None,:],axis=1)
    if oldrank:
        overlap=Q.T@Qp
        diagonal+=np.sum((overlap@oldcore)*overlap,axis=1)
    assert np.allclose(diagonal,np.diag(exact),rtol=1e-10,atol=1e-10)
    repaired=core.copy(); np.fill_diagonal(repaired,diagonal)
    E=G-Q@core@Q.T; Er=G-Q@repaired@Q.T
    e2=float(np.sum(E*E)); er2=float(np.sum(Er*Er))
    ce2=float(np.sum((core-exact)**2)); cr2=float(np.sum((repaired-exact)**2))
    item=dict(file=p.name,shape=[n,k],oldrank=oldrank,saved_rank=int(z['rank']),test_rank=rank,
       diagonal_formula_maxabs=float(np.max(np.abs(diagonal-np.diag(exact)))),
       operator_error2_recovered=1-er2/e2,core_error2_recovered=1-cr2/ce2,
       base_operator_relative_error=float(np.sqrt(e2/np.sum(G*G))),
       repaired_operator_relative_error=float(np.sqrt(er2/np.sum(G*G))),
       base_trace_relative_error=float(np.trace(E)/np.trace(G)),
       repaired_trace_relative_error=float(np.trace(Er)/np.trace(G)),
       base_core_trace=float(np.trace(core)),exact_core_trace=float(np.trace(exact)),
       diagonal_change_relative=float(np.linalg.norm(diagonal-np.diag(core))/np.linalg.norm(diagonal)),
       repaired_min_eigenvalue=float(np.linalg.eigvalsh(repaired)[0]),
       raw_flops_diagonal_new=3*rank*k,raw_flops_old=2*rank*oldrank*oldrank+2*rank*oldrank,
       exact_current_l10_factor_reduction_if_materialized='3*rank*total_old_columns; no oldcore multiply needed')
    rows.append(item); print(json.dumps(item),flush=True)
    del z,J,w,Qp,oldcore,d,G,Q,R,core,exact,Fj,diagonal,repaired,E,Er
out=dict(prereg=prereg,rows=rows,seconds=time.monotonic()-t0,
 peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
 limitations=['Historical captures; middle saved rank320 but test uses current-style384.',
 'Local Gram error is not final prediction MSE.', 'L7 exact diagonal previously null on fixed8; L13 core is unused.'])
(D/'RESULT_v1.json').open('x').write(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(done=True,seconds=out['seconds'],rss=out['peak_rss_bytes'])),flush=True)
