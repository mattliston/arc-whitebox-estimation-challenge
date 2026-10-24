from pathlib import Path
import numpy as np,json,hashlib,time
D=Path(__file__).resolve().parent;O=D.parents[2];C=O/'lead/runs/astra_native_terminal_linear_refit_v1/run_capture_s0_v1/train16_read_observer_v1'
out=D/'PAIR_ENERGY_ESTIMATOR_v1.json';assert not out.exists();rr=[]
for l in range(7,15):
 with np.load(C/f'L{l:02d}.npz')as z:
  A=z['left_young'].astype(float);lam=z['lambda_young'].astype(float);B=z['right_young'].astype(float);Q=z['Q'].astype(float);rows=z['rows'];cols=z['cols'];cut=int(z['c15']);v=z['vc'].astype(float)
 A=(A-Q@(Q.T@A))/v[rows,None];B=B[cut:]/np.sqrt(v[cols[cut:],None]);cols=cols[cut:]
 E=A@(B*lam).T;_,ri,ci=np.intersect1d(rows,cols,return_indices=True);E[ri,ci]=0
 re=np.sum(E*E,axis=1);ce=np.sum(E*E,axis=0);er=np.zeros(len(rows));ec=np.zeros(len(cols))
 bounds=np.r_[0,np.where(np.sign(lam[1:])!=np.sign(lam[:-1]))[0]+1,len(lam)]
 assert len(bounds)%2==1
 for i in range(0,len(bounds)-1,2):
  p=slice(bounds[i],bounds[i+1]);n=slice(bounds[i+1],bounds[i+2]);w=lam[p]**2;assert np.array_equal(lam[p],-lam[n])
  ap=A[:,p];an=A[:,n];bp=B[:,p];bn=B[:,n]
  er+=np.sum(w*(ap**2*np.sum(bp**2,axis=0)+an**2*np.sum(bn**2,axis=0)-2*ap*an*np.sum(bp*bn,axis=0)),axis=1)
  ec+=np.sum(w*(bp**2*np.sum(ap**2,axis=0)+bn**2*np.sum(an**2,axis=0)-2*bp*bn*np.sum(ap*an,axis=0)),axis=1)
  # Remove each pair's exact diagonal contribution, matching residual diagonal-zero rule.
  diag=(ap[ri]*bp[ci]-an[ri]*bn[ci])**2@w
  er[ri]-=diag;ec[ci]-=diag
 rr.append(dict(layer=l,exact_energy=float(re.sum()),pair_diagonal_energy=float(er.sum()),ratio=float(er.sum()/re.sum()),row_relative_rmse=float(np.linalg.norm(er-re)/np.linalg.norm(re)),col_relative_rmse=float(np.linalg.norm(ec-ce)/np.linalg.norm(ce)),col_corr=float(np.corrcoef(ec,ce)[0,1])))
 print(rr[-1],flush=True)
r=dict(status='OFFLINE_PAIR_DIAGONAL_ENERGY_CHECK',rows=rr,source='Saved QR TRAIN16 only; no truth targets, native calls, fit, or sweeps',limits=['Pair-aware terms exact; all cross-pair contributions omitted.','This estimates residual E squared only, not unavailable 2<Kcurrent,E> correction to production norm.','Same physics proxy as prior energy audit, not exact HF current outputs.'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
out.write_text(json.dumps(r,indent=2)+'\n');print(hashlib.sha256(out.read_bytes()).hexdigest())
