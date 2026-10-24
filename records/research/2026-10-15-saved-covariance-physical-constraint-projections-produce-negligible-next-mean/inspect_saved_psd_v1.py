from pathlib import Path
import hashlib,json,os,time
import numpy as np
D=Path(__file__).resolve().parent;O=D.parents[2]
assert all(os.environ.get(k)=="1" for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","VECLIB_NUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS"))
out=D/"TRAIN16_PSD_BOUND_v1.json";assert not out.exists()
p=json.loads((O/"lead/runs/astra_current_variance_capture_v1/CAPTURE_PROTOCOL_HELD_v1.json").read_text())
wpath=Path(p["weights_path"]);W=np.load(wpath,allow_pickle=False)
assert hashlib.sha256(W.tobytes()).hexdigest()==p["weights_raw_sha256"]
rows=[];t0=time.monotonic()
for l in (13,14,15):
 f=O/f"physicist/t128_cdump/t128_C_16_L{l}.npz"
 with np.load(f,allow_pickle=False) as z:C=z["C"].astype(float);m=z["mu"].astype(float)
 C=(C+C.T)*.5;n=len(m);lam,Q=np.linalg.eigh(C);ix=lam<0
 w=W[l].astype(float);v=np.einsum("ij,ij->j",w,C@w);mu=m@w
 U=Q[:,ix];neg=-lam[ix]
 if len(neg):
  UW=U.T@w;dv=neg@(UW*UW);dd=(U*U)@neg
 else:dv=np.zeros(n);dd=np.zeros(n)
 # Additive PSD cone projection is an oracle, not deployable candidate.
 # Also measure its zero-diagonal component, which isolates covariance transport.
 dv_off=dv-(w*w).T@dd
 s=np.sqrt(np.maximum(v,1e-20));a=mu/s;sen=np.exp(-.5*a*a)/(2*np.sqrt(2*np.pi)*s)
 row=dict(next_layer=l,minimum_eigenvalue=float(lam[0]),negative_modes=int(ix.sum()),negative_trace=float(neg.sum()),trace=float(np.trace(C)),negative_frobenius=float(np.linalg.norm(neg)),total_frobenius=float(np.linalg.norm(lam)),mean_rayleigh=float(m@C@m/(m@m)),negative_diagonals=int((np.diag(C)<0).sum()),projected_variance_mean=float(dv.mean()),projected_variance_rms=float(np.sqrt(np.mean(dv**2))),projected_offdiag_variance_rms=float(np.sqrt(np.mean(dv_off**2))),projected_mean_displacement_rms=float(np.sqrt(np.mean((sen*dv)**2))),projected_offdiag_mean_displacement_rms=float(np.sqrt(np.mean((sen*dv_off)**2))),input=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest())
 rows.append(row);print(json.dumps(row),flush=True)
x=dict(status="SAVED_TRAIN_ONLY_PSD_ORACLE",rows=rows,seconds=time.monotonic()-t0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),weights_path=str(wpath),weights_sha256=hashlib.sha256(wpath.read_bytes()).hexdigest(),models=0,new_truth=0,limits=["Historical S47e plain Torch covariance, not current submitted source.","Projection uses exact eigendecomposition only as an offline diagnostic; no candidate or deployable tariff.","Mean displacement is local Gaussian derivative sensitivity, not final MSE or actual accuracy.","The zero-diagonal component does not itself guarantee PSD."])
out.write_text(json.dumps(x,indent=2)+"\n");print(hashlib.sha256(out.read_bytes()).hexdigest(),flush=True)
