from pathlib import Path
import hashlib,json,os,time,math
import numpy as np
D=Path(__file__).resolve().parent;O=D.parents[2]
assert all(os.environ.get(k)=="1" for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","VECLIB_NUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS"))
out=D/"TRAIN16_RADIAL_SECOND_MOMENT_BOUND_v1.json";assert not out.exists()
p=json.loads((O/"lead/runs/astra_current_variance_capture_v1/CAPTURE_PROTOCOL_HELD_v1.json").read_text())
wpath=Path(p["weights_path"]);W=np.load(wpath,allow_pickle=False)
assert hashlib.sha256(W.tobytes()).hexdigest()==p["weights_raw_sha256"]
rows=[];t0=time.monotonic()
for l in (13,14,15):
 f=O/f"physicist/t128_cdump/t128_C_16_L{l}.npz"
 with np.load(f,allow_pickle=False) as z:C=z["C"].astype(float);m=z["mu"].astype(float)
 C=(C+C.T)*.5;n=len(m)
 log_er=.5*math.log(2)+math.lgamma((n+1)/2)-math.lgamma(n/2)
 beta=math.expm1(math.log(n)-2*log_er)
 lam,Q=np.linalg.eigh(C-beta*np.outer(m,m));ix=lam<0;U=Q[:,ix];neg=-lam[ix]
 w=W[l].astype(float);v=np.einsum("ij,ij->j",w,C@w);mu=m@w
 UW=U.T@w;dv=neg@(UW*UW);dd=(U*U)@neg;dv_off=dv-(w*w).T@dd
 s=np.sqrt(np.maximum(v,1e-20));a=mu/s;sen=np.exp(-.5*a*a)/(2*np.sqrt(2*np.pi)*s)
 R=C+np.outer(m,m);delta=np.maximum(-R,0);np.fill_diagonal(delta,0)
 dvr=np.einsum("ij,ij->j",w,delta@w)
 row=dict(next_layer=l,beta=beta,radial_minimum_eigenvalue=float(lam[0]),radial_negative_modes=int(ix.sum()),radial_negative_trace=float(neg.sum()),radial_negative_frobenius=float(np.linalg.norm(neg)),radial_projected_variance_rms=float(np.sqrt(np.mean(dv**2))),radial_projected_offdiag_variance_rms=float(np.sqrt(np.mean(dv_off**2))),radial_projected_mean_displacement_rms=float(np.sqrt(np.mean((sen*dv)**2))),radial_projected_offdiag_mean_displacement_rms=float(np.sqrt(np.mean((sen*dv_off)**2))),second_moment_min=float(R.min()),second_moment_offdiag_negative_entries=int((delta>0).sum()),second_moment_offdiag_negative_frobenius=float(np.linalg.norm(delta)),second_moment_offdiag_max_correction=float(delta.max()),second_moment_projected_variance_rms=float(np.sqrt(np.mean(dvr**2))),second_moment_projected_mean_displacement_rms=float(np.sqrt(np.mean((sen*dvr)**2))),input=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest())
 rows.append(row);print(json.dumps(row),flush=True)
x=dict(status="SAVED_TRAIN_ONLY_PHYSICAL_CONSTRAINT_ORACLE",rows=rows,seconds=time.monotonic()-t0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),weights_path=str(wpath),weights_sha256=hashlib.sha256(wpath.read_bytes()).hexdigest(),models=0,new_truth=0,limits=["Historical S47e plain Torch covariance, not current submitted source.","Eigenprojection is an offline diagnostic, not a candidate or deployable tariff.","Mean displacement is local Gaussian derivative sensitivity, not final MSE or actual accuracy.","Zero-diagonal projection does not guarantee PSD; elementwise second-moment clipping does not guarantee PSD."])
out.write_text(json.dumps(x,indent=2)+"\n");print(hashlib.sha256(out.read_bytes()).hexdigest(),flush=True)

