from pathlib import Path
import hashlib,json,os,time
import numpy as np
D=Path(__file__).resolve().parent;O=D.parents[2]
assert all(os.environ.get(k)=="1" for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","VECLIB_NUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS"))
out=D/"TRAIN16_EXACT_REMAINDER_v1.json";assert not out.exists()
p=json.loads((O/"lead/runs/astra_current_variance_capture_v1/CAPTURE_PROTOCOL_HELD_v1.json").read_text())
wpath=Path(p["weights_path"]);W=np.load(wpath,allow_pickle=False)
assert hashlib.sha256(W.tobytes()).hexdigest()==p["weights_raw_sha256"]
def integral(a,rho,order):
 nodes,weights=np.polynomial.legendre.leggauss(order);nodes=(nodes+1)/2;weights=weights/2
 aa=a[:,None]*a[:,None]+a[None,:]*a[None,:];ab=np.outer(a,a);ans=np.zeros_like(rho)
 for u,weight in zip(nodes,weights):
  t=rho*u;den=1-t*t
  ans+=weight*(1-u)*np.exp(-(aa-2*t*ab)/(2*den))/np.sqrt(den)/(2*np.pi)
 return ans
def rms(x):return float(np.sqrt(np.mean(x*x)))
def corr(x,y):return float(np.sum(x*y)/np.sqrt(np.sum(x*x)*np.sum(y*y)))
# Analytic centered-ReLU identity independently checks Price integral.
rr=np.array([[-.9,-.5,-.1,0,.1,.5,.9]])
aa=np.zeros(1)
# use scalar quadrature here to avoid nonsquare dimensions in the full helper
nodes,weights=np.polynomial.legendre.leggauss(64);u=(nodes+1)/2
integ=np.sum((weights/2)[:,None]*(1-u)[:,None]/np.sqrt(1-(u[:,None]*rr)**2),axis=0)/(2*np.pi)
numeric=rr[0]/4+rr[0]**2*integ
analytic=rr[0]/4+(np.sqrt(1-rr[0]**2)+rr[0]*np.arcsin(rr[0])-1)/(2*np.pi)
assert np.max(abs(numeric-analytic))<1e-14
rows=[];t0=time.monotonic()
for l in (13,14):
 f=O/f"physicist/t128_cdump/t128_C_16_L{l}.npz"
 with np.load(f,allow_pickle=False) as z:C=z["C"].astype(float);m=z["mu"].astype(float)
 w=W[l].astype(float);Cp=w.T@((C+C.T)*.5)@w;mu=m@w;v=np.diag(Cp);assert np.all(v>0)
 s=np.sqrt(v);a=mu/s;rho=Cp/np.outer(s,s);np.fill_diagonal(rho,0)
 assert np.max(abs(rho))<1
 phi=np.exp(-a*a/2)/np.sqrt(2*np.pi);pij=np.outer(phi,phi);aij=np.outer(a,a)
 R3=pij*(.5+rho*aij/6)
 d16=np.outer(s,s)*rho*rho*(integral(a,rho,16)-R3)
 d32=np.outer(s,s)*rho*rho*(integral(a,rho,32)-R3)
 d4=np.outer(s*phi*(a*a-1),s*phi*(a*a-1))*rho**4/24
 np.fill_diagonal(d16,0);np.fill_diagonal(d32,0);np.fill_diagonal(d4,0)
 fp=O/f"physicist/t128_cdump/t128_C_16_L{l+1}.npz"
 with np.load(fp,allow_pickle=False) as z:Cnext=z["C"].astype(float);mnext=z["mu"].astype(float)
 wn=W[l+1].astype(float);vn=np.einsum("ij,ij->j",wn,Cnext@wn);mun=mnext@wn
 sn=np.sqrt(np.maximum(vn,1e-20));an=mun/sn;sen=np.exp(-an*an/2)/(2*np.sqrt(2*np.pi)*sn)
 dv=np.einsum("ij,ij->j",wn,d32@wn);dv4=np.einsum("ij,ij->j",wn,d4@wn)
 off=~np.eye(len(m),dtype=bool)
 row=dict(layer=l,rho_abs_quantiles=dict(zip(["q0","q50","q90","q99","q99_9","max"],map(float,np.quantile(abs(rho[off]),[0,.5,.9,.99,.999,1])))),rho_abs_gt_half=int((abs(rho[off])>.5).sum()),alpha_abs_quantiles=list(map(float,np.quantile(abs(a),[0,.5,.9,.99,1]))),price_integral16_vs32_max=float(np.max(abs(d16-d32))),price_integral16_vs32_rms=rms(d16-d32),exact_remainder_frobenius=float(np.linalg.norm(d32)),r4_frobenius=float(np.linalg.norm(d4)),beyond_r4_frobenius=float(np.linalg.norm(d32-d4)),exact_vs_r4_cosine=corr(d32,d4),exact_next_variance_rms=rms(dv),r4_next_variance_rms=rms(dv4),beyond_r4_next_variance_rms=rms(dv-dv4),exact_next_mean_displacement_rms=rms(sen*dv),r4_next_mean_displacement_rms=rms(sen*dv4),beyond_r4_next_mean_displacement_rms=rms(sen*(dv-dv4)),exact_vs_r4_next_mean_cosine=corr(sen*dv,sen*dv4),relative_next_variance_rms=rms(dv/vn),source_snapshots=[dict(path=str(x),sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in (f,fp)])
 rows.append(row);print(json.dumps(row),flush=True)
x=dict(status="SAVED_TRAIN_ONLY_GAUSSIAN_EXACT_REMAINDER_DIAGNOSTIC",rows=rows,seconds=time.monotonic()-t0,centered_analytic_max_error=float(np.max(abs(numeric-analytic))),formula="Cov/s_i/s_j = rho*Phi(ai)*Phi(aj) + rho^2 integral_0^1 (1-u)*phi2(ai,aj;rho*u)du; subtract R3 exactly, restore diagonal zero.",source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),models=0,new_truth=0,limits=["Historical S47e TRAIN16 saved covariance, not current1c913 state.","Input covariance formed as exact W.T C W without current deadrow cut; Gaussian kernel formula is current active R3.","32-node quadrature is an offline reference, not a deployment-cost claim.","Local Gaussian sensitivity is not final accuracy."])
out.write_text(json.dumps(x,indent=2)+"\n");print(hashlib.sha256(out.read_bytes()).hexdigest(),flush=True)

