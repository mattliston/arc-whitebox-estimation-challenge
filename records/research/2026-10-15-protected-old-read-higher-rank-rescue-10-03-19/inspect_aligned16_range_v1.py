"""One fixed root-authorized aligned16 range: vc/mu +14 H-Gaussian columns."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
D=Path(__file__).resolve().parent;O=D.parents[2]
sys.path.insert(0,str(O/'lead/runs/astra_read_column_oracle_v1'))
from inspect_fixed_band_oracle_HELD_v1 import basis,zero_diag,energies
C=O/'lead/runs/astra_native_terminal_linear_refit_v1/run_capture_s0_v1/train16_read_observer_v1'
out=D/'TRAIN16_ALIGNED16_RANGE_v1.json';assert not out.exists()
rng=np.random.default_rng(20261015);rows=[];pooled={};pins={}
for path in sorted(C.glob('L*.npz')):
 with np.load(path,allow_pickle=False)as z:
  if 'old_inner'not in z:continue
  H=z['old_inner'].astype(float);B=z['old_right'].astype(float)
  ids=z['rows'].astype(int);cols=z['cols'].astype(int)[:len(B)]
  v,mu,g=[z[k].astype(float)for k in('vc','mu','gate')]
 pins[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 m,r=H.shape;packed=((3*m+255)//256)*64;oldcenter=packed-2;center=packed-16
 d=np.exp(-.5*mu[ids]**2/v[ids])/np.sqrt(2*np.pi*v[ids])
 q0=basis(v[ids],mu[ids]);E=(H-q0@(q0.T@H))@B.T;E[:oldcenter]=0
 zero_diag(E,ids,cols);cases={'failed_original_protected_rank2':energies(E,d,g[cols],ids,cols,len(mu))}
 Ho=H[center:];qfixed=basis(v[ids[center:]],mu[ids[center:]])
 assert qfixed.shape[1]==2
 omega=rng.standard_normal((r,30))[:,:14];Z=Ho@omega
 for _ in range(2):Z-=qfixed@(qfixed.T@Z)
 qr,rr=np.linalg.qr(Z,mode='reduced');assert qr.shape[1]==14
 assert np.min(np.abs(np.diag(rr)))>1e-12*np.max(np.abs(np.diag(rr)))
 q=np.column_stack((qfixed,qr))
 assert np.max(np.abs(q.T@q-np.eye(16)))<1e-10
 uh,sh,_=np.linalg.svd(Ho,full_matrices=False)
 for name,qq in [('fixed_2physics_14gaussian',q),('outer_inner_svd16',uh[:,:16])]:
  e=np.zeros((m,len(B)));e[center:]=(Ho-qq@(qq.T@Ho))@B.T
  zero_diag(e,ids,cols);cases[name]=energies(e,d,g[cols],ids,cols,len(mu))
 Y=Ho@B.T;W=.5*d[center:,None]*Y*g[cols][None,:]
 uw,sw,_=np.linalg.svd(W,full_matrices=False);qw=uw[:,:16]
 ew=np.zeros((m,len(B)));ew[center:]=W-qw@(qw.T@W);zero_diag(ew,ids,cols)
 full=np.zeros((len(mu),len(mu)));full[np.ix_(ids,cols)]=ew;sym=full+full.T
 cases['outer_weighted_output_svd16']=dict(weighted_directed=float(np.sum(ew*ew)),leading_symmetric_covariance=float(np.sum(sym*sym)))
 for name,vals in cases.items():
  for key,val in vals.items():pooled.setdefault(name,{}).setdefault(key,0.);pooled[name][key]+=val
 ratio={name:vals['leading_symmetric_covariance']/cases['failed_original_protected_rank2']['leading_symmetric_covariance']for name,vals in cases.items()if name!='failed_original_protected_rank2'}
 rows.append(dict(layer=int(path.stem[1:]),m=m,original_center=oldcenter,aligned_center=center,packed_height=packed,outer_rows=m-center,omega_shape=list(omega.shape),cases=cases,ratio_to_failed_original=ratio))
 print(path.stem,json.dumps(ratio),flush=True)
ratios={name:{key:vals[key]/pooled['failed_original_protected_rank2'][key]for key in('weighted_directed','leading_symmetric_covariance')}for name,vals in pooled.items()if name!='failed_original_protected_rank2'}
doc=dict(status='ONE_FIXED_ALIGNED16_SAVED_RANGE_CHECK',rank=16,seed=20261015,gaussian_rule="first14 of the already-fixed per-layer32-case30-column Gaussian table; no resampling",rule='original packed height; center=packed-16; QR of outer vc/mu plus14 independent Gaussian H directions orthogonalized twice; no power iteration',per_layer=rows,pooled=pooled,pooled_ratio_to_failed_original=ratios,evidence_sha256=pins,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limits=['Only saved TRAIN16 H and retained-prefix B; old_left unavailable. H Omega is attainable with a metric-aligned old-left sketch, whose factor formation must be charged.','No truths/models/captures/rank or random-seed search.','The two SVD16 results are nondeployable diagnostics at the same aligned exact-row budget.','Exact global diagonal residual excluded; leading symmetric covariance proxy is not final MSE.','Added14 approximated rows relative to the failed baseline are explicitly included.'])
out.write_text(json.dumps(doc,indent=2)+'\n');print('POOLED',json.dumps(ratios));print('SHA256',hashlib.sha256(out.read_bytes()).hexdigest())
