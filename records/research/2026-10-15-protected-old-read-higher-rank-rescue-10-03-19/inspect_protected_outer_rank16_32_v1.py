"""Fixed protected-center, outer-only rank16/32 TRAIN16 oracle; no models/truth."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
D=Path(__file__).resolve().parent; O=D.parents[2]
sys.path.insert(0,str(O/'lead/runs/astra_read_column_oracle_v1'))
from inspect_fixed_band_oracle_HELD_v1 import basis,zero_diag,energies
C=O/'lead/runs/astra_native_terminal_linear_refit_v1/run_capture_s0_v1/train16_read_observer_v1'
out=D/'TRAIN16_PROTECTED_OUTER_RANK16_32_v1.json'; assert not out.exists()
rows=[]; pooled={}; pins={}
for path in sorted(C.glob('L*.npz')):
 with np.load(path,allow_pickle=False)as z:
  if 'old_inner'not in z: continue
  H=z['old_inner'].astype(float); B=z['old_right'].astype(float)
  ids=z['rows'].astype(int); cols=z['cols'].astype(int)[:len(B)]
  v,mu,g=[z[k].astype(float)for k in('vc','mu','gate')]
 pins[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 m,r=H.shape; center=((3*m+255)//256)*64-2
 d=np.exp(-.5*mu[ids]**2/v[ids])/np.sqrt(2*np.pi*v[ids])
 Q=basis(v[ids],mu[ids]); E=(H-Q@(Q.T@H))@B.T; E[:center]=0
 zero_diag(E,ids,cols)
 cases={'failed_protected_rank2':energies(E,d,g[cols],ids,cols,len(mu))}
 Ho=H[center:]; Y=Ho@B.T; W=.5*d[center:,None]*Y*g[cols][None,:]
 uh,sh,_=np.linalg.svd(Ho,full_matrices=False)
 uw,sw,_=np.linalg.svd(W,full_matrices=False)
 for rank in(16,32):
  q=uh[:,:rank]
  e=np.zeros((m,len(B))); e[center:]=(Ho-q@(q.T@Ho))@B.T
  zero_diag(e,ids,cols)
  cases[f'outer_inner_svd_{rank}']=energies(e,d,g[cols],ids,cols,len(mu))
  q=uw[:,:rank]; ew=np.zeros((m,len(B)))
  ew[center:]=W-q@(q.T@W); zero_diag(ew,ids,cols)
  full=np.zeros((len(mu),len(mu))); full[np.ix_(ids,cols)]=ew
  sym=full+full.T
  cases[f'outer_weighted_output_svd_{rank}']=dict(weighted_directed=float(np.sum(ew*ew)),leading_symmetric_covariance=float(np.sum(sym*sym)))
 for name,vals in cases.items():
  for key,val in vals.items():pooled.setdefault(name,{}).setdefault(key,0.); pooled[name][key]+=val
 row=dict(layer=int(path.stem[1:]),m=m,center_exact=center,outer_rows=m-center,inner_rank=r,
  packed_rows={str(rank):((center+rank+63)//64)*64 for rank in(16,32)},cases=cases,
  ratio_to_failed={name:vals['leading_symmetric_covariance']/cases['failed_protected_rank2']['leading_symmetric_covariance']for name,vals in cases.items()if name!='failed_protected_rank2'})
 rows.append(row); print(path.stem,json.dumps(row['ratio_to_failed']),flush=True)
ratios={name:{key:vals[key]/pooled['failed_protected_rank2'][key]for key in('weighted_directed','leading_symmetric_covariance')}for name,vals in pooled.items()if name!='failed_protected_rank2'}
doc=dict(status='FIXED_PROTECTED_OUTER_RANK16_32_ORACLE',ranks=[16,32],central_rule='unchanged ceil(.75*m/64)*64-2 exact rows; only remaining outer rows projected',per_layer=rows,pooled=pooled,pooled_ratio_to_failed_protected_rank2=ratios,evidence_sha256=pins,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limits=['Accepted saved TRAIN16 only; no truths, new captures, candidate/model calls or rank sweep.','Both SVD bases require already-computed inner/output and are nondeployable ceilings for a future sketch.','Weighted output SVD optimizes directed weighted output before exact diagonal exclusion, not the symmetrized objective; actual symmetric error is reported.','Only captured old right prefix and leading covariance sensitivity are covered; native future-feedback/MSE improvement is unknown.'])
out.write_text(json.dumps(doc,indent=2)+'\n');print('POOLED',json.dumps(ratios));print('SHA256',hashlib.sha256(out.read_bytes()).hexdigest())
