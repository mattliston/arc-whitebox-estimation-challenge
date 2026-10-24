"""One fixed tail-driven power objective; saved TRAIN16 only, no truth/model."""
from pathlib import Path
import json,hashlib,time,importlib.util
import numpy as np
D=Path(__file__).resolve().parent;R=D.parent;C=R/'astra_native_terminal_linear_refit_v1/run_capture_s0_v1/train16_read_observer_v1'
spec=importlib.util.spec_from_file_location('oracle',R/'astra_read_column_oracle_v1/inspect_fixed_band_oracle_HELD_v1.py');o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)
rules=dict(seed=16,scope='All existing L02..L14 changed column reads; same rank2, same exactprefix/c2mask/oldterm.',candidate='Q0=stableQR(vc,mu); U=A@(Btail.T@(Btail@(A.T@Q0))); Qtail=stableQR(U), fallbackQ0 onrankloss. Replace original prefix-power objective; no weighting/strength/grid.',evaluation='Full young tail squared error, diagonal-removed young tail error, and symmetrized leading covariance error; no finalMSE or truth claims.',source_only=True)
rule_path=D/'OFFLINE_RULE_FROZEN_v1.json'
if rule_path.exists():assert json.loads(rule_path.read_text())==rules
else:rule_path.open('x').write(json.dumps(rules,indent=2)+'\n')
start=time.monotonic();rowsout=[];pooled={};pins={}
for p in sorted(C.glob('L*.npz')):
 pins[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 with np.load(p,allow_pickle=False)as z:
  A=z['left_young'].astype(float);B=z['right_young'][int(z['c15']):].astype(float)*z['lambda_young'].astype(float)
  Q=z['Q'].astype(float);rows=z['rows'].astype(int);cols=z['cols'][int(z['c15']):].astype(int)
  mu=z['mu'].astype(float);vc=z['vc'].astype(float);gate=z['gate'].astype(float)
  c15=int(z['c15']);c2=int(z['c2']);oldrank=int(z['old_inner'].shape[1])if 'old_inner'in z.files else 0
 Q0=o.basis(vc[rows],mu[rows]);U=A@(B.T@(B@(A.T@Q0)));Qt=o.basis(U[:,0],U[:,1]if U.shape[1]>1 else np.zeros(len(rows)))
 fallback=Qt.shape[1]<Q0.shape[1]
 if fallback:Qt=Q0
 Y=A@B.T;alpha=mu[rows]/np.sqrt(vc[rows]);d=np.exp(-.5*alpha**2)/(np.sqrt(2*np.pi)*np.sqrt(vc[rows]))
 cases={}
 for name,basis in [('incumbent_QR',Q),('fixed_tail_power',Qt)]:
  E=Y-basis@(basis.T@Y);raw=float(np.sum(E*E));o.zero_diag(E,rows,cols)
  cases[name]=dict(full_young_tail_error=raw,**o.energies(E,d,gate[cols],rows,cols,len(mu)))
  for key,val in cases[name].items():pooled.setdefault(name,{}).setdefault(key,0.);pooled[name][key]+=val
 ratio={k:cases['fixed_tail_power'][k]/max(v,1e-300)for k,v in cases['incumbent_QR'].items()}
 row=dict(layer=int(p.stem[1:]),m=A.shape[0],k=A.shape[1],tail=B.shape[0],c15=c15,c2=c2,oldrank=oldrank,rank=int(Qt.shape[1]),fallback=fallback,cases=cases,ratio_to_QR=ratio);rowsout.append(row)
 print(json.dumps(dict(layer=row['layer'],ratios=ratio)),flush=True)
result=dict(status='FIXED_SAVED_FACTOR_ONLY',rule=rules,per_layer=rowsout,pooled=pooled,pooled_ratio_to_QR={k:pooled['fixed_tail_power'][k]/max(v,1e-300)for k,v in pooled['incumbent_QR'].items()},evidence_sha256=pins,elapsed_s=time.monotonic()-start,models=0)
p=D/'OFFLINE_TAIL_POWER_RESULT_v1.json';p.open('x').write(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(pooled_ratio_to_QR=result['pooled_ratio_to_QR'],elapsed_s=result['elapsed_s'],sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
