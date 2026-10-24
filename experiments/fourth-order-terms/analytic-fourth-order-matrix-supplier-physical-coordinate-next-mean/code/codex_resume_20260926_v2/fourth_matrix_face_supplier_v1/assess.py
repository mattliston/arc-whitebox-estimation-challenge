"""Score the fixed deterministic supplier after its process has retired."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
ASTRA=HERE.parents[1]/'astra_phase2_reexamination_v1'
CAP=HERE.parent/'fourth_matrix_face_capacity_v1'
sys.path.insert(0,str(ASTRA));sys.path.insert(0,str(CAP))
from current_pair_oracle_math_v1 import exact_change,independent_fourth_map
from projection import vector

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while b:=f.read(1024**2):h.update(b)
 return h.hexdigest()

for name,digest in json.loads((HERE/'seal.json').read_text())['files'].items():assert sha(Path(name))==digest,name
out=HERE/'assessment.json';assert not out.exists() and not out.with_suffix('.npz').exists()
cap=json.loads((HERE/'capture.json').read_text());guard=json.loads((HERE/'capture_guard.json').read_text())
assert cap['passed'] and guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
assert sha(HERE/'capture.npz')==cap['output_sha256']
f=dict(np.load(ASTRA/'fourth_trace_oracle_inputs_r35_v1.npz',allow_pickle=False))
d=dict(np.load(HERE/'capture.npz',allow_pickle=False))
old=dict(np.load(CAP/'values.npz',allow_pickle=False))
assert sha(CAP/'values.npz')==json.loads((CAP/'assessment.json').read_text())['values_sha256']
C=(f['pre13_Cpre']+f['pre13_Cpre'].T)/2;D=(d['Dpre13']+d['Dpre13'].T)/2
v=np.diag(C);a=np.diag(D);s=f['pre13_sigma']
k4=v*a-f['pre13_k4']
q31=(v[:,None]*D+a[:,None]*C)/2-f['pre13_k31']
q22=(np.outer(v,a)+np.outer(a,v)+4*C*D)/6-f['pre13_k22']
ds=(k4/s**4,q31/(s[:,None]**3*s[None,:]),q22/np.outer(s*s,s*s))
dq4,dq31,dq22=(x.copy() for x in ds);np.fill_diagonal(dq31,0);np.fill_diagonal(dq22,0)
common=(f['pre13_Cpre']/np.outer(s,s),s,f['pre13_alpha'],f['pre13_pdf'],f['pre13_gate'],f['pre13_gaussian_mean']/s,f['pre13_m']/s)
dm,dc=exact_change(common,dQ22=dq22,dQ31=dq31,dq4=dq4)
fn=independent_fourth_map();base31=f['pre13_k31']/(s[:,None]**3*s[None,:]);np.fill_diagonal(base31,0)
bm,bc,*_=fn(f['pre13_mu'],f['pre13_Cpre'],f['pre13_K21'],f['pre13_k4'],f['pre13_k22'],'covariance',q31_override=base31)
cm,cc,*_=fn(f['pre13_mu'],f['pre13_Cpre'],f['pre13_K21'],f['pre13_k4']+k4,f['pre13_k22']+q22,'covariance',q31_override=base31+dq31)
discrepancy=max(float(np.max(np.abs(cm-bm-dm))),float(np.max(np.abs((cc-bc+cc.T-bc.T)/2-dc))))
assert discrepancy<=1e-12
w=f['W14'].astype(np.float64);nextmean=np.sum(w*(dc@w),axis=0)*f['response14']
preds=dict(mean=dm,covariance=dc,next_mean=nextmean)

def metrics(ta,tb,p):
 signal=float(np.sum(ta*tb));res=float(np.sum((ta-p)*(tb-p)))
 a=float(np.sum((p-ta)**2)/np.sum(ta*ta));b=float(np.sum((p-tb)**2)/np.sum(tb*tb))
 ratio=res/signal if signal>0 else None
 return dict(baseline_replica_cross_energy=signal,residual_replica_cross_energy=res,
             signal_residual_ratio=ratio,ordinary_prediction_targetA=a,ordinary_prediction_targetB=b,
             passed=signal>0 and ratio<=.75 and max(a,b)<=.85)

consumers={name:metrics(old[f'target_A_{name}'],old[f'target_B_{name}'],p) for name,p in preds.items()}
fv={}
i,j=np.triu_indices(len(s),1)
for stream in ('A','B'):
 t=dict(np.load(ASTRA/f'fourth_trace_oracle_r35_{stream}_v1.npz',allow_pickle=False))
 k=(t['k4']-f['pre13_k4'])/s**4
 p31=(t['Q31']-f['pre13_k31'])/(s[:,None]**3*s[None,:])
 p22=(t['Q22']-f['pre13_k22'])/np.outer(s*s,s*s);p22=(p22+p22.T)/2
 fv[stream]=dict(joint=vector(k,p31,p22),k4=k,q31=np.concatenate((p31[i,j],p31[j,i])),q22=p22[i,j])
predfaces=dict(joint=vector(*ds),k4=ds[0],q31=np.concatenate((ds[1][i,j],ds[1][j,i])),q22=ds[2][i,j])
face_scores={name:metrics(fv['A'][name],fv['B'][name],arr) for name,arr in predfaces.items()}
np.savez(out.with_suffix('.npz'),**preds,k4=k4,q31=q31,q22=q22)
record=dict(passed=consumers['covariance']['passed'] and consumers['next_mean']['passed'],
            consumers=consumers,faces=face_scores,exact_direct_map_max=discrepancy,
            capture_sha256=sha(HERE/'capture.npz'),output_sha256=sha(out.with_suffix('.npz')),
            utc_finished=datetime.now(timezone.utc).isoformat(),
            qualification='One fixed passive analytic post12 reset, unchanged historical parent; same reused rank35 truth as oracle capacity. Not full recurrence, S31 endpoint, adjusted score or generalization.',
            check32_accessed=False,valid40_accessed=False,new_samples=False,candidate_built=False)
with out.open('x') as o:json.dump(record,o,indent=2);o.write('\n')
print(json.dumps(record,indent=2),flush=True)
