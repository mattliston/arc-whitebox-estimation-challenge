"""Small nonsymmetric/cold and complete scalar/GNN derivative checks."""
import hashlib,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
from tangent import covariance_value,covariance_jacobian,terminal
HERE=Path(__file__).resolve().parent
assert all(t['num_threads']==1 for t in threadpool_info())
seal=json.loads((HERE/'sources_sealed.json').read_text())
for item in seal['sources']+seal['external_pins']:assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
rng=np.random.Generator(np.random.PCG64DXSM(93092011));n=12;k=5
G=rng.normal(size=(n,n));S=rng.normal(size=(n,n));C=G@G.T/n+np.eye(n)+.03*(S-S.T)
W=rng.normal(size=(n,n))/np.sqrt(n);indices=np.arange(8);alive=np.arange(n)<8;X=np.column_stack([rng.normal(size=(n,k-1)),np.ones(n)])
v,av=covariance_value(C,W,indices,alive,X,np.zeros(k));dv,da=covariance_jacobian(C,W,indices,alive,X)
fdv=[];fda=[]
for i in range(k):
 q=np.zeros(k);q[i]=1e-5;vp,ap=covariance_value(C,W,indices,alive,X,q);vm,am=covariance_value(C,W,indices,alive,X,-q)
 fdv.append((vp-vm)/2e-5);fda.append((ap-am)/2e-5)
rel=lambda x,y:float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
cv=rel(dv,np.stack(fdv,axis=1));ca=rel(da,np.stack(fda,axis=1));assert max(cv,ca)<1e-7
a=dict(C=C,W=W,indices=indices,alive=alive,H14=X[:,:-1],v=v,active_v=av,mu=rng.normal(size=n)*.15,k3base=rng.normal(size=n)*.003,bpre=rng.normal(size=n)*.002,radial=np.asarray(.01))
for name in ['m1','m2','m3']:a[name]=rng.normal(size=(n,2))*.2
for name in ['e1','e2','e3']:a[name]=rng.uniform(.01,.4,size=(n,2))
a.update(P_centers=np.zeros((16,30)),P_scales=np.ones((16,30))*2,P_enc0_w=rng.normal(size=(8,42))*.03,P_enc0_b=rng.normal(size=8)*.02,P_enc2_w=rng.normal(size=(7,8))*.1,P_enc2_b=rng.normal(size=7)*.02,P_enc4_w=rng.normal(size=(6,7))*.1,P_enc4_b=rng.normal(size=6)*.02,P_head=rng.normal(size=(16,6))*.3,P_head_b=rng.normal(size=16)*.01,P_gain=np.zeros(16),P_rms=np.ones(16)*.02)
J=terminal(a,dv,da).d;cols=[]
for i in range(k):
 q=np.zeros(k);q[i]=1e-5;vp,ap=covariance_value(C,W,indices,alive,X,q);vm,am=covariance_value(C,W,indices,alive,X,-q)
 p=terminal(a,dv*0,da*0,v_override=vp,active_override=ap).v;m=terminal(a,dv*0,da*0,v_override=vm,active_override=am).v
 cols.append((p-m)/2e-5)
full=rel(J,np.stack(cols,axis=1));assert full<1e-6
result=dict(passed=True,nonsymmetric_covariance=True,cold_diagonal=True,columns=k,covariance_relative_error=cv,active_covariance_relative_error=ca,complete_terminal_relative_error=full,truth_accessed=False,new_model_calls=0)
with (HERE/'unit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
