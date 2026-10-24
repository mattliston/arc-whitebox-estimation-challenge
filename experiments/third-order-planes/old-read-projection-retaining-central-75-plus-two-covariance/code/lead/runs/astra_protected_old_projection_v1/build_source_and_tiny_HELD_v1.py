from pathlib import Path
import ast,hashlib,json,numpy as np
O=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1');D=O/'lead/runs/astra_protected_old_projection_v1';P=O/'lead/runs/astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v3.py';F=D/'fragment_protected_old_HELD_v1.py';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(P)=='4d7739a5ad4444be8c20bdd5c5bc86f1cee5536a2c21e956d86dd95e6c83d8d9'
raw=P.read_bytes();candidate=raw+b'\n'+F.read_bytes();C=D/'candidate_protected_old_QR_ON_HELD_v1.py';C.write_bytes(candidate);compile(candidate,str(C),'exec');assert C.read_bytes()[:len(raw)]==raw
class Hold:
 def __call__(self,x):return x
class Est:
 def predict(self,*a,**k):return object()
rng=np.random.default_rng(20261015);cases=[]
for m in (1,32,64,96,128,192,512,608,768):
 for degenerate in (False,True):
  left=rng.normal(size=(m,48)).astype(np.float32);lam=rng.normal(size=48).astype(np.float32);factor=rng.normal(size=(48,16)).astype(np.float32);mu=rng.normal(size=m);vc=np.exp(rng.normal(size=m));rows=np.arange(m);calls=[]
  def parent(a,b,state):calls.append(a.shape);return (a*b)@factor
  def basis(v,u,hold):
   if degenerate:return np.ones((m,1),np.float32)/np.sqrt(m)
   return np.linalg.qr(np.stack((v,u),axis=1),mode='reduced')[0].astype(np.float32)
  ns=dict(_RTT_PARENT_INNER=parent,Estimator=Est,_RS_LAYER=[7],_RTT_ACTIVE=True,_E2Hold=Hold,_E2_LAYER=dict(ecut_rows=rows,vc=vc,mu=mu),_ASTRA_RLF_CONTEXT=[None],_rtp_qr_basis=basis,fnp=np)
  exec(compile(F.read_text(),str(F),'exec'),ns);before=left.tobytes();got=ns['_pop_inner'](left,lam,{})
  full=(left*lam)@factor;center=((3*m+255)//256)*64-2
  change=not degenerate and center>=1 and center+2<m
  if change:
   q=basis(vc,mu,None);expected=np.concatenate((full[:center],q[center:]@(q.T@full)),axis=0)
   assert calls==[(center+2,48)]
  else:expected=full;assert calls==[(m,48)]
  assert np.allclose(got,expected,atol=5e-5,rtol=1e-4) and left.tobytes()==before
  cases.append(dict(m=m,degenerate=degenerate,changed=change,max_abs=float(np.max(np.abs(got-expected)))))
# Existing RTT outer wrapper retains exactly the re-expanded result and one call.
tr=ast.parse(raw);n=[n for n in tr.body if isinstance(n,ast.FunctionDef) and n.name=='_cp_slr_inner_sq'][-1];state={};ns['_RTT_CONTEXT']=[dict(state=state,m=m,inner=None,inner_calls=0)];exec(compile(ast.Module(body=[n],type_ignores=[]),'<existing RTT capture>','exec'),ns);got=ns['_cp_slr_inner_sq'](left,lam,state);assert ns['_RTT_CONTEXT'][0]['inner']is got and ns['_RTT_CONTEXT'][0]['inner_calls']==1
r=dict(no_models=True,candidate=str(C),candidate_sha256=sha(C),parent=str(P),parent_sha256=sha(P),fragment=str(F),fragment_sha256=sha(F),parent_prefix_byte_exact=True,fixed_rule='L7..14 center=ceil(0.75*m/64)*64-2; retain center exact rows plus two vc/mu projections; full output expansion before original RTT capture; rank loss delegates unchanged parent',first7_expected_exact=True,young_path_and_exact_scalar_diagonal_unchanged=True,tiny_cases=cases,existing_RTT_capture_same_object_one_call=True,guard_capture_change=False,tariff_prior='1.4696339% warm saving TRAIN16; actual full bill mandatory',error_proxy_prior='9.1311x incumbent QR young-tail leading-covariance-weighted error; not MSE',source_pins={str(P):sha(P),str(F):sha(F)})
R=D/'SOURCE_READY_HELD_v1.json';R.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(candidate=sha(C),ready=sha(R),tiny_cases=len(cases))))
