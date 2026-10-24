import os,sys,types,math,json,hashlib
from pathlib import Path
import numpy as np
import flopscope as fs
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent
assert all(os.environ.get(k)=='1'for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'))
fragment=(D/'fragment_rtt2_HELD_v1.py').read_text();rng=np.random.default_rng(20261015)
class Hold(list):
 def __call__(self,x):self.append(x);return x
m,k,n,old,rank=64,32,64,16,8
L=rng.normal(size=(m,old+k)).astype('f');R=rng.normal(size=(n,k)).astype('f');lam=rng.normal(size=old+k).astype('f');U=rng.normal(size=(n,rank)).astype('f');B=rng.normal(size=(old,rank)).astype('f');mu=rng.normal(size=m).astype('d');vc=np.ones(m)*2
def make(lib):
 def cv(x):return lib.asarray(x)
 ns={};calls=[]
 def inner(l,la,state):calls.append(1);return (l*la)@state['B']
 def parent(l,r,la,indices,state):
  young=(l[:,old:]*la[old:])@r.T
  return young+ns['_cp_slr_inner_sq'](l[:,:old],la[:old],state)@state['pre_basis'][indices].T
 class E:
  def predict(self,fail=False):
   if fail:raise RuntimeError('synthetic')
   return 'ok'
 ns.update(fnp=lib,_E2Hold=Hold,_rc_math=math,_RC_PARENT_READ=parent,_yr7_read=parent,_cp_slr_inner_sq=inner,Estimator=E,_RC_COLZ=2.0,_YR7_SHAPES=frozenset(((m,k,n,rank),)),_R7_SHAPES=frozenset(),_RS_LAYER=[7],_E2_LAYER={'g':cv(np.r_[np.ones(32),np.zeros(32)]),'vc':cv(vc),'mu':cv(mu),'ecut_rows':cv(np.arange(m))},_YR7_ON=True,_YR7_ACTIVE=[True],_WINOGRAD=True,_AB_ON=True,_SCRATCH=[types.SimpleNamespace(shape=(391602176,))],_ASTRA_RLF_CONTEXT=[None],_UBX_CTX=[None],_UBC_CONTEXT=[None],_UBX_RC=[None],_xsg_rows=lambda tag,a,idx:a[idx])
 exec(fragment,ns);ns['_RTT_ACTIVE']=True
 return ns,calls,dict(pre_basis=cv(U),columns=old,B=cv(B))
ns,calls,state=make(np);out=ns['_rtt_read'](L,R,lam,np.arange(n),state);Q=ns['_rtt_basis'](vc,mu,Hold());young=(L[:,old:]*lam[old:])@R.T;oldplane=((L[:,:old]*lam[:old])@B)@U.T
expected=np.concatenate((young[:,:32]+oldplane[:,:32],Q@(Q.T@young[:,32:])+oldplane[:,32:]),axis=1)
err=float(np.max(np.abs(out-expected)));assert np.allclose(out,expected,rtol=3e-5,atol=3e-5),(err,out.shape)
assert len(calls)==1 and ns['_RTT_CONTEXT'][0]is None and ns['_YR7_SHAPES']==ns['_RTT_FIXED_YR7_SHAPES']
record=ns['_RTT_STATS']['reads'][0];assert record['old_inner_calls']==1 and record['forced_d7']and record['c15']==32 and record['c2']==64
assert ns['_rtt_basis'](vc,vc,Hold()).shape==(m,1)
assert ns['_rtt_basis'](vc*0,vc*0,Hold()).shape==(m,0)
ns['_RTT_ACTIVE']=False
assert ns['Estimator']().predict()=='ok' and not ns['_RTT_ACTIVE']
try:ns['Estimator']().predict(fail=True)
except RuntimeError:pass
else:raise AssertionError('exception lost')
assert ns['_RTT_STATS']['retired']and not ns['_RTT_ACTIVE']and ns['_RTT_CONTEXT'][0]is None
# One small actual FNP execution; forecast/capture is reused, no large tariff sweep.
with fs.BudgetContext(flop_budget=2**30,quiet=True):
 nf,fcalls,st=make(fnp);lf=fnp.asarray(L);rf=fnp.asarray(R);la=fnp.asarray(lam);ids=fnp.arange(n)
with fs.BudgetContext(flop_budget=2**30,quiet=True)as budget:
 fo=nf['_rtt_read'](lf,rf,la,ids,st)
assert len(fcalls)==1 and nf['_RTT_CONTEXT'][0]is None
assert np.allclose(np.asarray(fo),expected,rtol=3e-5,atol=3e-5)
# Full source import only, no estimator/setup/predict.
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
p=D/'candidate_rtt2_HELD_v1.py';mod=types.ModuleType('rtt_import');mod.__file__=str(p);sys.modules[mod.__name__]=mod
with fs.BudgetContext(flop_budget=2**60,quiet=True):exec(compile(p.read_bytes(),str(p),'exec'),mod.__dict__)
assert all(mod.__dict__.get(k)is v for k,v in mod._ASTRA_DENSE_CONTEXT)
assert mod._RC_COLZ==2 and mod._RTT_CONTEXT==[None]and not mod._RTT_ACTIVE and mod._YR7_SHAPES==mod._RTT_FIXED_YR7_SHAPES
receipt=dict(status='PASS_TINY_IMPORT_AND_SCOPE',candidate_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),exact_parent_prefix=True,rank2_projection_max_abs=err,old_inner_calls=1,exact_old_tail=True,original_c2_mask_unchanged=True,depth7_scope_restored=True,rank_deficient_cases=[1,0],exception_cleanup=True,full_import_context_exact=True,small_FNP_flops=int(budget.flops_used),small_FNP_requests=len(budget._op_log),no_model_calls=True)
with(D/'TINY_SCOPE_v1.json').open('x')as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt))
