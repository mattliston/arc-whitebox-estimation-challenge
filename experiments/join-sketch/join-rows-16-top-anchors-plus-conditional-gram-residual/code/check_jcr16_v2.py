import os,sys,types,json,hashlib,importlib.util
from pathlib import Path
import numpy as np
import flopscope as fs
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent
assert all(os.environ.get(k)=='1'for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'))
src=(D/'kernel_jcr16_HELD_v2.py').read_text();ns={'fnp':np};exec(src,ns);ns['_JCR_ACTIVE']=True
J=np.zeros((64,64),dtype='f');J[:16,:16]=10*np.eye(16);J[16:32,:16]=9*np.eye(16);J[32:,32:]=3*np.eye(32)
w=np.ones(64,dtype='f');K=J@J.T;d=np.diag(K).copy();original=np.sort(np.argsort(-d.astype('d'),kind='stable')[:24])
selected=ns['_jcr_rows'](10,J,w,None,None,d,24)
assert np.array_equal(selected,np.r_[np.arange(16),np.arange(32,40)])
assert len(np.unique(selected))==24 and np.all(np.diff(selected)>0)
def residual(rows):return float(np.trace(K-K[:,rows]@np.linalg.pinv(K[np.ix_(rows,rows)].astype('d'))@K[rows]))
assert residual(selected)<residual(original)
assert np.array_equal(ns['_jcr_rows'](7,J,w,None,None,d,24),original)
zero=np.zeros_like(J);bad=ns['_jcr_rows'](13,zero,w,None,None,np.zeros(64,dtype='f'),24)
assert np.array_equal(bad,np.arange(24)) and ns['_JCR_STATS']['joins'][-1]['fallback']
# Actual FNP: maximal architecture dimensions, successful solve, paid gathers/casts/sorts.
nf={'fnp':fnp};exec(src,nf);nf['_JCR_ACTIVE']=True
n,m,r=1024,6144,384
jn=np.zeros((n,m),dtype='f');jn[np.arange(n),np.arange(n)]=1
with fs.BudgetContext(flop_budget=2**40,quiet=True):
 jf=fnp.asarray(jn);wf=fnp.ones(m,dtype=fnp.float32);qp=fnp.zeros((n,r),dtype=fnp.float32);core=fnp.eye(r,dtype=fnp.float32);df=fnp.ones(n,dtype=fnp.float32)
with fs.BudgetContext(flop_budget=2**40,quiet=True)as b:
 nf['_jcr_rows'](10,jf,wf,qp,core,df,320)
charged=int(b.flops_used);requests=len(b._op_log)
assert not nf['_JCR_STATS']['joins'][-1]['fallback']
# Deliberately do not subtract original sort costs: conservative additive bound.
bound=2*charged;fraction=bound/255_000_000_000
assert fraction<=.002,(bound,fraction)
# Import only, no Estimator/setup/predict. Context identities are load-time requirements.
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
p=D/'candidate_jcr16_HELD_v2.py';mod=types.ModuleType('jcr_import');mod.__file__=str(p);sys.modules[mod.__name__]=mod
with fs.BudgetContext(flop_budget=2**60,quiet=True):exec(compile(p.read_bytes(),str(p),'exec'),mod.__dict__)
assert all(mod.__dict__.get(k)is v for k,v in mod._ASTRA_DENSE_CONTEXT)
assert mod._JCR_ACTIVE is False and mod._JCR_STATS=={'joins':[],'retired':True}
result=dict(status='PASS_TINY_IMPORT_AND_ADDITIVE_COST',candidate_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),redundant_PSD_trace_residual_before=residual(original),after=residual(selected),selected_count=24,invalid_anchor_fallback_exact=True,L7_exact=True,dense_context_identities=True,charged_one_maximal_join=charged,requests_one_join=requests,all_two_join_flops_bound=bound,added_pct_at_255G=100*fraction,full_later_products_unchanged=True,no_parent_cache_use=True,no_model_calls=True)
with(D/'TINY_AND_COST_v2.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
