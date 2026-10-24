import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import ast,hashlib,json
from pathlib import Path
import numpy as np
import flopscope as fs
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent
R=D.parents[2]
S=R/'astra_oct18_lead_v1/cov_first_diag_split_v1'
build=json.loads((S/'BUILD_v1.json').read_text())
candidate=Path(build['candidate']);assert hashlib.sha256(candidate.read_bytes()).hexdigest()==build['candidate_sha256']
assert fs.__version__.split('+')[0]=='0.12.3'
s=(S/'fragment_v1.py').read_text();lines=s.splitlines(keepends=True)
node=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)and n.name=='_o18_diag_split_first')
helper=''.join(lines[node.lineno-1:node.end_lineno])
rng=np.random.default_rng(935)
results=[]
for m,n in((5,8),(8,8),(9,3)):
 for previous in(-1,0,13,14):
  c=rng.normal(size=(m,m)).astype('f4');c=(c+c.T)*np.float32(.5);c[np.diag_indices(m)]+=np.float32(4)
  w=rng.normal(size=(m,n)).astype('f4')
  with fs.BudgetContext(flop_budget=2**30,quiet=True):C=fnp.asarray(c);W=fnp.asarray(w);scratch=fnp.empty((m,n),dtype=fnp.float32)
  ns=dict(fnp=fnp,_PHYS_LAYER=[previous],_O18_DS_STATS=dict(calls=0,layers=[]))
  inputs=[]
  def mm(a,b):
   inputs.append(a)
   fnp.copyto(scratch,fnp.matmul(a,b))
   return scratch
  ns['_mm']=mm;exec(helper,ns)
  with fs.BudgetContext(flop_budget=2**30,quiet=True)as baseline:plain=mm(C,W)
  dense=np.asarray(plain).copy()
  with fs.BudgetContext(flop_budget=2**30,quiet=True)as charged:answer=ns['_o18_diag_split_first'](C,W)
  got=np.asarray(answer).copy()
  assert np.asarray(C).tobytes()==c.tobytes()and np.asarray(W).tobytes()==w.tobytes()
  active=0<=previous<=13
  if active:
   assert not np.shares_memory(np.asarray(answer),np.asarray(scratch))
   assert not np.shares_memory(np.asarray(inputs[-1]),np.asarray(C))
   assert np.all(np.diag(np.asarray(inputs[-1]))==0)
   assert ns['_O18_DS_STATS']==dict(calls=1,layers=[previous+1])
   with fs.BudgetContext(flop_budget=2**30,quiet=True):fnp.copyto(scratch,fnp.full((m,n),99,dtype=fnp.float32))
   assert np.asarray(answer).tobytes()==got.tobytes()
  else:assert got.tobytes()==dense.tobytes()
  target=c.astype('d')@w.astype('d')
  assert np.allclose(got,target,atol=2e-6,rtol=2e-6)
  results.append(dict(m=m,n=n,previous=previous,active=active,extra_flops=int(charged.flops_used-baseline.flops_used),extra_charged_ops=len(charged._op_log)-len(baseline._op_log),max_abs_fp64_error=float(np.max(np.abs(got-target))),inputs_unchanged=True,owned_result_survives_kernel_reuse=active))
receipt=dict(passed=True,candidate_sha256=build['candidate_sha256'],flopscope=fs.__version__,scope='12 tiny publicSDK cases. Kernel substitute deliberately returns reusable scratch; active helper owns result and preserves inputs. No full estimator.',rows=results,
    source_peer='PASS exact95c source prefix; only4first-product callsites use helper; previous-layer0..13 guardsL1..14; hiddenrosters/star/secondproduct unchanged.',closure_search='No exact diagonal/offdiagonal first-product split found in bounded memory/TEAM_LOG search.')
with(D/'PEER_TINY_v1.json').open('x')as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(dict(passed=True,active_rows=[r for r in results if r['active']])))
