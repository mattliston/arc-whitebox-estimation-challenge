import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import ast,json
from pathlib import Path
import numpy as np
import flopscope as fs
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent
build=json.loads((D/'BUILD_v1.json').read_text())
source=Path(build['parent']).read_text();fragment=(D/'fragment_v1.py').read_text()
assert fs.__version__.split('+')[0]=='0.12.3'
def functions(text):
    lines=text.splitlines(keepends=True)
    return {n.name:''.join(lines[n.lineno-1:n.end_lineno]) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
original,changed=functions(source),functions(fragment)
class Hold(list):
    def __call__(self,x):self.append(x);return x
def namespace(branch,previous,modified):
    ns=dict(fnp=fnp,_E2Hold=Hold,_PHYS_LAYER=[previous],_O18_CDIAG_STATS=dict(calls=0,layers=[]),_SS_STATS=dict(planes=0),_DCV_STATS=dict(m=5),_PSDM_STATS=dict(padded=0),_DCV_SYM='tiles' if branch=='tiles' else'pad')
    order=np.array([0,2,4,6,1,3,5,7],dtype='i8')
    ns['_dcv_roster']=lambda gate,floor:None if branch=='full' else fnp.asarray(order[:5] if branch=='tiles' else order)
    ns['_ps32']=lambda C:C.astype(fnp.float32)
    ns['_ss_factors']=lambda wt,wt2,A,hold:(hold(wt[:,:2]),hold(wt2[:,:2]))
    ns['_astra_star_scaled_product']=lambda T,P,hold:fnp.matmul(T,P.T)/3
    def first(A,B):
        product=fnp.matmul(A,B);ns['first']=product;ns['weights']=B;return product
    def approximate(A,B,*unused):
        product=fnp.matmul(A,B)
        fnp.fill_diagonal(product,fnp.diag(product)+0.03125)
        return product
    ns.update(_mm=first,_sym_upper=approximate,_dcv_tiles=approximate,_psdm_sym_upper=approximate,_psdm_fits=lambda m:True,_xsg_rows=lambda tag,w,rows:w[rows])
    if modified:exec(changed['_o18_direct_covdiag'],ns)
    for name in ('_flms_cov_star','_dc_cov_star'):exec((changed if modified else original)[name],ns)
    return ns
rng=np.random.default_rng(935)
w=rng.normal(size=(8,8)).astype('f4');q=rng.normal(size=(8,8));C=q@q.T;A=rng.normal(size=(8,8)).astype('f4');gate=np.linspace(.1,.9,8)
records=[]
for branch in ('full','pad','tiles'):
 for previous in (-1,0,13,14):
  for retained in (False,True):
    outputs=[];bills=[];ops=[]
    for modified in (False,True):
      ns=namespace(branch,previous,modified)
      with fs.BudgetContext(flop_budget=2**40,quiet=True):inp=[fnp.asarray(x)for x in(w,C,A,gate)]
      with fs.BudgetContext(flop_budget=2**40,quiet=True)as budget:values=ns['_dc_cov_star'](*inp,0.01,retain32=retained)
      got=[np.asarray(v).copy()for v in values]
      expected=np.sum(np.asarray(ns['weights'])*np.asarray(ns['first']),axis=0,dtype=np.float64)
      for before,after in zip((w,C,A,gate),inp):assert before.tobytes()==np.asarray(after).tobytes()
      if modified and 0<=previous<=13:
        assert np.diag(got[0]).tobytes()==expected.tobytes()
        if retained:assert np.diag(got[2]).tobytes()==expected.astype('f4').tobytes()
        assert ns['_O18_CDIAG_STATS']==dict(calls=1,layers=[previous+1])
      assert got[0].dtype==np.float64
      if retained:
        assert got[2].dtype==np.float32
        assert not np.shares_memory(np.asarray(values[0]),np.asarray(values[2]))
      outputs.append(got);bills.append(int(budget.flops_used));ops.append(len(budget._op_log))
    mask=~np.eye(8,dtype=bool)
    for index,(before,after)in enumerate(zip(*outputs)):
      if index in(0,2)and 0<=previous<=13:assert before[mask].tobytes()==after[mask].tobytes()
      else:assert before.tobytes()==after.tobytes()
    records.append(dict(branch=branch,previous=previous,retained=retained,extra_flops=bills[1]-bills[0],extra_charged_ops=ops[1]-ops[0],inputs_unchanged=True,star_unchanged=True,offdiagonals_unchanged=True))
receipt=dict(passed=True,version=fs.__version__,candidate_sha256=build['candidate_sha256'],scope='24 fixed8-unit cases using public0.12.3. Full/cut-pad/cut-tiles branches, active/excluded layers, retain32 bothvalues. Parent finalproduct has a deliberate diagonal perturbation to prove repair placement; kernels are dense test substitutes.',rows=records)
with(D/'TINY_v1.json').open('x')as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(dict(passed=True,cases=len(records),active_extra_flops=sorted(set(r['extra_flops']for r in records if 0<=r['previous']<=13)),active_extra_ops=sorted(set(r['extra_charged_ops']for r in records if 0<=r['previous']<=13)))))
