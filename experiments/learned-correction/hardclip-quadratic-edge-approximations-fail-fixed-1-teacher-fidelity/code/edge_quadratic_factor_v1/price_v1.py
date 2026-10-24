"""Official tiny exact polynomial expansion plus metadata-only actual first7 prices.
This prices one prospective architecture; it does not fit/read training arrays.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
import ast,hashlib,importlib.util,json,resource,subprocess,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;V=HERE.parent;R=V.parent
BASE=V/'msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py'
SHA='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
OFFICIAL=R/'astra_crunch_runtime_v1/failure336631_v1/flopscope_0.12.3'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def main():
 assert sha(BASE)==SHA and not (HERE/'PRICE_v1.json').exists()
 sys.path.insert(0,str(OFFICIAL));import flopscope as fs
 from flopscope import numpy as fnp
 tree=ast.parse(BASE.read_text())
 defs={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 env=dict(fnp=fnp,_O20_ECUT_RADICAL_ON=True,_ECUT_STANDIN=True,_O20_ECUT_RADICAL_A=-.2684473340048872,
      _O20_ECUT_RADICAL_B=1.5215732663431163,_ECUT_STATS=dict(layers=0,kept=0,full=0),
      _EDGE_QUADRATIC_LITERAL=[.01*(i-6) for i in range(14)],_EDGE_QUADRATIC_CONSTANT=.023,_EDGE_QUADRATIC_OUTPUT_BOUND=.8)
 exec(compile(ast.Module(body=[defs['_E2Hold'],defs['_ecut_raw']],type_ignores=[]),'<edge>','exec'),env)
 exec((HERE/'helper_v1.py').read_text(),env)
 old,new=env['_ecut_raw'],env['_quadratic_edge_raw']
 rng=np.random.default_rng(20261021);cases=[]
 for n,ni,nj,mult in [(8,3,5,.3),(12,7,9,1.),(16,16,16,4.)]:
  with fs.BudgetContext(flop_budget=2**35,quiet=True) as budget:
   def arr(shape,scale=1):return fnp.asarray((rng.normal(size=shape)*scale).astype('f4'))
   left,right=arr((n,14),mult),arr((n,14),mult)
   t=dict(bias=arr((14,),.2),output=arr((14,),.1))
   k=dict(R=arr((n,n),mult),S=arr((n,n),mult),e0=arr((14,),.1),e1=arr((14,),.1),e2=arr((14,),.1))
   pf=arr((n,),.1);gf=arr((n,),.2)
   env['_E2_LAYER']=dict(ecut_rows=fnp.arange(ni,dtype=fnp.int64),ecut_cols=fnp.arange(nj,dtype=fnp.int64))
   args=(left,right,t,k,pf,gf,[])
   before,c=budget.flops_used,len(budget.op_log);old(*args);oldf,oldc=budget.flops_used-before,len(budget.op_log)-c
   bills=[];calls=[];ys=[]
   for phase in ('cold','warm'):
    before,c=budget.flops_used,len(budget.op_log);out=new(*args)
    bills.append(budget.flops_used-before);calls.append(len(budget.op_log)-c);ys.append(np.asarray(out).copy())
   np.testing.assert_array_equal(ys[0],ys[1])
   l,r,b=(np.asarray(x).astype('f8') for x in (left,right,t['bias']))
   e=np.stack([np.asarray(k['e'+str(i)]).astype('f8') for i in range(3)])
   rr,ss=np.asarray(k['R']).astype('f8'),np.asarray(k['S']).astype('f8')
   z=(l[:ni]+b)[:,None,:]+r[:nj][None,:,:]
   z+=rr[:ni,:nj,None]*e[0];z+=ss[:ni,:nj,None]*e[1];z+=ss[:nj,:ni].T[:,:,None]*e[2]
   q1=np.asarray(t['output']).astype('f8');q2=np.asarray(env['_EDGE_QUADRATIC_LITERAL'],dtype='f4').astype('f8')
   rawz=z@q1+(z*z)@q2+np.float32(env['_EDGE_QUADRATIC_CONSTANT'])
   rawz=np.clip(rawz,-.8,.8)
   rawz*=np.asarray(pf).astype('f8')[:ni,None]*np.asarray(gf).astype('f8')[None,:nj]
   oracle=np.zeros((n,n));oracle[:ni,:nj]=rawz
   error=float(np.max(np.abs(ys[-1]-oracle)));scale=max(float(np.max(np.abs(oracle))),1e-10)
   assert error<2e-8+3e-6*scale,(error,scale)
   cases.append(dict(n=n,rows=ni,cols=nj,old_flops=oldf,new_flops=bills,old_calls=oldc,new_calls=calls,
        max_expansion_abs_error=error,oracle_scale=scale,cold_warm_bytes_equal=True))
 # Existing reviewed metadata backend; all following large arrays are zero-stride.
 helper=R/'astra_oct20_bill_v1/fusion_shape_frontier_v1/price_v2.py'
 spec=importlib.util.spec_from_file_location('quad_metadata_tariff',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 mod,fn,fs2,measure,backend=module.load()
 # Add shape-only einsum support; numerical tiny checks above used real official execution.
 from flopscope import _budget
 old_backend=_budget._call_numpy
 einsum_counts={}
 def metadata_backend(function,*args,**kwargs):
  if getattr(function,'__name__','')=='matmul':
   a,b=args[:2];assert a.ndim>=1 and b.ndim>=1
   assert a.shape[-1]==b.shape[-1 if b.ndim==1 else -2]
   ashape=(1,a.shape[0]) if a.ndim==1 else a.shape
   bshape=(b.shape[0],1) if b.ndim==1 else b.shape
   shape=np.broadcast_shapes(ashape[:-2],bshape[:-2])+ashape[-2:-1]+bshape[-1:]
   if a.ndim==1:shape=shape[:-2]+shape[-1:]
   if b.ndim==1:shape=shape[:-1]
   if kwargs.get('out') is not None:return kwargs['out']
   return np.broadcast_to(np.zeros((),dtype=np.result_type(a.dtype,b.dtype)),shape)
  if getattr(function,'__name__','')=='clip':
   if kwargs.get('out') is not None:return kwargs['out']
   return np.broadcast_to(np.zeros((),dtype=args[0].dtype),args[0].shape)
  if getattr(function,'__name__','')=='einsum':
   equation=args[0];assert '...' not in equation and '->' in equation
   left,output=equation.split('->');terms=left.split(',');operands=args[1:]
   assert len(terms)==len(operands);dimensions={}
   for term,operand in zip(terms,operands):
    assert len(term)==len(operand.shape)
    for letter,size in zip(term,operand.shape):
     assert letter not in dimensions or dimensions[letter]==size
     dimensions[letter]=size
   shape=tuple(dimensions[letter] for letter in output)
   einsum_counts[equation]=einsum_counts.get(equation,0)+1
   return np.broadcast_to(np.zeros((),dtype=np.result_type(*[x.dtype for x in operands])),shape)
  return old_backend(function,*args,**kwargs)
 for name,m in list(sys.modules.items()):
  if name.startswith('flopscope') and getattr(m,'_call_numpy',None) is old_backend:m._call_numpy=metadata_backend
 env=dict(fnp=fn,_E2Hold=mod._E2Hold,_E2_LAYER={},_EDGE_QUADRATIC_LITERAL=[.01*(i-6) for i in range(14)],_EDGE_QUADRATIC_CONSTANT=.023,_EDGE_QUADRATIC_OUTPUT_BOUND=.8)
 exec((HERE/'helper_v1.py').read_text(),env)
 counts=json.loads((V/'edge_clip14_output_v1/CAPTURE_PRICE_INPUT_v1.json').read_text())
 shapes=sorted({(x['rows'],x['cols']) for r in counts['rows'].values() for x in r['layers']})
 prices={}
 for ni,nj in shapes:
  with fs2.BudgetContext(flop_budget=2**60,quiet=True):
   left,right=fn.empty((1024,14),dtype=fn.float32),fn.empty((1024,14),dtype=fn.float32)
   t=dict(bias=fn.empty(14,dtype=fn.float32),output=fn.empty(14,dtype=fn.float32))
   k=dict(R=fn.empty((1024,1024),dtype=fn.float32),S=fn.empty((1024,1024),dtype=fn.float32),**{'e'+str(i):fn.empty(14,dtype=fn.float32) for i in range(3)})
   pf,gf=fn.empty(1024,dtype=fn.float32),fn.empty(1024,dtype=fn.float32)
   ij=dict(ecut_rows=fn.arange(ni,dtype=fn.int64),ecut_cols=fn.arange(nj,dtype=fn.int64))
  mod._E2_LAYER=ij;env['_E2_LAYER']=ij
  args=(left,right,t,k,pf,gf,[])
  old=measure(lambda:mod._ecut_raw(*args))
  cold=measure(lambda:env['_quadratic_edge_raw'](*args))
  warm=measure(lambda:env['_quadratic_edge_raw'](*args))
  prices[str((ni,nj))]=dict(rows=ni,cols=nj,parent=old,cold=cold,warm=warm)
 rows=[]
 for rank,r in counts['rows'].items():
  credit=sum(prices[str((x['rows'],x['cols']))]['parent']['flops']-prices[str((x['rows'],x['cols']))]['warm']['flops'] for x in r['layers'])
  # Conservatively pay cold coefficient setup on every layer, not just once.
  cold=sum(prices[str((x['rows'],x['cols']))]['parent']['flops']-prices[str((x['rows'],x['cols']))]['cold']['flops'] for x in r['layers'])
  calls=sum(prices[str((x['rows'],x['cols']))]['warm']['charged']-prices[str((x['rows'],x['cols']))]['parent']['charged'] for x in r['layers'])
  rows.append(dict(rank=int(rank),warm_credit_flops=credit,warm_bill_pct=-100*credit/r['reference_flops'],
       conservative_allcold_bill_pct=-100*cold/r['reference_flops'],warm_charged_call_delta=calls))
 result=dict(utc=utc(),status='SOURCE_ONLY_COMPLETE_PATH_TARIFF_AND_POLYNOMIAL_ALGEBRA_PASS',
    parent_sha256=SHA,reference_submission=337422,helper_sha256=sha(HERE/'helper_v1.py'),script_sha256=sha(__file__),
    fixed_architecture='29-coefficient shared quadratic in existing14 preactivations; expanded pair algebra; fixed original-teacher output-L1 clamp before physical pdf/gate multiplication',
    numerical_cases=cases,first7=rows,first7_mean_warm_bill_pct=float(np.mean([r['warm_bill_pct'] for r in rows])),
    first7_conservative_cold_bill_pct=float(np.mean([r['conservative_allcold_bill_pct'] for r in rows])),
    priced_shapes=len(prices),prices=prices,metadata_backend=backend,metadata_einsum_shapes=einsum_counts,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    uses_training_arrays=False,uses_truth=False,native_predictions=0,
    limitations=['Polynomial fidelity and frequency of the fixed output clamp are untested.','Roster projections are first7 only; native feedback may change rosters.','No estimated adjusted gain.'])
 with (HERE/'PRICE_v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ('prices','metadata_backend')},indent=2))
if __name__=='__main__':main()

