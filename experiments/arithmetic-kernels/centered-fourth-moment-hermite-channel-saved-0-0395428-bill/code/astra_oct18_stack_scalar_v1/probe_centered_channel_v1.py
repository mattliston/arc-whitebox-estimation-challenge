"""Tiny pure algebra and public local flopscope 0.12.3 price, no model."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','GOTO_NUM_THREADS'): os.environ[k]='1'
import ast,json,resource,hashlib
from pathlib import Path
import numpy as np
import flopscope as fs
import flopscope.numpy as fnp
HERE=Path(__file__).resolve().parent
SRC=HERE.parent/'astra_oct18_accuracy_v1/setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py'
src=SRC.read_text();tree=ast.parse(src)
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_astra_hf_grouped')
code=ast.get_source_segment(src,node)
ns=dict(fnp=np,_rqd_get=lambda tag,owner,fn:fn())
exec(code,ns)
rng=np.random.default_rng(181621)
rows=[]
for dtype in (np.float64,np.float32):
 n=32; aa=rng.normal(size=(7,n)).astype(dtype);bb=rng.normal(size=(7,n)).astype(dtype)
 m=rng.normal(size=n).astype(dtype);s=np.exp(rng.normal(size=n)*.1).astype(dtype)
 rho=rng.normal(size=(n,n)).astype(dtype)*.03;rho=(rho+rho.T)/2
 t21=rng.normal(size=(n,n)).astype(dtype)*.03;t22=rng.normal(size=(n,n)).astype(dtype)*.001;t22=(t22+t22.T)/2
 t3=rng.normal(size=n).astype(dtype)*.02;t4=rng.normal(size=n).astype(dtype)*.01;q=rng.normal(size=n).astype(dtype)*.01
 def series(center):
  bc=bb-(2*m/s)*aa if center else bb
  F=np.stack((aa,bb,bc),axis=1)[:,:,:,None];G=np.stack((aa,aa,bc),axis=1)[:,:,None,:]
  F5=np.stack((F,F*t3[None,None,:,None],F*t4[None,None,:,None]));G5=np.stack((G,G*t3[None,None,None,:],G*t4[None,None,None,:]))
  pp=np.stack((rho,rho*rho/2,rho*rho*rho/6))[:,None]
  X2=np.empty((5,2,3,n,n),dtype=dtype);T=np.empty((5,3,n,n),dtype=dtype);Z=np.empty_like(T)
  a=dict(Z=Z,X1=np.empty((3,3,n,n),dtype=dtype),T=T,X2=X2,X2r1=X2[:,1:2],RR1=rho[None,None,None],div=np.asarray([2.,4.,6.,24.],dtype=dtype)[:,None,None,None])
  raw=ns['_astra_hf_grouped'](F,G,F5,G5,dict(P6=pp),t21,t21.T,t22,a,lambda x:x,q[None,:,None],q[None,None,:],persistent_fg=False)
  cov=raw[0]*np.outer(s,s);pk21=raw[1]*np.outer(s*s,s);pk22=raw[2]*np.outer(s*s,s*s)
  if center: post22=pk22-2*cov*cov
  else: post22=pk22-(2*m)[:,None]*pk21.T-(2*m)*pk21+(2*m)[:,None]*(2*m)*cov-2*cov*cov
  return cov,pk21,post22
 old=series(False);new=series(True)
 rows.append(dict(dtype=str(dtype),cov_bit_equal=bool(np.array_equal(old[0],new[0])),pk21_bit_equal=bool(np.array_equal(old[1],new[1])),post22_max_abs=float(np.max(np.abs(old[2]-new[2]))),post22_relative_fro=float(np.linalg.norm(old[2]-new[2])/np.linalg.norm(old[2])),radial_sum_old=float(np.sum(old[2],dtype=np.float64)),radial_sum_new=float(np.sum(new[2],dtype=np.float64))))
 assert np.array_equal(old[0],new[0]) and np.array_equal(old[1],new[1])
 if dtype==np.float64:assert np.allclose(old[2],new[2],atol=1e-12,rtol=1e-12)
n=1024
with fs.BudgetContext(flop_budget=10**10,quiet=True):
 aa,bb=fnp.asarray(rng.normal(size=(7,n))),fnp.asarray(rng.normal(size=(7,n)))
 m,s=fnp.asarray(rng.normal(size=n)),fnp.asarray(np.exp(rng.normal(size=n)*.1))
 cov,pk21,pk22=[fnp.asarray(rng.normal(size=(n,n)).astype('f4')) for _ in range(3)]
prices=[]
for center in (False,True):
 with fs.BudgetContext(flop_budget=10**10,quiet=True) as budget:
  m2=(2*m).astype(fnp.float32)
  if center:
   bc=bb-(2*m/s)*aa
   post22=pk22-(2*cov)*cov
  else:
   post22=((pk22-fnp.einsum('i,ji->ij',m2,pk21,optimize=False))-m2*pk21)+fnp.einsum('i,j,ij->ij',m2,m2,cov,optimize=False)-(2*cov)*cov
  post22=post22.astype(fnp.float64)
  fnp.fill_diagonal(post22,m)
  total=fnp.sum(post22)
 prices.append(int(budget.flops_used))
result=dict(source_sha256=hashlib.sha256(src.encode()).hexdigest(),flopscope_version=fs.__version__,numpy_version=np.__version__,algebra=rows,postprocess_prices=prices,flops_saved_per_layer=prices[0]-prices[1],flops_saved_14_layers=14*(prices[0]-prices[1]),peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Synthetic 32x32 actual grouped Hermite function, local FNP 1024x1024 postprocess; no boundary, model or native evidence.')
with (HERE/'PROBE_CENTERED_CHANNEL_v1.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
