import ast, hashlib, json
from pathlib import Path
import numpy as np
import flopscope as fs
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent
P=D.parent/'astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v3.py'
F=D/'fragment_hermite_group_lean_HELD_v1.py'
old=next(n for n in reversed(ast.parse(P.read_text()).body) if isinstance(n,ast.FunctionDef)and n.name=='_psp_series')
new=next(n for n in ast.parse(F.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='_astra_hf_grouped')
def namespace(xp):
 cache={}
 def get(k,o,f):
  key=(k,id(o));v=cache.get(key)
  if v is not None and v[0]is o:return v[1]
  v=f();cache[key]=(o,v);return v
 ns=dict(fnp=xp,_GTR_ENABLED=True,_rqd_get=get,_GTR_STATS=dict(calls=0,Q=0,H=0,credit=0,persistent_calls=0,gathered_calls=0))
 exec(compile(ast.Module(body=[old,new],type_ignores=[]),'extracted','exec'),ns)
 return ns
def make(xp, rng, dtype, channels=1, n=11, packed=False):
 tail=(channels,3,n,n) if packed else(channels,n,n)
 row=(channels,3,n,1) if packed else(channels,n,1)
 col=(channels,3,1,n) if packed else(channels,1,n)
 def rand(shape,scale=1):return xp.asarray(rng.normal(size=shape).astype(dtype)*scale)
 FF=rand((7,)+row);GG=rand((7,)+col)
 f3,f4,g3,g4=rand(row,.02),rand(row,.02),rand(col,.02),rand(col,.02)
 FF5=xp.stack((FF,FF*f3,FF*f4));GG5=xp.stack((GG,GG*g3,GG*g4))
 rho=rand((1,)+tail[1:],.06)
 pp=xp.stack((rho,rho*rho/2,rho*rho*rho/6))
 qrow,qcol=rand(row,.02),rand(col,.02)
 t21,t12,t22=rand(tail,.02),rand(tail,.02),rand(tail,.02)
 t31,t13=qrow*rho,qcol*rho
 def arena():
  X2=xp.empty((5,2)+tail,dtype=dtype)
  a=dict(X1=xp.empty((3,)+tail,dtype=dtype),X2=X2,X3=xp.empty((4,2)+tail,dtype=dtype),T=xp.empty((5,)+tail,dtype=dtype),U=xp.empty((4,)+tail,dtype=dtype),Z=xp.empty((5,)+tail,dtype=dtype),div=xp.asarray(np.array([2,4,6,24],dtype=dtype).reshape((4,)+(1,)*len(tail))))
  a['X2r1']=X2[:,1:2];a['RR1']=rho[None]
  return a
 return (FF,GG,FF5,GG5,dict(P6=pp,P12=pp[:2]),t21,t12,t22,t31,t13),arena,qrow,qcol
def calls(ns,args,arena,qr,qc):
 hold=lambda x:x
 a=arena(); b=arena()
 original=ns['_psp_series'](*args,a,hold,persistent_fg=True)
 grouped=ns['_astra_hf_grouped'](*args[:8],b,hold,qr,qc)
 return np.asarray(original).copy(),np.asarray(grouped).copy(),np.asarray(a['Z'][1:3]).copy(),np.asarray(b['Z'][1:3]).copy()

rng=np.random.default_rng(2026101502);cases=[]
for packed in (False,True):
 for dtype in (np.float64,np.float32):
  for channels in (1,2,3):
   args,arena,qr,qc=make(np,rng,dtype,channels,11,packed)
   oldout,newout,oldterms,newterms=calls(namespace(np),args,arena,qr,qc)
   ae=float(np.max(np.abs(oldout-newout)));rel=float(np.linalg.norm(oldout-newout)/np.linalg.norm(oldout))
   assert np.array_equal(oldterms,newterms)
   assert ae<(2e-14 if dtype==np.float64 else 2e-7),(packed,str(dtype),channels,ae)
   cases.append(dict(packed=packed,dtype=str(dtype),channels=channels,max_abs=ae,relative_l2=rel))
prices=[]
for packed,channels,n in ((False,1,11),(True,2,11),(False,1,1024),(True,2,128)):
 with fs.BudgetContext(flop_budget=2**42,quiet=True):
  args,arena,qr,qc=make(fnp,rng,np.float32,channels,n,packed);a,b=arena(),arena()
 ns=namespace(fnp);hold=lambda x:x
 before=[np.asarray(x).tobytes()for x in args if not isinstance(x,dict)]
 with fs.BudgetContext(flop_budget=2**42,quiet=True):
  ns['_psp_series'](*args,a,hold,persistent_fg=not packed);ns['_astra_hf_grouped'](*args[:8],b,hold,qr,qc)
 with fs.BudgetContext(flop_budget=2**42,quiet=True)as co:oldout=ns['_psp_series'](*args,a,hold,persistent_fg=not packed)
 with fs.BudgetContext(flop_budget=2**42,quiet=True)as cn:newout=ns['_astra_hf_grouped'](*args[:8],b,hold,qr,qc)
 assert np.allclose(np.asarray(oldout),np.asarray(newout),atol=2e-7,rtol=2e-5)
 assert np.array_equal(np.asarray(a['Z'][1:3]),np.asarray(b['Z'][1:3]))
 assert before==[np.asarray(x).tobytes()for x in args if not isinstance(x,dict)]
 prices.append(dict(packed=packed,n=n,channels=channels,tiles=3 if packed else 1,old_flops=int(co.flops_used),new_flops=int(cn.flops_used),saving=int(co.flops_used-cn.flops_used),old_ops=len(co._op_log),new_ops=len(cn._op_log)))
ba,pa=prices[-2:];credit=14*(ba['saving']+12*pa['saving']-2*36*128*4)
r=dict(status='TINY_KERNEL_PASS_NOT_FULL_SOURCE',cases=cases,prices=prices,e21_e22_bitwise_exact=True,input_bytes_unchanged=True,forecast_credit_flops=credit,forecast_pct245G=100*credit/245e9,forecast_charged_ops_delta=14*(ba['new_ops']-ba['old_ops']+pa['new_ops']-pa['old_ops']+2),with_dead_T31_credit_flops=credit+44040192,with_dead_T31_ops_delta=14*(ba['new_ops']-ba['old_ops']+pa['new_ops']-pa['old_ops']+2)-42,source_sha256=hashlib.sha256(F.read_bytes()).hexdigest(),models=0)
(D/'TINY_KERNEL_TARIFF_v1.json').open('x').write(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
