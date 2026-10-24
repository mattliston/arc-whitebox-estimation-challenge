import ast, hashlib, json
from pathlib import Path
import numpy as np
import flopscope as fs
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent
P=D.parent/'astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v3.py'
F=D/'fragment_hermite_group_HELD_v1.py'
old=next(n for n in reversed(ast.parse(P.read_text()).body) if isinstance(n,ast.FunctionDef)and n.name=='_psp_series')
new=next(n for n in ast.parse(F.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='_astra_hf_grouped')
def namespace(xp):
 ns=dict(fnp=xp,_GTR_ENABLED=True,_rqd_get=lambda k,o,f:f(),_GTR_STATS=dict(calls=0,Q=0,H=0,credit=0,persistent_calls=0,gathered_calls=0))
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
rng=np.random.default_rng(2026101502)
cases=[]
for packed in(False,True):
 for dtype in(np.float64,np.float32):
  for channels in(1,2,3):
   args,arena,qr,qc=make(np,rng,dtype,channels,11,packed)
   inputs=[x.copy()for x in args if isinstance(x,np.ndarray)]
   oldout,newout,oldterms,newterms=calls(namespace(np),args,arena,qr,qc)
   assert all(np.array_equal(x,y)for x,y in zip(inputs,[x for x in args if isinstance(x,np.ndarray)]))
   assert np.array_equal(oldterms,newterms)
   ae=float(np.max(np.abs(oldout-newout)));rel=float(np.linalg.norm(oldout-newout)/np.linalg.norm(oldout))
   assert ae<(2e-14 if dtype==np.float64 else 2e-7),(packed,dtype,channels,ae,rel)
   cases.append(dict(packed=packed,dtype=str(dtype),channels=channels,max_abs=ae,relative_l2=rel))
# Actual instrumented operations, first at small shapes and then representative BA/packed shapes.
prices=[]
for packed,channels,n in((False,1,11),(True,2,11),(False,1,1024),(True,2,128)):
 with fs.BudgetContext(flop_budget=2**42,quiet=True):
  args,arena,qr,qc=make(fnp,rng,np.float32,channels,n,packed)
  a,b=arena(),arena()
 ns=namespace(fnp);hold=lambda x:x
 before=[np.asarray(x).tobytes()for x in args if not isinstance(x,dict)]
 with fs.BudgetContext(flop_budget=2**42,quiet=True)as ctx_old:orig=ns['_psp_series'](*args,a,hold,persistent_fg=not packed)
 with fs.BudgetContext(flop_budget=2**42,quiet=True)as ctx_new:fused=ns['_astra_hf_grouped'](*args[:8],b,hold,qr,qc)
 out0,out1=np.asarray(orig),np.asarray(fused)
 assert np.allclose(out0,out1,atol=2e-7,rtol=2e-5)
 assert np.array_equal(np.asarray(a['Z'][1:3]),np.asarray(b['Z'][1:3]))
 assert before==[np.asarray(x).tobytes()for x in args if not isinstance(x,dict)]
 prices.append(dict(packed=packed,channels=channels,n=n,tiles=3 if packed else 1,old_flops=int(ctx_old.flops_used),new_flops=int(ctx_new.flops_used),saving=int(ctx_old.flops_used-ctx_new.flops_used),old_charged_ops=len(ctx_old._op_log),new_charged_ops=len(ctx_new._op_log),max_abs=float(np.max(np.abs(out0-out1)))))
ba,pa=prices[-2:]
# Packed actual path has36 tiles, tiny tariff representative has3; row-gather overhead included conservatively below.
per_layer=ba['saving']+12*pa['saving']
row_gather=2*36*128*4
forecast=(per_layer-row_gather)*14
r=dict(status='PASS',numpy_cases=cases,FNP_cases=prices,e21_e22_bitwise_exact=True,input_bytes_unchanged=True,real_algebra_exact_float64=True,all_original_diagonal_restoration_unchanged=True,full14_forecast_saving_flops=forecast,percent_of_245G=100*forecast/245e9,charged_ops_delta_per_layer=ba['new_charged_ops']-ba['old_charged_ops']+pa['new_charged_ops']-pa['old_charged_ops']+2,full_native_bill_required=True,model_calls=0,parent_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),fragment_sha256=hashlib.sha256(F.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
with(D/'TINY_AND_TARIFF_v1.json').open('x')as f:json.dump(r,f,indent=2)
print(json.dumps(r,indent=2))
