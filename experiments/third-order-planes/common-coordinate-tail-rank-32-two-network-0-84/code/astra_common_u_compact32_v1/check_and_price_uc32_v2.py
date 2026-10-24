from pathlib import Path
import numpy as np,json,hashlib,os,sys,types
import flopscope as fs
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent;R=D.parent
assert all(os.environ.get(k)=='1'for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'))
class Hold(list):
 def __call__(self,x):self.append(x);return x
rng=np.random.default_rng(20261014)
def split(F,blocks):
 return np.concatenate([(F[:256,a:a+h]+F[:256,a+h:a+2*h])*.5 for a,h in blocks],axis=1),np.concatenate([(F[:,a:a+h]-F[:,a+h:a+2*h])*.5 for a,h in blocks],axis=1),blocks,256
run=dict(uv=None,generation=('original',));seen=[]
def prod(left,right,uv,leg):seen.append((run['uv']is uv,run['generation']));return left@right
base=dict(fnp=np,_E2Hold=Hold,_uvp_split=split,_predict_binary=lambda *a,**k:0,_UVP_CUR=[10],_UBC_RUN=[run],_ubc_product=prod,_ubx_product=prod,_BH_HOLD=[])
env=dict(base);exec((D/'fragment_uc32_HELD_v1.py').read_text(),env);env['_UC32_ACTIVE']=True
old=dict(base);exec((R/'astra_common_u_compact8_v1/fragment_uc8_HELD_v1.py').read_text(),old);old['_UC8_ACTIVE']=True
cases=[]
for rank in (0,4,96):
 P=129;F=rng.normal(size=(384,2*P)).astype('f')
 if rank==0:F[:]=0
 elif rank==4:F[:]=rng.normal(size=(384,4)).astype('f')@rng.normal(size=(4,2*P)).astype('f')
 original=F.tobytes();uv=env['_uvp_split'](F,((0,P),));run['uv']=uv
 _,cuv,keep,joint,V=env['_UC32_CACHE'][id(uv)];assert len(keep)==224 and len(joint)==96 and set(keep)|set(joint)==set(range(320))
 H=np.zeros((384,256));H[keep,np.arange(224)]=1;H[np.ix_(joint,np.arange(224,256))]=V
 full=(F[:,:P]+F[:,P:])*.5;full[320:]=0
 assert np.allclose(cuv[0],H.T@full,rtol=2e-5,atol=3e-5)
 U=rng.normal(size=(17,384)).astype('f');S=rng.normal(size=(13,P)).astype('f');initial=rng.normal(size=(13,384)).astype('f')
 got=env['_uc32_recon'](U,uv);assert seen[-1]==(True,(('original',),'uc8-joint96-rank32'))
 inner=env['_uc32_inner'](initial.copy(),S,uv);assert seen[-1]==(True,(('original',),'uc8-joint96-rank32'))
 assert np.allclose(got,(U@H)@cuv[0],rtol=3e-5,atol=1e-4)
 assert np.allclose(inner,initial+(S@cuv[0].T)@H.T,rtol=3e-5,atol=1e-4)
 assert np.array_equal(inner[:,320:],initial[:,320:])and original==F.tobytes()
 assert run['uv']is uv and run['generation']==('original',)
 uv8=old['_uvp_split'](F,((0,P),));_,c8,k8,j8,V8=old['_UC8_CACHE'][id(uv8)];H8=np.zeros((384,256));H8[k8,np.arange(248)]=1;H8[np.ix_(j8,np.arange(248,256))]=V8
 e32=np.linalg.norm(full-H@cuv[0])**2;e8=np.linalg.norm(full-H8@c8[0])**2
 assert e32<=e8+1e-5*max(np.linalg.norm(full)**2,1)
 env['_UVP_CUR'][0]=15;assert np.array_equal(env['_uc32_recon'](U,uv),U[:,:256]@uv[0]);env['_UVP_CUR'][0]=10
 cases.append(dict(rank=rank,uc32_factor_error=float(e32),uc8_factor_error=float(e8),adjoint_coherent=True,Fv_full_and_tail_unchanged=True,L15_original_exact=True,input_immutable=True,private_generation_restored=True))
assert all(x[0] and x[1] in ((('original',),'uc8-joint96-rank32'),('original',)) for x in seen)
# Private generation must restore even when the routed product raises.
uv=env['_uvp_split'](F,((0,P),));run['uv']=uv
saved=env['_ubc_product'];env['_ubc_product']=lambda *a:(_ for _ in()).throw(ValueError('fixture'))
try:env['_uc32_recon'](U,uv)
except ValueError:pass
assert run['uv']is uv and run['generation']==('original',);env['_ubc_product']=saved
# Import-only and exact reverse of all three source substitutions.
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
p=D/'candidate_uc32_HELD_v1.py';mod=types.ModuleType('uc32_import');mod.__file__=str(p);sys.modules[mod.__name__]=mod
with fs.BudgetContext(flop_budget=2**60,quiet=True):exec(compile(p.read_bytes(),str(p),'exec'),mod.__dict__)
assert all(mod.__dict__.get(k)is v for k,v in mod._ASTRA_DENSE_CONTEXT)
s=p.read_text().split('\n\n# Fixed common256 budget:')[0].replace('    Pu = _uc32_recon(U, uv)\n',"    Pu = _ubc_product(U[:, :ru], Fu, uv, 'u')\n")
s=s.replace('    inner = _uc32_inner(inner, Su, uv)\n    del Su, Dv\n',"    inner_u = _ubx_product(Su, Fu.T, uv, 'ut')\n    _BH_HOLD.append(fnp.add(inner[:, :ru], inner_u, out=inner[:, :ru]))\n    _BH_HOLD.clear()\n    del Su, Dv, inner_u\n")
assert s==(R/'astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py').read_text()
# Actual tariff at all three archived fixed2 frame sizes and all used row shapes.
checks=json.loads((R/'astra_common_u_compact8_v1/run_uc8_diagnostic_v1/DIAGNOSTIC_RETIRED_v1.json').read_text())['checks'];prices={};maps={}
for k in (8,32):
 name='uc8'if k==8 else'uc32';file=R/'astra_common_u_compact8_v1/fragment_uc8_HELD_v1.py'if k==8 else D/'fragment_uc32_HELD_v1.py'
 ne=dict(fnp=fnp,_E2Hold=Hold,_uvp_split=lambda F,b:uv,_predict_binary=lambda *a:None,_UVP_CUR=[10]);exec(file.read_text(),ne);ne['_'+name.upper()+'_ACTIVE']=True
 for r,P in ((320,5856),(384,2400),(384,2432)):
  with fs.BudgetContext(flop_budget=2**40,quiet=True):F=fnp.zeros((r,2*P),dtype=fnp.float32);uv=(fnp.zeros((256,P),dtype=fnp.float32),fnp.zeros((r,P),dtype=fnp.float32),[(0,P)],256)
  with fs.BudgetContext(flop_budget=2**40,quiet=True)as b:ne['_uvp_split'](F,[(0,P)])
  prices[k,r,P]=dict(flops=int(b.flops_used),ops=len(b._op_log))
 for m in sorted({y[1]for x in checks for kind in('reconstruction','read')for y in x['uc8_stats'][kind]}|{1024}):
  with fs.BudgetContext(flop_budget=2**40,quiet=True):U=fnp.zeros((m,384),dtype=fnp.float32);keep=fnp.arange(256-k);joint=fnp.arange(256-k,320);V=fnp.zeros((64+k,k),dtype=fnp.float32);inner=fnp.zeros((m,384),dtype=fnp.float32);iu=fnp.zeros((m,256),dtype=fnp.float32)
  with fs.BudgetContext(flop_budget=2**40,quiet=True)as b:left=fnp.concatenate([U[:,keep],U[:,joint]@V],axis=1)
  rec=dict(flops=int(b.flops_used),ops=len(b._op_log))
  with fs.BudgetContext(flop_budget=2**40,quiet=True)as b:
   row=fnp.arange(m)*384;ki=row[:,None]+keep;ji=row[:,None]+joint;fnp.put(inner,ki,inner[:,keep]+iu[:,:256-k]);fnp.put(inner,ji,inner[:,joint]+iu[:,256-k:]@V.T)
  maps[k,m]=dict(recon=rec,read=dict(flops=int(b.flops_used)-m*256,ops=len(b._op_log)-1))
forecast=[]
for x in checks:
 delta=sum(prices[32,y[1],y[2]]['flops']-prices[8,y[1],y[2]]['flops']for y in x['uc8_stats']['builds'])
 delta+=sum(maps[32,y[1]][kind]['flops']-maps[8,y[1]][kind]['flops']for kind,key in [('recon','reconstruction'),('read','read')]for y in x['uc8_stats'][key])
 extra=x['arm_bill']-x['parent_bill']+delta
 forecast.append(dict(seed=x['seed'],whole_extra=extra,pct=100*extra/x['parent_bill'],includes_measured_uc8_cache_and_L15_debt=True,unchanged_routes_assumed=True))
result=dict(status='PASS_SOURCE_TINY_TARIFF_NO_MODELS',cases=cases,reverse_parent_byte_exact=True,original_dense_context_exact=True,exception_cache_restored=True,frame_tariffs={str(k):v for k,v in prices.items()},maps1024={str(k):v for k,v in maps.items()if k[1]==1024},whole_forecast=forecast,new_request_count_vs_uc8=0,new_setup_keys=0,main_common_width=256,difference_full_preserved=True,model_calls=0)
with(D/'TINY_AND_COST_v2.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(status=result['status'],whole=forecast,cases=cases)))
