from pathlib import Path
import json,hashlib,sys,types
import numpy as np
import flopscope as fs
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent;M=D.parent/'astra_native_terminal_mlp16_v1';C=D.parent/'astra_native_terminal_linear_refit_v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
with np.load(M/'all32_map_v1.npz',allow_pickle=False)as z:a={k:z[k]for k in z.files}
with np.load(D/'all32_context_map_v1.npz',allow_pickle=False)as z:weight=z['W']
records=json.loads((C/'ROOT_CAPTURE_RETIREMENT_v1.json').read_text())['evaluations']
answer=[None];calls=[]
def parent(*args):calls.append(args);return answer[0]
class Estimator:
 def predict(self,*args,**kwargs):return 'unchanged'
feature_answer=object()
ns=dict(fnp=fnp,gnn_step=parent,Estimator=Estimator,final_scalar_features=lambda *a:feature_answer)
with fs.BudgetContext(flop_budget=2**60,quiet=True)as setup:
 exec((M/'fragment_terminal_mlp_HELD_v1.py').read_text(),ns)
 exec((D/'fragment_terminal_context_HELD_v1.py').read_text(),ns)
checks=[];tariffs=[]
for seed in (0,16,31):
 e=next(e for e in records if e['seed']==seed);assert sha(e['capture_path'])==e['capture_sha256']
 with np.load(e['capture_path'],allow_pickle=False)as z:x=z['features30'];state=z['state64'];sig=z['sigma'];delta=z['delta'];host=z['host'];raw=z['raw_final_float64']
 X=np.column_stack((x,state)).astype('d');H=((X-a['center'])/a['scale'])@a['w0'].T+a['b0'];Y=(H/(1+np.exp(-H)))@a['w1'].T+a['b1'];mlp=(Y.reshape(-1)*float(a['target_rms']))*.5
 xd=x.astype('d');ctx=np.r_[1.,xd.mean(0),xd.std(0)];local=np.column_stack((np.ones(1024),xd[:,5],xd[:,6],xd[:,3],raw/sig,(xd[:,3]**2-1)*xd[:,6]));context=local@(ctx@weight)
 want=delta.astype('d')+sig*mlp+sig*context
 with fs.BudgetContext(flop_budget=2**60,quiet=True):xf=fnp.asarray(x);sf=fnp.asarray(state);vf=fnp.asarray(sig);df=fnp.asarray(delta);hf=fnp.asarray(host)
 originals=[np.asarray(v).tobytes()for v in(xf,sf,vf,df,hf)];answer[0]=(df,sf);calls.clear()
 assert ns['final_scalar_features'](15,hf,None,None)is feature_answer
 with fs.BudgetContext(flop_budget=2**60,quiet=True)as b:got=ns['gnn_step'](15,xf,vf,None,None,None)
 have=np.asarray(got[0]);assert len(calls)==1 and got[1]is sf and ns['_TCX_HOST']==[None]
 assert originals==[np.asarray(v).tobytes()for v in(xf,sf,vf,df,hf)]
 assert np.allclose(have,want,rtol=1e-11,atol=1e-12)
 native=np.maximum(host+have,0).astype('f');expected=np.maximum((host+(delta.astype('d')+sig*mlp))+sig*context,0).astype('f');assert native.tobytes()==expected.tobytes()
 checks.append(dict(seed=seed,parent_once=True,state_identity=True,input_unchanged=True,final_f32_exact=True,max_delta_error=float(np.max(abs(have-want)))))
 tariffs.append(dict(seed=seed,total_extra_flops=int(b.flops_used),charged_ops=len(b._op_log)))
 for layer in range(15):
  calls.clear()
  with fs.BudgetContext(flop_budget=2**60,quiet=True)as b:r=ns['gnn_step'](layer,xf,vf,None,None,None)
  assert r is answer[0] and len(calls)==1 and b.flops_used==0 and not b._op_log
 ns['_TCX_ON']=False;calls.clear()
 with fs.BudgetContext(flop_budget=2**60,quiet=True):off=ns['gnn_step'](15,xf,vf,None,None,None)
 assert np.asarray(off[0]).tobytes()==(delta.astype('d')+sig*mlp).tobytes()and off[1]is sf and len(calls)==1
 ns['_TCX_ON']=True
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
p=D/'candidate_QR_MLP_context_ON_HELD_v1.py';m=types.ModuleType('_tcx_full_import');m.__file__=str(p);sys.modules[m.__name__]=m
with fs.BudgetContext(flop_budget=2**60,quiet=True):exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__)
assert all(m.__dict__.get(k)is v for k,v in m._ASTRA_DENSE_CONTEXT)
assert m._RC_PARENT_READ is m._rtt_read and m._RTP_ON and m._RTT_CONTEXT==[None]and not m._RTT_ACTIVE and m._TCX_HOST==[None]
r=dict(status='PASS_NATIVE_EXPORT_TINY',candidate_sha256=sha(p),fragment_sha256=sha(D/'fragment_terminal_context_HELD_v1.py'),map_sha256=sha(D/'all32_context_map_v1.npz'),checks=checks,tariffs=tariffs,full_import_only=True,no_estimator_calls=True,first15_exact_delegation=True,coefficient_setup_flops=int(setup.flops_used),coefficient_setup_ops=len(setup._op_log))
with(D/'NATIVE_EXPORT_TINY_TARIFF_v1.json').open('x')as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r))
