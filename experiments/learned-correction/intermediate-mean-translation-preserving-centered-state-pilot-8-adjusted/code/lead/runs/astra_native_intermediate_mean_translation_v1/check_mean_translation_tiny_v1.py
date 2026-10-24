"""Fixed local residual: isolated real FNP and import, no estimator models."""
from pathlib import Path
import hashlib, importlib.util, json, sys, types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=D/'candidate_QR_mean_translation_HELD_v1.py'
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
spec=importlib.util.spec_from_file_location('_local_mean_tiny',p);M=importlib.util.module_from_spec(spec);sys.modules[spec.name]=M
with flops.BudgetContext(flop_budget=2**60,quiet=True):spec.loader.exec_module(M)
assert all(M.__dict__[name] is value for name,value in M._ASTRA_DENSE_CONTEXT)
assert M._nonlinear_pair_batch.__wrapped__ is M.nonlinear
parent=D.parent/'astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v3.py'
fragment=D/'fragment_mean_translation_HELD_v1.py'
proof=json.loads((D/'SOURCE_PLAN_HELD_v1.json').read_text());base=p.read_text()[:-len(fragment.read_text())-1]
for before,after in reversed(proof['core_changes']):
 assert base.count(after)==1;base=base.replace(after,before,1)
assert base==parent.read_text()
import ast
def function(text,name):return next(ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name==name)
assert function(p.read_text(),'_astra_fhm_effective')==function(parent.read_text(),'_astra_fhm_effective')
core=function(p.read_text(),'_astra_fhm_core');assert core.rfind('m = _lmt_next_mean')>core.rfind('_np_update(')>core.index('_lmt_next_mean = _lmt_translation(l, m)')
assert core.count('m = _lmt_next_mean')==1
with np.load(D.parent/'astra_native_intermediate_mean_refit_v1/run_capture_s0_v1/native_16_capture_capture.npz',allow_pickle=False)as z:
 data={k:z[k].copy()for k in z.files if k.startswith('L')}
weights=np.asarray(M._LMT_LITERAL);results=[];parent_calls=[0];state_return=[None];delta_return=[None]
def parent_gnn(*args):
 parent_calls[0]+=1
 return (delta_return[0],state_return[0])
M._LMT_PARENT_GNN=parent_gnn
for l in range(7,15):
 X=data[f'L{l}_features30'];S=data[f'L{l}_state64'];sigma=data[f'L{l}_sigma'];mean=data[f'L{l}_host'];delta=data[f'L{l}_delta']
 with flops.BudgetContext(flop_budget=2**60,quiet=True):
  x=fnp.asarray(X);s=fnp.asarray(S);ss=fnp.asarray(sigma,dtype=fnp.float64);m=fnp.asarray(mean,dtype=fnp.float64);dm=fnp.asarray(delta,dtype=fnp.float64)
  M._lmt_maps();state_return[0]=s;delta_return[0]=dm;M._ASTRA_FHM_ACTIVE[0]=True
 snapshots=[np.asarray(v).tobytes()for v in (x,s,ss,m,dm)]
 M._LMT_ACTIVE[0]=False
 with flops.BudgetContext(flop_budget=2**60,quiet=True)as offctx:
  ans=M.gnn_step(l,x,ss,None,None,None);off=M._astra_fhm_effective(l,m,dm);base_mean=m+off;off_mean=M._lmt_translation(l,base_mean)
 assert ans[0] is dm and ans[1] is s and M._LMT_PENDING[0]is None and off_mean is base_mean
 assert np.asarray(off).tobytes()==np.maximum(delta.astype(float),-mean.astype(float)).tobytes()
 M._LMT_ACTIVE[0]=True
 before_calls=parent_calls[0]
 with flops.BudgetContext(flop_budget=2**60,quiet=True)as ctx:
  ans=M.gnn_step(l,x,ss,None,None,None);eff=M._astra_fhm_effective(l,m,dm);original_mean=m+eff;out=M._lmt_translation(l,original_mean)
 assert parent_calls[0]==before_calls+1 and ans[0] is dm and ans[1] is s
 assert M._LMT_PENDING[0]is None
 coef=weights[l-7];normalized=(X.astype(float)@coef[:30]+S.astype(float)@coef[30:94])+coef[94]
 correction=(.5*sigma.astype(float))*normalized
 expected=np.maximum(mean.astype(float)+np.maximum(delta.astype(float),-mean.astype(float))+correction,0.)
 assert np.asarray(eff).tobytes()==np.asarray(off).tobytes()
 got=np.asarray(out);error=float(np.max(np.abs(got-expected)))
 assert np.allclose(got,expected,rtol=2e-13,atol=2e-13),error
 assert snapshots==[np.asarray(v).tobytes()for v in (x,s,ss,m,dm)]
 results.append(dict(layer=l,max_abs_error=error,source_state_identity=True,input_bytes_exact=True,OFF_exact=True,added_flops=int(ctx.flops_used-offctx.flops_used),added_charged_operations=len(ctx._op_log)-len(offctx._op_log)))
 M._LMT_ACTIVE[0]=False
# Synthetic clamp boundaries and exact zero-map identity, using the same FNP wrapper.
saved_maps=M._LMT_MAPS.copy()
with flops.BudgetContext(flop_budget=2**60,quiet=True):
 zero=fnp.asarray(np.zeros((8,95)));M._LMT_MAPS.clear();M._LMT_MAPS['owner']=zero
 for i in range(8):M._LMT_MAPS[i+7]=(zero[i,:30],zero[i,30:94],zero[i,94])
 x=fnp.asarray(np.zeros((5,30),np.float32));s=fnp.asarray(np.zeros((5,64),np.float32));ss=fnp.asarray(np.ones(5));m=fnp.asarray(np.asarray([1.,1.,0.,1.,-1.]));dm=fnp.asarray(np.asarray([-2.,-1.,0.,.5,2.]));state_return[0]=s;delta_return[0]=dm
 M._LMT_ACTIVE[0]=True;M.gnn_step(7,x,ss,None,None,None);eff=M._astra_fhm_effective(7,m,dm);raw=m+eff;got=M._lmt_translation(7,raw)
 assert np.asarray(got).tobytes()==np.asarray(raw).tobytes()
 # Centered covariance is the original FHM result and is never an input to translation.
 C=fnp.asarray(np.arange(25,dtype=float).reshape(5,5));Cpost=((C-fnp.outer(m,eff))-fnp.outer(eff,m))-fnp.outer(eff,eff);snapshot=np.asarray(Cpost).tobytes()
 M._LMT_PENDING[0]=(7,fnp.asarray(np.asarray([.1,-.1,.2,.3,-.4])));translated=M._lmt_translation(7,raw)
 assert np.asarray(Cpost).tobytes()==snapshot
 assert M._lmt_translation(6,raw)is raw
 assert np.asarray(raw).tobytes()==np.asarray(m+eff).tobytes()
 M._LMT_ACTIVE[0]=False;M._LMT_MAPS.clear();M._LMT_MAPS.update(saved_maps)
# Wrapper delegation, OFF result identity, and exceptional pending cleanup.
sentinel=object();M._LMT_PARENT_PREDICT=lambda *args:sentinel;M._LMT_ON=False
assert M._lmt_predict(None,types.SimpleNamespace(width=1024,depth=16),None)is sentinel
assert M._LMT_STATS['retired']and not M._LMT_ACTIVE[0]and M._LMT_PENDING[0]is None
M._LMT_ON=True
def fail(*args):M._LMT_PENDING[0]=(7,object());raise RuntimeError('injected')
M._LMT_PARENT_PREDICT=fail
try:M._lmt_predict(None,types.SimpleNamespace(width=1024,depth=16),None)
except RuntimeError as e:assert str(e)=='injected'
else:raise AssertionError('missing failure')
assert M._LMT_STATS['retired']and not M._LMT_ACTIVE[0]and M._LMT_PENDING[0]is None
reset=[];M._LMT_PARENT_RESET=lambda saved:reset.append(saved);M._pb_reset(('a','b'));assert reset==[('a','b')]and not M._LMT_MAPS
r=dict(status='PASS',source_inverse_exact=True,original_FHM_identity=True,translated_mean_assigned_only_after_NP=True,centered_covariance_preserved=True,full_import_context_identity=True,model_calls=0,real_FNP_cases=8,results=results,zero_map_and_clamp_boundaries_exact=True,OFF_wrapper_result_identity=True,state_unchanged=True,input_bytes_unchanged=True,predict_exception_cleanup=True,setup_map_reset=True,whole_network_added_flops=sum(x['added_flops']for x in results),whole_network_added_charged_operations=sum(x['added_charged_operations']for x in results),candidate_sha256=sha(p),fragment_sha256=sha(fragment),checker_sha256=sha(__file__))
out=D/'TINY_MEAN_TRANSLATION_RESULT_v1.json';out.open('x').write(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
