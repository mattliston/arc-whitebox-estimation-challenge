from pathlib import Path
import ast,hashlib,json,sys,types,os,time
import numpy as np
import flopscope as fs
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent;S=D/'candidate_jrs_HELD_v1.py';start=time.monotonic()
M=types.ModuleType('jrs_tiny');M.__file__=str(S);sys.modules[M.__name__]=M
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
with fs.BudgetContext(flop_budget=2**50,quiet=True):exec(compile(S.read_bytes(),str(S),'exec'),M.__dict__)
assert all(M.__dict__.get(k)is v for k,v in M._ASTRA_DENSE_CONTEXT)
assert M.nonlinear.__code__.co_firstlineno==844
assert M._OTT_ORIGINAL_PUSH is M._astra_bcc_push
assert '_astra_bcc_push_impl'in M._astra_bcc_push.__code__.co_names
tree=ast.parse(S.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_astra_bcc_push_impl')
assignment=next(n for n in ast.walk(node)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='_jn_rows'for t in n.targets))
expr=compile(ast.Expression(assignment.value),str(S),'eval')
rng=np.random.default_rng(20261015);cases=[]
for kind in ('normal','rank1_common','zero','nonfinite'):
 J=rng.normal(size=(12,16)).astype('f');w=np.abs(rng.normal(size=16)).astype('f');rank=4
 if kind=='rank1_common':J[:]=1;w[:]=1
 if kind=='zero':J[:]=0
 d=(J*J)@w
 if kind=='nonfinite':J[0,0]=np.nan
 order=np.argsort(-d.astype('d'),kind='stable');original=np.sort(order[:rank])
 with np.errstate(invalid='ignore'):
  yy=(J@(w*J.sum(0))).astype('d');s=yy.sum()
  score=d.astype('d')-yy*yy/s if np.isfinite(s)and s>0 else None
 expected_fallback=score is None or not np.all(np.isfinite(score))
 if expected_fallback:expected=original
 else:
  score[order[0]]=-np.inf;expected=np.sort(np.r_[order[0],np.argsort(-score,kind='stable')[:rank-1]])
 with fs.BudgetContext(flop_budget=2**50,quiet=True):
  jj,ww,dd=[fnp.asarray(x)for x in (J,w,d)]
 before=[np.asarray(x).tobytes()for x in (jj,ww,dd)]
 M._JRS_STATS.update(calls=0,records=[]);M._JRS_ACTIVE[0]=True
 with fs.BudgetContext(flop_budget=2**50,quiet=True)as budget:
  got=eval(expr,M.__dict__,dict(J=jj,weight=ww,_jn_d=dd,rank=rank,layer=7,old_columns=0,_e2_H=M._E2Hold()))
 assert np.array_equal(np.asarray(got),expected)
 assert M._JRS_STATS['calls']==1 and M._JRS_STATS['records'][0]['fallback']==expected_fallback
 assert before==[np.asarray(x).tobytes()for x in(jj,ww,dd)]
 cases.append(dict(kind=kind,rows=np.asarray(got).tolist(),fallback=expected_fallback,flops=int(budget.flops_used),ops=len(budget._op_log)))
# Caller OFF, non-L7, or already-old must execute exactly original selector.
for active,layer,old in ((False,7,0),(True,10,64),(True,7,64)):
 with fs.BudgetContext(flop_budget=2**50,quiet=True)as base:
  expect=fnp.sort(fnp.argsort(-dd.astype(fnp.float64),kind='stable')[:rank])
 M._JRS_ACTIVE[0]=active;beforecalls=M._JRS_STATS['calls']
 with fs.BudgetContext(flop_budget=2**50,quiet=True)as off:
  got=eval(expr,M.__dict__,dict(J=jj,weight=ww,_jn_d=dd,rank=rank,layer=layer,old_columns=old,_e2_H=M._E2Hold()))
 assert np.asarray(got).tobytes()==np.asarray(expect).tobytes() and off.flops_used==base.flops_used and len(off._op_log)==len(base._op_log)
 assert M._JRS_STATS['calls']==beforecalls
marker=object()
for throwing in (False,True):
 def fake(*a,**kw):
  assert M._JRS_ACTIVE==[True]
  if throwing:raise RuntimeError('synthetic')
  return marker
 M._JRS_PARENT_PREDICT=fake;M._JRS_ON=True
 try:answer=M._jrs_predict(None)
 except RuntimeError:assert throwing
 else:assert answer is marker and not throwing
 assert M._JRS_ACTIVE==[False]and M._JRS_STATS['retired']and M._JRS_STATS['calls']==0
r=dict(status='PASS',candidate_sha256=hashlib.sha256(S.read_bytes()).hexdigest(),all_context_identities_bound=True,active_BCC_call_chain=True,nonlinear_first_line_844=True,selector_cases=cases,inputs_immutable=True,OFF_later_old_rows_bytes_flops_ops_exact=True,success_exception_cleanup=True,no_setup_additions=True,no_model_calls=True,elapsed_seconds=time.monotonic()-start)
p=D/'TINY_INTEGRATION_v1.json'
with p.open('x')as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r));print(hashlib.sha256(p.read_bytes()).hexdigest())

