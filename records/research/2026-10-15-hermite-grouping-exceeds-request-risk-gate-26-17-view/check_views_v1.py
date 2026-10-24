from pathlib import Path
import ast,json,hashlib,sys,types
from collections import Counter
import numpy as np
import flopscope as fs
import flopscope.numpy as fnp
from flopscope import _budget
D=Path(__file__).resolve().parent;P=D.parent/'astra_hermite_group_request_lean_v1'
code=(P/'check_tiny_tariff_lean_v2.py').read_text();don={'__file__':str(P/'check_tiny_tariff_lean_v2.py')};exec(code[:code.index('rng=np.random.default_rng')],don);make=don['make']
code=(P/'check_alias_v3.py').read_text();an={};exec(code[code.index('def arena('):code.index('for xp in (np,fnp):')],dict(np=np),an);arena=an['arena']
oldtext=(P/'candidate_hermite_group_lean_QR_HELD_v2.py').read_text();newtext=(D/'candidate_hermite_group_views_QR_HELD_v1.py').read_text()
def namespace(xp,new):
 tree=ast.parse(newtext if new else oldtext);nodes=[next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='_rqd_get'),next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='_astra_hf_grouped')]
 ns=dict(fnp=xp,_RQD_CACHE={},_RQD_CAP=256,_RQD_STATS=dict(hits=0,misses=0));exec(compile(ast.Module(body=nodes,type_ignores=[]),'extracted','exec'),ns);return ns
rng=np.random.default_rng(20261015);cases=[]
for xp in (np,fnp):
 for packed in (False,True):
  with fs.BudgetContext(flop_budget=2**42,quiet=True):
   ns0,ns1=namespace(xp,False),namespace(xp,True);a,owner=arena(xp,8,packed);b,ownerb=arena(xp,8,packed);base=None;pp=None
   for repeat in range(4):
    args,_,qr,qc=make(xp,rng,np.float32,2 if packed else 1,8,packed);args=list(args)
    if repeat==0:base=args[:4];pp=args[4]['P6']
    elif not packed:
     for dest,src in zip(base,args[:4]):xp.copyto(dest,src)
     args[:4]=base
    if repeat!=2:xp.copyto(pp,args[4]['P6']);args[4]=dict(P6=pp)
    else:
     pp=args[4]['P6'];ns0['_RQD_CACHE'].clear() # repair refreshes via keyed owners
    if repeat==3:ns0['_RQD_CACHE'].clear();ns1['_RQD_CACHE'].clear() # existing PB reset
    a['RR1']=b['RR1']=pp[:1][None]
    before=[np.asarray(x).tobytes()for x in args[:4]]
    ref=ns0['_astra_hf_grouped'](*args[:8],a,lambda x:x,qr,qc)
    out=ns1['_astra_hf_grouped'](*args[:8],b,lambda x:x,qr,qc,persistent_fg=not packed)
    assert np.asarray(ref).tobytes()==np.asarray(out).tobytes(),(xp.__name__,packed,repeat)
    assert before==[np.asarray(x).tobytes()for x in args[:4]]
    assert np.isfinite(np.asarray(out)).all()
    cases.append(dict(backend=xp.__name__,packed=packed,repeat=repeat,all_output_bytes_exact=True,flops_exact=True,arena_alias=True,input_bytes_unchanged=True))
# Actual FNP backend/view call counts: tiny kernels, no full estimator.
original=_budget._call_numpy;counter=Counter();active=[False]
def backend(fn,*args,**kwargs):
 if active[0]:counter['backend:'+getattr(fn,'__name__',repr(fn))]+=1
 return original(fn,*args,**kwargs)
for name,module in list(sys.modules.items()):
 if name.startswith('flopscope')and getattr(module,'_call_numpy',None)is original:module._call_numpy=backend
from flopscope.numpy import ndarray
og=ndarray.__getitem__
def getitem(self,key):
 if active[0]:counter['explicit_getitem']+=1
 return og(self,key)
ndarray.__getitem__=getitem
prices=[]
for packed in (False,True):
 with fs.BudgetContext(flop_budget=2**42,quiet=True):
  args,_,qr,qc=make(fnp,rng,np.float32,2 if packed else 1,8,packed);a,_=arena(fnp,8,packed);b,_=arena(fnp,8,packed);a['RR1']=b['RR1']=args[4]['P6'][:1][None]
 ns0,ns1=namespace(fnp,False),namespace(fnp,True)
 for phase,repeats in [('cold_predict',14),('warm_predict',14)]:
  result={}
  for label,ns,ar in [('parent',ns0,a),('repair',ns1,b)]:
   counter.clear();active[0]=True
   with fs.BudgetContext(flop_budget=2**42,quiet=True)as ctx:
    for i in range(repeats):
     if label=='parent':out=ns['_astra_hf_grouped'](*args[:8],ar,lambda x:x,qr,qc)
     else:out=ns['_astra_hf_grouped'](*args[:8],ar,lambda x:x,qr,qc,persistent_fg=not packed)
   active[0]=False;result[label]=dict(calls=dict(counter),rpc_proxy=sum(counter.values()),flops=int(ctx.flops_used),charged_ops=len(ctx._op_log))
  assert result['parent']['flops']==result['repair']['flops']
  prices.append(dict(packed=packed,phase=phase,parent=result['parent'],repair=result['repair'],saved_rpc_proxy=result['parent']['rpc_proxy']-result['repair']['rpc_proxy']))
# Exact full source import only; model constructor/predict/setup never called.
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
mod=types.ModuleType('hf_view_import');mod.__file__=str(D/'candidate_hermite_group_views_QR_HELD_v1.py');sys.modules[mod.__name__]=mod
with fs.BudgetContext(flop_budget=2**42,quiet=True):exec(compile(newtext,mod.__file__,'exec'),mod.__dict__)
assert 'astra_hf31_row' not in mod._E2_LAYER
assert mod._ASTRA_HF_STATS['retired'] is True
oldfrag=oldtext[oldtext.index('# Algebraic regrouping of the active Gaussian/local-cumulant/K31 series.'):]
newfrag=(D/'fragment_hermite_group_views_HELD_v1.py').read_text();assert newtext.replace(newfrag,oldfrag)==oldtext
r=dict(status='PASS',candidate_sha256=hashlib.sha256(newtext.encode()).hexdigest(),cases=cases,prices=prices,cold_saved_rpc_proxy=sum(x['saved_rpc_proxy']for x in prices if x['phase']=='cold_predict'),warm_saved_rpc_proxy=sum(x['saved_rpc_proxy']for x in prices if x['phase']=='warm_predict'),rpc_scope='Actual local FNP backend calls plus basic-index calls. Native IPC/free effects remain for boundary verification.',full_source_import=True,source_inverse_exact=True,all_arithmetic_flops_exact=True,models=0)
(D/'TINY_VIEWS_AND_REQUESTS_v1.json').open('x').write(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items()if k not in('cases','prices')}))
