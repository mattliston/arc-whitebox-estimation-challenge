from pathlib import Path
import json,hashlib,ast,sys,types,numpy as np
import flopscope as fs
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent;b=json.loads((D/'SOURCE_BUILD_v1.json').read_text());F=Path(b['fragment']['path']);C=Path(b['candidate']['path']);P=Path(b['parent']['path']);tree=ast.parse(F.read_text());defs=[n for n in tree.body if isinstance(n,ast.FunctionDef)];rng=np.random.default_rng(20261015);cases=[]
for xp in (np,fnp):
 with fs.BudgetContext(flop_budget=2**42,quiet=True):
  A=xp.asarray(rng.normal(size=(7,11)).astype(np.float32));B=xp.asarray(rng.normal(size=(9,11)).astype(np.float32));lam=xp.asarray(rng.normal(size=11).astype(np.float32));ids=xp.asarray(np.arange(9,dtype=np.int64));state=dict(columns=0)
  original=[np.asarray(x).tobytes()for x in (A,B,lam,ids)];calls=[];marker=object()
  def full(a,r,w,i,s):calls.append('full');return(a*w)@r.T
  def parent(*args):calls.append('parent');return marker
  ns=dict(_ETR_ON=True,_ETR_ACTIVE=True,_ETR_LAYERS=(2,3),_RTT_ACTIVE=True,_RTP_ON=True,_RS_LAYER=[None],_RTT_CONTEXT=[None],_ETR_PARENT_READ=parent,_RTT_PARENT_READ=full,_RTT_STATS=dict(reads=[]),_ETR_STATS=dict(restored=[],shapes=[],retired=False),_RC_STATE=[marker]);exec(compile(ast.Module(body=defs,type_ignores=[]),'fragment','exec'),ns)
  for l in (1,2,3,4,10):
   ns['_RS_LAYER'][0]=l;calls.clear();out=ns['_rtt_read'](A,B,lam,ids,state)
   if l in (2,3):
    assert calls==['full'];ref=(A*lam)@B.T;assert np.asarray(out).tobytes()==np.asarray(ref).tobytes()
   else:assert calls==['parent']and out is marker
   assert ns['_RC_STATE'][0]is marker and ns['_RTT_CONTEXT'][0]is None
   cases.append(dict(backend=xp.__name__,layer=l,exact_delegate=True))
  assert ns['_ETR_STATS']['restored']==[2,3]
  assert all(r['c15']==r['c2']==9 and r['changed']is False for r in ns['_RTT_STATS']['reads'])
  ns['_ETR_ON']=False;ns['_RS_LAYER'][0]=2;assert ns['_rtt_read'](A,B,lam,ids,state)is marker
  assert original==[np.asarray(x).tobytes()for x in (A,B,lam,ids)]
  for throws in (False,True):
   ns['_ETR_ACTIVE']=False;ns['_ETR_ON']=True
   def pred(self):
    assert ns['_ETR_ACTIVE'];
    if throws:raise ValueError('synthetic failure')
    return marker
   ns['_ETR_PARENT_PREDICT']=pred
   try:out=ns['_etr_predict'](None);assert not throws and out is marker
   except ValueError:assert throws
   assert ns['_ETR_ACTIVE']is False and ns['_ETR_STATS']['retired']is True
# Full import, no estimator setup/predict. Exact parent prefix preserves captures.
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
m=types.ModuleType('early_tail_check');m.__file__=str(C);sys.modules[m.__name__]=m
with fs.BudgetContext(flop_budget=2**42,quiet=True):exec(compile(C.read_text(),str(C),'exec'),m.__dict__)
assert m._RC_PARENT_READ is m._rtt_read and m._RTT_PARENT_READ is m._yr7_read
assert all(m.__dict__[n]is v for n,v in m._ASTRA_DENSE_CONTEXT)
assert m._ETR_LAYERS==(2,3)and not m._ETR_ACTIVE and m._ETR_STATS['retired']
assert m._RC_COLZ==2.0 and len(m._PB_SINGLE)==649 and m._PB_WALL_S==3.97
assert C.read_text()==P.read_text()+'\n\n'+F.read_text()
source=P.read_text();assert "_RS_LAYER[0] = l\n    return _RS_PARENT_LEFT(l,"in source
cost=json.loads((D.parent/'astra_rtt2_layer_restore_audit_v1/TRAIN16_ERROR_ONSET_AND_TARIFF_v1.json').read_text());selected=[r for r in cost['per_layer_warm_tariff']if r['layer']in(2,3)]
r=dict(status='SOURCE_TINY_IMPORT_OFF_CLEANUP_PASS',candidate_sha256=b['candidate']['sha256'],cases=cases,OFF_exact_parent_delegate=True,inputs_unchanged=True,outer_c2_state_unchanged=True,full_read_called_once=True,exception_cleanup_pass=True,all_captured_dense_identities_unchanged=True,layer_index='Original zero-based l passed through _np_left into _RS_LAYER; restore exactly l=2,3, first2 outputs0/1 unchanged.',effective_RTT_record='L2/3 c15=c2, changed=false, basis/refine fieldsNone; old_rank0/innercalls0. Remaining layers originalrecord.',selected_actual_FNP_warm_tariff=selected,extra_whole_bill_percent_upper=sum(x['restore_extra_whole_bill_percent_upper']for x in selected),request_effect='Natural existing fullshape D7 routes retained; no added/disabled whitelist. Actual RPC change measured by pilot, no proxy guarantee.',models=0)
p=D/'TINY_SOURCE_AND_TARIFF_v1.json';p.open('x').write(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items()if k not in('cases','selected_actual_FNP_warm_tariff')}));print(hashlib.sha256(p.read_bytes()).hexdigest())
