"""Full held fragment exercised with current source-extracted inverse helpers.
No candidate module, estimator, card, model or truth import/evaluation.
"""
from pathlib import Path
import ast,hashlib,json,types
import numpy as np
import flopscope as flops
fnp=flops.numpy
D=Path(__file__).resolve().parent;O=D.parents[2];P=O/'engineer/a62_modules/s49/s56ubq1tw_x/candidate.py';s=P.read_text();t=ast.parse(s)
ns=dict(fnp=fnp,_BH_HOLD=[],_RQS_OUTV={},_RQS_STATS={'outv':0},_L3O_STATS={'pairs':0},_LVF_STATS={'fused_keys':0},_LVF_C=(slice(5,1,-3),slice(5,None,-5),slice(3,0,-2),slice(2,4)))
def put(n):exec(compile(ast.Module(body=[n],type_ignores=[]),str(P),'exec'),ns)
for name in ['_rs_nat_perm','_rs_level1','_rs_out_quadrants','_lvf_front','_lvf_at','_l3o_pairs']:
 put([n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name][-1])
fs=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='_l3o_last'];put(fs[0]);ns['_RQS_L3O_PARENT']=ns['_l3o_last'];put(fs[-1])
put([n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='_lvf_ops'][0])
put([n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='_ks_unwind' and n.lineno==11951][0]);baseline=ns['_ks_unwind'];ns['_ASTRA_RLF_PARENT_UNWIND']=baseline;ns['_ASTRA_RLF_CONTEXT']=[None]
class Estimator:
 def predict(self,*a):return 'parent'
ns.update(Estimator=Estimator,_PB_CACHES=(),_HZD_ON=True,_HZ7_PK=2048)
for name in ['_hzg_product','_hz_ks_product','_ubx_ks_product','_hz7_product']:ns[name]=lambda *a,**k:None
F=D/'fragment_inverse_output_prune_UNVALIDATED_v1.py';exec(compile(F.read_text(),str(F),'exec'),ns)
def setup(L,r,c,fused):
 T=fnp.empty((7,)*L+(r,c),dtype=fnp.float32);Z=[None,None];levels={}
 for j in range(2,L+1):
  nr=L-j;V=T[(slice(0,4),)*nr] if nr else T;Z.append(fnp.unstack(V,axis=nr));levels[j]=(V,nr)
 Z[1],slot=ns['_rs_level1'](2,L,T,(r,c))[0];ops=ns['_lvf_ops'](2,L,levels,Z,None) if fused else None
 basis=[]
 for j in range(2,L+1):
  ax=L-j;idx=[slice(0,4)]*L;idx[-1]=slice(None);lo=list(idx);lo[ax]=1;hi=list(idx);hi[ax]=3;basis.append((T[tuple(lo)],T[tuple(hi)]))
 out=fnp.empty(((1<<L)*r,(1<<L)*c),dtype=fnp.float32)
 return T,Z,slot,ops,basis,out
rng=np.random.default_rng(70219);rows=[]
for L,r,c in [(3,2,3),(4,3,2),(6,2,1),(7,1,1)]:
 for hr,hc in [(True,False),(False,True),(True,True)]:
  for fused in [False,True]:
   a=setup(L,r,c,fused);b=setup(L,r,c,fused);ns['_ASTRA_IOP_CACHE'].clear()
   for mode in ['OFF','ON','ON_dirty']:
    raw=rng.normal(size=(7,)*L+(r,c)).astype(np.float32)
    raw.flat[::19]=np.float32(-0.0);raw.flat[::23]*=np.float32(1e-39)
    fnp.copyto(a[0],fnp.asarray(raw));fnp.copyto(b[0],fnp.asarray(raw));fnp.copyto(a[5],np.float32(np.nan));fnp.copyto(b[5],np.float32(np.nan))
    ns['_ipt_side']=lambda *unused:(a[0],a[1],a[2],a[3],a[4])
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:baseline(a[0],a[1],a[3],a[4],a[5],L,(1<<L)*r,(1<<L)*c)
    oldbill=int(ctx.summary_dict()['flops_used']);oldops=len(ctx._op_log);ns['_BH_HOLD'].clear()
    ns['_ipt_side']=lambda *unused:(b[0],b[1],b[2],b[3],b[4]);ns['_ASTRA_IOP_ON']=mode!='OFF';ns['_ASTRA_IOP_CTX'][0]=dict(L=L,M=(1<<L)*r,Ns=((1<<L)*c,),hr=hr,hc=hc,family='ks')
    saved0=ns['_ASTRA_IOP_STATS']['saved_elements']
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:ns['_astra_iop_unwind'](b[0],b[1],b[3],b[4],b[5],L,(1<<L)*r,(1<<L)*c)
    bill=int(ctx.summary_dict()['flops_used']);ops=len(ctx._op_log);ns['_BH_HOLD'].clear();credit=ns['_ASTRA_IOP_STATS']['saved_elements']-saved0
    keep=np.ones((1<<L,r,1<<L,c),dtype=bool)
    if hr:keep[1::2,-1,:,:]=False
    if hc:keep[:,:,1::2,-1]=False
    x=np.asarray(a[5]).reshape(keep.shape);y=np.asarray(b[5]).reshape(keep.shape)
    assert x[keep].tobytes()==y[keep].tobytes(),(L,r,c,hr,hc,fused,mode,'bytes')
    if mode=='OFF':assert np.asarray(a[5]).tobytes()==np.asarray(b[5]).tobytes() and bill==oldbill and ops==oldops
    else:assert oldbill-bill==credit and credit>0,(L,r,c,hr,hc,fused,mode,'counter',oldbill-bill,credit)
    rows.append(dict(L=L,r=r,c=c,hr=hr,hc=hc,fused=fused,mode=mode,retained_bytes_exact=True,saved=credit,ops_delta=ops-oldops))
ns['_ASTRA_IOP_CTX'][0]=None;ns['_ASTRA_IOP_CACHE'].clear();ns['_ASTRA_IOP_ON']=True
assert Estimator().predict(None,None)=='parent' and ns['_ASTRA_IOP_CTX'][0] is None and not ns['_ASTRA_IOP_CACHE']
r=dict(status='PASS isolated full-fragment primitive, dirty scratch and OFF/ON transitions',candidate_module_imported=False,model_evaluations=0,parent_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),fragment_sha256=hashlib.sha256(F.read_bytes()).hexdigest(),rows=rows,cases=len(rows),cache_owner_reuse_tested=True,output_padding_poisoned=True,current_l3o_last_chain=True,retired_context_and_cache=True)
p=D/'TINY_FRAGMENT_DIRTY_RESULT_v1.json'
with p.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'cases':len(rows)}))
