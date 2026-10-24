"""Public-FNP arithmetic and candidate-local lifecycle proof; no model."""
import ast,hashlib,json,resource,time
from pathlib import Path
import numpy as np
import flopscope as fs
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent;R=D.parents[1];t0=time.monotonic()
source=R/'astra_crunch_runtime_v1/failure336657_v1/candidate_setup_freeze_v22b.py'
tree=ast.parse(source.read_text());nodes=[]
for name in ('_rs_nat_perm','_lvf_at','_lvf_front','_lvf_ops'):
 nodes.append(next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name))
nodes.append([n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_lvf_rs_level1'and 11900<n.lineno<12000][0])
ns=dict(fnp=fnp,_LVF_STATS={'fused_keys':0},_LVF_A=(slice(2,7,4),slice(3,0,-2)),_LVF_B=(slice(5,2,-2),slice(2,6,3),slice(6,3,-2)),_LVF_C=(slice(5,1,-3),slice(5,None,-5),slice(3,0,-2),slice(2,4)))
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
rng=np.random.default_rng(28401);rows=[]
def make(kind,L,values,crop):
 T=fnp.array(values,dtype=fnp.float32)
 if crop:T=T[...,:2,:3]
 data=slice(0,7,2)if kind==0 else slice(0,4);levels={};chunks=[None,None]
 for j in range(2,L+1):
  nr=L-j;Tj=T[(data,)*nr]if nr else T;levels[j]=(Tj,nr);chunks.append(fnp.unstack(Tj,axis=nr))
 lv1,groups=ns['_lvf_rs_level1'](kind,L,T,tuple(T.shape[-2:]));chunks[1]=lv1[0][0]
 return T,levels,chunks,groups[0]
def plain(kind,L,X):
 if kind==0:
  for j in range(1,L+1):
   z=X[j];fnp.subtract(z[4],z[0],out=z[5]);fnp.subtract(z[2],z[5],out=z[3]);fnp.subtract(z[6],z[5],out=z[1])
 elif kind==1:
  for j in range(1,L+1):
   z=X[j];fnp.add(z[1],z[0],out=z[5]);fnp.subtract(z[5],z[2],out=z[6]);fnp.subtract(z[3],z[5],out=z[4])
 else:
  for j in range(L,1,-1):
   z=X[j];fnp.add(z[0],z[5],out=z[5]);fnp.add(z[0],z[2],out=z[0]);fnp.add(z[5],z[4],out=z[5]);fnp.add(z[5],z[3],out=z[2]);fnp.add(z[5],z[1],out=z[3]);fnp.add(z[4],z[6],out=z[1])
for L in (3,4,5,6):
 for kind in (0,1,2):
  for crop in (False,True):
   values=rng.standard_normal((7,)*L+((3,5)if crop else(2,3))).astype(np.float32)
   with fs.BudgetContext(flop_budget=2**60,quiet=True):
    a,la,xa,ga=make(kind,L,values,crop);b,lb,xb,gb=make(kind,L,values,crop)
    ops=ns['_lvf_ops'](kind,L,lb,xb,gb)
   with fs.BudgetContext(flop_budget=2**60,quiet=True)as ca:plain(kind,L,xa)
   with fs.BudgetContext(flop_budget=2**60,quiet=True)as cb:
    for f,x,y,o in ops:f(x,y,out=o)
   assert np.array_equal(np.asarray(a),np.asarray(b)),(kind,L,crop)
   assert ca.flops_used==cb.flops_used
   rows.append(dict(kind=kind,L=L,cropped=crop,byte_equal=True,flops=int(ca.flops_used),plain_ops=len(ca._op_log),fused_ops=len(cb._op_log)))
fragment=D/'fragment_read_off_fusion192_v1.py.txt';src=fragment.read_text();compile(src,str(fragment),'exec')
# Scope/exception restoration is independent of arithmetic and public arrays.
proof=[]
for fail in(False,True):
 class Cache(dict):
  def touch(self,key):pass
 tables=({'p':1},{'r':2},{'read':3,'small':4});bound=[];caches=(Cache(),Cache(),Cache())
 def inner(_t):
  assert env['_PEEL_PUSH_PLANS']=={}and env['_PEEL_READ_PLANS']=={}and env['_SMX_FAMILIES']=={'small':4}
  if fail:raise ValueError('injected')
 def side(kind,L,r,c):
  entry=(None,None,[None,object()],('fused',),None);caches[kind][L,r,c]=entry;bound.append((kind,L,r,c));return entry
 env=dict(_peel_take=lambda k,x:True,_DPS_PARENT_BIND_V1=inner,_PEEL_PUSH_PLANS=tables[0],_PEEL_READ_PLANS=tables[1],_SMX_FAMILIES=tables[2],_SM12_PLANS={'a':('rec','small')},_SCRATCH=[object()],_LB_SETUP=[False],_LVF_EAGER=[False],_PB_SINGLE=tuple((i%3,6,64*(i+1),64)for i in range(224)),_ipt_side=side,_IPT_A=caches[0],_IPT_B=caches[1],_IPT_C=caches[2])
 exec(src,env)
 try:env['_ASH_PARENT_BIND'](0)
 except ValueError:assert fail
 assert all(env[name]is table for name,table in zip(('_PEEL_PUSH_PLANS','_PEEL_READ_PLANS','_SMX_FAMILIES'),tables))
 assert env['_LB_SETUP']==[False]and env['_LVF_EAGER']==[False]
 if not fail:assert bound==list(env['_PB_SINGLE'][128:192])
 proof.append(dict(injected_exception=fail,dict_identity_restored=True,flags_restored=True,binds=len(bound)))
out=dict(status='PASS',models=0,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),fragment_sha256=hashlib.sha256(fragment.read_bytes()).hexdigest(),arithmetic_cases=rows,lifecycle_cases=proof,seconds=time.monotonic()-t0,maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,limitations=['Tiny validates existing grouped fusion on packed and cropped public FNP views; all-read fallback preserves exact algebra but changes floating recombination, so whole predictions are not promised byte-identical.','No measured whole-prediction runtime saving; runtime owns integrated qualification.'])
with(D/'TINY_FUSION192_v1.json').open('x')as f:json.dump(out,f,indent=2)
print(json.dumps({k:out[k]for k in('status','fragment_sha256','seconds','maxrss')}))
