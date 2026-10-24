import ast,hashlib,json,resource,types,sys,os,math
from pathlib import Path
assert os.getpriority(os.PRIO_PROCESS,0)==10
import numpy as np
O=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1');D=O/'lead/runs/astra_full384_smirnov_v1';S=D/'candidate_full384_smirnov_HELD_v2.py'; text=S.read_text();tree=ast.parse(text)
sys.path.insert(0,str(O/'lead/runs/astra_smirnov_current24_virtual_v1'))
from smirnov_batched_slots_v2 import Kernel as Parent
G={'math':math,'_rc_math':math};names=['_ASTRA_SM40_FAMILIES','_ASTRA_SM40_RECORDS','_ASTRA_SM40_PLANS'];nodes=[]
for n in tree.body:
 if isinstance(n,(ast.FunctionDef,ast.ClassDef))and(n.name.startswith('_astra_sm40_')and n.lineno<18902 or n.name=='_AstraSm40Kernel'):nodes.append(n)
 elif isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id in names for x in n.targets):nodes.append(n)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(S),'exec'),G)
actualfamilies=G['_ASTRA_SM40_FAMILIES'];plans=G['_ASTRA_SM40_PLANS'];records=G['_ASTRA_SM40_RECORDS'];capacity=[]
for name,L,P,r,k,c in actualfamilies:
 count=7**L*P*(r*k+k*c+(0 if P==20 else r*c));assert count<=391602176;capacity.append(dict(family=name,main_bytes=count*4))
for (m,k,n,rank),(name,L,P)in plans.items():
 u=1<<L;r=math.ceil(m/(3*u));kk=math.ceil((k+rank)/(6*u));c=math.ceil(n/(3*u));assert any(f[1]==L and r<=f[3]and kk<=f[4]and c<=f[5]for f in actualfamilies)
M=types.ModuleType('tiny');M.fnp=np;M._BH_HOLD=[];M._TR_LAST=[None];M._IPT_REGION={L:(2**29,)*3 for L in range(1,8)}
body=[next(n for n in reversed(tree.body)if isinstance(n,ast.FunctionDef)and n.name==name)for name in ('_rs_nat_perm','_rs_basis_views')]
exec(compile(ast.Module(body=body,type_ignores=[]),str(S),'exec'),M.__dict__)
# Keep depth4, P40 and exact new81 inner; narrow output dimensions bound RAM.
G['_ASTRA_SM40_FAMILIES']=(('toy4_81',4,40,1,81,1),('toy5_pad',5,20,5,1,5))
size=max(7**L*P*(r*k+k*c+(0 if P==20 else r*c))for _,L,P,r,k,c in G['_ASTRA_SM40_FAMILIES'])
size=max(size,125*160*1296);owner=np.empty((size,),dtype='float32');M._SCRATCH=[owner];G['_astra_sm40_bind_universal'](M,owner)
raw={r['name']:r for r in json.loads((O.parent/'astra_phase2_reexamination_v1/smirnov_executable_transpose_certificate_v1.json').read_text())['records']};rng=np.random.default_rng(9384);rows=[]
for name,rec in records.items():
 p=Parent(M,raw[name]);q=G['_AstraSm40Kernel'](M,rec)
 for m,k,n,rank in [(47,7385,46,384),(48,7392,48,384),(47,7385,46,320),(48,7456,48,320)]:
  a=rng.standard_normal((m,k),dtype=np.float32);b=rng.standard_normal((k,n),dtype=np.float32);old=(rng.standard_normal((m,rank),dtype=np.float32),rng.standard_normal((rank,n),dtype=np.float32));before=(a.tobytes(),b.tobytes())
  expected=p(a,b,4,8,old);owner.fill(np.nan);M._sm40_extra_leaf.fill(np.nan);got=q(a,b,4,8,old);assert got.tobytes()==expected.tobytes();assert before==(a.tobytes(),b.tobytes())
  a2=a*.7;b2=b*-.8;expected2=p(a2,b2,4,8,old);owner.fill(np.nan);M._sm40_extra_leaf.fill(np.nan);got2=q(a2,b2,4,8,old);assert got2.tobytes()==expected2.tobytes();assert got.tobytes()==expected.tobytes()
  owner.fill(np.nan);M._sm40_extra_leaf.fill(np.nan)
  for v in M._sm40_outer.values():v.fill(np.nan)
  assert got2.tobytes()==expected2.tobytes();rows.append(dict(name=name,shape=[m,k,n,rank],exact=True))
out=dict(passed=True,candidate_sha256=hashlib.sha256(S.read_bytes()).hexdigest(),scope='Actual source AST extracted helper. All105 fixed plan shapes fit actualfamilies/arena. End-to-end depth4 exact newinner81/P40, bothorientations/rank384and320, changedinput/NaNpoison/detachedoutputs. No model.',capacity=capacity,rows=rows,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with(D/'TINY_FIXED384_v1.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
