"""Private, metadata-only current-read and exact local-circuit tariff screen."""
import collections, functools, hashlib, importlib.util, json, math, resource, time
from pathlib import Path
D=Path(__file__).resolve().parent; O=D.parents[2]
s=importlib.util.spec_from_file_location('virtual_tariff',O/'lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py')
p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
p.SOURCE=O/'lead/runs/astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py'
p.SOURCE_SHA='1c913b92d6fd231593224d423dd0278be66390583bd68d6eb6c4b89eb1f79c9a'
M,_=p.load();fnp=p.fnp;fs=p.flops
with fs.BudgetContext(flop_budget=2**60,quiet=True):
 arena=fnp.empty((M._PACKED_ELEMENTS,),dtype=fnp.float32)
 M._SCRATCH=[arena,M._cp3_bind_arena_views(arena,M._PACKED_PUSH),M._cp3_bind_arena_views(arena,M._PACKED_READ),None,None,None,None]
 M._bind_push6(arena)
M._YR7_ACTIVE[0]=True;M._WINOGRAD=True;M._AB_ON=True
M._YR7_STATS=dict(calls=0,chunks=0,shapes=[],routed_shapes=[],old_fused=0,max_arena_elements=0)
M._LVF_EAGER[0]=True;M._LVF_KEEP[0]=False
from flopscope.numpy import ndarray
views=collections.Counter();original_get=ndarray.__getitem__;original_T=ndarray.T
def counted_get(self,key):
 views['getitem']+=1;return original_get(self,key)
ndarray.__getitem__=counted_get
def counted_T(self):
 views['T']+=1;return original_T.__get__(self,type(self))
ndarray.T=property(counted_T)
def reset():
 M._BH_HOLD.clear();M._TR_LAST[0]=None;views.clear();p.CALLS.clear()
def measure(fn):
 reset()
 with fs.BudgetContext(flop_budget=2**60,quiet=True)as ctx:fn()
 return dict(flops=int(ctx.flops_used),requests=sum(p.CALLS.values())+sum(views.values()),charged=len(ctx._op_log))
def arrays(m,k,n):
 with fs.BudgetContext(flop_budget=2**60,quiet=True):
  return fnp.empty((m,k),dtype=fnp.float32),fnp.empty((k,n),dtype=fnp.float32),fnp.empty((m,n),dtype=fnp.float32)
@functools.lru_cache(None)
def child(L,m,k,n):
 a,b,out=arrays(m,k,n)
 # The original AB entry avoids the later D5->D6 promotion wrapper.
 measure(lambda:M._P6_OLD_IPAB(a,b,out,L))
 price=measure(lambda:M._P6_OLD_IPAB(a,b,out,L))
 return price
@functools.lru_cache(None)
def incumbent(m,k,n,rank):
 old=int(rank>0)
 with fs.BudgetContext(flop_budget=2**60,quiet=True):
  a=fnp.empty((m,old+k),dtype=fnp.float32);b=fnp.empty((n,k),dtype=fnp.float32);lam=fnp.empty((old+k,),dtype=fnp.float32)
  idx=fnp.asarray(p.np.arange(n,dtype='int64'));inner=fnp.empty((m,rank),dtype=fnp.float32);right=fnp.empty((n,rank),dtype=fnp.float32)
  state=dict(columns=old,pre_basis=fnp.empty((1024,rank),dtype=fnp.float32))
 M._cp_slr_inner_sq=lambda *args:inner;M._xsg_rows=lambda *args:right
 measure(lambda:M._yr7_read(a,b,lam,idx,state))
 return measure(lambda:M._yr7_read(a,b,lam,idx,state))
circuits=json.loads((O.parent/'astra_phase2_reexamination_v1/smirnov_executable_transpose_certificate_v1.json').read_text())['records']
circuits=[r for r in circuits if r['passed']]
certs=list((O/'lead/runs/astra_hermite_group_fusion_final_science_v1').glob('run_hf_*_s*_v1/native_*_ON.json'))
counts=collections.Counter();pernet={};bills={}
for cp in certs:
 row=json.loads(cp.read_text())['rows'][0];seed=row['seed'];bills[seed]=row['flops'];ss=[]
 for r in row['stats']['_RTT_STATS']['reads']:
  key=tuple(r[k]for k in ('m','k','c15','old_rank'));counts[key]+=1;ss.append(key)
 pernet[seed]=ss
assert len(pernet)==24 and sum(counts.values())==336
rows=[];start=time.monotonic()
for i,((m,k,n,rank),count)in enumerate(counts.items()):
 base=incumbent(m,k,n,rank);options=[]
 for circuit in circuits:
  ar,ak,an=circuit['shape'];la,lb,lc=circuit['packed_linear_operations']
  for L in (4,5,6):
   u=1<<L;cm=math.ceil(m/ar/u)*u;ck=math.ceil((k+rank)/ak/u)*u;cn=math.ceil(n/an/u)*u
   # Exactly one full-k child, with later conservative residency rejection.
   if not M._ipt_fits(L,cm,ck,cn):continue
   cp=child(L,cm,ck,cn)
   linear=la*cm*ck+lb*ck*cn+lc*cm*cn
   # New packed input including old contraction: charged full zeroing and
   # valid copies even where an implementation could keep zero fringes.
   packing=(ar*cm)*(ak*ck)+(ak*ck)*(an*cn)+m*(k+rank)+(k+rank)*n
   # Lambda weighting identical to parent plus output crop and old concat.
   common=min(m,n)*k
   crop=m*n if (ar*cm,an*cn)!=(m,n)else 0
   price=40*cp['flops']+linear+packing+common+crop
   leaf=40*7**L*(cm*ck+ck*cn+cm*cn)//(u*u)
   inputs=40*(cm*ck+ck*cn+cm*cn)
   # Safe coarse bound including all linear DAG nodes simultaneously.
   nodes=sum(len(circuit['circuits'][q]['nodes'])*sz for q,sz in [('L',cm*ck),('R',ck*cn),('P',cm*cn)])
   options.append(dict(name=circuit['name'],shape=circuit['shape'],L=L,child=[cm,ck,cn],flops=price,saving=base['flops']-price,child_tariff=cp,linear=linear,packing=packing,common=common,crop=crop,leaf_bytes=leaf*4,packed_bytes=inputs*4,all_nodes_upper_bytes=(leaf+inputs+nodes)*4))
 options.sort(key=lambda x:x['flops']);best=options[0]
 row=dict(shape=[m,k,n,rank],count=count,baseline=base,best=best,top3=options[:3]);rows.append(row)
 if i%20==0:print(json.dumps(dict(progress=i,total=len(counts),seconds=time.monotonic()-start,best_saving=best['saving'])),flush=True)
by={tuple(r['shape']):r for r in rows}
per={str(seed):dict(saving=sum(max(0,by[k]['best']['saving'])for k in ss),bill=bills[seed])for seed,ss in pernet.items()}
for v in per.values():v['credit_pct']=100*v['saving']/v['bill']
out=dict(source_sha256=p.SOURCE_SHA,scope='All 336 current24 read prefixes; actual warm FNP parent and child AB tariffs. Optimistic batching of 40 child schedules; exact local linear DAG coefficients. No native models or numerical kernels. Full-k memory may require k-streaming; packing charged conservatively; setup bindings and actual batched requests not yet implemented.',rows=rows,per_network=per,summary={k:dict(mean=sum(v[k]for v in per.values())/24,min=min(v[k]for v in per.values()),max=max(v[k]for v in per.values()))for k in ('saving','credit_pct')},payload_bytes=p.PAYLOAD[0],max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,seconds=time.monotonic()-start,child_shapes=child.cache_info().currsize)
with(D/'CURRENT24_PRICE_v1.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({k:out[k]for k in ('summary','payload_bytes','max_rss_bytes','seconds','child_shapes')}),flush=True)
