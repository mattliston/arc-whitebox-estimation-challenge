"""Virtual primitive tariffs only: no model, weights, predictions or truth arrays."""
from pathlib import Path
import ast, hashlib, importlib.util, json
D=Path(__file__).resolve().parent; O=D.parents[2]
P=O/'lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py'
s=importlib.util.spec_from_file_location('virtual_read256',P)
p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
p.SOURCE=O/'lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py'
p.SOURCE_SHA='d401532639bdc3d1b294518828f4af03a873727e510e0d4ec91399b1fc169d5a'
m,patched=p.load();fnp=p.fnp;flops=p.flops
tree=ast.parse(p.SOURCE.read_text())
node=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_ubx_ks_product' and 15000<x.lineno<16000][0]
clone=ast.get_source_segment(p.SOURCE.read_text(),node)
assert clone.count('if ctx is not None and not hzc:')==1
clone=clone.replace('def _ubx_ks_product(', 'def _read224_ubx_ks(').replace('if ctx is not None and not hzc:', 'if ctx is not None and (not hzc or n == 224):')
exec(compile(clone,'<read224_cache_admission>','exec'),m.__dict__)
m._ASTRA_IOP_PARENT_UBX=m._read224_ubx_ks

with flops.BudgetContext(flop_budget=2**60,quiet=True):
 arena=fnp.empty((m._PACKED_ELEMENTS,),dtype=fnp.float32)
 m._SCRATCH=[arena,m._cp3_bind_arena_views(arena,m._PACKED_PUSH),m._cp3_bind_arena_views(arena,m._PACKED_READ),None,None,None,None]
 m._bind_push6(arena)
history=O/'lead/runs/astra_difference_v192_v1/TARIFF_AND_HISTORY_PROPOSAL_v1.json'
sequences=json.loads(history.read_text())['detail'][:2]
# Both width256 and320 routes already exist in the parent (ut/vt). Warm the
# shared binding views for all these shapes before the price comparison; reset
# only numerical B-cache ownership before each measured sequence. Original
# same-shape setup keys are retained; no hypothetical free new binding keys.
for rank in (320,):
 for sequence in sequences:
  for row in sequence['products']:
   if row['leg'] != 'vt': continue
   n,Pn,_=row['shape']
   with flops.BudgetContext(flop_budget=2**60,quiet=True):
    a=fnp.empty((n,Pn),dtype=fnp.float32); b=fnp.empty((Pn,rank),dtype=fnp.float32)
    m._shared_old384_mm(a,b)
records=[]
for sequence in sequences:
 rows=[x for x in sequence['products'] if x['leg']=='vt']
 bills={}
 for rank in (320,224):
  m._UBX_STATE.update(run=None,cache={},bytes=0)
  m._UBX_STATS.update(products=0,hits=0,stores=0,uncached=0,skipped=0,strip_hits=0)
  m._TR_LAST[0]=None
  with flops.BudgetContext(flop_budget=2**60,quiet=True):
   owner=fnp.empty((320,11712),dtype=fnp.float32)
   Fv=fnp.empty((rank,5856),dtype=fnp.float32)
   Fu=fnp.empty((256,5856),dtype=fnp.float32)
  uv=(Fu,Fv,((0,5856),),256)
  m._UBC_RUN[0]=dict(owner=owner,uv=uv,generation=('read256-price',rank))
  calls=[]
  for row in rows:
   n,Pn,_=row['shape']
   with flops.BudgetContext(flop_budget=2**60,quiet=True):
    left=fnp.empty((n,Pn),dtype=fnp.float32)
   with flops.BudgetContext(flop_budget=2**60,quiet=True)as ctx:
    out=m._ubx_product(left,Fv.T,uv,'vt')
   assert tuple(out.shape)==(n,rank)
   calls.append(dict(layer=row['layer'],rows=n,bill=int(ctx.flops_used),requests=len(ctx._op_log)))
  bills[str(rank)]=dict(calls=calls,total=sum(c['bill']for c in calls),requests=sum(c['requests']for c in calls),cache_stats=dict(m._UBX_STATS))
  m._UBC_RUN[0]=None
 # Conservative scalar/gather/scatter allowance, no free copy claim. One fixed
 # frame selection over 320x5856; five full320 result zero/scatters. No projection.
 allowance=5*320*5856+4*224*5856+sum(8*x['shape'][0]*320 for x in rows)
 saving=bills['320']['total']-bills['224']['total']-allowance
 records.append(dict(seed=sequence['seed'],bills=bills,selection_and_scatter_allowance=allowance,
                     net_saving=saving,pct_of_257G=100*saving/257e9))
out=dict(status='SOURCE_ONLY_VIRTUAL_TARIFF_NO_MODELS',source_sha256=p.SOURCE_SHA,
         history_path=str(history),history_sha256=hashlib.sha256(history.read_bytes()).hexdigest(),
         caveat='Historical same-family response shapes; current routing may change. Common, reconstruction, full320 outer read and diagonal all retained. No accuracy claim.',
         binding_policy='Parent320 shapes bound first; new224 binding debt charged during its first calls; independent cold B caches and original strip/no-cache policy.',clone_sha256=hashlib.sha256(clone.encode()).hexdigest(),cache_change='Only permit original owned B tree/cache path at hzc n224; tag already pins actual n224. Existing half-column leaf views unchanged.',records=records,numerical_payload_bytes=p.PAYLOAD[0])
with(D/'READ224_CACHED_VIRTUAL_TARIFF_v1.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out,indent=2))
