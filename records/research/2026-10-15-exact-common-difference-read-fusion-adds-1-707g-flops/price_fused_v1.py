from pathlib import Path
import importlib.util,hashlib,json
D=Path(__file__).resolve().parent;O=D.parents[2]
s=importlib.util.spec_from_file_location('tariff',O/'lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py');p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
p.SOURCE=O/'lead/runs/astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v3.py';p.SOURCE_SHA=hashlib.sha256(p.SOURCE.read_bytes()).hexdigest();M,_=p.load();fnp=p.fnp;fs=p.flops
with fs.BudgetContext(flop_budget=2**60,quiet=True):
 a=fnp.empty((M._PACKED_ELEMENTS,),dtype=fnp.float32);M._SCRATCH=[a,M._cp3_bind_arena_views(a,M._PACKED_PUSH),M._cp3_bind_arena_views(a,M._PACKED_READ),None,None,None,None];M._bind_push6(a)
M._WINOGRAD=True;M._AB_ON=True;M._LVF_EAGER[0]=True;M._LVF_KEEP[0]=False
ctx=[None];bf=[0];orig=M._ipt_forward_b_ab
# Count actual B-forward bill separately; zero-forward result is an optimistic warm-cache normalization, not a measured production bill.
def counted(*args,**kwargs):
 before=ctx[0].flops_used if ctx[0] else 0;ans=orig(*args,**kwargs)
 if ctx[0]:bf[0]+=ctx[0].flops_used-before
 return ans
M._ipt_forward_b_ab=counted
cache={}
def price(m,k,n):
 key=(m,k,n)
 if key in cache:return cache[key]
 with fs.BudgetContext(flop_budget=2**60,quiet=True): a=fnp.empty((m,k),dtype=fnp.float32);b=fnp.empty((k,n),dtype=fnp.float32);M._shared_old384_mm(a,b)
 bf[0]=0
 with fs.BudgetContext(flop_budget=2**60,quiet=True)as c:
  ctx[0]=c;M._shared_old384_mm(a,b);ctx[0]=None
 M._BH_HOLD.clear();cache[key]=dict(total=int(c.flops_used),B_forward=bf[0],zero_B_forward=int(c.flops_used)-bf[0]);return cache[key]
tr=O/'scout_native/astra_ubcluster_native_v1/native_7100136_arm_trace.json'
ob=json.loads(tr.read_text())['observations'][0]['observation'];calls=ob['calls'];rows=[]
for i,c in enumerate(calls[:-1]):
 if c['name']!='_ubc_product' or c['args'][-1]!='u':continue
 v=calls[i+1]
 if v['name']!='_ubc_product' or v['args'][-1]!='v' or v['layer']!=c['layer']:continue
 au,bu=c['args'][:2];av,bv=v['args'][:2]
 m,ku=au['shape'];_,kv=av['shape'];n=bu['shape'][1]
 assert bv['shape'][1]==n
 old_u=price(m,ku,n);old_v=price(m,kv,n);fused=price(m,kv,2*n)
 old=old_u['zero_B_forward']+old_v['zero_B_forward'];new=fused['zero_B_forward']
 rows.append(dict(layer=c['layer'],m=m,ku=ku,kv=kv,n=n,old_warm_optimistic=old,fused_warm_optimistic=new,saving=old-new))
out=dict(status='OFFLINE_EXACT_PRODUCT_FUSION_COST_ONLY',rows=rows,saving=sum(r['saving']for r in rows),historical_bill=ob['total_flops'],source_sha256=p.SOURCE_SHA,limits=['Exact real arithmetic: U@[zero-padded Fu,Fv] replaces two products; rounding and packing not tested.','Current kernel on prior136 geometry; warm zero B-forward normalization only; no packing cost, cache reuse or deployment guarantee.','No estimator evaluation, no candidate, no scientific claim.'])
f=D/'WARM_TARIFF_v1.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out));print(hashlib.sha256(f.read_bytes()).hexdigest())
