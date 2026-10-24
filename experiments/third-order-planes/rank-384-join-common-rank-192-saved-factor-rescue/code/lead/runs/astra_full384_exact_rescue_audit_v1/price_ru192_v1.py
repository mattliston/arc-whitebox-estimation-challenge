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
tr=O/'scout_native/astra_ubcluster_native_v1/native_7100136_arm_trace.json';ob=json.loads(tr.read_text())['observations'][0]['observation'];rows=[]
for c in ob['calls']:
 if c['layer'] not in range(10,15) or c['name'] not in ['_ubc_product','_ubx_product'] or c['args'][-1] not in ['u','ut']:continue
 a,b=c['args'][:2];m,k=a['shape'];n=b['shape'][1];old=price(m,k,n);new=price(m,192,n)if c['args'][-1]=='u' else price(m,k,192)
 rows.append(dict(layer=c['layer'],leg=c['args'][-1],shape=[m,k,n],parent256=old,reduced192=new,zero_B_saving=old['zero_B_forward']-new['zero_B_forward'],uncached_saving=old['total']-new['total']))
out=dict(status='CURRENT_KERNEL_SHAPE_ONLY_OLD_FIXED136_GEOMETRY_NO_MODELS',source_sha256=p.SOURCE_SHA,trace=str(tr),trace_sha256=hashlib.sha256(tr.read_bytes()).hexdigest(),rows=rows,all_L10to14_zero_B_saving=sum(r['zero_B_saving']for r in rows),L10to12_zero_B_saving=sum(r['zero_B_saving']for r in rows if r['layer']<=12),all_L10to14_uncached_saving=sum(r['uncached_saving']for r in rows),trace_total_bill=ob['total_flops'],limits=['Only common-u reconstruction/read products; difference rank384 untouched.','Current kernels, prior geometry. Zero-B assumes perfect same-layout B-tree caching; copy/prebinding/transitions excluded.','L15 returns256, requiring a new base cache generation if192 usedL10-14.','Full384 cache admission currently hardcoded320/256, so proper192/384 ownership adaptation is mandatory.'])
f=D/'RU192_WARM_TARIFF_v1.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k not in ['rows','limits']}));print(hashlib.sha256(f.read_bytes()).hexdigest())
