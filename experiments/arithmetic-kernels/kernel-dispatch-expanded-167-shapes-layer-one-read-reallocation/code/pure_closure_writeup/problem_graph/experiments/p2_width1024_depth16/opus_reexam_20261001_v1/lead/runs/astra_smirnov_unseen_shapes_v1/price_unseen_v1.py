import os,sys,ast,json,time,hashlib,resource
from pathlib import Path
assert os.getpriority(os.PRIO_PROCESS,0)==10
O=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1');D=O/'lead/runs/astra_smirnov_unseen_shapes_v1';V=O/'lead/runs/astra_smirnov_current24_virtual_v1';g={'__file__':str(V/'price_v1.py')};exec((V/'price_v1.py').read_text().split('circuits=json.loads')[0],g)
M=g['M'];fnp=g['fnp'];fs=g['fs'];measure=g['measure'];arrays=g['arrays'];incumbent=g['incumbent']
sys.path.insert(0,str(O/'lead/runs/astra_smirnov_cfull_v1'));import smirnov_cfull_v1 as h
binding=measure(lambda:h.bind_universal(M,M._SCRATCH[0]));records=[r for r in json.loads((O.parent/'astra_phase2_reexamination_v1/smirnov_executable_transpose_certificate_v1.json').read_text())['records']if r['shape']==[3,6,3]];kernels={r['name']:h.Kernel(M,r)for r in records}
census=json.loads((D/'CENSUS_v1.json').read_text());rows=[];start=time.monotonic()
def clear():
 for name in ('_sm40_cache','_sm40_ccrop_cache','_sm40_io_cache','_sm40_outer_cache','_sm40_group_cache','_sm40_padding_grid','_sm40_padding_cache'):getattr(M,name).clear()
for i,r in enumerate(census['novel_shapes']):
 key=tuple(r['shape']);m,k,n,rank=key;base=incumbent(*key);a,b,_=arrays(m,k,n)
 with fs.BudgetContext(flop_budget=2**60,quiet=True):lam=fnp.empty((k,),dtype=fnp.float32);old=(fnp.empty((m,rank),dtype=fnp.float32),fnp.empty((rank,n),dtype=fnp.float32))if rank else None
 choices=[]
 for L in (4,5,6):
  u=1<<L;cm=(m+3*u-1)//(3*u)*u;ck=(k+rank+6*u-1)//(6*u)*u;cn=(n+3*u-1)//(3*u)*u
  try:family=h.family_for(L,cm,ck,cn)
  except StopIteration:continue
  for name,q in kernels.items():
   clear()
   def fn():return q(a*lam,b,L,family[2],old)if m<n else q(a,b*lam[:,None],L,family[2],old)
   first=measure(fn);warm=measure(fn);choices.append(dict(name=name,L=L,P=family[2],family=family,first=first,warm=warm,first_saving=base['flops']-first['flops'],warm_saving=base['flops']-warm['flops']))
 choices.sort(key=lambda x:x['first']['flops']);best=choices[0]if choices and choices[0]['first_saving']>0 else None
 rows.append(dict(shape=key,count=r['count'],baseline=base,best=best,choices=choices))
 if i%20==0:print(json.dumps(dict(at=i,total=len(census['novel_shapes']),seconds=time.monotonic()-start)),flush=True)
by={tuple(r['shape']):r for r in rows};per=[]
for r in census['networks']:
 eligible=[by[tuple(s)]['best']for s in r['shapes']if tuple(s)in by and by[tuple(s)]['best']];first=sum(q['first_saving']for q in eligible);warm=sum(q['warm_saving']for q in eligible);per.append(dict(seed=r['seed'],new_calls=len(eligible),first_saving=first,warm_saving=warm,first_credit_pct=100*first/r['flops'],warm_credit_pct=100*warm/r['flops']))
out=dict(scope='Cost-only exact virtual FNP. New shapes from32 independent saved networks, no MSE; current105 preserved. Choose minimum conservative isolated first-call FLOPs among both exact3x6x3 circuits and L4/5/6 only when same five prebound families fit. Baseline actual warm parent tariff excludes same old-inner product on both sides. Current setup/family unchanged.',census_sha256=hashlib.sha256((D/'CENSUS_v1.json').read_bytes()).hexdigest(),rows=rows,per_network=per,summary={k:dict(mean=sum(r[k]for r in per)/len(per),min=min(r[k]for r in per),max=max(r[k]for r in per))for k in ('new_calls','first_credit_pct','warm_credit_pct')},new_plans=sum(r['best']is not None for r in rows),seconds=time.monotonic()-start,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(D/'UNSEEN32_PRICE_v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k]for k in ('summary','new_plans','seconds','max_rss_bytes')}),flush=True)
