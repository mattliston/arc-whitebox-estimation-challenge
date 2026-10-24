import os,json,time,resource
from pathlib import Path
assert os.getpriority(os.PRIO_PROCESS,0)==10
O=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1');V=O/'lead/runs/astra_smirnov_current24_virtual_v1';D=O/'lead/runs/astra_read256_inner_cost_v1';D.mkdir(exist_ok=True);g={'__file__':str(V/'price_v1.py')};exec((V/'price_v1.py').read_text().split('circuits=json.loads')[0],g)
M=g['M'];fnp=g['fnp'];fs=g['fs'];measure=g['measure'];arrays=g['arrays'];rows=[];start=time.monotonic()
for cp in sorted((O/'lead/runs/astra_prefix_row_energy22_science_v1').glob('run_rpe_*_s*_v1/native_*_ON.json')):
 rr=json.loads(cp.read_text())['rows'][0];reads=rr['stats']['_RTT_STATS']['reads'];P=next(q['k']for q in reads if q['layer']==6)//2;ms=[q['m']for q in reads if 10<=q['layer']<=14];assert len(ms)==5;prices={};stats={}
 for rank in (320,256):
  with fs.BudgetContext(flop_budget=2**60,quiet=True):
   Fu=fnp.empty((256,P),dtype=fnp.float32);Fv=fnp.empty((320,P),dtype=fnp.float32);owner=fnp.empty((320,2*P),dtype=fnp.float32);right=Fv[:rank].T
  uv=(Fu,Fv,((0,P),),256);run=dict(owner=owner,uv=uv,generation=('fixed_read_base',P),cache={},bytes=0,stats=dict(generations=1,eligible=0,hits=0,misses=0,stores=0,copies=0,copy_elements=0,copied_bytes=0,max_resident_bytes=0,uncached=0,peak_cache_bytes=0,hit_slots={'u:0':0,'v:0':0,'v:4096':0,'ut:0':0,'vt:0':0,'u:4096':0}))
  M._UBC_RUN[0]=run;M._UBX_STATE.update(run=None,cache={},bytes=0);M._UBX_STATS.update(products=0,hits=0,stores=0,uncached=0,skipped=0,strip_hits=0);aa=[arrays(m,P,rank)[0]for m in ms]
  # Current B transform caching and all row-dependent structural views warmed.
  for _ in range(2):
   for a in aa:measure(lambda:M._ubx_product(a,right,uv,'vt'))
  prices[rank]=[measure(lambda:M._ubx_product(a,right,uv,'vt'))for a in aa];stats[rank]=dict(M._UBX_STATS)
  assert M._UBX_STATS['hits']>0 and M._UBX_STATS['uncached']==0
  M._UBC_RUN[0]=None;M._UBX_STATE.update(run=None,cache={},bytes=0)
 saving=sum(x['flops']for x in prices[320])-sum(x['flops']for x in prices[256]);pad=sum(ms)*320;row=dict(seed=rr['seed'],P=P,m=ms,baseline=prices[320],candidate=prices[256],cache_stats=stats,product_saving=saving,fullrank_padding_flops=pad,net_saving=saving-pad,credit_pct=100*(saving-pad)/rr['flops']);rows.append(row)
 print(json.dumps(dict(seed=row['seed'],P=P,credit_pct=row['credit_pct'])),flush=True)
out=dict(scope='Exact actual virtualFNP current UBXcached vt products only. Five L10–14 old-base read inners320→256; zero-pad tofull320 afterward. Fullstate/reconstruction/common/output contractions remainunchanged. P=halfcohort0..5 roster (layer6 read k), sourcejoins10(3..5); samebase through L14. Warm planning bound, not actual wholemodel firstbill or numerical claim.',rows=rows,summary={k:dict(mean=sum(r[k]for r in rows)/len(rows),min=min(r[k]for r in rows),max=max(r[k]for r in rows))for k in ('product_saving','net_saving','credit_pct')},seconds=time.monotonic()-start,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);(D/'CURRENT24_WARM_TARIFF_v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k]for k in ('summary','seconds','max_rss_bytes')}),flush=True)
