import os,json,time,resource,functools
from pathlib import Path
assert os.getpriority(os.PRIO_PROCESS,0)==10
O=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1');V=O/'lead/runs/astra_smirnov_current24_virtual_v1';D=O/'lead/runs/astra_read256_inner_cost_v1';g={'__file__':str(V/'price_v1.py')};exec((V/'price_v1.py').read_text().split('circuits=json.loads')[0],g)
M=g['M'];M._UBC_RUN[0]=None;fnp=g['fnp'];measure=g['measure'];arrays=g['arrays'];base={r['seed']:r for r in json.loads((D/'CURRENT24_WARM_TARIFF_v1.json').read_text())['rows']};start=time.monotonic();rows=[]
@functools.lru_cache(None)
def mm(m,k,n):
 a,b,_=arrays(m,k,n);measure(lambda:M._shared_old384_mm(a,b));return measure(lambda:M._shared_old384_mm(a,b))
for cp in sorted((O/'lead/runs/astra_prefix_row_energy22_science_v1').glob('run_rpe_*_s*_v1/native_*_ON.json')):
 rr=json.loads(cp.read_text())['rows'][0];reads={q['layer']:q for q in rr['stats']['_RTT_STATS']['reads']};Pearly=reads[3]['k']//2;Pbase=reads[6]['k']//2;Precent=(reads[9]['k']+reads[3]['k']-reads[6]['k'])//2;parts=[]
 for layer in (7,8,9,13,14):
  P=Pearly if layer<10 else Precent;m=reads[layer]['m'];a=mm(m,P,384);b=mm(m,P,320);credit=a['flops']-b['flops']-m*384;parts.append(dict(layer=layer,m=m,P=P,baseline=a,candidate=b,zero_pad_flops=m*384,credit=credit))
 extra=sum(p['credit']for p in parts);total=base[rr['seed']]['net_saving']+extra;row=dict(seed=rr['seed'],P_early=Pearly,P_base=Pbase,P_recent=Precent,base_cached_credit=base[rr['seed']]['net_saving'],uncached_parts=parts,uncached_credit=extra,total_credit=total,credit_pct=100*total/rr['flops']);rows.append(row)
out=dict(scope='One fixed uniform64-coordinate drop only inside all old-bank repeated-index read vt legs L7–14. Early/recent384→320 uncached plus base320→256 cached; full common256, state/reconstruction and finalrank remain unchanged viazero-padding. Exact virtual FNP warm tariff, no model/source or accuracy claim.',rows=rows,summary={k:dict(mean=sum(r[k]for r in rows)/len(rows),min=min(r[k]for r in rows),max=max(r[k]for r in rows))for k in ('base_cached_credit','uncached_credit','total_credit','credit_pct')},seconds=time.monotonic()-start,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);(D/'UNIFORM64_CURRENT24_WARM_TARIFF_v2.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k]for k in ('summary','seconds','max_rss_bytes')}))
