"""Price retained SWAP table changes only outside the frozen peel shape list."""
from pathlib import Path
import gc,hashlib,json,resource,time
HERE=Path(__file__).resolve().parent;R=HERE.parent
ns={'__file__':str(HERE/'reprice_v1.py'),'__name__':'swap_residual_metadata'}
exec((HERE/'reprice_v1.py').read_text().split('fixed=json.loads')[0],ns)
S,fn,fs,arr,cost,probe=(ns[k]for k in ('S','fn','fs','arr','cost','probe'))
original=dict(S._ASTRA_SM40_PLANS)
fragments=[R/'opus_smtable_v1/fragment_smtable_cap_v2.py.txt',R/'opus_releaseidea_v1/fragment_swap_drop12_v1.py.txt']
for p in fragments:exec(p.read_text(),S.__dict__)
swapped=dict(S._ASTRA_SM40_PLANS)
assert (len(original),len(swapped))==(167,530)
peel=json.loads((HERE/'DEPLOY_S2TR_CAP750_v2.json').read_text())
excluded={tuple(r['shape'])for r in peel['rows']if r['kind']=='read'}
filtered=dict(swapped)
for key in excluded:
    if key in original:filtered[key]=original[key]
    else:filtered.pop(key,None)
data=json.loads((R/'astra_crunch_compute_v1/bini_apa_v1/READ_TARIFF_v2.json').read_text())
todo=[r for r in data['rows']if tuple(r['shape'])not in excluded and original.get(tuple(r['shape']))!=filtered.get(tuple(r['shape']))]
t0=time.monotonic();rows=[]
for i,r in enumerate(todo):
    m,k,n,rank=r['shape'];old=int(rank>0)
    with fs.BudgetContext(flop_budget=2**60,quiet=True):
        full=arr((m,k+old));yr=arr((n,k));lam=arr((k+old,));inner=arr((m,rank));right=arr((n,rank))
        idx=fn.asarray(probe.np.arange(n,dtype='int64'));state=dict(columns=old,pre_basis=arr((1024,rank)))
    S._cp_slr_inner_sq=lambda *args:inner;S._xsg_rows=lambda *args:right
    def run():return S._astra_sm40_read(full,yr,lam,idx,state)
    S._ASTRA_SM40_PLANS=original;cost(run);base=cost(run)
    S._ASTRA_SM40_PLANS=filtered;cost(run);new=cost(run)
    rows.append(dict(shape=r['shape'],original_plan=original.get(tuple(r['shape'])),swap_plan=filtered.get(tuple(r['shape'])),
                     base=base,swap=new,gain=base['flops']-new['flops'],request_delta=new['requests']-base['requests']))
    S._BH_HOLD.clear();gc.collect()
by={tuple(r['shape']):r for r in rows};coverage=[]
for c in data['coverage']:
    used=[by[tuple(s)]for s in c['hits']if tuple(s)in by]
    gain=sum(r['gain']for r in used);requests=sum(r['request_delta']for r in used)
    coverage.append(dict(seed=c['seed'],gain=gain,pct=100*gain/c['bill'],request_delta=requests,
                         positive_calls=sum(max(0,r['request_delta'])for r in used),hits=[r['shape']for r in used]))
cb={r['seed']:r for r in coverage}
combined=[dict(seed=p['seed'],peel_pct=p['pct'],swap_pct=cb[p['seed']]['pct'],combined_pct=p['pct']+cb[p['seed']]['pct'],
               request_delta=p['net_extra_calls']+cb[p['seed']]['request_delta'],positive_calls=p['positive_calls_spent']+cb[p['seed']]['positive_calls'])for p in peel['matched8']]
summary=dict(changed_captured_nonpeel_shapes=len(rows),excluded_peel_read_shapes=len(excluded),
             swap24_mean_pct=sum(r['pct']for r in coverage)/len(coverage),swap24_mean_requests=sum(r['request_delta']for r in coverage)/len(coverage),
             swap24_max_requests=max(r['request_delta']for r in coverage),
             swap_matched8_mean_pct=sum(r['swap_pct']for r in combined)/len(combined),
             combined_matched8_mean_pct=sum(r['combined_pct']for r in combined)/len(combined),
             combined_matched8_mean_requests=sum(r['request_delta']for r in combined)/len(combined),
             combined_matched8_max_requests=max(r['request_delta']for r in combined),
             combined_matched8_max_positive_calls=max(r['positive_calls']for r in combined))
out=dict(status='SWAP_ONLY_ON_NONPEEL_READ_SHAPES_PUBLIC_METADATA',s2tr_source_sha=probe.SOURCE_SHA,
         fragment_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in fragments},
         original_plan_count=len(original),swap_plan_count=len(swapped),filtered_plan_count=len(filtered),
         excluded=[list(k)for k in sorted(excluded)],summary=summary,rows=rows,coverage=coverage,combined=combined,
         seconds=time.monotonic()-t0,maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         limits=['Only original two retained SWAP fragments; no retuning or added plan.',
                 'All selected peel read keys restored to their S2tr plan, even when peel allowance might reject them. No overlapping credit.',
                 'Combined numbers add independent SWAP calls outside750 peel allowance; a shared750 allowance needs replay before claiming750.',
                 'Warm call proxy and saved reference whole denominators; no numeric/native/runtime validation.'])
(HERE/'SWAP_RESIDUAL_v1.json').write_text(json.dumps(out,indent=2)+'\n')
changes={k:v for k,v in filtered.items()if original.get(k)!=v}
drops=sorted(k for k in original if k not in filtered)
fragment='# Retained SWAP changes only outside fixed peel read keys. No new plans.\n_ASTRA_SM40_PLANS.update('+repr(changes)+')\n'
fragment+='for _swap_key in '+repr(tuple(drops))+':\n    _ASTRA_SM40_PLANS.pop(_swap_key, None)\ndel _swap_key\n'if drops else ''
(HERE/'fragment_swap_nonpeel_v1.py').write_text(fragment)
print(json.dumps(dict(summary=summary,seconds=out['seconds'],maxrss=out['maxrss'],plans=[len(original),len(swapped),len(filtered)])),flush=True)
