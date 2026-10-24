"""Shape-only multiple-choice tariff frontier; no numeric or model execution."""
from pathlib import Path
import collections,hashlib,json,time,resource,warnings
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.sparse import lil_matrix

HERE=Path(__file__).resolve().parent;R=HERE.parent
peel=json.loads((R/'astra_crunch_compute_v1/peel_sm40_v1/TARIFF_v1.json').read_text())
tail=json.loads((HERE/'TARIFF_v1.json').read_text())
pc=json.loads((R/'astra_crunch_compute_v1/bini_apa_v1/OUTER_COMBINED_TARIFF_v3.json').read_text())['coverage']
rc=json.loads((R/'astra_crunch_compute_v1/bini_apa_v1/READ_TARIFF_v2.json').read_text())['coverage']
readseed={c['seed']:c for c in rc}
matched=[(p,readseed[p['seed']])for p in pc]
offers=collections.defaultdict(list)
for row in peel['rows']:
    key=(row['kind'],tuple(row['shape']))
    for trial in row['trials']:
        if trial['saving']>0:
            offers[key].append(dict(kind=key[0],shape=list(key[1]),policy='parent_tails',scheme=trial['scheme'],cuts=trial['cuts'],
                                    saving=trial['saving'],request_delta=trial['request_delta'],flops=trial['flops'],requests=trial['requests'],
                                    tail_plans=[dict(**v,method='parent')for v in trial['tails']]))
for policy in ('shallower','dense_skinny','shallower_or_dense'):
    for row in tail['policies'][policy]['rows']:
        if row['saving']>0:offers[row['kind'],tuple(row['shape'])].append(dict(row,policy=policy))

# A single structural product gets at most one implementation. Pareto pruning
# is exact because every coverage coefficient is nonnegative and shape-only.
options=[];groups=[]
for key,choices in sorted(offers.items()):
    unique={}
    for c in choices:
        signature=(c['saving'],c['request_delta'])
        unique.setdefault(signature,c)
    choices=list(unique.values())
    keep=[c for c in choices if not any(d['saving']>=c['saving']and d['request_delta']<=c['request_delta']and(d['saving']>c['saving']or d['request_delta']<c['request_delta'])for d in choices)]
    groups.append(list(range(len(options),len(options)+len(keep))))
    options.extend(keep)

counts={}
for kind,collection in [('push',pc),('read',rc)]:
    for c in collection:counts[kind,c['seed']]=collections.Counter(tuple(h)for h in c['hits'])
def weight(o,c,kind):
    return counts[kind,c['seed']][tuple(o['shape'])] if o['kind']==kind else 0
objective=[];calls=[]
for o in options:
    collection=pc if o['kind']=='push' else rc
    objective.append(sum(weight(o,c,o['kind'])*o['saving']/(c['total']if o['kind']=='push'else c['bill'])for c in collection)/len(collection))
    calls.append([(weight(o,p,'push')+weight(o,r,'read'))*o['request_delta']for p,r in matched])
objective=np.asarray(objective);calls=np.asarray(calls,dtype=float).T


def summarize(selected):
    lookup={(o['kind'],tuple(o['shape'])):o for o in selected}
    coverage={}
    for kind,collection in [('push',pc),('read',rc)]:
        rows=[]
        for c in collection:
            found=[lookup[kind,tuple(h)]for h in c['hits']if(kind,tuple(h))in lookup]
            den=c['total']if kind=='push'else c['bill']
            rows.append(dict(seed=c['seed'],gain=sum(x['saving']for x in found),pct=100*sum(x['saving']for x in found)/den,
                             request_delta=sum(x['request_delta']for x in found),calls=len(found),denominator=den))
        coverage[kind]=rows
    rby={x['seed']:x for x in coverage['read']}
    mm=[dict(seed=x['seed'],pct=100*(x['gain']+rby[x['seed']]['gain'])/x['denominator'],
             request_delta=x['request_delta']+rby[x['seed']]['request_delta'])for x in coverage['push']]
    return dict(objective_all24read_plus8push_pct=sum(sum(x['pct']for x in coverage[k])/len(coverage[k])for k in coverage),
                matched8_mean_pct=sum(x['pct']for x in mm)/len(mm),matched8_mean_request_delta=sum(x['request_delta']for x in mm)/len(mm),
                matched8_max_request_delta=max(x['request_delta']for x in mm),matched8_min_pct=min(x['pct']for x in mm),
                read24_mean_request_delta=sum(x['request_delta']for x in coverage['read'])/len(rc),
                read24_max_request_delta=max(x['request_delta']for x in coverage['read']),
                selected_shapes=len(selected),policy_counts=dict(collections.Counter(x['policy']for x in selected)),
                matched8=mm,coverage=coverage)


t0=time.monotonic();frontier={}
for cap_kind in ('matched8_mean','matched8_each'):
    cc=calls.mean(axis=0,keepdims=True)if cap_kind=='matched8_mean'else calls
    A=lil_matrix((len(groups)+cc.shape[0],len(options)),dtype=float)
    for i,g in enumerate(groups):A[i,g]=1
    A[len(groups):,:]=cc
    A=A.tocsr()
    rows=[]
    for cap in (0,250,500,750,1000):
        upper=np.r_[np.ones(len(groups)),np.full(cc.shape[0],cap,dtype=float)]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Unrecognized options detected')
            result=milp(-objective*1e6,integrality=np.ones(len(options)),bounds=Bounds(0,1),
                        constraints=LinearConstraint(A,np.full(len(upper),-np.inf),upper),
                        options=dict(time_limit=8,mip_rel_gap=1e-7,threads=1,parallel=False))
        assert result.x is not None,result
        indexes=np.flatnonzero(result.x>.5);chosen=[options[i]for i in indexes]
        assert max(cc@np.round(result.x))<=cap+1e-6
        assert all(sum(result.x[i]>.5 for i in g)<=1 for g in groups)
        summary=summarize(chosen)
        row=dict(cap=cap,status=int(result.status),message=result.message,mip_gap=float(result.mip_gap),**summary)
        rows.append(row)
        dispatch=dict(cap_kind=cap_kind,cap=cap,source_sha=tail['source_sha'],summary=row,
                      rows=chosen,fallback='Unchanged exact parent for all unlisted product shapes.',
                      objective='Maximize mean fractional push bill saving on8 captures plus mean fractional read bill saving on24 captures.',
                      constraint='Additional wrapper calls on matched8; '+('mean'if cap_kind.endswith('mean')else'each network'),
                      scopes='Existing four cuts with parent tails plus three previously priced tail policies at original selected best cuts only.')
        name=f'DISPATCH_{cap_kind}_{cap}_v1.json'
        (HERE/name).write_text(json.dumps(dispatch,indent=2)+'\n')
        row['dispatch']=name;row['dispatch_sha256']=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
        print(json.dumps(dict(cap_kind=cap_kind,**{k:row[k]for k in ('cap','matched8_mean_pct','matched8_mean_request_delta','matched8_max_request_delta','selected_shapes','mip_gap')})),flush=True)
    frontier[cap_kind]=rows
out=dict(status='SHAPE_ONLY_TARIFF_FRONTIER_NO_TRUTH_OR_NUMERIC_RUN',options=len(options),shape_groups=len(groups),
         objective='mean8 normalized push saving + mean24 normalized read saving',frontier=frontier,
         source_sha=tail['source_sha'],seconds=time.monotonic()-t0,maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         limits=['Only warm wrapper-call proxy is capped, not native residual instructions or wall time.',
                 'Each selected shape has positive FLOP saving. No accuracy outcomes or truth used.',
                 'Read-only capture shapes absent from matched8 have no matched8 call coefficient; their costs remain reported over all24read.',
                 'The three alternative tail policies were priced at original best cuts, not cross-multiplied with other cuts. All four original parent-tail cutsets are eligible.',
                 'MILP gap/status included; no exhaustive optimality claim if time-limited.'])
(HERE/'FRONTIER_v1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(options=out['options'],shapes=out['shape_groups'],seconds=out['seconds'],maxrss=out['maxrss'])),flush=True)
