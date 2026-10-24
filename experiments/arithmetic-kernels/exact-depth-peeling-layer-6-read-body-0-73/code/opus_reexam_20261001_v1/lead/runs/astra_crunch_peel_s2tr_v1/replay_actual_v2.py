"""Replay saved layer order with a conservative per-prediction request allowance."""
from pathlib import Path
import collections,hashlib,json
HERE=Path(__file__).resolve().parent;R=HERE.parent
x=json.loads((HERE/'TARIFF_v1.json').read_text())
wrapper_path=R/'astra_crunch_compute_v1/peel_sm40_v1/S2TR_WRAPPER_TARIFF_v1.json'
wrapper=json.loads(wrapper_path.read_text())
wb={(r['kind'],tuple(r['shape'])):r for r in wrapper['rows']}
for r in x['rows']:
    w=wb[r['kind'],tuple(r['shape'])]
    assert r['s2tr_saving']==w['gain']
    assert r['s2tr_price']['flops']==w['warm']['flops']
    r['recipe_request_delta']=r['s2tr_request_delta']
    r['s2tr_request_delta']=w['request_delta']
    r['s2tr_wrapper_warm']=w['warm']
    r['s2tr_wrapper_base']=w['base']
    r['s2tr_wrapper_cold']=w['cold']
table={(r['kind'],tuple(r['shape'])):r for r in x['rows']}
push={}
for path in (R/'opus_small_v1/shapes').glob('shapes_*.json'):
    z=json.loads(path.read_text());events=[]
    for i,r in enumerate(z['rows']):
        if r['entry']=='_mm'and r['site']=='_astra_bcc_newborn_impl'and r['b'][1]>1024:
            events.append(dict(kind='push',shape=[*r['a'],r['b'][1]],layer=r['layer'],order=i))
    push[z['seed']]=dict(denominator=z['total'],events=events,path=str(path))
read={}
for path in sorted((R/'astra_prefix_row_energy22_science_v1').glob('run_rpe_*_s*_v1/native_*_ON.json')):
    z=json.loads(path.read_text())['rows'][0];events=[]
    for i,r in enumerate(z['stats']['_RTT_STATS']['reads']):
        events.append(dict(kind='read',shape=[r['m'],r['k'],r['c15'],384 if r['old_rank']else 0],layer=r['layer'],order=i))
    assert z['seed']not in read
    read[z['seed']]=dict(denominator=z['flops'],events=events,path=str(path))


def replay(events,budget=750):
    spent=gain=net=0;accepted=[];skipped=[]
    for e in events:
        r=table.get((e['kind'],tuple(e['shape'])))
        if r is None or r['s2tr_saving']<=0:continue
        charge=max(0,r['s2tr_request_delta'])
        q=dict(e,charge=charge,saving=r['s2tr_saving'])
        if spent+charge>budget:skipped.append(q);continue
        spent+=charge;net+=r['s2tr_request_delta'];gain+=r['s2tr_saving'];accepted.append(q)
    assert net<=spent<=budget
    return dict(gain=gain,positive_calls_spent=spent,net_extra_calls=net,accepted=accepted,skipped=skipped)


read_rows=[];combined=[]
for seed,r in read.items():
    e=sorted(r['events'],key=lambda a:(a['layer'],a['order']))
    q=replay(e);read_rows.append(dict(seed=seed,**q,pct=100*q['gain']/r['denominator']))
    if seed in push:
        p=push[seed]
        # In the estimator loop inherited push precedes the CP read per layer.
        all_events=sorted(p['events']+r['events'],key=lambda a:(a['layer'],0 if a['kind']=='push'else 1,a['order']))
        q=replay(all_events);combined.append(dict(seed=seed,**q,pct=100*q['gain']/p['denominator']))
assert len(read_rows)==24 and len(combined)==8
summary=dict(matched8_mean_pct=sum(r['pct']for r in combined)/8,
             matched8_mean_net_extra_calls=sum(r['net_extra_calls']for r in combined)/8,
             matched8_max_net_extra_calls=max(r['net_extra_calls']for r in combined),
             matched8_max_positive_calls=max(r['positive_calls_spent']for r in combined),
             read24_mean_pct=sum(r['pct']for r in read_rows)/24,
             read24_max_net_extra_calls=max(r['net_extra_calls']for r in read_rows),
             read24_max_positive_calls=max(r['positive_calls_spent']for r in read_rows))
rows=[]
for r in x['rows']:
    q=dict(r,budget_cost=max(0,r['s2tr_request_delta']))
    rows.append(q)
out=dict(status='FROZEN_SHAPES_S2TR_750_ACTUAL_WRAPPER_POSITIVE_CALL_ALLOWANCE',wrapper_tariff_sha=hashlib.sha256(wrapper_path.read_bytes()).hexdigest(),source_sha=x['s2tr_source_sha'],body_source_sha=x['body_source_sha'],
         allowance=750,summary=summary,rows=rows,matched8=combined,read24=read_rows,
         runtime_rule='Reset remaining=750 per predict. For listed profitable shape, debit=max(0,s2tr_request_delta); if debit>remaining use exact original S2tr fallback, else subtract debit then run frozen recipe. Never credit negative deltas. Unlisted shapes always fallback.',
         guarantee='For any sequence of listed shapes, sum of net added WARM wrapper-operation proxies <=sum charged<=750. Unknown shapes add zero through exact fallback.',
         limits=['Bound is on measured warm API-call proxy, not native Python residual instructions, lazy setup cost, or wall time.',
                 'Replay uses existing separate push/read saved rosters, interleaved by layer with push first; no fresh trajectory.',
                 'Percent denominator is saved reference whole bill, not a new native S2tr measurement.'])
(HERE/'DEPLOY_S2TR_CAP750_v2.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(summary=summary,sha256=hashlib.sha256((HERE/'DEPLOY_S2TR_CAP750_v2.json').read_bytes()).hexdigest()),indent=2))
