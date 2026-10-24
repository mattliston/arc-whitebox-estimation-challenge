"""Pure Python historical ordered route/cap replay; no estimator import."""
from pathlib import Path
import json
D=Path(__file__).resolve().parent;R=D.parent
frontier=json.loads((D/'PEEL_DEUNION_FRONTIER_v1.json').read_text())
choice=next(x for x in frontier['frontier']if x['budget']==600)
by={(r['kind'],tuple(r['shape'])):r for r in choice['shapes']}
u=json.loads((R/'astra_crunch_compute_v1/small_stack_v25a_v1/UNION_PRICE_v2.json').read_text())
reads={r['seed']:r for r in json.loads((R/'astra_crunch_compute_v1/bini_apa_v1/READ_TARIFF_v2.json').read_text())['coverage']}
slow={('push',(1024,992,8064)),('push',(1024,992,8128)),('read',(1024,4096,960,0)),('read',(992,6144,928,0)),('read',(992,6144,896,0))}
out=[]
for net in u['networks']:
    seed=net['seed'];pairs=net['pairs'];rh=reads[seed]['hits'];assert len(pairs)==len(rh)==14
    events=[]
    for layer,(p,r) in enumerate(zip(pairs,rh),1):
        assert p['marker_layer']==layer
        for kind,shape in [('push',p['shape']),('read',r)]:
            price=by.get((kind,tuple(shape)))
            if price:
                events.append(dict(layer=layer,kind=kind,shape=shape,gain=price['gain'],request_delta=price['request_delta'],slow=(kind,tuple(shape[:3]if kind=='push'else shape))in slow))
    variants={}
    for tag,clamp in [('signed_credit',False),('candidate_nonnegative',True)]:
        left=600;accepted=[];rejected=[]
        for e in events:
            charge=max(0,e['request_delta'])if clamp else e['request_delta']
            if charge<=left:
                left-=charge;accepted.append(dict(e,cap_charge=charge))
            else:rejected.append(e)
        variants[tag]=dict(accepted=accepted,rejected=rejected,left=left,cap_spent=600-left,gain=sum(e['gain']for e in accepted),actual_request_delta=sum(e['request_delta']for e in accepted),slow_count=sum(e['slow']for e in accepted))
    assert variants['signed_credit']['gain']==sum(e['gain']for e in events)
    row=dict(seed=seed,denominator=net['denominator'],events=events,variants=variants)
    row['candidate_gain_pct']=100*variants['candidate_nonnegative']['gain']/net['denominator']
    row['clamp_loss_flops']=variants['signed_credit']['gain']-variants['candidate_nonnegative']['gain']
    out.append(row)
receipt=dict(status='PURE_PYTHON_ORDERED_HISTORICAL_REPLAY',models=0,rows=out,mean_gain_pct=sum(r['candidate_gain_pct']for r in out)/8,total_clamp_loss_flops=sum(r['clamp_loss_flops']for r in out),max_candidate_cap_spend=max(r['variants']['candidate_nonnegative']['cap_spent']for r in out),max_actual_request_delta=max(r['variants']['candidate_nonnegative']['actual_request_delta']for r in out),slow_counts=[r['variants']['candidate_nonnegative']['slow_count']for r in out],limitations=['Historical eight-roster sequence, not current 32-rank coverage.','Reads ordered one per layer1–14, paired with recorded push marker_layer.','Warm-wrapper pricing; no wall-time bound.'])
(D/'PEEL600_ORDERED_ROLLOUT_v1.json').open('x').write(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items()if k!='rows'}))
for r in out:print(json.dumps(dict(seed=r['seed'],gain_pct=r['candidate_gain_pct'],clamp_loss=r['clamp_loss_flops'],cap_spent=r['variants']['candidate_nonnegative']['cap_spent'],request_delta=r['variants']['candidate_nonnegative']['actual_request_delta'],slow=[e for e in r['variants']['candidate_nonnegative']['accepted']if e['slow']],rejected=r['variants']['candidate_nonnegative']['rejected'])))
