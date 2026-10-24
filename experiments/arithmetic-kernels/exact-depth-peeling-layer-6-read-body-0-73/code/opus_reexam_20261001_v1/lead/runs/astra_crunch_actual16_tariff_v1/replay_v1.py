"""Price recorded frozen16 routes; no model, new geometry, or target fitting."""
from pathlib import Path
import hashlib
import json
import statistics

D=Path(__file__).resolve().parent
R=D.parent
S=R/'astra_crunch_s2tr_fullstack_v1'
science_path=S/'FIXED16_SCIENCE_v1.json'
tariff_path=R/'astra_crunch_compute_v1/peel_sm40_v1/S2TR_WRAPPER_TARIFF_v1.json'
science=json.loads(science_path.read_text())
tariff=json.loads(tariff_path.read_text())
by={(x['kind'],tuple(x['shape'])):x for x in tariff['rows']}
W3=355273664
rows=[]
for scientific in science['rows']:
    rank=scientific['rank']
    if rank in (16,35,150,200,250,259,277,287):
        path=S/f'res_{rank}_v1.json'
        result=json.loads(path.read_text())
        peel=result['peel']
        w3=result['w3']['calls']
        assert result['source_sha']==science['source_sha']
    elif rank==293:
        path=R/f'astra_crunch_compute_v1/peel_fullstack8_v1/res_{rank}_v1.json'
        result=json.loads(path.read_text())
        peel=result['peel']
        w3=result['w3']['calls']
        assert result['source_sha']==science['source_sha']
    else:
        path=R/f'astra_crunch_compute_v1/peel_boundary_v1/science_{rank}_v1/rows.jsonl'
        result=json.loads(path.read_text().splitlines()[0])
        peel=result['peel_stats']
        w3=result['w3_delta']
        manifest=json.loads((path.parent/'manifest.json').read_text())
        assert manifest['sha256']==science['source_sha']
    assert w3==15
    prices=[]
    for kind in ('push','read'):
        shapes=peel[kind+'_shapes']
        assert len(shapes)==peel[kind]
        for shape in shapes:
            t=by[(kind,tuple(shape))]
            assert t['gain']==t['base']['flops']-t['warm']['flops']
            prices.append(dict(kind=kind,shape=shape,base=t['base']['flops'],warm=t['warm']['flops'],
                               gain=t['gain'],request_delta=t['request_delta']))
    peel_gain=sum(p['gain'] for p in prices)
    total=peel_gain+W3
    denominator=scientific['control_cold_flops']
    rows.append(dict(rank=rank,source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        push_calls=peel['push'],read_calls=peel['read'],prices=prices,
        peel_warm_gain=peel_gain,w3_warm_gain=W3,total_warm_gain=total,
        control_cold_reference_flops=denominator,
        estimated_warm_candidate_flops_using_cold_reference=denominator-total,
        peel_gain_pct_of_cold_reference=100*peel_gain/denominator,
        total_gain_pct_of_cold_reference=100*total/denominator,
        peel_warm_request_delta=sum(p['request_delta']for p in prices)))
den=sum(r['control_cold_reference_flops']for r in rows)
out=dict(science_source_sha256=science['source_sha'],models=0,n=16,rows=rows,
    input_artifacts={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in (science_path,tariff_path)},
    peel_mean_absolute_gain=statistics.mean(r['peel_warm_gain']for r in rows),
    w3_absolute_gain_per_network=W3,
    total_mean_absolute_gain=statistics.mean(r['total_warm_gain']for r in rows),
    mean_peel_gain_pct_of_cold_reference=statistics.mean(r['peel_gain_pct_of_cold_reference']for r in rows),
    mean_total_gain_pct_of_cold_reference=statistics.mean(r['total_gain_pct_of_cold_reference']for r in rows),
    pooled_peel_gain_pct_of_cold_reference=100*sum(r['peel_warm_gain']for r in rows)/den,
    pooled_total_gain_pct_of_cold_reference=100*sum(r['total_warm_gain']for r in rows)/den,
    mean_peel_request_delta=statistics.mean(r['peel_warm_request_delta']for r in rows),
    cold_prebind='Excluded from gains. No setup alias/frame/cache initialization charges included. Setup-limited first-model prices are separate.',
    denominator='Each network uses its saved S2tr native cold reference bill only to normalize exact warm-operation savings. Same-rank S2tr warm bills are unavailable; percentages are denominator approximations, not measured warm bill ratios.',
    overlap='Only recorded accepted peel routes priced. W3 is fixed per-network saving with15calls; no ZB4 arithmetic credit. No optional P40 gain included.',
    prior_historic_peel_mean_pct=0.671213199754,
    correction_to_historic_peel_pp=statistics.mean(r['peel_gain_pct_of_cold_reference']for r in rows)-0.671213199754)
P=R/'astra_crunch_smallsm_residual_v1'
small=json.loads((P/'FIXED8_RESULT_v1.json').read_text())
smalltariff=json.loads((P/'TARIFF_v1.json').read_text())
lo=min(r['gain']for r in smalltariff['rows']);hi=max(r['gain']for r in smalltariff['rows'])
optional=[]
for r in small['rows']:
    optional.append(dict(rank=r['rank'],actual_route_count=r['routes'],shapes_logged=False,
        warm_gain_lower=r['routes']*lo,warm_gain_upper=r['routes']*hi,
        lower_pct_of_full48c_cold_reference=100*r['routes']*lo/r['control_flops'],
        upper_pct_of_full48c_cold_reference=100*r['routes']*hi/r['control_flops']))
out['optional_P40_actual8']=dict(rows=optional,route_count=sum(r['actual_route_count']for r in optional),
    mean_routes=statistics.mean(r['actual_route_count']for r in optional),
    historic_mean_routes=statistics.mean(r['hits']for r in smalltariff['coverage']),
    exact_gain_identifiable=False,
    mean_gain_pct_lower=statistics.mean(r['lower_pct_of_full48c_cold_reference']for r in optional),
    mean_gain_pct_upper=statistics.mean(r['upper_pct_of_full48c_cold_reference']for r in optional),
    note='Existing optional boundary diagnostics record successful route counts only, not shapes. Bounds use min/max fixed12 per-call tariff; no unsupported exact point estimate or new model.')
(D/'ACTUAL16_WARM_TARIFF_v1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('rows','input_artifacts','optional_P40_actual8')},indent=2))
print(json.dumps({k:v for k,v in out['optional_P40_actual8'].items()if k!='rows'},indent=2))
