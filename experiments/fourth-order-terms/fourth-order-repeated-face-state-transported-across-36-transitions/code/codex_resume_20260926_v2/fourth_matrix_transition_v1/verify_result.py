"""Independent scalar and provenance replay after the fixed experiment."""
import math
from common import *

verify()
r=read(HERE/'result.json');u=read(HERE/'unit_result.json')
assert r['seal_sha256']==u['seal_sha256']==sha(HERE/'seal.json') and u['passed']
guards=[read(HERE/n) for n in ('unit_guard.json','run_guard.json')]
assert all(g['completed'] and g['whole_group_retired'] and not g['remaining_rows'] and g['max_swap_growth_bytes']==0 for g in guards)
for p in r['external_inputs']:check(p)
rows=r['rows'];logged=[json.loads(line) for line in (HERE/'cells.jsonl').read_text().splitlines()]
assert rows==logged and len(rows)==36
assert sorted((x['ordinal'],x['source_layer'],x['target_layer']) for x in rows)==sorted((i,l,l+1) for i in TRAIN for l in LAYERS)
differences=[];pooled={};nets={}
for group in rows[0]['metrics']:
    pooled[group]={};nets[group]=[]
    for metric in rows[0]['metrics'][group]:
        layer=[]
        for l in LAYERS:
            rr=[x['metrics'][group][metric] for x in rows if x['source_layer']==l]
            layer.append(math.fsum(x['residual_energy'] for x in rr)/math.fsum(x['baseline_energy'] for x in rr))
            for x in rr:
                calc=x['baseline_energy']+x['prediction_energy']-2*x['alignment']
                assert abs(calc-x['residual_energy'])<1e-10*max(x['baseline_energy'],x['residual_energy'])
        ratio=math.fsum(layer)/6
        pooled[group][metric]=ratio;differences.append(abs(ratio-r['pooled'][group][metric]))
    for ordinal in TRAIN:
        metrics={}
        for metric in pooled[group]:
            a=[];b=[]
            for l in LAYERS:
                base=math.fsum(x['metrics'][group][metric]['baseline_energy'] for x in rows if x['source_layer']==l)/6
                x=next(x for x in rows if x['ordinal']==ordinal and x['source_layer']==l)['metrics'][group][metric]
                a.append(x['residual_energy']/base);b.append(x['baseline_energy']/base)
            metrics[metric]=math.fsum(a)/math.fsum(b)
        original=next(x for x in r['networks'][group] if x['ordinal']==ordinal)['ratios']
        differences.extend(abs(v-original[k]) for k,v in metrics.items())
        nets[group].append(dict(ordinal=ordinal,ratios=metrics))
primary=('covariance','next_total_mean')
gates=dict(coupled=all(pooled['full_coupled'][k]<=.85 for k in primary),each_network=all(all(x['ratios'][k]<=.95 for k in primary) for x in nets['full_coupled']),fourth=all(pooled['fourth_fourth'][k]<=.80 for k in primary))
assert gates==r['gates'] and all(gates.values())==r['passed'] and max(differences)<1e-12
assert all(x['query_witness_max_abs']<=2e-11 and x['finite_consumer_max_abs']<=2e-11 for x in rows)
once(HERE/'verification.json',dict(passed=True,scientific_gates=gates,study_passed=r['passed'],maximum_aggregation_difference=max(differences),query_witness_max_abs=max(x['query_witness_max_abs'] for x in rows),finite_consumer_max_abs=max(x['finite_consumer_max_abs'] for x in rows),external_inputs_verified=len(r['external_inputs']),result_sha256=sha(HERE/'result.json'),verifier=pin(Path(__file__))))
print(json.dumps(dict(verified=True,study_passed=r['passed'],gates=gates,pooled=pooled,maximum_aggregation_difference=max(differences))))
