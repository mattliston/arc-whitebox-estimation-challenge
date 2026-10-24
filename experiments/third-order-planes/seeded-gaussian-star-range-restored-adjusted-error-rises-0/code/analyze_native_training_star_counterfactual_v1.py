"""Fixed eight-network native star-range comparison; no fitting or tuning."""
from pathlib import Path
import hashlib
import json

import numpy as np

HERE = Path(__file__).resolve().parent
protocol_path = HERE/'native_training_star_counterfactual_protocol_v1.json'
protocol = json.loads(protocol_path.read_text())
build = json.loads((HERE/'native_training_star_counterfactual_build_v1.json').read_text())
assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == build['protocol_sha256']
assert hashlib.sha256((HERE/protocol['parent']).read_bytes()).hexdigest() == protocol['parent_sha256']
assert hashlib.sha256((HERE/build['candidate']).read_bytes()).hexdigest() == build['candidate_sha256']
tags = ['training_star_counterfactual_train2_v1','training_star_counterfactual_train6_v1']
rows, guards = [], []
e_all, d_all = [], []
for candidate_tag, control_tag, ranks in zip(tags, protocol['controls'], protocol['groups']):
    guard = json.loads((HERE/f'{candidate_tag}_guard.json').read_text())
    assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes'] == 0
    guards.append(guard)
    manifest = json.loads((HERE/candidate_tag/'manifest.json').read_text())
    assert manifest['sha256'] == build['candidate_sha256'] and manifest['ranks'] == ranks
    ca = [json.loads(s) for s in (HERE/candidate_tag/'rows.jsonl').read_text().splitlines()]
    co = [json.loads(s) for s in (HERE/control_tag/'rows.jsonl').read_text().splitlines()]
    assert [r['rank'] for r in ca] == [r['rank'] for r in co] == ranks
    for i, (a,b) in enumerate(zip(ca,co)):
        rank = a['rank']
        assert a['random_range_star_calls'] == 15*(i+1)
        y = np.load(HERE/candidate_tag/f'pred_{rank}.npy')
        parent = np.load(HERE/control_tag/f'pred_{rank}.npy')
        with np.load(HERE.parent/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz') as f:
            truth = f['mc_post'][15].astype(np.float64)
        e, d = parent[15]-truth, y[15]-parent[15]
        E, D, ED = map(float, (np.mean(e*e),np.mean(d*d),np.mean(e*d)))
        assert np.isclose(E,b['mse'],rtol=1e-12,atol=0)
        assert abs((E+2*ED+D)-a['mse']) < 1e-20
        e_all.append(e); d_all.append(d)
        rows.append(dict(rank=rank,base=b,counterfactual=a,
                         final_difference_mse=D,final_difference_energy_over_parent_error=D/E,
                         final_residual_difference_cosine=ED/np.sqrt(E*D),
                         final_cross_term_over_parent_error=2*ED/E,
                         per_layer_rms_difference=np.sqrt(np.mean((y-parent)**2,axis=1)).tolist(),
                         raw_pct=100*(a['mse']/b['mse']-1),
                         adjusted_pct=100*(a['adjusted']/b['adjusted']-1),
                         bill_pct=100*(a['flops']/b['flops']-1)))
assert [row['rank'] for row in rows] == protocol['ranks']
base = {key:np.asarray([row['base'][key] for row in rows]) for key in ['mse','adjusted','flops','cpu_s']}
case = {key:np.asarray([row['counterfactual'][key] for row in rows]) for key in base}
summary = {key+'_pct':float(100*(case[key].mean()/base[key].mean()-1)) for key in base}
summary['raw_wins'] = int(np.count_nonzero(case['mse'] < base['mse']))
summary['adjusted_wins'] = int(np.count_nonzero(case['adjusted'] < base['adjusted']))
rng = np.random.default_rng(20260927)
ix = rng.integers(0,len(rows),size=(50000,len(rows)))
boot = case['adjusted'][ix].mean(axis=1)/base['adjusted'][ix].mean(axis=1)-1
summary['adjusted_ci95_pct'] = (100*np.quantile(boot,[.025,.975])).tolist()
e, d = np.concatenate(e_all), np.concatenate(d_all)
E, D, ED = map(float,(np.mean(e*e),np.mean(d*d),np.mean(e*d)))
summary['difference_energy_over_parent_error'] = D/E
summary['difference_rms_over_parent_error_rms'] = float(np.sqrt(D/E))
summary['residual_difference_cosine'] = float(ED/np.sqrt(E*D))
summary['cross_term_over_parent_error'] = 2*ED/E
passed = summary['adjusted_pct'] <= -2 and summary['adjusted_wins'] >= 6 and summary['adjusted_ci95_pct'][1] < 0
report = dict(protocol_sha256=build['protocol_sha256'],candidate_sha256=build['candidate_sha256'],
              parent_sha256=protocol['parent_sha256'],summary=summary,rows=rows,guards=guards,
              direct_candidate_gate_passed=passed,
              scope=protocol['scope'],training_interpretation=protocol['training_interpretation'],
              fitted_parameters=False,validation_accessed=False)
with (HERE/'native_training_star_counterfactual_result_v1.json').open('x') as fp:
    json.dump(report,fp,indent=2,allow_nan=False)
print(json.dumps(dict(summary=summary,direct_candidate_gate_passed=passed),indent=2))
