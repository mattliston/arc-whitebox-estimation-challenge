"""Paired NC scores against immediate Cost75 and graded Cost150 parents."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
D = HERE.parent
TRUTH = D.parent / 'public_g19_attribution_v1'
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    panel = read(HERE / 'panel/completed.json')
    assert panel['completed'] and len(panel['records']) == 14
    assert panel['candidate_sha256'] == read(HERE / 'build.json')['candidate_sha256']
    parent_panels = {key: read(D / f'{key}_panel_v1/completed.json') for key in ('cost75', 'cost150')}
    assert parent_panels['cost75']['candidate_sha256'] == read(HERE / 'build.json')['parent_sha256']
    assert parent_panels['cost150']['candidate_sha256'] == 'f06314111d7f41a7c1fbcb8d888a509d0e46222cd31b5ab227ec1e334e582a1f'
    parents = {key: {r['rank']: r['call'] for r in pp['records']} for key, pp in parent_panels.items()}
    joined = []
    for entry in panel['records']:
        rank = entry['rank']
        arms = {'candidate': entry['call'], **{key: calls[rank] for key, calls in parents.items()}}
        assert len({call['weights_sha256'] for call in arms.values()}) == 1
        for call in arms.values():
            assert call['includes_initialization'] and sha(call['output']) == call['output_sha256']
        joined.append(dict(rank=rank, arms=arms))
    with (HERE / 'panel_comparison_sealed.json').open('x') as stream:
        json.dump(dict(candidate_sha256=panel['candidate_sha256'], joined=joined), stream, indent=2)
    rows = []
    for entry in joined:
        rank = entry['rank']
        path = TRUTH / (f'fresh_{rank}_truth4M.npz' if rank >= 3016500 else f'mps_truth_rank{rank:04d}_indep4M.npz')
        with np.load(path) as file:
            truth = file['mc_post'][15].astype(np.float64)
            noise = float(np.mean(file['mc_se'][15].astype(np.float64)**2))
        row = dict(rank=rank, truth_sha256=sha(path), noise_mse=noise)
        for arm, call in entry['arms'].items():
            pred = np.load(call['output']).astype(np.float64)
            assert pred.shape == (16, 1024) and np.isfinite(pred).all() and (pred >= 0).all()
            mse = float(np.mean((pred[15] - truth)**2))
            row[arm] = dict(raw=mse, raw_nc=mse-noise, flops=call['flops'], multiplier=call['multiplier'],
                            adjusted=mse*call['multiplier'], adjusted_nc=(mse-noise)*call['multiplier'])
        row['adjusted_nc_ratio'] = row['candidate']['adjusted_nc']/row['cost75']['adjusted_nc']
        rows.append(row)
    groups = {}
    for name, group in [('public8', rows[:8]), ('fresh6', rows[8:]), ('all14', rows)]:
        aggregate = {arm: {key: float(np.mean([r[arm][key] for r in group]))
                           for key in ('raw', 'raw_nc', 'flops', 'multiplier', 'adjusted', 'adjusted_nc')}
                     for arm in ('cost75', 'cost150', 'candidate')}
        aggregate['adjusted_nc_ratio'] = aggregate['candidate']['adjusted_nc']/aggregate['cost75']['adjusted_nc']
        aggregate['graded_parent_nc_ratio'] = aggregate['candidate']['adjusted_nc']/aggregate['cost150']['adjusted_nc']
        aggregate['wins_against_cost75'] = sum(r['adjusted_nc_ratio'] < 1 for r in group)
        groups[name] = aggregate
    projection = 4.015535762022727e-9 * groups['all14']['graded_parent_nc_ratio']
    result = dict(rows=rows, groups=groups, official_bank_ratio_projection=projection,
                   projection_caveat='Existing-panel paired NC estimate against graded Cost150; not an official score or an untouched validation set.')
    result['both_panels_improve'] = all(groups[g]['adjusted_nc_ratio'] < 1 for g in ('public8', 'fresh6'))
    result['largest_individual_nc_ratio'] = max(r['adjusted_nc_ratio'] for r in rows)
    result['panel_gate_passed'] = result['both_panels_improve'] and result['largest_individual_nc_ratio'] <= 1.03
    with (HERE / 'panel_score.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(groups=groups, official_bank_ratio_projection=projection,
                          panel_gate_passed=result['panel_gate_passed'],
                          largest_individual_nc_ratio=result['largest_individual_nc_ratio'])), flush=True)


if __name__ == '__main__':
    main()
