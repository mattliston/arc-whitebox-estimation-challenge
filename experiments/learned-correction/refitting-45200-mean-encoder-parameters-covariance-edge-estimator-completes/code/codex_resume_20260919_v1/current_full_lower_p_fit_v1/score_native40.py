from endpoint_common import *
endpoint_verify()
done('seal_predictions')
sealed = read(HERE / 'native40_predictions_sealed.json')
assert sealed['completed'] and sealed['before_validation_targets'] and sealed['own_complete_bills'] and sealed['repeats_excluded']
for p in sealed['pins']:
    check(p)
inputs = read(HERE / 'inputs.json')
assert sealed['scored_ranks'] == inputs['valid']
roles = {k: read(v['result']['path'])['records'][:-1] for k, v in sealed['roles'].items()}
import numpy as np
assert np.__version__ == '2.4.6'
predictions = {k: [np.load(r['prediction']['path'], allow_pickle=False) for r in rs] for k, rs in roles.items()}
for index, rank in enumerate(inputs['valid']):
    a, b = predictions['parent'][index], predictions['trained'][index]
    assert a.shape == b.shape == (16, 1024) and a.dtype == b.dtype == np.float32
    assert a[0].tobytes() == b[0].tobytes()
rows = []
for index, rank in enumerate(inputs['valid']):
    tp = inputs['targets'][str(rank)]
    check(tp)
    with np.load(tp['path'], allow_pickle=False) as z:
        target = z['mc_post'].astype(np.float64)
    assert target.shape == (16, 1024) and np.isfinite(target).all()
    row = dict(rank=rank, target=tp, roles={})
    for role in ('parent', 'trained'):
        r = roles[role][index]
        assert r['rank'] == rank
        mse = np.mean((predictions[role][index].astype(np.float64) - target) ** 2, axis=1)
        assert np.isfinite(mse).all() and (mse >= 0).all()
        row['roles'][role] = dict(final_raw=float(mse[-1]), final_adjusted=float(mse[-1] * r['multiplier']),
            per_layer_raw=mse.tolist(), flops=r['flops'], multiplier=r['multiplier'], prediction=r['prediction'])
    row['adjusted_ratio'] = row['roles']['trained']['final_adjusted'] / row['roles']['parent']['final_adjusted']
    rows.append(row)
means = {role: {key: sum(r['roles'][role][key] for r in rows) / 40
    for key in ('final_raw', 'final_adjusted', 'flops')} for role in ('parent', 'trained')}
ratio = means['trained']['final_adjusted'] / means['parent']['final_adjusted']
wins = sum(r['adjusted_ratio'] < 1 for r in rows)
passed = ratio < 1 and wins >= 24
write(HERE / 'native40_score.json', dict(completed=True, integrity_passed=True, passed=passed,
    adjusted_ratio=ratio, raw_ratio=means['trained']['final_raw'] / means['parent']['final_raw'],
    adjusted_wins=wins, means=means, rows=rows, first_layer_byte_exact=True,
    all_predictions_and_bills_verified_before_validation_targets=True,
    prediction_seal=pin(HERE / 'native40_predictions_sealed.json'),
    native_score_measured=True, native_derivative_certified=False, reused_VALID40=True,
    advances_to_downstream_checks=passed, candidate=read(HERE / 'build.json')['candidate'],
    official_changed=False, goal_achieved=False))
print(dict(completed=True, passed=passed, adjusted_ratio=ratio, wins=wins, means=means), flush=True)
