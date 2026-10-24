"""Paired recomputation against the existing independent Monte Carlo panel."""
from common import (HERE, D, NETS, CANDIDATE_SHA, read, write, pin,
                    verify, verify_pin, verify_call)
import numpy as np

verify()
panel = read(HERE / 'panel/completed.json')
assert panel['completed'] and panel['candidate_sha256'] == CANDIDATE_SHA
assert tuple(r['rank'] for r in panel['records']) == NETS
comparisons = []
for entry in panel['records']:
    arrays = {}
    for name, call in [('candidate', entry['call']), ('parent', entry['parent'])]:
        verify_call(call)
        a = np.load(call['output'], allow_pickle=False)
        assert a.shape == (16,1024) and a.dtype == np.float32
        assert np.isfinite(a).all() and (a >= 0).all()
        arrays[name] = a
    for key in ('result', 'watch'):
        if entry[key] is not None:
            verify_pin(entry[key])
    prefix_exact = [arrays['candidate'][i].tobytes() == arrays['parent'][i].tobytes() for i in range(16)]
    delta = arrays['candidate'].astype(np.float64) - arrays['parent'].astype(np.float64)
    comparisons.append(dict(rank=entry['rank'], first_two_rows_exact=all(prefix_exact[:2]), per_layer_bytes_exact=prefix_exact,
        per_layer_delta_mse=np.mean(delta*delta, axis=1).tolist(),
        all_layer_maxabs=float(np.max(np.abs(delta)))))
write(HERE / 'predictions_sealed.json', dict(panel=pin(HERE/'panel/completed.json'),
    comparisons=comparisons, references_decoded=False))
prior = {r['rank']:r for r in read(D/'lazy_rotation_objective_v1/panel_score.json')['rows']}
rows = []
for entry, comparison in zip(panel['records'], comparisons):
    rank = entry['rank']
    verify_pin(entry['truth'])
    with np.load(entry['truth']['path'], allow_pickle=False) as z:
        truth = z['mc_post'][15].astype(np.float64)
        noise = float(np.mean(z['mc_se'][15].astype(np.float64)**2))
    row = dict(rank=rank, truth=entry['truth'], noise_mse=noise, comparison=comparison)
    for name, call in [('candidate', entry['call']), ('parent', entry['parent'])]:
        prediction = np.load(call['output'], allow_pickle=False)[15].astype(np.float64)
        raw = float(np.mean((prediction-truth)**2))
        row[name] = dict(raw=raw, raw_nc=raw-noise, flops=call['flops'],
            multiplier=call['multiplier'], adjusted=raw*call['multiplier'],
            adjusted_nc=(raw-noise)*call['multiplier'])
    for key in ('raw', 'raw_nc', 'adjusted', 'adjusted_nc', 'flops'):
        assert abs(row['parent'][key]-prior[rank]['candidate'][key]) < 1e-20, (rank,key)
    row['raw_nc_ratio'] = row['candidate']['raw_nc']/row['parent']['raw_nc']
    row['adjusted_nc_ratio'] = row['candidate']['adjusted_nc']/row['parent']['adjusted_nc']
    row['adjusted_ratio'] = row['candidate']['adjusted']/row['parent']['adjusted']
    rows.append(row)
groups = {}
for name, selected in [('development2', rows[:2]), ('new_public6', rows[2:8]),
                       ('public8', rows[:8]), ('historical_fresh6', rows[8:]),
                       ('new_calls12', rows[2:]), ('all14', rows)]:
    group = {arm:{key:float(np.mean([r[arm][key] for r in selected]))
                  for key in ('raw','raw_nc','flops','multiplier','adjusted','adjusted_nc')}
             for arm in ('candidate','parent')}
    group['adjusted_nc_ratio'] = group['candidate']['adjusted_nc']/group['parent']['adjusted_nc']
    group['raw_nc_ratio'] = group['candidate']['raw_nc']/group['parent']['raw_nc']
    group['wins'] = sum(r['adjusted_nc_ratio'] < 1 for r in selected)
    groups[name] = group
advance = (groups['all14']['adjusted_nc_ratio'] <= .99 and
           groups['public8']['adjusted_nc_ratio'] < 1 and
           groups['historical_fresh6']['adjusted_nc_ratio'] < 1 and
           max(r['adjusted_nc_ratio'] for r in rows) <= 1.10)
result = dict(completed=True, candidate_sha256=CANDIDATE_SHA, rows=rows, groups=groups,
    panel_advancement_passed=advance, old_development_gate_passed=False,
    parent_reproduced=True, official_score=False,
    caveat='Existing paired4M panel, reused research networks; NC subtracts estimated reference variance. No new samples or fit.')
write(HERE / 'panel_score.json', result)
print({k:v for k,v in result.items() if k != 'rows'}, flush=True)
