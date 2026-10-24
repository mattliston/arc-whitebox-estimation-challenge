"""Only decode targets after the entire fixed roster and own bills are sealed."""
import argparse
from common import *
ap = argparse.ArgumentParser()
ap.add_argument('stage',choices=['panel','confirmation','independent'])
stage = ap.parse_args().stage
stage_ready(stage)
clean_watch(HERE/f'seal_{stage}_watch.json')
sealed = read(HERE/f'{stage}_predictions_sealed.json')
assert sealed['completed'] and sealed['stage'] == stage and sealed['all_predictions_and_bills_before_targets']
assert sealed['own_complete_bills'] and sealed['repeats_excluded']
for p in sealed['pins']:
    check(p)
meta = read(HERE/'inputs.json')
ranks = INDEPENDENT if stage == 'independent' else PANEL if stage == 'panel' else CONFIRM
assert sealed['ranks'] == ranks
import numpy as np
assert np.__version__ == '2.4.6'
predictions = {role:[np.load(r['prediction']['path'],allow_pickle=False) for r in rs]
               for role,rs in sealed['roles'].items()}
for index,rank in enumerate(ranks):
    a,b = predictions['parent'][index],predictions['trained'][index]
    assert a.shape == b.shape == (16,1024) and a.dtype == b.dtype == np.float32
    assert np.isfinite(a).all() and np.isfinite(b).all() and (a >= 0).all() and (b >= 0).all()
    assert a[0].tobytes() == b[0].tobytes(), rank
rows = []
for index,rank in enumerate(ranks):
    tp = meta['independent_truths'][str(rank)] if stage == 'independent' else meta['baselines'][stage][index]['truth']
    check(tp)
    with np.load(tp['path'],allow_pickle=False) as z:
        target = z['mc_post'][15].astype(np.float64)
        noise = 0.
        if stage == 'independent':
            se = z['mc_se'][15].astype(np.float64)
            assert se.shape == (1024,) and np.isfinite(se).all() and (se >= 0).all()
            noise = float(np.mean(se**2))
        elif 'P' in z:
            assert int(z['P']) == 1_000_000_000
    assert target.shape == (1024,) and np.isfinite(target).all()
    row = dict(rank=rank,target=tp,mc_se_squared_mean=noise,roles={})
    for role in ['parent','trained']:
        r = sealed['roles'][role][index]
        assert r['rank'] == rank and r['multiplier'] == max(.1,r['flops']/2**41)
        raw = float(np.mean((predictions[role][index][15].astype(np.float64)-target)**2))
        assert np.isfinite(raw) and raw >= 0
        row['roles'][role] = dict(raw_observed=raw,raw_noise_corrected=raw-noise,
            adjusted_observed=raw*r['multiplier'],adjusted_noise_corrected=(raw-noise)*r['multiplier'],
            flops=r['flops'],multiplier=r['multiplier'],prediction=r['prediction'])
    parent = row['roles']['parent']['adjusted_observed']
    assert parent > 0
    row['adjusted_observed_ratio'] = row['roles']['trained']['adjusted_observed']/parent
    rows.append(row)
fields = ['raw_observed','raw_noise_corrected','adjusted_observed','adjusted_noise_corrected','flops']
groups = [('public8',PUBLIC),('virgin6',VIRGIN),('combined14',INDEPENDENT)] if stage == 'independent' else [(stage,ranks)]
summaries = {}
for name,roster in groups:
    selected = [r for r in rows if r['rank'] in roster]
    assert [r['rank'] for r in selected] == roster
    means = {role:{key:sum(r['roles'][role][key] for r in selected)/len(selected) for key in fields}
             for role in ['parent','trained']}
    denom = means['parent']['adjusted_noise_corrected']
    summaries[name] = dict(means=means,
        adjusted_observed_ratio=means['trained']['adjusted_observed']/means['parent']['adjusted_observed'],
        adjusted_noise_corrected_ratio=means['trained']['adjusted_noise_corrected']/denom if denom != 0 else None,
        adjusted_noise_corrected_difference=means['trained']['adjusted_noise_corrected']-denom,
        observed_adjusted_wins=sum(r['adjusted_observed_ratio'] < 1 for r in selected),
        worst_observed_adjusted_ratio=max(r['adjusted_observed_ratio'] for r in selected),
        strict_noise_corrected_gain=means['trained']['adjusted_noise_corrected'] < denom)
if stage == 'independent':
    passed = all(summaries[name]['strict_noise_corrected_gain'] for name in ['public8','virgin6'])
    gate = dict(strict_noise_corrected_adjusted_improvement_in_both_groups=True,unclipped=True)
else:
    s = summaries[stage]
    minimum_wins = 8 if stage == 'panel' else 5
    passed = s['adjusted_observed_ratio'] < 1 and s['observed_adjusted_wins'] >= minimum_wins and s['worst_observed_adjusted_ratio'] <= 1.10
    gate = dict(adjusted_pooled_ratio_strictly_below=1,minimum_wins=minimum_wins,worst_ratio_at_most=1.10)
write(HERE/f'{stage}_score.json',dict(completed=True,integrity_passed=True,passed=passed,stage=stage,
    summaries=summaries,rows=rows,gate=gate,first_layer_byte_exact=True,
    all_predictions_and_bills_verified_before_target_arrays=True,prediction_seal=pin(HERE/f'{stage}_predictions_sealed.json'),
    overlap=meta['overlap'],pristine_model_holdout=False,candidate=admission()['candidate'],
    official_changed=False,goal_achieved=False,no_refit=True))
print(dict(completed=True,stage=stage,passed=passed,summaries=summaries),flush=True)
