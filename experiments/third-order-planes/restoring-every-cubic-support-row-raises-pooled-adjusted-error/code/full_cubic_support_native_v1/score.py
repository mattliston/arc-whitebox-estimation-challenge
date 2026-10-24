import math
import resource
import numpy as np
from common import *

seal_check(); done('qualify'); assert read(HERE / 'qualification.json')['passed']
sealed = read(HERE / 'predictions_sealed.json')
check(sealed['candidate']); check(sealed['qualification'])
for p in sealed['result_records']:
    check(p)
rows = []
for pair in sealed['pairs']:
    for p in [pair['candidate']['prediction'], pair['parent_prediction'], pair['truth']]:
        check(p)
    c, p = pair['candidate'], pair['parent']
    ca = np.load(c['prediction']['path'], allow_pickle=False)
    pa = np.load(pair['parent_prediction']['path'], allow_pickle=False)
    with np.load(pair['truth']['path'], allow_pickle=False) as z:
        truth = z['mc_post'][-1].astype(np.float64)
    scores = {}
    for name, array, rec in [('candidate', ca, c), ('parent', pa, p)]:
        assert rec['includes_initialization'] and rec['multiplier'] == max(.1, rec['flops']/2**41)
        delta = array[-1].astype(np.float64) - truth
        raw = math.fsum(float(x)*float(x) for x in delta) / delta.size
        nr = float(np.mean(delta*delta)); assert abs(nr-raw) <= 1e-14*raw
        scores[name] = dict(raw=raw, adjusted=raw*rec['multiplier'], flops=rec['flops'])
    rows.append(dict(rank=pair['rank'], **scores,
        raw_ratio=scores['candidate']['raw']/scores['parent']['raw'], cost_ratio=c['flops']/p['flops'],
        adjusted_ratio=scores['candidate']['adjusted']/scores['parent']['adjusted'],
        max_parent_difference=float(np.max(np.abs(ca.astype(float)-pa.astype(float))))))
pooled = sum(r['candidate']['adjusted'] for r in rows) / sum(r['parent']['adjusted'] for r in rows)
passed = all(r['adjusted_ratio'] < 1 for r in rows) and pooled <= .99
score = dict(rows=rows, pooled_adjusted_ratio=pooled, passed=passed,
    gate='both adjusted ratios<1; pooled adjusted ratio<=.99',
    references='existing ordinary1B ranks35/100', pristine_holdout=False, official_score=False,
    fit_steps=0, new_samples=0, independent_loss_reduction_agreed=True,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
write(HERE / 'score.json', score); print(score, flush=True)

