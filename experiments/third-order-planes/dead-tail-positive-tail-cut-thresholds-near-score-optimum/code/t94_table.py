#!/usr/bin/env python3
"""TAG opus-physicist T94 table: tail-threshold arms against the fired thresholds (arm A), merged over run directories (the fired arm of
t93_tailscan is reused: same request, same worker, deterministic). Per arm: final-layer change pooled, mean log (se), networks lower,
all-layer change; mean rows cut per layer 7-14 (old tail, young tail, push dead) from the simulator's own row-cut logs.
usage: t94_table.py <run dir> [<run dir> ...]"""
import json, math, sys
from pathlib import Path
import numpy as np
R, C = {}, []
for d_ in map(Path, sys.argv[1:]):
    for f in sorted((d_ / 'results').glob('*.json')):
        d = json.loads(f.read_text()); _, s, a = d['id'].rsplit('_', 2); R.setdefault(int(s), {})[a] = (d['mse'], float(np.mean(d['mse_layers'])))
    if (d_ / 'cut_counts.jsonl').exists(): C += [json.loads(l) for l in (d_ / 'cut_counts.jsonl').read_text().splitlines()]
arms = sorted({a for r in R.values() for a in r})
for arm in arms:
    nets = sorted(s for s in R if 'A' in R[s] and arm in R[s])
    seeds = [s for s in nets]
    rows = {k: np.zeros(16) for k in ('old_tail', 'young_tail', 'dead')}; n = 0
    for s in seeds:
        rr = {r['layer']: r for r in C if r['tag'] == '%s_%d' % (arm, s)}
        if len(rr) < 15: continue
        n += 1
        for l, r in rr.items():
            for k in rows: rows[k][l] += r.get(k, 0)
    cnt = ' | '.join('%s %s' % (k, ' '.join('%.0f' % (rows[k][l] / max(n, 1)) for l in range(7, 15))) for k in rows)
    if arm == 'A':
        print('A (fired) on %d networks; mean rows cut per layer 7-14: %s' % (len(nets), cnt)); continue
    a = np.array([R[s][arm][0] for s in nets]); b = np.array([R[s]['A'][0] for s in nets]); lr = np.log(a / b)
    al = np.array([R[s][arm][1] for s in nets]); bl = np.array([R[s]['A'][1] for s in nets])
    print('%s vs A on %d: final %+.3f%% pooled, mean log %+.3f%% (se %.3f%%), lower on %d of %d; all-layer %+.3f%%; rows: %s' % (
        arm, len(nets), 100 * (a.mean() / b.mean() - 1), 100 * lr.mean(), 100 * lr.std(ddof=1) / math.sqrt(len(lr)), int((a < b).sum()), len(nets),
        100 * (al.mean() / bl.mean() - 1), cnt))
