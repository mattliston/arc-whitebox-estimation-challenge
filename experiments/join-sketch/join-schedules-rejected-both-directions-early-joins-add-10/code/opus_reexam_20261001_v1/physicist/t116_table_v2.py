#!/usr/bin/env python3
"""TAG opus-physicist T116 table v2 = t116_table.py + the arm 'nolast' (7/10/-, no third join), each arm's per-layer pooled MSE change against the base
(layers 5-15) and, for runs of worker v2, each join's mean captured energy fraction of the joined covariance at ranks 256 / 320 / 352 / 384 (the card's
own join log). Every arm is paired per seed with the base 7/10/13 (plain torch card, S47e / Nd learned parts fixed, one-direction star, legacy truths).
usage: t116_table_v2.py"""
import json, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ARMS = ('early', 'nolast', 'j7_11', 'j8_11', 'late', 'latelast', 'late71114', 'fullrank', 'dumpq0', 'sketchw2', 'sketchw1', 'pcopy', 'pairbase', 'pairk3', 'colk3', 'colk3p3')
R, CNT, JL = {}, {}, {}
for name in ('base',) + ARMS:
    run = HERE / ('t115_' + name)
    if not run.exists():
        continue
    for f in (run / 'results').glob('*.json'):
        d = json.loads(f.read_text()); s = int(d['id'].split('_')[1]); R.setdefault(s, {})[name] = np.array(d['mse_layers'], float)
    if (run / 'push_counts.jsonl').exists():
        rows = [json.loads(l) for l in (run / 'push_counts.jsonl').read_text().splitlines()]
        CNT[name] = np.array([[(r['counts'].get(str(l), {}).get('cols', 0), r['counts'].get(str(l), {}).get('old', 0)) for l in range(16)] for r in rows], dtype=float)
        for r in rows:
            for j in r.get('joinlog') or []:
                JL.setdefault(name, {}).setdefault(j['layer'], []).append(j)
for name in ARMS:
    nets = sorted(s for s in R if 'base' in R[s] and name in R[s])
    if len(nets) < 2:
        continue
    a = np.array([R[s][name] for s in nets]); b = np.array([R[s]['base'] for s in nets]); lr = np.log(a[:, -1] / b[:, -1])
    print('%-9s vs base on %d: final %+.3f%% pooled, mean log %+.3f%% (se %.3f%%), lower on %d of %d; all-layer %+.3f%%' % (
        name, len(nets), 100 * (a[:, -1].mean() / b[:, -1].mean() - 1), 100 * lr.mean(), 100 * lr.std(ddof=1) / math.sqrt(len(lr)),
        int((a[:, -1] < b[:, -1]).sum()), len(nets), 100 * (a.mean(1).mean() / b.mean(1).mean() - 1)))
    print('          per layer 5-15 pooled %: ' + ' '.join('L%d %+.2f' % (l, 100 * (a[:, l].mean() / b[:, l].mean() - 1)) for l in range(5, 16)))
for name, c in CNT.items():
    m = c.mean(0)
    print('%-9s (%d networks) young columns per layer 1-15: %s | young column-layers total %.0f' % (
        name, len(c), ' '.join('%.0f' % x for x in m[1:, 0] - m[1:, 1]), (m[1:, 0] - m[1:, 1]).sum()))
for name, layers in JL.items():
    for layer, js in sorted(layers.items()):
        E = {k: np.mean([j['E'][k] for j in js]) for k in ('256', '320', '352', '384')}
        print('%-9s join at L%d (%d networks): rank kept %d, joined columns %.0f, live units %.0f, captured energy at 256 / 320 / 352 / 384 = %s' % (
            name, layer, len(js), js[0]['r'], np.mean([j['joined'] for j in js]), np.mean([j['live'] for j in js]),
            ' / '.join('%.4f' % E[k] for k in ('256', '320', '352', '384'))))
