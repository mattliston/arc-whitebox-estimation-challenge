#!/usr/bin/env python3
"""TAG opus-physicist T129 table: births floor RAISED at late layers (S50 frozen point, learned parts on), each arm paired per network with the base (floor 0.01):
final-layer MSE pooled change, mean log (se), networks lower, all-layer; births per layer 7-14 (mean over networks, base and arm) and the change in young
column-layers (sum over layers of all pushed minus old-tier columns), with a PROXY bill change = sum_l d(young columns)_l x the engineer's young price per
column at layer l (S47e census, Mflop: L8 1.40, L9 1.29, L10 1.24, L11 1.20, L12 1.15, L13 1.09, L14 1.13, L15 0.61) / 255 Gflop (the native factor for births
is about 2x more: the engineer prices natively). usage: t129_table.py"""
import json, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; R = HERE / 't129_floor_raise'
PRICE = {1: 1.47, 2: 1.67, 3: 1.74, 4: 1.75, 5: 1.71, 6: 1.63, 7: 1.50, 8: 1.40, 9: 1.29, 10: 1.24, 11: 1.20, 12: 1.15, 13: 1.09, 14: 1.13, 15: 0.61}
res = {}
for f in (R / 'results').glob('*.json'):
    d = json.loads(f.read_text()); _, s, arm = d['id'].split('_', 2); res.setdefault(int(s), {})[arm] = np.array(d['mse_layers'], float)
sel = {}; push = {}
for line in (R / 'sel_counts.jsonl').read_text().splitlines() if (R / 'sel_counts.jsonl').exists() else []:
    d = json.loads(line); arm, s = d['tag'].rsplit('_', 1); sel.setdefault(arm, {})[int(s)] = {int(k): v.get('births', 0) for k, v in d['counts'].items()}
for line in (R / 'push_counts.jsonl').read_text().splitlines() if (R / 'push_counts.jsonl').exists() else []:
    d = json.loads(line); arm, s = d['tag'].rsplit('_', 1)
    push.setdefault(arm, {})[int(s)] = {int(k): v['cols'] - v['old'] for k, v in d['counts'].items()}
for arm in ('b02_8_14', 'b03_8_14', 'b05_8_14', 'b03_11_14'):
    nets = sorted(s for s in res if 'base' in res[s] and arm in res[s])
    if len(nets) < 2:
        continue
    a = np.array([res[s][arm] for s in nets]); b = np.array([res[s]['base'] for s in nets]); lr = np.log(a[:, -1] / b[:, -1])
    line = '%-10s n %2d: final %+.3f%% pooled, mean log %+.3f%% (se %.3f%%), lower on %d of %d; all-layer %+.3f%%' % (
        arm, len(nets), 100 * (a[:, -1].mean() / b[:, -1].mean() - 1), 100 * lr.mean(), 100 * lr.std(ddof=1) / math.sqrt(len(lr)), int((a[:, -1] < b[:, -1]).sum()),
        len(nets), 100 * (a.mean(1).mean() / b.mean(1).mean() - 1))
    pn = sorted(s for s in nets if s in push.get(arm, {}) and s in push.get('base', {}))
    if pn:
        dy = {l: np.mean([push[arm][s].get(l, 0) - push['base'][s].get(l, 0) for s in pn]) for l in range(1, 16)}
        yb = np.mean([sum(push['base'][s].get(l, 0) for l in range(1, 16)) for s in pn])
        bill = sum(dy[l] * PRICE[l] for l in dy) / 255e3
        line += ' | young column-layers %+.0f (%+.2f%% of base %.0f), proxy bill %+.2f%%' % (sum(dy.values()), 100 * sum(dy.values()) / yb, yb, 100 * bill)
    sn = sorted(s for s in nets if s in sel.get(arm, {}) and s in sel.get('base', {}))
    if sn:
        line += ' | births L7-14 base %s arm %s' % (' '.join('%.0f' % np.mean([sel['base'][s].get(l, 0) for s in sn]) for l in range(7, 15)),
                                                    ' '.join('%.0f' % np.mean([sel[arm][s].get(l, 0) for s in sn]) for l in range(7, 15)))
    print(line)
