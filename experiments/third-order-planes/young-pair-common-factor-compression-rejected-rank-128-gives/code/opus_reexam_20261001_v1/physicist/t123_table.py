#!/usr/bin/env python3
"""TAG opus-physicist T123 table: each pair-split arm against the plain card's base, final-layer MSE pooled / mean log (se) / networks lower / all-layer, per
layer 5-15; and per layer 8-14 the mean c energy kept and (mode A) the young plane's relative change (mean over networks).
usage: t123_table.py"""
import json, math
import numpy as np
from t123_common import HERE, arm, base
b = base()
for name in ('a128', 'n128', 'a256', 'a384', 'a512', 'b384', 'b512', 'r32', 'r64'):
    a = arm(name); nets = sorted(s for s in a if s in b)
    if len(nets) < 2:
        continue
    A = np.array([a[s] for s in nets]); Bm = np.array([b[s] for s in nets]); lr = np.log(A[:, -1] / Bm[:, -1])
    print('%-5s on %2d: final %+.3f%% pooled, mean log %+.3f%% (se %.3f%%), lower on %2d; all-layer %+.3f%% | L5-15: %s' % (
        name, len(nets), 100 * (A[:, -1].mean() / Bm[:, -1].mean() - 1), 100 * lr.mean(), 100 * lr.std(ddof=1) / math.sqrt(len(lr)), int((A[:, -1] < Bm[:, -1]).sum()),
        100 * (A.mean(1).mean() / Bm.mean(1).mean() - 1), ' '.join('%+.2f' % (100 * (A[:, l].mean() / Bm[:, l].mean() - 1)) for l in range(5, 16))))
    log = HERE / ('t123_' + name) / 't123_log.jsonl'
    if log.exists():
        rows = [json.loads(x)['layers'] for x in log.read_text().splitlines()]
        for key in ('c_kept', 'c_kept_scheme', 'c_kept_oracle', 'dplane_rel', 'k', 'rows_check'):
            vals = {l: [r[l][key] for r in rows if l in r and key in r[l]] for l in map(str, range(8, 15))}
            if any(vals.values()):
                print('      %-10s L8-14: %s' % (key, ' '.join('%.6f' % np.mean(v) if v else '-' for v in vals.values())))
