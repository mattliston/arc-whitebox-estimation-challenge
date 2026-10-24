"""TAG opus-trainer (2026-10-02): paired comparisons between result labels on the judging ranks they share (final-layer MSE, torch, forward
only): pooled change, paired standard error, networks better, all-layer change.
usage (from O/trainer): python -B cert/fold/pairs.py <ranks file> <a>:<b> [<a>:<b> ...]      (labels without the a_ prefix)"""
import json
import sys
from pathlib import Path

import numpy as np

R = Path('run_M/results')
ranks = [int(x) for x in Path(sys.argv[1]).read_text().split(',')]
L = lambda n, r: json.loads((R / ('a_%s_r%04d.json' % (n, r))).read_text())
has = lambda n, r: (R / ('a_%s_r%04d.json' % (n, r))).exists()
for pair in sys.argv[2:]:
    a_name, b_name = pair.split(':')
    rs = [r for r in ranks if has(a_name, r) and has(b_name, r)]
    if len(rs) < 2:
        print('%-34s too few shared ranks (%d)' % (pair, len(rs)))
        continue
    a = np.array([L(a_name, r)['mse'] for r in rs]); b = np.array([L(b_name, r)['mse'] for r in rs]); ch = 100 * (a / b - 1)
    la = np.array([np.mean(L(a_name, r)['mse_layers']) for r in rs]); lb = np.array([np.mean(L(b_name, r)['mse_layers']) for r in rs])
    print('%-34s %2d ranks: %.4e against %.4e: %+.3f%% (se %.3f, better on %d/%d; all layers %+.3f%%)' % (
        pair, len(rs), a.mean(), b.mean(), 100 * (a.mean() / b.mean() - 1), ch.std(ddof=1) / len(rs) ** .5, int((a < b).sum()), len(rs), 100 * (la.mean() / lb.mean() - 1)))
