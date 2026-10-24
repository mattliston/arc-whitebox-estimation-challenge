"""TAG opus-trainer RAD (written 2026-10-13T11:1xZ; lead 10:3xZ K4L ages 3-4 in the RAD card from S56): zero-compute t-statistics of the K4L age 3-4
strength gradients in a round's 64 gradient records (worker v16 computes them; they are not in ROUTES, so no RAD arm dials them). Per band, the
band-summed gradient of k4la3, of k4la4 and of both moved together (k4l34 = k4la3 + k4la4), in the jmix units (x 1e8 / scale), with the strict rule's
halves (first / last 32). A negative t means raising that strength lowers the final-layer MSE.
usage (from O/trainer): python -B cert/RAD/k4l_grad_t_v1.py <round, e.g. S56> [<gradient records dir>]"""
import sys
from pathlib import Path

import numpy as np
import torch

NAME = sys.argv[1]
D = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('run_RAD_%s_gen/results' % NAME)
rows = [torch.load(p, map_location='cpu', weights_only=False) for p in sorted(D.glob('*.pt'))]
rows = [r for r in rows if 'k4la3' in r['pl_grads'] and 'k4la4' in r['pl_grads']]
print('%s: %d gradient records with k4la3 / k4la4 in %s' % (NAME, len(rows), D))
if not rows:
    sys.exit(0)
t = lambda x: float(x.mean() / (x.std(ddof=1) / len(x) ** .5)) if len(x) > 1 and x.std() > 0 else float('nan')
G = {f: np.stack([r['pl_grads'][f].double().numpy() * (1e8 / r['scale']) for r in rows]) for f in ('k4la3', 'k4la4')}
G['k4l34'] = G['k4la3'] + G['k4la4']
h = len(rows) // 2
for f in ('k4la3', 'k4la4', 'k4l34'):
    out = []
    for lo, hi in ((4, 7), (8, 11), (12, 14), (15, 15), (8, 15)):
        s = G[f][:, lo:hi + 1].sum(1)
        out.append('L%d-%d t %+.2f (halves %+.2f / %+.2f, mean %+.4f)' % (lo, hi, t(s), t(s[:h]), t(s[h:]), s.mean()))
    print('%-6s %s' % (f, ' | '.join(out)))
