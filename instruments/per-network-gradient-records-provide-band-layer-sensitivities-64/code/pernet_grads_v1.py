"""TAG opus-trainer RAD (written 2026-10-13T00:4xZ; lead 00:4xZ: does a dial's per-network gradient track the L12-15 kink fraction?): writes the
PER-NETWORK gradients of S53's 64 gradient records (A36 7100264-327, S51's point, gk11 on) for the analyst, zero compute (reads the .pt records only).
Units: d(1e8 x final-layer MSE) / d(dial), the jmix convention (each record's gradient x 1e8 / its scale). Per network:
  mse, mse_layers (16), worker / threads of the record;
  bands: family / band -> the band-summed per-layer gradient (L5-9 / L10-11 / L12-14 / L15 / all 0-15) for iso, k21n, X0, X1, X2, kappa, bsrc, ds
         (the lead's list) and for every other per-layer family present;
  pl: family -> the 16 per-layer gradients (every per-layer family in the record);
  dial: the global dials' gradients (dial_grads).
usage (from O/trainer): python -B cert/RAD/pernet_grads_v1.py <round, e.g. S53> <out json>"""
import json
import sys
from pathlib import Path

import torch

NAME, OUT = sys.argv[1], Path(sys.argv[2])
seeds = [int(x) for x in Path('cert/Nf/gen_allow_grad_v1.txt').read_text().split(',')]
BANDS = {'5-9': (5, 9), '10-11': (10, 11), '12-14': (12, 14), '15': (15, 15), 'all': (0, 15)}
LEAD = ('iso', 'k21n', 'X0', 'X1', 'X2', 'kappa', 'bsrc', 'ds')
out = dict(schema='trainer_pernet_grads_v1', round=NAME, records='run_RAD_%s_gen/results/a_R%sg_r<seed>.pt' % (NAME, NAME),
           units='d(1e8 x final-layer MSE) / d(dial): record gradient x 1e8 / record scale', bands={k: list(v) for k, v in BANDS.items()},
           lead_families=list(LEAD), networks={})
for s in seeds:
    r = torch.load('run_RAD_%s_gen/results/a_R%sg_r%d.pt' % (NAME, NAME, s), map_location='cpu', weights_only=False)
    k = 1e8 / r['scale']
    pl = {f: [float(x) * k for x in v.double()] for f, v in r['pl_grads'].items()}
    out['networks'][str(s)] = dict(mse=r['mse'], mse_layers=r['mse_layers'], worker=r.get('worker'), threads=r.get('threads'),
                                   bands={'%s/%s' % (f, b): sum(pl[f][lo:hi + 1]) for f in pl for b, (lo, hi) in BANDS.items()},
                                   pl=pl, dial={d: float(v) * k for d, v in r['dial_grads'].items()})
OUT.open('x').write(json.dumps(out))
print('wrote', OUT, len(out['networks']), 'networks;', len(next(iter(out['networks'].values()))['pl']), 'per-layer families')
