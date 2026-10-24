"""TAG opus-trainer (2026-10-02): verdict on one joint dial solve on top of a carrier (m_mix.py, label <label>): its optimum run for
real on the 16 line-search ranks (against the solve's base: did the model's prediction hold?) and on the judging ranks that have
both the optimum and the base (same card, same carrier, default dials). The last line printed is the TEAM_LOG line.
usage: python -B cert/fold/dial_verdict.py <label> <spec.json> [<tag, e.g. _min3>]"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

T = Path(__file__).resolve().parents[2]
LABEL, SPEC, TAG = sys.argv[1], Path(sys.argv[2]), (sys.argv[3] if len(sys.argv) > 3 else '')
spec = json.loads(SPEC.read_text())
R = T / 'run_M/results'
L = lambda name, r: json.loads((R / ('%s_r%04d.json' % (name, r))).read_text())
has = lambda name, r: (R / ('%s_r%04d.json' % (name, r))).exists()
ls = [int(x) for x in (T / 'cert/ls16_ranks.txt').read_text().split(',')]
val = [int(x) for x in (T / 'cert/runM_val50.txt').read_text().split(',')]
opt = json.loads((SPEC.parent / ('%s_optimum%s.json' % (LABEL, TAG))).read_text())
J, B = 'a_%sjudge%s' % (LABEL, TAG), 'a_%sjbase' % LABEL
val = [r for r in val if has(J, r) and has(B, r)]


def cmp(a_name, b_name, ranks):
    a = np.array([L(a_name, r)['mse'] for r in ranks]); b = np.array([L(b_name, r)['mse'] for r in ranks])
    la = np.array([np.mean(L(a_name, r)['mse_layers']) for r in ranks]); lb = np.array([np.mean(L(b_name, r)['mse_layers']) for r in ranks])
    ch = 100 * (a / b - 1)
    return 100 * (a.mean() / b.mean() - 1), ch.std(ddof=1) / len(ch) ** .5, int((a < b).sum()), len(ranks), 100 * (la.mean() / lb.mean() - 1), a.mean(), b.mean()


real = cmp('a_%sopt%s' % (LABEL, TAG), 'a_%s_base' % LABEL, ls)
judge = cmp(J, B, val)
stored = torch.load(spec['base']['file'], map_location='cpu', weights_only=True).get('pl', {}) if 'file' in spec['base'] else {}
families = sorted({family for arm in spec['arms'].values() for family in arm.get('pl', {})})
shift = lambda family: [a - b for a, b in zip(opt['shifts'].get(family, [0.0] * 16), [float(x) for x in stored.get(family, [0.0] * 16)])]
show = lambda v: ('%+.3f' % v[8]) if len({round(x, 9) for x in v[1:15]}) == 1 else 'layers 1-4 %+.3f, 5-9 %+.3f, 10-14 %+.3f' % (v[2], v[7], v[12])
print('weights:', {k: round(v, 3) for k, v in opt['weights'].items()})
print('JOINT DIAL SOLVE %s%s (O/trainer/%s; request fields %s): on the 16 line-search ranks the model said %+.2f%% (leave-one-network-out %+.2f%%) against its base and the '
      'real run of the optimum gave %+.2f%% (se %.2f, better on %d/%d). JUDGED on the first %d judging ranks, optimum against the same card and carrier at default dials '
      '(final-layer MSE %.4e against %.4e): %+.3f%% (se %.3f, better on %d/%d; all layers %+.3f%%). SHIFTS AT THE OPTIMUM (added to the default value): %s. '
      'Weights of the arms (1 = the arm as run): %s. Parameter file O/trainer/%s.' % (
          LABEL, TAG, SPEC.relative_to(T) if SPEC.is_absolute() else SPEC, json.dumps(spec.get('request', {})), opt['model_pct'], opt['leave_one_out_pct'], *real[:4],
          len(val), judge[5], judge[6], *judge[:5], '; '.join('%s %s' % (family, show(shift(family))) for family in families),
          ', '.join('%s %+.2f' % (k, v) for k, v in opt['weights'].items()), (SPEC.parent / ('%s_optimum%s.pt' % (LABEL, TAG)))))
