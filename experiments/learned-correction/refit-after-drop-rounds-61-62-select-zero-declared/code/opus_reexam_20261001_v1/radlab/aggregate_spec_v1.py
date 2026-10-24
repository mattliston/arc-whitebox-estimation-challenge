"""radlab aggregate_spec_v1 (10-12, for the trainer's S53+ arm stage; FROZEN once used: later changes go to a v2): ONE aggregate weak-band dial arm
from a round's own 64 start-point gradients, as a fixed multi-family spec (JM_FIXED format 'family:lo:hi:shift,...', applied TOGETHER in one arm).
Bands: trainer/cert/RAD/ROUTES.json (per_layer: 1-6 / 7-10 / 11-14 / all; all_only: all; pending: per layer iff the engineer's line
'S51 PORT PARITY <arm>: PASS' exists, out on FAILED, else all; bands: the family's own list). Strict = make_s53_v3's rule (t^2 > 6, both halves of
the same sign at |t| > 1.5): those are rule (b)'s own dial arms and are NOT in the aggregate. Weak = |t| >= 2, not strict, not overlapping a strict
band of the same family, not overlapping a band sized as a dial arm in the previous round (optional keep file), one band per family region (the
strongest of overlapping weak bands). Each component moves against its gradient sign by its own dial-arm trial (jmix: -0.5% first-order change,
cap 0.02) times |t| / sum |t|, so the whole arm's first-order change is about -0.5% of the error; the freeze sizes it as one arm.
Gradients scaled 1e8 / scale and mse x 1e8 as in jmix_generated_v7 / make_s53_v3. numpy only (radlab/ptload.py, no torch).
usage: python -B radlab/aggregate_spec_v1.py <results dir> <gradient id prefix> <out json> [previous round keep json]
  e.g. (from O) radlab/aggregate_spec_v1.py trainer/run_RAD_S53_gen/results a_RS53g trainer/cert/RAD/S53/aggregate_spec_v1.json trainer/cert/RAD/S52/keep_v1.json"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ptload import load

O = Path(__file__).resolve().parents[1]
res_dir, pre, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
prev_keep = json.loads(Path(sys.argv[4]).read_text())['keep'] if len(sys.argv) > 4 and Path(sys.argv[4]).exists() else []
assert not out.exists(), out
seeds = [int(x) for x in (O / 'trainer/cert/Nf/gen_allow_grad_v1.txt').read_text().split(',')]
rows = [load(str(res_dir / ('%s_r%d.pt' % (pre, r)))) for r in seeds]
assert len(rows) == 64
routes = json.loads((O / 'trainer/cert/RAD/ROUTES.json').read_text())
ENG = [l for l in (O / 'TEAM_LOG.md').read_text().splitlines() if l.startswith('[engineer] ')]
STD = ((1, 6), (7, 10), (11, 14), (0, 15))


def bands_of(fam):
    if fam in routes.get('bands', {}):
        return [tuple(b) for b in routes['bands'][fam]]
    if fam in routes.get('per_layer', []):
        return list(STD)
    if fam in routes.get('all_only', []):
        return [(0, 15)]
    if fam in routes.get('pending', {}):
        arm = routes['pending'][fam]
        if any('S51 PORT PARITY %s: FAILED' % arm in l for l in ENG):
            return []
        return list(STD) if any('S51 PORT PARITY %s: PASS' % arm in l for l in ENG) else [(0, 15)]
    return []


t = lambda x: float(x.mean() / (x.std(ddof=1) / len(x) ** .5)) if x.std() > 0 else 0.0
mse = float(np.mean([float(r['mse']) for r in rows])) * 1e8
sc = np.array([1e8 / float(r['scale']) for r in rows])
prev = []      # previous round's dial arms 'd<fam>_<lo>_<hi>' (or 'd<fam>' = all)
for a in prev_keep:
    if a.startswith('d'):
        parts = a[1:].rsplit('_', 2)
        prev.append((parts[0], int(parts[1]), int(parts[2])) if len(parts) == 3 and parts[1].isdigit() else (a[1:], 0, 15))
strict, weak = [], []
for fam in sorted(rows[0]['pl_grads'], key=str.lower):
    if fam == 'kappa':
        continue
    G = np.stack([np.asarray(r['pl_grads'][fam], dtype=np.float64) for r in rows]) * sc[:, None]
    if not np.abs(G).max() > 0:
        continue
    cand = []
    for lo, hi in bands_of(fam):
        s = G[:, lo:hi + 1].sum(1); tt, ta, tb = t(s), t(s[:32]), t(s[32:])
        cand.append(dict(family=fam, lo=lo, hi=hi, t=tt, halves=[ta, tb], slope=float(s.mean()),
                         strict=bool(tt * tt > 6 and ta * tb > 0 and min(abs(ta), abs(tb)) > 1.5)))
    st = [c for c in cand if c['strict']]
    strict += st
    over = lambda c, d: not (c['hi'] < d['lo'] or c['lo'] > d['hi'])
    kept = []
    for c in sorted((c for c in cand if not c['strict'] and abs(c['t']) >= 2), key=lambda c: -abs(c['t'])):
        if any(over(c, d) for d in st) or any(over(c, dict(lo=lo, hi=hi)) for f, lo, hi in prev if f == fam) or any(over(c, d) for d in kept):
            continue
        kept.append(c)
    weak += kept
T = sum(abs(c['t']) for c in weak)
for c in weak:
    trial = min(0.005 * mse / max(abs(c['slope']), 1e-300), 0.02)
    c['shift'] = float(-np.sign(c['slope']) * trial * abs(c['t']) / T)
q = float(np.sqrt(sum(c['t'] ** 2 for c in weak)))
spec = ','.join('%s:%d:%d:%.6g' % (c['family'], c['lo'], c['hi'], c['shift']) for c in weak)
rec = dict(schema='radlab_aggregate_spec_v1', utc=os.popen('date -u +%FT%TZ').read().strip(), gradients=str(res_dir), prefix=pre, mse_1e8=mse,
           routes=str(O / 'trainer/cert/RAD/ROUTES.json'), previous_dial_arms=prev, strict=[dict(c, shift=None) for c in strict], components=weak,
           aggregate_A36_t=q, expected_fresh_t=[round(0.45 * q, 2), round(0.54 * q, 2)], spec=spec,
           rule='weak = |t| >= 2, not strict, not overlapping a strict band / a previous-round dial band / each other; shift = -sign(slope) x '
                'min(0.005 mse / |slope|, 0.02) x |t| / sum |t|; one arm, sized by the freeze')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rec, indent=1))
print('strict (rule b): %s' % ('; '.join('%s:%d-%d %+.2f' % (c['family'], c['lo'], c['hi'], c['t']) for c in strict) or 'none'))
print('aggregate arm (%d components, A36 t %.2f, expected fresh t %.2f-%.2f): %s' % (len(weak), q, 0.45 * q, 0.54 * q, spec or 'EMPTY (no arm)'))
