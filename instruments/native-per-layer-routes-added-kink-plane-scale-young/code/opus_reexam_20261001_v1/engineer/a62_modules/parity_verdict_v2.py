"""TAG opus_engineer a62 v2 (file written 2026-10-12T07:30Z; lead 07:3xZ: positive control + a size floor) = parity_verdict_v1.py with
  - a FLOOR: on a network where BOTH the torch and the native final-layer prediction change are below FLOOR of the prediction's size
    (default 2e-6: the native rounding-only change at layer 15 measured on 7100136, exact pair split psr0 vs S48b, was 1.73e-6), the two are
    NOT COMPARABLE there and that network is left out (a strength arm that barely moves predictions is not judged on noise); a change above
    the floor on either side is judged (a native change torch does not make, or the reverse, is a real mismatch);
  - the rule on every comparable network: correlation >= 0.70 and native size within 1/3 .. 3 x torch's;
  - verdict PASS (all comparable networks pass, at least one comparable), FAIL (any comparable network fails), NOT COMPARABLE (none);
  - record names PREFIX_<BASENAME>_r<seed> / PREFIX_<arm>_r<seed> (env PREFIX default a_RS51, BASENAME default base); several native arm
    files may be given as a comma list (one seed each, matched by seed).
It writes nothing anywhere (the TEAM_LOG verdict line is posted by hand in the trainer's format).
usage: parity_verdict_v2.py <torch results dir> <arm> <native base gate json> <native arm gate json[,json...]> <seed> [<seed> ...]"""
import json
import os
import sys
import numpy as np

R, arm, nb, na = sys.argv[1:5]
seeds = sys.argv[5:]
P, BN, FLOOR = os.environ.get('PREFIX', 'a_RS51'), os.environ.get('BASENAME', 'base'), float(os.environ.get('FLOOR', '2e-6'))
rows = lambda f: {str(x['seed']): x for x in json.load(open(f))['rows']}
arm_files = na.split(',')
results = []
for s in seeds:
    tb = json.load(open('%s/%s_%s_r%s.json' % (R, P, BN, s))); ta = json.load(open('%s/%s_%s_r%s.json' % (R, P, arm, s)))
    pt = np.asarray(tb['final_prediction'], float)
    dt = np.asarray(ta['final_prediction'], float) - pt
    fa = [f for f in arm_files if s in rows(f)][0]
    qb = np.load(nb.replace('.json', '_pred.npz'))[s].astype(float)[15]; qa = np.load(fa.replace('.json', '_pred.npz'))[s].astype(float)[15]
    dn = qa - qb
    scale = float(np.linalg.norm(pt))
    nt, nn = float(np.linalg.norm(dt)), float(np.linalg.norm(dn))
    c = float(dt @ dn) / (nt * nn) if nt > 0 and nn > 0 else float('nan')
    ratio = nn / nt if nt > 0 else float('inf')
    rb, ra = rows(nb)[s], rows(fa)[s]
    if nt / scale < FLOOR and nn / scale < FLOOR:
        state = 'not comparable (both changes under the floor)'
    else:
        state = 'ok' if (c >= 0.70 and 1 / 3 <= ratio <= 3) else 'NOT OK'
    results.append(state)
    print('%s %s: corr %+.3f, native/torch size %.2f, change torch %.2e native %.2e of the prediction (floor %.0e), MSE change torch %+.3f%% native %+.3f%% -> %s' % (
        arm, s, c, ratio, nt / scale, nn / scale, FLOOR, 100 * (ta['mse'] / tb['mse'] - 1), 100 * (ra['mse_final'] / rb['mse_final'] - 1), state))
judged = [r for r in results if not r.startswith('not comparable')]
v = 'NOT COMPARABLE' if not judged else ('PASS' if all(r == 'ok' for r in judged) else 'FAIL')
print('VERDICT %s: %s' % (arm, v))
