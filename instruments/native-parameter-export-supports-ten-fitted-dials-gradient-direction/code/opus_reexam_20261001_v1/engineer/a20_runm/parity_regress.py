"""TAG opus_engineer a20 (2026-10-02): native against torch for a build that changes SEVERAL dials at once (run M export rehearsal).
The native prediction change (test build minus base build, same network) is compared with
  (1) the torch card's change for the same dial values (arm `all` minus arm `ref`): cosine and size ratio per layer, and
  (2) the torch card's change for each dial ALONE (single arms minus `ref`): least squares of the native change on those single
      changes. A dial the native build implements as the card does gets a coefficient near 1; a missing dial 0; two dials swapped
      show as two wrong coefficients. The standard error comes from the residual (treated as independent per unit).
The same fit of torch `all` on the singles shows how far the dials are from adding up (1 for each = they add).
usage: parity_regress.py <torch arms .npz> <native base pred .npy> <native test pred .npy> [single arm names, default: every arm but ref and all]"""
import sys

import numpy as np

t = np.load(sys.argv[1])
base = np.load(sys.argv[2]).astype(np.float64)
test = np.load(sys.argv[3]).astype(np.float64)
singles = sys.argv[4:] or [k[5:] for k in t.files if k not in ('pred_ref', 'pred_all')]
dn = test - base
da = t['pred_all'] - t['pred_ref']
D = {name: t['pred_' + name] - t['pred_ref'] for name in singles}
rms = lambda x: float(np.sqrt(np.mean(x * x)))
print('native change against torch `all`, per layer: cosine / size ratio (native / torch) / rms of the torch change x1e6')
for l in range(16):
    if rms(da[l]) == 0.0 and rms(dn[l]) == 0.0:
        print('  layer %2d: no change in either' % l); continue
    cos = float(dn[l] @ da[l]) / max(float(np.linalg.norm(dn[l]) * np.linalg.norm(da[l])), 1e-300)
    print('  layer %2d: cos %.3f  ratio %.3f  torch rms %.2f  left after subtracting torch %.2f' % (
        l, cos, rms(dn[l]) / max(rms(da[l]), 1e-300), 1e6 * rms(da[l]), 1e6 * rms(dn[l] - da[l])))


def fit(y, label, layers):
    X = np.stack([np.concatenate([D[name][l] for l in layers]) for name in singles], axis=1)
    yy = np.concatenate([y[l] for l in layers])
    coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
    res = yy - X @ coef
    dof = len(yy) - len(singles)
    cov = np.linalg.inv(X.T @ X) * float(res @ res) / dof
    print('%s: residual rms %.2fe-6 of %.2fe-6' % (label, 1e6 * rms(res), 1e6 * rms(yy)))
    for name, c, s in zip(singles, coef, np.sqrt(np.diag(cov))):
        print('    %-6s coefficient %+.3f  (se %.3f)   rms of this dial\'s torch change %.2fe-6' % (name, c, s, 1e6 * rms(X[:, singles.index(name)])))


for layers, tag in (([15], 'final layer'), (list(range(1, 16)), 'layers 1-15')):
    fit(dn, 'NATIVE change on the torch single-dial changes, %s' % tag, layers)
    fit(da, 'torch `all` on the torch single-dial changes, %s' % tag, layers)
    # the dials interact (the row above is not all ones), so the sharp test is on what is LEFT: a dial ported as in the card leaves
    # a coefficient of 0 here, a missing dial -1, a dial applied twice +1
    fit(dn - da, 'NATIVE change MINUS torch `all`, on the torch single-dial changes, %s (0 = ported as in the card)' % tag, layers)
