"""TAG opus_engineer a33 SS unit check (light, one process, a few seconds): the built candidate's star functions against the formula
in float64 numpy, on a random plane with the real one's shape (a strong row-sum part, a few large directions, a flat rest).
  s2 = (W o W)' L' W + m (W' (A 1 - L 1))',  L = the candidate's own factors Q B (or 0 at rank 0),  m = each unit's mean squared weight
  (one-direction form, _SS_RANK entry -1:  s2 = ((W o W)' A' a) (W' a)' / a'a,  a = A 1)
Checked: the plane returned by _flms_cov_star (off the diagonal s2 / 3, on it diag s2), the float32 copy, the final layer's rows
(_flms_star_diagonal), and the bill of the star part against the parent file's (the covariance product is replaced by a plain
matmul in both, so the difference is the lever's own cost). usage: unit_ss.py <SS candidate.py> <parent candidate.py>"""
from pathlib import Path
import sys
import types
import numpy as np
import flopscope as flops
fnp = flops.numpy


def load(path, name):
    stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
    sys.modules['whestbench'] = stub
    mod = types.ModuleType(name); mod.__file__ = str(path); sys.modules[name] = mod
    exec(compile(Path(path).read_text(), str(path), 'exec', dont_inherit=True), mod.__dict__)
    mod._mm = lambda a, b: a @ b                    # the covariance product is not under test: plain matmuls in both files
    mod._sym_upper = lambda wt, first: wt @ first
    mod._PHYS_LAYER[0] = 0                          # the fragment reads the form of the layer the star is moved into
    return mod


cand, parent = load(sys.argv[1], 'unit_ss_candidate'), load(sys.argv[2], 'unit_ss_parent')
K = int(cand._SS_RANK[0]) if hasattr(cand, '_SS_RANK') else int(cand._LR_K)
rng = np.random.default_rng(11)
n = 1024
W = (rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32)
C = np.eye(n)
head = rng.standard_normal((n, 6)) @ (rng.standard_normal((6, n)) * np.array([8, 4, 3, 2, 1.5, 1])[:, None])
A = (np.outer(rng.uniform(0.5, 1.5, n), rng.uniform(0.2, 1.8, n)) * 3 + head * 0.05 + rng.standard_normal((n, n)) * 0.6).astype(np.float32) * 1e-6
rows = np.sort(rng.choice(n, 700, replace=False))
out = {}
for name, mod in (('candidate', cand), ('parent', parent)):
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        Cpre, plane, cov32, plane32 = mod._flms_cov_star(fnp.asarray(W), fnp.asarray(C), fnp.asarray(A), retain32=True)
        used = int(ctx.summary_dict()['flops_used']); ops = len(ctx._op_log)
        final = mod._flms_star_diagonal(fnp.asarray(W)[:, fnp.asarray(rows)], fnp.asarray(A))
        used_final = int(ctx.summary_dict()['flops_used']) - used
        Q, B = (None, None) if (name == 'candidate' and K <= 0) else mod._lr_factor(fnp.asarray(A))
    out[name] = dict(plane=np.asarray(plane, dtype=np.float64), plane32=np.asarray(plane32, dtype=np.float64), final=np.asarray(final, dtype=np.float64), used=used, ops=ops, used_final=used_final,
                     L=np.zeros((n, n)) if Q is None else np.asarray(Q, dtype=np.float64) @ np.asarray(B, dtype=np.float64))
W64, A64 = W.astype(np.float64), A.astype(np.float64)
wt2 = W64.T * W64.T
m = wt2.mean(1)
L = out['candidate']['L']
s2 = wt2 @ L.T @ W64 + np.outer(m, (A64.sum(1) - L.sum(1)) @ W64)
if K < 0:
    a = A64.sum(1); s2 = np.outer(wt2 @ (A64.T @ a), a @ W64) / (a @ a)
want = s2 / 3
np.fill_diagonal(want, np.diag(s2))
scale = np.abs(want).max()
errs = dict(plane=np.abs(out['candidate']['plane'] - want).max() / scale, plane32=np.abs(out['candidate']['plane32'] - want).max() / scale,
            final=np.abs(out['candidate']['final'] - np.diag(s2)[rows]).max() / np.abs(np.diag(s2)).max())
exact = wt2 @ A64.T @ W64
off = ~np.eye(n, dtype=bool)
sp = wt2 @ out['parent']['L'].T @ W64
print('rank %d: largest difference / largest value: plane %.1e, float32 copy %.1e, final-layer rows %.1e' % (K, errs['plane'], errs['plane32'], errs['final']))
print('on this random plane, relative error of the transported star against the exact one: candidate %.4f, parent %.4f' % (np.sqrt(((s2 - exact)[off] ** 2).mean() / (exact[off] ** 2).mean()), np.sqrt(((sp - exact)[off] ** 2).mean() / (exact[off] ** 2).mean())))
print('bill of one layer (covariance as a plain matmul in both): candidate %d, parent %d, difference %+d = %+.2f n^2; charged ops %d against %d; final layer %+d flops' % (
    out['candidate']['used'], out['parent']['used'], out['candidate']['used'] - out['parent']['used'], (out['candidate']['used'] - out['parent']['used']) / n ** 2, out['candidate']['ops'], out['parent']['ops'],
    out['candidate']['used_final'] - out['parent']['used_final']))
print('PASS' if max(errs.values()) < 2e-5 else 'FAIL')
