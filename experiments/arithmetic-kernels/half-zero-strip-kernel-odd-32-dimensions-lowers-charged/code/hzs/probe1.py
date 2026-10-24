"""TAG opus_engineer hzs probe1: (1) meter the parent's _mm on a few read shapes (bill, ops, seconds, error vs float64);
(2) determine empirically which of the 7 deepest-level chunk products a half-zero leaf row / leaf column / inner index needs."""
import json
import sys
import numpy as np
import flopscope as flops
from lab import load, Meter, rand
fnp = flops.numpy
mod, est = load()
print(json.dumps(dict(setup_log=[list(x) for x in mod._PB_LOG])), flush=True)


def relerr(x, ref):
    x = np.asarray(x, dtype=np.float64)
    return float(np.linalg.norm(x-ref)/np.linalg.norm(ref))


# (1) the parent's own path on read shapes
for (m, k, n) in [(672, 10240, 864), (704, 8832, 864), (640, 7296, 864), (928, 8128, 992), (704, 10240, 896), (640, 10240, 832)]:
    a = rand((m, k), 1); b = rand((k, n), 2)
    ref = np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64)
    for rep in range(2):
        with Meter() as mt:
            c = mod._mm(a, b)
        print(json.dumps(dict(shape=[m, k, n], rep=rep, flops=mt.flops, ratio=mt.flops/(2*m*k*n), ops=mt.ops, seconds=round(mt.seconds, 3),
                              relerr=relerr(c, ref), ks=dict(mod._KS_STATS))), flush=True)
    del a, b, c, ref

# (2) chunk needs: one depth-6 product (M, q, N) with interleaved half-zero pads on rows, on columns, and on the inner index
L = 6


def tree_product(aa, bb, M, q, N, edit=None):
    """the kernel's own forward A, forward B, leaf matmul, unwind; edit(TC) may zero chunk products before the unwind"""
    with flops.BudgetContext(flop_budget=2**44, quiet=True):
        TA = mod._ipt_forward_a_ab(aa, L, M, q)
        TB = mod._ipt_forward_b_ab(bb, L, q, N)
        TC, Z1, _pair, ops, basis = mod._ipt_side(2, L, M, N)
        fnp.matmul(TA, TB, out=TC)
        if edit is not None:
            edit(np.asarray(TC), np.asarray(TA), np.asarray(TB))
        o = fnp.empty((M, N), dtype=fnp.float32)
        mod._ks_unwind(TC, Z1, ops, basis, o, L, M, N)
        mod._BH_HOLD.clear()
        mod._TR_LAST[0] = None
    return np.asarray(o, dtype=np.float64)


rng = np.random.default_rng(5)
a_leaf, b_leaf, qc = 10, 13, 20          # main leaf rows / cols; inner leaf dim
M, N, q = 64*(a_leaf+1), 64*(b_leaf+1), 64*qc
A3 = np.zeros((64, a_leaf+1, q), dtype=np.float32)
A3[:, :a_leaf, :] = rng.standard_normal((64, a_leaf, q))
A3[0::2, a_leaf, :] = rng.standard_normal((32, q))          # strip rows at even blocks (deepest row bit = 0); odd blocks stay zero
B3 = np.zeros((q, 64, b_leaf+1), dtype=np.float32)
B3[:, :, :b_leaf] = rng.standard_normal((q, 64, b_leaf))
B3[:, 0::2, b_leaf] = rng.standard_normal((q, 32))          # strip columns at even blocks
aa = fnp.asarray(A3.reshape(M, q)); bb = fnp.asarray(B3.reshape(q, N))
ref = (A3.reshape(M, q).astype(np.float64) @ B3.reshape(q, N).astype(np.float64)).reshape(64, a_leaf+1, 64, b_leaf+1)
full = tree_product(aa, bb, M, q, N).reshape(64, a_leaf+1, 64, b_leaf+1)
print(json.dumps(dict(check='full padded product', relerr=float(np.linalg.norm(full-ref)/np.linalg.norm(ref)))), flush=True)
scale = float(np.sqrt(np.mean(ref**2)))
regions = dict(main=(slice(None), slice(0, a_leaf), slice(None), slice(0, b_leaf)),
               rowstrip=(slice(0, 64, 2), a_leaf, slice(None), slice(0, b_leaf)),
               colstrip=(slice(None), slice(0, a_leaf), slice(0, 64, 2), b_leaf),
               corner=(slice(0, 64, 2), a_leaf, slice(0, 64, 2), b_leaf))
out = {}
for name, zero in (('rowstrip', lambda TC, c: TC[c][..., a_leaf, :b_leaf].__setitem__(Ellipsis, 0)),
                   ('colstrip', lambda TC, c: TC[c][..., :a_leaf, b_leaf].__setitem__(Ellipsis, 0)),
                   ('corner', lambda TC, c: TC[c][..., a_leaf, b_leaf].__setitem__(Ellipsis, 0))):
    need = {}
    for c in range(7):
        got = tree_product(aa, bb, M, q, N, edit=lambda TC, TA, TB, c=c, zero=zero: zero(TC, c)).reshape(64, a_leaf+1, 64, b_leaf+1)
        need[c] = {reg: float(np.max(np.abs(got[idx]-full[idx]))/scale) for reg, idx in regions.items()}
    out[name] = need
    print(json.dumps(dict(zeroed_strip=name, change_by_chunk={c: {r: round(v, 6) for r, v in d.items()} for c, d in need.items()})), flush=True)
# exact zeros of the tree at the strips
with flops.BudgetContext(flop_budget=2**44, quiet=True):
    TA = np.asarray(mod._ipt_forward_a_ab(aa, L, M, q)); mod._TR_LAST[0] = None
    TB = np.asarray(mod._ipt_forward_b_ab(bb, L, q, N))
    mod._BH_HOLD.clear()
print(json.dumps(dict(A_strip_row_zero_chunks=[c for c in range(7) if not np.any(TA[c][..., a_leaf, :])],
                      B_strip_col_zero_chunks=[c for c in range(7) if not np.any(TB[c][..., :, b_leaf])])), flush=True)
print('DONE', flush=True)
