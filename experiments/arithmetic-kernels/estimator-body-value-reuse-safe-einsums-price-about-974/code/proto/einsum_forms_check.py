#!/usr/bin/env python3
"""TAG fable-engineer2 (E-71 group 5 prototype, unit certificate): every einsum form the --g5e prototype uses, against the chain it
replaces, through flopscope in-process (the server's flopscope + numpy): bitwise equality (signed zeros compared as equal values, and
counted), identical bill, and result dtype. usage: einsum_forms_check.py"""
import numpy as np, flopscope as flops, flopscope.numpy as fnp
rng = np.random.default_rng(7); n = 1024
def arr(*shape, dt=np.float64): return fnp.asarray(rng.standard_normal(shape).astype(dt))
def bill(f):
    with flops.BudgetContext(flop_budget=2**40, quiet=True) as ctx:
        r = f()
    return ctx.flops_used, np.asarray(r)
def c(v, dt): return fnp.asarray(v, dtype=dt)
a, b, x, s = arr(n), arr(n), arr(n), arr(n)
M, K = arr(n, n), arr(n, n)
P, D = arr(4, n), arr(4, n, n)
j1m = fnp.asarray([1.0, 0.0, 1.0, 1.0])
m32, R32 = arr(n, dt=np.float32), arr(n, n, dt=np.float32)
births = fnp.asarray(rng.random(n) < 0.3)
cases = [
    ("derivatives ((sign*h)*pdf)", lambda: (-1 * a) * b, lambda: fnp.einsum(',i,i->i', c(-1.0, 'float64'), a, b, optimize=False)),
    ("k31 ((c1*mu)*k3)", lambda: (0.37 * a) * b, lambda: fnp.einsum(',i,i->i', c(0.37, 'float64'), a, b, optimize=False)),
    ("gn ((iso*kink)*k3)", lambda: (-0.094 * a) * b, lambda: fnp.einsum(',i,i->i', c(-0.094, 'float64'), a, b, optimize=False)),
    ("gn x[:,None]*Cpre", lambda: a[:, None] * M, lambda: fnp.einsum('i,ij->ij', a, M, optimize=False)),
    ("gn v[:,None]*s[None,:]", lambda: a[:, None] * s[None, :], lambda: fnp.einsum('i,j->ij', a, s, optimize=False)),
    ("gn k31*(t4*r31) (0-d last)", lambda: -0.212 * (a * b), lambda: fnp.einsum('i,i,->i', a, b, c(-0.212, 'float64'), optimize=False)),
    ("gn x32[:,None]*rho32", lambda: m32[:, None] * R32, lambda: fnp.einsum('i,ij->ij', m32, R32, optimize=False)),
    ("bd (skew*X)*pdf", lambda: (a * b) * x, lambda: fnp.einsum('i,i,i->i', a, b, x, optimize=False)),
    ("bd half_d[:,None]*K21 out=", lambda: fnp.multiply(a[:, None], K, out=fnp.empty((n, n))), lambda: fnp.einsum('i,ij->ij', a, K, out=fnp.empty((n, n)), optimize=False)),
    ("bd nhg[:,None]*K21.T out=", lambda: fnp.multiply(a[:, None], K.T, out=fnp.empty((n, n))), lambda: fnp.einsum('i,ji->ij', a, K, out=fnp.empty((n, n)), optimize=False)),
    ("bd Cpre*gate_jet[:,None] out=", lambda: fnp.multiply(M, a[:, None], out=fnp.empty((n, n))), lambda: fnp.einsum('ij,i->ij', M, a, out=fnp.empty((n, n)), optimize=False)),
    ("bd projection*_j1m[:,None]", lambda: P * j1m[:, None], lambda: fnp.einsum('aj,a->aj', P, j1m, optimize=False)),
    ("bd bp*projection[:,None,:]", lambda: M * P[:, None, :], lambda: fnp.einsum('ij,aj->aij', M, P, optimize=False)),
    ("bd (a64*d)*births (bool)", lambda: (a * b) * births, lambda: fnp.einsum('i,i,i->i', a, b, births, optimize=False)),
]
ok = True
for label, f, g in cases:
    bf, rf = bill(f); bg, rg = bill(g)
    same_bits = rf.dtype == rg.dtype and rf.shape == rg.shape and np.array_equal(rf.view(np.uint8), rg.view(np.uint8))
    same_vals = rf.dtype == rg.dtype and np.array_equal(rf, rg)
    zeros = int(np.sum((rf == 0) & (np.signbit(rf) != np.signbit(rg))))
    good = same_vals and bf == bg
    ok &= good
    print(f"{'OK  ' if good else 'FAIL'} {label:32s} bills {bf} / {bg}  dtype {rf.dtype}/{rg.dtype}  bitwise {same_bits}  signed-zero flips {zeros}")
print("ALL FORMS OK" if ok else "SOME FORMS FAIL")
