#!/usr/bin/env python3
"""TAG fable-engineer (E-41): price symmetric-output alternatives on the candidate's own kernel (warm calls: each variant runs twice,
the second call is billed, so frame binding is excluded). Units: dense = 2*m*k*n flops of the product.
  gram      G = X^T X (1024x1024x1024): current _mm vs fnp.einsum('ai,aj->ij') vs a block recursion (off-diagonal blocks through _mm,
            diagonal leaves as the symmetric einsum) at 1-3 levels
  sandwich  G = w^T (C w): current second product _mm(w^T, P) vs an upper-block recursion (skip G21 = G12^T, dense leaves)
  chol      G = (L^T w)^T (L^T w) with L = cholesky(C): cholesky + _mm(L^T, w) + gram recursion, vs the current two products
  wgram     old-tier Fj diag(g) Fj^T (384 x k): current _shared_old384_mm vs einsum('ia,a,ja->ij')
  assembly  copyto of a block and of a transposed block; ops on a SymmetricTensor
usage: sym_price.py <candidate.py>"""
import sys, types, warnings
from pathlib import Path
import numpy as np
import flopscope as flops, flopscope.numpy as fnp
warnings.filterwarnings("ignore")
cand = Path(sys.argv[1]).resolve()
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("sym_price_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
rng = np.random.default_rng(0)
ctx = flops.BudgetContext(flop_budget=2**50, quiet=True); ctx.__enter__()
arena = fnp.empty((mod._PACKED_ELEMENTS,), dtype=fnp.float32)
mod._SCRATCH = [arena, mod._cp3_bind_arena_views(arena, mod._PACKED_PUSH), mod._cp3_bind_arena_views(arena, mod._PACKED_READ), None, None, None, None]
mod._bind_push6(mod._SCRATCH[0])

def billed(fn, *a):
    fn(*a)                                   # first call binds frames
    b0 = ctx._flops_used; out = fn(*a); return ctx._flops_used - b0, out

def raw(x): return np.asarray(x).view(np.ndarray).astype(np.float64)

def sym_gram(X, levels):                     # X^T X, X is k x m
    m = int(X.shape[1])
    if levels == 0:
        return fnp.einsum('ai,aj->ij', X, X)
    h = m // 2
    X1, X2 = X[:, :h], X[:, h:]
    G11, G22, G12 = sym_gram(X1, levels - 1), sym_gram(X2, levels - 1), mod._mm(X1.T, X2)
    out = fnp.empty((m, m), dtype=fnp.float32)
    fnp.copyto(out[:h, :h], G11); fnp.copyto(out[h:, h:], G22); fnp.copyto(out[:h, h:], G12); fnp.copyto(out[h:, :h], G12.T)
    return out

def sym_upper(A, B, levels):                 # A @ B known symmetric: A is m x k, B is k x m
    m = int(A.shape[0])
    if levels == 0:
        return mod._mm(A, B)
    h = m // 2
    G11, G22, G12 = sym_upper(A[:h], B[:, :h], levels - 1), sym_upper(A[h:], B[:, h:], levels - 1), mod._mm(A[:h], B[:, h:])
    out = fnp.empty((m, m), dtype=fnp.float32)
    fnp.copyto(out[:h, :h], G11); fnp.copyto(out[h:, h:], G22); fnp.copyto(out[:h, h:], G12); fnp.copyto(out[h:, :h], G12.T)
    return out

n = 1024; dense = 2 * n ** 3
X = fnp.asarray((rng.standard_normal((n, n)) / 32).astype(np.float32))
w = fnp.asarray((rng.standard_normal((n, n)) / 32).astype(np.float32))
Z = rng.standard_normal((n, n)).astype(np.float32) / 32
C = fnp.asarray((Z @ Z.T / n + 1e-3 * np.eye(n)).astype(np.float32))
ref = raw(X).T @ raw(X)
print("GRAM X^T X, 1024^3 (units of 2n^3)")
b, g = billed(lambda: mod._mm(X.T, X)); print(f"  current _mm(X.T, X)          {b/dense:6.3f}  err {np.abs(raw(g)-ref).max():.1e}")
b, g = billed(lambda: fnp.einsum('ai,aj->ij', X, X)); print(f"  einsum ai,aj->ij             {b/dense:6.3f}  err {np.abs(raw(g)-ref).max():.1e}  type {type(g).__name__}")
for L in (1, 2, 3):
    b, g = billed(lambda: sym_gram(X, L)); print(f"  block recursion {L} levels     {b/dense:6.3f}  err {np.abs(raw(g)-ref).max():.1e}")
print("SANDWICH w^T (C w), 1024^3 each")
b1, P = billed(lambda: mod._mm(C, w)); print(f"  first product _mm(C, w)      {b1/dense:6.3f}")
ref2 = raw(w).T @ raw(P)
b, g = billed(lambda: mod._mm(w.T, P)); print(f"  current second _mm(w.T, P)   {b/dense:6.3f}  err {np.abs(raw(g)-ref2).max():.1e}")
for L in (1, 2, 3):
    b, g = billed(lambda: sym_upper(w.T, P, L)); print(f"  upper-block recursion {L}     {b/dense:6.3f}  err {np.abs(raw(g)-ref2).max():.1e}")
print("CHOLESKY route")
b, Lc = billed(lambda: fnp.linalg.cholesky(C)); print(f"  cholesky(C) f32              {b/dense:6.3f}")
b, Y = billed(lambda: mod._mm(Lc.T, w)); print(f"  _mm(L^T, w)                  {b/dense:6.3f}")
for L in (0, 2, 3):
    b, g = billed(lambda: sym_gram(Y, L)); print(f"  gram of L^T w, {L} levels      {b/dense:6.3f}  err vs w^T C w {np.abs(raw(g)-raw(w).T @ raw(C) @ raw(w)).max():.1e}")
print("WEIGHTED GRAM Fj diag(g) Fj^T, 384 x k")
for k in (768, 3072, 6144):
    F = fnp.asarray((rng.standard_normal((384, k)) / 32).astype(np.float32)); gv = fnp.asarray(rng.random(k).astype(np.float32))
    d = 2 * 384 * 384 * k
    b, g = billed(lambda: mod._shared_old384_mm(F, gv[:, None] * F.T)); print(f"  k={k:5d} current            {b/d:6.3f} (incl. the {k*384} scaling flops)")
    b, g = billed(lambda: fnp.einsum('ia,a,ja->ij', F, gv, F)); print(f"  k={k:5d} einsum ia,a,ja->ij  {b/d:6.3f}  type {type(g).__name__}")
print("ASSEMBLY and SymmetricTensor ops (per element of a 1024^2 matrix)")
out = fnp.empty((n, n), dtype=fnp.float32); G12 = fnp.asarray(np.ones((512, 512), np.float32))
b, _ = billed(lambda: fnp.copyto(out[:512, 512:], G12)); print(f"  copyto block                 {b/512**2:6.3f}")
b, _ = billed(lambda: fnp.copyto(out[512:, :512], G12.T)); print(f"  copyto transposed block      {b/512**2:6.3f}")
S = fnp.einsum('ai,aj->ij', X, X); D = mod._mm(X.T, X)
for label, f in (("S + S", lambda: S + S), ("D + D", lambda: D + D), ("S * S", lambda: S * S), ("S.astype(f64)", lambda: S.astype(fnp.float64)),
                 ("D.astype(f64)", lambda: D.astype(fnp.float64)), ("exp(S)", lambda: fnp.exp(S)), ("S @ X", lambda: S @ X)):
    b, r = billed(f); print(f"  {label:14s} {b/n**2:8.3f} per element  -> {type(r).__name__}")
