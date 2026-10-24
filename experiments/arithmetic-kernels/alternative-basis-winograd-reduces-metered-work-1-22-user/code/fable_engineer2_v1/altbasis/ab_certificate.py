#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-ABC): numerical certificate of the derived alternative-basis Strassen-Winograd (12 adds per level, 1-add basis
changes): A' replaces a21 by S1 = a21 + a22; B' replaces b12 by T3 = b22 - b12; core encodings S2 = S1 - a11, S4 = a12 - S2, S3 = a22 - S2
and T2 = T3 + b11, T4 = T2 - b21, T1 = b22 - T2; products m1..m7 as Winograd; decode C11 = m1 + m2, U2 = m1 + m6, U4 = U2 + m5,
C12 = U4 + m3, C22 = U4 + m7, D = m5 + m4 into (C11, C12, D, C22); inverse C21 = C22 - D at every level. Checks vs A @ B (float64)."""
import numpy as np
def q(X): r, c = X.shape[0] // 2, X.shape[1] // 2; return X[:r, :c], X[:r, c:], X[r:, :c], X[r:, c:]
def join(a, b, c, d): return np.block([[a, b], [c, d]])
def basis_A(X, L):
    if L == 0: return X.copy()
    x11, x12, x21, x22 = q(X); return join(basis_A(x11, L - 1), basis_A(x12, L - 1), basis_A(x21 + x22, L - 1), basis_A(x22, L - 1))
def basis_B(Y, L):
    if L == 0: return Y.copy()
    y11, y12, y21, y22 = q(Y); return join(basis_B(y11, L - 1), basis_B(y22 - y12, L - 1), basis_B(y21, L - 1), basis_B(y22, L - 1))
def inv_C(Z, L):
    if L == 0: return Z.copy()
    c11, c12, d, c22 = q(Z); return join(inv_C(c11, L - 1), inv_C(c12, L - 1), inv_C(c22 - d, L - 1), inv_C(c22, L - 1))
def core(A, B, L):
    if L == 0: return A @ B
    a11, a12, s1, a22 = q(A); b11, t3, b21, b22 = q(B)
    S2 = s1 - a11; S4 = a12 - S2; S3 = a22 - S2
    T2 = t3 + b11; T4 = T2 - b21; T1 = b22 - T2
    m1 = core(a11, b11, L - 1); m2 = core(a12, b21, L - 1); m3 = core(S4, b22, L - 1); m4 = core(a22, T4, L - 1)
    m5 = core(s1, T1, L - 1); m6 = core(S2, T2, L - 1); m7 = core(S3, t3, L - 1)
    C11 = m1 + m2; U2 = m1 + m6; U4 = U2 + m5; C12 = U4 + m3; C22 = U4 + m7; D = m5 + m4
    return join(C11, C12, D, C22)
rng = np.random.default_rng(0)
for (m, k, n, L) in ((8, 8, 8, 1), (16, 32, 16, 2), (32, 64, 96, 3), (64, 128, 64, 4), (64, 64, 128, 5), (128, 64, 64, 6)):
    A = rng.standard_normal((m, k)); B = rng.standard_normal((k, n))
    C = inv_C(core(basis_A(A, L), basis_B(B, L), L), L); ref = A @ B
    print(f"m,k,n,L = {m},{k},{n},{L}: max rel err {np.abs(C - ref).max() / np.abs(ref).max():.2e}")
