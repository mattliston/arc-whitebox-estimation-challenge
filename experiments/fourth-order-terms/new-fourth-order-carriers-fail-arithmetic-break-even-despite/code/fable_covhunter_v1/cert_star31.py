#!/usr/bin/env python3
"""TAG fable-covhunter (CH-41b): certificate of the plane-only (3,1) transport formula of star31_transport.py. A symmetric 4-tensor is built
with support ONLY on (a,a,a,a), (a,a,a,c) and (a,a,c,c) patterns (n = 6), pushed exactly by einsum through W, and its (3,1) plane
kappa4(h_b,h_b,h_b,h_d) compared with the closed form."""
import itertools
import numpy as np
rng = np.random.default_rng(0); n = 6
k4 = rng.standard_normal(n); K31 = rng.standard_normal((n, n)); K22 = rng.standard_normal((n, n)); K22 = (K22 + K22.T) / 2
np.fill_diagonal(K31, 0.); np.fill_diagonal(K22, 0.)
T = np.zeros((n,) * 4)
for i, j, k, l in itertools.product(range(n), repeat=4):
    idx = (i, j, k, l); u = sorted(set(idx))
    if len(u) == 1: T[idx] = k4[i]
    elif len(u) == 2:
        a, c = u; ca, cc = idx.count(a), idx.count(c)
        if ca == 3: T[idx] = K31[a, c]
        elif cc == 3: T[idx] = K31[c, a]
        else: T[idx] = K22[a, c]
W = rng.standard_normal((n, n))
exact4 = np.einsum('abcd,ai,bj,ck,dl->ijkl', T, W, W, W, W)
exact = np.array([[exact4[b, b, b, d] for d in range(n)] for b in range(n)])
w2, w3 = W * W, W * W * W; M = K31 @ W
formula = w3.T @ (k4[:, None] * W) + w3.T @ M + 3 * ((w2 * M).T @ W) + 3 * (((w2.T @ K22) * W.T) @ W)
print('max abs diff exact vs formula:', float(np.abs(exact - formula).max()), ' scale', float(np.abs(exact).max()))
