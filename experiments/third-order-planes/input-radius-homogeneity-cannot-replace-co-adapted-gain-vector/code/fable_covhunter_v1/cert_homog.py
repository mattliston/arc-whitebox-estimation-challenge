#!/usr/bin/env python3
"""TAG fable-covhunter (CH-37): numerical certificate of the input-radial term. For a bias-free ReLU net and x ~ N(0, I_n), z = r g(xhat) with
r = |x| independent of xhat, so kappa3(z) = E r^3 kappa3(g) + (1/E r^2) Sym(C (x) mu) + O(mu^3/n^2), C and mu being z's covariance and mean.
Tiny MC (n = 16 inputs, two hidden layers of 16): empirical kappa3(z)_(a,a,c) vs the right-hand side built from the empirical moments of g
= z / r and the exact chi moments (E r = sqrt2 G((n+1)/2)/G(n/2), E r^2 = n, E r^3 = (n+1) E r)."""
import math
import numpy as np
from scipy.special import gammaln
rng = np.random.default_rng(0); n, N = 16, 4_000_000
W1 = rng.standard_normal((n, n)) * math.sqrt(2 / n); W2 = rng.standard_normal((n, n)) * math.sqrt(2 / n)
x = rng.standard_normal((N, n)); r = np.linalg.norm(x, axis=1)
z = np.maximum(x @ W1, 0) @ W2                  # layer-2 pre-activation of a bias-free ReLU net
g = z / r[:, None]
Er = math.sqrt(2) * math.exp(gammaln((n + 1) / 2) - gammaln(n / 2)); Er2 = n; Er3 = (n + 1) * Er
def k21(y):                                      # kappa3(y_a, y_a, y_c)
    m = y.mean(0); yc = y - m
    return np.einsum('ia,ic->ac', yc * yc, yc) / len(y)
mu = z.mean(0); C = np.cov(z.T, bias=True); v = np.diag(C)
lhs = k21(z)
sym = v[:, None] * mu[None, :] + 2 * C * mu[:, None]
rhs = Er3 * k21(g) + sym / Er2
off = ~np.eye(n, dtype=bool)
print(f'|lhs - rhs| / |lhs| (off-diag) = {np.linalg.norm((lhs - rhs)[off]) / np.linalg.norm(lhs[off]):.4f}')
print(f'input-radial share of |lhs| (off-diag): |sym/n| / |lhs| = {np.linalg.norm((sym / Er2)[off]) / np.linalg.norm(lhs[off]):.4f}')
# noise scale: split-half difference of lhs
h = N // 2; noise = np.linalg.norm((k21(z[:h]) - k21(z[h:]))[off]) / 2 / np.linalg.norm(lhs[off])
print(f'MC noise of lhs (half-split) = {noise:.4f}')
