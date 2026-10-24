#!/usr/bin/env python3
"""CERTIFICATE (tiny world, n=32): first-order response of the POST third cumulant of a DISTINCT
triple (a,b,c) to the PRE fourth cumulant, all index classes with multiplicities from the
Edgeworth identity E[f] = E_G[f] + sum kappa4_ijkl/4! E_G[d_ijkl f]:
  (2,1,1): 1/2 k4_aabc E_G[phi''_a phi'_b phi'_c] (+ b,c doubled)   <- the class the engine lacks
  (3,1):   1/6 k4_aaab E_G[phi'''_a phi'_b phi_c] (+ perms)
  (2,2):   1/4 k4_aabb E_G[phi''_a phi''_b phi_c] (+ perms)
  (4):     1/24 k4_aaaa E_G[phi''''_a phi_b phi_c] (+ perms)
Kernels: phi' = step, phi'' = delta, phi''' = delta', phi'''' = delta'' -> conditional Gaussian
expectations at h_a = 0 (2D Gauss-Hermite on (h_b,h_c)|h_a), derivative kernels by central
finite differences in the conditioning value.  Truth: pre h = A z + s * He3(z_1) e_s (kappa3 = 0
exactly, kappa4 linear in s + O(s^2)); post kappa3 response measured as the SAME-DRAW difference
between s and 0 (noise-cancelled), MC N samples.  Also reports the separable approximation of the
(2,1,1) kernel (pdf_a/sd_a * cdf_b * cdf_c) vs the exact conditional orthant.  Usage: cert_cum4_feed.py"""
import json, sys
import numpy as np
from math import erf
rng = np.random.default_rng(11)
n, N = 32, 6_000_000
A = rng.normal(size=(n, n)) / np.sqrt(n); A += 0.35 * np.outer(np.ones(n), np.ones(n)) / n * np.sqrt(n)   # positive-ish correlations
m0 = rng.normal(0, 0.3, n)
svec = rng.normal(size=n) * 0.12
def draw(s):
    z = rng.standard_normal((N, n))
    h = z @ A.T + m0 + s * (z[:, 0] ** 3 - 3 * z[:, 0])[:, None] * svec[None, :]
    return h
# same draws for s and 0: regenerate with the same seed state
st = rng.bit_generator.state
SPERT = 0.3
h1 = draw(0.0); rng.bit_generator.state = st; h2 = draw(SPERT)
def cum3_triples(h, trip):
    x = np.maximum(h, 0.0); mu = x.mean(0)
    out = []
    for a, b, c in trip:
        xa, xb, xc = x[:, a], x[:, b], x[:, c]
        E3 = (xa * xb * xc).mean(); Eab, Eac, Ebc = (xa * xb).mean(), (xa * xc).mean(), (xb * xc).mean()
        out.append(E3 - mu[a] * Ebc - mu[b] * Eac - mu[c] * Eab + 2 * mu[a] * mu[b] * mu[c])
    return np.array(out)
trip = [tuple(sorted(rng.choice(n, 3, replace=False))) for _ in range(60)]
from math import factorial as _fact
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from k3gen import hermite_cols
def gauss_triple(h, trip, nmax=14):
    mu_ = h.mean(0); Cc = np.cov(h.T); sd_ = np.sqrt(np.diag(Cc)); rho_ = Cc / np.outer(sd_, sd_)
    c1, _, _, _, _, _ = hermite_cols(mu_, np.diag(Cc)); out = []
    for a, b, c in trip:
        tot = 0.0
        for p_ in range(nmax + 1):
            for q_ in range(nmax + 1 - p_):
                for r_ in range(nmax + 1 - p_ - q_):
                    if (p_ == 0 and q_ == 0) or (p_ == 0 and r_ == 0) or (q_ == 0 and r_ == 0): continue
                    tot += c1[p_ + r_][a] * c1[p_ + q_][b] * c1[q_ + r_][c] * rho_[a, b] ** p_ * rho_[b, c] ** q_ * rho_[a, c] ** r_ / (_fact(p_) * _fact(q_) * _fact(r_))
        out.append(tot)
    return np.array(out)
# NON-Gaussian response: (MC - Gaussian series) at s, minus the same at s=0 (removes the residual series error, same draws)
resp_mc = (cum3_triples(h2, trip) - gauss_triple(h2, trip)) - (cum3_triples(h1, trip) - gauss_triple(h1, trip))
base = cum3_triples(h1, trip)
# pre kappa4 of h2 (exact to O(s^2)): empirical from the same samples
hc = h2 - h2.mean(0)
C = hc.T @ hc / N
def k4_entry(i, j, k, l):
    return (hc[:, i] * hc[:, j] * hc[:, k] * hc[:, l]).mean() - C[i, j] * C[k, l] - C[i, k] * C[j, l] - C[i, l] * C[j, k]
# Gaussian reference for kernels: mean m0-ish (use empirical mean of h2) and covariance C
mu = h2.mean(0); sd = np.sqrt(np.diag(C))
xg, wg = np.polynomial.hermite_e.hermegauss(120); wg = wg / wg.sum()
def cond_bc(a, b, c, ha):
    """conditional mean/cov of (h_b,h_c) given h_a = ha"""
    idx = [b, c]; Cba = C[idx, a]; Cbb = C[np.ix_(idx, idx)]
    mcond = mu[idx] + Cba * (ha - mu[a]) / C[a, a]
    Ccond = Cbb - np.outer(Cba, Cba) / C[a, a]
    return mcond, Ccond
def E_cond(a, b, c, ha, fb, fc):
    """E[fb(h_b) fc(h_c) | h_a = ha] by 2D Gauss-Hermite"""
    mcond, Ccond = cond_bc(a, b, c, ha); L = np.linalg.cholesky(Ccond + 1e-12 * np.eye(2))
    X, Y = np.meshgrid(xg, xg, indexing="ij"); U = np.stack([X.ravel(), Y.ravel()], 1) @ L.T + mcond
    Wt = (wg[:, None] * wg[None, :]).ravel()
    return float((Wt * fb(U[:, 0]) * fc(U[:, 1])).sum())
pdf = lambda a, ha: np.exp(-0.5 * ((ha - mu[a]) / sd[a]) ** 2) / (np.sqrt(2 * np.pi) * sd[a])
step = lambda x: (x > 0).astype(float); relu = lambda x: np.maximum(x, 0.0); one = lambda x: np.ones_like(x)
def kern(a, b, c, da, fb, fc):
    """E_G[phi^(da)_a fb(h_b) fc(h_c)] with da in {2,3,4}: delta, delta', delta'' at h_a = 0 (finite differences)"""
    g = lambda ha: pdf(a, ha) * E_cond(a, b, c, ha, fb, fc)
    eps = 0.02 * sd[a]
    if da == 2: return g(0.0)
    if da == 3: return -(g(eps) - g(-eps)) / (2 * eps)
    if da == 4: return (g(eps) - 2 * g(0.0) + g(-eps)) / eps ** 2
pred = np.zeros(len(trip)); pred211 = np.zeros(len(trip)); pred211_sep = np.zeros(len(trip))
for t, (a, b, c) in enumerate(trip):
    for (x, y, zz) in [(a, b, c), (b, a, c), (c, a, b)]:            # doubled index x, others y, zz
        k = k4_entry(x, x, y, zz)
        v211 = 0.5 * k * kern(x, y, zz, 2, step, step); pred211[t] += v211
        pred211_sep[t] += 0.5 * k * pdf(x, 0.0) * 0.5 * (1 + erf(((mu[y]) / sd[y]) / np.sqrt(2))) * 0.5 * (1 + erf(((mu[zz]) / sd[zz]) / np.sqrt(2)))
        pred[t] += v211
        pred[t] += (1 / 6) * k4_entry(x, x, x, y) * kern(x, y, zz, 3, step, relu) + (1 / 6) * k4_entry(x, x, x, zz) * kern(x, zz, y, 3, step, relu)
        pred[t] += (1 / 24) * k4_entry(x, x, x, x) * kern(x, y, zz, 4, relu, relu)
    for (x, y, zz) in [(a, b, c), (a, c, b), (b, c, a)]:            # (2,2): doubled x and y, spectator zz
        pred[t] += 0.25 * k4_entry(x, x, y, y) * (pdf(x, 0.0) * E_cond(x, y, zz, 0.0, lambda u: np.exp(-0.5 * 0) * 0 + 0, relu) if False else 0.0)
# (2,2) kernel E[delta_x delta_y phi_zz] needs the joint density at (0,0): do it explicitly
for t, (a, b, c) in enumerate(trip):
    for (x, y, zz) in [(a, b, c), (a, c, b), (b, c, a)]:
        idx = [x, y]; Cxy = C[np.ix_(idx, idx)]; mxy = mu[idx]
        det = np.linalg.det(Cxy); q = float(mxy @ np.linalg.solve(Cxy, mxy))
        p2 = np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(det))
        Cz = C[zz, idx]; mcond = mu[zz] + Cz @ np.linalg.solve(Cxy, -mxy); vcond = C[zz, zz] - Cz @ np.linalg.solve(Cxy, Cz)
        sdz = np.sqrt(max(vcond, 1e-12)); al = mcond / sdz
        Erelu = sdz * np.exp(-0.5 * al * al) / np.sqrt(2 * np.pi) + mcond * 0.5 * (1 + erf(al / np.sqrt(2)))
        pred[t] += 0.25 * k4_entry(x, x, y, y) * p2 * Erelu
rel = lambda p: float(np.linalg.norm(p - resp_mc) / np.linalg.norm(resp_mc))
out = {"resp_mc_rms": float(np.sqrt((resp_mc ** 2).mean())), "base_k3_rms": float(np.sqrt((base ** 2).mean())),
       "corr_all_classes": float(np.corrcoef(pred, resp_mc)[0, 1]), "rel_err_all_classes": rel(pred),
       "beta_all": float(np.dot(pred, resp_mc) / np.dot(pred, pred)),
       "corr_211_only": float(np.corrcoef(pred211, resp_mc)[0, 1]), "share_211_of_pred": float(np.linalg.norm(pred211) / np.linalg.norm(pred)),
       "sep_vs_exact_211_ratio": float(np.dot(pred211_sep, pred211) / np.dot(pred211, pred211)), "corr_sep_exact_211": float(np.corrcoef(pred211_sep, pred211)[0, 1])}
print(json.dumps(out, indent=1))
