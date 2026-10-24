#!/usr/bin/env python3
"""CERTIFICATE v2 (tiny world): first-order response of the post kappa3 of distinct triples to the pre
kappa4, with the pre kappa4 known ANALYTICALLY (no estimation noise). Pre: h = A z + m0 + s * He3(z_1) * svec.
Exact joint cumulants of h through order 4 via the polynomial structure (Gaussian z): kappa3 = 0 exactly;
kappa4_ijkl = s * 6 * [A_i1 A_j1 A_k1 sv_l + 3 perms] + s^2 * 36 * [A_i1 A_j1 sv_k sv_l + 5 perms]
              + s^3 * c3 * [A_i1 sv_j sv_k sv_l + 3 perms] + s^4 * c4 * sv_i sv_j sv_k sv_l,
   c3 = kappa(z1, He3, He3, He3)-type joint cumulant = E[z1 He3^3] = 3*E[z^4 He3^2]... computed numerically
   from 1D Gaussian moments below; c4 = kappa4(He3(z)) = E[He3^4] - 3 E[He3^2]^2.
Measured response: (MC post kappa3 - Gaussian triple series) at s minus the same at s=0, same draws, N chunked.
Prediction: sum over classes with multiplicities 1/2 (2,1,1), 1/6 (3,1), 1/4 (2,2), 1/24 (4) and conditional-Gaussian kernels."""
import json, sys
import numpy as np
from math import erf, factorial
sys.path.insert(0, ".")
from k3gen import hermite_cols
rng = np.random.default_rng(11)
n, N, CH = 32, 100_000_000, 2_000_000
A = rng.normal(size=(n, n)) / np.sqrt(n); A += 0.35 * np.outer(np.ones(n), np.ones(n)) / n * np.sqrt(n)
m0 = rng.normal(0, 0.3, n); sv = rng.normal(size=n) * 0.12; SP = float(__import__("os").environ.get("SP", "0.1"))
trip = [tuple(sorted(rng.choice(n, 3, replace=False))) for _ in range(60)]
# --- 1D Gaussian moments for He3 = z^3 - 3z
def gm(k):  # E[z^k]
    return 0.0 if k % 2 else float(np.prod(np.arange(1, k, 2))) if k > 0 else 1.0
def E_z_he3(pz, ph):  # E[z^pz * He3(z)^ph]
    from itertools import product
    tot = 0.0
    # He3^ph = sum over binomial expansion of (z^3 - 3z)^ph
    for j in range(ph + 1):
        coef = factorial(ph) / (factorial(j) * factorial(ph - j)) * (1.0) ** j * (-3.0) ** (ph - j)
        tot += coef * gm(pz + 3 * j + (ph - j))
    return tot
E1 = {(pz, ph): E_z_he3(pz, ph) for pz in range(5) for ph in range(5)}
def kappa4_exact(i, j, k, l, s):
    """joint cumulant of h_i..h_l with h = A z + s He3(z_1) sv (Gaussian parts contribute nothing beyond order 2)."""
    idx = [i, j, k, l]
    # write each h_u = g_u + s sv_u He3(z1), g Gaussian; kappa4 = sum over subsets S (which slots take the He3 factor)
    # of s^|S| prod sv_u * joint cumulant(g_{rest}, He3,...,He3). g_rest depends on z1 through A_u1 z1 and on other z's
    # (independent of He3 -> only the z1-components couple). joint cumulant of (A_u1 z1 ...) and He3 factors = cumulant of
    # polynomials of a single Gaussian z1: compute via moments of z1 and the cumulant expansion for <= 4 variables.
    from itertools import combinations
    tot = 0.0
    for r in range(1, 5):
        for S in combinations(range(4), r):
            coef = s ** r * np.prod([sv[idx[u]] for u in S])
            rest = [u for u in range(4) if u not in S]
            # variables: X_u = A_{idx[u],1} z1 for u in rest (only the z1 part couples with He3; other z components are
            # independent of He3 and of each other's z1 parts -> they contribute only to 2nd-order cumulants among g's,
            # which never enter a mixed cumulant with an He3 factor), Y = He3(z1) for u in S
            avec = [A[idx[u], 0] for u in rest]
            # joint cumulant kappa(X_1..X_p, Y..Y (r times)) of polynomials in z1: use the moment->cumulant formula for the
            # 4-variable cumulant via all set partitions
            vars_ = [("z", a) for a in avec] + [("h", 1.0)] * r
            def mom(sub):
                pz = sum(1 for v in sub if v[0] == "z"); ph = sum(1 for v in sub if v[0] == "h")
                c = np.prod([v[1] for v in sub if v[0] == "z"]) if pz else 1.0
                return c * E1[(pz, ph)]
            # set partitions of 4 elements (Bell 15)
            def partitions(lst):
                if not lst: yield []; return
                first = lst[0]
                for smaller in partitions(lst[1:]):
                    for i2, sub in enumerate(smaller):
                        yield smaller[:i2] + [[first] + sub] + smaller[i2 + 1:]
                    yield [[first]] + smaller
            cum = 0.0
            for part in partitions(list(range(4))):
                b = len(part); term = (-1) ** (b - 1) * factorial(b - 1)
                for blk in part:
                    term *= mom([vars_[q] for q in blk])
                cum += term
            tot += coef * cum
    return tot
# --- streaming MC for post kappa3 (same draws for s and 0)
def stream(sval, seed=777):
    rng_ = np.random.default_rng(seed)
    S1 = np.zeros(n); S2 = np.zeros((n, n)); X1 = np.zeros(n); E3 = np.zeros(len(trip)); X2 = {}
    for _ in range(N // CH):
        z = rng_.standard_normal((CH, n))
        h = z @ A.T + m0 + sval * (z[:, 0] ** 3 - 3 * z[:, 0])[:, None] * sv[None, :]
        S1 += h.sum(0); S2 += h.T @ h
        x = np.maximum(h, 0.0); X1 += x.sum(0)
        for t, (a, b, c) in enumerate(trip):
            E3[t] += (x[:, a] * x[:, b] * x[:, c]).sum()
            for pr in [(a, b), (a, c), (b, c)]:
                X2[pr] = X2.get(pr, 0.0) + (x[:, pr[0]] * x[:, pr[1]]).sum()
    mu_ = S1 / N; C_ = S2 / N - np.outer(mu_, mu_); X1 /= N; X2 = {k: v / N for k, v in X2.items()}; E3 /= N
    k3 = np.array([E3[t] - X1[a] * X2[(b, c)] - X1[b] * X2[(a, c)] - X1[c] * X2[(a, b)] + 2 * X1[a] * X1[b] * X1[c] for t, (a, b, c) in enumerate(trip)])
    return mu_, C_, k3
def gauss_series(mu_, C_, nmax=14):
    sd_ = np.sqrt(np.diag(C_)); rho_ = C_ / np.outer(sd_, sd_); c1, _, _, _, _, _ = hermite_cols(mu_, np.diag(C_)); out = []
    for a, b, c in trip:
        tot = 0.0
        for p_ in range(nmax + 1):
            for q_ in range(nmax + 1 - p_):
                for r_ in range(nmax + 1 - p_ - q_):
                    if (p_ == 0 and q_ == 0) or (p_ == 0 and r_ == 0) or (q_ == 0 and r_ == 0): continue
                    tot += c1[p_ + r_][a] * c1[p_ + q_][b] * c1[q_ + r_][c] * rho_[a, b] ** p_ * rho_[b, c] ** q_ * rho_[a, c] ** r_ / (factorial(p_) * factorial(q_) * factorial(r_))
        out.append(tot)
    return np.array(out)
mu0, C0, k30 = stream(0.0); mus, Cs, k3s = stream(SP)
resp = (k3s - gauss_series(mus, Cs)) - (k30 - gauss_series(mu0, C0))
# --- kernels at the perturbed Gaussian reference (mus, Cs)
mu, C = mus, Cs; sd = np.sqrt(np.diag(C))
xg, wg = np.polynomial.hermite_e.hermegauss(120); wg = wg / wg.sum()
def E_cond(a, b, c, ha, fb, fc):
    idx = [b, c]; Cba = C[idx, a]; Cbb = C[np.ix_(idx, idx)]
    mcond = mu[idx] + Cba * (ha - mu[a]) / C[a, a]; Ccond = Cbb - np.outer(Cba, Cba) / C[a, a]
    L = np.linalg.cholesky(Ccond + 1e-12 * np.eye(2)); X, Y = np.meshgrid(xg, xg, indexing="ij")
    U = np.stack([X.ravel(), Y.ravel()], 1) @ L.T + mcond; Wt = (wg[:, None] * wg[None, :]).ravel()
    return float((Wt * fb(U[:, 0]) * fc(U[:, 1])).sum())
pdf = lambda a, ha: np.exp(-0.5 * ((ha - mu[a]) / sd[a]) ** 2) / (np.sqrt(2 * np.pi) * sd[a])
step = lambda x: (x > 0).astype(float); relu = lambda x: np.maximum(x, 0.0)
def kern(a, b, c, da, fb, fc):
    g = lambda ha: pdf(a, ha) * E_cond(a, b, c, ha, fb, fc); eps = 0.02 * sd[a]
    if da == 2: return g(0.0)
    if da == 3: return -(g(eps) - g(-eps)) / (2 * eps)
    return (g(eps) - 2 * g(0.0) + g(-eps)) / eps ** 2
pred = np.zeros(len(trip)); p211 = np.zeros(len(trip)); p211s = np.zeros(len(trip))
for t, (a, b, c) in enumerate(trip):
    for (x, y, zz) in [(a, b, c), (b, a, c), (c, a, b)]:
        k = kappa4_exact(x, x, y, zz, SP); v = 0.5 * k * kern(x, y, zz, 2, step, step); p211[t] += v; pred[t] += v
        p211s[t] += 0.5 * k * pdf(x, 0.0) * 0.5 * (1 + erf((mu[y] / sd[y]) / np.sqrt(2))) * 0.5 * (1 + erf((mu[zz] / sd[zz]) / np.sqrt(2)))
        pred[t] += (1 / 6) * kappa4_exact(x, x, x, y, SP) * kern(x, y, zz, 3, step, relu) + (1 / 6) * kappa4_exact(x, x, x, zz, SP) * kern(x, zz, y, 3, step, relu)
        pred[t] += (1 / 24) * kappa4_exact(x, x, x, x, SP) * kern(x, y, zz, 4, relu, relu)
    for (x, y, zz) in [(a, b, c), (a, c, b), (b, c, a)]:
        idx = [x, y]; Cxy = C[np.ix_(idx, idx)]; mxy = mu[idx]; det = np.linalg.det(Cxy)
        p2 = np.exp(-0.5 * float(mxy @ np.linalg.solve(Cxy, mxy))) / (2 * np.pi * np.sqrt(det))
        Cz = C[zz, idx]; mcond = mu[zz] + Cz @ np.linalg.solve(Cxy, -mxy); vcond = C[zz, zz] - Cz @ np.linalg.solve(Cxy, Cz)
        sdz = np.sqrt(max(vcond, 1e-12)); al = mcond / sdz
        Erelu = sdz * np.exp(-0.5 * al * al) / np.sqrt(2 * np.pi) + mcond * 0.5 * (1 + erf(al / np.sqrt(2)))
        pred[t] += 0.25 * kappa4_exact(x, x, y, y, SP) * p2 * Erelu
out = {"N": N, "resp_rms": float(np.sqrt((resp ** 2).mean())), "corr_all": float(np.corrcoef(pred, resp)[0, 1]),
       "beta_all": float(np.dot(pred, resp) / np.dot(pred, pred)), "rel_err_all": float(np.linalg.norm(pred - resp) / np.linalg.norm(resp)),
       "corr_211": float(np.corrcoef(p211, resp)[0, 1]), "share_211": float(np.linalg.norm(p211) / np.linalg.norm(pred)),
       "sep_over_exact_211": float(np.dot(p211s, p211) / np.dot(p211, p211))}
print(json.dumps(out, indent=1))
