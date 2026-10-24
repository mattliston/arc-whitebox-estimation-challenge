#!/usr/bin/env python3
"""G23 coincidence-resummed one-step per-unit k3 generator (GEMM-class).
k3'_i = sum_{jkl} W_ji W_ki W_li kappa3(x_j,x_k,x_l), x = ReLU(h), h ~ (m, v, C; k3, k4, K21, K22, K31)
split by index coincidence and resummed exactly where bonds are coincident:
  [D] j=k=l : exact univariate post third cumulant (Edgeworth-4 univariate, same channel pattern as the engine's r2)
  [P] j=k!=l: 3 * W_ji^2 W_li * K21post_jl, K21post = the exact bivariate pair series in the OFF-diagonal bond
             (pairmap.pair_M11_M21 -> M21 - 2 r1 M11 - r2 r1 + 2 r1^2 r1), i.e. the engine's emission_K21
  [T] all distinct: Gaussian triple Hermite (Mehler) expansion, words rho_jk^a rho_kl^b rho_jl^c/(a!b!c!) c_j[a+c]c_k[a+b]c_l[b+c];
             PATH words (one bond power zero) evaluated with off-diagonal Hadamard powers R^{oa} (R = rho - I) as GEMMs,
             with the j=l coincident terms removed (they belong to [P]); TRIANGLE words (a,b,c all >= 1) DROPPED (1024^4).
Certificates: (1) triple series vs Monte Carlo on a Gaussian triple; (2) GEMM path-word contraction vs the explicit
triple sum on n=24; (3) univariate k3post (Gaussian part) vs MC.  Then per-layer k3 vs truth on r35."""
import os, sys, json
from math import erf, factorial
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
G19 = HERE.parent / "public_g19_attribution_v1"; G21 = HERE.parent / "public_g21_pair_consumer_v1"
sys.path.insert(0, str(G21)); sys.path.insert(0, str(G19))
from pairmap import columns, pair_M11_M21
from run_attribution import edg2_relu_mean
_erf = np.vectorize(erf)
NB = int(os.environ.get("K3GEN_NB", "6"))       # max bond order a+b of the path words


def hermite_cols(m, v, qmax=30):
    """c1[t] = E[phi(h) He_t(z)], c2, c3 = same for phi^2, phi^3 (engine conventions)."""
    c1, c2, sd, pdf, cdf = columns(m, v, qmax)
    v_ = np.maximum(v, 1e-30); al = m / sd; ma = -al
    He = [np.ones_like(al), ma]
    for k in range(2, qmax + 2):
        He.append(ma * He[-1] - (k - 1) * He[-2])
    raw1 = c1[0]; raw2 = (m * m + v_) * cdf + m * sd * pdf
    raw3 = (m ** 3 + 3 * m * v_) * cdf + (m * m + 2 * v_) * sd * pdf
    c3 = [raw3, 3 * sd * raw2, 6 * v_ * raw1, 6 * v_ * sd * cdf]
    for q in range(4, qmax):
        c3.append(6 * v_ * sd * He[q - 4] * pdf)
    return c1, c2, np.stack(c3), sd, pdf, cdf


def univariate_k3post(m, v, k3, k4):
    """Edgeworth-4 univariate: E[phi^p] = raw_p + (g3/6)c_p[3] + (g4/24)c_p[4] + (g3^2/72)c_p[6]; k3 = r3 - 3 r2 r1 + 2 r1^3."""
    c1, c2, c3, sd, pdf, cdf = hermite_cols(m, v)
    g3, g4 = k3 / sd ** 3, k4 / sd ** 4
    corr = lambda c: (g3 / 6) * c[3] + (g4 / 24) * c[4] + (g3 * g3 / 72) * c[6]
    r1 = edg2_relu_mean(m, v, k3, k4)
    r2 = c2[0] + corr(c2)
    r3 = c3[0] + corr(c3)
    return r3 - 3 * r2 * r1 + 2 * r1 ** 3, r1, r2, r3


def k3_next(m, v, C, k3, k4, K21, K22, K31, W, nb=NB, parts=("D", "P", "T"), nb_min=2):
    """One-step per-unit k3 at the next layer from the pre-state at this layer; W = weights[l+1] (in, out)."""
    n = m.shape[0]
    c1, c2, c3, sd, pdf, cdf = hermite_cols(m, v)
    out = np.zeros(W.shape[1]); comp = {}
    if "D" in parts:
        k3p, r1, r2, r3 = univariate_k3post(m, v, k3, k4)
        comp["D"] = (W ** 3).T @ k3p; out += comp["D"]
    if "P" in parts:
        M11, M21, r1, r2, _ = pair_M11_M21(m, v, C, k3, k4, K21, K22, K31, 2)
        K21p = M21 - 2 * r1[:, None] * M11 - r2[:, None] * r1[None, :] + 2 * (r1 ** 2)[:, None] * r1[None, :]
        np.fill_diagonal(K21p, 0.0)
        comp["P"] = 3.0 * np.einsum("ji,li,jl->i", W ** 2, W, K21p, optimize=True); out += comp["P"]
    if "T" in parts:
        rho = np.clip(C / np.outer(sd, sd), -0.9999, 0.9999); R = rho.copy(); np.fill_diagonal(R, 0.0)
        # Hadamard powers of the off-diagonal correlation, shared across words
        Rp = {1: R}
        for s in range(2, nb + 1):
            Rp[s] = Rp[s - 1] * R
        # u^(a)_{ki} = sum_j R_kj^a W_ji c_j[a]   (GEMM per a)
        U = {a: Rp[a] @ (W * c1[a][:, None]) for a in range(1, nb)}
        T = np.zeros(W.shape[1])
        for a in range(1, nb):
            for b in range(1, nb + 1 - a):
                if a + b < nb_min:
                    continue          # words below nb_min are already in the card (its CP births are the (1,1) word)
                coef = 3.0 / (factorial(a) * factorial(b))
                hub = W * c1[a + b][:, None]                                # W_ki c_k[a+b]
                T += coef * np.einsum("ki,ki,ki->i", hub, U[a], U[b])      # path words, hub k
                # remove the coincident j=l terms (they are plane-class): sum_{j,k} W_ji^2 c_j[a]c_j[b] W_ki c_k[a+b] R_jk^{a+b}
                T -= coef * np.einsum("ki,ki->i", hub, Rp[a + b] @ ((W ** 2) * (c1[a] * c1[b])[:, None]))
        comp["T"] = T; out += T
    return out, comp


# ---------------- certificates ----------------
def cert_triple_series(seed=0, nmc=4_000_000):
    """(1) Gaussian triple: kappa3(x_j,x_k,x_l) series (path + triangle words, a+b+c<=12) vs MC."""
    rng = np.random.default_rng(seed)
    m = np.array([0.3, -0.2, 0.1]); v = np.array([1.0, 0.7, 1.3])
    L = np.array([[1, 0, 0], [0.4, 1, 0], [-0.3, 0.5, 1.0]]); Cc = L @ L.T
    d = np.sqrt(np.diag(Cc)); Cc = Cc / np.outer(d, d) * np.outer(np.sqrt(v), np.sqrt(v))
    z = rng.standard_normal((nmc, 3)) @ np.linalg.cholesky(Cc).T + m
    x = np.maximum(z, 0)
    mc = (x[:, 0] * x[:, 1] * x[:, 2]).mean() - x[:, 0].mean() * (x[:, 1] * x[:, 2]).mean() \
        - x[:, 1].mean() * (x[:, 0] * x[:, 2]).mean() - x[:, 2].mean() * (x[:, 0] * x[:, 1]).mean() + 2 * np.prod(x.mean(0))
    c1, _, _, sd, _, _ = hermite_cols(m, v)
    rho = Cc / np.outer(sd, sd)
    ser = 0.0; path = 0.0
    for a in range(0, 13):
        for b in range(0, 13 - a):
            for c in range(0, 13 - a - b):
                iso = (a == 0 and b == 0) or (a == 0 and c == 0) or (b == 0 and c == 0)
                if iso: continue
                w = c1[a + c][0] * c1[a + b][1] * c1[b + c][2] * rho[0, 1] ** a * rho[1, 2] ** b * rho[0, 2] ** c / (factorial(a) * factorial(b) * factorial(c))
                ser += w
                if min(a, b, c) == 0: path += w
    return {"mc": float(mc), "series_all": float(ser), "series_path_only": float(path), "rel_err_all": float(abs(ser - mc) / abs(mc)), "triangle_share": float((ser - path) / ser)}


def cert_gemm_path(n=24, seed=1, nb=4):
    """(2) GEMM path-word contraction == explicit sum over distinct triples (path words only)."""
    rng = np.random.default_rng(seed)
    m = rng.normal(0, 0.3, n); v = rng.uniform(0.5, 1.5, n)
    A = rng.normal(size=(n, n)) / np.sqrt(n); C = A @ A.T; C = C / np.sqrt(np.outer(np.diag(C), np.diag(C))) * np.sqrt(np.outer(v, v))
    W = rng.normal(size=(n, 5)) / np.sqrt(n)
    c1, _, _, sd, _, _ = hermite_cols(m, v)
    rho = C / np.outer(sd, sd)
    ref = np.zeros(5)
    for i in range(5):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    if j == k or k == l or j == l: continue
                    s = 0.0
                    for a in range(1, nb):
                        for b in range(1, nb + 1 - a):
                            s += c1[a][j] * c1[a + b][k] * c1[b][l] * rho[j, k] ** a * rho[k, l] ** b / (factorial(a) * factorial(b))
                    ref[i] += 3.0 * W[j, i] * W[k, i] * W[l, i] * s
    got, comp = k3_next(m, v, C, np.zeros(n), np.zeros(n), np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n)), W, nb=nb, parts=("T",))
    return {"rel_err": float(np.linalg.norm(got - ref) / np.linalg.norm(ref))}


def cert_univariate(seed=2, nmc=4_000_000):
    """(3) Gaussian univariate k3post vs MC (k3=k4=0)."""
    rng = np.random.default_rng(seed)
    m = np.array([0.2, -0.4, 0.0]); v = np.array([1.0, 0.6, 2.0])
    z = rng.standard_normal((nmc, 3)) * np.sqrt(v) + m; x = np.maximum(z, 0)
    mc = (x ** 3).mean(0) - 3 * (x ** 2).mean(0) * x.mean(0) + 2 * x.mean(0) ** 3
    k3p = univariate_k3post(m, v, np.zeros(3), np.zeros(3))[0]
    return {"rel_err": float(np.linalg.norm(k3p - mc) / np.linalg.norm(mc))}


if __name__ == "__main__":
    print(json.dumps({"cert_triple_series": cert_triple_series(), "cert_gemm_path": cert_gemm_path(), "cert_univariate": cert_univariate()}, indent=1))
