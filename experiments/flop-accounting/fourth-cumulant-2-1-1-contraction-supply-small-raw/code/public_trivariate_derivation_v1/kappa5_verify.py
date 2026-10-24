#!/usr/bin/env python
"""
TAG: public_trivariate_derivation_v1/kappa5_verify.py
Brute verification + magnitude gate for KAPPA5_CLOSURE.md.

T1 closure, univariate:  kappa5 = 5 k3 k4 / v - (40/9) k3^3 / v^2  (standardized: v=1)
T2 closure, bivariate:   K41/K32 plane laws (diagonal-chaos frame) vs exact joint cumulants
T3 emission:             post-relu k4 from pre-tilts; the He5 kappa5-column (and the full
                         third-order column set He5/He7/He9) must shrink the residual by an order
T4 magnitude gate:       at the measured ladder tilts (|g3| 0.1-0.25, g4 0.03-0.07), is the
                         kappa5-fed post-k4 correction the size of the 0.95@L5->0.84@L15 decay
                         (~1.2 percent/layer relative)?  FAIL-FAST if 10x too small.

Pair/marginal rig reused from l5_verify (exact cumulants; eps-graded chaos), c-tables from
verify.py extended to p<=4.  numpy + math only.
"""
import json
import math
import sys
from math import factorial

import numpy as np

sys.path.insert(0, "/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_trivariate_derivation_v1")
import verify as V
import l5_verify as L5

HERE = "/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_trivariate_derivation_v1"


# ---------------------------------------------------------------- c-table p<=4 (exact recursion)
def ctable_p4(mu, sd, qmax=10):
    w = -mu / sd
    nmax = qmax + 5
    I = np.zeros(nmax + 1)
    I[0] = V.Phi(-w)
    pw = V.phi(w)
    I[1] = pw
    for n in range(2, nmax + 1):
        I[n] = w ** (n - 1) * pw + (n - 1) * I[n - 2]
    lin = np.array([mu, sd])
    pows = {1: lin}
    for p in (2, 3, 4):
        pows[p] = np.convolve(pows[p - 1], lin)
    c = {p: np.zeros(qmax + 1) for p in (1, 2, 3, 4)}
    for p in c:
        for q in range(qmax + 1):
            poly = np.convolve(pows[p], V.HE[q])
            c[p][q] = float(np.dot(poly, I[: len(poly)]))
    return c


def stein_check(log):
    """c[3,q]=6 sd^3 He_{q-4}(w) phi(w) (q>=4), c[4,q]=24 sd^4 He_{q-5}(w) phi(w) (q>=5)."""
    worst = 0.0
    for (mu, sd) in [(0.3, 0.9), (-0.5, 1.2), (0.8, 0.7)]:
        w = -mu / sd
        c = ctable_p4(mu, sd)
        for q in range(4, 10):
            ref = 6 * sd ** 3 * V.he_val(q - 4, w) * V.phi(w)
            worst = max(worst, abs(c[3][q] - ref) / max(abs(ref), 1e-12))
        for q in range(5, 10):
            ref = 24 * sd ** 4 * V.he_val(q - 5, w) * V.phi(w)
            worst = max(worst, abs(c[4][q] - ref) / max(abs(ref), 1e-12))
    log(f"c-table p<=4: recursion vs Stein closed forms max rel dev {worst:.2e}")
    assert worst < 1e-9, worst


# ---------------------------------------------------------------- closures
def kappa5_closure(k3, k4):
    return 5.0 * k3 * k4 - (40.0 / 9.0) * k3 ** 3


def K41_closure(k3i, k4i, k3j, k4j, rho):
    return (rho * (3.0 * k3i * k4i - (28.0 / 9.0) * k3i ** 3)
            + rho ** 2 * k3j * k4i
            + rho ** 3 * (k3i * k4j - (4.0 / 3.0) * k3i * k3j ** 2))


def K32_closure(k3i, k4i, k3j, k4j, rho):
    ai, aj = k3i / 6.0, k3j / 6.0
    bi = (k4i - (4.0 / 3.0) * k3i ** 2) / 24.0
    bj = (k4j - (4.0 / 3.0) * k3j ** 2) / 24.0
    return (48.0 * ai ** 3 * rho ** 2
            + 6.0 * ai ** 2 * aj * (32.0 * rho ** 3 + 16.0 * rho)
            + 3.0 * ai * aj ** 2 * (16.0 * rho ** 4 + 32.0 * rho ** 2)
            + 216.0 * ai * bi * rho ** 2
            + 6.0 * bi * aj * (24.0 * rho ** 3 + 12.0 * rho)
            + 6.0 * ai * bj * (12.0 * rho ** 4 + 24.0 * rho ** 2)
            + 72.0 * aj * bj * rho ** 3)


def std_pair_state(Pi, Pj, rho_g):
    kap = L5.cumulants_from_moments(L5.joint_moments(Pi, Pj, rho_g))
    si, sj = math.sqrt(kap[2, 0]), math.sqrt(kap[0, 2])
    out = {"mi": kap[1, 0], "mj": kap[0, 1], "si": si, "sj": sj,
           "rho": kap[1, 1] / (si * sj)}
    for a in range(6):
        for b in range(6 - a):
            if 3 <= a + b <= 5:
                out[(a, b)] = kap[a, b] / si ** a / sj ** b
    return out


def t1_t2_closure(log):
    rows = []
    for eps in [0.32, 0.2263, 0.16, 0.1131, 0.08, 0.0566, 0.04]:
        Pi = L5.pair_poly(0.3, 0.9, eps * 0.6, eps * eps * 0.4)
        Pj = L5.pair_poly(-0.2, 1.1, eps * -0.45, eps * eps * 0.5)
        st = std_pair_state(Pi, Pj, 0.45)
        k3i, k4i, k5i = st[(3, 0)], st[(4, 0)], st[(5, 0)]
        k3j, k4j = st[(0, 3)], st[(0, 4)]
        c5 = kappa5_closure(k3i, k4i)
        c41 = K41_closure(k3i, k4i, k3j, k4j, st["rho"])
        c32 = K32_closure(k3i, k4i, k3j, k4j, st["rho"])
        rows.append(dict(eps=eps, k3=k3i, k4=k4i,
                         k5_exact=k5i, k5_closure=c5, k5_rel=(c5 - k5i) / k5i,
                         K41_exact=st[(4, 1)], K41_closure=c41,
                         K41_rel=(c41 - st[(4, 1)]) / st[(4, 1)],
                         K32_exact=st[(3, 2)], K32_closure=c32,
                         K32_rel=(c32 - st[(3, 2)]) / st[(3, 2)]))
        log(f"T1/T2 eps={eps:.4f}: k5 exact {k5i:+.5e} closure {c5:+.5e} rel {rows[-1]['k5_rel']:+.3f} | "
            f"K41 rel {rows[-1]['K41_rel']:+.3f} | K32 rel {rows[-1]['K32_rel']:+.3f}")
    le = np.log([r["eps"] for r in rows[-4:]])
    for key in ("k5_rel", "K41_rel", "K32_rel"):
        ly = np.log([abs(r[key]) for r in rows[-4:]])
        s = float(np.polyfit(le, ly, 1)[0])
        log(f"T1/T2 {key} log-log slope (4 smallest eps): {s:.2f} (predict 2: residual is eps^2 relative)")
    return rows


# ---------------------------------------------------------------- emission
def relu_marginal_moments_exact(P, n=140, Z=9.0):
    """E[relu(P(g))^p], p=1..4, root-split GL. Exact."""
    xg, wg = V._gl_cache(n)
    segs = L5._segments(P, Z)
    m = np.zeros(5)
    for lo, hi in zip(segs[:-1], segs[1:]):
        g, wq = V._map_nodes(lo, hi, xg, wg)
        g, wq = g.ravel(), wq.ravel()
        x = np.maximum(np.polynomial.polynomial.polyval(g, P), 0.0)
        dens = wq * V.phi(g)
        for p in range(1, 5):
            m[p] += float(np.sum(dens * x ** p))
    return m


def k4_from_moments(m):
    m1, m2, m3, m4 = m[1], m[2], m[3], m[4]
    return m4 - 4 * m3 * m1 - 3 * m2 * m2 + 12 * m2 * m1 * m1 - 6 * m1 ** 4


def model_post_moments(mu, sd, kstd, columns):
    """Edgeworth post moments: m_p = c[p,0] + sum_cols weight * c[p,q]."""
    c = ctable_p4(mu, sd)
    m = np.zeros(5)
    for p in range(1, 5):
        m[p] = c[p][0]
        for q, wgt in columns(kstd):
            m[p] += wgt * c[p][q]
    return m


def cols_base(k):
    return [(3, k["k3"] / 6.0), (4, k["k4"] / 24.0), (6, k["k3"] ** 2 / 72.0)]


def cols_k5(k):
    return cols_base(k) + [(5, k["k5"] / 120.0)]


def cols_order3(k):
    return cols_k5(k) + [(7, k["k3"] * k["k4"] / 144.0), (9, k["k3"] ** 3 / 1296.0),
                         (6, k["k6"] / 720.0)]


def dk4_dcolumn(mu, sd, q):
    """d k4_post / d(weight of He_q column), connected differential."""
    c = ctable_p4(mu, sd)
    m1, m2, m3 = c[1][0], c[2][0], c[3][0]
    d1, d2, d3, d4 = c[1][q], c[2][q], c[3][q], c[4][q]
    return (d4 - 4 * m1 * d3 - 4 * m3 * d1 - 6 * m2 * d2
            + 12 * m1 * m1 * d2 + 24 * m2 * m1 * d1 - 24 * m1 ** 3 * d1)


def t3_emission(log):
    rows = []
    for eps in [0.32, 0.2263, 0.16, 0.1131, 0.08, 0.0566]:
        P = L5.pair_poly(0.3, 0.9, eps * 0.6, eps * eps * 0.4)
        kap = L5.cumulants_from_moments(L5.joint_moments(P, P, 0.0), deg=5)
        mu, var = kap[1, 0], kap[2, 0]
        sd = math.sqrt(var)
        # kappa6 of the marginal needs deg-6 moments: extend via 1D GH directly
        xg, wg = np.polynomial.hermite.hermgauss(30)
        z = math.sqrt(2.0) * xg
        wn = wg / math.sqrt(math.pi)
        h = np.polynomial.polynomial.polyval(z, P)
        m1d = np.array([float(np.sum(wn * h ** p)) for p in range(7)])
        a6 = np.zeros((7, 7))
        for p in range(7):
            a6[p, 0] = m1d[p]
        kap6 = L5.cumulants_from_moments(a6, deg=6)[6, 0]
        k = {"k3": kap[3, 0] / sd ** 3, "k4": kap[4, 0] / sd ** 4,
             "k5": kap[5, 0] / sd ** 5, "k6": kap6 / sd ** 6}
        truth = k4_from_moments(relu_marginal_moments_exact(P))
        base = k4_from_moments(model_post_moments(mu, sd, k, cols_base))
        withk5 = k4_from_moments(model_post_moments(mu, sd, k, cols_k5))
        full3 = k4_from_moments(model_post_moments(mu, sd, k, cols_order3))
        rows.append(dict(eps=eps, kstd=k, truth=truth, r_base=truth - base,
                         r_k5=truth - withk5, r_full3=truth - full3))
        log(f"T3 eps={eps:.4f}: post-k4 truth {truth:+.6e}  resid base {truth - base:+.2e}  "
            f"+He5(k5) {truth - withk5:+.2e}  +order3 set {truth - full3:+.2e}  "
            f"(k3={k['k3']:+.3f} k4={k['k4']:+.3f} k5={k['k5']:+.4f} k6={k['k6']:+.4f})")
    le = np.log([r["eps"] for r in rows[-4:]])
    for key, pred in (("r_base", 3), ("r_full3", 4)):
        ly = np.log([max(abs(r[key]), 1e-300) for r in rows[-4:]])
        s = float(np.polyfit(le, ly, 1)[0])
        log(f"T3 {key} slope {s:.2f} (predict {pred})")
    return rows


# ---------------------------------------------------------------- magnitude gate
def t4_magnitude(log):
    decay_needed = 1.0 - (0.82 / 0.93) ** (1.0 / 8.0)   # per-layer relative decay L7->L15
    log(f"T4 target: masked-median k4 ratio 0.93@L7 -> 0.82@L15 (all reset residuals threaded) "
        f"=> needed injection ~{100 * decay_needed:.2f}%/layer relative")
    out = []
    for g3 in (0.10, 0.175, 0.25):
        for g4 in (0.03, 0.05, 0.07):
            k5c = kappa5_closure(g3, g4)
            for alpha in (-0.5, 0.0, 0.5):
                mu, sd = alpha, 1.0
                sens5 = dk4_dcolumn(mu, sd, 5) / 120.0
                sens7 = dk4_dcolumn(mu, sd, 7) / 144.0
                sens9 = dk4_dcolumn(mu, sd, 9) / 1296.0
                k = {"k3": g3, "k4": g4}
                base = k4_from_moments(model_post_moments(mu, sd, k, cols_base))
                d_k5 = k5c * sens5
                d_l34 = g3 * g4 * sens7
                d_l333 = g3 ** 3 * sens9
                out.append(dict(g3=g3, g4=g4, alpha=alpha, k5_closure=k5c,
                                post_k4_base=base,
                                dk4_k5col=d_k5, dk4_he7=d_l34, dk4_he9=d_l333,
                                rel_k5=d_k5 / base, rel_he7=d_l34 / base,
                                rel_order3=(d_k5 + d_l34 + d_l333) / base))
    med5 = float(np.median([abs(r["rel_k5"]) for r in out]))
    med3 = float(np.median([abs(r["rel_order3"]) for r in out]))
    hi5 = float(np.max([abs(r["rel_k5"]) for r in out]))
    log(f"T4 kappa5-column share of post-k4 (closure-fed): median {100 * med5:.2f}%/layer, "
        f"max {100 * hi5:.2f}%/layer;  full order-3 emission set median {100 * med3:.2f}%/layer")
    for r in out:
        if r["alpha"] == 0.0:
            log(f"  g3={r['g3']:.3f} g4={r['g4']:.2f} a=0: k5_closure={r['k5_closure']:+.4f} "
                f"post_k4={r['post_k4_base']:+.4f} dk4(He5)={r['dk4_k5col']:+.5f} "
                f"({100 * r['rel_k5']:+.2f}%)  dk4(He7)={r['dk4_he7']:+.5f} "
                f"dk4(He9)={r['dk4_he9']:+.5f}  order3 total {100 * r['rel_order3']:+.2f}%")
    ratio = med5 / decay_needed
    verdict = ("RIGHT SIZE" if 0.3 <= ratio <= 3.0 else
               ("TOO SMALL (fail-fast)" if ratio < 0.1 else
                ("small but nonneglibible" if ratio < 0.3 else "LARGER than needed")))
    log(f"T4 VERDICT: kappa5-column median injection / needed decay = {ratio:.2f}  -> {verdict}")
    return out, decay_needed, ratio, verdict


def run():
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    log(f"TAG public_trivariate_derivation_v1/kappa5_verify.py | numpy {np.__version__}")
    stein_check(log)
    closure_rows = t1_t2_closure(log)
    emission_rows = t3_emission(log)
    mag_rows, decay_needed, ratio, verdict = t4_magnitude(log)
    with open(f"{HERE}/KAPPA5_CLOSURE.json", "w") as f:
        json.dump({
            "tag": "public_trivariate_derivation_v1/kappa5",
            "date": "2026-08-26",
            "closure": {
                "diagonal": "kappa5 = 5*k3*k4/v - (40/9)*k3^3/v^2 (standardized: v=1); "
                            "equivalently 5*k3*k4_intrinsic + (20/9)*k3^3 with "
                            "k4_intrinsic = k4 - (4/3)k3^2",
                "K41": "rho*(3 k3i k4i - 28/9 k3i^3) + rho^2 k3j k4i + rho^3*(k3i k4j - 4/3 k3i k3j^2)",
                "K32": "see KAPPA5_CLOSURE.md eq (3.3) / K32_closure() in kappa5_verify.py",
                "kappa6": "NOT free: leading order needs He4/chaos-4 content the closure does not carry",
                "frame": "chaos-graded generation (He2 ~ eps, He3 ~ eps^2), the same frame that "
                         "underlies the deployed k4 = sum_x K21^2/v law; exact at leading order in-frame",
            },
            "emission": {
                "dm_p/dk5": "c[p,5]/120", "dm_p/dk6": "c[p,6]/720",
                "dk4post_dcol": "c[4,q] - 4 m1 c[3,q] - 4 m3 c[1,q] - 6 m2 c[2,q] "
                                "+ 12 m1^2 c[2,q] + 24 m2 m1 c[1,q] - 24 m1^3 c[1,q], all over the "
                                "column's factorial weight",
                "stein_columns": "c[1,5]=sd He3(w)phi(w); c[2,5]=2 sd^2 He2(w)phi(w); "
                                 "c[3,5]=6 sd^3 He1(w)phi(w); c[4,5]=24 sd^4 phi(w); w=-mu/sd",
            },
            "verification": {"closure_rows": closure_rows, "emission_rows": emission_rows},
            "magnitude_gate": {"rows": mag_rows, "needed_per_layer": decay_needed,
                               "median_ratio": ratio, "verdict": verdict},
        }, f, indent=1, default=float)
    log("artifact written: KAPPA5_CLOSURE.json")


if __name__ == "__main__":
    run()
