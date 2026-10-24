#!/usr/bin/env python
"""
TAG: public_trivariate_derivation_v1/l5_verify.py
Verifies L5_DERIVATION.md:
  W-test : derived L3L4 / L3^3/6 weight tables == lambda3 shell's literal arithmetic (machine eq).
  Physics: bivariate pair with EXACT joint cumulants (cubic Hermite transform of correlated
           Gaussians, eps-graded: k3~eps, k4~eps^2, k5~eps^3, k6~eps^4); exact E[x_i x_j] by
           root-split 2D Gauss-Legendre; base (1+L3+L4+L3^2/2) vs +L5class (deployable) vs
           +Lambda5 (completion). Residuals must drop by the predicted order (eps^3 -> eps^4).
Convention replicated from estimator.py exactly: channels (a,b), weights kappa/(C(r,a)/r!)-blocks,
Mehler sum_t rho^t/t! c_i[1,a+t] c_j[1,b+t], t <= 24 (t30 delta reported).
Outputs: L5_FORMULA.json, L5_VERIFICATION.md.  numpy + math only.
"""
import json
import math
import sys
from fractions import Fraction
from math import factorial

import numpy as np

sys.path.insert(0, "/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_trivariate_derivation_v1")
import verify as V

HERE = "/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_trivariate_derivation_v1"
TMAX = 24  # the estimator's frozen _Q24_TERMS - 1

W3 = {(3, 0): Fraction(1, 6), (2, 1): Fraction(1, 2), (1, 2): Fraction(1, 2), (0, 3): Fraction(1, 6)}
W4 = {(4, 0): Fraction(1, 24), (3, 1): Fraction(1, 6), (2, 2): Fraction(1, 4),
      (1, 3): Fraction(1, 6), (0, 4): Fraction(1, 24)}
W5 = {(5, 0): Fraction(1, 120), (4, 1): Fraction(1, 24), (3, 2): Fraction(1, 12),
      (2, 3): Fraction(1, 12), (1, 4): Fraction(1, 24), (0, 5): Fraction(1, 120)}


def l3sq_terms():
    """channel -> [(Fraction coeff, (pat3, pat3)), ...] for L3^2/2."""
    out = {}
    for p1, c1 in W3.items():
        for p2, c2 in W3.items():
            key = (p1[0] + p2[0], p1[1] + p2[1])
            out.setdefault(key, []).append((Fraction(1, 2) * c1 * c2, (p1, p2)))
    return out


def l3l4_terms():
    out = {}
    for p1, c1 in W3.items():
        for p2, c2 in W4.items():
            key = (p1[0] + p2[0], p1[1] + p2[1])
            out.setdefault(key, []).append((c1 * c2, (p1, p2)))
    return out


def l3cubed_terms():
    out = {}
    for p1, c1 in W3.items():
        for p2, c2 in W3.items():
            for p3, c3 in W3.items():
                key = (p1[0] + p2[0] + p3[0], p1[1] + p2[1] + p3[1])
                out.setdefault(key, []).append((Fraction(1, 6) * c1 * c2 * c3, (p1, p2, p3)))
    return out


def eval_terms(terms, k):
    """Numeric channel weights from a standardized-cumulant dict k[(a,b)]."""
    return {ch: float(sum(float(c) * math.prod(k[p] for p in pats) for c, pats in lst))
            for ch, lst in terms.items()}


def shell_weights_transcribed(k):
    """Literal transcription of estimator_x.py _edgeworth_weights lines 592-615."""
    w03, w12, w21, w30 = k[(0, 3)] / 6, k[(1, 2)] / 2, k[(2, 1)] / 2, k[(3, 0)] / 6
    w04, w13, w22s, w31, w40 = (k[(0, 4)] / 24, k[(1, 3)] / 6, k[(2, 2)] / 4,
                                k[(3, 1)] / 6, k[(4, 0)] / 24)
    h06 = 0.5 * w03 * w03
    h15 = w03 * w12
    h24 = w03 * w21 + 0.5 * w12 * w12
    h33 = w03 * w30 + w12 * w21
    h42 = w12 * w30 + 0.5 * w21 * w21
    h51 = w21 * w30
    h60 = 0.5 * w30 * w30
    third = 1.0 / 3.0
    out = {(0, 6): h06, (1, 5): h15, (2, 4): h24, (3, 3): h33, (4, 2): h42, (5, 1): h51, (6, 0): h60,
           (0, 7): w03 * w04, (1, 6): w03 * w13 + w12 * w04,
           (2, 5): w03 * w22s + w12 * w13 + w21 * w04,
           (3, 4): w03 * w31 + w12 * w22s + w21 * w13 + w30 * w04,
           (4, 3): w03 * w40 + w12 * w31 + w21 * w22s + w30 * w13,
           (5, 2): w12 * w40 + w21 * w31 + w30 * w22s,
           (6, 1): w21 * w40 + w30 * w31, (7, 0): w30 * w40,
           (0, 9): third * (h06 * w03), (1, 8): third * (h06 * w12 + h15 * w03),
           (2, 7): third * (h06 * w21 + h15 * w12 + h24 * w03),
           (3, 6): third * (h06 * w30 + h15 * w21 + h24 * w12 + h33 * w03),
           (4, 5): third * (h15 * w30 + h24 * w21 + h33 * w12 + h42 * w03),
           (5, 4): third * (h24 * w30 + h33 * w21 + h42 * w12 + h51 * w03),
           (6, 3): third * (h33 * w30 + h42 * w21 + h51 * w12 + h60 * w03),
           (7, 2): third * (h42 * w30 + h51 * w21 + h60 * w12),
           (8, 1): third * (h51 * w30 + h60 * w21), (9, 0): third * (h60 * w30)}
    return out


def w_test(log):
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(200):
        k = {(a, b): float(rng.uniform(-1, 1))
             for a in range(6) for b in range(6) if 3 <= a + b <= 5}
        mine = {}
        mine.update(eval_terms(l3sq_terms(), k))
        mine.update(eval_terms(l3l4_terms(), k))
        mine.update(eval_terms(l3cubed_terms(), k))
        shell = shell_weights_transcribed(k)
        for ch, wv in shell.items():
            d = abs(mine[ch] - wv) / max(abs(wv), 1e-12)
            worst = max(worst, d)
    log(f"W-test: derived weight tables vs shell transcription, 200 random cumulant draws, "
        f"27 channels: max rel dev {worst:.2e}")
    assert worst < 1e-13, worst
    return worst


# ---------------------------------------------------------------- exact-cumulant pair
def pair_poly(m, s, e2, e3):
    """h(g) = m + s*(g + e2 He2(g) + e3 He3(g)) as coeffs low->high."""
    return np.array([m - s * e2, s * (1.0 - 3.0 * e3), s * e2, s * e3])


def joint_moments(Pi, Pj, rho_g, deg=5, nodes=24):
    xg, wg = np.polynomial.hermite.hermgauss(nodes)
    z = math.sqrt(2.0) * xg
    wn = wg / math.sqrt(math.pi)
    s = math.sqrt(1.0 - rho_g * rho_g)
    g1 = z[:, None] + 0.0 * z[None, :]
    g2 = rho_g * z[:, None] + s * z[None, :]
    W = wn[:, None] * wn[None, :]
    hi = np.polynomial.polynomial.polyval(g1, Pi)
    hj = np.polynomial.polynomial.polyval(g2, Pj)
    m = np.zeros((deg + 1, deg + 1))
    hip = [np.ones_like(hi)]
    hjp = [np.ones_like(hj)]
    for _ in range(deg):
        hip.append(hip[-1] * hi)
        hjp.append(hjp[-1] * hj)
    for p in range(deg + 1):
        for q in range(deg + 1 - p):
            m[p, q] = float(np.sum(W * hip[p] * hjp[q]))
    return m


def cumulants_from_moments(m, deg=5):
    """2D truncated series log of the MGF: kappa[p,q] = p! q! [log M]_{pq}."""
    a = np.zeros_like(m)
    for p in range(deg + 1):
        for q in range(deg + 1 - p):
            a[p, q] = m[p, q] / (factorial(p) * factorial(q))
    B = a.copy()
    B[0, 0] = 0.0

    def mul(x, y):
        out = np.zeros_like(x)
        for p in range(deg + 1):
            for q in range(deg + 1 - p):
                acc = 0.0
                for r in range(p + 1):
                    for t in range(q + 1):
                        acc += x[r, t] * y[p - r, q - t]
                out[p, q] = acc
        return out

    L = np.zeros_like(a)
    Bk = None
    for k in range(1, deg + 1):
        Bk = B if Bk is None else mul(Bk, B)
        L += ((-1.0) ** (k + 1)) / k * Bk
    kap = np.zeros_like(L)
    for p in range(deg + 1):
        for q in range(deg + 1 - p):
            kap[p, q] = L[p, q] * factorial(p) * factorial(q)
    return kap


def standardized_state(Pi, Pj, rho_g):
    kap = cumulants_from_moments(joint_moments(Pi, Pj, rho_g))
    mi, mj = kap[1, 0], kap[0, 1]
    si, sj = math.sqrt(kap[2, 0]), math.sqrt(kap[0, 2])
    rho = kap[1, 1] / (si * sj)
    k = {}
    for a in range(6):
        for b in range(6 - a):
            if a + b >= 3:
                k[(a, b)] = kap[a, b] / (si ** a * sj ** b)
    return mi, si, mj, sj, rho, k


# ---------------------------------------------------------------- model in the code convention
def model_m11(mi, si, mj, sj, rho, k, level, tmax=TMAX):
    qmax = 9 + tmax
    ci = V.ctable_stein(mi, si, qmax)[1]
    cj = V.ctable_stein(mj, sj, qmax)[1]
    channels = {(0, 0): 1.0}
    for (a, b), c in W3.items():
        channels[(a, b)] = channels.get((a, b), 0.0) + float(c) * k[(a, b)]
    for (a, b), c in W4.items():
        channels[(a, b)] = channels.get((a, b), 0.0) + float(c) * k[(a, b)]
    for ch, w in eval_terms(l3sq_terms(), k).items():
        channels[ch] = channels.get(ch, 0.0) + w
    if level in ("l5", "l5k5"):
        for ch, w in eval_terms(l3l4_terms(), k).items():
            channels[ch] = channels.get(ch, 0.0) + w
        for ch, w in eval_terms(l3cubed_terms(), k).items():
            channels[ch] = channels.get(ch, 0.0) + w
    if level == "l5k5":
        for (a, b), c in W5.items():
            channels[(a, b)] = channels.get((a, b), 0.0) + float(c) * k[(a, b)]
    rp = rho ** np.arange(tmax + 1)
    invf = np.array([1.0 / factorial(t) for t in range(tmax + 1)])
    total = 0.0
    for (a, b), w in channels.items():
        if w == 0.0:
            continue
        kern = float(np.sum(rp * invf * ci[a:a + tmax + 1] * cj[b:b + tmax + 1]))
        total += w * kern
    return total


# ---------------------------------------------------------------- exact truth
def _segments(P, Z=9.0):
    c = np.trim_zeros(np.asarray(P, dtype=float), "b")
    pts = [-Z, Z]
    if len(c) > 1:
        for r in np.roots(c[::-1]):
            if abs(r.imag) < 1e-12 and -Z < r.real < Z:
                pts.append(float(r.real))
    return sorted(pts)


def truth_pair_moment(Pi, Pj, rho_g, n=110, Z=9.0):
    xg, wg = V._gl_cache(n)
    s = math.sqrt(1.0 - rho_g * rho_g)

    def nodes_for(P):
        segs = _segments(P, Z)
        gs, ws = [], []
        for lo, hi in zip(segs[:-1], segs[1:]):
            g, w = V._map_nodes(lo, hi, xg, wg)
            gs.append(g.ravel())
            ws.append(w.ravel())
        return np.concatenate(gs), np.concatenate(ws)

    g1, w1 = nodes_for(Pi)
    g2, w2 = nodes_for(Pj)
    xi = np.maximum(np.polynomial.polynomial.polyval(g1, Pi), 0.0)
    xj = np.maximum(np.polynomial.polynomial.polyval(g2, Pj), 0.0)
    dens = V.phi((g2[None, :] - rho_g * g1[:, None]) / s) / s
    inner = dens @ (w2 * xj)
    return float(np.sum(w1 * V.phi(g1) * xi * inner))


# ---------------------------------------------------------------- ladder
CONFIGS = [
    ("L5-C1", dict(mi=0.3, si=0.9, mj=-0.2, sj=1.1, g2i=0.6, g3i=0.4, g2j=-0.45, g3j=0.5,
                   rho_g=0.45)),
    ("L5-C2", dict(mi=0.0, si=1.0, mj=0.5, sj=0.8, g2i=-0.5, g3i=0.5, g2j=0.55, g3j=-0.35,
                   rho_g=0.25)),
]
EPS = [0.32, 0.2263, 0.16, 0.1131, 0.08, 0.0566, 0.04, 0.0283, 0.02, 0.0141, 0.01]


def tune_gamma3_null_k5(cfg, eps_probe=0.05):
    """Per-unit gamma3 that nulls the marginal fifth cumulant at leading order.
    kappa5_marginal is linear in gamma3 at fixed gamma2: solve from two probes."""
    out = dict(cfg)
    for unit, (mk, sk, g2k, g3k) in (("i", ("mi", "si", "g2i", "g3i")),
                                     ("j", ("mj", "sj", "g2j", "g3j"))):
        def k5_of(g3):
            P = pair_poly(cfg[mk], cfg[sk], eps_probe * cfg[g2k], eps_probe ** 2 * g3)
            kap = cumulants_from_moments(joint_moments(P, P, 0.0))
            return kap[5, 0]
        a0, a1 = k5_of(0.0), k5_of(1.0)
        out[g3k] = -a0 / (a1 - a0)
    return out


def run():
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    log(f"TAG public_trivariate_derivation_v1/l5_verify.py | TMAX={TMAX} | numpy {np.__version__}")
    wdev = w_test(log)

    c3 = tune_gamma3_null_k5(CONFIGS[0][1])
    log(f"[L5-C3] = C1 with per-unit gamma3 retuned to null marginal kappa5 at leading order: "
        f"g3i {CONFIGS[0][1]['g3i']:+.3f} -> {c3['g3i']:+.4f}, "
        f"g3j {CONFIGS[0][1]['g3j']:+.3f} -> {c3['g3j']:+.4f}")
    all_configs = CONFIGS + [("L5-C3", c3)]

    rows = []
    for name, cfg in all_configs:
        # Gaussian-limit anchor: eps=0 must reproduce truth to the Mehler floor.
        Pi = pair_poly(cfg["mi"], cfg["si"], 0.0, 0.0)
        Pj = pair_poly(cfg["mj"], cfg["sj"], 0.0, 0.0)
        st = standardized_state(Pi, Pj, cfg["rho_g"])
        t0 = truth_pair_moment(Pi, Pj, cfg["rho_g"])
        b0 = model_m11(*st[:5], st[5], "base")
        log(f"[{name}] eps=0 anchor: truth={t0:.12f} base={b0:.12f} dev={b0 - t0:+.2e} "
            f"(Mehler t<=24 floor at rho_g={cfg['rho_g']})")
        for eps in EPS:
            Pi = pair_poly(cfg["mi"], cfg["si"], eps * cfg["g2i"], eps * eps * cfg["g3i"])
            Pj = pair_poly(cfg["mj"], cfg["sj"], eps * cfg["g2j"], eps * eps * cfg["g3j"])
            mi, si, mj, sj, rho, k = standardized_state(Pi, Pj, cfg["rho_g"])
            T = truth_pair_moment(Pi, Pj, cfg["rho_g"])
            T2 = truth_pair_moment(Pi, Pj, cfg["rho_g"], n=160)
            B = model_m11(mi, si, mj, sj, rho, k, "base")
            F = model_m11(mi, si, mj, sj, rho, k, "l5")
            F5 = model_m11(mi, si, mj, sj, rho, k, "l5k5")
            F5b = model_m11(mi, si, mj, sj, rho, k, "l5k5", tmax=30)
            rows.append(dict(config=name, eps=eps, truth=T, quad_refine=T2 - T,
                             kstats=dict(k30=k[(3, 0)], k03=k[(0, 3)], k21=k[(2, 1)],
                                         k12=k[(1, 2)], k40=k[(4, 0)], k22=k[(2, 2)],
                                         k50=k[(5, 0)], k05=k[(0, 5)], rho=rho),
                             base=B, l5=F, l5k5=F5, mehler_t30_delta=F5b - F5,
                             r_base=T - B, r_l5=T - F, r_l5k5=T - F5,
                             shift_l5class=F - B, shift_lambda5=F5 - F))
            log(f"[{name}] eps={eps:.4f} truth={T:.9f} (refine {T2 - T:+.1e})  "
                f"resid base={T - B:+.3e}  +L5class={T - F:+.3e}  +Lambda5={T - F5:+.3e}  "
                f"shifts: L5class {F - B:+.2e} Lambda5 {F5 - F:+.2e}  "
                f"(t30d {F5b - F5:+.0e}; k30={k[(3, 0)]:+.3f} k40={k[(4, 0)]:+.3f} "
                f"k50={k[(5, 0)]:+.4f} k05={k[(0, 5)]:+.4f})")

    # slopes on the clean tail (smallest four eps)
    verdicts = {}
    for name, _ in all_configs:
        sub = [r for r in rows if r["config"] == name][-4:]
        le = np.log([r["eps"] for r in sub])

        def slope(key):
            ly = np.log([max(abs(r[key]), 1e-300) for r in sub])
            return float(np.polyfit(le, ly, 1)[0])

        verdicts[name] = dict(slope_base=slope("r_base"), slope_l5=slope("r_l5"),
                              slope_l5k5=slope("r_l5k5"))
        v = verdicts[name]
        log(f"[{name}] log-log residual slopes (smallest 4 eps): base {v['slope_base']:.2f} "
            f"(predict 3), +L5class {v['slope_l5']:.2f}, +Lambda5 {v['slope_l5k5']:.2f} (predict 4)")

    # third-order class: predicted vs measured
    for name, _ in all_configs:
        for r in [x for x in rows if x["config"] == name]:
            p3 = r["l5k5"] - r["base"]
            r["p3_ratio"] = r["r_base"] / p3 if p3 != 0 else float("nan")
        tail = [x for x in rows if x["config"] == name][-3:]
        log(f"[{name}] (truth-base)/(predicted order-3 class): "
            + ", ".join(f"eps={x['eps']:.3f}: {x['p3_ratio']:.3f}" for x in tail)
            + "  (-> 1 as eps -> 0)")

    write_artifacts(rows, verdicts, wdev)
    log("artifacts written: L5_FORMULA.json, L5_VERIFICATION.md")
    return rows


def _fraction_terms_json(terms, names):
    out = []
    for (a, b), lst in sorted(terms.items()):
        agg = {}
        for c, pats in lst:
            key = tuple(sorted(names[p] for p in pats))
            agg[key] = agg.get(key, Fraction(0)) + c
        out.append({"channel_a_on_i": a, "channel_b_on_j": b,
                    "terms": [{"coefficient": str(c), "factors": list(kk)}
                              for kk, c in sorted(agg.items())]})
    return out


def write_artifacts(rows, verdicts, wdev):
    n3 = {p: f"k{p[0]}{p[1]}" for p in W3}
    n4 = {p: f"k{p[0]}{p[1]}" for p in W4}
    names = {**n3, **n4}
    formula = {
        "tag": "public_trivariate_derivation_v1/L5",
        "date": "2026-08-25",
        "convention": {
            "pair_moment": "M11[i,j] = sum_channels W_ab * sum_{t=0}^{24} rho^t/t! "
                           "c_i[1,a+t]/sd_i^a * c_j[1,b+t]/sd_j^b  (c[p,q]=E[X_+^p He_q(Z)])",
            "kab": "standardized joint pre-activation cumulant, a indices on i, b on j "
                   "(raw kappa_ab/(sd_i^a sd_j^b); identical algebra to the code's raw-plane "
                   "weights times inverse_sigma powers)",
            "base_channels": "1 + L3 (kab*C(3,a)/6) + L4 (kab*C(4,a)/24) + L3^2/2 "
                             "(0.5 * selfconv of normalized L3 block)",
        },
        "l3l4_channels": _fraction_terms_json(l3l4_terms(), names),
        "l3cubed_channels": _fraction_terms_json(l3cubed_terms(), names),
        "lambda5_completion_requires_kappa5": [
            {"channel_a_on_i": a, "channel_b_on_j": b, "coefficient": str(c),
             "factors": [f"k{a}{b}"]} for (a, b), c in sorted(W5.items())],
        "exp_coefficients": {"L3*L4": "1", "L3^3": "1/6", "Lambda5": "1",
                             "statement": "third tilt-order class = Lambda5 + Lambda3*Lambda4 + "
                                          "Lambda3^3/6, complete"},
        "shell_comparison": {
            "verdict": "IDENTICAL: the lambda3 shell's weights[17..34] equal these tables "
                       "term-for-term (symbolic match in L5_DERIVATION.md Section 4; numeric "
                       "W-test below). Shell failure on the current host is NOT term error.",
            "w_test_max_rel_dev": wdev,
        },
        "verification_rows": rows,
        "slope_verdicts": verdicts,
    }
    with open(f"{HERE}/L5_FORMULA.json", "w") as f:
        json.dump(formula, f, indent=1)
    L = ["# L5 VERIFICATION — public_trivariate_derivation_v1",
         "Pair: h_u = m + s(g + eps g2 He2 + eps^2 g3 He3), exact cumulants (2D GH + series log),",
         "truth by root-split GL (column refine = its own error). Model = code convention, t<=24.",
         "",
         "| config | eps | truth | resid base | resid +L5class | resid +Lambda5 | shift L5class | shift Lambda5 | (T-B)/pred3 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| {} | {:.4f} | {:.9f} | {:+.3e} | {:+.3e} | {:+.3e} | {:+.2e} | {:+.2e} | {:.3f} |".format(
            r["config"], r["eps"], r["truth"], r["r_base"], r["r_l5"], r["r_l5k5"],
            r["shift_l5class"], r["shift_lambda5"], r.get("p3_ratio", float("nan"))))
    L.append("")
    for name, v in verdicts.items():
        L.append(f"- {name}: log-log residual slopes (4 smallest eps): base {v['slope_base']:.2f} "
                 f"(predicted 3), +L5class {v['slope_l5']:.2f}, +Lambda5 {v['slope_l5k5']:.2f} "
                 f"(predicted 4).")
    L.append("- W-test: shell weight arithmetic == derived tables, max rel dev "
             f"{wdev:.2e} over 200 random cumulant draws (27 channels).")
    with open(f"{HERE}/L5_VERIFICATION.md", "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    run()
