#!/usr/bin/env python3
"""TAG g19a-attribution-v1: exact telescoping attribution of the Q24 host's
per-layer mean error vs offline MC truth (weights-only, seeded, no targets).
See PREREGISTER.md (sha d65e0daf...). Usage: run_attribution.py <rank>"""
import importlib.util
import json
import sys
from math import erf
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
BASE = P2 / "public_q24_native_transport_energy_final_workspace_v4/estimator.py"
STORE = (P2 / "learned_residual_observer_design/v4_feature_store_v3/"
         "sealed_target_blind_feature_store_v3")
P_SAMPLES = 8_000_000
BATCH = 65_536
SEED_XOR = 0x47414D454F4E5F31


def load_host(tag):
    spec = importlib.util.spec_from_file_location(tag, BASE)
    m = importlib.util.module_from_spec(spec)
    sys.modules[tag] = m
    spec.loader.exec_module(m)
    return m


def tap_host_state(mod, out):
    orig = mod._univariate_edgeworth

    def tap(coefficients, raw, sigma, alpha, pdf, k3, k4, sigma2, sigma3,
            inverse_context=None):
        s = np.asarray(sigma, dtype=np.float64).view(np.ndarray)
        a = np.asarray(alpha, dtype=np.float64).view(np.ndarray)
        out.append({
            "m": (a * s).copy(), "v": (s * s).copy(),
            "k3": np.asarray(k3, dtype=np.float64).view(np.ndarray).copy(),
            "k4": np.asarray(k4, dtype=np.float64).view(np.ndarray).copy(),
        })
        return orig(coefficients, raw, sigma, alpha, pdf, k3, k4, sigma2,
                    sigma3, inverse_context)

    mod._univariate_edgeworth = tap


_erf = np.vectorize(erf)


def edg2_relu_mean(m, v, k3, k4):
    v = np.maximum(v, 1e-30)
    s = np.sqrt(v)
    a = m / s
    pdf = np.exp(-0.5 * a * a) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1.0 + _erf(a / np.sqrt(2)))
    base = s * pdf + m * cdf
    ma = -a
    he1 = ma
    he2 = ma * ma - 1.0
    he4 = ma ** 4 - 6.0 * ma * ma + 3.0
    c1_3 = s * he1 * pdf
    c1_4 = s * he2 * pdf
    c1_6 = s * he4 * pdf
    g3 = k3 / s ** 3
    g4 = k4 / s ** 4
    return base + (g3 / 6.0) * c1_3 + (g4 / 24.0) * c1_4 + (g3 * g3 / 72.0) * c1_6


def mc_truth(weights, mlp_seed):
    rng = np.random.Generator(np.random.PCG64DXSM((mlp_seed ^ SEED_XOR)
                                                   & ((1 << 64) - 1)))
    n_layers = 16
    post_s1 = np.zeros((n_layers, 1024))
    post_s2 = np.zeros((n_layers, 1024))
    pre_s = np.zeros((n_layers, 4, 1024))
    done = 0
    w32 = [np.ascontiguousarray(weights[l], dtype=np.float32)
           for l in range(n_layers)]
    while done < P_SAMPLES:
        b = min(BATCH, P_SAMPLES - done)
        x = rng.standard_normal(size=(b, 1024), dtype=np.float32)
        for l in range(n_layers):
            h = x @ w32[l]
            h64 = h.astype(np.float64)
            pre_s[l, 0] += h64.sum(axis=0)
            pre_s[l, 1] += (h64 ** 2).sum(axis=0)
            pre_s[l, 2] += (h64 ** 3).sum(axis=0)
            pre_s[l, 3] += (h64 ** 4).sum(axis=0)
            x = np.maximum(h, 0.0)
            x64 = x.astype(np.float64)
            post_s1[l] += x64.sum(axis=0)
            post_s2[l] += (x64 ** 2).sum(axis=0)
        done += b
        if done % (BATCH * 16) == 0:
            print(json.dumps({"samples": done}), flush=True)
    P = float(P_SAMPLES)
    m1 = pre_s[:, 0] / P
    m2 = pre_s[:, 1] / P
    m3 = pre_s[:, 2] / P
    m4 = pre_s[:, 3] / P
    pre_mean = m1
    pre_var = m2 - m1 ** 2
    pre_k3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    pre_k4 = (m4 - 4 * m1 * m3 - 3 * m2 ** 2 + 12 * m1 ** 2 * m2
              - 6 * m1 ** 4)
    post_mean = post_s1 / P
    post_var = post_s2 / P - post_mean ** 2
    post_se = np.sqrt(np.maximum(post_var, 0) / P)
    return pre_mean, pre_var, pre_k3, pre_k4, post_mean, post_se


def main():
    rank = int(sys.argv[1])
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    rmp = STORE / entry["path"]
    row = json.loads(rmp.read_text())
    weights = np.load(rmp.parent / "weights.npy", mmap_mode="r",
                      allow_pickle=False)

    import flopscope as flops
    from threadpoolctl import threadpool_limits
    from whestbench.domain import MLP
    from whestbench.seeds import derive_estimator_seed
    mod = load_host(f"g19a_{rank}")
    host_states = []
    tap_host_state(mod, host_states)
    mlp = MLP(width=1024, depth=16,
              weights=[weights[l] for l in range(16)],
              seed=int(derive_estimator_seed(int(row["mlp_seed"]))),
              name=f"g19a-{rank}")
    mlp.validate()
    with threadpool_limits(limits=2):
        with flops.BudgetContext(flop_budget=2 ** 43,
                                 wall_time_limit_s=600.0, quiet=True):
            host_v, _, _ = mod.predict_with_native_tape(mlp)
    host_post = np.asarray(host_v, dtype=np.float64)
    assert host_post.shape == (16, 1024)

    pre_m, pre_v, pre_k3, pre_k4, mc_post, mc_se = mc_truth(
        weights, int(row["mlp_seed"]))

    np.savez(HERE / f"states_rank{rank:04d}.npz",
             mc_pre_m=pre_m, mc_pre_v=pre_v, mc_pre_k3=pre_k3,
             mc_pre_k4=pre_k4, mc_post=mc_post, mc_se=mc_se,
             host_post=host_post,
             host_m=np.stack([t["m"] for t in host_states]),
             host_v=np.stack([t["v"] for t in host_states]),
             host_k3=np.stack([t["k3"] for t in host_states]),
             host_k4=np.stack([t["k4"] for t in host_states]))
    report = {"schema": "g19a-attribution-v1", "rank": rank,
              "P": P_SAMPLES, "layers": {}}
    n_taps = len(host_states)
    for l in range(1, 16):
        tap = host_states[l - 1] if l - 1 < n_taps else None
        e = mc_post[l] - host_post[l]
        edg_true = edg2_relu_mean(pre_m[l], pre_v[l], pre_k3[l], pre_k4[l])
        term_a = mc_post[l] - edg_true
        if tap is not None:
            edg_host = edg2_relu_mean(tap["m"], tap["v"], tap["k3"], tap["k4"])
            term_b = edg_true - edg_host
            term_c = edg_host - host_post[l]
        else:
            term_b = np.zeros(1024)
            term_c = e - term_a
        noise_energy = float(np.sum(mc_se[l] ** 2))
        raw_energy = float(np.sum(e ** 2))
        rec = {
            "raw_error_energy": raw_energy,
            "mc_noise_energy": noise_energy,
            "nc_error_energy": max(raw_energy - noise_energy, 0.0),
            "term_A_energy": float(np.sum(term_a ** 2)) - noise_energy,
            "term_B_energy": float(np.sum(term_b ** 2)),
            "term_C_energy": float(np.sum(term_c ** 2)),
            "cross_AB": float(2 * np.sum(term_a * term_b)),
            "cross_AC": float(2 * np.sum(term_a * term_c)),
            "cross_BC": float(2 * np.sum(term_b * term_c)),
            "tap_present": tap is not None,
        }
        report["layers"][str(l)] = rec
    (HERE / f"ATTRIBUTION_rank{rank:04d}.json").write_text(
        json.dumps(report, indent=1) + "\n")
    late = [report["layers"][str(l)] for l in range(10, 16)]
    tot = sum(r["nc_error_energy"] for r in late)
    print(json.dumps({
        "rank": rank,
        "late_nc_energy": tot,
        "late_A": sum(r["term_A_energy"] for r in late) / tot if tot else None,
        "late_B": sum(r["term_B_energy"] for r in late) / tot if tot else None,
        "late_C": sum(r["term_C_energy"] for r in late) / tot if tot else None,
        "noise_over_raw": sum(r["mc_noise_energy"] for r in late)
                          / sum(r["raw_error_energy"] for r in late),
    }, indent=1))


if __name__ == "__main__":
    main()
