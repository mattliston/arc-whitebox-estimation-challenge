#!/usr/bin/env python3
"""P2R0F3A design probe: is _push_quartic's star transport a birth mechanism?

TAG=star-transport-fidelity  width=16  seed=7  synthetic-only (targets_read=0)

Compares the deployed Q24 host's quartic star transport (_push_quartic, the
exact source function from public_q24_native_transport_energy_final_repeatability_v3)
against the exact symmetric 4-tensor push-forward T' = T x4 W on three synthetic
cumulant families, in the three consumed plane channels (K4 diagonal, K31, K22).

Supersedes STAR_TRANSPORT_FIDELITY_PROBE.json (v1), whose exact-reference k4
channel was poisoned by a numpy aliasing bug: einsum("aaab->ab", Tp) returns a
VIEW into Tp, and np.fill_diagonal on that view zeroed Tp[a,a,a,a] before the
k4 view was materialized, collapsing the reference denominator (the
vanishing-denominator law, self-inflicted). Fix: .copy() every extraction.
v1's k31/k22 numbers were unaffected and are reproduced here.
"""
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EST = (
    HERE.parent
    / "public_q24_native_transport_energy_final_repeatability_v3"
    / "estimator.py"
)

N = 16
SEED = 7


def load_module():
    spec = importlib.util.spec_from_file_location("q24_v3_est", EST)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def set_pattern(T, idx, value):
    for p in set(itertools.permutations(idx)):
        T[p] = value


def exact_planes(T, W):
    """Exact push-forward planes; .copy() prevents view aliasing (v1 bug)."""
    Tp = np.einsum("ijkl,ia,jb,kc,ld->abcd", T, W, W, W, W, optimize=True)
    k4 = np.einsum("aaaa->a", Tp).copy()
    k31 = np.einsum("aaab->ab", Tp).copy()
    np.fill_diagonal(k31, 0.0)
    k22 = np.einsum("aabb->ab", Tp).copy()
    np.fill_diagonal(k22, 0.0)
    return k4, k31, k22


def replica_planes(sd, so, W):
    """Line-faithful float64 transcription of _push_quartic's star part
    (estimator.py lines 1046-1064; pairing terms zero here)."""
    W2, W3 = W**2, W**3
    transported = so @ W
    mixed = W2.T @ (W * transported)
    diag = (sd[:, None] * W2 * W2 + 4.0 * W3 * transported).sum(axis=0)
    k31 = W3.T @ (sd[:, None] * W) + W3.T @ transported \
        + 3.0 * (W2 * transported).T @ W
    k22 = W2.T @ (sd[:, None] * W2) + 2.0 * (mixed + mixed.T)
    np.fill_diagonal(k31, 0.0)
    np.fill_diagonal(k22, 0.0)
    return diag, k31, k22


def module_planes(mod, sd, so, W):
    """The deployed function itself, f32 fnp semantics, pairing zero."""
    fnp = mod.fnp
    w32 = fnp.asarray(W, dtype=fnp.float32)
    wc = (w32, fnp.transpose(w32), w32 * w32, fnp.transpose(w32 * w32),
          w32 * w32 * w32, fnp.transpose(w32 * w32 * w32))
    star_np = so.astype(np.float32)
    np.fill_diagonal(star_np, sd.astype(np.float32))
    star_h = fnp.asarray(star_np, dtype=fnp.float32)
    eye = np.eye(N, dtype=bool)
    zero = fnp.zeros((N, N), dtype=fnp.float64)
    zerov = fnp.zeros(N, dtype=fnp.float64)
    d, k31, k22 = mod._push_quartic(
        star_h, 0.0, wc, zero, zerov, eye, None,
    )
    return (np.asarray(d, dtype=np.float64),
            np.asarray(k31, dtype=np.float64),
            np.asarray(k22, dtype=np.float64))


def rel(a, b):
    db = np.linalg.norm(b)
    if db < 1e-30:
        return float(np.linalg.norm(a - b))  # absolute when reference vanishes
    return float(np.linalg.norm(a - b) / db)


def channel_report(planes, exact):
    keys = ("k4", "k31", "k22")
    out = {k: rel(p, e) for k, p, e in zip(keys, planes, exact)}
    num = np.sqrt(sum(np.linalg.norm(p - e) ** 2 for p, e in zip(planes, exact)))
    den = np.sqrt(sum(np.linalg.norm(e) ** 2 for e in exact))
    out["pooled_missed_fraction"] = float(num / den) if den > 0 else 0.0
    out["exact_norms"] = {k: float(np.linalg.norm(e)) for k, e in zip(keys, exact)}
    return out


def main():
    mod = load_module()
    rng = np.random.default_rng(SEED)
    W = rng.standard_normal((N, N)) / np.sqrt(N)

    TA = np.zeros((N,) * 4)
    for i in range(N):
        set_pattern(TA, (i, i, i, i), rng.standard_normal())
        for j in range(N):
            if i != j:
                set_pattern(TA, (i, i, i, j), rng.standard_normal())
    TB = np.zeros((N,) * 4)
    for i in range(N):
        for j in range(i + 1, N):
            set_pattern(TB, (i, i, j, j), rng.standard_normal())
    G = rng.standard_normal((N,) * 4)
    TC = sum(np.transpose(G, p) for p in itertools.permutations(range(4))) / 24.0

    results = {}
    for name, T in (("A_star_family_positive_control", TA),
                    ("B_k22_residual_only", TB),
                    ("C_generic_symmetric", TC)):
        sd = np.einsum("iiii->i", T).copy()
        so = np.einsum("iiij->ij", T).copy()
        np.fill_diagonal(so, 0.0)
        exact = exact_planes(T, W)
        results[name] = {
            "float64_replica_vs_exact": channel_report(replica_planes(sd, so, W), exact),
            "deployed_f32_module_vs_exact": channel_report(module_planes(mod, sd, so, W), exact),
        }

    payload = {
        "schema": "p2r0f3a-star-transport-fidelity-probe-v2",
        "node": "P2R0F3A",
        "status": "COMPLETE_SYNTHETIC_NO_DATA",
        "supersedes": "STAR_TRANSPORT_FIDELITY_PROBE.json",
        "supersession_reason": (
            "v1's exact-reference k4 channel read zeros: einsum('aaab->ab') "
            "returned a writable view into the transported tensor and "
            "np.fill_diagonal on it erased Tp[a,a,a,a] before the k4 view was "
            "materialized. v2 copies every extraction; v1 k31/k22 unaffected."
        ),
        "width": N,
        "seed": SEED,
        "estimator_source": str(EST.relative_to(HERE.parent)),
        "estimator_sha256": hashlib.sha256(EST.read_bytes()).hexdigest(),
        "results": results,
        "targets_read": 0,
    }
    out = HERE / "STAR_TRANSPORT_FIDELITY_PROBE_v2.json"
    assert not out.exists(), "append-only: v2 receipt already present"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.rename(out)
    print(json.dumps(results, indent=1))
    print("receipt_sha256:", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
