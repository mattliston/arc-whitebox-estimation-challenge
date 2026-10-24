#!/usr/bin/env python3
"""F3A read-only probe: spectral structure of the quartic/cubic residual stars.

Captures full star planes from the frozen instrumented X estimator on the
frozen eight-row roster and measures per-layer singular-value energy decay.
Target-free; writes only STAR_SPECTRUM_PROBE.json.
"""
import hashlib, importlib.util, json, os, sys
from pathlib import Path
import numpy as np
import flopscope as flops
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
SOURCE = P2 / "public_lambda3_shell_probe_v1/instrumented_estimator_x.py"
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
ROSTER = [0, 1, 2, 55, 100, 201, 400, 643]
RANKS_KEPT = [1, 4, 16, 64, 256]

def sha(p):
    d = hashlib.sha256()
    with open(p, "rb") as h:
        for b in iter(lambda: h.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()

def main():
    flops.configure(symmetry_warnings=False)
    spec = importlib.util.spec_from_file_location("star_probe_x", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    captures = []
    orig = module._reset_stars
    def wrapped(*args, **kwargs):
        out = orig(*args, **kwargs)
        cubic, quartic, alpha = out
        cn = cubic.view(np.ndarray) if hasattr(cubic, "view") else np.asarray(cubic)
        qn = quartic.view(np.ndarray) if hasattr(quartic, "view") else np.asarray(quartic)
        captures.append((np.array(cn, dtype=np.float32, copy=True),
                         np.array(qn, dtype=np.float32, copy=True)))
        return out
    module._reset_stars = wrapped
    manifest = json.loads((STORE / "manifest.json").read_text())
    entries = manifest["roles"]["observer_fit"]["row_manifests"]
    # energy fraction captured by top-r singular values, aggregated per layer index
    quartic_frac = {r: [] for r in RANKS_KEPT}
    cubic_frac = {r: [] for r in RANKS_KEPT}
    per_layer_q64 = np.zeros((8, 15))
    for row_index, rank in enumerate(ROSTER):
        entry = entries[rank]
        rmp = STORE / entry["path"]
        row = json.loads(rmp.read_text())
        weights = np.load(rmp.parent / "weights.npy", mmap_mode="r", allow_pickle=False)
        mlp = MLP(width=1024, depth=16, weights=[weights[l] for l in range(16)],
                  seed=int(derive_estimator_seed(int(row["mlp_seed"]))), name=f"star-spec-{rank}")
        mlp.validate()
        start = len(captures)
        with threadpool_limits(limits=2):
            with flops.BudgetContext(flop_budget=2**42, quiet=True):
                module.predict_with_native_tape(mlp)
        rowcaps = captures[start:]
        assert len(rowcaps) == 15, len(rowcaps)
        for layer_index, (cn, qn) in enumerate(rowcaps):
            sq = np.linalg.svd(qn.astype(np.float64), compute_uv=False)
            sc = np.linalg.svd(cn.astype(np.float64), compute_uv=False)
            eq = np.square(sq); ec = np.square(sc)
            tq = eq.sum(); tc = ec.sum()
            for r in RANKS_KEPT:
                quartic_frac[r].append(float(eq[:r].sum() / max(tq, 1e-300)))
                cubic_frac[r].append(float(ec[:r].sum() / max(tc, 1e-300)))
            per_layer_q64[row_index, layer_index] = float(eq[:64].sum() / max(tq, 1e-300))
        print(json.dumps({"rank": rank, "q64_mean": float(per_layer_q64[row_index].mean())}), flush=True)
    payload = {
        "schema": "p2r0f3a-star-spectrum-probe-v1",
        "node": "P2R0F3A",
        "status": "COMPLETE_TARGET_FREE",
        "source_sha256": sha(SOURCE),
        "roster": ROSTER,
        "quartic_energy_fraction_by_rank": {str(r): {"mean": float(np.mean(v)), "min": float(np.min(v))} for r, v in quartic_frac.items()},
        "cubic_energy_fraction_by_rank": {str(r): {"mean": float(np.mean(v)), "min": float(np.min(v))} for r, v in cubic_frac.items()},
        "quartic_top64_by_layer_mean": per_layer_q64.mean(axis=0).tolist(),
        "targets_read": 0,
    }
    out = HERE / "STAR_SPECTRUM_PROBE.json"
    assert not out.exists()
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"quartic": payload["quartic_energy_fraction_by_rank"], "cubic": payload["cubic_energy_fraction_by_rank"]}, indent=1))

if __name__ == "__main__":
    main()
