#!/usr/bin/env python3
"""Extended-tape screen: do supply-era observables reconstruct the supplied
host's residual where the 20-channel tape (cap 3.18e-8) cannot?
Capture 4 candidate channels on 30 rows; correlate vs per-layer residual^2."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import flopscope as flops
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
BASE = P2 / "public_q24_native_transport_energy_final_workspace_v4/estimator.py"
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
ROWS = list(range(35, 235, 7))[:30]
CH = ["K_row_norm", "k4_supply", "k22_supply_rownorm", "K_diag_neighbor"]


def load(tag, p):
    spec = importlib.util.spec_from_file_location(tag, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[tag] = m
    spec.loader.exec_module(m)
    return m


def make(mod, chans_out):
    state = {"K22resid": None, "W": None}
    orig_stars = mod._reset_stars
    orig_assemble = mod._assemble_quartic_provider
    orig_push = mod._push_quartic

    def stars_wrap(*args, **kwargs):
        result = orig_stars(*args, **kwargs)
        k4, k31, k22 = args[2], args[3], args[4]
        qpp = args[9] if len(args) > 9 else kwargs.get("quartic_provider_planes")
        k4r, k31r, k22r = mod._quartic_residual(k4, k31, k22, qpp)
        arr = np.asarray(k22r, dtype=np.float64).view(np.ndarray)
        sym = 0.5 * (arr + arr.T)
        var = np.asarray(args[6], dtype=np.float64).view(np.ndarray)
        cov = np.asarray(args[5], dtype=np.float64).view(np.ndarray)
        pairing = np.outer(var, var) + 2.0 * cov * cov
        np.fill_diagonal(pairing, 0.0)
        np.fill_diagonal(sym, 0.0)
        state["K22resid"] = sym - float(result[2]) * pairing
        return result

    def push_wrap(star_h, pairing_alpha, weight_context, pre_covariance, pre_variance,
                  diagonal_context, fusion_context):
        state["W"] = np.asarray(weight_context[0], dtype=np.float64).view(np.ndarray)
        return orig_push(star_h, pairing_alpha, weight_context, pre_covariance,
                         pre_variance, diagonal_context, fusion_context)

    def assemble_wrap(host_planes, provider_planes):
        k4, k31, k22 = orig_assemble(host_planes, provider_planes)
        if state["K22resid"] is not None and state["W"] is not None:
            W = state["W"]; W2 = W * W
            K = state["K22resid"]
            KW2 = K @ W2
            S = 3.0 * np.einsum("ia,ia->a", W2, KW2)
            k4 = k4 + mod.fnp.asarray(S, dtype=mod.fnp.float64)
            W2TK = W2.T @ K
            k31a = 3.0 * ((W2TK * W.T) @ W)
            k22a = W2.T @ KW2
            np.fill_diagonal(k31a, 0.0)
            np.fill_diagonal(k22a, 0.0)
            k31 = k31 + mod.fnp.asarray(k31a, dtype=mod.fnp.float64)
            k22 = k22 + mod.fnp.asarray(k22a, dtype=mod.fnp.float64)
            Kabs = np.abs(K)
            chans_out.append(np.stack([
                Kabs.mean(axis=1),
                np.abs(S),
                np.abs(k22a).mean(axis=1),
                np.abs(np.diag(K @ K)) ** 0.5,
            ], axis=-1).astype(np.float32))
        return k4, k31, k22

    mod._reset_stars = stars_wrap
    mod._push_quartic = push_wrap
    mod._assemble_quartic_provider = assemble_wrap
    return state


def main():
    manifest = json.loads((STORE / "manifest.json").read_text())
    entries = manifest["roles"]["observer_fit"]["row_manifests"]
    table = pq.read_table(TARGETS, columns=["mlp_id", "all_layer_means"])
    values = table.column("all_layer_means").combine_chunks()
    storage = getattr(values, "storage", values)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    targets = storage.to_numpy(zero_copy_only=False).reshape(688, 16, 1024).astype(np.float64)
    residuals, channels = [], []
    for rank in ROWS:
        mod = load(f"nt_{rank}", BASE)
        chans_out = []
        make(mod, chans_out)
        entry = entries[rank]
        rmp = STORE / entry["path"]
        row = json.loads(rmp.read_text())
        weights = np.load(rmp.parent / "weights.npy", mmap_mode="r", allow_pickle=False)
        mlp = MLP(width=1024, depth=16, weights=[weights[l] for l in range(16)],
                  seed=int(derive_estimator_seed(int(row["mlp_seed"]))), name=f"nt-{rank}")
        mlp.validate()
        with threadpool_limits(limits=2):
            with flops.BudgetContext(flop_budget=2**43, wall_time_limit_s=600.0, quiet=True):
                host_v, _, _ = mod.predict_with_native_tape(mlp)
        out = np.asarray(host_v, dtype=np.float64)
        residuals.append((targets[rank] - out).astype(np.float32))
        channels.append(np.stack(chans_out))
        print(json.dumps({"rank": rank, "done": True}), flush=True)
    resid = np.stack(residuals)
    chan = np.stack(channels)
    report = {}
    for c, name in enumerate(CH):
        pooled_unit = []
        for L in range(2, 16):
            x = chan[:, L - 2, :, c].astype(np.float64).ravel()
            y = (resid[:, L, :].astype(np.float64) ** 2).ravel()
            xc, yc = x - x.mean(), y - y.mean()
            d = np.sqrt((xc * xc).sum() * (yc * yc).sum())
            pooled_unit.append(float((xc * yc).sum() / d) if d > 0 else 0.0)
        rx = chan[:, :, :, c].mean(axis=(1, 2)).astype(np.float64)
        ry = (resid[:, 15, :].astype(np.float64) ** 2).mean(axis=1)
        rxc, ryc = rx - rx.mean(), ry - ry.mean()
        d = np.sqrt((rxc * rxc).sum() * (ryc * ryc).sum())
        report[name] = {
            "mean_unit_corr_L2_15": round(float(np.mean(pooled_unit)), 3),
            "max_unit_corr": round(float(np.max(np.abs(pooled_unit))), 3),
            "row_corr": round(float((rxc * ryc).sum() / d) if d > 0 else 0.0, 3),
        }
    (HERE / "NEWTAPE_SCREEN.json").write_text(
        json.dumps({"schema": "newtape-screen-v1", "rows": ROWS,
                    "report": report}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
