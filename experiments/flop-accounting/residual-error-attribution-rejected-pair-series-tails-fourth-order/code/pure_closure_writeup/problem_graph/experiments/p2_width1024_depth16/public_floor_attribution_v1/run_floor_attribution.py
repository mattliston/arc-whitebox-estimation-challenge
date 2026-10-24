#!/usr/bin/env python3
"""Floor attribution: which analytic approximation owns the ~3.1e-8 floor?

The floor realization is the deployed banked card's residual (truth minus
host-plus-frozen-corrector, clamped) - the error no deployed component
removes. Correlates its square against the R-arm's 10 source-uncertainty
channels (literal diagnostics of the host's own approximations) per layer
and pooled, plus row-level. All inputs already exist: the sealed K22-pilot
base tapes, the R-arm channel rows (same roster), and the lawfully open
truth. The strongest family becomes the next carrier target.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
BASE_EST = P2 / "public_q24_native_transport_energy_final_workspace_v4/estimator.py"
BASE_ROWS = P2 / "public_k22_pilot_v1/rows/base"
R_ROWS = P2 / "public_native_r10_source_uncertainty_pilot_v1/rows"
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
ROSTER = list(range(35, 235))
CHANNEL_NAMES = [
    "cp4_dropped_diagonal", "cp4_dropped_diagonal_absolute",
    "cubic_star_row_rms", "quartic_star_row_rms", "quartic_star_diagonal",
    "pairing_alpha", "mehler_t24_row_coupling", "max_offdiagonal_abs_rho",
    "rho2_neighbor_alpha", "rho8_row_concentration",
]


def main():
    spec = importlib.util.spec_from_file_location("fa_base", BASE_EST)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    manifest = json.loads((STORE / "manifest.json").read_text())
    entries = manifest["roles"]["observer_fit"]["row_manifests"]
    table = pq.read_table(TARGETS, columns=["mlp_id", "all_layer_means"])
    values = table.column("all_layer_means").combine_chunks()
    storage = getattr(values, "storage", values)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    truth = storage.to_numpy(zero_copy_only=False).reshape(688, 16, 1024)[35:235].astype(np.float64)

    floor_error = np.empty((200, 16, 1024), dtype=np.float64)
    channels = np.empty((200, 16, 1024, 10), dtype=np.float32)
    for local, rank in enumerate(ROSTER):
        d = BASE_ROWS / f"row_{rank:04d}"
        host = np.load(d / "host.npy", allow_pickle=False)
        raw20 = np.load(d / "raw20.npy", allow_pickle=False)
        sigma = np.load(d / "sigma.npy", allow_pickle=False)
        rmp = STORE / entries[rank]["path"]
        weights = np.load(rmp.parent / "weights.npy", mmap_mode="r", allow_pickle=False)
        correction = module.transport_correction(
            raw20, sigma, [weights[l] for l in range(16)], module._TRANSPORT_PAYLOAD)
        deployed = np.maximum(
            np.asarray(host, dtype=np.float64) + np.asarray(correction, dtype=np.float64), 0.0)
        floor_error[local] = truth[local] - deployed
        channels[local] = np.load(R_ROWS / f"row_{rank:04d}/r10_locals.npy",
                                  allow_pickle=False)
        if local % 50 == 0:
            print(json.dumps({"loaded": local}), flush=True)

    floor_sq = floor_error ** 2
    report = {"per_layer_pooled": {}, "final_layer": {}, "row_level": {}}
    for c, name in enumerate(CHANNEL_NAMES):
        mag = np.abs(channels[:, :, :, c].astype(np.float64))
        layer_corrs = []
        for layer in range(2, 16):
            x = mag[:, layer, :].ravel()
            y = floor_sq[:, layer, :].ravel()
            x = x - x.mean(); y = y - y.mean()
            denom = np.sqrt((x * x).sum() * (y * y).sum())
            layer_corrs.append(float((x * y).sum() / denom) if denom > 0 else 0.0)
        xf = mag[:, 15, :].ravel(); yf = floor_sq[:, 15, :].ravel()
        xf = xf - xf.mean(); yf = yf - yf.mean()
        df = np.sqrt((xf * xf).sum() * (yf * yf).sum())
        row_x = mag.mean(axis=(1, 2)); row_y = floor_sq[:, 15, :].mean(axis=1)
        row_x = row_x - row_x.mean(); row_y = row_y - row_y.mean()
        dr = np.sqrt((row_x * row_x).sum() * (row_y * row_y).sum())
        report["per_layer_pooled"][name] = {
            "mean_layer_corr": float(np.mean(layer_corrs)),
            "consistent_sign_layers": int(sum(1 for v in layer_corrs if v * np.sign(np.mean(layer_corrs)) > 0)),
        }
        report["final_layer"][name] = float((xf * yf).sum() / df) if df > 0 else 0.0
        report["row_level"][name] = float((row_x * row_y).sum() / dr) if dr > 0 else 0.0

    payload = {
        "schema": "floor-attribution-v1",
        "node": "P2R0F3A",
        "floor_definition": "truth minus (base host + frozen corrector, clamped) - the deployed banked card's residual on the 200 protected rows (mean final MSE 2.958e-8; in-sample flattered but the only deployable floor realization)",
        "report": report,
        "floor_final_mse": float(floor_sq[:, 15, :].mean()),
        "truth_use": "lawful all-layer means, rows 35..234 (S-localization precedent)",
    }
    out = HERE / "FLOOR_ATTRIBUTION.json"
    assert not out.exists()
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=1))
    print("receipt_sha256:", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
