#!/usr/bin/env python3
"""TAG g22-refit-gate: cross-ROLE held-out gate for the k4-panel refit payload.
Scores the refit corrector (REFIT_K4PANEL_MODEL.pt) on rows of the
observer_SELECTION role (88 nets with sealed targets, DISJOINT from the
observer_fit role the operator was fitted on): deployed = clamp(host +
corrector(tape), 0) vs the sealed all_layer_means, final layer, plain MSE
(the protocol's valid-40 convention). Reports host-only, refit-corrected,
and the removal fraction per net and pooled. Usage: score_refit_gate.py"""
import importlib.util, json, os, sys
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
J_SOURCE = P2 / "public_full688_transport_energy_coherence_cv_v1/run_experiment.py"
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
import os
ROWS = Path(os.environ.get("G22_SEL_ROWS", str(HERE / "rows_k4panel_observer_selection")))
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_selection.parquet"
if os.environ.get("G22_LOCAL_FEATURES"): J_SOURCE = HERE / "jext/run_experiment_ext.py"
spec = importlib.util.spec_from_file_location("gate_j", J_SOURCE); j = importlib.util.module_from_spec(spec); sys.modules["gate_j"] = j; spec.loader.exec_module(j)
art = torch.load(Path(os.environ.get("G22_MODEL", str(HERE / "REFIT_K4PANEL_MODEL.pt"))), map_location="cpu", weights_only=False)
sd = art["state_dict"]
model = j.TransportEnergyMultiStateResidual(sd["centers"].numpy(), sd["scales"].numpy(), sd["layer_rms"].numpy())
model.load_state_dict(sd); model.eval()
man = json.load(open(STORE / "manifest.json"))["roles"]["observer_selection"]["row_manifests"]
t = pq.read_table(TARGETS, columns=["mlp_id", "all_layer_means"]); ids = t.column("mlp_id").combine_chunks().to_numpy(zero_copy_only=False)
v = t.column("all_layer_means").combine_chunks(); st = getattr(v, "storage", v)
while not pa.types.is_primitive(st.type): st = st.flatten()
tg = st.to_numpy(zero_copy_only=False).reshape(len(ids), 16, 1024)
ranks = sorted(int(p.name[4:]) for p in ROWS.glob("row_*") if (p / "result.json").exists())
rows = []
with torch.no_grad():
    for r in ranks:
        d = ROWS / f"row_{r:04d}"; rec = json.loads((d / "result.json").read_text())
        host = np.load(d / "host.npy").astype(np.float64)
        r20 = np.load(d / "raw20.npy").astype(np.float32)
        if (d / "mech.npy").exists(): r20 = np.concatenate([r20, np.load(d / "mech.npy").astype(np.float32)], axis=2)
        raw = torch.from_numpy(r20[None]); sig = torch.from_numpy(np.load(d / "sigma.npy")[None].astype(np.float32))
        w = torch.from_numpy(np.asarray(np.load(Path(STORE / man[r]["path"]).parent / "weights.npy", mmap_mode="r"))[None].astype(np.float32))
        corr = model(raw, sig, w, exposure=1.0).double().numpy()[0]
        k = int(np.where(ids == rec["mlp_id"])[0][0]); truth = tg[k].astype(np.float64)
        dep = np.maximum(host + corr, 0.0)
        rows.append({"rank": r, "mlp_id": rec["mlp_id"], "host_final_mse": float(np.mean((host[15] - truth[15]) ** 2)),
                     "refit_final_mse": float(np.mean((dep[15] - truth[15]) ** 2))})
h = float(np.mean([x["host_final_mse"] for x in rows])); f = float(np.mean([x["refit_final_mse"] for x in rows]))
out = {"role": "observer_selection (disjoint from the fit role)", "nets": len(rows), "host_final_mse_mean": h,
       "refit_final_mse_mean": f, "removal_fraction": 1 - f / h, "per_net": rows}
(Path(os.environ.get("G22_GATE_OUT", str(HERE / "REFIT_K4PANEL_GATE_SELECTION.json")))).write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: out[k] for k in out if k != "per_net"}), flush=True)
