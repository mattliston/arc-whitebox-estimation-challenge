#!/usr/bin/env python3
"""TAG g22-rows-meanleak-mech (T-112): the ORACLE recursion (organizers' 1e9 means fed back, one-step host rows) + G28
mechanistic features per row (mech.npy). Derived from gen_rows_mech.py (round 3).
Original docstring: generate the learned-operator training rows
(host / raw20 / sigma, the frozen 160/40 protocol's row format) from the
k4-RESCALED host: teammate A's estimator_retro.py (gated copy of the
deployed D11 lineage) with RETRO_FORCE=k4 + RETRO_K4_RESCALE=<frozen
panel-mean table> so that at every layer_index li the card's own carried
k4 is replaced by card_k4 / s_li (k4_scale_panel8.json, key
s_l_ratio_rms_by_layer_index). In rescale mode the truth vector handed to
_retro_forced_value is IGNORED (estimator_retro.py:1087-1089); the
RETRO_FORCE_TRUTH file is only required to exist for shape/indexing, so a
fixed dummy (r35 P8M) is passed and NO truth enters the prediction (asserted
below by re-deriving the code path). Same MLP/seed law as
public_joint_floor_probe_v1/generate_joint_rows.py. Usage: gen_rows_k4panel.py <rank>"""
import importlib.util, json, os, sys
from pathlib import Path
import numpy as np
import flopscope as flops
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
SOURCE = P2 / "public_g20_emission_retrofit_v1/estimator_retro.py"
TABLE = HERE / "k4_scale_panel8.json"
DUMMY_TRUTH = P2 / "public_g19_attribution_v1/mps_truth_rank0035_P8M.npz"
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
ROLE = os.environ.get("G22_ROLE", "observer_fit")
OUT = Path(os.environ["G22_OUT"]) if os.environ.get("G22_OUT") else (HERE / ("rows_meanleak_mech15" if ROLE == "observer_fit" else f"rows_meanleak_mech15_{ROLE}"))   # T-116g: honour G22_OUT (honest rows go to rows_meanleak_mech15b)

os.environ["RETRO_MEANLEAK_MEAN"] = "1"
os.environ["RETRO_MEANLEAK_ROWS"] = "pre"
os.environ["RETRO_DUMP_FEAT"] = "1"
MECHTMP = HERE / "rows_meanleak_mech15_tmp"; MECHTMP.mkdir(exist_ok=True)
os.environ["RETRO_DUMP_FEAT_PATH"] = str(MECHTMP / "mechfeat")
sys.path.insert(0, str(P2 / "public_g28_mech_features_v1"))
for k in ("RETRO_FORCE", "RETRO_K4_RESCALE", "RETRO_FORCE_TRUTH", "RETRO_K3_RESCALE", "RETRO_K22_CHAIN", "RETRO_CLOSED_LOOP",
          "RETRO_FORCE_NOISE", "RETRO_FORCE_MIX", "RETRO_RESID_CP", "RETRO_CP4_FEED",
          "RETRO_DUMP_K4", "RETRO_DUMP_PLANES", "RETRO_DUMP_CP", "RETRO_DUMP_CP4", "CP4_CAP"):
    os.environ.pop(k, None)


def one(rank):
    out_dir = OUT / f"row_{rank:04d}"
    if (out_dir / "result.json").exists():
        return {"rank": rank, "status": "done"}
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"][ROLE]["row_manifests"][rank]
    rmp = STORE / entry["path"]
    row = json.loads(rmp.read_text())
    weights = np.load(rmp.parent / "weights.npy", mmap_mode="r", allow_pickle=False)
    mlp = MLP(width=1024, depth=16, weights=[weights[l] for l in range(16)],
              seed=int(derive_estimator_seed(int(row["mlp_seed"]))), name=f"mlmech15-{ROLE}-{rank}")
    mlp.validate()
    os.environ["RETRO_MEANLEAK_TRUTH"] = str(HERE / "truthpost" / f"truthpost_rank{rank:04d}.npz")
    spec = importlib.util.spec_from_file_location(f"retro_{os.getpid()}", SOURCE)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    assert not m._RETRO_FORCE and m._RETRO_K4SCALE is None and m._RETRO_K3SCALE is None, "contract: no forcing, no tables"
    with threadpool_limits(limits=5):
        with flops.BudgetContext(flop_budget=2**42, wall_time_limit_s=900.0, quiet=True) as ctx:
            host_v, raw_v, sigma_v = m.predict_with_native_tape(mlp)
    host = np.ascontiguousarray(np.asarray(host_v), dtype="<f4")
    from mechfeat15 import mechfeat_from_run
    fp = MECHTMP / f"mechfeat_{mlp.name}.npz"
    feat = np.load(fp)
    X = np.ascontiguousarray(mechfeat_from_run(feat, mlp.weights), dtype="<f4")
    assert X.shape == (16, 1024, 31) and np.isfinite(X).all()
    fp.unlink()
    raw20 = np.ascontiguousarray(np.asarray(raw_v), dtype="<f4")
    sigma = np.ascontiguousarray(np.asarray(sigma_v), dtype="<f4")
    assert host.shape == (16, 1024) and np.isfinite(host).all()
    assert np.isfinite(raw20).all() and np.isfinite(sigma).all() and (sigma > 0).all()
    tmp = out_dir.with_name(f".{out_dir.name}.{os.getpid()}")
    tmp.mkdir(parents=True, exist_ok=True)
    np.save(tmp / "host.npy", host); np.save(tmp / "raw20.npy", raw20); np.save(tmp / "sigma.npy", sigma)
    np.save(tmp / "mech.npy", X)
    # T-116m: four pre-covariance contractions per unit per layer (zeros where the dump lacks them)
    covfeat = np.zeros((16, 1024, 4), dtype="<f4")
    for l in range(16):
        if f"covfeat_{l}" in feat.files: covfeat[l] = np.asarray(feat[f"covfeat_{l}"], dtype="<f4")
    np.save(tmp / "covfeat.npy", covfeat)
    pf = np.zeros((16, 1024, 20), dtype="<f4")                       # T-137c: correlation-graph features from the instrument (puffifeat_{l})
    for l in range(16):
        if f"puffifeat_{l}" in feat.files: pf[l] = np.asarray(feat[f"puffifeat_{l}"], dtype="<f4")
    np.save(tmp / "puffifeat2.npy", pf)
    (tmp / "result.json").write_text(json.dumps({
        "rank": rank, "mlp_id": int(entry["mlp_id"]), "flops_used": int(ctx.flops_used),
        "status": "complete", "targets_read": 0,
        "host": "estimator_retro.py oracle recursion (RETRO_MEANLEAK_MEAN=1, rows=pre) + G28 mech.npy (16,1024,31), T-112",
        "k4_table_sha256_prefix": "n/a (literal in host)"}) + "\n")
    tmp.replace(out_dir)
    return {"rank": rank, "flops": int(ctx.flops_used)}


if __name__ == "__main__":
    print(json.dumps(one(int(sys.argv[1]))), flush=True)
