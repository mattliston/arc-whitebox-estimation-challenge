#!/usr/bin/env python3
"""TAG g22-rows-closedloop (T-109 scheduled sampling): generate the learned-operator training rows (host / raw20 / sigma, the frozen 160/40
protocol's row format) from the DERIVED host v1 = estimator_retro.py with the in-card fourth-order plane chain
(RETRO_K22_CHAIN=1: pre_k22 := the engine's plane chain run in-card, public_g29_bond_kernel_v1/k22_chain.py; certified
against the engine's deploy-frame planes at corr 0.9999, host 3.092e-8 on r35 F2J off). No forcing, no truth file, no
feed, no residual carry (rows v1). Same MLP/seed law and row format as gen_rows_k4panel_role.py.
Usage: gen_rows_mixed025.py <rank>   (G22_ROLE=observer_fit|observer_selection)"""
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
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
ROLE = os.environ.get("G22_ROLE", "observer_fit")
OUT = HERE / ("rows_mixed025" if ROLE == "observer_fit" else f"rows_mixed025_{ROLE}")

# T-105: the ORACLE recursion - the organizers' 1e9 all-layer means fed back into the recursion (mean + consistent
# covariance); the row's host = the ONE-STEP mean before the correction; the refit target = truth - one-step host.
os.environ["RETRO_CLOSED_LOOP"] = "1"
os.environ["RETRO_CLOSED_LOOP_ROWS"] = "pre"
os.environ["RETRO_TRANSPORT_PAYLOAD"] = os.environ["G22_CL_PAYLOAD"]   # iteration-1 operator fed back
os.environ["RETRO_CLOSED_LOOP_EXPOSURE"] = os.environ.get("G22_CL_EXPOSURE", "0.5")   # truth share of the fed-back correction
os.environ["RETRO_MEANLEAK_OFF"] = "1"   # the truth file feeds only the blend, not the oracle hook
for k in ("RETRO_FORCE", "RETRO_K4_RESCALE", "RETRO_FORCE_TRUTH", "RETRO_FORCE_NOISE", "RETRO_FORCE_MIX",
          "RETRO_RESID_CP", "RETRO_CP4_FEED", "RETRO_DUMP_K4", "RETRO_DUMP_PLANES", "RETRO_DUMP_CP",
          "RETRO_DUMP_CP4", "CP4_CAP", "RETRO_FORCE_PLANES", "RETRO_K21_INJECT", "RETRO_K21_SUPPLY",
          "RETRO_V_INJECT", "RETRO_K3_INJECT", "RETRO_K3_RESCALE", "RETRO_PLANE_SCALE", "RETRO_K22_CHAIN_DUMP"):
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
              seed=int(derive_estimator_seed(int(row["mlp_seed"]))), name=f"mixed-{ROLE}-{rank}")
    mlp.validate()
    os.environ["RETRO_MEANLEAK_TRUTH"] = str(HERE / "truthpost" / f"truthpost_rank{rank:04d}.npz")
    spec = importlib.util.spec_from_file_location(f"retro_{os.getpid()}", SOURCE)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    assert not m._RETRO_FORCE and not m._RETRO_FORCE_PLANES and not m._RETRO_K22_CHAIN, "contract: no forcing, no chain"
    with threadpool_limits(limits=5):
        with flops.BudgetContext(flop_budget=2**42, wall_time_limit_s=900.0, quiet=True) as ctx:
            host_v, raw_v, sigma_v = m.predict_with_native_tape(mlp)
    host = np.ascontiguousarray(np.asarray(host_v), dtype="<f4")
    raw20 = np.ascontiguousarray(np.asarray(raw_v), dtype="<f4")
    sigma = np.ascontiguousarray(np.asarray(sigma_v), dtype="<f4")
    assert host.shape == (16, 1024) and np.isfinite(host).all()
    assert np.isfinite(raw20).all() and np.isfinite(sigma).all() and (sigma > 0).all()
    tmp = out_dir.with_name(f".{out_dir.name}.{os.getpid()}")
    tmp.mkdir(parents=True, exist_ok=True)
    np.save(tmp / "host.npy", host); np.save(tmp / "raw20.npy", raw20); np.save(tmp / "sigma.npy", sigma)
    (tmp / "result.json").write_text(json.dumps({
        "rank": rank, "mlp_id": int(entry["mlp_id"]), "flops_used": int(ctx.flops_used),
        "status": "complete", "targets_read": 0,
        "host": "estimator_retro.py RETRO_CLOSED_LOOP=1 RETRO_CLOSED_LOOP_ROWS=pre (scheduled sampling: alpha x truth + (1-alpha) x operator fed back, one-step host rows, T-109)",
        "k4_table_sha256_prefix": "n/a (literal in host)"}) + "\n")
    tmp.replace(out_dir)
    return {"rank": rank, "flops": int(ctx.flops_used)}


if __name__ == "__main__":
    print(json.dumps(one(int(sys.argv[1]))), flush=True)
