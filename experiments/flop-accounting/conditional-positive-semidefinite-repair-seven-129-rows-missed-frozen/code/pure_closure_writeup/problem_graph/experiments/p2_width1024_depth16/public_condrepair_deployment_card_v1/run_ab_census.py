#!/usr/bin/env python3
"""P2R0F2W A/B census: banked H7B estimator vs conditional-repair W card."""
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import flopscope as flops
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
BASE = P2 / "public_q24_native_transport_energy_final_workspace_v4/estimator.py"
W_SOURCE = HERE / "estimator.py"
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
BASE_SHA = "2fd2d40d4e30cc32269afb0113e244ee76c0a51005bce166c7625510f66a8a76"
W_SHA = "24ed2325e4db0c0287a222189a986c2d6462f80832c717827c27b783a6655ee2"
TAILS = [55, 71, 106, 317, 339, 458, 485, 507, 540, 575, 602]
HARD = [201, 310, 599, 643]

def sha(p):
    d = hashlib.sha256()
    with open(p, "rb") as h:
        for b in iter(lambda: h.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()

def roster():
    rows = sorted(set(TAILS) | set(HARD) | set(range(0, 688, 6)))
    return rows

def load(tag, path):
    spec = importlib.util.spec_from_file_location(tag, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[tag] = m
    spec.loader.exec_module(m)
    return m

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=2)
    args = parser.parse_args()
    assert sha(BASE) == BASE_SHA and sha(W_SOURCE) == W_SHA
    flops.configure(symmetry_warnings=False)
    base = load("w_census_base", BASE)
    wmod = load("w_census_w", W_SOURCE)
    skip_count = {"skips": 0, "fallbacks": 0}
    import flopscope.numpy as fnp
    orig_chol = fnp.linalg.cholesky
    # Every W gauge-arm repair attempts exactly one Cholesky; counting the
    # attempts and failures gives the full skip/fallback path census.
    def counting_chol(x):
        try:
            r = orig_chol(x)
        except Exception:
            skip_count["fallbacks"] += 1
            raise
        skip_count["skips"] += 1
        return r
    fnp.linalg.cholesky = counting_chol
    manifest = json.loads((STORE / "manifest.json").read_text())
    entries = manifest["roles"]["observer_fit"]["row_manifests"]
    rows = [r for i, r in enumerate(roster()) if i % args.shards == args.shard]
    out_path = HERE / f"census_shard_{args.shard}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            done.add(json.loads(line)["rank"])
    with out_path.open("a") as sink:
        for rank in rows:
            if rank in done:
                continue
            entry = entries[rank]
            rmp = STORE / entry["path"]
            row = json.loads(rmp.read_text())
            weights = np.load(rmp.parent / "weights.npy", mmap_mode="r", allow_pickle=False)
            def run(module, name):
                mlp = MLP(width=1024, depth=16, weights=[weights[l] for l in range(16)],
                          seed=int(derive_estimator_seed(int(row["mlp_seed"]))), name=name)
                mlp.validate()
                with threadpool_limits(limits=2):
                    with flops.BudgetContext(flop_budget=2**41, wall_time_limit_s=300.0, quiet=True) as ctx:
                        out = module.Estimator().predict(mlp, 2**41)
                return np.asarray(out, dtype=np.float64), int(ctx.flops_used)
            t0 = time.perf_counter()
            pb, fb = run(base, f"census-base-{rank}")
            pw, fw = run(wmod, f"census-w-{rank}")
            delta = pw - pb
            record = {
                "rank": rank,
                "finite": bool(np.isfinite(pw).all()),
                "base_flops": fb,
                "w_flops": fw,
                "flops_saved": fb - fw,
                "delta_final_rms": float(np.sqrt(np.mean(np.square(delta[-1])))),
                "delta_max_abs": float(np.abs(delta).max()),
                "skips_cum": skip_count["skips"],
                "fallbacks_cum": skip_count["fallbacks"],
                "is_tail": rank in TAILS,
                "is_hard": rank in HARD,
                "wall_seconds": time.perf_counter() - t0,
            }
            sink.write(json.dumps(record, sort_keys=True) + "\n")
            sink.flush()
            print(json.dumps({k: record[k] for k in ("rank", "finite", "flops_saved", "delta_final_rms")}), flush=True)
    print(json.dumps({"shard": args.shard, "status": "SHARD_DONE", "rows": len(rows)}))

if __name__ == "__main__":
    main()
