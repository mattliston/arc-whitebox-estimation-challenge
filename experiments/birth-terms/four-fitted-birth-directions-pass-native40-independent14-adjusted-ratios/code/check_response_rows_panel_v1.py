"""INERT fixed14 validation: --role response, fresh --role bank, then --score.

No refit, diagnostics, variant or fixture option.  Each role is one fresh
nice10 native process serving14 MLPs sequentially, with initialization billed
on its first call.  BOTH roles' unmodified served outputs and complete bills
are authenticated before any independent4M truth access.  These native
measurements are not a persistent client/server production-runtime gate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE / "response_rows_panel_v1"
DEV = HERE / "response_rows_adjusted_reevaluation_v1"
G19 = HERE.parent / "public_g19_attribution_v1"
HELPER = HERE / "check_response_rows_adjusted_v1.py"
HELPER_SHA = "8e99890bdc7742164a53303a23691e9f1cc54909c9cccc9cf64fc362eb459304"
BANK_SHA = "3b7f55f9a814789743467bb4bcda0837c3c1aad15d120e4cdab632594bf54757"
RESPONSE_SHA = "5a07832fee84b94c3a98589dc3ef19d718c1b062f5c4d3e04452a56ab9f11172"
PROTOCOL_SHA = "ba7b74f2c00bab570e4b722393a39a56eebc905d19d284f2d69dec2fe4c95cbb"
DEV_SCORE_SHA = "a5e4a8aa64968218931d3b11aa80141580371ec17931f7a1532b7381ab5437d0"
PUBLIC = (35, 100, 150, 200, 250, 300, 400, 500)
VIRGIN = (3016500, 3016501, 3016502, 3016503, 3016504, 3016505)
RANKS = PUBLIC + VIRGIN
SCHEMA = "response_rows_independent14_panel_v1"
BLAS_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def load_helper():
    # Authenticate the inert helper BEFORE importing any of its definitions.
    if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:
        raise RuntimeError("Frozen helper hash mismatch")
    spec = importlib.util.spec_from_file_location("fixed_response_panel_helpers", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.require(module.digest(HELPER) == HELPER_SHA, "Helper changed during import")
    return module


def sources(h):
    h.require(h.PINNED[h.BANK] == BANK_SHA and h.PINNED[h.RESPONSE] == RESPONSE_SHA
              and h.PINNED[h.PROTOCOL] == PROTOCOL_SHA, "Candidate/protocol pin mismatch")
    bound = h.source_bindings(RESPONSE_SHA)
    bound[Path(__file__).name] = h.binding(__file__)
    # Reuse the qualified development parity/repeat; do not recapture features.
    for role in ("response", "bank"):
        h.verify_role(DEV, role, RESPONSE_SHA)
        for name in ("completion.json", "result.json", "predictions_and_bills_sealed.json"):
            bound[f"development_{role}_{name}"] = h.binding(DEV / role / name)
    bound["development_score"] = h.binding(DEV / "score/result.json")
    h.require(bound["development_score"]["sha256"] == DEV_SCORE_SHA,
              "Only the frozen passing development score is admissible")
    dev = h.load_json(DEV / "score/result.json")
    h.require(dev["qualification_passed"] and dev["hashes_unchanged"]
              and dev["advances_to_independent_14_panel"], "Development prerequisite failed")
    return bound


def fixtures(h):
    manifest_path = h.STORE / "manifest.json"
    bound = {"manifest": h.binding(manifest_path)}
    manifest = h.load_json(manifest_path)["roles"]["observer_fit"]["row_manifests"]
    rows = []
    for rank in RANKS:
        if rank in PUBLIC:
            entry = manifest[rank] if isinstance(manifest, list) else manifest[str(rank)]
            path = (h.STORE / entry["path"]).resolve()
            h.require(path.is_relative_to(h.STORE.resolve()), "Public fixture escaped store")
            bound[f"row_{rank}"] = h.binding(path)
            weights, seed = path.parent / "weights.npy", int(h.load_json(path)["mlp_seed"])
        else:
            weights, seed = G19 / f"fresh_{rank}_weights.npy", rank
        bound[f"weights_{rank}"] = h.binding(weights)
        rows.append(dict(rank=rank, weights_path=str(weights.resolve()), mlp_seed=seed))
    h.check_bindings(bound)
    return bound, rows


def verify_role(h, role):
    directory = ROOT / role
    complete = h.load_json(directory / "completion.json")
    h.require(complete["schema"] == SCHEMA and complete["role"] == role
              and complete["qualification_passed"], "Panel role did not complete successfully")
    h.require(h.digest(directory / "result.json") == complete["result_sha256"], "Role result seal changed")
    result = h.load_json(directory / "result.json")
    h.require(result["schema"] == SCHEMA and result["role"] == role and result["completed"]
              and result["qualification_passed"] and result["hashes_unchanged"]
              and not result["truth_files_accessed"] and not result["instrumented"], "Invalid panel acquisition")
    candidate = h.RESPONSE if role == "response" else h.BANK
    h.require(result["candidate"] == candidate and result["candidate_sha256"] == h.PINNED[candidate],
              "Wrong candidate assigned to panel role")
    h.require(result["source_bindings"] == sources(h), "Panel source/provenance set changed")
    h.check_bindings(result["fixture_bindings"])
    h.check_bindings(result["artifacts"])
    h.require(h.digest(directory / "started.json") == result["started_sha256"], "Start seal changed")
    started = h.load_json(directory / "started.json")
    h.require(started == result["started"], "Acquisition header differs from its seal")
    h.require(started["pid"] == result["pid"] and started["source_bindings"] == result["source_bindings"]
              and started["fixture_bindings"] == result["fixture_bindings"]
              and started["versions"] == result["versions"] and started["nice"] >= 10,
              "Acquisition header/terminal provenance differs")
    h.require(result["versions"]["flopscope"] == "0.12.0" and result["versions"]["whestbench"] == "0.16.0"
              and result["blas_env"] == dict.fromkeys(BLAS_ENV, "1"), "Native runtime frame changed")
    h.require([t["rank"] for t in result["trials"]] == list(RANKS), "Wrong14-net panel sequence")
    expected_artifacts = {f"call_{i:02d}_rank{rank}_output.npy" for i, rank in enumerate(RANKS)}
    expected_artifacts.update(f"call_{i:02d}_bill.json" for i in range(len(RANKS)))
    h.require(set(result["artifacts"]) == expected_artifacts, "All28 outputs/bills must be sealed")
    for call, trial in enumerate(result["trials"]):
        h.require(trial["call"] == call and 0 < trial["flops"] <= h.BUDGET
                  and trial["context_wall_s"] < 120 and trial["peak_rss_bytes"] < h.RSS_LIMIT
                  and math.isfinite(trial["native_residual_s"]), "Invalid completed native call")
        h.require(h.load_json(directory / f"call_{call:02d}_bill.json") == trial, "Per-call bill changed")
    for name, item in result["artifacts"].items():
        h.require(Path(item["path"]).resolve() == (directory / name).resolve(), "Artifact escaped role directory")
    return result


def acquire(h, role, directory):
    h.require(Path(sys.prefix).resolve() == h.GATE_ENV.resolve() and "anaconda3" not in sys.executable
              and "anaconda3" not in sys.prefix, "Use OFFPY data_gate/.venv; never AP/Torch")
    h.require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "Launch at nice>=10")
    env = {key: os.environ.get(key) for key in BLAS_ENV}
    h.require(env == dict.fromkeys(BLAS_ENV, "1"), "Set all three BLAS launch variables to1 BEFORE imports")
    prior = verify_role(h, "response") if role == "bank" else None
    h.require(prior is None or prior["pid"] != os.getpid(), "Roles require separate fresh processes")
    source_set = sources(h)
    fixture_set, rows = fixtures(h)
    versions = {name: importlib.metadata.version(name) for name in ("flopscope", "whestbench", "numpy", "threadpoolctl")}
    h.require(versions["flopscope"] == "0.12.0" and versions["whestbench"] == "0.16.0", "Wrong native versions")
    import numpy as np
    import flopscope as flops
    from threadpoolctl import threadpool_info, threadpool_limits
    from whestbench.domain import MLP
    from whestbench.seeds import derive_estimator_seed
    flops.configure(symmetry_warnings=False)
    started = dict(schema=SCHEMA, role=role, pid=os.getpid(), utc=h.utc(), executable=sys.executable,
                   nice=os.getpriority(os.PRIO_PROCESS, 0), blas_env=env, versions=versions,
                   sequence=list(RANKS), source_bindings=source_set, fixture_bindings=fixture_set,
                   truth_files_accessed=False, instrumented=False, flop_budget=h.BUDGET, wall_limit_s=120)
    start_sha = h.write_json(directory / "started.json", started)
    candidate = h.RESPONSE if role == "response" else h.BANK
    t0 = time.perf_counter()
    spec = importlib.util.spec_from_file_location("fixed_panel_unwrapped_candidate", HERE / candidate)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    import_wall = time.perf_counter() - t0
    trials, artifacts = [], {}
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        h.require(all(p["num_threads"] == 1 for p in pools if p.get("user_api") == "blas"), "Reported BLAS threads differ from1")
        for call, row in enumerate(rows):
            weights = np.load(row["weights_path"], mmap_mode="r", allow_pickle=False)
            h.require(weights.shape == (16, 1024, 1024) and weights.dtype.kind == "f"
                      and bool(np.isfinite(weights).all()), "Invalid Phase2 weights")
            mlp = MLP(width=1024, depth=16, weights=list(weights),
                      seed=int(derive_estimator_seed(row["mlp_seed"])), name=f"response-panel-{row['rank']}")
            began = h.utc(); t0 = time.perf_counter()
            with flops.BudgetContext(flop_budget=h.BUDGET, wall_time_limit_s=120, quiet=True) as context:
                returned = module.Estimator().predict(mlp, h.BUDGET)
            elapsed = time.perf_counter() - t0
            out = np.asarray(returned).copy()  # Preserve served bytes: NO clamp/cast repair.
            h.assert_output(np, out)
            name = f"call_{call:02d}_rank{row['rank']}_output.npy"
            artifacts[name] = h.save_npy(np, directory / name, out)
            trial = dict(call=call, rank=row["rank"], began_utc=began, ended_utc=h.utc(),
                         flops=int(context.flops_used), context_wall_s=elapsed,
                         native_residual_s=float(context.residual_wall_time_s), peak_rss_bytes=h.peak_rss_bytes(),
                         output_artifact=name, output_bytes_sha256=hashlib.sha256(out.tobytes(order="C")).hexdigest(),
                         includes_first_call_initialization=call == 0, instrumented=False)
            trials.append(trial)
            bill = f"call_{call:02d}_bill.json"
            h.write_json(directory / bill, trial); artifacts[bill] = h.binding(directory / bill)
            print(json.dumps(dict(role=role, **trial)), flush=True)
            h.require(0 < trial["flops"] <= h.BUDGET and elapsed < 120
                      and trial["peak_rss_bytes"] < h.RSS_LIMIT and math.isfinite(trial["native_residual_s"]),
                      "Native budget/wall/RSS qualification failed; preserve this call")
            del weights, mlp, returned, out
    h.check_bindings(source_set); h.check_bindings(fixture_set); h.check_bindings(artifacts)
    result = dict(schema=SCHEMA, role=role, pid=os.getpid(), completed=True, qualification_passed=True,
                  hashes_unchanged=True, truth_files_accessed=False, instrumented=False,
                  started=started, started_sha256=start_sha, completed_utc=h.utc(), versions=versions,
                  blas_env=env, threadpools=pools, candidate=candidate, candidate_sha256=source_set[candidate]["sha256"],
                  candidate_import_wall_s=import_wall, source_bindings=source_set, fixture_bindings=fixture_set,
                  trials=trials, artifacts=artifacts, production_residual_validated=False)
    digest = h.write_json(directory / "result.json", result)
    h.write_json(directory / "completion.json", dict(schema=SCHEMA, role=role, result_sha256=digest,
                 qualification_passed=True, completed_utc=h.utc(), truth_files_accessed=False))
    print(json.dumps(dict(role=role, qualification_passed=True, result_sha256=digest)), flush=True)


def score(h, directory):
    roles = {role: verify_role(h, role) for role in ("response", "bank")}
    response, bank = roles["response"], roles["bank"]
    h.require(response["pid"] != bank["pid"] and response["completed_utc"] <= bank["started"]["utc"], "Wrong fresh-process role order")
    h.require(response["source_bindings"] == bank["source_bindings"]
              and response["fixture_bindings"] == bank["fixture_bindings"]
              and response["versions"] == bank["versions"], "Roles do not share the frozen frame")
    import numpy as np  # No candidate/FlopScope import or estimator invocation during scoring.
    h.require(importlib.metadata.version("numpy") == bank["versions"]["numpy"], "Scoring NumPy differs")
    outputs = {}
    for role, record in roles.items():
        outputs[role] = []
        for trial in record["trials"]:
            out = np.load(ROOT / role / trial["output_artifact"], allow_pickle=False)
            h.assert_output(np, out)
            h.require(hashlib.sha256(out.tobytes(order="C")).hexdigest() == trial["output_bytes_sha256"], "Served bytes changed")
            outputs[role].append(out)
    sealed = {f"{role}_{name}": h.binding(ROOT / role / name) for role in roles
              for name in ("completion.json", "result.json", "started.json")}
    before_sha = h.write_json(directory / "both_roles_verified_before_truth.json", dict(
        schema=SCHEMA, utc=h.utc(), role_bindings=sealed, truth_files_accessed=False,
        source_bindings=sources(h), ranks=list(RANKS)))
    truth_time = h.utc()
    h.write_json(directory / "truth_access_started.json", dict(utc=truth_time, prior_seal_sha256=before_sha))
    truths, scored = {}, {role: [] for role in roles}
    for index, rank in enumerate(RANKS):
        path = G19 / (f"mps_truth_rank{rank:04d}_indep4M.npz" if rank in PUBLIC else f"fresh_{rank}_truth4M.npz")
        truths[str(rank)] = h.binding(path)
        with np.load(path, allow_pickle=False) as archive:
            mean, se = archive["mc_post"][15].astype(np.float64), archive["mc_se"][15].astype(np.float64)
        h.require(mean.shape == se.shape == (1024,) and bool(np.isfinite(mean).all())
                  and bool(np.isfinite(se).all()) and bool((se >= 0).all()), "Invalid independent4M truth")
        noise = float(np.mean(se * se))
        for role, record in roles.items():
            trial = record["trials"][index]
            raw = float(np.mean((outputs[role][index][15].astype(np.float64) - mean) ** 2))
            nc = raw - noise  # Deliberately NOT clipped: the established noise-corrected estimator.
            multiplier = max(0.1, trial["flops"] / h.BUDGET)
            scored[role].append(dict(**trial, raw_observed=raw, mc_se_squared_mean=noise,
                                    raw_noise_corrected=nc, multiplier=multiplier,
                                    adjusted_noise_corrected=nc * multiplier, adjusted_observed=raw * multiplier))
    summaries = {}
    for role, rows in scored.items():
        summaries[role] = {}
        for group, ranks in (("public8", PUBLIC), ("virgin6", VIRGIN), ("combined14", RANKS)):
            subset = [r for r in rows if r["rank"] in ranks]
            fields = ("raw_observed", "raw_noise_corrected", "mc_se_squared_mean", "flops", "adjusted_observed", "adjusted_noise_corrected")
            summaries[role][group] = dict(n=len(subset), **{f"mean_{k}": sum(r[k] for r in subset) / len(subset) for k in fields})
    wins = {g: summaries["response"][g]["mean_adjusted_noise_corrected"] < summaries["bank"][g]["mean_adjusted_noise_corrected"]
            for g in ("public8", "virgin6")}
    h.check_bindings(truths); h.check_bindings(sealed)
    for record in roles.values():
        for key in ("source_bindings", "fixture_bindings", "artifacts"):
            h.check_bindings(record[key])
    result = dict(schema=SCHEMA, qualification_passed=True, hashes_unchanged=True, completed_utc=h.utc(),
                  both_roles_verified_before_truth_sha256=before_sha, truth_access_started_utc=truth_time,
                  truth_bindings=truths, source_bindings=sources(h), per_role=scored, summaries=summaries,
                  role_candidates={role: dict(name=r["candidate"], sha256=r["candidate_sha256"])
                                   for role, r in roles.items()},
                  group_improved=wins, retain_validated_gain=all(wins.values()),
                  aggregation="Mean of actual per-net (observed MSE-mean(mc_se15^2))*max(.1,F/2^41); no clipping; call0 initialization included",
                  frame="Existing independent4M public8/virgin6 panel; distinct from1e9 development and official scoring",
                  native_timing_only=True, production_residual_validated=False, official_score=False,
                  milestone_achieved=False, goal_achieved=False)
    result_sha = h.write_json(directory / "result.json", result)
    h.write_json(directory / "completion.json", dict(schema=SCHEMA, qualification_passed=True,
                 result_sha256=result_sha, retain_validated_gain=all(wins.values()), completed_utc=h.utc()))
    print(json.dumps(dict(group_improved=wins, retain_validated_gain=all(wins.values()),
                          summaries=summaries, result_sha256=result_sha)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--role", choices=("response", "bank"))
    mode.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if args.role is None and not args.score:
        parser.print_help(); return 0
    h = load_helper()
    if args.role == "response":
        ROOT.mkdir(exist_ok=False)
    h.require(ROOT.is_dir() and not ROOT.is_symlink(), "Acquire response first in the fixed new root")
    directory = ROOT / ("score" if args.score else args.role)
    directory.mkdir(exist_ok=False)
    try:
        score(h, directory) if args.score else acquire(h, args.role, directory)
    except Exception as error:
        h.write_json(directory / "failure.json", dict(schema=SCHEMA, utc=h.utc(), qualification_passed=False,
                     error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
