"""Inert fixed Gram512 remote prepare/qualification; never launch or fit here."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import math
import os
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
ROOT = HERE / "remote"
PANEL = HERE / "panel_v2"
CANDIDATE = "resident_gram512_v1/estimator.py"
CANDIDATE_SHA = "0c21db7a74ae0a271d52fddb5b925318be61100e0676393c87f2f15ce9a997dc"
PARENT_SHA = "07b71a03e13f8daed17e0ee529c4113e88cd89c2cceb29ea0e2f148084e74d30"
TAG = "resident_gram512_openblas_series_v1"
OLD_TAG = "terminal_dead_consumers_openblas_series_v1"
SCHEMA = "resident_gram512_remote_v1"
RANKS = (35, 100, 500, 3016500, 35)
PANEL_RANKS = (35, 100, 150, 200, 250, 300, 400, 500,
               3016500, 3016501, 3016502, 3016503, 3016504, 3016505)
UTILITY = NEW / "terminal_dead_consumers_v1/check_remote_v2.py"
UTILITY_SHA = "76f6510e724e03505d8060fa159dc0df40ad8f9ea75a740d171b5c3d24c16fa4"
PINS = {
    CANDIDATE: CANDIDATE_SHA,
    "terminal_dead_consumers_v1/estimator.py": PARENT_SHA,
    "resident_gram512_v1/source.py": "939ca1f8aaf8c4b3a6961231631d98e93d4723e6e3fbd3fc311487357b05dc96",
    "resident_gram512_v1/build.json": "3f04e57a820508928f409ddb45083cfcb7dee7499f6d62a5d85b6545ca267da3",
    "resident_gram512_v1/PROTOCOL.md": "e262eb588587a148205b791fd3ed79a368808ac164f18aba885e03e21b5ec191",
    "resident_gram512_v1/dev/score/result.json": "fd015efa3d31488f7f15b940d9ceec0d2f7020b7dd1cbdc4cb262479e4154866",
    "terminal_dead_consumers_v1/remote/result.json": "b9695e839d4ba190fc82d933b5981af9409ef679c869efe8c8f470bc70d713bf",
    "terminal_dead_consumers_v1/panel/candidate/result.json": "302d9bbaf1327e0f12aa8746406536e24ed85beca6ec23eb807c8a2a03cd7344",
    "run_remote_series_v1.py": "00eabab0b7c1ec0c113b998032d441b60aa882670f364732f02a741282634cb1",
    "check_remote_series_v1.py": "f3b0b9dd9e9edd27bf4f85b5a34ab0d72131e22881dd0531b3201dc29642da55",
}


def utility():
    if hashlib.sha256(UTILITY.read_bytes()).hexdigest() != UTILITY_SHA:
        raise RuntimeError("Frozen inert provenance utility changed")
    spec = importlib.util.spec_from_file_location("gram_remote_utilities", UTILITY)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import inert provenance utility")
    r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r)
    return r


def command(r):
    return [str(r.ENV / "bin/python"), "-u", str(NEW / "run_remote_series_v1.py"),
            "--candidate", CANDIDATE, "--tag", TAG]


def verify_inputs(r, audit, pins):
    for name, sha in PINS.items():
        audit.bind(name, NEW / name, sha=sha)
    audit.bind("utility", UTILITY, sha=UTILITY_SHA)
    audit.bind("remote_checker", __file__)
    audit.bind("remote_protocol", HERE / "REMOTE_PROTOCOL.md")
    for key, path in (("panel_checker", HERE / "check_panel_v2.py"),
                      ("panel_acquire", PANEL / "candidate/result.json"),
                      ("panel_score", PANEL / "score/result.json")):
        audit.bind(key, path, sha=pins[key + "_sha256"])
    acq = r.load(PANEL / "candidate/result.json")
    score = r.load(PANEL / "score/result.json")
    for role, record, key in (("candidate", acq, "panel_acquire"), ("score", score, "panel_score")):
        path = PANEL / role / "completion.json"
        audit.bind(role + "_completion", path)
        seal = r.load(path)
        r.require(seal["qualification_passed"] and record["qualification_passed"]
                  and record["completed"] and record["hashes_unchanged"]
                  and seal["result_sha256"] == pins[key + "_sha256"]
                  and record["candidate_sha256"] == CANDIDATE_SHA
                  and record["parent_sha256"] == PARENT_SHA, "Unqualified native14 receipt")
    r.require(not acq["instrumented"] and not acq["truth_files_accessed"]
              and score["retain_local_component"] and score["baseline_recomputed_exact"]
              and score["group_strict_gain_gate"] == {"public8": True, "virgin6": True}
              and all(0 < score["adjusted_candidate_over_parent"][g] < 1
                      for g in ("public8", "virgin6", "combined14"))
              and acq["source_bindings"] == score["source_bindings"], "Native adjusted gate failed")
    audit.recorded("native_source/", acq["source_bindings"])
    audit.recorded("native_fixture/", acq["fixture_bindings"])
    audit.recorded("native_artifact/", acq["artifacts"])
    audit.recorded("native_score_seal/", score["role_seals"])
    r.require([t["rank"] for t in acq["trials"]] == list(PANEL_RANKS)
              and len(acq["comparisons"]) == 14, "Changed native panel sequence")
    expected = {f"call_{i:02d}_rank{rank}_output.npy" for i, rank in enumerate(PANEL_RANKS)}
    expected.update(f"call_{i:02d}_bill.json" for i in range(14))
    r.require(set(acq["artifacts"]) == expected, "Incomplete native artifacts")
    for name, item in acq["artifacts"].items():
        r.require(Path(item["path"]).resolve() == (PANEL / "candidate" / name).resolve(), "Native artifact path")
    parent_native = r.load(NEW / "terminal_dead_consumers_v1/panel/candidate/result.json")
    for i, (trial, old) in enumerate(zip(acq["trials"], parent_native["trials"], strict=True)):
        r.require(trial["call"] == i and trial["includes_first_call_initialization"] == (i == 0)
                  and type(trial["flops"]) is int and 0 < trial["flops"] < old["flops"] <= 2**41
                  and trial["served_materialized_byte_exact"]
                  and trial["output_artifact"] == f"call_{i:02d}_rank{PANEL_RANKS[i]}_output.npy"
                  and r.load(PANEL / "candidate" / f"call_{i:02d}_bill.json") == trial,
                  "Native complete bill or output identity changed")
        r.require(score["per_role"]["candidate"][i]["flops"] == trial["flops"]
                  and score["per_role"]["candidate"][i]["output_bytes_sha256"] == trial["output_bytes_sha256"],
                  "Scored output differs from measured one")
    start_path = PANEL / "candidate/started.json"
    audit.bind("native_start", start_path, sha=acq["started_sha256"])
    r.require(r.load(start_path) == acq["started"]
              and acq["started"]["sequence"] == list(PANEL_RANKS)
              and acq["started"]["one_estimator_instance"]
              and acq["started"]["initialization_once_in_call0"]
              and acq["started"]["no_warmup"] and acq["started"]["no_reset"], "Native process contract")
    seal_path = PANEL / "candidate/predictions_and_bills_sealed.json"
    audit.bind("native_predictions_seal", seal_path, sha=acq["predictions_and_bills_seal_sha256"])
    sealed = r.load(seal_path)
    for key in ("trials", "artifacts", "comparisons", "source_bindings", "fixture_bindings", "candidate_sha256"):
        r.require(sealed[key] == acq[key], "Native prediction seal changed")
    audit.bind("native_before_truth", PANEL / "score/both_roles_verified_before_truth.json",
               sha=score["both_roles_verified_before_truth_sha256"])
    audit.bind("native_truth_marker", PANEL / "score/truth_access_started.json")
    before = r.load(PANEL / "score/both_roles_verified_before_truth.json")
    marker = r.load(PANEL / "score/truth_access_started.json")
    r.require(not sealed["truth_files_accessed"] and not before["truth_files_accessed"]
              and before["role_seals"] == score["role_seals"]
              and marker["prior_seal_sha256"] == score["both_roles_verified_before_truth_sha256"]
              and r.timestamp(sealed["sealed_utc"]) <= r.timestamp(before["utc"]) <= r.timestamp(marker["utc"]),
              "Native truth chronology differs")
    # The native score's truth bindings are metadata here, NOT followed.
    old_qualified = r.load(NEW / "terminal_dead_consumers_v1/remote/result.json")
    audit.bind("parent_remote_completion", NEW / "terminal_dead_consumers_v1/remote/completion.json")
    old_complete = r.load(NEW / "terminal_dead_consumers_v1/remote/completion.json")
    r.require(old_qualified["passed"] and old_qualified["hashes_unchanged"]
              and old_qualified["candidate_sha256"] == PARENT_SHA and old_qualified["tag"] == OLD_TAG
              and old_complete["passed"]
              and old_complete["result_sha256"] == PINS["terminal_dead_consumers_v1/remote/result.json"],
              "Wrong qualified remote parent")
    audit.recorded("parent_remote_input/", old_qualified["input_bindings"])
    old_remote = r.load(NEW / (OLD_TAG + ".json"))
    r.require(old_remote["status"] == "completed" and old_remote["source_sha256"] == PARENT_SHA
              and old_remote["repeated_output_byte_identical"] and len(old_remote["records"]) == 5,
              "Wrong parent remote series")
    for i, rank in enumerate(RANKS):
        row = old_remote["records"][i]
        item = audit.bind(f"parent_output/{i}", NEW / f"{OLD_TAG}_{i}_r{rank}.f32", sha=row["output_sha256"])
        r.require(row["index"] == i and row["rank"] == rank and item["bytes"] == 65536,
                  "Wrong parent remote output")
        r.require(acq["fixture_bindings"][f"weights_{rank}"] ==
                  old_qualified["input_bindings"][f"prelaunch/panel_fixture/weights_{rank}"],
                  "Native/remote fixture ownership differs")
    audit.recheck()
    return acq, score, old_remote


def prepare(r, audit, pins):
    paths = [NEW / (TAG + suffix) for suffix in (".json", "_resources.json", "_server.log")]
    paths += [NEW / f"{TAG}_{i}_r{rank}.f32" for i, rank in enumerate(RANKS)]
    r.require(not any(path.exists() for path in paths), "Refusing existing remote artifacts")
    acq, _, _ = verify_inputs(r, audit, pins)
    record = dict(schema=SCHEMA, utc=r.utc(), candidate=CANDIDATE, candidate_sha256=CANDIDATE_SHA,
                  parent_sha256=PARENT_SHA, tag=TAG, sequence=list(RANKS), provenance_pins=pins,
                  input_bindings=audit.files, fixture_bindings=acq["fixture_bindings"],
                  env=r.LAUNCH_ENV, command=command(r), cwd=str(NEW), nice=os.getpriority(os.PRIO_PROCESS, 0),
                  remote_launched=False, estimator_executed=False, truth_arrays_accessed=False)
    sha = r.write(ROOT / "launch.json", record)
    r.write(ROOT / "prepare_completion.json", dict(passed=True, launch_sha256=sha, utc=r.utc()))
    print(__import__("json").dumps(dict(prepared=True, launch_sha256=sha, command=record["command"],
                                       env=r.LAUNCH_ENV, cwd=str(NEW))), flush=True)


def qualify(r, audit, launch_sha):
    launch_item = audit.bind("launch", ROOT / "launch.json", sha=launch_sha)
    launch = r.load(ROOT / "launch.json")
    audit.bind("prepare_completion", ROOT / "prepare_completion.json")
    complete = r.load(ROOT / "prepare_completion.json")
    r.require(complete["passed"] and complete["launch_sha256"] == launch_sha
              and launch["schema"] == SCHEMA and launch["candidate"] == CANDIDATE
              and launch["candidate_sha256"] == CANDIDATE_SHA and launch["parent_sha256"] == PARENT_SHA
              and launch["tag"] == TAG and launch["sequence"] == list(RANKS)
              and launch["env"] == r.LAUNCH_ENV and launch["command"] == command(r)
              and launch["cwd"] == str(NEW) and launch["nice"] >= 10
              and not launch["remote_launched"] and not launch["estimator_executed"]
              and not launch["truth_arrays_accessed"], "Wrong prepared contract")
    current = r.Audit()
    acq, score, old_series = verify_inputs(r, current, launch["provenance_pins"])
    r.require(current.files == launch["input_bindings"] and acq["fixture_bindings"] == launch["fixture_bindings"],
              "Inputs changed after preparation")
    audit.recorded("prelaunch/", current.files)
    r.require(all(item["mtime_ns"] <= launch_item["mtime_ns"] for item in current.files.values()),
              "Input newer than launch")
    for suffix in (".json", "_resources.json", "_server.log"):
        item = audit.bind("new_remote" + suffix, NEW / (TAG + suffix))
        r.require(item["mtime_ns"] >= launch_item["mtime_ns"], "Remote artifact predates launch")
    series = r.load(NEW / (TAG + ".json"))
    resources = r.load(NEW / (TAG + "_resources.json"))
    r.require(series["status"] == "completed" and series["source_sha256"] == CANDIDATE_SHA
              and series["candidate"] == resources["candidate"] == CANDIDATE
              and series["repeated_output_byte_identical"] and len(series["records"]) == 5,
              "Fixed remote series incomplete")
    import numpy as np
    r.require(importlib.metadata.version("numpy") == acq["versions"]["numpy"] == "2.4.6", "Comparison environment")

    def valid(value):
        r.require(value.shape == (16, 1024) and value.dtype == np.dtype("float32")
                  and bool(np.isfinite(value).all()) and bool((value >= 0).all()), "Invalid served values")

    checks, predictions, parent_predictions = [], [], []
    for i, rank in enumerate(RANKS):
        row, old = series["records"][i], old_series["records"][i]
        r.require(row["index"] == i and row["rank"] == rank and row["output_shape"] == [16, 1024]
                  and row["output_dtype"] == "float32", "Remote sequence/output schema")
        item = audit.bind(f"new_output/{i}", NEW / f"{TAG}_{i}_r{rank}.f32", sha=row["output_sha256"])
        r.require(item["bytes"] == 65536 and item["mtime_ns"] >= launch_item["mtime_ns"], "Remote output binding")
        raw = Path(item["path"]).read_bytes()
        r.require(hashlib.sha256(raw).hexdigest() == row["output_sha256"], "Changed remote bytes")
        value = np.frombuffer(raw, dtype=np.float32).reshape(16, 1024)
        old_raw = (NEW / f"{OLD_TAG}_{i}_r{rank}.f32").read_bytes()
        r.require(hashlib.sha256(old_raw).hexdigest() == old["output_sha256"], "Changed parent bytes")
        parent = np.frombuffer(old_raw, dtype=np.float32).reshape(16, 1024)
        valid(value); valid(parent)
        error_parent = np.max(np.abs(value.astype(np.float64) - parent.astype(np.float64)), axis=1)
        r.require(bool((error_parent <= 1e-5).all()), "Same-backend parent trajectory drift")
        timing, flops = row["timing"], row["flops"]
        r.require(all(isinstance(t, (float, int)) and not isinstance(t, bool)
                      and math.isfinite(t) and t >= 0 for t in timing.values()), "Invalid remote timings")
        r.require(0 < timing["wall_time_s"] < 60 and 0 <= timing["residual_wall_time_s"] < .4
                  and type(flops) is int and 0 < flops < old["flops"] <= 2**41
                  and row["util"] == flops / 2**41, "Remote complete cost/wall/residual gate")
        call = PANEL_RANKS.index(rank)
        trial = acq["trials"][call]
        native = np.load(PANEL / "candidate" / trial["output_artifact"], allow_pickle=False)
        valid(native)
        r.require(hashlib.sha256(native.tobytes()).hexdigest() == trial["output_bytes_sha256"], "Native bytes")
        diff = value.astype(np.float64) - native.astype(np.float64)
        r.require(float(np.max(np.abs(diff))) < 1e-5, "Native/remote candidate drift")
        predictions.append(value); parent_predictions.append(parent)
        checks.append(dict(index=i, rank=rank, flops=flops, parent_flops=old["flops"],
                           saved_flops=old["flops"] - flops, wall_s=timing["wall_time_s"],
                           residual_s=timing["residual_wall_time_s"], output_sha256=row["output_sha256"],
                           parent_per_layer_max_abs=error_parent.tolist(), native14_max_abs=float(np.max(np.abs(diff))),
                           native14_rms=float(np.sqrt(np.mean(diff * diff))), finite=True, nonnegative=True))
    r.require(predictions[0].tobytes() == predictions[4].tobytes()
              and series["records"][0]["flops"] - series["records"][4]["flops"] == 1383913666,
              "Remote repeat bytes/initialization bill changed")
    samples = resources["samples"]
    r.require(samples and all(isinstance(s, list) and len(s) == 2 and r.positive(s[0])
                             and type(s[1]) is int and s[1] >= 0 for s in samples), "RSS samples")
    peak = resources["peak_combined_rss_bytes"]
    r.require(type(peak) is int and peak == max(s[1] for s in samples) and 0 < peak < 8_000_000_000
              and r.positive(resources["wall_s"]), "Remote RSS/resources gate")
    audit.recheck()
    seal_sha = r.write(ROOT / "both_remote_roles_sealed_before_truth.json", dict(
        utc=r.utc(), input_bindings=audit.files, checks=checks, truth_arrays_accessed=False))
    r.write(ROOT / "truth_access_started.json", dict(utc=r.utc(), prior_seal_sha256=seal_sha))
    per_role = {"parent": [], "candidate": []}
    truth_bindings = {}
    for i, rank in enumerate(RANKS[:4]):
        expected = score["truth_bindings"][str(rank)]
        truth_bindings[str(rank)] = r.binding(expected["path"])
        r.require(truth_bindings[str(rank)] == expected, "Independent4M truth binding changed")
        with np.load(expected["path"], allow_pickle=False) as data:
            mean = data["mc_post"][15].astype(np.float64)
            se = data["mc_se"][15].astype(np.float64)
        r.require(mean.shape == se.shape == (1024,) and bool(np.isfinite(mean).all())
                  and bool(np.isfinite(se).all()) and bool((se >= 0).all()), "Independent4M truth")
        noise = float(np.mean(se * se))
        for role, outputs, record in (("parent", parent_predictions, old_series), ("candidate", predictions, series)):
            raw = float(np.mean((outputs[i][15].astype(np.float64) - mean) ** 2))
            flops = record["records"][i]["flops"]
            mult = max(.1, flops / 2**41)
            per_role[role].append(dict(rank=rank, flops=flops, multiplier=mult, raw_observed=raw,
                                      noise=noise, raw_noise_corrected=raw-noise,
                                      adjusted_observed=raw*mult, adjusted_noise_corrected=(raw-noise)*mult))
    groups = {"public3": (35, 100, 500), "virgin1": (3016500,), "combined4": RANKS[:4]}
    summaries = {role: {g: {"mean_" + key: sum(t[key] for t in rows if t["rank"] in ranks) / len(ranks)
                                  for key in ("raw_observed", "raw_noise_corrected", "flops", "adjusted_noise_corrected")}
                       for g, ranks in groups.items()} for role, rows in per_role.items()}
    r.require(all(math.isfinite(summaries["parent"][g]["mean_adjusted_noise_corrected"])
                  and summaries["parent"][g]["mean_adjusted_noise_corrected"] > 0 for g in groups),
              "Invalid parent adjusted denominator; noise correction is not clipped")
    ratios = {g: summaries["candidate"][g]["mean_adjusted_noise_corrected"] /
                 summaries["parent"][g]["mean_adjusted_noise_corrected"] for g in groups}
    r.require(all(math.isfinite(v) and v > 0 for v in ratios.values()), "Invalid adjusted ratios")
    passed = all(v < 1 for v in ratios.values())
    audit.recheck()
    for item in truth_bindings.values():
        r.require(r.binding(item["path"]) == item, "Truth changed after scoring")
    result = dict(schema=SCHEMA, passed=passed, completed_utc=r.utc(), candidate_sha256=CANDIDATE_SHA,
                  parent_sha256=PARENT_SHA, candidate=CANDIDATE, tag=TAG, checks=checks,
                  input_bindings=audit.files, hashes_unchanged=True, prior_native14_pass_verified=True,
                  runtime_gates_passed=True, strict_complete_cost_saving_all5=True,
                  repeated_rank35_byte_exact=True, native14_comparison_limit=1e-5,
                  peak_combined_rss_bytes=peak, launch_sha256=launch_sha,
                  both_roles_before_truth_sha256=seal_sha, truth_bindings=truth_bindings,
                  per_role=per_role, summaries=summaries, adjusted_candidate_over_parent=ratios,
                  backend_subset_adjusted_gate_passed=passed, official_score=False, milestone_achieved=False,
                  child_environment_independently_observed=False, archived_runtime_not_contemporaneous=True,
                  frame="Fixed four-unique remote compatibility subset of independent14; repeat excluded. Actual same-backend outputs/full bills. Not a new holdout or official/Linux guarantee.")
    sha = r.write(ROOT / "result.json", result)
    r.write(ROOT / "completion.json", dict(passed=passed, result_sha256=sha, utc=r.utc()))
    print(__import__("json").dumps(dict(passed=passed, checks=checks, peak_combined_rss_bytes=peak,
                                       summaries=summaries, adjusted_candidate_over_parent=ratios,
                                       result_sha256=sha)), flush=True)
    return 0 if passed else 1


def main():
    r = utility()
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--qualify", action="store_true")
    for name in ("panel-acquire", "panel-score", "panel-checker", "launch"):
        parser.add_argument("--" + name + "-sha256", type=r.sha_arg)
    args = parser.parse_args()
    if not args.prepare and not args.qualify:
        parser.print_help(); return 0
    keys = ("panel_acquire_sha256", "panel_score_sha256", "panel_checker_sha256")
    r.require((args.prepare and args.launch_sha256 is None and all(getattr(args, k) for k in keys))
              or (args.qualify and args.launch_sha256 and all(getattr(args, k) is None for k in keys)),
              "Prepare needs all panel pins; qualify needs only launch pin")
    if args.prepare:
        ROOT.mkdir(exist_ok=False)
    r.require(ROOT.is_dir() and not ROOT.is_symlink(), "Exclusive prepared directory required")
    stage = "prepare" if args.prepare else "qualify"
    audit = r.Audit()
    r.write(ROOT / (stage + "_started.json"), dict(utc=r.utc(), arguments=vars(args), source=r.binding(__file__)))
    try:
        r.environment()
        if args.prepare:
            prepare(r, audit, {k: getattr(args, k) for k in keys}); return 0
        return qualify(r, audit, args.launch_sha256)
    except BaseException:
        r.write(ROOT / (stage + "_failure.json"), dict(passed=False, utc=r.utc(), error=traceback.format_exc(),
                input_bindings=audit.files))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
