"""Inert terminal-component remote provenance and artifact qualification.

--prepare requires pinned passing native acquisition, score and checker hashes.
It writes an exclusive launch record/command; it DOES NOT launch that command.
--qualify requires the launch hash and checks completed fixed remote artifacts.
Neither stage imports an estimator, executes a network, or opens truth arrays.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
NEW = HERE.parent
ENV = NEW.parent / "data_gate/.venv"
PANEL = HERE / "panel"
ROOT = HERE / "remote"
CANDIDATE = "terminal_dead_consumers_v1/estimator.py"
TAG = "terminal_dead_consumers_openblas_series_v1"
OLD_TAG = "pair_polynomial_f32_openblas_series_v1"
SCHEMA = "terminal_dead_consumers_remote_v1"
PANEL_SCHEMA = "terminal_dead_consumers_fixed14_v1"
RANKS = (35, 100, 500, 3016500, 35)
PANEL_RANKS = (35, 100, 150, 200, 250, 300, 400, 500,
               3016500, 3016501, 3016502, 3016503, 3016504, 3016505)
PARENT_SHA = "0b84ab6fa96d32804c9adec350d41766e47d09a71a936b95652893bb1e364a96"
CANDIDATE_SHA = "07b71a03e13f8daed17e0ee529c4113e88cd89c2cceb29ea0e2f148084e74d30"
LAUNCH_ENV = dict(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
                  MKL_NUM_THREADS="1", BLIS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                  PYTHONDONTWRITEBYTECODE="1", FLOPSCOPE_MAX_ARRAY_BYTES="4294967296")
PINS = {
    CANDIDATE: CANDIDATE_SHA,
    "candidate_response_pair_polynomial_f32_v1.py": PARENT_SHA,
    "terminal_dead_consumers_v1/source.py": "f87af2c86acbea4d83ba8ad34ef73759a3a38857dc4ac4f4ae622df98dea6e17",
    "terminal_dead_consumers_v1/PROTOCOL.md": "cc1d6202ab22f575ec571c794182a85bd8e2ecd3f894102049c54b0e17a8b1c5",
    "terminal_dead_consumers_v1/build.json": "37a99946bc5993e144d1d4e7fc78ed049d7c669cb9a88ea5d93ca05177afb4ed",
    "terminal_dead_consumers_v1/check_synthetic.py": "501d9393b362662ad198ef77efeba56f95760ca5d178fdcf8e013e366f6baf3b",
    "terminal_dead_consumers_v1/synthetic/result.json": "0be71565fed36857f51b826c4116c570074126b4de73b9713109103ed890a54d",
    "run_remote_series_v1.py": "00eabab0b7c1ec0c113b998032d441b60aa882670f364732f02a741282634cb1",
    "check_remote_series_v1.py": "f3b0b9dd9e9edd27bf4f85b5a34ab0d72131e22881dd0531b3201dc29642da55",
    "pair_polynomial_f32_remote_qualified_v1.json": "b039087055743e97be5006723894ee3dac26eb1db1eedc1cb0213d8d669a8e4e",
    OLD_TAG + ".json": "1ee2833f75770c5e2e8e0d899d03d6dd61326c90fd4553bd474838d452ebb273",
    "pair_polynomial_f32_panel_v1/candidate/result.json": "72b41dbc14cd9b98f823b8f3b0aa9116839d5181264a8b597c8f3f47b835e595",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    stamp = datetime.fromisoformat(value)
    require(stamp.utcoffset() is not None, "UTC timestamp lacks timezone")
    return stamp.timestamp()


def binding(path):
    path = Path(path).resolve()
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            f"Artifact changed during hashing: {path}")
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=after.st_size, mtime_ns=after.st_mtime_ns)


def load(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return binding(path)["sha256"]


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def sha_arg(value):
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise argparse.ArgumentTypeError("Expected lowercase SHA256")
    return value


def environment():
    require(Path(sys.prefix).resolve() == ENV.resolve() and "anaconda3" not in sys.executable,
            "Use OFFPY data_gate/.venv only, not AP")
    require(sys.platform == "darwin" and sys.byteorder == "little", "Fixed macOS little-endian scope")
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "nice >=10 required")
    require({key: os.environ.get(key) for key in LAUNCH_ENV} == LAUNCH_ENV,
            "Explicit BLAS1/no-bytecode/4GiB-array launch controls required")


class Audit:
    def __init__(self):
        self.files = {}

    def bind(self, name, path, expected=None, sha=None):
        item = binding(path)
        require(name not in self.files or self.files[name] == item, f"Rebound artifact: {name}")
        self.files[name] = item
        require(expected is None or item == expected, f"Artifact binding differs: {name}")
        require(sha is None or item["sha256"] == sha, f"Artifact SHA differs: {name}")
        return item

    def recorded(self, prefix, items):
        for name, item in items.items():
            self.bind(prefix + name, item["path"], expected=item)

    def recheck(self):
        for name, expected in self.files.items():
            require(binding(expected["path"]) == expected, f"Artifact changed: {name}")


def verify_inputs(audit, pins):
    for name, expected in PINS.items():
        audit.bind(name, NEW / name, sha=expected)
    audit.bind("remote_checker", __file__)
    audit.bind("panel_checker", HERE / "check_panel.py", sha=pins["panel_checker_sha256"])
    build = load(HERE / "build.json")
    gate = load(HERE / "synthetic/result.json")
    audit.bind("synthetic_completion", HERE / "synthetic/completion.json")
    complete = load(HERE / "synthetic/completion.json")
    require(build["passed"] and build["executed"] is False and build["inverse_source_byte_exact"]
            and build["all_other_definitions_byte_exact"] and build["imports_and_learned_payloads_unchanged"]
            and gate["passed"] and gate["sources_unchanged"] and complete["passed"], "Unqualified build/synthetic")
    require(build["candidate_sha256"] == CANDIDATE_SHA and build["parent_sha256"] == gate["parent_sha256"] == PARENT_SHA
            and build["source_sha256"] == gate["source_sha256"] == PINS["terminal_dead_consumers_v1/source.py"]
            and build["protocol_sha256"] == gate["protocol_sha256"] == PINS["terminal_dead_consumers_v1/PROTOCOL.md"]
            and build["synthetic_result_sha256"] == complete["result_sha256"] == PINS["terminal_dead_consumers_v1/synthetic/result.json"]
            and gate["checker_sha256"] == PINS["terminal_dead_consumers_v1/check_synthetic.py"]
            and build["terminal_function_sha256"] == gate["terminal_function_sha256"], "Build/source ownership differs")

    acq_path, score_path = PANEL / "candidate/result.json", PANEL / "score/result.json"
    audit.bind("panel_acquisition", acq_path, sha=pins["panel_acquire_sha256"])
    audit.bind("panel_score", score_path, sha=pins["panel_score_sha256"])
    acq, score = load(acq_path), load(score_path)
    for role in ("candidate", "score"):
        audit.bind("panel_" + role + "_completion", PANEL / role / "completion.json")
        seal = load(PANEL / role / "completion.json")
        require(seal["schema"] == PANEL_SCHEMA and seal["qualification_passed"]
                and seal["result_sha256"] == audit.files["panel_acquisition" if role == "candidate" else "panel_score"]["sha256"],
                "Panel completion seal differs")
        if role == "score":
            require(seal["retain_local_component"], "Panel completion did not retain component")
    require(acq["schema"] == score["schema"] == PANEL_SCHEMA and acq["role"] == "candidate"
            and all(record["completed"] and record["qualification_passed"] and record["hashes_unchanged"] for record in (acq, score))
            and acq["candidate_sha256"] == score["candidate_sha256"] == CANDIDATE_SHA
            and acq["parent_sha256"] == score["parent_sha256"] == PARENT_SHA
            and not acq["truth_files_accessed"] and not acq["instrumented"]
            and acq["exact_outputs_and_lower_complete_cost"] and score["exact_all16_bytes_all14"]
            and score["strictly_lower_complete_flops_all14"] and score["baseline_recomputed_exact"]
            and score["retain_local_component"] and score["scientific_status"] == "exact_component_pass"
            and score["group_strict_gain_gate"] == {"public8": True, "virgin6": True}, "No qualified native14 pass")
    require(acq["source_bindings"] == score["source_bindings"] and acq["provenance_pins"] == score["provenance_pins"]
            and acq["provenance_pins"] == dict(build_sha256=PINS["terminal_dead_consumers_v1/build.json"],
                synthetic_sha256=PINS["terminal_dead_consumers_v1/synthetic/result.json"],
                protocol_sha256=PINS["terminal_dead_consumers_v1/PROTOCOL.md"]), "Panel provenance differs")
    require(all(positive(score["adjusted_candidate_over_parent"][group])
                and score["adjusted_candidate_over_parent"][group] < 1
                for group in ("public8", "virgin6", "combined14")), "Recorded panel group gains differ")
    audit.recorded("panel_source/", acq["source_bindings"])
    for key, path in (("candidate", HERE / "estimator.py"), ("builder", HERE / "source.py"),
                      ("panel_checker", HERE / "check_panel.py"), ("protocol", HERE / "PROTOCOL.md")):
        require(acq["source_bindings"][key] == binding(path), "Panel source path/bytes differ")
    audit.recorded("panel_fixture/", acq["fixture_bindings"])
    expected_artifacts = {f"call_{i:02d}_rank{rank}_output.npy" for i, rank in enumerate(PANEL_RANKS)}
    expected_artifacts.update(f"call_{i:02d}_bill.json" for i in range(14))
    require(set(acq["artifacts"]) == expected_artifacts, "Incomplete native14 artifacts")
    for name, item in acq["artifacts"].items():
        require(Path(item["path"]).resolve() == (PANEL / "candidate" / name).resolve(), "Native artifact escaped panel")
        audit.bind("panel_artifact/" + name, item["path"], expected=item)
    audit.recorded("score_role_seal/", score["role_seals"])
    start_path = PANEL / "candidate/started.json"
    audit.bind("panel_start", start_path, sha=acq["started_sha256"])
    require(load(start_path) == acq["started"] and acq["started"]["sequence"] == list(PANEL_RANKS)
            and acq["started"]["initialization_once_in_call0"] and acq["started"]["one_estimator_instance"]
            and acq["started"]["no_warmup"] and acq["started"]["no_reset"]
            and acq["blas_env"] == acq["started"]["blas_env"] ==
                {key: "1" for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}, "Native start differs")
    seal_path = PANEL / "candidate/predictions_and_bills_sealed.json"
    audit.bind("panel_predictions_seal", seal_path, sha=acq["predictions_and_bills_seal_sha256"])
    sealed = load(seal_path)
    for key in ("trials", "artifacts", "comparisons", "source_bindings", "fixture_bindings", "provenance_pins", "candidate_sha256"):
        require(sealed[key] == acq[key], "Native prediction seal differs: " + key)
    require(sealed["truth_files_accessed"] is False, "Prediction seal includes truth access")
    before_path = PANEL / "score/both_roles_verified_before_truth.json"
    audit.bind("before_truth_seal", before_path, sha=score["both_roles_verified_before_truth_sha256"])
    before = load(before_path)
    audit.bind("truth_access_marker", PANEL / "score/truth_access_started.json")
    marker = load(PANEL / "score/truth_access_started.json")
    require(before["role_seals"] == score["role_seals"] and before["source_bindings"] == acq["source_bindings"]
            and before["truth_files_accessed"] is False and before["ranks"] == list(PANEL_RANKS)
            and marker["prior_seal_sha256"] == score["both_roles_verified_before_truth_sha256"]
            and marker["utc"] == score["truth_access_started_utc"]
            and timestamp(sealed["sealed_utc"]) <= timestamp(before["utc"]) <= timestamp(marker["utc"]),
            "Outputs/bills were not sealed before the recorded truth boundary")
    # Read only truth-boundary metadata. Do not follow score['truth_bindings'].
    parent = load(NEW / "pair_polynomial_f32_panel_v1/candidate/result.json")
    require(acq["baseline_result_sha256"] == PINS["pair_polynomial_f32_panel_v1/candidate/result.json"]
            and parent["candidate_sha256"] == PARENT_SHA
            and len(acq["trials"]) == len(acq["comparisons"]) == len(parent["trials"]) == 14,
            "Different archived native parent")
    for i, rank in enumerate(PANEL_RANKS):
        trial, comparison, old = acq["trials"][i], acq["comparisons"][i], parent["trials"][i]
        require(trial["call"] == comparison["call"] == old["call"] == i
                and trial["rank"] == comparison["rank"] == old["rank"] == rank
                and trial["output_artifact"] == f"call_{i:02d}_rank{rank}_output.npy"
                and comparison["all16_byte_exact"] and comparison["per_layer_byte_exact"] == [True] * 16
                and comparison["strict_complete_cost_saving"]
                and trial["output_bytes_sha256"] == old["output_bytes_sha256"]
                and type(trial["flops"]) is int and 0 < trial["flops"] < old["flops"] <= 2**41
                and comparison["parent_flops"] == old["flops"]
                and comparison["saved_flops"] == old["flops"] - trial["flops"], "Native exactness/cost gate differs")
        require(load(PANEL / "candidate" / f"call_{i:02d}_bill.json") == trial, "Native bill differs")
        require(score["per_role"]["candidate"][i]["output_bytes_sha256"] == trial["output_bytes_sha256"]
                and score["per_role"]["candidate"][i]["flops"] == trial["flops"], "Scored prediction differs")

    old_qualification = load(NEW / "pair_polynomial_f32_remote_qualified_v1.json")
    old_remote = load(NEW / (OLD_TAG + ".json"))
    require(old_qualification["pass_gate"] and old_qualification["hashes_unchanged"]
            and old_qualification["source_sha256"] == old_remote["source_sha256"] == PARENT_SHA
            and old_qualification["tag"] == OLD_TAG and old_remote["status"] == "completed"
            and old_remote["repeated_output_byte_identical"] and len(old_remote["records"]) == 5,
            "Archived remote parent is unqualified")
    for name in (OLD_TAG + ".json", OLD_TAG + "_resources.json", OLD_TAG + "_server.log"):
        audit.bind("old_remote/" + name, NEW / name, expected=old_qualification["input_bindings"][name])
    for i, rank in enumerate(RANKS):
        row = old_remote["records"][i]
        require(row["index"] == i and row["rank"] == rank, "Wrong archived remote order")
        name = f"{OLD_TAG}_{i}_r{rank}.f32"
        item = audit.bind("old_remote/" + name, NEW / name, expected=old_qualification["input_bindings"][name])
        require(item["sha256"] == row["output_sha256"] and item["bytes"] == 65536, "Archived remote output differs")
        fixture = acq["fixture_bindings"][f"weights_{rank}"]
        require(fixture == old_qualification["input_bindings"][f"weights_{rank}"], "Remote/native weight fixtures differ")
    audit.recheck()
    return acq, score, old_remote


def command():
    return [str(ENV / "bin/python"), "-u", str(NEW / "run_remote_series_v1.py"),
            "--candidate", CANDIDATE, "--tag", TAG]


def prepare(audit, pins):
    require(not any((NEW / (TAG + suffix)).exists() for suffix in (".json", "_resources.json", "_server.log"))
            and not any((NEW / f"{TAG}_{i}_r{r}.f32").exists() for i, r in enumerate(RANKS)),
            "Refusing existing remote artifacts")
    acq, _, _ = verify_inputs(audit, pins)
    record = dict(schema=SCHEMA, utc=utc(), candidate=CANDIDATE, candidate_sha256=CANDIDATE_SHA, tag=TAG,
                  planned_sequence=list(RANKS), panel_gate_passed=True, provenance_pins=pins,
                  input_bindings=audit.files, fixture_bindings={f"weights_{r}": acq["fixture_bindings"][f"weights_{r}"]
                                                              for r in sorted(set(RANKS))},
                  env=LAUNCH_ENV, nice=os.getpriority(os.PRIO_PROCESS, 0), pid=os.getpid(),
                  cwd=str(NEW), command=command(), remote_launched=False, estimator_executed=False,
                  truth_arrays_accessed=False, raw_or_adjusted_scored=False,
                  qualification="Explicit prelaunch source/fixture/environment record; the command is NOT executed here. Child environment is not independently observed.")
    audit.recheck()
    digest = write(ROOT / "launch.json", record)
    write(ROOT / "prepare_completion.json", dict(schema=SCHEMA, passed=True, utc=utc(), launch_sha256=digest))
    print(json.dumps(dict(prepared=True, launch_sha256=digest, command=record["command"], cwd=str(NEW),
                          env=LAUNCH_ENV, remote_launched=False), sort_keys=True))


def qualify(audit, launch_sha):
    launch_binding = audit.bind("launch", ROOT / "launch.json", sha=launch_sha)
    launch = load(ROOT / "launch.json")
    audit.bind("prepare_completion", ROOT / "prepare_completion.json")
    prepared = load(ROOT / "prepare_completion.json")
    require(prepared["passed"] and prepared["launch_sha256"] == launch_sha
            and launch["schema"] == SCHEMA and launch["candidate"] == CANDIDATE
            and launch["candidate_sha256"] == CANDIDATE_SHA and launch["tag"] == TAG
            and launch["planned_sequence"] == list(RANKS) and launch["panel_gate_passed"]
            and launch["env"] == LAUNCH_ENV and launch["nice"] >= 10
            and launch["command"] == command() and launch["cwd"] == str(NEW)
            and launch["remote_launched"] is False and launch["estimator_executed"] is False,
            "Different prelaunch contract")
    current = Audit()
    acq, _, old_remote = verify_inputs(current, launch["provenance_pins"])
    require(current.files == launch["input_bindings"], "Inputs changed after prelaunch")
    audit.recorded("prelaunch/", current.files)
    require(launch["fixture_bindings"] == {f"weights_{r}": acq["fixture_bindings"][f"weights_{r}"] for r in sorted(set(RANKS))},
            "Prelaunch fixture list differs")
    require(timestamp(launch["utc"]) <= launch_binding["mtime_ns"] / 1e9
            and all(item["mtime_ns"] <= launch_binding["mtime_ns"] for item in current.files.values()),
            "Prelaunch record predates a required input")
    for suffix in (".json", "_resources.json", "_server.log"):
        audit.bind("new_remote" + suffix, NEW / (TAG + suffix))
    series, resources = load(NEW / (TAG + ".json")), load(NEW / (TAG + "_resources.json"))
    require(series["status"] == "completed" and series["candidate"] == CANDIDATE
            and series["source_sha256"] == CANDIDATE_SHA and series["repeated_output_byte_identical"]
            and len(series["records"]) == 5 and resources["candidate"] == CANDIDATE, "Remote series did not complete")
    import numpy as np
    require(importlib.metadata.version("numpy") == acq["versions"]["numpy"], "Native comparison NumPy differs")

    def valid_array(array):
        require(array.shape == (16, 1024) and array.dtype == np.dtype("float32")
                and bool(np.isfinite(array).all()) and bool((array >= 0).all()), "Invalid served float32 output")

    checks, repeat = [], None
    for i, rank in enumerate(RANKS):
        row, old = series["records"][i], old_remote["records"][i]
        require(row["index"] == i and row["rank"] == rank and row["output_shape"] == [16, 1024]
                and row["output_dtype"] == "float32", "Wrong remote output/sequence")
        path = NEW / f"{TAG}_{i}_r{rank}.f32"
        item = audit.bind(f"new_remote_output/{i}", path, sha=row["output_sha256"])
        require(item["bytes"] == 65536 and item["mtime_ns"] >= launch_binding["mtime_ns"], "Output predates launch/has wrong size")
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == item["sha256"], "Remote output changed while reading")
        old_raw = (NEW / f"{OLD_TAG}_{i}_r{rank}.f32").read_bytes()
        require(raw == old_raw and row["output_sha256"] == old["output_sha256"], "Same-backend parent bytes differ")
        array = np.frombuffer(raw, dtype=np.float32).reshape(16, 1024)
        valid_array(array)
        flops, timing = row["flops"], row["timing"]
        require(type(flops) is int and 0 < flops < old["flops"] <= 2**41
                and row["util"] == flops / 2**41, "Strict complete remote cost saving failed")
        wall, residual = timing["wall_time_s"], timing["residual_wall_time_s"]
        require(positive(wall) and wall < 120 and isinstance(residual, (int, float))
                and not isinstance(residual, bool) and math.isfinite(residual) and 0 <= residual < .4,
                "Remote wall/residual gate failed")
        call = PANEL_RANKS.index(rank)
        trial = acq["trials"][call]
        native = np.load(PANEL / "candidate" / f"call_{call:02d}_rank{rank}_output.npy", allow_pickle=False)
        valid_array(native)
        require(hashlib.sha256(native.tobytes(order="C")).hexdigest() == trial["output_bytes_sha256"], "Native array bytes differ")
        difference = array.astype(np.float64) - native.astype(np.float64)
        error = float(np.max(np.abs(difference)))
        require(error <= 5e-6, "Remote/native14 maximum absolute drift exceeds5e-6")
        if i == 0:
            repeat = raw
        if i == 4:
            require(raw == repeat, "Repeated remote rank35 bytes differ")
        checks.append(dict(index=i, rank=rank, output_sha256=item["sha256"], parent_byte_exact=True,
                           flops=flops, parent_flops=old["flops"], saved_flops=old["flops"] - flops,
                           wall_s=wall, residual_s=residual, native14_call=call,
                           native14_max_abs=error, native14_rms=float(np.sqrt(np.mean(difference * difference))),
                           finite=True, nonnegative=True))
    samples = resources["samples"]
    require(isinstance(samples, list) and samples and all(isinstance(sample, list) and len(sample) == 2
            and positive(sample[0]) and type(sample[1]) is int and sample[1] >= 0 for sample in samples), "Invalid RSS samples")
    peak = resources["peak_combined_rss_bytes"]
    require(type(peak) is int and peak == max(sample[1] for sample in samples) and 0 < peak < 8_000_000_000,
            "Combined sampled RSS limit exceeded")
    require(positive(resources["wall_s"]) and type(resources["max_single_child_rss_bytes"]) is int
            and resources["max_single_child_rss_bytes"] >= 0, "Invalid resource metadata")
    audit.recheck()
    result = dict(schema=SCHEMA, passed=True, completed_utc=utc(), candidate=CANDIDATE,
                  candidate_sha256=CANDIDATE_SHA, parent_sha256=PARENT_SHA, tag=TAG, checks=checks,
                  input_bindings=audit.files, launch_sha256=launch_sha, hashes_unchanged=True,
                  explicit_launch_env=launch["env"], prior_native14_pass_verified=True,
                  parent_all5_byte_exact=True, strict_complete_cost_saving_all5=True,
                  native14_comparison_limit=5e-6, repeated_rank35_byte_exact=True,
                  peak_combined_rss_bytes=peak, sampled_rss_observations=len(samples),
                  max_single_child_rss_bytes=resources["max_single_child_rss_bytes"],
                  series_wall_s=resources["wall_s"], executable=sys.executable, numpy_version=np.__version__,
                  estimator_executed_by_checker=False, truth_arrays_accessed=False, raw_or_adjusted_scored=False,
                  native14_accuracy_recomputed=False, official_grade=False, milestone_achieved=False,
                  child_environment_independently_observed=False, archived_runtime_not_contemporaneous=True,
                  qualification="Five-context persistent NumPy-free participant on the configured macOS OpenBLAS server. Exact archived parent bytes, lower same-backend complete bills, tolerance against five matching outputs from the native14 panel, and local sampled RSS/wall/residual gates only; no official grade, contemporaneous speedup, Linux cgroup guarantee, packaging or submission claim.")
    digest = write(ROOT / "result.json", result)
    write(ROOT / "completion.json", dict(schema=SCHEMA, passed=True, utc=utc(), result_sha256=digest))
    print(json.dumps(dict(passed=True, result_sha256=digest, checks=checks, peak_combined_rss_bytes=peak), sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--qualify", action="store_true")
    for name in ("panel-acquire", "panel-score", "panel-checker", "launch"):
        parser.add_argument("--" + name + "-sha256", type=sha_arg)
    args = parser.parse_args()
    if not args.prepare and not args.qualify:
        parser.print_help()
        return 0
    if args.prepare and (args.launch_sha256 is not None or any(getattr(args, key) is None for key in
                        ("panel_acquire_sha256", "panel_score_sha256", "panel_checker_sha256"))):
        parser.error("--prepare needs all three panel SHA pins, and no launch SHA")
    if args.qualify and (args.launch_sha256 is None or any(getattr(args, key) is not None for key in
                        ("panel_acquire_sha256", "panel_score_sha256", "panel_checker_sha256"))):
        parser.error("--qualify needs only --launch-sha256")
    if args.prepare:
        ROOT.mkdir(exist_ok=False)
    require(ROOT.is_dir() and not ROOT.is_symlink(), "An exclusive prepared remote directory is required")
    stage = "prepare" if args.prepare else "qualify"
    require(not any((ROOT / name).exists() for name in (stage + "_started.json", stage + "_failure.json"))
            and (args.prepare or not any((ROOT / name).exists() for name in ("result.json", "completion.json"))),
            "Refusing an existing stage/result/failure")
    audit = Audit()
    started = time.monotonic()
    write(ROOT / (stage + "_started.json"), dict(schema=SCHEMA, utc=utc(), arguments=vars(args),
          source_sha256=binding(__file__)["sha256"], estimator_executed=False, truth_arrays_accessed=False))
    try:
        environment()
        if args.prepare:
            pins = {key: getattr(args, key) for key in ("panel_acquire_sha256", "panel_score_sha256", "panel_checker_sha256")}
            prepare(audit, pins)
        else:
            qualify(audit, args.launch_sha256)
        return 0
    except BaseException:
        write(ROOT / (stage + "_failure.json"), dict(schema=SCHEMA, passed=False, utc=utc(),
              elapsed_s=time.monotonic() - started, error=traceback.format_exc(), input_bindings=audit.files,
              estimator_executed=False, truth_arrays_accessed=False))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
