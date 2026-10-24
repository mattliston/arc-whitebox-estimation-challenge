"""Inert exact-cleanup remote preparation and sealed-output qualification.

--prepare authenticates the completed native14+repeat and writes a launch
command without running it. --qualify reads the five completed remote
outputs; neither stage imports an estimator, runs a network, or opens truth
arrays. Parent remote raw scores are decoded only after exact-output proof
is sealed. Default invocation only prints help.
"""
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
NEW = HERE.parent
ROOT = HERE / "remote"
NATIVE = HERE / "native"
PARENT_NATIVE = NEW / "resident_gram512_v1/panel_v2/candidate"
PARENT_REMOTE = NEW / "resident_gram512_v1/remote"
CANDIDATE = "unused_materializations_v1/estimator.py"
CANDIDATE_SHA = "409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f"
PARENT_SHA = "0c21db7a74ae0a271d52fddb5b925318be61100e0676393c87f2f15ce9a997dc"
NATIVE_CHECKER_SHA = "3cfab9e100d0fa3ebd82222d0d67ca2197657caf1de1028f7d40df9fa2f654ac"
PARENT_REMOTE_SHA = "3cbe4d5354b84e95209952a95248955c04f7e7862c2f908919083ffdc3d61be4"
PARENT_NATIVE_SHA = "095e8ca059f71cde099781aef1aa5d9e38ce9f8ae70d4b58eadc7b93312a7bff"
PARENT_NATIVE_SCORE_SHA = "e3dc33f0018ffb6d2e1b9855112bb4c7ae14e9805253694d147493a2bcf724cf"
TAG = "unused_materializations_openblas_series_v1"
OLD_TAG = "resident_gram512_openblas_series_v1"
SCHEMA = "unused_materializations_remote_v1"
NATIVE_SCHEMA = "unused_materializations_exact14_v1"
RANKS = (35, 100, 500, 3016500, 35)
PANEL_RANKS = (35, 100, 150, 200, 250, 300, 400, 500,
               3016500, 3016501, 3016502, 3016503, 3016504, 3016505)
INITIALIZATION = 1383913666
BUDGET = 2**41
VERSIONS = dict(flopscope="0.12.0", whestbench="0.16.0", numpy="2.4.6", threadpoolctl="3.6.0")
UTILITY = NEW / "terminal_dead_consumers_v1/check_remote_v2.py"
UTILITY_SHA = "76f6510e724e03505d8060fa159dc0df40ad8f9ea75a740d171b5c3d24c16fa4"
PINS = {
    CANDIDATE: CANDIDATE_SHA,
    "resident_gram512_v1/estimator.py": PARENT_SHA,
    "unused_materializations_v1/build.py": "21c90b150b43c56617b4ff85fc93b0772fd374a484a65592327d7b566ca51530",
    "unused_materializations_v1/build.json": "5e324e0226cfad657fec3861e3266070525bcd3614b21ec0e11f18afcf81553e",
    "unused_materializations_v1/PROTOCOL.md": "e3cc5f1696bad8196601e23c2da6e0d08a5db77be7cc7950b6732621eb617805",
    "resident_gram512_v1/check_remote.py": "41a2f1aa75c269d81a46095701760da0eb6c3b170404233cb03c7018e1aacb7e",
    "resident_gram512_v1/remote/result.json": PARENT_REMOTE_SHA,
    "resident_gram512_v1/remote/launch.json": "83c4cb1011b67c4d14a6b3e46830ea6e364351b34db75bc7daf1a8af9250094e",
    "resident_gram512_v1/panel_v2/candidate/result.json": PARENT_NATIVE_SHA,
    OLD_TAG + ".json": "229b2ff838cdd91c094b6a8a3baf8a0c9c3195571499185657f724f013363c0a",
    OLD_TAG + "_resources.json": "61fc4d899b4357a723aeb0b53f739bd2448e97e3a415133907f38b3e17051f7d",
    OLD_TAG + "_server.log": "ee8e3ed840d2c02a2b6e9167cd3548ba36c7f0f8994e438915cc85b43cf3ae95",
    "run_remote_series_v1.py": "00eabab0b7c1ec0c113b998032d441b60aa882670f364732f02a741282634cb1",
    "check_remote_series_v1.py": "f3b0b9dd9e9edd27bf4f85b5a34ab0d72131e22881dd0531b3201dc29642da55",
}


def utility():
    if hashlib.sha256(UTILITY.read_bytes()).hexdigest() != UTILITY_SHA:
        raise RuntimeError("Frozen inert provenance utility changed")
    spec = importlib.util.spec_from_file_location("unused_remote_utilities", UTILITY)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load inert provenance utility")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def command(r):
    return [str(r.ENV / "bin/python"), "-u", str(NEW / "run_remote_series_v1.py"),
            "--candidate", CANDIDATE, "--tag", TAG]


def nonnegative(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def verify_inputs(r, audit, pins):
    r.require(pins["native_checker_sha256"] == NATIVE_CHECKER_SHA, "Different fixed native checker")
    for name, sha in PINS.items():
        audit.bind(name, NEW / name, sha=sha)
    audit.bind("utility", UTILITY, sha=UTILITY_SHA)
    audit.bind("remote_checker", __file__)
    audit.bind("remote_protocol", HERE / "REMOTE_PROTOCOL.md")
    audit.bind("native_checker", HERE / "check.py", sha=pins["native_checker_sha256"])
    build = r.load(HERE / "build.json")
    r.require(build["passed"] and build["executed"] is False and build["inverse_source_byte_exact"]
              and build["all_other_source_byte_exact"] and build["imports_and_payloads_unchanged"]
              and build["candidate_sha256"] == CANDIDATE_SHA and build["parent_sha256"] == PARENT_SHA
              and build["builder_sha256"] == PINS["unused_materializations_v1/build.py"]
              and build["protocol_sha256"] == PINS["unused_materializations_v1/PROTOCOL.md"], "Build mismatch")
    records = []
    for name, directory in (("native_acquire", NATIVE), ("native_score", HERE / "score")):
        audit.bind(name, directory / "result.json", sha=pins[name + "_sha256"])
        audit.bind(name + "_completion", directory / "completion.json")
        record, seal = r.load(directory / "result.json"), r.load(directory / "completion.json")
        r.require(seal["passed"] and record["passed"] and seal["result_sha256"] == pins[name + "_sha256"]
                  and record["schema"] == NATIVE_SCHEMA and record["candidate_sha256"] == CANDIDATE_SHA
                  and record["parent_sha256"] == PARENT_SHA and not record["truth_files_accessed"], "Native seal failed")
        records.append(record)
    acq, score = records
    r.require(acq["hashes_unchanged"] and acq["versions"] == VERSIONS
              and acq["source_bindings"] == score["source_bindings"]
              and acq["fixture_bindings"] == score["fixture_bindings"]
              and score["acquisition_sha256"] == pins["native_acquire_sha256"]
              and score["parent_score_sha256"] == PARENT_NATIVE_SCORE_SHA
              and score["raw_reused_by_exact_output_identity"], "Native source/raw ownership differs")
    audit.recorded("native_source/", acq["source_bindings"])
    audit.recorded("native_fixture/", acq["fixture_bindings"])
    source_paths = {Path(v["path"]).resolve(): v["sha256"] for v in acq["source_bindings"].values()}
    r.require(source_paths[HERE / "check.py"] == NATIVE_CHECKER_SHA
              and source_paths[HERE / "estimator.py"] == CANDIDATE_SHA, "Native source paths differ")
    audit.bind("native_start", NATIVE / "started.json", sha=acq["started_sha256"])
    started = r.load(NATIVE / "started.json")
    r.require(started == acq["started"] and started["sequence"] == list(PANEL_RANKS) + [35]
              and started["source_bindings"] == acq["source_bindings"]
              and started["fixture_bindings"] == acq["fixture_bindings"]
              and started["versions"] == VERSIONS and started["nice"] >= 10
              and started["initialization_once"] and started["no_warmup"] and not started["instrumented"]
              and started["blas_env"] == {k: "1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
              "Native process boundary differs")
    audit.bind("native_output_seal", NATIVE / "predictions_and_bills_sealed.json", sha=acq["outputs_seal_sha256"])
    sealed = r.load(NATIVE / "predictions_and_bills_sealed.json")
    for key in ("candidate_sha256", "source_bindings", "fixture_bindings", "artifacts", "trials"):
        r.require(sealed[key] == acq[key], "Native output/bill seal differs")
    audit.bind("native_before_score", HERE / "score/output_identity_verified_before_score_receipt.json")
    before = r.load(HERE / "score/output_identity_verified_before_score_receipt.json")
    r.require(not sealed["truth_files_accessed"] and not before["truth_files_accessed"]
              and before["acquisition_sha256"] == pins["native_acquire_sha256"]
              and r.timestamp(sealed["utc"]) <= r.timestamp(before["utc"]) <= r.timestamp(score["completed_utc"]),
              "Native raw-reuse chronology differs")
    parent = r.load(PARENT_NATIVE / "result.json")
    audit.bind("parent_native_completion", PARENT_NATIVE / "completion.json")
    pc = r.load(PARENT_NATIVE / "completion.json")
    r.require(parent["qualification_passed"] and parent["hashes_unchanged"] and pc["qualification_passed"]
              and pc["result_sha256"] == PARENT_NATIVE_SHA and parent["candidate_sha256"] == PARENT_SHA
              and parent["versions"] == VERSIONS and parent["fixture_bindings"] == acq["fixture_bindings"], "Wrong native parent")
    audit.recorded("parent_native_artifact/", parent["artifacts"])
    sequence = PANEL_RANKS + (35,)
    expected = {f"call_{i:02d}_rank{rank}_output.npy" for i, rank in enumerate(sequence)}
    expected.update(f"call_{i:02d}_bill.json" for i in range(15))
    r.require(set(acq["artifacts"]) == expected and [t["rank"] for t in acq["trials"]] == list(sequence)
              and [p["rank"] for p in score["per_net"]] == list(PANEL_RANKS), "Wrong native artifacts/sequence")
    for name, item in acq["artifacts"].items():
        r.require(Path(item["path"]).resolve() == (NATIVE / name).resolve(), "Native artifact escaped directory")
        audit.bind("native_artifact/" + name, item["path"], expected=item)
    for i, trial in enumerate(acq["trials"]):
        prior = parent["trials"][i if i < 14 else 0]
        parent_flops = prior["flops"] - (INITIALIZATION if i == 14 else 0)
        r.require(trial["call"] == i and trial["includes_initialization"] == (i == 0)
                  and trial["all16_byte_exact"] and trial["output_bytes_sha256"] == prior["output_bytes_sha256"]
                  and type(trial["flops"]) is int and 0 < trial["flops"] < parent_flops <= BUDGET
                  and trial["parent_flops"] == parent_flops and trial["saved_flops"] == parent_flops - trial["flops"]
                  and trial["output_artifact"] == f"call_{i:02d}_rank{sequence[i]}_output.npy"
                  and r.load(NATIVE / f"call_{i:02d}_bill.json") == trial, "Native bytes/full bill differ")
        r.require(r.positive(trial["context_wall_s"]) and trial["context_wall_s"] < 120
                  and r.positive(trial["meter_wall_s"]) and trial["meter_wall_s"] < 120
                  and type(trial["rss_bytes"]) is int and 0 < trial["rss_bytes"] < 8_000_000_000
                  and all(nonnegative(trial[k]) for k in ("backend_s", "overhead_s", "native_residual_s")), "Native resources differ")
        if i < 14:
            p = score["per_net"][i]
            r.require(p["flops"] == trial["flops"] and p["saved_flops"] == trial["saved_flops"]
                      and p["multiplier"] == max(.1, trial["flops"] / BUDGET)
                      and p["adjusted_noise_corrected"] == p["raw_noise_corrected"] * p["multiplier"], "Native scored bill differs")
    r.require(acq["trials"][0]["flops"] - acq["trials"][-1]["flops"] == INITIALIZATION
              and acq["trials"][0]["output_bytes_sha256"] == acq["trials"][-1]["output_bytes_sha256"], "Native repeat changed")
    for group, count in (("public8", 8), ("virgin6", 6), ("combined14", 14)):
        s = score["summaries"][group]
        r.require(s["n"] == count and r.positive(s["adjusted_ratio"]) and s["adjusted_ratio"] < 1, "Native group gain failed")
    # Keep this scored parent result opaque until the new remote proof is sealed.
    audit.bind("parent_remote_completion", PARENT_REMOTE / "completion.json")
    pc = r.load(PARENT_REMOTE / "completion.json")
    r.require(pc["passed"] and pc["result_sha256"] == PARENT_REMOTE_SHA, "Unqualified remote parent seal")
    parent_launch = r.load(PARENT_REMOTE / "launch.json")
    r.require(parent_launch["candidate_sha256"] == PARENT_SHA and parent_launch["tag"] == OLD_TAG
              and parent_launch["sequence"] == list(RANKS)
              and parent_launch["fixture_bindings"] == acq["fixture_bindings"], "Remote/native fixture ownership differs")
    audit.recorded("parent_launch_input/", parent_launch["input_bindings"])
    old_series = r.load(NEW / (OLD_TAG + ".json"))
    r.require(old_series["status"] == "completed" and old_series["source_sha256"] == PARENT_SHA
              and old_series["candidate"] == "resident_gram512_v1/estimator.py"
              and old_series["repeated_output_byte_identical"] and len(old_series["records"]) == 5, "Wrong parent series")
    for i, rank in enumerate(RANKS):
        old = old_series["records"][i]
        item = audit.bind(f"parent_output/{i}", NEW / f"{OLD_TAG}_{i}_r{rank}.f32", sha=old["output_sha256"])
        r.require(old["index"] == i and old["rank"] == rank and item["bytes"] == 65536
                  and old["output_shape"] == [16, 1024] and old["output_dtype"] == "float32"
                  and type(old["flops"]) is int and 0 < old["flops"] <= BUDGET, "Parent output/full bill differs")
    r.require(old_series["records"][0]["flops"] - old_series["records"][4]["flops"] == INITIALIZATION, "Parent initialization differs")
    audit.recheck()
    return acq, score, old_series


def prepare(r, audit, pins):
    paths = [NEW / (TAG + suffix) for suffix in (".json", "_resources.json", "_server.log")]
    paths += [NEW / f"{TAG}_{i}_r{rank}.f32" for i, rank in enumerate(RANKS)]
    r.require(not any(p.exists() or p.is_symlink() for p in paths), "Refusing existing remote artifacts")
    acq, _, _ = verify_inputs(r, audit, pins)
    record = dict(schema=SCHEMA, utc=r.utc(), candidate=CANDIDATE, candidate_sha256=CANDIDATE_SHA,
                  parent_sha256=PARENT_SHA, tag=TAG, sequence=list(RANKS), provenance_pins=pins,
                  input_bindings=audit.files, fixture_bindings=acq["fixture_bindings"], env=r.LAUNCH_ENV,
                  command=command(r), cwd=str(NEW), nice=os.getpriority(os.PRIO_PROCESS, 0),
                  remote_launched=False, estimator_executed=False, truth_arrays_accessed=False,
                  parent_remote_raw_decoded=False, parent_all5_byte_identity_required=True)
    audit.recheck()
    sha = r.write(ROOT / "launch.json", record)
    r.write(ROOT / "prepare_completion.json", dict(passed=True, launch_sha256=sha, utc=r.utc()))
    print(json.dumps(dict(prepared=True, launch_sha256=sha, command=record["command"], env=r.LAUNCH_ENV,
                          cwd=str(NEW), required_nice_min=10, remote_launched=False)), flush=True)


def qualify(r, audit, launch_sha):
    launch_item = audit.bind("launch", ROOT / "launch.json", sha=launch_sha)
    launch = r.load(ROOT / "launch.json")
    audit.bind("prepare_completion", ROOT / "prepare_completion.json")
    pc = r.load(ROOT / "prepare_completion.json")
    r.require(pc["passed"] and pc["launch_sha256"] == launch_sha and launch["schema"] == SCHEMA
              and launch["candidate"] == CANDIDATE and launch["candidate_sha256"] == CANDIDATE_SHA
              and launch["parent_sha256"] == PARENT_SHA and launch["tag"] == TAG
              and launch["sequence"] == list(RANKS) and launch["env"] == r.LAUNCH_ENV
              and launch["command"] == command(r) and launch["cwd"] == str(NEW) and launch["nice"] >= 10
              and launch["parent_all5_byte_identity_required"] and not launch["remote_launched"]
              and not launch["estimator_executed"] and not launch["truth_arrays_accessed"]
              and not launch["parent_remote_raw_decoded"], "Wrong prepared contract")
    current = r.Audit()
    acq, _, old_series = verify_inputs(r, current, launch["provenance_pins"])
    r.require(current.files == launch["input_bindings"] and acq["fixture_bindings"] == launch["fixture_bindings"]
              and all(v["mtime_ns"] <= launch_item["mtime_ns"] for v in current.files.values()), "Inputs changed after preparation")
    audit.recorded("prelaunch/", current.files)
    for suffix in (".json", "_resources.json", "_server.log"):
        item = audit.bind("new_remote" + suffix, NEW / (TAG + suffix))
        r.require(item["mtime_ns"] >= launch_item["mtime_ns"], "Remote artifact predates launch")
    series, resources = r.load(NEW / (TAG + ".json")), r.load(NEW / (TAG + "_resources.json"))
    r.require(series["status"] == "completed" and series["source_sha256"] == CANDIDATE_SHA
              and series["candidate"] == resources["candidate"] == CANDIDATE
              and series["repeated_output_byte_identical"] and len(series["records"]) == 5, "Incomplete remote series")
    import numpy as np
    r.require(importlib.metadata.version("numpy") == acq["versions"]["numpy"] == "2.4.6", "Comparison NumPy differs")

    def valid(value):
        r.require(value.shape == (16, 1024) and value.dtype == np.dtype("float32")
                  and bool(np.isfinite(value).all()) and bool((value >= 0).all()), "Invalid served values")

    checks, repeat = [], None
    for i, rank in enumerate(RANKS):
        row, old = series["records"][i], old_series["records"][i]
        r.require(row["index"] == i and row["rank"] == rank and row["output_shape"] == [16, 1024]
                  and row["output_dtype"] == "float32", "Remote output/sequence changed")
        item = audit.bind(f"new_output/{i}", NEW / f"{TAG}_{i}_r{rank}.f32", sha=row["output_sha256"])
        r.require(item["bytes"] == 65536 and item["mtime_ns"] >= launch_item["mtime_ns"], "Remote output binding differs")
        raw = Path(item["path"]).read_bytes()
        old_raw = (NEW / f"{OLD_TAG}_{i}_r{rank}.f32").read_bytes()
        r.require(hashlib.sha256(raw).hexdigest() == row["output_sha256"] == old["output_sha256"]
                  and raw == old_raw, "Exact same-backend parent bytes failed")
        value = np.frombuffer(raw, dtype=np.float32).reshape(16, 1024)
        valid(value)
        timing, flops = row["timing"], row["flops"]
        r.require(all(nonnegative(t) for t in timing.values()) and 0 < timing["wall_time_s"] < 60
                  and 0 <= timing["residual_wall_time_s"] < .4 and type(flops) is int
                  and 0 < flops < old["flops"] <= BUDGET and row["util"] == flops / BUDGET, "Remote resource/full-bill gate")
        call = PANEL_RANKS.index(rank)
        trial = acq["trials"][call]
        native = np.load(NATIVE / trial["output_artifact"], allow_pickle=False)
        valid(native)
        r.require(hashlib.sha256(native.tobytes()).hexdigest() == trial["output_bytes_sha256"], "Native bytes changed")
        diff = value.astype(np.float64) - native.astype(np.float64)
        error = float(np.max(np.abs(diff)))
        r.require(error <= 1e-5, "Native/remote candidate drift")
        if i == 0:
            repeat = raw
        if i == 4:
            r.require(raw == repeat, "Remote repeat bytes differ")
        checks.append(dict(index=i, rank=rank, flops=flops, parent_flops=old["flops"],
                           saved_flops=old["flops"] - flops, wall_s=timing["wall_time_s"],
                           residual_s=timing["residual_wall_time_s"], timing=timing,
                           output_sha256=row["output_sha256"], parent_byte_exact=True,
                           parent_per_layer_max_abs=[0.0] * 16, native14_max_abs=error,
                           native14_rms=float(np.sqrt(np.mean(diff * diff))), finite=True, nonnegative=True))
        del value, native, diff, raw, old_raw
    r.require(series["records"][0]["flops"] - series["records"][4]["flops"] == INITIALIZATION, "Remote initialization changed")
    samples, peak = resources["samples"], resources["peak_combined_rss_bytes"]
    r.require(isinstance(samples, list) and samples and all(isinstance(s, list) and len(s) == 2
              and r.positive(s[0]) and type(s[1]) is int and s[1] >= 0 for s in samples), "Invalid RSS samples")
    r.require(type(peak) is int and peak == max(s[1] for s in samples) and 0 < peak < 8_000_000_000
              and r.positive(resources["wall_s"]) and type(resources["max_single_child_rss_bytes"]) is int
              and resources["max_single_child_rss_bytes"] >= 0, "Remote RSS/resource gate")
    audit.recheck()
    proof_sha = r.write(ROOT / "exact_remote_outputs_sealed_before_raw_reuse.json", dict(
        utc=r.utc(), candidate_sha256=CANDIDATE_SHA, parent_sha256=PARENT_SHA,
        input_bindings=audit.files, checks=checks, parent_all5_byte_exact=True,
        truth_arrays_accessed=False, parent_remote_raw_decoded=False))
    audit.bind("exact_output_proof", ROOT / "exact_remote_outputs_sealed_before_raw_reuse.json", sha=proof_sha)
    marker_sha = r.write(ROOT / "raw_reuse_started.json", dict(utc=r.utc(), prior_seal_sha256=proof_sha,
            parent_score_sha256=PARENT_REMOTE_SHA, truth_arrays_accessed=False))
    audit.bind("raw_reuse_marker", ROOT / "raw_reuse_started.json", sha=marker_sha)
    # First decode of the scored parent remote receipt: no truth bindings are followed.
    parent_score = r.load(PARENT_REMOTE / "result.json")
    r.require(parent_score["passed"] and parent_score["hashes_unchanged"]
              and parent_score["runtime_gates_passed"] and parent_score["backend_subset_adjusted_gate_passed"]
              and parent_score["candidate_sha256"] == PARENT_SHA and parent_score["tag"] == OLD_TAG,
              "Qualified remote raw-score parent changed")
    audit.recorded("parent_qualification_input/", parent_score["input_bindings"])
    r.require(len(parent_score["per_role"]["candidate"]) == 4 and len(parent_score["checks"]) == 5, "Parent score sequence")
    per_role = {"parent": [], "candidate": []}
    for i, rank in enumerate(RANKS[:4]):
        p, old = parent_score["per_role"]["candidate"][i], old_series["records"][i]
        r.require(p["rank"] == rank and p["flops"] == old["flops"]
                  and parent_score["checks"][i]["output_sha256"] == old["output_sha256"]
                  and all(nonnegative(p[k]) for k in ("raw_observed", "noise"))
                  and math.isfinite(p["raw_noise_corrected"])
                  and p["raw_noise_corrected"] == p["raw_observed"] - p["noise"], "Parent scored-output ownership differs")
        for role, f in (("parent", old["flops"]), ("candidate", series["records"][i]["flops"])):
            multiplier = max(.1, f / BUDGET)
            per_role[role].append(dict(rank=rank, flops=f, multiplier=multiplier, raw_observed=p["raw_observed"],
                noise=p["noise"], raw_noise_corrected=p["raw_noise_corrected"], adjusted_observed=p["raw_observed"] * multiplier,
                adjusted_noise_corrected=p["raw_noise_corrected"] * multiplier,
                raw_provenance="Exact all16 same-backend output bytes; authenticated parent remote score " + PARENT_REMOTE_SHA))
        r.require(per_role["parent"][-1]["adjusted_noise_corrected"] == p["adjusted_noise_corrected"], "Parent product changed")
    groups = {"public3": (35, 100, 500), "virgin1": (3016500,), "combined4": RANKS[:4]}
    summaries = {role: {group: {"mean_" + key: sum(t[key] for t in rows if t["rank"] in ranks) / len(ranks)
                               for key in ("raw_observed", "raw_noise_corrected", "flops", "adjusted_noise_corrected")}
                        for group, ranks in groups.items()} for role, rows in per_role.items()}
    r.require(all(r.positive(summaries["parent"][g]["mean_adjusted_noise_corrected"]) for g in groups), "Parent score denominator")
    ratios = {g: summaries["candidate"][g]["mean_adjusted_noise_corrected"] /
                 summaries["parent"][g]["mean_adjusted_noise_corrected"] for g in groups}
    r.require(all(r.positive(v) for v in ratios.values()), "Invalid adjusted ratios")
    passed = all(v < 1 for v in ratios.values())
    audit.recheck()
    result = dict(schema=SCHEMA, passed=passed, completed_utc=r.utc(), candidate=CANDIDATE,
        candidate_sha256=CANDIDATE_SHA, parent_sha256=PARENT_SHA, tag=TAG, checks=checks,
        input_bindings=audit.files, hashes_unchanged=True, launch_sha256=launch_sha,
        explicit_launch_env=launch["env"], prior_native14_pass_verified=True, runtime_gates_passed=True,
        parent_all5_byte_exact=True, strict_complete_cost_saving_all5=True, repeated_rank35_byte_exact=True,
        native14_comparison_limit=1e-5, native14_comparison_inclusive=True, peak_combined_rss_bytes=peak,
        sampled_rss_observations=len(samples), max_single_child_rss_bytes=resources["max_single_child_rss_bytes"],
        series_wall_s=resources["wall_s"], both_roles_before_raw_reuse_sha256=proof_sha,
        parent_raw_score_sha256=PARENT_REMOTE_SHA, truth_bindings={}, truth_arrays_accessed=False,
        raw_provenance="Inherited only after sealed all16 byte identity to the qualified 0c21 remote outputs; no truth access.",
        raw_reused_by_exact_output_identity=True, per_role=per_role, summaries=summaries,
        adjusted_candidate_over_parent=ratios, backend_subset_adjusted_gate_passed=passed,
        estimator_executed_by_checker=False, official_score=False, milestone_achieved=False,
        child_environment_independently_observed=False, archived_runtime_not_contemporaneous=True,
        frame="Fixed four-unique remote compatibility subset of independent14; repeat excluded. Same-backend outputs/full bills, inherited same-output raw. Not new holdout, official/Linux guarantee, packaging or submission.")
    sha = r.write(ROOT / "result.json", result)
    r.write(ROOT / "completion.json", dict(passed=passed, result_sha256=sha, utc=r.utc()))
    print(json.dumps(dict(passed=passed, result_sha256=sha, checks=checks, peak_combined_rss_bytes=peak,
                         summaries=summaries, adjusted_candidate_over_parent=ratios)), flush=True)
    return 0 if passed else 1


def main():
    r = utility()
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--qualify", action="store_true")
    for name in ("native-acquire", "native-score", "native-checker", "launch"):
        parser.add_argument("--" + name + "-sha256", type=r.sha_arg)
    args = parser.parse_args()
    if not args.prepare and not args.qualify:
        parser.print_help(); return 0
    keys = ("native_acquire_sha256", "native_score_sha256", "native_checker_sha256")
    r.require((args.prepare and args.launch_sha256 is None and all(getattr(args, k) for k in keys))
              or (args.qualify and args.launch_sha256 and all(getattr(args, k) is None for k in keys)),
              "Prepare needs all native pins; qualify needs only launch pin")
    if args.prepare:
        ROOT.mkdir(exist_ok=False)
    r.require(ROOT.is_dir() and not ROOT.is_symlink(), "Exclusive prepared directory required")
    stage = "prepare" if args.prepare else "qualify"
    r.require(not any((ROOT / name).exists() for name in (stage + "_started.json", stage + "_failure.json"))
              and (args.prepare or not any((ROOT / name).exists() for name in ("result.json", "completion.json"))), "Refusing existing stage")
    audit, began = r.Audit(), time.monotonic()
    r.write(ROOT / (stage + "_started.json"), dict(utc=r.utc(), arguments=vars(args), source=r.binding(__file__),
                                                estimator_executed=False, truth_arrays_accessed=False))
    try:
        r.environment()
        r.require({name: importlib.metadata.version(name) for name in VERSIONS} == VERSIONS, "OFFPY versions differ")
        if args.prepare:
            prepare(r, audit, {k: getattr(args, k) for k in keys}); return 0
        return qualify(r, audit, args.launch_sha256)
    except BaseException:
        r.write(ROOT / (stage + "_failure.json"), dict(passed=False, utc=r.utc(), elapsed_s=time.monotonic()-began,
                error=traceback.format_exc(), input_bindings=audit.files, estimator_executed=False, truth_arrays_accessed=False))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
