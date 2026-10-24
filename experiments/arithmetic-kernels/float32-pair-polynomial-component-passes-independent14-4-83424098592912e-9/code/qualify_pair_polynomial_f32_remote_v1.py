"""Inert qualification of completed remote artifacts; no estimator or truth access."""

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
ENV = HERE.parent / "data_gate/.venv"
TAG = "pair_polynomial_f32_openblas_series_v1"
SOURCE = "candidate_response_pair_polynomial_f32_v1.py"
NATIVE = HERE / "pair_polynomial_f32_fixed_dev_v1/candidate"
PANEL = HERE / "pair_polynomial_f32_panel_v1"
LAUNCH = HERE / "pair_polynomial_f32_remote_launch_v1.json"
STORE = HERE.parent / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
OUT = HERE / "pair_polynomial_f32_remote_qualified_v1.json"
FAIL = HERE / "pair_polynomial_f32_remote_qualification_failure_v1.json"
RANKS = (35, 100, 500, 3016500, 35)
LAUNCH_ENV = dict(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
                  MKL_NUM_THREADS="1", BLIS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                  PYTHONDONTWRITEBYTECODE="1", FLOPSCOPE_MAX_ARRAY_BYTES="4294967296")
PINS = {
    SOURCE: "0b84ab6fa96d32804c9adec350d41766e47d09a71a936b95652893bb1e364a96",
    "run_remote_series_v1.py": "00eabab0b7c1ec0c113b998032d441b60aa882670f364732f02a741282634cb1",
    "check_remote_series_v1.py": "f3b0b9dd9e9edd27bf4f85b5a34ab0d72131e22881dd0531b3201dc29642da55",
    "PAIR_POLYNOMIAL_F32_REMOTE_RELEASE_V1.md": "5365ccb8e056bf5d06753b54edb52a2d67d034d6fde37b5c4cb08bd028d6bd3d",
    "check_pair_polynomial_f32_panel_v1.py": "9f17e45aaa3e968f5eaca236115e6d060bb2dfc70e450478e1282e5d4ed7588f",
    "prepare_pair_polynomial_f32_remote_launch_v1.py": "052098822e56b69e32c3fc2b3a0f6421ab129c164a8c4257e9a43def22a6bf2f",
    "pair_polynomial_f32_fixed_dev_v1/candidate/result.json": "11a66c5a85cabde9c23bca95dba81e0585d0ede999db09ac26c0d538a988a08a",
    "pair_polynomial_f32_panel_v1/candidate/result.json": "72b41dbc14cd9b98f823b8f3b0aa9116839d5181264a8b597c8f3f47b835e595",
    "pair_polynomial_f32_panel_v1/score/result.json": "7fc755056d504aa827eda0a2d6141bc9f64cf2eb3a1c0739fdd6b7bd4f34c5e3",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def binding(path):
    path = Path(path).resolve()
    before = path.stat()
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            f"Artifact changed while hashing: {path}")
    return dict(path=str(path), sha256=value.hexdigest(), bytes=after.st_size,
                mtime_ns=after.st_mtime_ns)


def load(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def finite_nonnegative(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value) and value >= 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualify", action="store_true")
    args = parser.parse_args()
    if not args.qualify:
        print("Inert. --qualify checks only the fixed completed remote artifacts.")
        return 0
    require(not OUT.exists() and not FAIL.exists(), "Refusing an existing qualification or failure")
    started, clock_start = utc(), time.monotonic()
    before, checks = {}, []

    def bind(name, path):
        item = binding(path)
        require(name not in before, f"Duplicate binding: {name}")
        before[name] = item
        return item

    try:
        require(Path(sys.prefix).resolve() == ENV.resolve() and "anaconda3" not in sys.executable,
                "Use OFFPY data_gate/.venv only; never AP/threadpoolctl")
        require(sys.platform == "darwin" and sys.byteorder == "little", "Fixed macOS raw-float32 scope")
        require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "nice >=10 required")
        require(os.environ.get("PYTHONDONTWRITEBYTECODE") == "1", "No bytecode writes")
        for name, expected in PINS.items():
            require(bind(name, HERE/name)["sha256"] == expected, f"Source/receipt pin differs: {name}")
        bind(Path(__file__).name, __file__)
        launch_binding = bind(LAUNCH.name, LAUNCH)
        launch = load(LAUNCH)
        launch_time = datetime.fromisoformat(launch["utc"])
        require(launch_time.utcoffset() is not None
                and launch_time.timestamp() <= launch_binding["mtime_ns"]/1e9,
                "Invalid prelaunch UTC timestamp")
        require(launch["candidate"] == SOURCE and launch["source_sha256"] == PINS[SOURCE]
                and launch["tag"] == TAG and launch["planned_sequence"] == list(RANKS)
                and launch["env"] == LAUNCH_ENV and launch["nice"] >= 10
                and launch["candidate_executed"] is False and launch["remote_launched"] is False
                and launch["panel_gate_passed"] is True, "Invalid explicit prelaunch contract")
        launch_sources = {SOURCE, "check_pair_polynomial_f32_panel_v1.py",
                          "PAIR_POLYNOMIAL_F32_REMOTE_RELEASE_V1.md", "run_remote_series_v1.py",
                          "check_remote_series_v1.py", "prepare_pair_polynomial_f32_remote_launch_v1.py",
                          Path(__file__).name}
        require(set(launch["source_bindings"]) == launch_sources, "Incomplete prelaunch source bindings")
        for name, expected in launch["source_bindings"].items():
            require(bind("launch_source/"+name, HERE/name) == expected
                    and expected["mtime_ns"] <= launch_binding["mtime_ns"],
                    f"Source differs from prelaunch: {name}")

        panel_score = load(PANEL/"score/result.json")
        panel_acquisition = load(PANEL/"candidate/result.json")
        panel_score_binding = before["pair_polynomial_f32_panel_v1/score/result.json"]
        panel_completion_binding = bind("panel_score_completion", PANEL/"score/completion.json")
        panel_completion = load(PANEL/"score/completion.json")
        require(launch["panel_score_sha256"] == panel_score_binding["sha256"]
                and panel_completion["result_sha256"] == panel_score_binding["sha256"]
                and panel_completion["qualification_passed"] is True
                and panel_completion["retain_local_component"] is True
                and panel_score["qualification_passed"] is True and panel_score["hashes_unchanged"] is True
                and panel_score["retain_local_component"] is True
                and panel_score["scientific_status"] == "panel_component_pass"
                and panel_score["candidate"] == SOURCE and panel_score["candidate_sha256"] == PINS[SOURCE]
                and panel_score["source_bindings"] == panel_acquisition["source_bindings"]
                and max(panel_score_binding["mtime_ns"], panel_completion_binding["mtime_ns"])
                    <= launch_binding["mtime_ns"], "No sealed matching panel pass before launch")
        require(panel_score["role_seals"]["candidate/result.json"]
                == before["pair_polynomial_f32_panel_v1/candidate/result.json"], "Panel acquisition seal differs")
        fixtures = panel_acquisition["fixture_bindings"]
        manifest_binding = bind("fixture_manifest", STORE/"manifest.json")
        require(manifest_binding == fixtures["manifest"]
                and manifest_binding["mtime_ns"] <= launch_binding["mtime_ns"], "Panel fixture manifest changed")
        manifest = load(STORE/"manifest.json")["roles"]["observer_fit"]["row_manifests"]
        require(set(launch["fixture_bindings"]) == {f"weights_{r}" for r in set(RANKS)},
                "Incomplete prelaunch four-fixture bindings")
        for rank in sorted(set(RANKS)):
            if rank < 1000:
                entry = manifest[rank] if isinstance(manifest, list) else manifest[str(rank)]
                row = (STORE/entry["path"]).resolve()
                require(row.is_relative_to(STORE.resolve()), "Fixture escaped canonical store")
                path = row.parent/"weights.npy"
            else:
                path = HERE.parent/"public_g19_attribution_v1"/f"fresh_{rank}_weights.npy"
            name = f"weights_{rank}"
            require(bind(name, path) == launch["fixture_bindings"][name] == fixtures[name]
                    and before[name]["mtime_ns"] <= launch_binding["mtime_ns"],
                    f"Fixture differs from prelaunch/panel: {rank}")
        for suffix in (".json", "_resources.json", "_server.log"):
            bind(TAG+suffix, HERE/(TAG+suffix))
        for name in ("started.json", "completion.json"):
            bind("native_"+name, NATIVE/name)

        receipt = load(HERE/(TAG+".json"))
        resources = load(HERE/(TAG+"_resources.json"))
        native = load(NATIVE/"result.json")
        completion = load(NATIVE/"completion.json")
        require(receipt["status"] == "completed" and receipt["candidate"] == SOURCE
                and receipt["source_sha256"] == PINS[SOURCE]
                and receipt["repeated_output_byte_identical"] is True, "Remote series did not complete")
        records = receipt["records"]
        require(len(records) == 5 and [r["index"] for r in records] == list(range(5))
                and tuple(r["rank"] for r in records) == RANKS, "Wrong five-context sequence")
        native_pin = PINS["pair_polynomial_f32_fixed_dev_v1/candidate/result.json"]
        require(completion["qualification_passed"] is True and completion["role"] == "candidate"
                and completion["result_sha256"] == native_pin, "Native completion seal differs")
        require(native["completed"] and native["qualification_passed"] and native["hashes_unchanged"]
                and native["compatibility_passed"] and native["repeat_byte_identical"]
                and native["candidate"] == SOURCE and native["candidate_sha256"] == PINS[SOURCE]
                and not native["truth_files_accessed"] and not native["instrumented"], "Native baseline unqualified")
        require(native["started_sha256"] == before["native_started.json"]["sha256"]
                and native["started"] == load(NATIVE/"started.json"), "Native start seal differs")
        require([t["rank"] for t in native["trials"]] == [35, 100, 35], "Wrong native comparison frame")
        require(all(before[f"weights_{rank}"] == native["fixture_bindings"][f"weights_{rank}"]
                    for rank in (35, 100)), "Remote/native fixtures differ")

        import numpy as np

        require(importlib.metadata.version("numpy") == native["versions"]["numpy"], "Native NumPy version differs")

        def output(array):
            require(array.shape == (16, 1024) and array.dtype == np.dtype("float32")
                    and bool(np.isfinite(array).all()) and bool((array >= 0).all()),
                    "Output must be finite/nonnegative float32(16,1024)")

        remote_arrays = []
        for r in records:
            index, rank = r["index"], r["rank"]
            name = f"{TAG}_{index}_r{rank}.f32"
            artifact = bind(name, HERE/name)
            require(artifact["sha256"] == r["output_sha256"] and artifact["bytes"] == 16*1024*4
                    and r["output_shape"] == [16, 1024] and r["output_dtype"] == "float32", "Remote output identity differs")
            require(launch_binding["mtime_ns"] <= artifact["mtime_ns"], "Launch receipt postdates remote output")
            array = np.fromfile(HERE/name, dtype=np.float32).reshape(16, 1024)
            output(array)
            require(hashlib.sha256(array.tobytes(order="C")).hexdigest() == artifact["sha256"],
                    "Remote bytes changed while reading")
            remote_arrays.append(array)
            timing, flops = r["timing"], r["flops"]
            require(type(flops) is int and 0 < flops <= 2**41, "Remote complete FLOP gate failed")
            require(finite_nonnegative(timing["wall_time_s"]) and 0 < timing["wall_time_s"] < 60
                    and finite_nonnegative(timing["residual_wall_time_s"])
                    and timing["residual_wall_time_s"] < .4, "Remote wall/residual gate failed")
            require(r["util"] == flops/2**41, "Remote utilization field differs")
            check = dict(index=index, rank=rank, finite=True, nonnegative=True, flops=flops,
                         multiplier=max(.1, flops/2**41), wall_s=timing["wall_time_s"],
                         residual_s=timing["residual_wall_time_s"], output_sha256=artifact["sha256"])
            if index < 2:
                trial = native["trials"][index]
                native_name = f"call_{index:02d}_rank{rank}_output.npy"
                require(trial["call"] == index and trial["output_artifact"] == native_name, "Wrong native output selection")
                nb = bind("native_"+native_name, NATIVE/native_name)
                require(nb == native["artifacts"][native_name], "Native output artifact binding differs")
                ref = np.load(NATIVE/native_name, allow_pickle=False)
                output(ref)
                require(hashlib.sha256(ref.tobytes(order="C")).hexdigest() == trial["output_bytes_sha256"],
                        "Native served bytes differ")
                error = array.astype(np.float64)-ref.astype(np.float64)
                check["native_output_max_abs"] = float(np.max(np.abs(error)))
                check["native_output_rms"] = float(np.sqrt(np.mean(error*error)))
                require(check["native_output_max_abs"] < 1e-5, "Native/remote backend parity failed")
            checks.append(check)
        require(remote_arrays[0].tobytes() == remote_arrays[-1].tobytes(), "Repeated remote35 bytes differ")
        require(resources["candidate"] == SOURCE, "Resource receipt belongs to another candidate")
        samples = resources["samples"]
        require(isinstance(samples, list) and len(samples) > 0
                and all(len(s) == 2 and finite_nonnegative(s[0])
                        and type(s[1]) is int and s[1] >= 0 for s in samples), "Invalid RSS samples")
        peak = resources["peak_combined_rss_bytes"]
        require(type(peak) is int and peak == max(s[1] for s in samples)
                and 0 < peak < 8_000_000_000, "Combined sampled RSS gate failed")
        require(finite_nonnegative(resources["wall_s"]) and resources["wall_s"] > 0
                and finite_nonnegative(resources["max_single_child_rss_bytes"]), "Invalid resource metadata")
        for name, expected in before.items():
            require(binding(expected["path"]) == expected, f"Artifact changed during qualification: {name}")
        result = dict(pass_gate=True, candidate=SOURCE, source_sha256=PINS[SOURCE], tag=TAG,
                      qualification_started_utc=started, completed_utc=utc(), input_bindings=before,
                      launch_receipt_sha256=launch_binding["sha256"], launch_record=launch,
                      prelaunch_sources_and_fixtures_unchanged=True, explicit_launch_env=launch["env"],
                      panel_score_sha256=panel_score_binding["sha256"], prior_panel_pass_verified=True,
                      hashes_unchanged=True, repeated_output_byte_identical=True, checks=checks,
                      peak_combined_rss_bytes=peak, max_single_child_rss_bytes=resources["max_single_child_rss_bytes"],
                      sampled_rss_observations=len(samples), remote_series_wall_s=resources["wall_s"],
                      executable=sys.executable, numpy_version=np.__version__, platform=sys.platform,
                      qualifier_wall_s=time.monotonic()-clock_start,
                      truth_accessed=False, estimator_executed_by_qualifier=False, raw_or_adjusted_scored=False,
                      official_grade=False, independent_panel_qualified_by_this_receipt=False,
                      whest_contract_validated_by_this_receipt=False, archive_inspected_by_this_receipt=False,
                      milestone_achieved=False, launch_environment_observed_by_qualifier=False,
                      qualification="Completed fixed local NumPy-free participant/OpenBLAS server artifacts on macOS; sampled RSS is not Linux cgroup memory or an official grade. The separately sealed independent14 pass and explicit prelaunch source/fixture/environment record are verified, including before/after fixture hashes. The old launcher inherits those declared controls; child environment was not independently sampled. This qualifier neither recomputes panel accuracy nor supplies Whest/package validation.")
        write(OUT, result)
        print(json.dumps(dict(pass_gate=True, receipt_sha256=binding(OUT)["sha256"], checks=checks,
                              peak_combined_rss_bytes=peak), sort_keys=True, allow_nan=False), flush=True)
        return 0
    except BaseException:
        write(FAIL, dict(pass_gate=False, candidate=SOURCE, tag=TAG, qualification_started_utc=started,
                         failed_utc=utc(), error=traceback.format_exc(), input_bindings=before, checks=checks,
                         truth_accessed=False, estimator_executed_by_qualifier=False,
                         qualifier_wall_s=time.monotonic()-clock_start))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
