"""Inert exclusive prelaunch provenance; no server/client or estimator execution."""

from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OUT = HERE / "pair_polynomial_f32_remote_launch_v1.json"
TAG = "pair_polynomial_f32_openblas_series_v1"
CANDIDATE = "candidate_response_pair_polynomial_f32_v1.py"
PANEL = "check_pair_polynomial_f32_panel_v1.py"
PINS = {
    CANDIDATE: "0b84ab6fa96d32804c9adec350d41766e47d09a71a936b95652893bb1e364a96",
    PANEL: "9f17e45aaa3e968f5eaca236115e6d060bb2dfc70e450478e1282e5d4ed7588f",
    "PAIR_POLYNOMIAL_F32_REMOTE_RELEASE_V1.md": "5365ccb8e056bf5d06753b54edb52a2d67d034d6fde37b5c4cb08bd028d6bd3d",
    "run_remote_series_v1.py": "00eabab0b7c1ec0c113b998032d441b60aa882670f364732f02a741282634cb1",
    "check_remote_series_v1.py": "f3b0b9dd9e9edd27bf4f85b5a34ab0d72131e22881dd0531b3201dc29642da55",
}
ENV = dict(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
           MKL_NUM_THREADS="1", BLIS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
           PYTHONDONTWRITEBYTECODE="1", FLOPSCOPE_MAX_ARRAY_BYTES="4294967296")


def main():
    if sys.argv[1:] != ["--prepare"]:
        print("Inert. Explicit --prepare required after the panel passes.")
        return
    if OUT.exists() or any((HERE/(TAG+suffix)).exists() for suffix in (".json", "_server.log", "_resources.json")):
        raise FileExistsError("Refusing existing launch or remote artifact")
    if sha256((HERE/PANEL).read_bytes()).hexdigest() != PINS[PANEL]:
        raise RuntimeError("Panel helper changed")
    spec = importlib.util.spec_from_file_location("fixed_fp32_remote_panel", HERE/PANEL)
    p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p)
    h, old_panel, dev = p.modules()
    h.require(Path(sys.prefix).resolve() == h.GATE_ENV.resolve(), "Use OFFPY")
    h.require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "nice10 required")
    h.require({key: os.environ.get(key) for key in ENV} == ENV, "Explicit remote launch environment required")
    acquisition, baseline, _ = p.verify_acquisition(h, old_panel, dev)
    score_path = p.ROOT/"score/result.json"
    score = h.load_json(score_path)
    complete = h.load_json(p.ROOT/"score/completion.json")
    h.require(complete["result_sha256"] == h.digest(score_path) and complete["qualification_passed"]
              and complete["retain_local_component"] and score["retain_local_component"]
              and score["scientific_status"] == "panel_component_pass" and score["hashes_unchanged"]
              and score["source_bindings"] == acquisition["source_bindings"], "No matching panel pass")
    for key in ("role_seals", "truth_bindings"):
        h.check_bindings(score[key])
    source_set = {name: h.binding(HERE/name) for name in (*PINS, Path(__file__).name,
                  "qualify_pair_polynomial_f32_remote_v1.py")}
    h.require(all(source_set[name]["sha256"] == value for name, value in PINS.items()), "Source pin mismatch")
    fixture_set = {f"weights_{rank}": acquisition["fixture_bindings"][f"weights_{rank}"]
                   for rank in (35,100,500,3016500)}
    h.check_bindings(fixture_set)
    record = dict(utc=datetime.now(timezone.utc).isoformat(), tag=TAG, candidate=CANDIDATE,
                  source_sha256=PINS[CANDIDATE], source_bindings=source_set, fixture_bindings=fixture_set,
                  env=ENV, panel_score_sha256=h.digest(score_path), panel_gate_passed=True,
                  planned_sequence=[35,100,500,3016500,35], nice=os.getpriority(os.PRIO_PROCESS, 0),
                  pid=os.getpid(), candidate_executed=False, remote_launched=False,
                  qualification="Explicit prelaunch source/fixture/environment attestation; child environment not independently sampled")
    with OUT.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(launch_receipt_sha256=h.digest(OUT), panel_score_sha256=record["panel_score_sha256"],
                         remote_launched=False, planned_sequence=record["planned_sequence"]), sort_keys=True))


if __name__ == "__main__":
    main()
