"""Read-only-array qualification of already completed fixed remote outputs.

No estimator execution or truth access. Source, served-byte, native-parity,
and resource checks are recorded in an exclusively created new receipt.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TAG = "response_rows_current_runtime_openblas_series_v2"
SOURCE = "candidate_response_rows_current_runtime_v1.py"
PINS = {
    SOURCE: "5a07832fee84b94c3a98589dc3ef19d718c1b062f5c4d3e04452a56ab9f11172",
    "run_remote_series_v1.py": "00eabab0b7c1ec0c113b998032d441b60aa882670f364732f02a741282634cb1",
    "check_remote_series_v1.py": "f3b0b9dd9e9edd27bf4f85b5a34ab0d72131e22881dd0531b3201dc29642da55",
    "RESPONSE_ROWS_ADJUSTED_EXCEPTION_V1.md": "ba7b74f2c00bab570e4b722393a39a56eebc905d19d284f2d69dec2fe4c95cbb",
    "response_rows_adjusted_reevaluation_v1/response/result.json": "b65e248340c4bc3dd37948254a150fe28e75be906d0d133215b30d45996f511e",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    outfile = HERE / "response_rows_current_runtime_remote_qualified_v1.json"
    if outfile.exists():
        raise FileExistsError(outfile)
    for name, expected in PINS.items():
        assert sha(HERE / name) == expected, name
    client_path = HERE / (TAG + ".json")
    resources_path = HERE / (TAG + "_resources.json")
    receipt = json.loads(client_path.read_text())
    resources = json.loads(resources_path.read_text())
    assert receipt["status"] == "completed" and receipt["source_sha256"] == PINS[SOURCE]
    assert receipt["repeated_output_byte_identical"]
    records = receipt["records"]
    assert [r["rank"] for r in records] == [35, 100, 500, 3016500, 35]
    native_dir = HERE / "response_rows_adjusted_reevaluation_v1/response"
    native = json.loads((native_dir / "result.json").read_text())
    checks, output_bindings, arrays = [], {}, []
    for r in records:
        index, rank = r["index"], r["rank"]
        path = HERE / f"{TAG}_{index}_r{rank}.f32"
        output_bindings[path.name] = sha(path)
        assert output_bindings[path.name] == r["output_sha256"]
        assert r["output_shape"] == [16, 1024] and r["output_dtype"] == "float32"
        array = np.fromfile(path, dtype=np.float32).reshape(16, 1024)
        assert np.isfinite(array).all() and (array >= 0).all()
        arrays.append(array)
        timing = r["timing"]
        check = dict(index=index, rank=rank, finite=True, nonnegative=True,
                     flops=r["flops"], multiplier=max(.1, r["flops"] / 2**41),
                     wall_s=timing["wall_time_s"], residual_s=timing["residual_wall_time_s"])
        assert 0 < r["flops"] <= 2**41
        assert timing["wall_time_s"] < 60 and timing["residual_wall_time_s"] < .4
        if index < 2:
            trial = native["trials"][index]
            path = native_dir / trial["output_artifact"]
            ref = np.load(path, allow_pickle=False)
            assert hashlib.sha256(ref.tobytes()).hexdigest() == trial["output_bytes_sha256"]
            error = array.astype(np.float64) - ref.astype(np.float64)
            check["native_output_max_abs"] = float(np.max(np.abs(error)))
            check["native_output_rms"] = float(np.sqrt(np.mean(error**2)))
            assert check["native_output_max_abs"] < 1e-5
        checks.append(check)
    assert arrays[0].tobytes() == arrays[-1].tobytes()
    assert resources["candidate"] == SOURCE
    assert 0 < resources["peak_combined_rss_bytes"] < 8_000_000_000
    for name, expected in PINS.items():
        assert sha(HERE / name) == expected, name
    for name, expected in output_bindings.items():
        assert sha(HERE / name) == expected, name
    result = dict(pass_gate=True, candidate=SOURCE, source_sha256=PINS[SOURCE],
                  input_sha256=PINS, client_receipt_sha256=sha(client_path),
                  resource_receipt_sha256=sha(resources_path), output_sha256=output_bindings,
                  repeated_output_byte_identical=True, checks=checks,
                  peak_combined_rss_bytes=resources["peak_combined_rss_bytes"],
                  max_single_child_rss_bytes=resources["max_single_child_rss_bytes"],
                  truth_accessed=False, estimator_executed=False,
                  official_grade=False, panel_qualified=False, milestone_achieved=False,
                  qualification="Local NumPy-free participant, explicit one-thread OpenBLAS server; not an official Linux grade.")
    with outfile.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(pass_gate=True, receipt_sha256=sha(outfile),
                         peak_combined_rss_bytes=result["peak_combined_rss_bytes"], checks=checks)), flush=True)


if __name__ == "__main__":
    main()
