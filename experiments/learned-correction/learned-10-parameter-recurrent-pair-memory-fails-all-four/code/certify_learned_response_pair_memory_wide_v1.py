"""Inert, inputs-only rank35 wide output/full-parameter-gradient certificate.

One fixed ten-vector, both history/RESET, one synthetic linear output probe.
No loss.npz, empirical target, fit, parameter/step/seed sweep or bank replay.
Root reviews and executes; the author does not open numerical inputs.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time


HERE = Path(__file__).resolve().parent
ROW = HERE / "learned_response_pair_memory_data_v1" / "row_0035"
INPUTS = ROW / "inputs.npz"
ROW_RECEIPT = ROW / "result.json"
PREPARATION_RECEIPT = ROW.parent / "rank35_receipt.json"
SYNTHETIC = HERE / "learned_response_pair_memory_synthetic_certificate_v1.json"
OUT = HERE / "learned_response_pair_memory_wide_certificate_v1.json"
START = HERE / "learned_response_pair_memory_wide_certificate_started_v1.json"
PINS = {
    "LEARNED_RESPONSE_PAIR_MEMORY_PROTOCOL_DRAFT_V1.md":
        "1e277cd16c6596cca947172d0dcd2bd7c21c8962256596ca8ea070c65fc8754c",
    "LEARNED_RESPONSE_PAIR_MEMORY_ROOT_ACCEPTANCE_V1.md":
        "45c590e373fb35c6af0c9a248addf8b98d5e6830c8b6b66861fd5ced3b537e2a",
    "learned_response_pair_memory_numpy_v1.py":
        "5a24a77c24f08ba521fa8d0cd36a569a9eeee6bf3045cb4ee15d0117fbf4129e",
    "learned_response_pair_memory_torch_v1.py":
        "54e4040947cfd818246b1cc604fd79516e426771ca23b1568eb2391b394c2789",
    "certify_learned_response_pair_memory_v1.py":
        "e34bab8fdaa6074b58394450d84f94765c8f8d86f80363a91258702c7fdebb4e",
}
PARAMETERS = [0.17, -0.23, 0.31, 0.11, -0.13, 0.19, 0.07, -0.29, 0.37, -0.41]
SEED, N, HISTORY = 930174, 1024, 13
RSS_LIMIT = 6*(1 << 30)
TOLERANCE = 1e-4
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def source_path(name):
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts,
            f"Source key is not relative to NEW: {name}")
    require(relative.name != "loss.npz", "The wide certificate must never open loss.npz")
    path = (HERE / relative).resolve()
    require(path.is_relative_to(HERE), f"Source escaped NEW: {name}")
    return path


def execute():
    require(not OUT.exists() and not START.exists(), "Preserve existing certificate/start; no retry")
    started = time.perf_counter()
    source_hashes, checked_files = {}, {}
    result = dict(schema="learned-response-pair-memory-wide-certificate-v1",
        passed=False, rank=35, width=N, history_length=HISTORY,
        parameters=PARAMETERS, synthetic_probe_seed=SEED, arms={}, stage="preflight",
        target_access=False, loss_file_opened=False, predictor_fit=False,
        bank_replayed=False, raw_or_information_claim=False,
        synthetic_probe_definition="fresh default_rng(930174).normal((1024,1024))/sqrt(1024*1024); sum(output*probe)",
        gates=dict(cpu64_vs_numpy64_output_relative=TOLERANCE,
                   mps32_vs_numpy64_output_relative=TOLERANCE,
                   mps32_vs_cpu64_full_gradient_relative=TOLERANCE,
                   mps32_vs_cpu64_theta_gradient_relative=TOLERANCE,
                   peak_process_rss_bytes=RSS_LIMIT),
        started_utc=datetime.now(timezone.utc).isoformat())
    try:
        require("/anaconda3/" in sys.executable, "Use AP/Torch interpreter")
        require(not any(k == "torch" or k.startswith("torch.") for k in sys.modules),
                "Use a fresh process before MPS environment setup")
        require("threadpoolctl" not in sys.modules, "AP must never import threadpoolctl")
        require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "Launch nice >=10")
        for key in THREAD_ENV:
            os.environ[key] = "2"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        for name, expected in PINS.items():
            digest = sha(source_path(name))
            require(digest == expected, f"Frozen prerequisite source changed: {name}")
            source_hashes[name] = digest
        source_hashes[Path(__file__).name] = sha(Path(__file__).resolve())
        source_hashes[SYNTHETIC.name] = sha(SYNTHETIC)
        tiny = json.loads(SYNTHETIC.read_text())
        require(tiny["schema"] == "learned-response-pair-memory-synthetic-certificate-v1" and
                tiny["passed"] and tiny["hashes_unchanged"] and tiny["synthetic_only"] and
                tiny["target_access"] is False and tiny["seed"] == SEED and tiny["width"] == 7 and
                tiny["history_length"] == HISTORY and tiny["parameters"] == PARAMETERS and
                tiny["fd_step"] == 1e-5,
                "The exact fixed synthetic/FD certificate has not qualified")
        require(set(tiny["arms"]) == {"history", "reset"} and
                all(row["passed"] for row in tiny["arms"].values()), "Incomplete synthetic arms")
        for name, digest in tiny["source_hashes"].items():
            require(sha(source_path(name)) == digest, f"Synthetic source changed: {name}")
            if name in source_hashes:
                require(source_hashes[name] == digest, f"Conflicting prerequisite digest: {name}")
            source_hashes[name] = digest
        for name, digest in PINS.items():
            require(tiny["source_hashes"][name] == digest, f"Synthetic fixture used another source: {name}")

        receipt_name = str(ROW_RECEIPT.relative_to(HERE))
        checked_files[receipt_name] = sha(ROW_RECEIPT)
        completion_name = str(PREPARATION_RECEIPT.relative_to(HERE))
        checked_files[completion_name] = sha(PREPARATION_RECEIPT)
        completion = json.loads(PREPARATION_RECEIPT.read_text())
        require(completion["passed"] and completion["hashes_unchanged"] and
                len(completion["rows"]) == 1 and completion["rows"][0]["rank"] == 35 and
                completion["rows"][0]["passed"] and
                completion["rows"][0]["result_sha256"] == checked_files[receipt_name],
                "The complete rank35 preparation/source-manifest seal is missing or changed")
        for name, digest in completion["source_hashes"].items():
            require(sha(source_path(name)) == digest, f"Preparation completion source changed: {name}")
            if name in source_hashes:
                require(source_hashes[name] == digest, f"Completion/prerequisite mismatch: {name}")
            source_hashes[name] = digest
        row = json.loads(ROW_RECEIPT.read_text())
        require(row["passed"] and row["rank"] == 35 and row["peak_rss_bytes"] <= RSS_LIMIT,
                "The exact rank35 causal row did not qualify")
        for name, digest in row["source_hashes"].items():
            require(sha(source_path(name)) == digest, f"Data-preparation source changed: {name}")
            if name in source_hashes:
                require(source_hashes[name] == digest, f"Data/prerequisite source mismatch: {name}")
            source_hashes[name] = digest
        input_name = str(INPUTS.relative_to(HERE))
        checked_files[input_name] = sha(INPUTS)
        require(checked_files[input_name] == row["artifacts"]["inputs.npz"]["sha256"],
                "The sealed causal input artifact changed")
        # Intentionally do not iterate row.artifacts or original_input_manifest:
        # either can name teacher artifacts. Only inputs.npz is decoded here.
        result.update(source_hashes=source_hashes, checked_files=checked_files,
            data_receipt_sha256=checked_files[receipt_name], inputs_sha256=checked_files[input_name],
            preparation_completion_sha256=checked_files[completion_name],
            synthetic_certificate_sha256=source_hashes[SYNTHETIC.name], stage="qualified_inputs",
            runtime=dict(interpreter=sys.executable, nice=os.getpriority(os.PRIO_PROCESS, 0),
                environment={k: os.environ[k] for k in (*THREAD_ENV, "PYTORCH_ENABLE_MPS_FALLBACK")}))
        write_json(START, result)

        import numpy as np
        import torch
        import learned_response_pair_memory_numpy_v1 as numpy_model
        import learned_response_pair_memory_torch_v1 as torch_model

        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        require("threadpoolctl" not in sys.modules, "Unexpected threadpoolctl import in AP")
        require(torch.backends.mps.is_available(), "Native MPS required; no CPU fallback")
        require(numpy_model.DATA_KEYS == torch_model.DATA_KEYS and
                numpy_model.PARAMETER_ORDER == torch_model.PARAMETER_ORDER and
                numpy_model.RESET_INACTIVE_PARAMETER_INDICES == (1, 3) and
                torch_model.RESET_INACTIVE_PARAMETER_INDICES == (1, 3), "Pure model contracts differ")
        result["runtime"].update(numpy_version=np.__version__, torch_version=torch.__version__,
            cpu_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads())

        def finite(x):
            return bool(np.isfinite(x).all())

        def relative(x, reference):
            return float(np.linalg.norm(x-reference)/max(float(np.linalg.norm(reference)), 1e-30))

        def resource_gate():
            peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            require(peak <= RSS_LIMIT, f"Wide certificate exceeded6GiB RSS: {peak}")

        def array_digest(arrays):
            digest = hashlib.sha256()
            for name in sorted(arrays):
                value = np.ascontiguousarray(arrays[name])
                digest.update(name.encode())
                digest.update(str(value.shape).encode())
                digest.update(str(value.dtype).encode())
                digest.update(memoryview(value).cast("B"))
            return digest.hexdigest()

        with np.load(INPUTS, allow_pickle=False) as saved:
            require(set(saved.files) == numpy_model.DATA_KEYS, "Wrong causal schema or extra loss field")
            data64 = {}
            input_shapes, input_dtypes = {}, {}
            for name in saved.files:
                value = saved[name]
                input_shapes[name], input_dtypes[name] = list(value.shape), str(value.dtype)
                require(finite(value), f"Nonfinite prepared causal field: {name}")
                require(value.dtype == (np.dtype("bool") if name == "mask" else np.dtype("float32")),
                        f"Expected immutable prepared float32/bool data: {name}")
                data64[name] = value.copy() if name == "mask" else value.astype(np.float64)
                del value
        require(data64["K"].shape == (N, N) and data64["V"].shape == (HISTORY, N, N),
                "This is not fixed rank35 width1024/history13")
        require(np.count_nonzero(data64["V"][0]) == 0, "Prepared V0 is not zero")
        require(not np.diag(data64["mask"]).any(), "Mask includes the excluded current query diagonal")
        p64 = np.asarray(PARAMETERS, dtype=np.float64)
        probe = np.random.default_rng(SEED).normal(size=(N, N))/math.sqrt(N*N)
        require(finite(probe), "Nonfinite synthetic probe")
        before_inputs = array_digest(data64)
        result.update(input_shapes=input_shapes, prepared_input_dtypes=input_dtypes,
            cast_policy="prepared float32 -> CPU/NumPy64; original float32 restored for native MPS32",
            in_memory_input_sha256=before_inputs,
            synthetic_probe_sha256=array_digest(dict(probe=probe)))
        resource_gate()

        def torch_reference(memory, device, dtype):
            started_call = time.perf_counter()
            tensors = {name: torch.tensor(value, device=device,
                       dtype=torch.bool if name == "mask" else dtype)
                       for name, value in data64.items()}
            parameter = torch.tensor(PARAMETERS, device=device, dtype=dtype, requires_grad=True)
            weights = torch.tensor(probe, device=device, dtype=dtype)
            require(all(not value.requires_grad for value in tensors.values()),
                    "Causal inputs became trainable")
            output = torch_model.predict(tensors, parameter, memory=memory)
            resource_gate()
            linear = torch.sum(output*weights)
            gradient, = torch.autograd.grad(linear, parameter)
            if device == "mps":
                torch.mps.synchronize()
            value = output.detach().cpu().numpy().astype(np.float64, copy=True)
            grad = gradient.detach().cpu().numpy().astype(np.float64, copy=True)
            require(finite(value) and finite(grad), f"Nonfinite {device} output/gradient")
            require(value.shape == (N, N) and grad.shape == (10,), "Wrong wide output/gradient shape")
            require(float(np.linalg.norm(grad)) > 0 and float(np.linalg.norm(grad[:8])) > 0,
                    "Vacuous fixed-probe parameter/gate gradient")
            if not memory:
                require(np.count_nonzero(grad[[1, 3]]) == 0, f"{device} RESET inactive gradients changed")
            memory_stats = dict(process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            if device == "mps":
                memory_stats.update(mps_allocated_bytes=int(torch.mps.current_allocated_memory()),
                                    mps_driver_bytes=int(torch.mps.driver_allocated_memory()))
            del output, gradient, linear, parameter, tensors, weights
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            resource_gate()
            return value, grad, dict(wall_s=time.perf_counter()-started_call, **memory_stats)

        for memory in (True, False):
            name = "history" if memory else "reset"
            record = result["arms"][name] = dict(passed=False, allocated_parameters=10,
                structurally_active_parameters=10 if memory else 8)
            result["stage"] = name+"_numpy64"
            started_call = time.perf_counter()
            reference = numpy_model.predict(data64, p64, memory=memory)
            require(finite(reference) and float(np.linalg.norm(reference)) > 0,
                    "Vacuous or nonfinite NumPy64 wide output")
            record["numpy64_wall_s"] = time.perf_counter()-started_call
            record["numpy64_output_norm"] = float(np.linalg.norm(reference))
            require(np.count_nonzero(reference[~data64["mask"]]) == 0, "NumPy output escaped mask")
            resource_gate()
            result["stage"] = name+"_cpu64_autograd"
            cpu_value, cpu_grad, cpu_resources = torch_reference(memory, "cpu", torch.float64)
            record.update(cpu64_vs_numpy64_output_relative=relative(cpu_value, reference),
                cpu64_gradient=cpu_grad.tolist(), cpu64_resources=cpu_resources)
            require(record["cpu64_vs_numpy64_output_relative"] <= TOLERANCE,
                    "Wide CPU64/NumPy64 output mismatch")
            del cpu_value
            gc.collect()
            result["stage"] = name+"_mps32_autograd"
            mps_value, mps_grad, mps_resources = torch_reference(memory, "mps", torch.float32)
            record.update(mps32_vs_numpy64_output_relative=relative(mps_value, reference),
                mps32_vs_cpu64_full_gradient_relative=relative(mps_grad, cpu_grad),
                mps32_vs_cpu64_theta_gradient_relative=relative(mps_grad[:8], cpu_grad[:8]),
                mps32_vs_cpu64_beta_gradient_relative=relative(mps_grad[8:], cpu_grad[8:]),
                mps32_gradient=mps_grad.tolist(), mps32_resources=mps_resources)
            require(record["mps32_vs_numpy64_output_relative"] <= TOLERANCE and
                    record["mps32_vs_cpu64_full_gradient_relative"] <= TOLERANCE and
                    record["mps32_vs_cpu64_theta_gradient_relative"] <= TOLERANCE,
                    "Wide native MPS32 output/full-history parameter-gradient mismatch")
            require(np.count_nonzero(mps_value[~data64["mask"]]) == 0, "MPS output escaped mask")
            record["passed"] = True
            print(json.dumps(dict(stage=name+"_qualified", **record)), flush=True)
            del reference, cpu_grad, mps_value, mps_grad
            gc.collect()
            resource_gate()

        require(array_digest(data64) == before_inputs, "A pure model mutated causal inputs")
        require("threadpoolctl" not in sys.modules, "Unexpected threadpoolctl import in AP")
        result["input_arrays_unchanged"] = True
        result["stage"] = "complete"
        result["passed"] = True
    except BaseException as exc:
        result["passed"] = False
        result["exception_type"], result["exception"] = type(exc).__name__, str(exc)
    finally:
        final_sources, final_inputs = {}, {}
        for name in source_hashes:
            try:
                final_sources[name] = sha(source_path(name))
            except BaseException:
                final_sources[name] = None
        for name in checked_files:
            try:
                final_inputs[name] = sha(source_path(name))
            except BaseException:
                final_inputs[name] = None
        result.update(source_hashes=source_hashes, checked_files=checked_files,
            final_source_hashes=final_sources, final_checked_files=final_inputs,
            hashes_unchanged=(final_sources == source_hashes and final_inputs == checked_files),
            wall_s=time.perf_counter()-started, finished_utc=datetime.now(timezone.utc).isoformat(),
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        result["passed"] = bool(result["passed"] and result["hashes_unchanged"] and
                                result["peak_rss_bytes"] <= RSS_LIMIT)
        write_json(OUT, result)
        print(json.dumps({key: value for key, value in result.items()
              if key not in ("source_hashes", "final_source_hashes", "checked_files",
                             "final_checked_files", "arms")}), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-reviewed-wide-certificate", action="store_true")
    args = parser.parse_args()
    if args.execute_reviewed_wide_certificate:
        execute()
    else:
        print(json.dumps(dict(execute=False, rank=35, input_path=str(INPUTS),
            output_path=str(OUT), both_arms=True, arrays_opened=False,
            numerical_imports=False, loss_file_opened=False)))
