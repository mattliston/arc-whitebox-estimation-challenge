"""Inert, one-fixture numerical certificate for the accepted pair-memory law.

Synthetic n7/history13, seed930174; no real weights, moments or loss data.
No execution without --execute in a fresh nice10 AP/Torch process. Both
history and RESET use one fixed nonzero parameter vector and linear loss.
No parameter/step/seed/shape sweep, retry, training or fallback is present.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time


HERE = Path(__file__).resolve().parent
OUT = HERE / "learned_response_pair_memory_synthetic_certificate_v1.json"
START = HERE / "learned_response_pair_memory_synthetic_certificate_started_v1.json"
PINS = {
    "LEARNED_RESPONSE_PAIR_MEMORY_PROTOCOL_DRAFT_V1.md":
        "1e277cd16c6596cca947172d0dcd2bd7c21c8962256596ca8ea070c65fc8754c",
    "LEARNED_RESPONSE_PAIR_MEMORY_ROOT_ACCEPTANCE_V1.md":
        "45c590e373fb35c6af0c9a248addf8b98d5e6830c8b6b66861fd5ced3b537e2a",
    "learned_response_pair_memory_numpy_v1.py":
        "5a24a77c24f08ba521fa8d0cd36a569a9eeee6bf3045cb4ee15d0117fbf4129e",
    "learned_response_pair_memory_torch_v1.py":
        "54e4040947cfd818246b1cc604fd79516e426771ca23b1568eb2391b394c2789",
}
SEED, N, HISTORY = 930174, 7, 13
PARAMETERS = [0.17, -0.23, 0.31, 0.11, -0.13, 0.19, 0.07, -0.29, 0.37, -0.41]
FD_STEP = 1e-5
RSS_LIMIT = 2 * (1 << 30)
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def execute():
    require("/anaconda3/" in sys.executable, "Use the requested AP/Torch interpreter")
    require(not any(k == "torch" or k.startswith("torch.") for k in sys.modules),
            "A fresh process is required before setting MPS/thread environment")
    require("threadpoolctl" not in sys.modules, "AP must not import threadpoolctl")
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "Launch at nice >=10")
    require(not OUT.exists() and not START.exists(), "Refusing overwrite or retry")
    for name in THREAD_ENV:
        os.environ[name] = "2"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    paths = {name: HERE / name for name in PINS}
    paths[Path(__file__).name] = Path(__file__).resolve()
    hashes = {name: sha(path) for name, path in paths.items()}
    require(all(hashes[name] == digest for name, digest in PINS.items()),
            "A frozen model/protocol/acceptance source changed")
    started = time.perf_counter()
    result = dict(schema="learned-response-pair-memory-synthetic-certificate-v1",
                  passed=False, synthetic_only=True, target_access=False,
                  raw_or_information_claim=False, seed=SEED, width=N,
                  history_length=HISTORY, parameters=PARAMETERS, fd_step=FD_STEP,
                  source_hashes=hashes, arms={}, stage="started",
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  runtime=dict(interpreter=sys.executable, nice=os.getpriority(os.PRIO_PROCESS, 0),
                               environment={k: os.environ[k] for k in (*THREAD_ENV,
                                            "PYTORCH_ENABLE_MPS_FALLBACK")}),
                  gates=dict(cpu64_output_relative=1e-6, cpu64_parameter_fd_relative=1e-6,
                             mps32_output_relative=1e-4, mps32_gradient_relative=1e-4,
                             unit_conversion_relative=1e-12,
                             symmetry_and_query_parity_float64=1e-12,
                             symmetry_and_query_parity_float32=1e-4,
                             peak_process_rss_bytes=RSS_LIMIT))
    write_json(START, result)
    try:
        import numpy as np
        import torch
        import learned_response_pair_memory_numpy_v1 as numpy_model
        import learned_response_pair_memory_torch_v1 as torch_model

        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        require("threadpoolctl" not in sys.modules, "Unexpected threadpoolctl import in AP")
        require(torch.backends.mps.is_available(), "Native MPS is required; no fallback")
        require(numpy_model.PARAMETER_ORDER == torch_model.PARAMETER_ORDER and
                numpy_model.DATA_KEYS == torch_model.DATA_KEYS and
                numpy_model.RESET_INACTIVE_PARAMETER_INDICES == (1, 3) and
                torch_model.RESET_INACTIVE_PARAMETER_INDICES == (1, 3),
                "Model contracts disagree")
        result["runtime"].update(numpy_version=np.__version__, torch_version=torch.__version__,
                                 cpu_threads=torch.get_num_threads(),
                                 interop_threads=torch.get_num_interop_threads())

        def finite(x):
            return bool(np.isfinite(x).all())

        def relative(x, reference):
            return float(np.linalg.norm(x-reference) /
                         max(float(np.linalg.norm(reference)), 1e-30))

        def digest_arrays(arrays):
            h = hashlib.sha256()
            for name in sorted(arrays):
                a = np.asarray(arrays[name])
                h.update(name.encode())
                h.update(str(a.dtype).encode())
                h.update(str(a.shape).encode())
                h.update(np.ascontiguousarray(a).tobytes())
            return h.hexdigest()

        def resource_gate():
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= RSS_LIMIT,
                    "Synthetic certificate exceeded2GiB process RSS")

        # All values below come from this sole deterministic synthetic stream.
        # Positive scales make V and the independent J spelling coherent.
        rng = np.random.default_rng(SEED)
        w = rng.normal(size=(HISTORY, N, N)) / math.sqrt(N)
        post_sigma = np.exp(0.15*rng.normal(size=(HISTORY, N)))
        pre_sigma = np.exp(0.15*rng.normal(size=(HISTORY, N)))
        g = rng.uniform(0.05, 0.95, size=(HISTORY, N))
        alive = np.ones((HISTORY, N), dtype=np.bool_)
        alive[np.arange(HISTORY), np.arange(HISTORY) % N] = False
        D = rng.normal(scale=0.5, size=(HISTORY, N, N))
        d = rng.uniform(-0.9, 0.9, size=(HISTORY, N))
        D[:, np.arange(N), np.arange(N)] = d
        B = 0.5*(d[:, :, None]*D + D.transpose(0, 2, 1)*d[:, None, :])
        B[:, np.arange(N), np.arange(N)] = 0
        # Direct moderate-input psi is an independent fixture-source spelling.
        b = B/np.sqrt(1+B*B)
        V = np.zeros((HISTORY, N, N), dtype=np.float64)
        for layer in range(1, HISTORY):
            V[layer] = post_sigma[layer-1, :, None]*w[layer]/pre_sigma[layer, None, :]
        post_scale = g*alive*pre_sigma/post_sigma
        query_w = rng.normal(size=(N, N))/math.sqrt(N)
        query_sigma = np.exp(0.15*rng.normal(size=N))
        query_V = post_sigma[-1, :, None]*query_w/query_sigma[None, :]
        E = D[-1]-(2/3)*np.diag(d[-1])
        Q = E@query_V
        K = (query_V*query_V).T@Q + 2*(query_V*Q).T@query_V
        query_alive = np.array([True, True, False, True, True, True, False])
        mask = query_alive[:, None] & query_alive[None, :]
        np.fill_diagonal(mask, False)
        data64 = dict(V=V, b=b, d=d, g=g, post_scale=post_scale,
                      query_V=query_V, K=K, mask=mask)
        data32 = {k: v.copy() if k == "mask" else v.astype(np.float32)
                  for k, v in data64.items()}
        p64 = np.asarray(PARAMETERS, dtype=np.float64)
        p32 = p64.astype(np.float32)
        loss_weight64 = rng.normal(size=(N, N))/math.sqrt(N*N)
        sign64 = np.array([1., -1., 1., -1., -1., 1., -1.])
        result["fixture_sha256"] = digest_arrays(dict(data64, parameters=p64,
                                                     loss_weight=loss_weight64))
        before_data = digest_arrays(data64), digest_arrays(data32)
        require(set(data64) == numpy_model.DATA_KEYS and all(finite(x) for x in data64.values()),
                "Synthetic causal contract is invalid")

        # Independent real-unit identity, one H generated from the same stream.
        hu = rng.normal(size=(N, N))
        hu = 0.5*(hu+hu.T)
        unit_errors = []
        for layer in range(1, HISTORY):
            J = (post_sigma[layer-1, :, None]*w[layer] *
                 (g[layer]*alive[layer]/post_sigma[layer])[None, :])
            physical = (J.T@hu)@J
            scaled = np.outer(post_scale[layer], post_scale[layer])*((V[layer].T@hu)@V[layer])
            unit_errors.append(relative(scaled, physical))
        result["unit_conversion_relative"] = max(unit_errors)
        require(max(unit_errors) <= 1e-12, "Post/pre unit-conversion identity failed")

        def numpy_value(data, parameters, memory):
            out, state = numpy_model.predict(data, parameters, memory=memory, return_state=True)
            require(all(finite(x) for x in (out, *state.values())), "Nonfinite NumPy predictor")
            return out, state

        def torch_value(data, parameters, memory, device, dtype):
            tensors = {k: torch.tensor(v, device=device,
                        dtype=torch.bool if k == "mask" else dtype) for k, v in data.items()}
            parameter = torch.tensor(parameters, dtype=dtype, device=device, requires_grad=True)
            out, state = torch_model.predict(tensors, parameter, memory=memory, return_state=True)
            weights = torch.tensor(loss_weight64, dtype=dtype, device=device)
            gradient, = torch.autograd.grad(torch.sum(out*weights), parameter)
            if device == "mps":
                torch.mps.synchronize()
            value = out.detach().cpu().numpy()
            saved = {k: v.detach().cpu().numpy() for k, v in state.items()}
            grad = gradient.detach().cpu().numpy().astype(np.float64)
            require(all(finite(x) for x in (value, grad, *saved.values())),
                    f"Nonfinite Torch {device} predictor/gradient")
            require(all(not v.requires_grad for v in tensors.values()), "Synthetic inputs changed ownership")
            return value, saved, grad

        def check_state(value, state, tolerance):
            require(set(state) == {"H", "query_U", "query_A", "F0", "F1"}, "Wrong state contract")
            require(all(x.shape == (N, N) for x in (value, *state.values())), "Wrong output shape")
            symmetry = max(relative(state[k], state[k].T) for k in ("H", "query_A"))
            maximum = max(float(np.max(np.abs(state[k]))) for k in ("H", "query_A"))
            require(symmetry <= tolerance and maximum <= 1+tolerance, "Latent symmetry/bound failed")
            require(np.count_nonzero(value[~mask]) == 0, "Correction escaped the live offdiagonal mask")
            return dict(symmetry_relative=symmetry, max_abs_bounded_field=maximum,
                        output_norm=float(np.linalg.norm(value)))

        def sign_data(data):
            sign = sign64.astype(data["K"].dtype)
            return dict(data, query_V=data["query_V"]*sign[None, :], K=data["K"]*sign[None, :])

        def sign_error(base, changed):
            out, state = base
            out_s, state_s = changed
            errors = [relative(out_s, out*sign64[None, :]), relative(state_s["H"], state["H"])]
            for name in ("query_U", "query_A"):
                errors.append(relative(state_s[name], sign64[:, None]*state[name]*sign64[None, :]))
            for name in ("F0", "F1"):
                errors.append(relative(state_s[name], state[name]*sign64[None, :]))
            return max(errors)

        for memory in (True, False):
            name = "history" if memory else "reset"
            result["stage"] = name+"_cpu64"
            record = result["arms"][name] = dict(allocated_parameters=10,
                        structurally_active_parameters=10 if memory else 8)
            n64, ns64 = numpy_value(data64, p64, memory)
            c64, cs64, cg64 = torch_value(data64, p64, memory, "cpu", torch.float64)
            fd = np.empty(10, dtype=np.float64)
            for coordinate in range(10):
                plus, minus = p64.copy(), p64.copy()
                plus[coordinate] += FD_STEP
                minus[coordinate] -= FD_STEP
                yp = numpy_model.predict(data64, plus, memory=memory)
                ym = numpy_model.predict(data64, minus, memory=memory)
                fd[coordinate] = np.sum((yp-ym)*loss_weight64)/(2*FD_STEP)
            require(finite(fd) and float(np.linalg.norm(cg64)) > 0, "Vacuous/nonfinite gradient fixture")
            record.update(cpu64_output_relative=relative(c64, n64),
                          cpu64_autograd_vs_numpy64_fd_relative=relative(cg64, fd),
                          cpu64_gate_autograd_vs_numpy64_fd_relative=relative(cg64[:8], fd[:8]),
                          cpu64_gradient=cg64.tolist(), numpy64_fd_gradient=fd.tolist(),
                          numpy64_state=check_state(n64, ns64, 1e-12),
                          cpu64_state=check_state(c64, cs64, 1e-12))
            require(record["cpu64_output_relative"] <= 1e-6 and
                    record["cpu64_autograd_vs_numpy64_fd_relative"] <= 1e-6 and
                    record["cpu64_gate_autograd_vs_numpy64_fd_relative"] <= 1e-6,
                    "CPU64 output/parameter-gradient qualification failed")
            require(float(np.linalg.norm(cg64[:8])) > 0, "No gate-gradient signal in fixed fixture")
            if not memory:
                require(np.count_nonzero(cg64[[1, 3]]) == 0 and np.count_nonzero(fd[[1, 3]]) == 0,
                        "RESET inactive coefficients have nonzero derivatives")
            sn64, sns64 = numpy_value(sign_data(data64), p64, memory)
            sc64, scs64, _ = torch_value(sign_data(data64), p64, memory, "cpu", torch.float64)
            record["numpy64_query_sign_relative"] = sign_error((n64, ns64), (sn64, sns64))
            record["cpu64_query_sign_relative"] = sign_error((c64, cs64), (sc64, scs64))
            require(max(record["numpy64_query_sign_relative"], record["cpu64_query_sign_relative"]) <= 1e-12,
                    "Float64 query sign parity failed")

            result["stage"] = name+"_mps32"
            n32, ns32 = numpy_value(data32, p32, memory)
            m32, ms32, mg32 = torch_value(data32, p32, memory, "mps", torch.float32)
            record.update(mps32_vs_numpy32_output_relative=relative(m32, n32),
                          mps32_vs_fd_qualified_cpu64_gradient_relative=relative(mg32, cg64),
                          mps32_vs_fd_qualified_cpu64_gate_gradient_relative=relative(mg32[:8], cg64[:8]),
                          mps32_gradient=mg32.tolist(),
                          numpy32_state=check_state(n32, ns32, 1e-4),
                          mps32_state=check_state(m32, ms32, 1e-4))
            require(record["mps32_vs_numpy32_output_relative"] <= 1e-4 and
                    record["mps32_vs_fd_qualified_cpu64_gradient_relative"] <= 1e-4 and
                    record["mps32_vs_fd_qualified_cpu64_gate_gradient_relative"] <= 1e-4,
                    "MPS32 output/gradient qualification failed")
            if not memory:
                require(np.count_nonzero(mg32[[1, 3]]) == 0, "MPS RESET inactive gradient is nonzero")
            sn32, sns32 = numpy_value(sign_data(data32), p32, memory)
            sm32, sms32, _ = torch_value(sign_data(data32), p32, memory, "mps", torch.float32)
            record["numpy32_query_sign_relative"] = sign_error((n32, ns32), (sn32, sns32))
            record["mps32_query_sign_relative"] = sign_error((m32, ms32), (sm32, sms32))
            require(max(record["numpy32_query_sign_relative"], record["mps32_query_sign_relative"]) <= 1e-4,
                    "Float32 query sign parity failed")
            record["passed"] = True
            resource_gate()

        require(before_data == (digest_arrays(data64), digest_arrays(data32)), "Predictor mutated causal inputs")
        require("threadpoolctl" not in sys.modules, "Unexpected threadpoolctl import in AP")
        result["input_arrays_unchanged"] = True
        result["mps_memory"] = dict(current_allocated_bytes=int(torch.mps.current_allocated_memory()),
                                    driver_allocated_bytes=int(torch.mps.driver_allocated_memory()))
        result["stage"] = "complete"
        result["passed"] = True
        resource_gate()
    except BaseException as exc:
        result["passed"] = False
        result["exception_type"], result["exception"] = type(exc).__name__, str(exc)
    finally:
        result["wall_s"] = time.perf_counter()-started
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        result["peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        result["final_source_hashes"] = {name: sha(path) for name, path in paths.items()}
        result["hashes_unchanged"] = result["final_source_hashes"] == hashes
        result["passed"] = bool(result["passed"] and result["hashes_unchanged"] and
                                result["peak_rss_bytes"] <= RSS_LIMIT)
        write_json(OUT, result)
        print(json.dumps({k: v for k, v in result.items() if k not in
                          ("source_hashes", "final_source_hashes")}, allow_nan=False), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        execute()
    else:
        print("INERT: fixed synthetic fixture only; --execute required after source review.")
