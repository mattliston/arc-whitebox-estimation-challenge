"""Fixed small synthetic MPS arithmetic/gradient check, not a fit."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def run():
    check("anaconda3" in sys.executable, "AP Torch interpreter required")
    check("threadpoolctl" not in sys.modules, "No threadpoolctl with Torch")
    check(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "nice10 required")
    check(os.environ.get("PYTHONDONTWRITEBYTECODE") == "1", "No bytecode writes")
    outdir = HERE / "numerical_qualification"
    outdir.mkdir(exist_ok=False)
    result = dict(passed=False, training=False, truth_access=False,
                  model_sha256=sha(HERE / "memory_model.py"),
                  script_sha256=sha(Path(__file__)))
    write(outdir / "started.json", dict(**result, pid=os.getpid()))
    start = time.perf_counter()

    def alarm(signum, frame):
        raise TimeoutError("Fixed180s numerical qualification cap")

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(180)
    try:
        import numpy as np
        import torch
        check("threadpoolctl" not in sys.modules, "Unexpected threadpoolctl")
        check(torch.backends.mps.is_available(), "MPS unavailable")
        torch.set_num_threads(1)
        spec = importlib.util.spec_from_file_location("fixed_memory_model", HERE / "memory_model.py")
        physics = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(physics)
        rng = np.random.default_rng(9062047)
        n = 11
        matrices = {k: rng.normal(size=(4, n, n)) for k in ("V", "rho", "Kstar")}
        matrices["V"] *= .16
        matrices["rho"] *= .12
        matrices["Kstar"] *= .02
        vectors = dict(alpha=rng.normal(size=(4, n)),
                       gate=rng.uniform(.05, .95, size=(4, n)),
                       pdf=rng.uniform(.01, .4, size=(4, n)))
        arrays = {k: np.asarray(v, dtype=np.float64) for k, v in {**matrices, **vectors}.items()}
        initial = rng.normal(size=(n, n)) * .07
        base = physics.make_model(torch)
        # Nonzero fixed readout is essential: a zero head would hide gradient bugs.
        with torch.no_grad():
            base[4].weight.copy_(torch.from_numpy(rng.normal(size=(2, 8)).astype(np.float32) * .03))
            base[4].bias.copy_(torch.tensor([.007, -.011], dtype=torch.float32))
        frozen = {k: v.detach().numpy().astype(np.float64) for k, v in base.state_dict().items()}
        max_reference, max_output, max_gradient = 0., 0., 0.
        by_arm = []
        for memory in (False, True):
            cpu = physics.make_model(torch).double()
            cpu.load_state_dict({k: torch.from_numpy(v) for k, v in frozen.items()})
            mps = physics.make_model(torch).to("mps")
            mps.load_state_dict({k: torch.from_numpy(v.astype(np.float32)).to("mps") for k, v in frozen.items()})
            inputs_cpu = {k: torch.from_numpy(v) for k, v in arrays.items()}
            inputs_mps = {k: torch.from_numpy(v.astype(np.float32)).to("mps") for k, v in arrays.items()}
            hp, hc, hm = initial.copy(), torch.from_numpy(initial.copy()), torch.from_numpy(initial.astype(np.float32)).to("mps")
            loss_c, loss_m = 0., 0.
            arm_reference, arm_output = 0., 0.
            for l in range(4):
                yp, hp = physics.step_numpy(np, frozen, hp, *(arrays[k][l] for k in
                       ("V", "rho", "Kstar", "alpha", "gate", "pdf")), l, memory=memory)
                yc, hc = physics.step(torch, cpu, hc, *(inputs_cpu[k][l] for k in
                       ("V", "rho", "Kstar", "alpha", "gate", "pdf")), l, memory=memory)
                ym, hm = physics.step(torch, mps, hm, *(inputs_mps[k][l] for k in
                       ("V", "rho", "Kstar", "alpha", "gate", "pdf")), l, memory=memory)
                loss_c = loss_c + (yc * yc).mean() + .01 * (hc * hc).mean()
                loss_m = loss_m + (ym * ym).mean() + .01 * (hm * hm).mean()
                for a, b in ((yp, yc.detach().numpy()), (hp, hc.detach().numpy())):
                    arm_reference = max(arm_reference, float(np.max(np.abs(a-b))))
                for a, b in ((yc, ym), (hc, hm)):
                    delta = a.detach().numpy()-b.detach().cpu().numpy().astype(np.float64)
                    arm_output = max(arm_output, float(np.max(np.abs(delta))))
            loss_c.backward()
            loss_m.backward()
            grad_c = np.concatenate([p.grad.detach().numpy().reshape(-1) for p in cpu.parameters()])
            grad_m = np.concatenate([p.grad.detach().cpu().numpy().reshape(-1) for p in mps.parameters()])
            grad_error = float(np.linalg.norm(grad_c-grad_m) / max(np.linalg.norm(grad_c), 1e-12))
            check(arm_reference < 1e-11 and arm_output < 2e-5 and grad_error < 2e-4,
                  f"Numerical model gate failed: {memory, arm_reference, arm_output, grad_error}")
            max_reference = max(max_reference, arm_reference)
            max_output = max(max_output, arm_output)
            max_gradient = max(max_gradient, grad_error)
            by_arm.append(dict(memory=memory, reference_abs=arm_reference,
                               mps_abs=arm_output, gradient_relative=grad_error))
            del cpu, mps, inputs_cpu, inputs_mps, hc, hm, yc, ym, loss_c, loss_m
        torch.mps.synchronize()
        check(sha(HERE / "memory_model.py") == result["model_sha256"], "Model source changed")
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        check(peak < 2 * 1024**3, "Tiny qualification exceeded2GiB host RSS")
        result.update(passed=True, arms=by_arm, max_numpy_reference_abs=max_reference,
                      max_mps_abs=max_output, max_gradient_relative=max_gradient,
                      torch_version=torch.__version__, numpy_version=np.__version__,
                      mps_driver_bytes=int(torch.mps.driver_allocated_memory()), peak_rss_bytes=peak)
    except BaseException:
        result["error"] = traceback.format_exc()
    finally:
        signal.alarm(0)
    result["elapsed_s"] = time.perf_counter()-start
    write(outdir / "result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        raise SystemExit(run())
    parser.print_help()
