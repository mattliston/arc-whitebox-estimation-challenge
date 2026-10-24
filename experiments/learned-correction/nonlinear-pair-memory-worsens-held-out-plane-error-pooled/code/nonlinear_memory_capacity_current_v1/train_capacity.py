"""Fixed matched two-memory capacity fit; inert without explicit mode."""

import argparse
import gc
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
TRAIN = (35, 100, 150, 200, 250, 300)
HELDOUT = (400, 500)
FEATURES = ("V", "rho", "Kstar", "alpha", "gate", "pdf")
MODEL = HERE / "memory_model.py"
PROTOCOL = HERE / "TRAINING_PROTOCOL.md"
PREPARER = HERE / "prepare_rows.py"
SMALL_RESULT = HERE / "numerical_qualification/result.json"
SMALL_SCRIPT = HERE / "qualify_model.py"
SMALL_RESULT_SHA = "18891a191cbfeed98e42408c29b2d1f0daaab1b28f7863905a5970ee1d92f234"


def check(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return digest(path)


def load_json(path):
    return json.loads(Path(path).read_text())


def load_physics():
    spec = importlib.util.spec_from_file_location("fixed_capacity_model", MODEL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute(mode):
    check("anaconda3" in sys.executable and "threadpoolctl" not in sys.modules,
          "AP/Torch only, no threadpoolctl")
    check(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "nice10 required")
    check(os.environ.get("PYTHONDONTWRITEBYTECODE") == "1", "No bytecode writes")
    check(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0", "MPS fallback must be disabled")
    sources = {str(p): digest(p) for p in (MODEL, PROTOCOL, PREPARER, SMALL_SCRIPT, SMALL_RESULT, Path(__file__))}
    numerical = load_json(SMALL_RESULT)
    check(sources[str(SMALL_RESULT)] == SMALL_RESULT_SHA and numerical["passed"]
          and numerical["model_sha256"] == sources[str(MODEL)]
          and numerical["script_sha256"] == sources[str(SMALL_SCRIPT)],
          "Small numerical qualification missing")
    if mode == "train":
        qualification_path = HERE / "wide_qualification/result.json"
        qualification_completion = load_json(HERE / "wide_qualification/completion.json")
        check(qualification_completion["passed"] and
              qualification_completion["result_sha256"] == digest(qualification_path),
              "Full-width qualification receipt seal changed")
        qualification = load_json(qualification_path)
        check(qualification["passed"] and qualification["sources"] == sources,
              "Exact-source full-width resource qualification missing")
    directory = HERE / ("wide_qualification" if mode == "qualify" else "training")
    directory.mkdir(exist_ok=False)
    result = dict(passed=False, completed=False, mode=mode, sources=sources,
                  teacher_forced=True, raw_score_evidence=False, truth_access=False,
                  heldout_access_before_seal=False)
    write_json(directory / "started.json", dict(**result, pid=os.getpid(), began_unix=time.time()))
    start = time.perf_counter()
    wall_cap = 240 if mode == "qualify" else 3600

    def alarm(signum, frame):
        raise TimeoutError(f"Fixed {wall_cap}s wall cap reached")

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(wall_cap)
    try:
        import numpy as np
        import torch
        from torch.utils.checkpoint import checkpoint
        check(torch.backends.mps.is_available() and "threadpoolctl" not in sys.modules,
              "MPS unavailable or forbidden import")
        torch.set_num_threads(1)
        physics = load_physics()
        prepared_bindings = {}

        def resource_check():
            torch.mps.synchronize()
            peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            driver = int(torch.mps.driver_allocated_memory())
            check(peak < 6*1024**3, "Host peak RSS exceeded6GiB")
            check(driver < 4*1024**3, "MPS driver allocation exceeded4GiB")
            check(time.perf_counter()-start < wall_cap, "Fixed wall cap reached")
            return peak, driver

        def load_row(rank):
            path = HERE / "prepared" / f"row_{rank:04d}.npz"
            receipt_path = HERE / "prepared" / f"row_{rank:04d}.json"
            completion = load_json(HERE / "prepared" / f"row_{rank:04d}.completion.json")
            check(completion["passed"] and completion["result_sha256"] == digest(receipt_path),
                  "Prepared receipt seal changed")
            receipt = load_json(receipt_path)
            check(receipt["passed"] and receipt["completed"] and receipt["rank"] == rank
                  and receipt["prepare_sha256"] == sources[str(PREPARER)]
                  and receipt["capture_sha256"] == "3950d1b43e62b660093bd1140d5e9662b3e6f64a979b66807c1d9aed32fb918d",
                  f"Unqualified prepared rank{rank}")
            # Preparation's receipt exposes one unambiguous artifact binding.
            expected = receipt["output"]["sha256"]
            check(Path(receipt["output"]["path"]).resolve() == path.resolve()
                  and digest(path) == expected, "Prepared payload changed")
            prepared_bindings[str(rank)] = dict(path=str(path), sha256=expected,
                            receipt_path=str(receipt_path), receipt_sha256=digest(receipt_path),
                            completion_path=str(HERE / "prepared" / f"row_{rank:04d}.completion.json"),
                            completion_sha256=digest(HERE / "prepared" / f"row_{rank:04d}.completion.json"))
            with np.load(path, allow_pickle=False) as z:
                check(set(z.files) == set(FEATURES) | {"target_loss_only"}, "Unexpected prepared field")
                row = {k: torch.from_numpy(z[k].copy()) for k in z.files}
            for k, value in row.items():
                shape = (15, 1024, 1024) if k in ("V", "rho", "Kstar", "target_loss_only") else (15, 1024)
                check(tuple(value.shape) == shape and value.dtype == torch.float32
                      and bool(torch.isfinite(value).all()), "Bad prepared array")
            return row

        ranks = (35,) if mode == "qualify" else TRAIN
        rows = {rank: load_row(rank) for rank in ranks}
        count, energy = 0, 0.
        for row in rows.values():
            d = row["target_loss_only"][1:].double() - row["Kstar"][1:].double()
            energy += float((d*d).sum())
            count += d.numel()
            del d
        normalizer = max(energy/count, 1e-12)
        result["train_only_zero_energy_normalizer"] = normalizer

        def objective(model, row, memory):
            # Each block freezes its layer argument for checkpoint recomputation.
            hstate = torch.zeros_like(row["Kstar"][0])
            loss = torch.zeros((), device="mps")
            for l in range(15):
                def block(previous, V, rho, Kstar, alpha, gate, pdf, target, layer=l):
                    prediction, new_state = physics.step(torch, model, previous, V, rho,
                              Kstar, alpha, gate, pdf, layer, memory=memory)
                    current = ((prediction-target)*(prediction-target)).mean() / (normalizer*14) if layer else prediction.sum()*0
                    return new_state, current
                hstate, part = checkpoint(block, hstate,
                          *(row[k][l] for k in FEATURES), row["target_loss_only"][l],
                          use_reentrant=False, preserve_rng_state=False)
                loss = loss + part
            return loss

        model_seals, logs = {}, []
        epochs = 1 if mode == "qualify" else 100
        for arm, memory in (("control", False), ("memory", True)):
            model = physics.make_model(torch).to("mps")
            optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9,.999), eps=1e-8, weight_decay=0)
            for epoch in range(1, epochs+1):
                epoch_start, total_loss = time.perf_counter(), 0.
                for rank in ranks:
                    device_row = {k: v.to("mps") for k, v in rows[rank].items()}
                    optimizer.zero_grad(set_to_none=True)
                    loss = objective(model, device_row, memory)
                    check(bool(torch.isfinite(loss)), "Nonfinite loss")
                    loss.backward()
                    check(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                              for p in model.parameters()), "Nonfinite/missing gradient")
                    if mode == "train":
                        optimizer.step()
                    check(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "Nonfinite parameter")
                    total_loss += float(loss.detach().cpu())
                    del loss, device_row
                    peak, driver = resource_check()
                entry = dict(arm=arm, epoch=epoch, normalized_train_loss=total_loss/len(ranks),
                             epoch_s=time.perf_counter()-epoch_start,
                             elapsed_s=time.perf_counter()-start, peak_rss_bytes=peak, mps_driver_bytes=driver)
                logs.append(entry)
                print(json.dumps(entry), flush=True)
                if mode == "train" and epoch in (25, 50, 75, 100):
                    path = directory / f"{arm}_epoch{epoch:03d}.npz"
                    with path.open("xb") as stream:
                        np.savez(stream, **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
                    if epoch == 100:
                        model_seals[arm] = dict(path=str(path), sha256=digest(path))
            del optimizer, model
            gc.collect()
            torch.mps.empty_cache()
        resource_check()
        # No held-out array is decoded in this program, including after fitting.
        if mode == "train":
            write_json(directory / "models_sealed.json", dict(models=model_seals,
                       sources=sources, prepared_bindings=prepared_bindings,
                       normalizer=normalizer, heldout_access=False))
        check(all(digest(p) == v for p, v in sources.items()), "Source changed during execution")
        for bound in prepared_bindings.values():
            check(digest(bound["path"]) == bound["sha256"] and
                  digest(bound["receipt_path"]) == bound["receipt_sha256"] and
                  digest(bound["completion_path"]) == bound["completion_sha256"],
                  "Prepared input or receipt changed during execution")
        result.update(passed=True, completed=True, logs=logs, models=model_seals,
                      prepared_bindings=prepared_bindings, torch_version=torch.__version__,
                      peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    except BaseException:
        result["error"] = traceback.format_exc()
        if "logs" in locals():
            result["logs"] = logs
    finally:
        signal.alarm(0)
    result["elapsed_s"] = time.perf_counter()-start
    seal = write_json(directory / "result.json", result)
    write_json(directory / "completion.json", dict(passed=result["passed"], result_sha256=seal))
    print(json.dumps({k: result.get(k) for k in ("mode", "passed", "elapsed_s", "peak_rss_bytes", "error")}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("qualify", "train"))
    args = parser.parse_args()
    if args.mode:
        raise SystemExit(execute(args.mode))
    parser.print_help()
