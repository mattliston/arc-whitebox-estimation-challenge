"""Post-fit sealed predictions through exact current-bank local consumers."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
PARENT = NEW / "candidate_response_pair_polynomial_f32_v1.py"
PARENT_SHA = "0b84ab6fa96d32804c9adec350d41766e47d09a71a936b95652893bb1e364a96"
HELPER = NEW / "check_response_rows_adjusted_v1.py"
HELPER_SHA = "8e99890bdc7742164a53303a23691e9f1cc54909c9cccc9cf64fc362eb459304"
RANKS = (35, 100, 150, 200, 250, 300, 400, 500)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def execute():
    if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:
        raise RuntimeError("Frozen helper changed")
    h = load_module("current_memory_consumer_helpers", HELPER)
    h.require(Path(sys.prefix).resolve() == h.GATE_ENV.resolve()
              and "anaconda3" not in sys.executable and "torch" not in sys.modules, "OFFPY only")
    h.require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "nice10 required")
    h.require(os.environ.get("PYTHONDONTWRITEBYTECODE") == "1" and all(os.environ.get(k) == "1" for k in
              ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")), "Launch controls required")
    h.require(h.digest(PARENT) == PARENT_SHA, "Current candidate changed")
    source_bindings = {str(p): h.binding(p) for p in (PARENT, HELPER, Path(__file__), HERE / "CONSUMER_PROTOCOL.md")}
    training_dir = HERE / "training"
    train_completion = h.load_json(training_dir / "completion.json")
    h.require(train_completion["passed"] and train_completion["result_sha256"] == h.digest(training_dir / "result.json"),
              "Training seal missing")
    train_result = h.load_json(training_dir / "result.json")
    h.require(train_result["passed"] and train_result["completed"], "Fixed training did not finish")
    model_seals = h.load_json(training_dir / "models_sealed.json")
    h.require(model_seals["models"] == train_result["models"] and model_seals["sources"] == train_result["sources"],
              "Model sealing inconsistent")
    for k, bound in model_seals["models"].items():
        h.require(h.digest(bound["path"]) == bound["sha256"], f"Model changed:{k}")
    for path, expected_sha in model_seals["sources"].items():
        source_bindings[path] = h.binding(Path(path))
        h.require(source_bindings[path]["sha256"] == expected_sha, "A training/model source changed")
    predictions_dir = HERE / "predictions"
    prediction_completion = h.load_json(predictions_dir / "completion.json")
    h.require(prediction_completion["passed"] and
              prediction_completion["result_sha256"] == h.digest(predictions_dir / "result.json"),
              "Prediction qualification seal missing")
    prediction_result = h.load_json(predictions_dir / "result.json")
    h.require(prediction_result["passed"] and prediction_result["completed"], "Prediction evaluation incomplete")
    for path, bound in prediction_result["source_bindings"].items():
        actual = h.binding(Path(path))
        h.require(actual["sha256"] == bound["sha256"], "Prediction evaluator/source changed")
        source_bindings[path] = actual
    for name in ("result.json", "models_sealed.json"):
        bound = prediction_result["input_bindings"][f"training/{name}"]
        h.require(Path(bound["path"]).resolve() == (training_dir/name).resolve() and
                  bound["sha256"] == h.digest(training_dir/name), "Prediction training lineage mismatch")
    for arm, declared in model_seals["models"].items():
        bound = prediction_result["models"][arm]
        h.require(Path(bound["path"]).resolve() == Path(declared["path"]).resolve()
                  and bound["sha256"] == declared["sha256"], "Prediction model lineage mismatch")
    h.require(prediction_result["predictions_sealed_sha256"] == h.digest(predictions_dir / "predictions_sealed.json"),
              "Prediction manifest seal changed")
    initial_bindings = {str(p): h.binding(p) for p in
                       (training_dir / "result.json", training_dir / "models_sealed.json",
                        training_dir / "completion.json", predictions_dir / "result.json",
                        predictions_dir / "completion.json", predictions_dir / "predictions_sealed.json")}
    for bound in model_seals["models"].values():
        initial_bindings[bound["path"]] = h.binding(Path(bound["path"]))
    outdir = HERE / "consumers"
    outdir.mkdir(exist_ok=False)
    result = dict(passed=False, completed=False, teacher_forced=True,
                  raw_or_adjusted_score_evidence=False, no_fit=True, truth_access=False,
                  source_bindings=source_bindings, sealed_input_bindings=initial_bindings,
                  learned_models=model_seals["models"])
    h.write_json(outdir / "started.json", dict(**result, utc=h.utc(), pid=os.getpid()))
    start = time.perf_counter()

    def alarm(signum, frame):
        raise TimeoutError("Fixed600s consumer cap reached")

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(600)
    all_metrics, inputs, artifacts, native_bills = [], {}, {}, []
    try:
        import numpy as np
        import flopscope as flops
        from threadpoolctl import threadpool_info, threadpool_limits
        flops.configure(symmetry_warnings=False)
        module = load_module("current_memory_consumer_candidate", PARENT)
        fnp = module.fnp

        def resources():
            h.require(h.peak_rss_bytes() < 4*1024**3, "Consumer RSS exceeded4GiB")
            h.require(time.perf_counter()-start < 600, "Consumer wall cap reached")

        def bind_file(path, expected_sha=None):
            binding = h.binding(path)
            if expected_sha is not None:
                h.require(binding["sha256"] == expected_sha, "Input changed")
            inputs[str(path)] = binding

        def replay(l, pre, aux_native, head, w, plane=None):
            mu = fnp.asarray(pre["mu"], dtype=fnp.float64)
            cpre = fnp.asarray(pre["C"], dtype=fnp.float64)
            vv = fnp.diag(cpre)
            radial = fnp.asarray(pre["incoming_radial"], dtype=fnp.float64)
            k4 = radial * vv * vv
            k22 = radial * (fnp.outer(vv, vv) + 2*cpre*cpre) / 3
            if plane is None:
                k21 = fnp.asarray(pre["K21_loss_only"], dtype=fnp.float64)
            else:
                ss = fnp.asarray(aux_native["sigma"], dtype=fnp.float64)
                k21 = fnp.asarray(plane, dtype=fnp.float32).astype(fnp.float64) * (ss*ss)[:, None] * ss[None, :]
            m, cov, unused_target, unused_radial, aux = module.nonlinear(mu, cpre, k21, k4, k22, "covariance")
            xx = module.features(l, m, cov, k21, aux, w)
            previous = fnp.asarray(head["state_previous_consumer_only"], dtype=fnp.float32)
            dm, state_unused = module.gnn_step(l, xx, aux["sigma"], w, previous, module._GNN_PARAMS)
            if plane is None:
                h.require(np.asarray(xx).shape == head["features_consumer_only"].shape and
                          np.asarray(xx).dtype == head["features_consumer_only"].dtype and
                          np.asarray(xx).tobytes() == head["features_consumer_only"].tobytes(),
                          f"Teacher current features differ at{l}")
                h.require(np.asarray(dm).shape == head["delta_consumer_only"].shape and
                          np.asarray(dm).dtype == head["delta_consumer_only"].dtype and
                          np.asarray(dm).tobytes() == head["delta_consumer_only"].tobytes(),
                          f"Teacher current delta differs at{l}")
            dm = fnp.asarray(dm, dtype=fnp.float64)
            cov = cov - fnp.outer(m, dm) - fnp.outer(dm, m) - fnp.outer(dm, dm)
            m = m + dm
            alive = fnp.asarray(pre["alive"], dtype=fnp.bool_)
            live = alive[:, None] * alive[None, :]
            diagonal = fnp.diag(cov).copy()
            cov = cov * live
            fnp.fill_diagonal(cov, diagonal)
            mm, cc = np.asarray(m).copy(), np.asarray(cov).copy()
            h.require(np.isfinite(mm).all() and np.isfinite(cc).all(), "Nonfinite consumer")
            return mm, cc

        with threadpool_limits(limits=1):
            pools = threadpool_info()
            h.require(all(p["num_threads"] == 1 for p in pools if p.get("user_api") == "blas"), "BLAS controls failed")
            result["threadpools"] = pools
            for rank in RANKS:
                rowdir = HERE / "rows" / f"row_{rank:04d}"
                completion = h.load_json(rowdir / "completion.json")
                h.require(completion["passed"] and completion["result_sha256"] == h.digest(rowdir / "result.json"),
                          "Capture result changed")
                capture = h.load_json(rowdir / "result.json")
                h.require(capture["passed"] and capture["archive_output_and_bill_exact"] and
                          capture["source_bindings"][str(PARENT)]["sha256"] == PARENT_SHA, "Wrong capture frame")
                lineage = prediction_result["input_bindings"][f"capture/{rank}/result"]
                h.require(Path(lineage["path"]).resolve() == (rowdir/"result.json").resolve() and
                          lineage["sha256"] == completion["result_sha256"], "Prediction capture lineage mismatch")
                bind_file(rowdir / "result.json", completion["result_sha256"])
                weights_bound = capture["fixture_bindings"][f"weights_{rank}"]
                bind_file(Path(weights_bound["path"]), weights_bound["sha256"])
                weights = np.load(weights_bound["path"], mmap_mode="r", allow_pickle=False)
                predictions = {}
                for arm in ("control", "memory"):
                    name = f"row_{rank:04d}_{arm}.npz"
                    # Evaluation publishes a flat artifact table keyed by filename.
                    bound = prediction_result["artifacts"][name]
                    path = predictions_dir / name
                    bind_file(path, bound["sha256"])
                    with np.load(path, allow_pickle=False) as z:
                        predictions[arm] = z["Dpre"].copy()
                    h.require(predictions[arm].shape == (15,1024,1024) and predictions[arm].dtype == np.float32,
                              "Wrong frozen prediction shape/dtype")
                t0 = time.perf_counter()
                row_metrics = []
                with flops.BudgetContext(flop_budget=2**41, wall_time_limit_s=120, quiet=True) as ctx:
                    if module._GNN_PARAMS is None:
                        module._GNN_PARAMS = {k: fnp.asarray(v, dtype=fnp.float32) for k,v in module._GNN_LITERAL.items()}
                    for l in range(1,15):
                        loaded = {}
                        for kind in ("pre", "aux", "head", "post"):
                            name = f"{kind}_{l:02d}.npz"
                            bind_file(rowdir / name, capture["artifacts"][name]["sha256"])
                            with np.load(rowdir / name, allow_pickle=False) as z:
                                wanted = {"pre": ("mu","C","K21_loss_only","incoming_radial","alive"),
                                          "aux": ("sigma",), "head": ("state_previous_consumer_only","features_consumer_only","delta_consumer_only"),
                                          "post": ("m","C")}[kind]
                                loaded[kind] = {k: z[k].copy() for k in wanted}
                        teacher_m, teacher_c = replay(l, loaded["pre"], loaded["aux"], loaded["head"], weights[l])
                        h.require(teacher_m.dtype == loaded["post"]["m"].dtype and
                                  teacher_m.tobytes() == loaded["post"]["m"].tobytes() and
                                  teacher_c.dtype == loaded["post"]["C"].dtype and
                                  teacher_c.tobytes() == loaded["post"]["C"].tobytes(),
                                  f"Teacher mean/covariance byte parity failed at{rank}:{l}")
                        next_w = weights[l+1].astype(np.float64)
                        for arm in ("control", "memory"):
                            mm, cc = replay(l, loaded["pre"], loaded["aux"], loaded["head"], weights[l], predictions[arm][l])
                            delta_m, delta_c = mm-teacher_m, cc-teacher_c
                            variance_response = np.sum(next_w*(delta_c @ next_w), axis=0, dtype=np.float64)
                            h.require(np.isfinite(variance_response).all(), "Nonfinite next-variance consumer")
                            row_metrics.append(dict(rank=rank, layer=l, arm=arm,
                                mean_discrepancy_mse=float(np.mean(delta_m*delta_m)),
                                next_variance_discrepancy_mse=float(np.mean(variance_response*variance_response)),
                                covariance_discrepancy_mse=float(np.mean(delta_c*delta_c)),
                                minimum_post_variance=float(np.min(np.diag(cc))),
                                negative_post_variances=int((np.diag(cc)<0).sum())))
                        resources()
                native_bills.append(dict(rank=rank, flops=int(ctx.flops_used), wall_s=time.perf_counter()-t0,
                           residual_s=float(ctx.residual_wall_time_s), diagnostic_only=True))
                all_metrics.extend(row_metrics)
                path = outdir / f"row_{rank:04d}.json"
                h.write_json(path, dict(rank=rank, teacher_byte_parity=True, metrics=row_metrics))
                artifacts[path.name] = h.binding(path)
                print(json.dumps(dict(rank=rank, teacher_byte_parity=True,
                                      local_consumer_done=True, wall_s=time.perf_counter()-t0)), flush=True)
        summaries = []
        for rank in RANKS:
            summary = dict(rank=rank, heldout=rank in (400,500))
            for metric in ("mean_discrepancy_mse", "next_variance_discrepancy_mse", "covariance_discrepancy_mse"):
                vals = {arm: float(np.mean([x[metric] for x in all_metrics if x["rank"]==rank and x["arm"]==arm]))
                        for arm in ("control","memory")}
                summary[metric] = dict(**vals, memory_over_control=(vals["memory"]/vals["control"] if vals["control"]>1e-30 else None))
            summaries.append(summary)
        h.check_bindings(source_bindings)
        h.check_bindings(initial_bindings)
        h.check_bindings(inputs)
        h.check_bindings(artifacts)
        resources()
        result.update(passed=True, completed=True, every_teacher_replay_byte_exact=True,
                      summaries=summaries, layer_metrics=all_metrics, native_diagnostic_bills=native_bills,
                      fp64_next_variance_arithmetic_subtotal=8*14*2*(4*1024**3+2*1024**2-2*1024))
    except BaseException:
        result["error"] = traceback.format_exc()
    finally:
        signal.alarm(0)
    result.update(artifacts=artifacts, input_bindings=inputs, elapsed_s=time.perf_counter()-start,
                  peak_rss_bytes=h.peak_rss_bytes(), ended_utc=h.utc())
    result_sha = h.write_json(outdir / "result.json", result)
    h.write_json(outdir / "completion.json", dict(passed=result["passed"], result_sha256=result_sha))
    print(json.dumps({k: result.get(k) for k in ("passed","completed","elapsed_s","peak_rss_bytes","summaries","error")}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        raise SystemExit(execute())
    parser.print_help()
