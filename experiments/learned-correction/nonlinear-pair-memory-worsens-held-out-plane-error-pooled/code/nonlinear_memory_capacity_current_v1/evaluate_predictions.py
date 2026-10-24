"""Inert final-model capacity evaluation; no optimizer, bank/head or MC imports.

Both final model seals are authenticated before any prepared row is decoded.
Every prediction is saved and sealed before current action labels are decoded.
Reported errors concern teacher-forced standardized planes, not raw/adjusted score.
"""

import argparse
import gc
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback


SELF = Path(__file__).resolve()
HERE = SELF.parent
MODEL = HERE / "memory_model.py"
TRAINER = HERE / "train_capacity.py"
PREPARER = HERE / "prepare_rows.py"
PROTOCOL = HERE / "TRAINING_PROTOCOL.md"
SMALL_SCRIPT = HERE / "qualify_model.py"
SMALL_RESULT = HERE / "numerical_qualification/result.json"
WIDE_RESULT = HERE / "wide_qualification/result.json"
WIDE_COMPLETION = HERE / "wide_qualification/completion.json"
CAPTURE = HERE / "capture_current.py"
CAPTURE_PROTOCOL = HERE / "CAPTURE_PROTOCOL.md"
PINS = {
    MODEL: "b0f43b1da4e64d8d644cf379bff98745282273457666fae6d68e516b6508831b",
    TRAINER: "f53d3c577cb8c198faccdc3c14bd4689828cef4600471194962ad32eefed9cc6",
    PREPARER: "9f34cd20dde99851426db8fb0713b7513022097cac975393cd8b3801f63e2fd7",
    PROTOCOL: "df12da9dd2179e7442e9ddcb60e9f4debbddd00bdf7042e8b36bac6dc3212cef",
    SMALL_SCRIPT: "2b1bba5c8b5e56dfc10f455918619513c312e43d330529302e9ab745379fe640",
    SMALL_RESULT: "18891a191cbfeed98e42408c29b2d1f0daaab1b28f7863905a5970ee1d92f234",
    WIDE_RESULT: "8ef4dfbe707a5f5d540929b8457d91cd4c2a0ef9aa3ac6f424b9400859abe22c",
    WIDE_COMPLETION: "9f845a226467de19375e027e304b7611970811f1367619c1a31316446f67e279",
    CAPTURE: "3950d1b43e62b660093bd1140d5e9662b3e6f64a979b66807c1d9aed32fb918d",
    CAPTURE_PROTOCOL: "d610cf7dc1000ca13c9fbb586e82e3f7fb671ea3aaca3dca427c613e105e4947",
}
TRAIN = (35, 100, 150, 200, 250, 300)
HELDOUT = (400, 500)
RANKS = TRAIN + HELDOUT
ARMS = (("control", False), ("memory", True))
FEATURES = ("V", "rho", "Kstar", "alpha", "gate", "pdf")
N, LAYERS = 1024, 15
SYNTHETIC_N, SYNTHETIC_LAYERS, SYNTHETIC_SEED = 11, 4, 9062111
SYNTHETIC_ABS_LIMIT = 2e-5
WALL_LIMIT, RSS_LIMIT, DRIVER_LIMIT = 600, 6 * 1024**3, 4 * 1024**3
SCHEMA = "nonlinear_memory_current_predictions_v1"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authenticate_training(h, sources):
    """Only sealed metadata and raw-byte hashes; no model/row array decoding."""
    directory = HERE / "training"
    paths = {name: directory / name for name in
             ("started.json", "result.json", "completion.json", "models_sealed.json")}
    inputs = {f"training/{name}": h.binding(path) for name, path in paths.items()}
    completion = h.read_json(paths["completion.json"])
    h.require(completion["passed"] and completion["result_sha256"] == inputs["training/result.json"]["sha256"],
              "Training completion/result seal mismatch")
    record = h.read_json(paths["result.json"])
    start = h.read_json(paths["started.json"])
    sealed = h.read_json(paths["models_sealed.json"])
    expected_sources = {str(path): PINS[path] for path in
                        (MODEL, PROTOCOL, PREPARER, SMALL_SCRIPT, SMALL_RESULT, TRAINER)}
    h.require(record["passed"] and record["completed"] and record["mode"] == "train" and
              record["teacher_forced"] and record["raw_score_evidence"] is False and
              record["truth_access"] is False and record["heldout_access_before_seal"] is False,
              "Both fixed training arms must finish successfully before evaluation")
    h.require(record["sources"] == start["sources"] == sealed["sources"] == expected_sources,
              "Wrong training source frame")
    h.require(record["models"] == sealed["models"] and set(record["models"]) == {arm for arm, _ in ARMS} and
              record["prepared_bindings"] == sealed["prepared_bindings"] and
              set(record["prepared_bindings"]) == {str(rank) for rank in TRAIN} and
              sealed["heldout_access"] is False, "Final-model ownership mismatch")
    normalizer = record["train_only_zero_energy_normalizer"]
    h.require(math.isfinite(normalizer) and normalizer >= 1e-12 and sealed["normalizer"] == normalizer,
              "Wrong TRAIN-only normalizer")
    h.require([(item["arm"], item["epoch"]) for item in record["logs"]] ==
              [(arm, epoch) for arm, _ in ARMS for epoch in range(1, 101)],
              "Both exact100-epoch arms were not completed")
    h.require(record["elapsed_s"] < 3600 and record["peak_rss_bytes"] < RSS_LIMIT and
              all(math.isfinite(item["normalized_train_loss"]) and
                  item["peak_rss_bytes"] < RSS_LIMIT and item["mps_driver_bytes"] < DRIVER_LIMIT
                  for item in record["logs"]), "Training numerical/resource receipt failed")
    model_bindings = {}
    for arm, _ in ARMS:
        path = directory / f"{arm}_epoch100.npz"
        declared = record["models"][arm]
        h.require(Path(declared["path"]).resolve() == path.resolve(), "Non-final checkpoint requested")
        model_bindings[arm] = h.binding(path)
        h.require(model_bindings[arm]["sha256"] == declared["sha256"], "Epoch100 model changed")
        inputs[f"model/{arm}"] = model_bindings[arm]
    small = h.read_json(SMALL_RESULT)
    wide = h.read_json(WIDE_RESULT)
    wide_done = h.read_json(WIDE_COMPLETION)
    h.require(small["passed"] and small["model_sha256"] == PINS[MODEL] and
              small["script_sha256"] == PINS[SMALL_SCRIPT] and wide["passed"] and
              wide["completed"] and wide["sources"] == expected_sources and
              wide_done["passed"] and wide_done["result_sha256"] == PINS[WIDE_RESULT],
              "Numerical/wide qualification frame mismatch")
    return record, model_bindings, inputs


def authenticate_prepared(h, rank, training, sources, inputs):
    """Called only after both epoch100 models have been authenticated."""
    base = HERE / "prepared" / f"row_{rank:04d}"
    paths = {name: Path(str(base) + suffix) for name, suffix in
             (("receipt", ".json"), ("completion", ".completion.json"), ("causal", ".causal.json"), ("array", ".npz"))}
    bound = {name: h.binding(path) for name, path in paths.items()}
    done = h.read_json(paths["completion"])
    h.require(done["passed"] and done["rank"] == rank and
              done["result_sha256"] == bound["receipt"]["sha256"], "Prepared completion/receipt mismatch")
    receipt = h.read_json(paths["receipt"])
    h.require(receipt["schema"] == "nonlinear_memory_current_prepared_v1" and
              receipt["passed"] and receipt["completed"] and receipt["rank"] == rank and
              receipt["prepare_sha256"] == PINS[PREPARER] and receipt["capture_sha256"] == PINS[CAPTURE] and
              receipt["source_hashes_unchanged"] and receipt["input_hashes_unchanged"] and
              receipt["predictor_fields_unchanged_after_labels"] and
              receipt["mc_truth_files_accessed"] is False and receipt["head_files_accessed"] is False,
              "Prepared row is not qualified current-frame data")
    h.require(receipt["output"]["sha256"] == bound["array"]["sha256"] and
              Path(receipt["output"]["path"]).resolve() == paths["array"].resolve(), "Prepared output changed")
    h.require(receipt["causal_predictors_seal_sha256"] == bound["causal"]["sha256"],
              "Causal predictor seal changed")
    causal = h.read_json(paths["causal"])
    h.require(causal["rank"] == rank and causal["current_teacher_labels_decoded"] is False and
              causal["prepare_sha256"] == PINS[PREPARER] and
              set(causal["predictors"]) == set(FEATURES) and
              all(causal["predictors"][key] == receipt["array_contract"][key] for key in FEATURES),
              "Predictor fields lack the pre-label seal")
    for path, expected in {**h.PINS, PREPARER: PINS[PREPARER]}.items():
        item = receipt["source_bindings"][str(path)]
        h.require(item["sha256"] == expected, "Prepared source frame changed")
        sources.setdefault(str(path), h.authenticate(item, path))
    capture_dir = HERE / "rows" / f"row_{rank:04d}"
    capture_result_path, capture_completion_path = capture_dir / "result.json", capture_dir / "completion.json"
    capture_bound = h.binding(capture_result_path)
    capture_done_bound = h.binding(capture_completion_path)
    capture_done = h.read_json(capture_completion_path)
    h.require(capture_done["passed"] and capture_done["result_sha256"] == capture_bound["sha256"] ==
              receipt["capture_result_sha256"] == causal["capture_result_sha256"], "Capture-frame seal changed")
    capture = h.read_json(capture_result_path)
    h.require(capture["passed"] and capture["completed"] and capture["rank"] == rank and
              capture["capture_sha256"] == PINS[CAPTURE] and capture["protocol_sha256"] == PINS[CAPTURE_PROTOCOL] and
              capture["target_access"] is False and capture["archive_output_and_bill_exact"] and
              capture["same_source_control_exact"] and
              capture["source_bindings"][str(h.CANDIDATE)]["sha256"] == h.PINS[h.CANDIDATE],
              "Prepared teacher is not the qualified current0b84 capture")
    h.require(receipt["input_bindings"][f"fixture/weights_{rank}"]["sha256"] ==
              capture["fixture_bindings"][f"weights_{rank}"]["sha256"], "Teacher weight ownership differs")
    if rank in TRAIN:
        trained = training["prepared_bindings"][str(rank)]
        h.require(trained["sha256"] == bound["array"]["sha256"] and
                  trained["receipt_sha256"] == bound["receipt"]["sha256"] and
                  trained["completion_sha256"] == bound["completion"]["sha256"], "TRAIN data differ from fit")
    for name, item in bound.items():
        inputs[f"prepared/{rank}/{name}"] = item
    inputs[f"capture/{rank}/result"] = capture_bound
    inputs[f"capture/{rank}/completion"] = capture_done_bound
    return receipt, paths["array"]


def execute():
    timer = time.perf_counter()
    if hashlib.sha256(PREPARER.read_bytes()).hexdigest() != PINS[PREPARER]:
        raise RuntimeError("Frozen preparation helper changed")
    # This module has only standard-library top-level imports; its execute is never called.
    h = load_module("capacity_evaluation_seal_helpers", PREPARER)
    h.require("anaconda3" in sys.executable and
              not any(name == "threadpoolctl" or name.startswith("threadpoolctl.") for name in sys.modules),
              "AP/Torch only; threadpoolctl is forbidden")
    h.require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "nice10 required")
    h.require(os.environ.get("PYTHONDONTWRITEBYTECODE") == "1" and
              os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0", "Launch controls missing")
    source_pins = dict(PINS)
    for path, expected in h.PINS.items():
        h.require(path not in source_pins or source_pins[path] == expected, "Inconsistent source pin")
        source_pins[path] = expected
    sources = {str(path): h.binding(path) for path in (*source_pins, SELF)}
    h.require(all(sources[str(path)]["sha256"] == expected for path, expected in source_pins.items()),
              "Frozen evaluation source/qualification changed")
    training, model_bindings, inputs = authenticate_training(h, sources)
    directory = HERE / "predictions"
    directory.mkdir(exist_ok=False)
    result = dict(schema=SCHEMA, passed=False, completed=False, source_bindings=sources,
                  input_bindings=inputs, models=model_bindings, model_sha256=PINS[MODEL],
                  teacher_forced=True, raw_score_evidence=False, adjusted_score_evidence=False,
                  mc_truth_files_accessed=False, head_files_accessed=False,
                  native_candidate_imported=False, models_authenticated_before_prepared_decode=True,
                  heldout_arrays_decoded=False, current_teacher_labels_decoded=False,
                  train_normalizer_report_only=training["train_only_zero_energy_normalizer"],
                  layer0_prediction_saved=True, scored_layers=list(range(1, 15)))
    started_sha = h.write_json(directory / "started.json", dict(**result, utc=h.utc(), pid=os.getpid(),
                       nice=os.getpriority(os.PRIO_PROCESS, 0), wall_limit_s=WALL_LIMIT,
                       rss_limit_bytes=RSS_LIMIT, mps_driver_limit_bytes=DRIVER_LIMIT))
    result["started_sha256"] = started_sha
    artifacts, prediction_records, prepared_rows = {}, {}, {}
    peak_driver = 0

    def watchdog(signum, frame):
        raise TimeoutError("Fixed600s prediction/evaluation watchdog exhausted")

    signal.signal(signal.SIGALRM, watchdog)
    signal.alarm(max(1, math.ceil(WALL_LIMIT - (time.perf_counter()-timer))))
    try:
        h.require(time.perf_counter()-timer < WALL_LIMIT, "Provenance preflight exhausted600s")
        import numpy as np
        import torch
        h.require(torch.backends.mps.is_available() and "threadpoolctl" not in sys.modules, "MPS unavailable/forbidden import")
        h.require(torch.__version__ == training["torch_version"], "Torch version differs from training")
        torch.set_num_threads(1)
        physics = load_module("fixed_final_capacity_model", MODEL)

        def resources():
            nonlocal peak_driver
            torch.mps.synchronize()
            driver = int(torch.mps.driver_allocated_memory())
            peak_driver = max(peak_driver, driver)
            h.require(h.peak_rss_bytes() < RSS_LIMIT and driver < DRIVER_LIMIT, "Evaluation memory cap exceeded")
            h.require(time.perf_counter()-timer < WALL_LIMIT, "Evaluation600s cap exhausted")

        def array(value, shape, name):
            h.require(value.shape == shape and value.dtype == np.float32 and bool(np.isfinite(value).all()),
                      f"Bad FP32 array: {name}")
            return value

        def raw_record(value):
            h.require(value.flags.c_contiguous, "Noncontiguous output")
            return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
                        sha256=hashlib.sha256(memoryview(value)).hexdigest())

        def checked_feature(value, key, receipt):
            shape = (LAYERS, N, N) if key in ("V", "rho", "Kstar", "target_loss_only") else (LAYERS, N)
            array(value, shape, key)
            h.require(raw_record(value) == receipt["array_contract"][key], f"Prepared field hash differs: {key}")
            return value

        shapes = {"0.weight": (8, 16), "0.bias": (8,), "2.weight": (8, 8),
                  "2.bias": (8,), "4.weight": (2, 8), "4.bias": (2,)}
        models, params64 = {}, {}
        for arm, _ in ARMS:
            with np.load(model_bindings[arm]["path"], allow_pickle=False) as z:
                h.require(set(z.files) == set(shapes), "Unexpected final-model parameters")
                params = {key: array(z[key], shape, f"{arm}:{key}") for key, shape in shapes.items()}
            h.require(bool(np.any(params["4.weight"][0] != 0)) or params["4.bias"][0] != 0,
                      "Final learned D decoder is still identically zero")
            model = physics.make_model(torch)
            model.load_state_dict({key: torch.from_numpy(value.copy()) for key, value in params.items()}, strict=True)
            model.requires_grad_(False)
            models[arm] = model.to("mps").eval()
            params64[arm] = {key: value.astype(np.float64) for key, value in params.items()}
            del params, model
        result["both_final_models_decoded_before_prepared_rows"] = True

        # ONE fixed target-free fixture; exact FP32 inputs are promoted for NumPy64.
        rng = np.random.default_rng(SYNTHETIC_SEED)
        n, depth = SYNTHETIC_N, SYNTHETIC_LAYERS
        fixture = {key: np.zeros((depth, n, n), dtype=np.float32) for key in ("V", "rho", "Kstar")}
        fixture.update({key: np.zeros((depth, n), dtype=np.float32) for key in ("alpha", "gate", "pdf")})
        for l in range(depth):
            factor = rng.normal(size=(n, n))
            cov = factor.T @ factor + np.eye(n)
            scale = np.sqrt(np.diag(cov))
            fixture["rho"][l] = (cov / np.outer(scale, scale)).astype(np.float32)
            if l:
                fixture["V"][l] = (rng.normal(size=(n, n)) * (.4/math.sqrt(n))).astype(np.float32)
                fixture["Kstar"][l] = (rng.normal(size=(n, n)) * .05).astype(np.float32)
                fixture["alpha"][l] = rng.uniform(-2, 2, size=n).astype(np.float32)
            aa = fixture["alpha"][l].astype(np.float64)
            fixture["gate"][l] = np.array([.5*(1+math.erf(float(x)/math.sqrt(2))) for x in aa], dtype=np.float32)
            fixture["pdf"][l] = (np.exp(-aa*aa/2)/math.sqrt(2*math.pi)).astype(np.float32)
        fixture_mps = {key: torch.from_numpy(value).to("mps") for key, value in fixture.items()}
        certificate = dict(seed=SYNTHETIC_SEED, width=n, layers=depth, abs_limit=SYNTHETIC_ABS_LIMIT,
                           actual_final_models=model_bindings, inputs={key: raw_record(value) for key, value in fixture.items()}, arms=[])
        with torch.inference_mode():
            for arm, memory in ARMS:
                previous64 = np.zeros((n, n), dtype=np.float64)
                previous32 = torch.zeros((n, n), device="mps", dtype=torch.float32)
                entries = []
                for l in range(depth):
                    predicted64, previous64 = physics.step_numpy(np, params64[arm], previous64,
                        *(fixture[key][l].astype(np.float64) for key in FEATURES), l, memory=memory)
                    predicted32, previous32 = physics.step(torch, models[arm], previous32,
                        *(fixture_mps[key][l] for key in FEATURES), l, memory=memory)
                    out32, h32 = predicted32.cpu().numpy(), previous32.cpu().numpy()
                    array(out32, (n, n), f"synthetic {arm} Dpre")
                    array(h32, (n, n), f"synthetic {arm} H")
                    h.require(np.isfinite(predicted64).all() and np.isfinite(previous64).all() and
                              np.isfinite(out32).all() and np.isfinite(h32).all(), "Nonfinite synthetic learned-model output")
                    errors = dict(layer=l, Dpre_max_abs=float(np.max(np.abs(out32.astype(np.float64)-predicted64))),
                                  H_max_abs=float(np.max(np.abs(h32.astype(np.float64)-previous64))))
                    entries.append(errors)
                    resources()
                arm_passed = all(max(item["Dpre_max_abs"], item["H_max_abs"]) <= SYNTHETIC_ABS_LIMIT for item in entries)
                certificate["arms"].append(dict(arm=arm, memory=memory, passed=arm_passed, layers=entries))
            certificate["passed"] = all(item["passed"] for item in certificate["arms"])
            h.write_json(directory / "learned_model_numerical.json", certificate)
            artifacts["learned_model_numerical.json"] = h.binding(directory / "learned_model_numerical.json")
            h.require(certificate["passed"], "Fixed final-learned-model numerical qualification failed; no retry/refit")
            del fixture_mps, previous32, predicted32, previous64, predicted64, fixture, params64
            torch.mps.empty_cache()

            for rank in RANKS:
                receipt, path = authenticate_prepared(h, rank, training, sources, inputs)
                prepared_rows[rank] = (receipt, path)
                if rank in HELDOUT:
                    result["heldout_arrays_decoded"] = True
                # Deliberately do not access target_loss_only, even though it shares the archive.
                with np.load(path, allow_pickle=False) as z:
                    h.require(set(z.files) == set(FEATURES) | {"target_loss_only"}, "Unexpected prepared fields")
                    cpu = {key: checked_feature(z[key], key, receipt) for key in FEATURES}
                device = {key: torch.from_numpy(value).to("mps") for key, value in cpu.items()}
                del cpu
                prediction_records[str(rank)] = {}
                for arm, memory in ARMS:
                    previous = torch.zeros((N, N), device="mps", dtype=torch.float32)
                    output = np.empty((LAYERS, N, N), dtype=np.float32)
                    state_diagnostics = []
                    for l in range(LAYERS):
                        predicted, previous = physics.step(torch, models[arm], previous,
                                      *(device[key][l] for key in FEATURES), l, memory=memory)
                        h.require(tuple(predicted.shape) == (N, N) and tuple(previous.shape) == (N, N) and
                                  predicted.dtype == torch.float32 and previous.dtype == torch.float32 and
                                  bool(torch.isfinite(predicted).all()) and bool(torch.isfinite(previous).all()),
                                  "Nonfinite/wrong-shape learned prediction/state")
                        output[l] = predicted.cpu().numpy()
                        absolute = previous.abs()
                        state_diagnostics.append(dict(layer=l, H_abs_max=float(absolute.max().cpu()),
                            H_rms=float(torch.sqrt(torch.mean(previous*previous)).cpu()),
                            H_abs_ge_0_99_fraction=float((absolute >= .99).float().mean().cpu()),
                            H_abs_ge_0_999_fraction=float((absolute >= .999).float().mean().cpu())))
                        h.require(state_diagnostics[-1]["H_abs_max"] <= 1., "Latent tanh bound failed")
                        resources()
                    name = f"row_{rank:04d}_{arm}.npz"
                    with (directory / name).open("xb") as stream:
                        np.savez(stream, Dpre=output)
                    artifacts[name] = h.binding(directory / name)
                    prediction_records[str(rank)][arm] = dict(artifact=name, output=artifacts[name],
                        Dpre=raw_record(output), model=model_bindings[arm], H_diagnostics=state_diagnostics)
                    del output, previous, predicted, absolute
                    resources()
                del device
                gc.collect()
                torch.mps.empty_cache()
                print(f"Predictions sealed for rank {rank}; current labels not decoded", flush=True)

        prediction_seal_sha = h.write_json(directory / "predictions_sealed.json", dict(schema=SCHEMA,
            utc=h.utc(), evaluator_sha256=sources[str(SELF)]["sha256"], models=model_bindings,
            source_bindings=sources, input_bindings=inputs, predictions=prediction_records,
            current_teacher_labels_decoded=False, mc_truth_files_accessed=False))
        artifacts["predictions_sealed.json"] = h.binding(directory / "predictions_sealed.json")
        result["all_predictions_sealed_before_label_decode"] = True
        result["predictions_sealed_sha256"] = prediction_seal_sha
        result["current_teacher_labels_decoded"] = True

        def ratio(numerator, denominator):
            return numerator/denominator if denominator > 0 else None

        metrics = []
        for rank in RANKS:
            receipt, path = prepared_rows[rank]
            with np.load(path, allow_pickle=False) as z:
                target = checked_feature(z["target_loss_only"], "target_loss_only", receipt)
                zero = checked_feature(z["Kstar"], "Kstar", receipt)
            predictions = {}
            for arm, _ in ARMS:
                item = prediction_records[str(rank)][arm]
                h.authenticate(item["output"], directory / item["artifact"])
                with np.load(directory / item["artifact"], allow_pickle=False) as z:
                    h.require(set(z.files) == {"Dpre"}, "Unexpected prediction fields")
                    predictions[arm] = array(z["Dpre"], (LAYERS, N, N), arm)
                h.require(raw_record(predictions[arm]) == item["Dpre"], "Prediction values differ from seal")
            layers = []
            for l in range(1, LAYERS):
                truth = target[l].astype(np.float64)
                residual = zero[l].astype(np.float64)-truth
                base_mse = float(np.mean(residual*residual))
                beyond_range = np.maximum(np.abs(residual)-.05, 0)
                values = dict(layer=l, zero_decoder_mse=base_mse,
                              ideal_pointwise_range_floor_mse=float(np.mean(beyond_range*beyond_range)))
                for arm, _ in ARMS:
                    delta = predictions[arm][l].astype(np.float64)-truth
                    values[f"{arm}_mse"] = float(np.mean(delta*delta))
                    values[f"{arm}_over_zero"] = ratio(values[f"{arm}_mse"], base_mse)
                values["memory_over_control"] = ratio(values["memory_mse"], values["control_mse"])
                h.require(all(math.isfinite(values[key]) and values[key] >= 0 for key in
                    ("zero_decoder_mse", "control_mse", "memory_mse", "ideal_pointwise_range_floor_mse")),
                    "Nonfinite/negative plane metric")
                layers.append(values)
                resources()
            pooled = {key: sum(item[key] for item in layers)/len(layers) for key in
                      ("zero_decoder_mse", "control_mse", "memory_mse", "ideal_pointwise_range_floor_mse")}
            pooled.update(memory_over_control=ratio(pooled["memory_mse"], pooled["control_mse"]),
                          memory_over_zero=ratio(pooled["memory_mse"], pooled["zero_decoder_mse"]),
                          control_over_zero=ratio(pooled["control_mse"], pooled["zero_decoder_mse"]))
            metrics.append(dict(rank=rank, group="train" if rank in TRAIN else "heldout", layers=layers, pooled=pooled))
            del target, zero, predictions, truth, residual, beyond_range, delta
        groups = {}
        for name, members in (("train", TRAIN), ("heldout", HELDOUT), ("all", RANKS)):
            selected = [item["pooled"] for item in metrics if item["rank"] in members]
            values = {key: sum(item[key] for item in selected)/len(selected) for key in
                      ("zero_decoder_mse", "control_mse", "memory_mse", "ideal_pointwise_range_floor_mse")}
            values.update(memory_over_control=ratio(values["memory_mse"], values["control_mse"]),
                          memory_over_zero=ratio(values["memory_mse"], values["zero_decoder_mse"]),
                          control_over_zero=ratio(values["control_mse"], values["zero_decoder_mse"]))
            groups[name] = values
        h.check_bindings(sources)
        h.check_bindings(inputs)
        h.check_bindings(artifacts)
        h.require("threadpoolctl" not in sys.modules, "Forbidden threadpoolctl import")
        resources()
        result.update(passed=True, completed=True, started_sha256=started_sha, source_hashes_unchanged=True,
                      input_and_model_hashes_unchanged=True, prediction_hashes_unchanged=True,
                      learned_model_numerical=certificate, predictions=prediction_records,
                      metrics=metrics, pooled_groups=groups, zero_denominator_ratio_is_null=True,
                      no_scientific_advancement_gate_evaluated=True, torch_version=torch.__version__,
                      numpy_version=np.__version__)
    except BaseException:
        result["error"] = traceback.format_exc()
    finally:
        signal.alarm(0)
    result.update(artifacts=artifacts, ended_utc=h.utc(), elapsed_s=time.perf_counter()-timer,
                  peak_rss_bytes=h.peak_rss_bytes(), peak_mps_driver_bytes=peak_driver)
    result_sha = h.write_json(directory / "result.json", result)
    h.write_json(directory / "completion.json", dict(schema=SCHEMA, passed=result["passed"],
                 result_sha256=result_sha, utc=h.utc()))
    print({key: result.get(key) for key in ("passed", "elapsed_s", "peak_rss_bytes", "peak_mps_driver_bytes", "error")}, flush=True)
    return 0 if result["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.print_help()
        return 0
    return execute()


if __name__ == "__main__":
    raise SystemExit(main())
