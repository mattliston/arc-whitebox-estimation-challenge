"""Fixed Phase2 matched Dirichlet-Gram information probe; inert by default.

Run ONLY after source review with --execute-reviewed-probe. Both40-epoch
payloads are sealed, then all NumPy/MPS held-out predictions are sealed,
BEFORE any VALID truth file is even hashed. No VALID interim loss exists.
All artifacts are exclusive-create under dirichlet_gram_probe_pilot_v1.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
OUT = HERE / "dirichlet_gram_probe_pilot_v1"
ROWS = HERE / "rows_dirichlet_gram_probe_v1"
PILOT = "dirichlet_gram_rows_pilot_v1.json"
Q_NAME = "payload_fullpair_final_head_v1.npz"
RANKS = tuple(range(35, 59))
VALID = (37, 39, 42, 48)
TRAIN = tuple(rank for rank in RANKS if rank not in VALID)
ARMS = (("control", False), ("sources", True))
SEED = 930080
EPOCHS = 40
BATCH_SIZE = 8192
WIDTH = 1024
WALL_CAP = 600.0
RSS_CAP = 6 * 1024**3
DELTA_TOLERANCE = 2e-9
FEATURE_ORDER = (
    "log_diag_Kpre15_over_scalar_variance",
    "mean_offdiagonal_D",
    "rms_offdiagonal_D",
    "D_alpha_over_n",
    "D_host_over_sigma_over_n",
    "D_gate_over_n",
    "mean_D_rho",
    "mean_D_rho_squared",
)
PINS = {
    "DIRICHLET_GRAM_INFORMATION_DESIGN_V1.md":
        "1ddd561a782b19cdb81c95c4c3d92960b16e311114d7118a9ed061de7e9d302c",
    "DIRICHLET_GRAM_ROOT_ACCEPTANCE_V1.md":
        "7101d3ed78424e6e2a6fb38c183752b0e5893883585f0625e28f392b3a48231a",
    "dirichlet_gram_observer_v1.py":
        "a7456ab9861568f04a7c6b7884677a0921448924f89f1f67ec45938e5a6f21af",
    "generate_dirichlet_gram_row_v1.py":
        "a0f382100b66d0a3ada7901a57a74525949be7243bf8032d65bf7b87bba0b003",
    "prepare_dirichlet_gram_pilot_v1.py":
        "3fb3cec96e6580759375c202c42576cf1e4a4d6fea5f8c5e4a4af5c90f2406b4",
    "certify_dirichlet_gram_observer_v1.py":
        "8c48de79712e9253d1b94d2a7090c828263c1b8d55a2738262cdf2b97e13972c",
    "dirichlet_gram_small_certificate_v1.json":
        "f0952df6679abd11a7055b4052d5e478ff719cc8f860da438f918908cab90eda",
    "compact_final_head_v1.py":
        "acaee5719a619518caa989f4122847033ace9f15526bbe283144dc1535c3702a",
    PILOT: "737dae5bfac496a1e6cbbd3f67b7abbf5ebb59e32e073ddacb9e4c85ca393991",
    Q_NAME: "848420b459f08e3e39a0a83916db16d7a4400381fcf256a373cc0ac139c12c63",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(name, value):
    path = OUT / name
    record = dict(value, written_at_utc=utc_now(), written_monotonic_ns=time.monotonic_ns())
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return dict(path=name, sha256=digest(path), written_at_utc=record["written_at_utc"],
                written_monotonic_ns=record["written_monotonic_ns"])


def write_npz(np, name, arrays):
    path = OUT / name
    with path.open("xb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    return dict(path=name, sha256=digest(path), sealed_at_utc=utc_now(),
                sealed_monotonic_ns=time.monotonic_ns())


class Bindings:
    """Only explicit, NEW-relative input/source paths; no truth globbing."""

    def __init__(self):
        self.files = {}

    def add(self, relative, expected=None):
        relative = str(relative)
        path = (HERE / relative).resolve()
        require(path.is_relative_to(HERE), "Input binding escaped NEW")
        require("truthpost" not in path.parts and path.name != "loss.npz",
                "Truth/loss is forbidden in input provenance traversal")
        relative = path.relative_to(HERE).as_posix()
        if relative not in self.files:
            stat = path.stat()
            self.files[relative] = dict(sha256=digest(path), size=stat.st_size,
                                        mtime_ns=stat.st_mtime_ns)
        if expected is not None:
            require(self.files[relative]["sha256"] == expected, "Digest mismatch: " + relative)
        return self.files[relative]["sha256"]

    def recheck(self):
        for relative, before in self.files.items():
            path = HERE / relative
            stat = path.stat()
            require(stat.st_size == before["size"] and stat.st_mtime_ns == before["mtime_ns"],
                    "Input/source stat changed: " + relative)
            require(digest(path) == before["sha256"], "Input/source content changed: " + relative)


def assert_artifacts(artifacts):
    for item in artifacts:
        require(digest(OUT / item["path"]) == item["sha256"],
                "Frozen output artifact changed: " + item["path"])


def check_resources(started):
    require("threadpoolctl" not in sys.modules, "threadpoolctl must not be imported in AP")
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    elapsed = time.monotonic() - started
    require(peak <= RSS_CAP, "Peak process RSS exceeded6GiB")
    require(elapsed <= WALL_CAP, "Fixed600s execution/fit cap exceeded")
    return dict(wall_seconds=elapsed, peak_rss_bytes=peak)


def configure_process():
    require(sys.platform == "darwin", "This fixed MPS/RSS contract requires macOS")
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "Launch this worker at nice>=10")
    require(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0", "MPS fallback must be explicitly0")
    require("torch" not in sys.modules and "numpy" not in sys.modules,
            "Numerical packages must not be imported before the thread contract")
    require("threadpoolctl" not in sys.modules, "Forbidden preloaded threadpoolctl")
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
    for name in names:
        require(os.environ.get(name, "2") == "2", name + " must be2")
        os.environ[name] = "2"
    return {name: os.environ[name] for name in names}


def qualify_inputs(bindings):
    for name, expected in PINS.items():
        bindings.add(name, expected)
    bindings.add(Path(__file__).name)
    pilot = read_json(HERE / PILOT)
    require(pilot["passed"] is True and pilot["target_access"] is False,
            "The fixed24-row pilot is not input-only qualified")
    require(pilot["source_unchanged"] is True and tuple(pilot["ranks"]) == RANKS,
            "Pilot rank/source seal mismatch")
    require(tuple(row["rank"] for row in pilot["rows"]) == RANKS, "Pilot row roster mismatch")
    for name, expected in pilot["source_sha256"].items():
        bindings.add(name, expected)
    certificate = read_json(HERE / "dirichlet_gram_small_certificate_v1.json")
    require(certificate["passed"] is True, "Synthetic certificate is not passing")
    require(certificate["source_sha256"] == PINS["dirichlet_gram_observer_v1.py"],
            "Synthetic certificate observer mismatch")
    for item in pilot["rows"]:
        rank = item["rank"]
        require(item["passed"] is True and item["peak_rss_bytes"] <= RSS_CAP,
                "Input row qualification/resource failure")
        prefix = f"rows_dirichlet_gram_probe_v1/row_{rank:04d}"
        bindings.add(prefix + "/result.json", item["record_sha256"])
        bindings.add(prefix + "/data.npz", item["data_sha256"])
        record = read_json(HERE / prefix / "result.json")
        require(record["rank"] == rank and record["passed"] is True,
                "Row identity/qualification mismatch")
        require(record["target_access"] is False and record["inputs_unchanged"] is True,
                "Row has no causal input seal")
        require(record["data_sha256"] == item["data_sha256"] and all(record["byte_parity"].values()),
                "Row data/parent byte parity failed")
        require(set(record["byte_parity"]) == {"x", "sigma", "host", "delta", "h", "prediction"},
                "Parent byte-parity contract changed")
        require(record["source_dtypes"]["final_sigma"] == "float64"
                and record["source_dtypes"]["final_host"] == "float64",
                "Scalar-final dtype ownership changed")
        if rank == 35:
            require(record["deterministic_replay"] is True, "Rank35 smoke replay did not pass")
        expected_keys = {
            "dirichlet_gram_observer_v1.py", "generate_dirichlet_gram_row_v1.py",
            "DIRICHLET_GRAM_INFORMATION_DESIGN_V1.md", "DIRICHLET_GRAM_ROOT_ACCEPTANCE_V1.md",
            "dirichlet_gram_small_certificate_v1.json", Q_NAME,
            f"rows_matrix_error_observer_v1/row_{rank:04d}/result.json",
            f"rows_matrix_error_observer_v1/row_{rank:04d}/final.npz",
            *(f"rows_current_transport_shadow_v2/row_{rank:04d}/{name}"
              for name in ("result.json", "weights.npy", "pre_cov.npy", "x.npy", "sigma.npy")),
        }
        require(set(record["input_source_sha256"]) == expected_keys,
                "Unexpected input source/loss path in row provenance")
        for name, expected in record["input_source_sha256"].items():
            bindings.add(name, expected)
        bank_record = read_json(HERE / f"rows_matrix_error_observer_v1/row_{rank:04d}/result.json")
        require(bank_record["passed"] is True and bank_record["target_access"] is False,
                "Original scalar-final row is not qualified")
        require(all(bank_record["prefix_feature_sigma_byte_parity"])
                and all(bank_record["existing_current_scalar_final_byte_parity"].values()),
                "Original prefix/scalar-final parity is not passing")
        require(set(bank_record["source_hashes"]) == {"payload_compact_feed_fullpair_v1.npz", Q_NAME},
                "Unexpected parent P/Q payload roster")
        for name, expected in bank_record["source_hashes"].items():
            bindings.add(name, expected)
    bindings.recheck()
    return pilot


def serve(np, host, delta):
    return np.maximum(host + delta, 0).astype(np.float32)


def load_inputs(np):
    rows = {}
    shapes = {"x": (WIDTH, 30), "sigma": (WIDTH,), "host": (WIDTH,),
              "delta": (WIDTH,), "h": (WIDTH, 126), "prediction": (16, WIDTH),
              "sources": (WIDTH, 8)}
    for rank in RANKS:
        with np.load(ROWS / f"row_{rank:04d}/data.npz", allow_pickle=False) as archive:
            require(set(archive.files) == set(shapes), "Input NPZ schema mismatch")
            row = {key: archive[key].copy() for key in shapes}
        for key, shape in shapes.items():
            require(row[key].shape == shape and np.isfinite(row[key]).all(),
                    f"Invalid shape/finite field: {rank}/{key}")
        require(row["h"].dtype == np.float32 and row["sources"].dtype == np.float32,
                "Head/source float32 contract changed")
        require(row["host"].dtype == np.float64 and row["sigma"].dtype == np.float64,
                "Actual scalar-final host/sigma must remain float64")
        require(np.all(row["sigma"] > 0), "Nonpositive scalar-final sigma")
        require(np.array_equal(serve(np, row["host"], row["delta"]),
                               np.maximum(row["prediction"][-1], 0).astype(np.float32)),
                "Served parent host+delta does not match archived parent")
        rows[rank] = row
    with np.load(HERE / Q_NAME, allow_pickle=False) as archive:
        q = {key: archive[key].copy() for key in archive.files}
    for key, shape in {"w0": (128, 126), "b0": (128,), "w2": (128, 128),
                       "b2": (128,), "w4": (1, 128), "b4": (1,)}.items():
        require(q[key].shape == shape and q[key].dtype == np.float32 and np.isfinite(q[key]).all(),
                "Frozen Q parameter contract mismatch: " + key)
    require(q["rms"].size == 1 and np.isfinite(q["rms"]).all() and float(q["rms"][0]) > 0,
            "Invalid frozen Q output scale")
    train_sources = np.concatenate([rows[rank]["sources"] for rank in TRAIN], axis=0)
    # Match the old trainer's float32 feature-statistic arithmetic, using TRAIN only.
    center = train_sources.mean(axis=0)
    std = np.maximum(train_sources.std(axis=0), 1e-6)
    require(center.shape == std.shape == (8,) and np.isfinite(center).all()
            and np.isfinite(std).all(), "Invalid TRAIN normalizer")
    for row in rows.values():
        row["extra"] = np.clip((row["sources"] - center) / std, -8, 8).astype(np.float32)
    return rows, q, center, std


def read_truth(np, rank, state, stage):
    if stage == "train":
        require(rank in TRAIN and state["stage"] == "train_truth_access",
                "TRAIN truth access outside its barrier")
    else:
        require(stage == "valid" and rank in VALID and state["stage"] == "validation_judge",
                "VALID truth access outside its barrier")
        require(state.get("payload_seal") and state.get("prediction_seal"),
                "VALID truth requires BOTH payloads and all predictions sealed")
        assert_artifacts([state["payload_seal"], state["prediction_seal"]])
        assert_artifacts(list(state["payloads"].values()) + [state["predictions"]])
    event = dict(rank=rank, split=stage, first_access_at_utc=utc_now(),
                 first_access_monotonic_ns=time.monotonic_ns())
    state["truth_accesses"].append(event)
    path = HERE.parent / "public_g22_operator_features_v1" / "truthpost" / f"truthpost_rank{rank:04d}.npz"
    # For VALID, even this first byte hash is strictly AFTER the prediction seal.
    before = digest(path)
    with np.load(path, allow_pickle=False) as archive:
        values = archive["mc_post"]
        require(values.shape == (16, WIDTH) and np.isfinite(values).all(), "Invalid frozen truth array")
        result = values[-1].astype(np.float64).copy()
    require(digest(path) == before, "Truth file changed during its authorized read")
    event.update(path=str(path), sha256=before, completed_at_utc=utc_now())
    write_json(f"truth_access_{stage}_{rank:04d}.json", event)
    return result


def build_model(torch, q, device, *, fitted=False):
    model = torch.nn.Sequential(torch.nn.Linear(134, 128), torch.nn.SiLU(),
                                torch.nn.Linear(128, 128), torch.nn.SiLU(),
                                torch.nn.Linear(128, 1)).to(device)
    with torch.no_grad():
        model[0].weight.zero_()
        if fitted:
            model[0].weight.copy_(torch.tensor(q["w0"], dtype=torch.float32, device=device))
        else:
            model[0].weight[:, :126].copy_(torch.tensor(q["w0"], dtype=torch.float32, device=device))
        model[0].bias.copy_(torch.tensor(q["b0"], dtype=torch.float32, device=device))
        for index in (2, 4):
            model[index].weight.copy_(torch.tensor(q[f"w{index}"], dtype=torch.float32, device=device))
            model[index].bias.copy_(torch.tensor(q[f"b{index}"], dtype=torch.float32, device=device))
    return model


def run_probe(started, state):
    environment = configure_process()
    bindings = Bindings()
    state["stage"] = "input_qualification"
    qualify_inputs(bindings)
    import numpy as np
    import torch
    from compact_final_head_v1 import final_delta
    require(Path(sys.modules["compact_final_head_v1"].__file__).resolve()
            == HERE / "compact_final_head_v1.py", "Wrong NumPy head module imported")
    require("threadpoolctl" not in sys.modules, "Numerical import loaded forbidden threadpoolctl")
    require(torch.backends.mps.is_available(), "MPS is unavailable; no fallback arm")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    require(torch.get_num_threads() == 2, "Torch thread contract failed")
    device = torch.device("mps")
    rows, q, center, std = load_inputs(np)
    check_resources(started)
    bindings.recheck()
    state["input_seal"] = write_json("input_qualification.json", dict(
        passed=True, target_access=False, ranks=RANKS, train=TRAIN, valid=VALID,
        source_feature_order=FEATURE_ORDER, files=bindings.files,
        normalizer_split="TRAIN only", environment=environment,
        numpy_version=np.__version__, torch_version=torch.__version__,
        nice=os.getpriority(os.PRIO_PROCESS, 0), fallback=os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
        output_transform="maximum(float64_host + delta, 0).astype(float32)",
        **check_resources(started)))

    # This is the first target-array access anywhere in the execution.
    state["stage"] = "train_truth_access"
    truths = {rank: read_truth(np, rank, state, "train") for rank in TRAIN}
    require(set(truths) == set(TRAIN), "Only TRAIN targets may be present during fitting")
    target_np = np.concatenate([truths[rank] - rows[rank]["host"] for rank in TRAIN])
    sigma_np = np.concatenate([rows[rank]["sigma"] for rank in TRAIN])
    target = torch.tensor(target_np, dtype=torch.float32, device=device)
    sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
    normalizer = float(target.square().mean().cpu())
    require(math.isfinite(normalizer) and normalizer > 0, "Invalid TRAIN loss normalizer")
    scale = float(q["rms"][0])
    steps_per_epoch = math.ceil(len(target_np) / BATCH_SIZE)
    require(len(target_np) == 20 * WIDTH and steps_per_epoch == 3, "Fixed training roster changed")
    state["stage"] = "matched_training"
    state["payloads"] = {}
    histories = {}
    permutation_hashes = {}

    for name, enabled in ARMS:
        torch.manual_seed(SEED)
        # A reset CPU generator guarantees identical permutations for both arms.
        permutation_generator = torch.Generator(device="cpu").manual_seed(SEED)
        input_np = np.concatenate([
            np.concatenate((rows[rank]["h"], rows[rank]["extra"] if enabled
                            else np.zeros_like(rows[rank]["extra"])), axis=1)
            for rank in TRAIN], axis=0)
        x = torch.tensor(input_np, dtype=torch.float32, device=device)
        model = build_model(torch, q, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=2e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS * steps_per_epoch, eta_min=5e-5)
        history, orders = [], []
        update_count = 0
        for epoch in range(1, EPOCHS + 1):
            order_cpu = torch.randperm(len(target_np), generator=permutation_generator)
            orders.append(hashlib.sha256(order_cpu.numpy().tobytes()).hexdigest())
            order = order_cpu.to(device)
            model.train()
            for offset in range(0, len(target_np), BATCH_SIZE):
                index = order[offset:offset + BATCH_SIZE]
                predicted_delta = sigma[index] * scale * model(x[index])[:, 0]
                loss = (predicted_delta - target[index]).square().mean() / normalizer
                require(bool(torch.isfinite(loss).item()), "Nonfinite TRAIN loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                update_count += 1
                check_resources(started)
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    train_mse = float((sigma * scale * model(x)[:, 0] - target).square().mean().cpu())
                require(math.isfinite(train_mse), "Nonfinite TRAIN progress loss")
                record = dict(arm=name, epoch=epoch, updates=update_count, train_mse=train_mse,
                              learning_rate=float(optimizer.param_groups[0]["lr"]),
                              valid_truth_access=False, **check_resources(started))
                history.append(record)
                write_json(f"progress_{name}_epoch_{epoch:02d}.json", record)
                print(json.dumps(record, sort_keys=True), flush=True)
        require(update_count == 120, "Incomplete fixed40-epoch fit")
        require(abs(float(optimizer.param_groups[0]["lr"]) - 5e-5) < 1e-15,
                "Cosine endpoint changed")
        payload = {f"{kind}{index}": getattr(model[index], "weight" if kind == "w" else "bias")
                   .detach().cpu().numpy().copy()
                   for index in (0, 2, 4) for kind in ("w", "b")}
        require(all(np.isfinite(value).all() for value in payload.values()), "Nonfinite fitted payload")
        if not enabled:
            require(np.count_nonzero(payload["w0"][:, 126:]) == 0,
                    "Zero-feature control acquired active extra columns")
        payload.update(rms=q["rms"].copy(), source_center=center.copy(), source_std=std.copy(),
                       source_enabled=np.asarray([enabled], dtype=np.bool_),
                       source_feature_order=np.asarray(FEATURE_ORDER))
        state["payloads"][name] = write_npz(np, f"payload_{name}.npz", payload)
        histories[name] = dict(epochs=history, updates=update_count, final_epoch=EPOCHS)
        permutation_hashes[name] = orders
        del model, optimizer, scheduler, x, input_np, payload, predicted_delta, loss, order, order_cpu
        torch.mps.empty_cache()

    require(permutation_hashes["control"] == permutation_hashes["sources"],
            "Matched minibatch permutations differ")
    require(all(event["split"] == "train" for event in state["truth_accesses"]),
            "Unexpected VALID access before both fits ended")
    bindings.recheck()
    assert_artifacts(list(state["payloads"].values()))
    state["payload_seal"] = write_json("both_payloads_sealed_before_validation.json", dict(
        passed=True, valid_truth_access=False, payloads=state["payloads"], train=TRAIN, valid=VALID,
        seed=SEED, epochs=EPOCHS, updates_per_arm=120, source_feature_order=FEATURE_ORDER,
        train_loss_normalizer=normalizer, permutation_hashes=permutation_hashes,
        input_seal=state["input_seal"], input_source_hashes_unchanged=True,
        **check_resources(started)))

    # Held-out INPUTS only; no target dictionary contains a VALID rank.
    state["stage"] = "heldout_prediction_only"
    predictions, discrepancies = {}, {}
    for rank in VALID:
        predictions[f"parent_{rank}"] = serve(np, rows[rank]["host"], rows[rank]["delta"])
    for name, enabled in ARMS:
        item = state["payloads"][name]
        assert_artifacts([item])
        with np.load(OUT / item["path"], allow_pickle=False) as archive:
            fitted = {key: archive[key].copy() for key in archive.files}
        require(fitted["w0"].shape == (128, 134), "Fitted architecture changed")
        require(tuple(fitted["source_feature_order"].tolist()) == FEATURE_ORDER
                and bool(fitted["source_enabled"][0]) == enabled,
                "Payload feature-order/arm ownership changed")
        require(np.array_equal(fitted["source_center"], center)
                and np.array_equal(fitted["source_std"], std), "Payload TRAIN normalizer changed")
        model = build_model(torch, fitted, device, fitted=True).eval()
        for rank in VALID:
            row = rows[rank]
            extra = np.clip((row["sources"] - fitted["source_center"]) / fitted["source_std"],
                            -8, 8).astype(np.float32) if enabled else np.zeros_like(row["sources"])
            inputs = np.concatenate((row["h"], extra), axis=1)
            numpy_delta = final_delta(inputs, row["sigma"], fitted)
            with torch.no_grad():
                tx = torch.tensor(inputs, dtype=torch.float32, device=device)
                ts = torch.tensor(row["sigma"], dtype=torch.float32, device=device)
                mps_delta = (ts * float(fitted["rms"][0]) * model(tx)[:, 0]).cpu().numpy().copy()
            require(np.isfinite(numpy_delta).all() and np.isfinite(mps_delta).all(),
                    "Nonfinite held-out input-only predictions")
            discrepancy = float(np.max(np.abs(numpy_delta.astype(np.float64)
                                              - mps_delta.astype(np.float64))))
            require(discrepancy <= DELTA_TOLERANCE, f"NumPy/MPS delta mismatch: {name}/{rank}")
            discrepancies[f"{name}_{rank}"] = discrepancy
            predictions[f"{name}_{rank}"] = serve(np, row["host"], numpy_delta)
            predictions[f"{name}_delta_{rank}"] = numpy_delta.copy()
            predictions[f"{name}_mps_delta_{rank}"] = mps_delta.copy()
            check_resources(started)
        del model, fitted, tx, ts
        torch.mps.empty_cache()
    require(all(value.dtype == np.float32 and np.isfinite(value).all() for value in predictions.values()),
            "Predictions must be finite served-float32 arrays or float32 deltas")
    state["predictions"] = write_npz(np, "heldout_predictions_before_truth.npz", predictions)
    bindings.recheck()
    assert_artifacts(list(state["payloads"].values()) + [state["payload_seal"], state["predictions"]])
    require(all(event["split"] == "train" for event in state["truth_accesses"]),
            "VALID truth was touched before prediction seal")
    state["prediction_seal"] = write_json("all_predictions_sealed_before_valid_truth.json", dict(
        passed=True, valid_truth_access=False, ranks=VALID, predictions=state["predictions"],
        payload_seal=state["payload_seal"], payloads=state["payloads"],
        numpy_mps_delta_max=discrepancies, tolerance=DELTA_TOLERANCE,
        input_source_hashes_unchanged=True, **check_resources(started)))

    # No model object survives this barrier; judge only the sealed NumPy arrays.
    state["stage"] = "validation_judge"
    with np.load(OUT / state["predictions"]["path"], allow_pickle=False) as archive:
        served = {key: archive[key].copy() for key in archive.files}
    records = []
    for rank in VALID:
        truth = read_truth(np, rank, state, "valid")
        risks = {name: float(np.mean((served[f"{name}_{rank}"].astype(np.float64) - truth)**2))
                 for name in ("parent", "control", "sources")}
        require(all(math.isfinite(value) for value in risks.values())
                and risks["parent"] > 0 and risks["control"] > 0,
                "Nonfinite/zero held-out risk denominator: inconclusive")
        records.append(dict(rank=rank, parent_raw=risks["parent"], control_raw=risks["control"],
                            sources_raw=risks["sources"], sources_to_parent=risks["sources"]/risks["parent"],
                            sources_to_control=risks["sources"]/risks["control"],
                            control_numpy_mps_max=discrepancies[f"control_{rank}"],
                            sources_numpy_mps_max=discrepancies[f"sources_{rank}"]))
    parent_ratio = sum(row["sources_raw"] for row in records) / sum(row["parent_raw"] for row in records)
    control_ratio = sum(row["sources_raw"] for row in records) / sum(row["control_raw"] for row in records)
    wins = sum(row["sources_raw"] < row["parent_raw"] for row in records)
    gates = dict(pooled_source_to_parent_at_most_point65=parent_ratio <= .65,
                 pooled_source_to_control_at_most_point70=control_ratio <= .70,
                 at_least_three_of_four_parent_wins=wins >= 3)
    bindings.recheck()
    assert_artifacts(list(state["payloads"].values()) +
                     [state["input_seal"], state["payload_seal"], state["predictions"], state["prediction_seal"]])
    for event in state["truth_accesses"]:
        require(digest(Path(event["path"])) == event["sha256"], "Authorized truth source changed")
        if event["split"] == "valid":
            require(event["first_access_monotonic_ns"] > state["prediction_seal"]["written_monotonic_ns"],
                    "VALID target seal ordering violation")
    state["stage"] = "complete"
    result = dict(
        schema="dirichlet-gram-matched-final-probe-v1", qualification_passed=True,
        expansion_gate=all(gates.values()), gates=gates, train=TRAIN, valid=VALID,
        source_feature_order=FEATURE_ORDER, seed=SEED, epochs=EPOCHS, batch_size=BATCH_SIZE,
        architecture=[134, 128, 128, 1], learning_rate=5e-4, minimum_learning_rate=5e-5,
        weight_decay=2e-5, gradient_norm_cap=5, records=records, histories=histories,
        pooled_sources_to_parent=parent_ratio, pooled_sources_to_control=control_ratio, parent_wins=wins,
        payloads=state["payloads"], payload_seal=state["payload_seal"],
        predictions=state["predictions"], prediction_seal=state["prediction_seal"],
        truth_accesses=state["truth_accesses"], input_source_hashes_unchanged=True,
        input_files=bindings.files, train_only_progress=True, final_epoch_only=True,
        qualification=("One fixed40-epoch matched final-only information probe on20TRAIN/4 reused research "
                       "holdouts. Both payloads and all held-out NumPy/MPS predictions sealed before any "
                       "VALID truth bytes. Served max0/float32 outputs; no selection, true Jacobian claim, "
                       "native acquisition/parity, full bill, independent14-net panel, candidate or board score."),
        **check_resources(started))
    write_json("result.json", result)
    print(json.dumps({key: result[key] for key in ("qualification_passed", "expansion_gate", "gates", "records",
                                                   "pooled_sources_to_parent", "pooled_sources_to_control",
                                                   "parent_wins", "wall_seconds", "peak_rss_bytes")}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-reviewed-probe", action="store_true")
    args = parser.parse_args()
    if not args.execute_reviewed_probe:
        print("INERT: explicit --execute-reviewed-probe is required after source review.")
        return
    require(not OUT.exists(), "Refusing to overwrite an existing versioned probe directory")
    OUT.mkdir()
    started = time.monotonic()
    state = dict(stage="starting", truth_accesses=[])

    def timeout_handler(_signum, _frame):
        raise TimeoutError("Fixed600s wall cap exceeded")

    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, WALL_CAP)
    try:
        write_json("started.json", dict(schema="dirichlet-gram-matched-final-probe-v1",
                   source_sha256=digest(Path(__file__)), train=TRAIN, valid=VALID,
                   wall_cap_seconds=WALL_CAP, rss_cap_bytes=RSS_CAP, target_access=False))
        run_probe(started, state)
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        write_json("failure.json", dict(passed=False, error_type=type(exc).__name__, error=str(exc),
                   traceback=traceback.format_exc(), state=state,
                   wall_seconds=time.monotonic()-started,
                   peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                   source_sha256=digest(Path(__file__))))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


if __name__ == "__main__":
    main()
