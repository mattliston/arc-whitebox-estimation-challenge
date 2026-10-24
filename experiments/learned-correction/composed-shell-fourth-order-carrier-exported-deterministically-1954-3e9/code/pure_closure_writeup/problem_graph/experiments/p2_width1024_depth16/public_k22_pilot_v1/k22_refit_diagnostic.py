#!/usr/bin/env python3
"""Stage-2 secondary diagnostic: frozen J refit on each pilot arm's tape.

Diagnostic only per the preregistration — never gating. Runs AFTER
score_pilot.py (requires PILOT_RAW_RESULT.json to exist, so truth has
already been lawfully opened by the primary scorer). Exactly the frozen
J-native transfer protocol per arm: same 160/40 protected split, exact
architecture (25,600 parameters), model_seed 1753362478, shuffle stream 40,
six epochs, fixed final checkpoint. Measures whether the raw error the
carrier removes is the component J could already reconstruct (the
additive-invariant question).
"""
from dataclasses import dataclass
import hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
J_DIR = P2 / "public_full688_transport_energy_coherence_cv_v1"
J_SOURCE = J_DIR / "run_experiment.py"
J_SOURCE_SHA256 = "1aedf41e6733ffb2a62a9881dc28945f97a4233372b272f6df6204cd525b8810"
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
RAW_RESULT = HERE / "PILOT_RAW_RESULT.json"
RESULT = HERE / "REFIT_DIAGNOSTIC.json"
ROSTER = tuple(range(35, 235))
EXPECTED_VALID_GLOBAL = (
    37, 39, 42, 48, 59, 63, 66, 67, 71, 73, 75, 79, 93, 101, 122, 130,
    132, 135, 139, 145, 146, 149, 150, 151, 155, 159, 163, 169, 173, 175,
    181, 192, 198, 199, 204, 219, 224, 227, 233, 234,
)
MODEL_SEED = 1_753_362_478
SHUFFLE_STREAM_ID = 40


def sha(p):
    d = hashlib.sha256()
    with open(p, "rb") as h:
        for b in iter(lambda: h.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def load_module(tag, path):
    spec = importlib.util.spec_from_file_location(tag, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[tag] = m
    spec.loader.exec_module(m)
    return m


def refit_arm(arm, records, valid, train, targets):
    rows_dir = HERE / "rows" / arm
    hosts = np.ascontiguousarray(np.stack([
        np.load(rows_dir / f"row_{rank:04d}/host.npy", allow_pickle=False)
        for rank in ROSTER
    ]), dtype=np.float32)
    error = np.ascontiguousarray(targets - hosts, dtype=np.float32)

    @dataclass(frozen=True)
    class JoinedRow:
        raw_locals: np.ndarray
        sigma: np.ndarray
        weights_path: Path

    class ArmDataset:
        def __init__(self):
            self.mlp_ids = np.asarray([int(r["mlp_id"]) for r in records], dtype=np.int64)

        def load(self, local):
            rank = ROSTER[int(local)]
            d = rows_dir / f"row_{rank:04d}"
            return JoinedRow(
                raw_locals=np.load(d / "raw20.npy", mmap_mode="r", allow_pickle=False),
                sigma=np.load(d / "sigma.npy", mmap_mode="r", allow_pickle=False),
                weights_path=STORE / f"rows/observer_fit/{rank:04d}/weights.npy",
            )

    sums = np.zeros((200, 16, 19), dtype=np.float64)
    sums2 = np.zeros_like(sums)
    for local, rank in enumerate(ROSTER):
        raw = np.load(rows_dir / f"row_{rank:04d}/raw20.npy", mmap_mode="r", allow_pickle=False)
        tail = np.asarray(raw[:, :, 1:], dtype=np.float64)
        sums[local] = tail.sum(axis=1, dtype=np.float64)
        sums2[local] = np.square(tail).sum(axis=1, dtype=np.float64)

    j = load_module(f"k22_refit_j_{arm}", J_SOURCE)
    try:
        j.configure()
    except RuntimeError:
        pass  # torch interop threads are process-global; second arm inherits them
    centers, scales, layer_rms = j.moments_for(train, error, sums, sums2)
    torch.manual_seed(MODEL_SEED)
    model = j.TransportEnergyMultiStateResidual(centers, scales, layer_rms)
    assert j.parameter_count(model) == 25_600
    trace, _rss = j.fit_fold(SHUFFLE_STREAM_ID, model, ArmDataset(), train, error)
    autonomous = j.score(model, ArmDataset(), valid, error)
    host_squared = np.square(error[valid].astype(np.float64))
    host_final = float(np.mean(host_squared[:, -1], dtype=np.float64))
    corrected_final = float(autonomous["final_mse"])
    return {
        "host_valid40_final_mse": host_final,
        "corrected_valid40_final_mse": corrected_final,
        "j_removal_fraction": 1.0 - corrected_final / host_final,
        "finite": bool(autonomous["finite_full_rollout"]),
        "training_trace_final": trace[-1],
    }


def main():
    assert RAW_RESULT.exists(), "run score_pilot.py first (primary opens truth)"
    assert sha(J_SOURCE) == J_SOURCE_SHA256
    if RESULT.exists():
        raise FileExistsError(RESULT)
    records = []
    for rank in ROSTER:
        receipt = json.loads((HERE / "rows/base" / f"row_{rank:04d}/result.json").read_text())
        records.append(receipt)
    keyed = []
    for local, receipt in enumerate(records):
        digest = hashlib.blake2s(
            f"p2r0f2j-q24-native-v1:{int(receipt['mlp_id'])}".encode("ascii"),
            digest_size=16).digest()
        keyed.append((digest, local))
    valid = np.asarray(sorted(local for _, local in sorted(keyed)[:40]), dtype=np.int64)
    train = np.asarray(sorted(set(range(200)) - set(valid.tolist())), dtype=np.int64)
    assert tuple(ROSTER[int(l)] for l in valid) == EXPECTED_VALID_GLOBAL and len(train) == 160

    table = pq.read_table(TARGETS, columns=["mlp_id", "all_layer_means"])
    ids = table.column("mlp_id").combine_chunks().to_numpy(zero_copy_only=False)
    values = table.column("all_layer_means").combine_chunks()
    storage = getattr(values, "storage", values)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    targets = storage.to_numpy(zero_copy_only=False).reshape(688, 16, 1024)
    expected_ids = np.asarray([int(r["mlp_id"]) for r in records], dtype=np.int32)
    assert np.array_equal(ids[35:235], expected_ids)
    targets = np.ascontiguousarray(targets[35:235], dtype=np.float32)

    started = time.perf_counter()
    arms = {}
    for arm in ("base", "k22r4"):
        arms[arm] = refit_arm(arm, records, valid, train, targets)
    absorption = {
        "host_raw_change": arms["k22r4"]["host_valid40_final_mse"]
        - arms["base"]["host_valid40_final_mse"],
        "corrected_change": arms["k22r4"]["corrected_valid40_final_mse"]
        - arms["base"]["corrected_valid40_final_mse"],
        "interpretation": (
            "corrected_change near zero while host improves = the carrier "
            "removed J-reconstructable error (additive invariant); corrected "
            "improving with the host = genuinely new information supplied"
        ),
    }
    result = {
        "schema": "k22pilot-refit-diagnostic-v1",
        "node": "P2R0F3A",
        "role": "secondary-diagnostic-never-gating",
        "arms": arms,
        "absorption": absorption,
        "elapsed_seconds": time.perf_counter() - started,
        "bindings": {"j_source_sha256": J_SOURCE_SHA256,
                     "raw_result_sha256": sha(RAW_RESULT)},
    }
    tmp = RESULT.with_name(f".{RESULT.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    tmp.replace(RESULT)
    print(json.dumps({"arms": arms, "absorption": absorption}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
