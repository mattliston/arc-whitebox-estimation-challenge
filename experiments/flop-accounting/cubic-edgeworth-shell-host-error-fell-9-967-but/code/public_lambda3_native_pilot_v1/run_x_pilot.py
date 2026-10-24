#!/usr/bin/env python3
"""Frozen 160/40 lambda3-X pilot: exact J on the X native tape."""
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
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
X_ROWS = HERE / "rows"
PREREGISTRATION = HERE / "PREREGISTRATION.json"
SEAL = HERE / "PREREGISTRATION_SEAL.json"
RESULT = HERE / "PILOT_RESULT.json"
MODEL = HERE / "PILOT_MODEL.pt"
J_SOURCE_SHA256 = "1aedf41e6733ffb2a62a9881dc28945f97a4233372b272f6df6204cd525b8810"
BASE_HOST_FINAL = 8.919980524938374e-8
J_NATIVE_PILOT_BASELINE = 3.107235383605394e-8
SCALING_THRESHOLD = 3.045090675933286e-8
ROSTER = tuple(range(35, 235))
EXPECTED_VALID_GLOBAL = (
    37, 39, 42, 48, 59, 63, 66, 67, 71, 73, 75, 79, 93, 101, 122, 130,
    132, 135, 139, 145, 146, 149, 150, 151, 155, 159, 163, 169, 173, 175,
    181, 192, 198, 199, 204, 219, 224, 227, 233, 234,
)
MODEL_SEED = 1_753_362_478
SHUFFLE_STREAM_ID = 40
MAX_RSS_BYTES = 8 * (1 << 30)

def sha(p):
    d = hashlib.sha256()
    with open(p, "rb") as h:
        for b in iter(lambda: h.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()

def resident_bytes():
    v = int(__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss)
    return v if sys.platform == "darwin" else v * 1024

def load_module(tag, path):
    spec = importlib.util.spec_from_file_location(tag, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[tag] = m
    spec.loader.exec_module(m)
    return m

def main():
    seal = json.loads(SEAL.read_text())
    assert seal["preregistration_sha256"] == sha(PREREGISTRATION)
    assert seal["pilot_program_sha256"] == sha(Path(__file__))
    assert seal["j_source_sha256"] == sha(J_SOURCE) == J_SOURCE_SHA256
    if RESULT.exists() or MODEL.exists():
        raise FileExistsError("X pilot output exists")
    records = []
    for rank in ROSTER:
        receipt = json.loads((X_ROWS / f"row_{rank:04d}/result.json").read_text())
        assert receipt["status"] == "complete-target-free-x"
        assert receipt["source_sha256"] == seal["instrumented_x_sha256"]
        assert int(receipt["role_rank"]) == rank
        records.append(receipt)
    keyed = []
    for local, receipt in enumerate(records):
        digest = hashlib.blake2s(f"p2r0f2j-q24-native-v1:{int(receipt['mlp_id'])}".encode("ascii"), digest_size=16).digest()
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
    hosts = np.ascontiguousarray(np.stack([
        np.load(X_ROWS / f"row_{rank:04d}/host.npy", allow_pickle=False) for rank in ROSTER
    ]), dtype=np.float32)
    error = np.ascontiguousarray(targets - hosts, dtype=np.float32)

    @dataclass(frozen=True)
    class XJoinedRow:
        raw_locals: np.ndarray
        sigma: np.ndarray
        weights_path: Path

    class XDataset:
        def __init__(self):
            self.mlp_ids = np.asarray([int(r["mlp_id"]) for r in records], dtype=np.int64)
        def load(self, local):
            rank = ROSTER[int(local)]
            d = X_ROWS / f"row_{rank:04d}"
            return XJoinedRow(
                raw_locals=np.load(d / "raw20.npy", mmap_mode="r", allow_pickle=False),
                sigma=np.load(d / "sigma.npy", mmap_mode="r", allow_pickle=False),
                weights_path=STORE / f"rows/observer_fit/{rank:04d}/weights.npy",
            )

    dataset = XDataset()
    sums = np.zeros((200, 16, 19), dtype=np.float64)
    sums2 = np.zeros_like(sums)
    for local, rank in enumerate(ROSTER):
        raw = np.load(X_ROWS / f"row_{rank:04d}/raw20.npy", mmap_mode="r", allow_pickle=False)
        tail = np.asarray(raw[:, :, 1:], dtype=np.float64)
        sums[local] = tail.sum(axis=1, dtype=np.float64)
        sums2[local] = np.square(tail).sum(axis=1, dtype=np.float64)
    j = load_module("x_pilot_j", J_SOURCE)
    j.configure()
    centers, scales, layer_rms = j.moments_for(train, error, sums, sums2)
    torch.manual_seed(MODEL_SEED)
    model = j.TransportEnergyMultiStateResidual(centers, scales, layer_rms)
    assert j.parameter_count(model) == 25_600
    started = time.perf_counter()
    trace, training_max_rss = j.fit_fold(SHUFFLE_STREAM_ID, model, dataset, train, error)
    autonomous = j.score(model, dataset, valid, error)
    host_squared = np.square(error[valid].astype(np.float64))
    host_layer = np.mean(host_squared, axis=(0, 2), dtype=np.float64)
    host_network_final = np.mean(host_squared[:, -1], axis=1, dtype=np.float64)
    candidate_by_local = {int(r["role_rank"]): float(r["final_mse"]) for r in autonomous["per_network"]}
    wins = sum(candidate_by_local[int(l)] < float(h) for l, h in zip(valid, host_network_final, strict=True))
    candidate_final = float(autonomous["final_mse"])
    host_final = float(host_layer[-1])
    maximum_network = max(float(r["final_mse"]) for r in autonomous["per_network"])
    gate = {
        "x_host_below_base_host": host_final < BASE_HOST_FINAL,
        "finite_full_rollout": bool(autonomous["finite_full_rollout"]),
        "validation_final_mse_below_5e_8": candidate_final < 5.0e-8,
        "validation_ratio_to_host_below_0_6": candidate_final / host_final < 0.6,
        "networkwise_wins_at_least_28": wins >= 28,
        "maximum_network_final_mse_below_5e_7": maximum_network < 5.0e-7,
    }
    gate["conjunctive_pass"] = all(gate.values())
    scaling = {
        "j_native_pilot_baseline": J_NATIVE_PILOT_BASELINE,
        "fractional_improvement_vs_j_native_pilot": 1.0 - candidate_final / J_NATIVE_PILOT_BASELINE,
        "scaling_threshold": SCALING_THRESHOLD,
        "full688_continuation_licensed": bool(gate["conjunctive_pass"] and candidate_final < SCALING_THRESHOLD),
    }
    max_rss = max(training_max_rss, resident_bytes())
    assert max_rss < MAX_RSS_BYTES
    artifact = {"schema": "lambda3-x-pilot-model-v1", "state_dict": model.state_dict(),
                "model_seed": MODEL_SEED, "preregistration_sha256": seal["preregistration_sha256"]}
    tmp = MODEL.with_name(f".{MODEL.name}.{os.getpid()}.tmp")
    torch.save(artifact, tmp)
    tmp.replace(MODEL)
    result = {
        "schema": "lambda3-x-pilot-result-v1",
        "status": "complete",
        "node": "P2R0F2X",
        "validation": {
            "host_final_mse": host_final,
            "base_host_final_mse": BASE_HOST_FINAL,
            "host_fractional_improvement": 1.0 - host_final / BASE_HOST_FINAL,
            "autonomous_final_mse": candidate_final,
            "autonomous_layer_mse": autonomous["layer_mse"],
            "host_layer_mse": host_layer.tolist(),
            "autonomous_ratio_to_host": candidate_final / host_final,
            "maximum_per_network_final_mse": maximum_network,
            "networkwise_wins": wins,
            "networks": 40,
        },
        "decision_gate": gate,
        "scaling_rule": scaling,
        "training_trace_final": trace[-1],
        "resource": {"maximum_process_rss_bytes": max_rss, "elapsed_seconds": time.perf_counter() - started},
        "bindings": {
            "seal_sha256": sha(SEAL),
            "preregistration_sha256": seal["preregistration_sha256"],
            "pilot_program_sha256": seal["pilot_program_sha256"],
            "instrumented_x_sha256": seal["instrumented_x_sha256"],
            "model_sha256": sha(MODEL),
        },
        "observer_selection_targets_opened": 0,
        "name_or_board_join": False,
    }
    tmp2 = RESULT.with_name(f".{RESULT.name}.{os.getpid()}.tmp")
    tmp2.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    tmp2.replace(RESULT)
    print(json.dumps({"validation": result["validation"], "gate": gate, "scaling": scaling}, sort_keys=True), flush=True)

if __name__ == "__main__":
    main()
