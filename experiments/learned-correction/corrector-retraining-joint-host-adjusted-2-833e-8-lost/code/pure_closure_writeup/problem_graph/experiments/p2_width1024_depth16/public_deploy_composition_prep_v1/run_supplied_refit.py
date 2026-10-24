#!/Users/matt/anaconda3/bin/python3
"""Frozen-protocol J refit on the k22full SUPPLIED host's rows + fire economics.

TAG=supplied-refit  Same drill as run_joint_refit.py on the full-rank
K22-residual state-supply host (host raw -61.0% on the 8-row probe, mean
8.35e-8 -> 3.26e-8, public_k211_supply_v1/k22full_raw.log).  Exactly the
frozen J-native transfer protocol: 160/40 protected split, 25,600 parameters,
model_seed 1753362478, shuffle stream 40, six epochs, fixed final checkpoint.
All 200 receipts verified BEFORE the single lawful stage-2 target read.

THE floor question: does the valid-40 refit floor drop below the 3.13e-8 the
base and joint hosts both sat at?  Then the deployed frame (clamped) on
valid-40 and all-200, and fire economics at the projected supplied-card
utilization: fire beats the bank iff all-200 deployed x 0.8535 < 2.2843e-8.
Writes SUPPLIED_REFIT_DEPLOYED.json.  No submission, no store writes.
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
J_SOURCE = P2 / "public_full688_transport_energy_coherence_cv_v1/run_experiment.py"
J_SOURCE_SHA256 = "1aedf41e6733ffb2a62a9881dc28945f97a4233372b272f6df6204cd525b8810"
ROWS = P2 / "public_k211_supply_v1/rows"
SUPPLY_GENERATOR = P2 / "public_k211_supply_v1/generate_supplied_row.py"
SUPPLY_GENERATOR_SHA256 = "dd1f8be5a5a41b8d00ae40355a5d11a85ea038a4e11e0b83a51546844e0b78a4"
SUPPLY_CORE = P2 / "public_k211_supply_v1/supply_core.py"
SUPPLY_CORE_SHA256 = "fb5df7ad0895df36435207f085f16dbd47aa9065fd69848a9ab16f4705ba0cb7"
EXPECTED_HOST_KIND = "k22full-supplied-monkeypatch"
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
MODEL = HERE / "SUPPLIED_REFIT_MODEL.pt"
RESULT = HERE / "SUPPLIED_REFIT_DEPLOYED.json"
ROSTER = tuple(range(35, 235))
EXPECTED_VALID_GLOBAL = (
    37, 39, 42, 48, 59, 63, 66, 67, 71, 73, 75, 79, 93, 101, 122, 130,
    132, 135, 139, 145, 146, 149, 150, 151, 155, 159, 163, 169, 173, 175,
    181, 192, 198, 199, 204, 219, 224, 227, 233, 234,
)
MODEL_SEED = 1_753_362_478
SHUFFLE_STREAM_ID = 40

# Handed-down comparison constants (team lead, 2026-08-25):
DEPLOYED_BASELINE_ALL200 = 2.958e-8      # all-200, in-sample, frozen corrector
BANK_BOARD_ADJUSTED = 2.2843e-8          # current bank's board adjusted
SUPPLIED_UTIL_MULTIPLIER = 0.8535        # projected: no-eigh 1716e9 + supply ~161e9 -> ~1877e9
FIRE_THRESHOLD_ALL200 = 2.676e-8         # = bank / multiplier (lead's rounding)
# Floor references this run interrogates:
FLOOR_BASE_REFIT_VALID40 = 3.133419038664782e-08
FLOOR_JOINT_REFIT_VALID40 = 3.133856737168449e-08


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


def main():
    assert sha(J_SOURCE) == J_SOURCE_SHA256
    assert sha(SUPPLY_GENERATOR) == SUPPLY_GENERATOR_SHA256, (
        "supplied-row generator drifted from the pinned rows provenance"
    )
    assert sha(SUPPLY_CORE) == SUPPLY_CORE_SHA256, (
        "supply core drifted from the pinned rows provenance"
    )
    if RESULT.exists() or MODEL.exists():
        raise FileExistsError("supplied refit output exists; version, never overwrite")

    # Stage 1 closure: verify all 200 receipts BEFORE any target read.
    records = []
    for rank in ROSTER:
        d = ROWS / f"row_{rank:04d}"
        receipt = json.loads((d / "result.json").read_text())
        assert receipt["status"] == "complete", (rank, receipt["status"])
        assert int(receipt["rank"]) == rank
        assert int(receipt.get("targets_read", 0)) == 0
        assert receipt.get("host_kind") == EXPECTED_HOST_KIND, (rank, receipt)
        for name in ("host.npy", "raw20.npy", "sigma.npy"):
            assert (d / name).exists(), (rank, name)
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

    hosts = np.ascontiguousarray(np.stack([
        np.load(ROWS / f"row_{rank:04d}/host.npy", allow_pickle=False)
        for rank in ROSTER
    ]), dtype=np.float32)
    assert hosts.shape == (200, 16, 1024) and np.isfinite(hosts).all()
    error = np.ascontiguousarray(targets - hosts, dtype=np.float32)

    @dataclass(frozen=True)
    class JoinedRow:
        raw_locals: np.ndarray
        sigma: np.ndarray
        weights_path: Path

    class SuppliedDataset:
        def __init__(self):
            self.mlp_ids = np.asarray([int(r["mlp_id"]) for r in records], dtype=np.int64)

        def load(self, local):
            rank = ROSTER[int(local)]
            d = ROWS / f"row_{rank:04d}"
            return JoinedRow(
                raw_locals=np.load(d / "raw20.npy", mmap_mode="r", allow_pickle=False),
                sigma=np.load(d / "sigma.npy", mmap_mode="r", allow_pickle=False),
                weights_path=STORE / f"rows/observer_fit/{rank:04d}/weights.npy",
            )

    sums = np.zeros((200, 16, 19), dtype=np.float64)
    sums2 = np.zeros_like(sums)
    for local, rank in enumerate(ROSTER):
        raw = np.load(ROWS / f"row_{rank:04d}/raw20.npy", mmap_mode="r", allow_pickle=False)
        tail = np.asarray(raw[:, :, 1:], dtype=np.float64)
        sums[local] = tail.sum(axis=1, dtype=np.float64)
        sums2[local] = np.square(tail).sum(axis=1, dtype=np.float64)

    j = load_module("supplied_refit_j", J_SOURCE)
    try:
        j.configure()
    except RuntimeError:
        pass
    dataset = SuppliedDataset()
    centers, scales, layer_rms = j.moments_for(train, error, sums, sums2)
    torch.manual_seed(MODEL_SEED)
    model = j.TransportEnergyMultiStateResidual(centers, scales, layer_rms)
    assert j.parameter_count(model) == 25_600
    started = time.perf_counter()
    trace, _rss = j.fit_fold(SHUFFLE_STREAM_ID, model, dataset, train, error)
    autonomous = j.score(model, dataset, valid, error)
    training_seconds = time.perf_counter() - started

    artifact = {
        "schema": "supplied-refit-model-v1",
        "state_dict": model.state_dict(),
        "model_seed": MODEL_SEED,
        "shuffle_stream_id": SHUFFLE_STREAM_ID,
        "j_source_sha256": J_SOURCE_SHA256,
        "rows_source": "public_k211_supply_v1/rows (generator "
                       + SUPPLY_GENERATOR_SHA256[:8] + ", core "
                       + SUPPLY_CORE_SHA256[:8] + ")",
    }
    tmp = MODEL.with_name(f".{MODEL.name}.{os.getpid()}.tmp")
    torch.save(artifact, tmp)
    tmp.replace(MODEL)

    # Deployed-frame offline scoring: deployed = clamp(host + correction, 0).
    depth, width = 16, 1024
    sse_deployed = np.zeros(depth, dtype=np.float64)
    sse_unclamped = np.zeros(depth, dtype=np.float64)
    sse_host = np.zeros(depth, dtype=np.float64)
    final_sq_deployed = np.zeros(200, dtype=np.float64)
    final_sq_host = np.zeros(200, dtype=np.float64)
    clamp_engaged_elements = 0
    finite = True
    model.eval()
    with torch.no_grad():
        for offset in range(0, 200, 2):
            group = np.arange(offset, min(offset + 2, 200), dtype=np.int64)
            raw, sigma, weights, _truth = j.load_batch(dataset, group, error)
            prediction = model(raw, sigma, weights, exposure=1.0)
            correction = prediction.double().cpu().numpy()
            finite = finite and bool(np.isfinite(correction).all())
            host_group = hosts[group].astype(np.float64)
            target_group = targets[group].astype(np.float64)
            unclamped = host_group + correction
            deployed = np.maximum(unclamped, 0.0)
            clamp_engaged_elements += int((unclamped < 0.0).sum())
            sse_deployed += np.square(deployed - target_group).sum(axis=(0, 2))
            sse_unclamped += np.square(unclamped - target_group).sum(axis=(0, 2))
            sse_host += np.square(host_group - target_group).sum(axis=(0, 2))
            final_sq_deployed[group] = np.square(
                deployed[:, -1] - target_group[:, -1]
            ).mean(axis=1)
            final_sq_host[group] = np.square(
                host_group[:, -1] - target_group[:, -1]
            ).mean(axis=1)

    all200_deployed_layer = sse_deployed / (200 * width)
    all200_unclamped_layer = sse_unclamped / (200 * width)
    all200_host_layer = sse_host / (200 * width)
    valid40_deployed_final = float(final_sq_deployed[valid].mean())
    train160_deployed_final = float(final_sq_deployed[train].mean())
    all200_deployed_final = float(all200_deployed_layer[-1])
    host_valid40_final = float(final_sq_host[valid].mean())
    host_all200_final = float(all200_host_layer[-1])
    refit_valid40 = float(autonomous["final_mse"])

    projected_adjusted = all200_deployed_final * SUPPLIED_UTIL_MULTIPLIER
    beats_bank = projected_adjusted < BANK_BOARD_ADJUSTED
    result = {
        "schema": "supplied-refit-deployed-v1",
        "verdict": "FIRE_ECONOMICS_PASS" if beats_bank else "FIRE_ECONOMICS_FAIL",
        "protocol": {
            "law": "frozen 160/40 J-native transfer protocol (k22_refit_diagnostic spelling)",
            "model_seed": MODEL_SEED,
            "shuffle_stream_id": SHUFFLE_STREAM_ID,
            "epochs": 6,
            "parameters": 25600,
            "j_source_sha256": J_SOURCE_SHA256,
            "rows": "public_k211_supply_v1/rows ranks 35..234",
            "rows_generator_sha256": SUPPLY_GENERATOR_SHA256,
            "rows_supply_core_sha256": SUPPLY_CORE_SHA256,
            "rows_host_kind": EXPECTED_HOST_KIND,
            "targets_opened_after_full_receipt_verification": True,
        },
        "floor_question": {
            "supplied_refit_valid40_final_mse": refit_valid40,
            "base_refit_floor_valid40": FLOOR_BASE_REFIT_VALID40,
            "joint_refit_floor_valid40": FLOOR_JOINT_REFIT_VALID40,
            "floor_moved_vs_base": 1.0 - refit_valid40 / FLOOR_BASE_REFIT_VALID40,
            "below_3p13e8_floor": refit_valid40 < FLOOR_BASE_REFIT_VALID40,
        },
        "training": {
            "valid40_corrected_final_mse_protocol_unclamped": refit_valid40,
            "valid40_host_final_mse": host_valid40_final,
            "j_removal_fraction_valid40": 1.0 - refit_valid40 / host_valid40_final,
            "finite_full_rollout": bool(autonomous["finite_full_rollout"]),
            "training_trace_final": trace[-1],
            "training_seconds": training_seconds,
            "model_sha256": sha(MODEL),
        },
        "raw_host_no_corrector": {
            "valid40_final_mse": host_valid40_final,
            "all200_final_mse": host_all200_final,
            "all200_layer_mse": all200_host_layer.tolist(),
        },
        "deployed_frame": {
            "definition": "deployed = max(supplied_host + corrector(tape), 0) vs truth; torch forward as the corrector (flopscope parity banked at 1.75e-10)",
            "valid40_final_mse": valid40_deployed_final,
            "train160_final_mse_in_sample": train160_deployed_final,
            "all200_final_mse_IN_SAMPLE": all200_deployed_final,
            "all200_note": "160 of these 200 rows trained the corrector; the all-200 number is in-sample-optimistic",
            "all200_unclamped_final_mse": float(all200_unclamped_layer[-1]),
            "clamp_engaged_elements_of_3276800": clamp_engaged_elements,
            "all200_deployed_layer_mse": all200_deployed_layer.tolist(),
            "finite": finite,
        },
        "fire_economics": {
            "rule": "fire beats the bank iff all200_deployed x 0.8535 < 2.2843e-8, i.e. all200_deployed < 2.676e-8",
            "supplied_util_multiplier": SUPPLIED_UTIL_MULTIPLIER,
            "projected_bill_note": "no-eigh 1716e9 + supply ~161e9 -> ~1877e9 (multiplier handed down; the composed+supply card must be metered before any actual fire)",
            "threshold_all200_deployed_raw": FIRE_THRESHOLD_ALL200,
            "threshold_exact_bank_over_multiplier": BANK_BOARD_ADJUSTED / SUPPLIED_UTIL_MULTIPLIER,
            "measured_all200_deployed_IN_SAMPLE": all200_deployed_final,
            "projected_adjusted_IN_SAMPLE_OPTIMISTIC": projected_adjusted,
            "bank_board_adjusted": BANK_BOARD_ADJUSTED,
            "projected_vs_bank_ratio": projected_adjusted / BANK_BOARD_ADJUSTED,
            "beats_bank_IN_SAMPLE_OPTIMISTIC": beats_bank,
            "deployed_baseline_all200_in_sample": DEPLOYED_BASELINE_ALL200,
            "ratio_vs_deployed_baseline": all200_deployed_final / DEPLOYED_BASELINE_ALL200,
        },
        "caveats": (
            "all-200 is in-sample for the corrector (train160 included); the "
            "in-sample optimism measured on the joint refit was ~1.7% "
            "(3.188 vs 3.134e-8 valid-40); fire decisions also need the "
            "actual composed+supply card metered bill and the virgin gate, "
            "per standing law"
        ),
    }
    tmp2 = RESULT.with_name(f".{RESULT.name}.{os.getpid()}.tmp")
    tmp2.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    tmp2.replace(RESULT)
    print(json.dumps({
        "floor_question_valid40": refit_valid40,
        "below_3p13e8_floor": result["floor_question"]["below_3p13e8_floor"],
        "raw_host_all200": host_all200_final,
        "deployed_valid40": valid40_deployed_final,
        "deployed_all200_IN_SAMPLE": all200_deployed_final,
        "projected_adjusted": projected_adjusted,
        "verdict": result["verdict"],
    }, indent=1), flush=True)


if __name__ == "__main__":
    main()
