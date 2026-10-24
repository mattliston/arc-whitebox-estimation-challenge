#!/Users/matt/anaconda3/bin/python3
"""Frozen-protocol J refit on the joint host's rows + deployed-frame scoring.

TAG=joint-refit  Exactly the frozen J-native transfer protocol
(public_k22_pilot_v1/k22_refit_diagnostic.py): 160/40 protected split,
25,600 parameters, model_seed 1753362478, shuffle stream 40, six epochs,
fixed final checkpoint.  Rows: public_joint_floor_probe_v1/rows (all 200
receipts verified BEFORE the single lawful stage-2 target read).  Saves the
trained model (JOINT_REFIT_MODEL.pt) for payload splicing, then scores the
DEPLOYED frame offline: deployed = clamp(joint_host + corrector(tape), 0)
against truth, valid-40 and all-200 (the all-200 number is in-sample for the
corrector: 160 of its rows trained the model).  Writes
JOINT_REFIT_DEPLOYED.json.  No submission, no store writes.
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
ROWS = P2 / "public_joint_floor_probe_v1/rows"
JOINT_ESTIMATOR = P2 / "public_joint_floor_probe_v1/estimator_joint.py"
JOINT_ESTIMATOR_SHA256 = "4c4ef84ff3cc2fa82efffb731aa6f2ce1bc3ceb2d7488353f3b23e7b79c3522a"
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
MODEL = HERE / "JOINT_REFIT_MODEL.pt"
RESULT = HERE / "JOINT_REFIT_DEPLOYED.json"
ROSTER = tuple(range(35, 235))
EXPECTED_VALID_GLOBAL = (
    37, 39, 42, 48, 59, 63, 66, 67, 71, 73, 75, 79, 93, 101, 122, 130,
    132, 135, 139, 145, 146, 149, 150, 151, 155, 159, 163, 169, 173, 175,
    181, 192, 198, 199, 204, 219, 224, 227, 233, 234,
)
MODEL_SEED = 1_753_362_478
SHUFFLE_STREAM_ID = 40

# Handed-down comparison constants (team lead, 2026-08-25):
DEPLOYED_BASELINE_ALL200 = 2.958e-8      # all-200, in-sample, prior corrector
BANK_BOARD_ADJUSTED = 2.2843e-8          # current bank's board adjusted
COMPOSED_UTIL_MULTIPLIER = 0.8887        # at the composed card's 1954.3e9 bill
# Floor-probe outcome this run CONFIRMS in the deployed frame (receipts:
# public_joint_floor_probe_v1/JOINT_FLOOR_RESULT.json): refit floor FLAT.
FLOOR_JOINT_REFIT_VALID40 = 3.133856737168449e-08
FLOOR_BASE_REFIT_VALID40 = 3.133419038664782e-08
FLOOR_JOINT_HOST_VALID40 = 5.026953062499852e-08


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
    assert sha(JOINT_ESTIMATOR) == JOINT_ESTIMATOR_SHA256, (
        "joint estimator drifted from the pinned rows provenance"
    )
    if RESULT.exists() or MODEL.exists():
        raise FileExistsError("joint refit output exists; version, never overwrite")

    # Stage 1 closure: verify all 200 receipts BEFORE any target read.  The
    # joint generator's receipts carry no source sha; provenance is pinned
    # via JOINT_ESTIMATOR_SHA256 above instead.
    records = []
    for rank in ROSTER:
        d = ROWS / f"row_{rank:04d}"
        receipt = json.loads((d / "result.json").read_text())
        assert receipt["status"] == "complete", (rank, receipt["status"])
        assert int(receipt["rank"]) == rank
        assert int(receipt.get("targets_read", 0)) == 0
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

    class JointDataset:
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

    j = load_module("joint_refit_j", J_SOURCE)
    try:
        j.configure()
    except RuntimeError:
        pass
    dataset = JointDataset()
    centers, scales, layer_rms = j.moments_for(train, error, sums, sums2)
    torch.manual_seed(MODEL_SEED)
    model = j.TransportEnergyMultiStateResidual(centers, scales, layer_rms)
    assert j.parameter_count(model) == 25_600
    started = time.perf_counter()
    trace, _rss = j.fit_fold(SHUFFLE_STREAM_ID, model, dataset, train, error)
    autonomous = j.score(model, dataset, valid, error)
    training_seconds = time.perf_counter() - started

    artifact = {
        "schema": "joint-refit-model-v1",
        "state_dict": model.state_dict(),
        "model_seed": MODEL_SEED,
        "shuffle_stream_id": SHUFFLE_STREAM_ID,
        "j_source_sha256": J_SOURCE_SHA256,
        "rows_source": "public_joint_floor_probe_v1/rows (estimator_joint.py "
                       + JOINT_ESTIMATOR_SHA256[:8] + ")",
    }
    tmp = MODEL.with_name(f".{MODEL.name}.{os.getpid()}.tmp")
    torch.save(artifact, tmp)
    tmp.replace(MODEL)

    # Deployed-frame offline scoring: deployed = clamp(host + correction, 0).
    # The torch forward stands in for the spliced flopscope corrector; parity
    # is banked at 1.75e-10 max abs err (REAL_ARTIFACT_EXPORT_RESULT.json).
    depth, width = 16, 1024
    sse_deployed = np.zeros(depth, dtype=np.float64)
    sse_unclamped = np.zeros(depth, dtype=np.float64)
    sse_host = np.zeros(depth, dtype=np.float64)
    final_sq_deployed = np.zeros(200, dtype=np.float64)
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

    all200_deployed_layer = sse_deployed / (200 * width)
    all200_unclamped_layer = sse_unclamped / (200 * width)
    all200_host_layer = sse_host / (200 * width)
    valid40_deployed_final = float(final_sq_deployed[valid].mean())
    train160_deployed_final = float(final_sq_deployed[train].mean())
    all200_deployed_final = float(all200_deployed_layer[-1])
    host_valid40_final = float(
        np.square(error[valid].astype(np.float64))[:, -1].mean()
    )

    projected_adjusted = all200_deployed_final * COMPOSED_UTIL_MULTIPLIER
    result = {
        "schema": "joint-refit-deployed-v1",
        "verdict": "CONFIRMATION_NO_FIRE",
        "context": {
            "why_no_fire": "the joint refit floor came in FLAT vs base "
            "(3.1339e-8 vs 3.1334e-8, JOINT_FLOOR_RESULT.json), so the "
            "composed card + retrained corrector cannot beat the bank "
            "(parity raw at util 0.889 vs the bank's 0.78); this run is the "
            "deployed-frame confirmation receipt, not a promotion candidate",
            "floor_receipt": "public_joint_floor_probe_v1/JOINT_FLOOR_RESULT.json",
            "floor_joint_refit_valid40": FLOOR_JOINT_REFIT_VALID40,
            "floor_base_refit_valid40": FLOOR_BASE_REFIT_VALID40,
            "floor_joint_host_valid40": FLOOR_JOINT_HOST_VALID40,
        },
        "raw_host_no_corrector": {
            "valid40_final_mse": host_valid40_final,
            "all200_final_mse": float(all200_host_layer[-1]),
            "all200_layer_mse": all200_host_layer.tolist(),
            "note": "the corrector-free joint-host trajectory is the live "
            "lane; the all-200 number pins the valid-40 reading",
        },
        "protocol": {
            "law": "frozen 160/40 J-native transfer protocol (k22_refit_diagnostic spelling)",
            "model_seed": MODEL_SEED,
            "shuffle_stream_id": SHUFFLE_STREAM_ID,
            "epochs": 6,
            "parameters": 25600,
            "j_source_sha256": J_SOURCE_SHA256,
            "rows": "public_joint_floor_probe_v1/rows ranks 35..234",
            "rows_host_source_sha256": JOINT_ESTIMATOR_SHA256,
            "targets_opened_after_full_receipt_verification": True,
        },
        "training": {
            "valid40_corrected_final_mse_protocol_unclamped": float(autonomous["final_mse"]),
            "valid40_host_final_mse": host_valid40_final,
            "j_removal_fraction_valid40": 1.0 - float(autonomous["final_mse"]) / host_valid40_final,
            "finite_full_rollout": bool(autonomous["finite_full_rollout"]),
            "reproduction_delta_vs_floor_receipt": float(autonomous["final_mse"])
            - FLOOR_JOINT_REFIT_VALID40,
            "training_trace_final": trace[-1],
            "training_seconds": training_seconds,
            "model_sha256": sha(MODEL),
        },
        "deployed_frame": {
            "definition": "deployed = max(joint_host + corrector(joint_tape), 0) vs truth; torch forward as the corrector (flopscope parity banked at 1.75e-10)",
            "valid40_final_mse": valid40_deployed_final,
            "train160_final_mse_in_sample": train160_deployed_final,
            "all200_final_mse_IN_SAMPLE": all200_deployed_final,
            "all200_note": "160 of these 200 rows trained the corrector; the all-200 number is in-sample-optimistic",
            "all200_unclamped_final_mse": float(all200_unclamped_layer[-1]),
            "all200_host_final_mse": float(all200_host_layer[-1]),
            "clamp_engaged_elements_of_3276800": clamp_engaged_elements,
            "all200_deployed_layer_mse": all200_deployed_layer.tolist(),
            "finite": finite,
        },
        "comparison": {
            "deployed_baseline_all200_in_sample": DEPLOYED_BASELINE_ALL200,
            "delta_vs_deployed_baseline": all200_deployed_final - DEPLOYED_BASELINE_ALL200,
            "ratio_vs_deployed_baseline": all200_deployed_final / DEPLOYED_BASELINE_ALL200,
            "bank_board_adjusted": BANK_BOARD_ADJUSTED,
            "composed_util_multiplier": COMPOSED_UTIL_MULTIPLIER,
            "projected_adjusted_IN_SAMPLE_OPTIMISTIC": projected_adjusted,
            "projected_vs_bank_ratio": projected_adjusted / BANK_BOARD_ADJUSTED,
        },
        "transfer_caveat": (
            "the corrector was trained on rows from the OLD instrumented "
            "spelling (estimator_joint.py) while the deployment card is the "
            "no-eigh+t7-trim spelling; the two hosts differ by the t7 trim "
            "and the eigh removal, each measured at 2-5e-5 of the error "
            "scale in output effect, so first-order transfer holds; the "
            "deployed-frame numbers here are computed in the JOINT rows' "
            "own frame"
        ),
    }
    tmp2 = RESULT.with_name(f".{RESULT.name}.{os.getpid()}.tmp")
    tmp2.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    tmp2.replace(RESULT)
    print(json.dumps({
        "valid40_protocol": result["training"]["valid40_corrected_final_mse_protocol_unclamped"],
        "valid40_deployed": valid40_deployed_final,
        "all200_deployed_IN_SAMPLE": all200_deployed_final,
        "projected_adjusted_IN_SAMPLE_OPTIMISTIC": projected_adjusted,
    }, indent=1), flush=True)


if __name__ == "__main__":
    main()
