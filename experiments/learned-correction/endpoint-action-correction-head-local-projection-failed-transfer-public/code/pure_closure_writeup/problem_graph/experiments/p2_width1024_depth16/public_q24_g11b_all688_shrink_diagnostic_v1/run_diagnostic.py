#!/Users/matt/anaconda3/bin/python3
"""Target-blind all-688 energy diagnostic for the integrated G11B correction."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
PILOT_ROWS = P2 / "public_q24_native_transport_energy_pilot_v1/runs/native_rows_v1"
FINAL_ROWS = P2 / "public_q24_native_transport_energy_final_v1/runs/native_rows_remaining_v1"
FEATURE_STORE = (
    P2 / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3"
)
J_SOURCE = P2 / "public_full688_transport_energy_coherence_cv_v1/run_experiment.py"
Q_SOURCE = P2 / "public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py"
ALPHA_MODEL = P2 / "public_q24_native_j_checkpoint_interpolation_v1/INTERPOLATED_MODEL.pt"
Q_MODEL = P2 / "public_q24_native_j_q10q15_residual_head_v1/Q_SUFFIX_HEAD.pt"
PREREGISTRATION = HERE / "PREREGISTRATION.json"
RESULT = HERE / "RESULT.json"

J_SOURCE_SHA256 = "1aedf41e6733ffb2a62a9881dc28945f97a4233372b272f6df6204cd525b8810"
Q_SOURCE_SHA256 = "1e4c906936f4685bd800f922ded6c582af8213ecad1fbe41da651faff82be9c3"
ALPHA_MODEL_SHA256 = "fc65528720c7e1378ed7011b7b201f4086217a544b092de208ea1064656d9ef5"
Q_MODEL_SHA256 = "5fb57a510d933940f19ddd5ab11650c3a966c2039c6bd12f73988457c2ce91b8"
NATIVE_AGGREGATE_SHA256 = "204097b1e43387fdc2080eb690606d7dc34aeda7ca4794cb98382137ebfa3db3"
R0_D1_RAW_MSE = 2.9409214512554626e-8
R1_G11B_RAW_MSE = 3.065038697513956e-8
SUFFIX = tuple(range(10, 16))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(tag: str, path: Path):
    specification = importlib.util.spec_from_file_location(tag, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[tag] = module
    specification.loader.exec_module(module)
    return module


def row_dir(rank: int) -> Path:
    root = PILOT_ROWS if 35 <= rank < 235 else FINAL_ROWS
    return root / f"row_{rank:04d}"


def verify_inputs() -> list[dict[str, object]]:
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if prereg.get("status") != "FROZEN_BEFORE_TARGET_BLIND_DIAGNOSTIC":
        raise AssertionError("preregistration status changed")
    expected = {
        "program_sha256": sha256(Path(__file__)),
        "j_source_sha256": sha256(J_SOURCE),
        "q_source_sha256": sha256(Q_SOURCE),
        "alpha_model_sha256": sha256(ALPHA_MODEL),
        "q_model_sha256": sha256(Q_MODEL),
    }
    literals = {
        "j_source_sha256": J_SOURCE_SHA256,
        "q_source_sha256": Q_SOURCE_SHA256,
        "alpha_model_sha256": ALPHA_MODEL_SHA256,
        "q_model_sha256": Q_MODEL_SHA256,
    }
    for key, value in expected.items():
        if prereg["bindings"].get(key) != value:
            raise AssertionError(f"preregistered input changed: {key}")
    for key, value in literals.items():
        if expected[key] != value:
            raise AssertionError(f"literal input changed: {key}")
    records: list[dict[str, object]] = []
    aggregate = hashlib.sha256()
    for rank in range(688):
        result_path = row_dir(rank) / "result.json"
        record = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            record.get("status") != "complete-target-free"
            or record.get("role") != "observer_fit"
            or int(record.get("role_rank", -1)) != rank
            or record.get("target_columns_opened") != []
            or record.get("observer_selection_opened") is not False
            or record.get("name_or_board_join") is not False
        ):
            raise AssertionError(f"native target-free receipt changed at rank {rank}")
        for name in ("raw20.npy", "sigma.npy"):
            path = row_dir(rank) / name
            if path.stat().st_size != int(record["files"][name]["bytes"]):
                raise AssertionError(f"native payload size changed at rank {rank}: {name}")
        weights = FEATURE_STORE / f"rows/observer_fit/{rank:04d}/weights.npy"
        if weights.stat().st_size != 67_108_992:
            raise AssertionError(f"weight shape storage changed at rank {rank}")
        result_hash = sha256(result_path)
        aggregate.update(rank.to_bytes(4, "big"))
        aggregate.update(result_hash.encode("ascii"))
        records.append(record)
    if aggregate.hexdigest() != NATIVE_AGGREGATE_SHA256:
        raise AssertionError("native row aggregate changed")
    return records


def load_models(j, q):
    alpha_artifact = torch.load(ALPHA_MODEL, map_location="cpu", weights_only=False)
    if alpha_artifact.get("alpha") != 0.75:
        raise AssertionError("alpha model changed")
    state = alpha_artifact["state_dict"]
    base = j.TransportEnergyMultiStateResidual(
        state["centers"].numpy(), state["scales"].numpy(), state["layer_rms"].numpy()
    )
    base.load_state_dict(state, strict=True)
    base.eval()
    q_artifact = torch.load(Q_MODEL, map_location="cpu", weights_only=False)
    head = q.QSuffixHead(base.centers, base.scales, base.layer_rms)
    head.load_state_dict(q_artifact["state_dict"], strict=True)
    head.eval()
    return base, head


def main() -> None:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    records = verify_inputs()
    torch.manual_seed(0x51313015)
    np.random.seed(0x51313015)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    j = load_module("g11b_energy_exact_j", J_SOURCE)
    q = load_module("g11b_energy_exact_q", Q_SOURCE)
    base, head = load_models(j, q)
    network_energy = np.empty(688, dtype=np.float64)
    network_mean = np.empty(688, dtype=np.float64)
    network_max_abs = np.empty(688, dtype=np.float64)
    with torch.no_grad():
        for rank in range(688):
            raw_np = np.load(row_dir(rank) / "raw20.npy", mmap_mode="r", allow_pickle=False)
            sigma_np = np.load(row_dir(rank) / "sigma.npy", mmap_mode="r", allow_pickle=False)
            weights_np = np.load(
                FEATURE_STORE / f"rows/observer_fit/{rank:04d}/weights.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            raw = torch.from_numpy(np.ascontiguousarray(raw_np, dtype=np.float32))[None]
            sigma = torch.from_numpy(np.ascontiguousarray(sigma_np, dtype=np.float32))[None]
            weights = torch.from_numpy(np.ascontiguousarray(weights_np, dtype=np.float32))[None]
            correction = base(raw, sigma, weights, exposure=1.0)
            if not isinstance(correction, torch.Tensor):
                raise TypeError("alpha J returned non-tensor")
            a = correction[0].cpu().numpy().astype(np.float32, copy=False)
            qstate = np.zeros(1024, dtype=np.float32)
            for layer in SUFFIX:
                phi = np.asarray(raw_np[layer, :, 5], dtype=np.float64)
                layer_weights = np.asarray(weights_np[layer], dtype=np.float64)
                qmessage = phi * (qstate.astype(np.float64) @ layer_weights)
                a_message = phi * (a[layer - 1].astype(np.float64) @ layer_weights)
                base_q = (a[layer].astype(np.float64) - a_message).astype(np.float32)
                source = head.source(
                    torch.from_numpy(np.asarray(raw_np[layer]).copy()),
                    torch.from_numpy(np.asarray(sigma_np[layer]).copy()),
                    torch.from_numpy(a[layer]),
                    torch.from_numpy(base_q),
                    layer,
                ).cpu().numpy()
                qstate = (qmessage + source.astype(np.float64)).astype(np.float32)
            integrated = a[-1].astype(np.float64) + qstate.astype(np.float64)
            network_energy[rank] = float(np.mean(integrated * integrated))
            network_mean[rank] = float(np.mean(integrated))
            network_max_abs[rank] = float(np.max(np.abs(integrated)))
            if (rank + 1) % 32 == 0:
                print(json.dumps({"stage": "target_blind_energy", "rows": rank + 1}), flush=True)
    energy = float(np.mean(network_energy))
    cross = float((energy + R0_D1_RAW_MSE - R1_G11B_RAW_MSE) / 2.0)
    beta = float(np.clip(cross / energy, 0.0, 1.0)) if energy > 0 else 0.0
    implied_raw = float(R0_D1_RAW_MSE + beta * beta * energy - 2.0 * beta * cross)
    quantiles = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    result = {
        "schema": "q24-g11b-all688-target-blind-shrink-diagnostic-v1",
        "status": "COMPLETE_TARGET_BLIND_NO_SUBMISSION_AUTHORITY",
        "networks": 688,
        "definition": "integrated final correction = frozen alpha=.75 native-J correction plus frozen autonomous q10:q15 state",
        "fleet_mean_squared_integrated_correction_E": energy,
        "same_board_raw": {
            "R0_d1_submission_328586": R0_D1_RAW_MSE,
            "R1_g11b_submission_328616": R1_G11B_RAW_MSE,
            "inferred_error_correction_cross_moment_c": cross,
            "beta_star_clipped_0_1": beta,
            "implied_quadratic_raw_at_beta_star": implied_raw,
            "implied_raw_gain_vs_d1": R0_D1_RAW_MSE - implied_raw,
        },
        "network_distribution": {
            "energy_quantiles": {
                str(qvalue): float(np.quantile(network_energy, qvalue)) for qvalue in quantiles
            },
            "mean_correction_quantiles": {
                str(qvalue): float(np.quantile(network_mean, qvalue)) for qvalue in quantiles
            },
            "max_abs_correction_quantiles": {
                str(qvalue): float(np.quantile(network_max_abs, qvalue)) for qvalue in quantiles
            },
            "per_network": [
                {
                    "role_rank": rank,
                    "mlp_id": int(records[rank]["mlp_id"]),
                    "energy": float(network_energy[rank]),
                    "mean": float(network_mean[rank]),
                    "max_abs": float(network_max_abs[rank]),
                }
                for rank in range(688)
            ],
        },
        "interpretation_limit": "quadratic beta uses only two same-board aggregate raw scores and target-blind correction energy; it is a shrink lead, not validation or submission authority",
        "access_declarations": {
            "weights_only": True,
            "target_columns_opened": [],
            "scores_used": [328586, 328616],
            "names_opened": [],
            "network": False,
            "submission": False,
        },
        "bindings": {
            "program_sha256": sha256(Path(__file__)),
            "preregistration_sha256": sha256(PREREGISTRATION),
            "j_source_sha256": J_SOURCE_SHA256,
            "q_source_sha256": Q_SOURCE_SHA256,
            "alpha_model_sha256": ALPHA_MODEL_SHA256,
            "q_model_sha256": Q_MODEL_SHA256,
            "native_row_aggregate_sha256": NATIVE_AGGREGATE_SHA256,
        },
    }
    temporary = RESULT.with_name(f".{RESULT.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(RESULT)
    print(json.dumps({key: result[key] for key in ("status", "fleet_mean_squared_integrated_correction_E", "same_board_raw")}, sort_keys=True))


if __name__ == "__main__":
    main()
