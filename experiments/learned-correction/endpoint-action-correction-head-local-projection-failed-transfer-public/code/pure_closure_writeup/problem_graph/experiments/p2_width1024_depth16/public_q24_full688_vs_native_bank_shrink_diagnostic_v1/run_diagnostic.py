#!/Users/matt/anaconda3/bin/python3
"""One-shot same-688 native-bank versus promoted-qhead shrink diagnostic."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
BASE_RUNNER = P2 / "public_q24_qhead_full688_refit_v1/run_refit.py"
Q_SOURCE = P2 / "public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py"
J_SOURCE = P2 / "public_full688_transport_energy_coherence_cv_v1/run_experiment.py"
NATIVE_RESULT = P2 / "public_q24_native_transport_energy_final_v1/FINAL_FIT_RESULT.json"
NATIVE_MODEL = P2 / "public_q24_native_transport_energy_final_v1/FINAL_MODEL.pt"
QHEAD_RESULT = P2 / "public_q24_qhead_full688_refit_v1/RESULT.json"
QHEAD_MODEL = P2 / "public_q24_qhead_full688_refit_v1/FULL688_Q_SUFFIX_HEAD.pt"
ALPHA_MODEL = P2 / "public_q24_native_j_checkpoint_interpolation_v1/INTERPOLATED_MODEL.pt"
PREREGISTRATION = HERE / "PREREGISTRATION.json"
RESULT = HERE / "RESULT.json"

EXPECTED = {
    NATIVE_RESULT: "e8261895c5d8f5f1feb1e15f257612d1d1b36a90e8bde1ab05e2844100a8ba08",
    NATIVE_MODEL: "21ad1dede3373c90d2694158b6599c4c172d7181c72634a6ec8e362449b7d1f8",
    QHEAD_RESULT: "87d22be8d48d7c73821c02bb3562b3e8ad67955022df0e2185c2f94838e87a25",
    QHEAD_MODEL: "d57407e0534e0be8b29a8b3f8bb80f617e83671584ddb32af757a3631e0fc58f",
    ALPHA_MODEL: "fc65528720c7e1378ed7011b7b201f4086217a544b092de208ea1064656d9ef5",
    J_SOURCE: "1aedf41e6733ffb2a62a9881dc28945f97a4233372b272f6df6204cd525b8810",
    Q_SOURCE: "1e4c906936f4685bd800f922ded6c582af8213ecad1fbe41da651faff82be9c3",
    BASE_RUNNER: "7ffaae1a3d03a7d1f3fcfe3e9026d1674892c52291f7899978ff379f8da78cd2",
}
SUFFIX = tuple(range(10, 16))
NATIVE_RECEIPT_MSE = 2.9736792457285063e-8
STRICT_SCIENTIFIC_GATE = 2.9672528546945005e-8
BANK_BOARD_RAW = 2.9409214512554626e-8
BANK_BOARD_ADJ = 2.1258043062525995e-8
D3_MAX_COMPUTE_RATIO = 0.8839413768621204
MAX_RSS = 8 * (1 << 30)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(tag: str, path: Path):
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[tag] = module
    spec.loader.exec_module(module)
    return module


def resident_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    return value


def load_transport(j, path: Path, expected_schema: str):
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("schema") != expected_schema:
        raise AssertionError(f"transport schema changed: {path}")
    state = artifact["state_dict"]
    model = j.TransportEnergyMultiStateResidual(
        state["centers"].cpu().numpy(),
        state["scales"].cpu().numpy(),
        state["layer_rms"].cpu().numpy(),
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_qhead(q, alpha):
    artifact = torch.load(QHEAD_MODEL, map_location="cpu", weights_only=False)
    if (
        artifact.get("schema") != "q24-native-j-q10q15-residual-head-checkpoint-v1"
        or artifact.get("refit_kind") != "all688-fixed-epoch13-after-four-fold-oof-gate"
    ):
        raise AssertionError("full688 q-head schema changed")
    model = q.QSuffixHead(alpha.centers, alpha.scales, alpha.layer_rms)
    model.load_state_dict(artifact["state_dict"], strict=True)
    model.eval()
    return model


def beta_for(cross: np.ndarray, energy: np.ndarray, selected: np.ndarray) -> float:
    denominator = float(np.sum(energy[selected], dtype=np.float64))
    if not math.isfinite(denominator) or denominator <= 0.0:
        return 0.0
    return float(np.clip(np.sum(cross[selected], dtype=np.float64) / denominator, 0.0, 1.0))


def mse_at(base: np.ndarray, cross: np.ndarray, energy: np.ndarray, beta: float) -> np.ndarray:
    return base - 2.0 * beta * cross + beta * beta * energy


def write_once(payload: dict[str, object]) -> None:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    temporary = RESULT.with_name(f".{RESULT.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(RESULT)


def main() -> None:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if prereg.get("status") != "FROZEN_BEFORE_DIAGNOSTIC_TARGET_REOPEN":
        raise AssertionError("preregistration status changed")
    for path, expected in EXPECTED.items():
        observed = sha256(path)
        if observed != expected:
            raise AssertionError((str(path), observed, expected))

    base_runner = load_module("full688_shrink_base_runner", BASE_RUNNER)
    j = load_module("full688_shrink_j", J_SOURCE)
    q = load_module("full688_shrink_q", Q_SOURCE)
    records, _ = base_runner.verify_native_rows()
    folds = base_runner.balanced_folds(records)
    targets = base_runner.load_targets(records)
    hosts = np.ascontiguousarray(
        np.stack(
            [
                np.load(base_runner.row_dir(rank) / "host.npy", allow_pickle=False)
                for rank in range(688)
            ]
        ),
        dtype=np.float32,
    )
    error = np.ascontiguousarray(targets - hosts, dtype=np.float32)
    dataset = base_runner.All688Dataset(records)

    torch.manual_seed(base_runner.MODEL_SEED)
    np.random.seed(base_runner.MODEL_SEED & 0xFFFF_FFFF)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    native = load_transport(j, NATIVE_MODEL, "p2r0f2j-q24-native-all688-final-model-v1")
    alpha = load_transport(j, ALPHA_MODEL, "q24-native-j-interpolated-checkpoint-v1")
    qhead = load_qhead(q, alpha)

    native_mse = np.empty(688, dtype=np.float64)
    candidate_mse = np.empty(688, dtype=np.float64)
    energy = np.empty(688, dtype=np.float64)
    cross = np.empty(688, dtype=np.float64)
    correction_mean = np.empty(688, dtype=np.float64)
    correction_rms = np.empty(688, dtype=np.float64)
    with torch.no_grad():
        for rank in range(688):
            raw_t, sigma_t, weights_t, _ = j.load_batch(
                dataset, np.asarray([rank], dtype=np.int64), error
            )
            native_a = native(raw_t, sigma_t, weights_t, exposure=1.0)[0].cpu().numpy()
            alpha_a = alpha(raw_t, sigma_t, weights_t, exposure=1.0)[0].cpu().numpy()
            raw = raw_t[0].cpu().numpy()
            sigma = sigma_t[0].cpu().numpy()
            weights = weights_t[0].cpu().numpy()
            q_state = np.zeros(1024, dtype=np.float32)
            for layer in SUFFIX:
                phi = raw[layer, :, 5].astype(np.float64)
                base_message = phi * (
                    alpha_a[layer - 1].astype(np.float64) @ weights[layer].astype(np.float64)
                )
                base_q = (alpha_a[layer].astype(np.float64) - base_message).astype(np.float32)
                source = qhead.source(
                    torch.from_numpy(raw[layer]),
                    torch.from_numpy(sigma[layer]),
                    torch.from_numpy(alpha_a[layer]),
                    torch.from_numpy(base_q),
                    layer,
                ).cpu().numpy()
                transported = phi * (
                    q_state.astype(np.float64) @ weights[layer].astype(np.float64)
                )
                q_state = (transported + source.astype(np.float64)).astype(np.float32)

            native_correction = native_a[-1].astype(np.float64)
            full_correction = alpha_a[-1].astype(np.float64) + q_state.astype(np.float64)
            delta = full_correction - native_correction
            residual = error[rank, -1].astype(np.float64) - native_correction
            native_mse[rank] = np.mean(residual * residual)
            candidate_mse[rank] = np.mean((residual - delta) ** 2)
            energy[rank] = np.mean(delta * delta)
            cross[rank] = np.mean(residual * delta)
            correction_mean[rank] = np.mean(delta)
            correction_rms[rank] = math.sqrt(energy[rank])
            if (rank + 1) % 32 == 0:
                print(json.dumps({"stage": "inference", "rows": rank + 1}), flush=True)
            if resident_bytes() >= MAX_RSS:
                raise MemoryError(f"RSS gate failed at rank {rank}: {resident_bytes()}")

    observed_native = float(np.mean(native_mse))
    # The frozen final-fit receipt scored larger Torch batches, while this
    # diagnostic deliberately uses one-network deployment-shaped inference.
    # BLAS accumulation changes the aggregate by about 9e-18 absolute only.
    if not np.isclose(observed_native, NATIVE_RECEIPT_MSE, rtol=0.0, atol=2e-17):
        raise AssertionError(("native receipt parity", observed_native, NATIVE_RECEIPT_MSE))
    candidate_all = float(np.mean(candidate_mse))
    folds_out = []
    oof_mse = np.empty(688, dtype=np.float64)
    oof_beta = np.empty(688, dtype=np.float64)
    all_ranks = np.arange(688, dtype=np.int64)
    for fold in range(4):
        train = all_ranks[folds != fold]
        held = all_ranks[folds == fold]
        beta = beta_for(cross, energy, train)
        scores = mse_at(native_mse[held], cross[held], energy[held], beta)
        oof_mse[held] = scores
        oof_beta[held] = beta
        folds_out.append(
            {
                "fold": fold,
                "train_networks": int(len(train)),
                "held_networks": int(len(held)),
                "beta_selected_on_other_folds": beta,
                "native_mse": float(np.mean(native_mse[held])),
                "unshrunk_full688_final_model_mse": float(np.mean(candidate_mse[held])),
                "held_shrunk_mse": float(np.mean(scores)),
                "held_relative_gain_vs_native": float(
                    1.0 - np.mean(scores) / np.mean(native_mse[held])
                ),
                "network_win_fraction_vs_native": float(np.mean(scores < native_mse[held])),
            }
        )
    final_beta = beta_for(cross, energy, all_ranks)
    final_scores = mse_at(native_mse, cross, energy, final_beta)
    oof_mean = float(np.mean(oof_mse))
    optimistic_d3_adj = BANK_BOARD_ADJ * (oof_mean / BANK_BOARD_RAW) * D3_MAX_COMPUTE_RATIO
    gates = {
        "oof_below_measured_native_all688": oof_mean < NATIVE_RECEIPT_MSE,
        "oof_below_stricter_scientific_threshold": oof_mean < STRICT_SCIENTIFIC_GATE,
        "every_fold_improves_native": all(
            row["held_shrunk_mse"] < row["native_mse"] for row in folds_out
        ),
        "network_win_fraction_above_half": float(np.mean(oof_mse < native_mse)) > 0.5,
    }
    result = {
        "schema": "q24-full688-vs-native-bank-global-shrink-diagnostic-result-v1",
        "status": "PASS_BROAD_ABSOLUTE_GAIN" if all(gates.values()) else "CLOSED_NO_BROAD_ABSOLUTE_GAIN",
        "access_declarations": {
            "role": "observer_fit",
            "observer_fit_targets_reopened_under_existing_refit_authority": True,
            "observer_selection_targets_opened": [],
            "board_opened": False,
            "submission": False,
            "training": False,
            "estimator_calls": 0,
        },
        "semantic_provenance": {
            "measured_native_all688_mse": NATIVE_RECEIPT_MSE,
            "deployment_shaped_replay_native_mse": observed_native,
            "deployment_shaped_minus_receipt_mse": observed_native - NATIVE_RECEIPT_MSE,
            "measured_native_all688_receipt": str(NATIVE_RESULT.relative_to(P2)),
            "stricter_threshold": STRICT_SCIENTIFIC_GATE,
            "stricter_threshold_meaning": "frozen 10%-below-P2R0F2D OOF scientific gate, not a second native-bank measurement",
            "qhead_oof_receipt_mse_for_context_only": 3.066199675740282e-8,
            "endpoint_models_are_all688_fits": True,
            "only_beta_is_strictly_out_of_fold": True,
        },
        "same_688_endpoint_comparison": {
            "networks": 688,
            "native_bank_final_mse": observed_native,
            "full688_final_fit_qhead_mse": candidate_all,
            "full688_final_fit_ratio_to_native": candidate_all / observed_native,
            "full688_final_fit_network_win_fraction": float(np.mean(candidate_mse < native_mse)),
            "delta_correction_mean_squared_energy": float(np.mean(energy)),
            "delta_correction_rms": math.sqrt(float(np.mean(energy))),
            "delta_correction_network_rms_quantiles": {
                "minimum": float(np.min(correction_rms)),
                "q25": float(np.quantile(correction_rms, 0.25)),
                "median": float(np.median(correction_rms)),
                "q75": float(np.quantile(correction_rms, 0.75)),
                "maximum": float(np.max(correction_rms)),
            },
            "delta_correction_network_mean_quantiles": {
                "minimum": float(np.min(correction_mean)),
                "median": float(np.median(correction_mean)),
                "maximum": float(np.max(correction_mean)),
            },
        },
        "strict_oof_beta_diagnostic": {
            "folds": folds_out,
            "oof_mse": oof_mean,
            "oof_ratio_to_native": oof_mean / observed_native,
            "oof_network_win_fraction_vs_native": float(np.mean(oof_mse < native_mse)),
            "oof_beta_values": [float(row["beta_selected_on_other_folds"]) for row in folds_out],
            "final_beta_frozen_from_all_688": final_beta,
            "final_beta_in_sample_mse_diagnostic": float(np.mean(final_scores)),
            "gates": gates,
            "conjunctive_pass": all(gates.values()),
        },
        "score_projection": {
            "optimistic_assumption": "scalar blend has unchanged D3+qhead fixture compute; this is accounting-only and not deployment authority",
            "bank_board_raw": BANK_BOARD_RAW,
            "bank_board_adjusted": BANK_BOARD_ADJ,
            "d3_max_fixture_compute_ratio": D3_MAX_COMPUTE_RATIO,
            "optimistic_d3_adjusted_from_oof_ratio": optimistic_d3_adj,
            "deployability_caveat": "Exact native-to-qhead interpolation requires both distinct transport endpoints unless a separately proved fused spelling exists; no zero-added-FLOP source is established.",
        },
        "resource": {"maximum_process_rss_bytes": resident_bytes()},
        "bindings": {
            "preregistration_sha256": sha256(PREREGISTRATION),
            "program_sha256": sha256(Path(__file__)),
            **{path.name + "_sha256": value for path, value in EXPECTED.items()},
        },
    }
    write_once(result)
    print(json.dumps({"status": result["status"], "result": str(RESULT)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
