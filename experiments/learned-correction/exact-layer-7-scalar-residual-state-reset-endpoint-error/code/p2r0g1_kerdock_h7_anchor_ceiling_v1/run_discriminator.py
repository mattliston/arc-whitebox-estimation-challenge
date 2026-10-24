#!/Users/matt/anaconda3/bin/python3
"""One frozen post-ReLU h7 exact-state-reset ceiling."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "public_exposure_trained_residual_state_v1"
SOURCE_PROGRAM = SOURCE / "run_experiment.py"
SOURCE_MODEL = SOURCE / "MODEL.pt"
SOURCE_RESULT = SOURCE / "RESULT.json"
PREREGISTRATION = HERE / "PREREGISTRATION.json"
RESULT = HERE / "RESULT.json"

ANCHOR_LAYER = 6
EXPECTED_MODEL_SHA256 = "c998c4410bf36d3b4f1152c45d86fd4c654c7e05abe85e297f07b15f16d58cc7"
EXPECTED_SOURCE_RESULT_SHA256 = "fe2f2f3c8cfa890d416b520f443fbbe475ea8fabafbd16956006a9154b100440"
EXPECTED_SOURCE_PROGRAM_SHA256 = "5c16a68bb907b0e3c945bce883a877c99743fb40fffca3d8039054fd2eb566e1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source():
    spec = importlib.util.spec_from_file_location("frozen_residual_source", SOURCE_PROGRAM)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE_PROGRAM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_single_reset(model, source, panel: dict[str, object], ranks: np.ndarray) -> dict[str, object]:
    raw = panel["raw"]
    sigma = panel["sigma"]
    error = panel["error"]
    paths = panel["weight_paths"]
    assert isinstance(raw, np.ndarray)
    assert isinstance(sigma, np.ndarray)
    assert isinstance(error, np.ndarray)
    assert isinstance(paths, list)

    layer_sse = np.zeros(source.DEPTH, dtype=np.float64)
    per_network: list[float] = []
    maximum_absolute = 0.0
    finite = True
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(ranks), 2):
            group = ranks[offset : offset + 2]
            batch_raw = source.tensor_rows(raw, group)
            batch_sigma = source.tensor_rows(sigma, group)
            truth = source.tensor_rows(error, group)
            weights = source.load_weights(paths, group)
            state = torch.zeros(len(group), source.WIDTH, dtype=batch_raw.dtype)
            outputs: list[torch.Tensor] = []
            gain = model.transport_gain()
            for layer in range(source.DEPTH):
                if layer < source.FIRST_CORRECTED_LAYER:
                    state = torch.zeros_like(state)
                    outputs.append(state)
                    continue
                prior = state
                message = torch.bmm(prior[:, None, :], weights[:, layer]).squeeze(1)
                message = batch_raw[:, layer, :, 5] * message
                local = model.innovation(batch_raw[:, layer], batch_sigma[:, layer], message, layer)
                state = gain[layer] * message + local
                if layer == ANCHOR_LAYER:
                    state = truth[:, layer].clone()
                outputs.append(state)
            prediction = torch.stack(outputs, dim=1)
            finite = finite and bool(torch.isfinite(prediction).all())
            maximum_absolute = max(maximum_absolute, float(prediction.abs().max()))
            squared = (prediction - truth).double().square()
            layer_sse += squared.sum(dim=(0, 2)).cpu().numpy()
            per_network.extend(squared[:, -1].mean(dim=1).cpu().numpy().tolist())
    layer_mse = layer_sse / (len(ranks) * source.WIDTH)
    return {
        "finite_full_rollout": finite,
        "maximum_absolute_correction": maximum_absolute,
        "final_mse": float(layer_mse[-1]),
        "layer_mse": layer_mse.tolist(),
        "per_network_final_mse": per_network,
    }


def main() -> None:
    if RESULT.exists():
        raise FileExistsError("single-run result already exists")
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if prereg.get("status") != "frozen-before-metrics":
        raise AssertionError("preregistration is not frozen")
    if sha256(SOURCE_MODEL) != EXPECTED_MODEL_SHA256:
        raise AssertionError("source model changed")
    if sha256(SOURCE_RESULT) != EXPECTED_SOURCE_RESULT_SHA256:
        raise AssertionError("source result changed")
    if sha256(SOURCE_PROGRAM) != EXPECTED_SOURCE_PROGRAM_SHA256:
        raise AssertionError("source program changed")

    started = time.perf_counter()
    source = load_source()
    source.configure()
    panel = source.load_panel()
    raw = panel["raw"]
    error = panel["error"]
    assert isinstance(raw, np.ndarray)
    assert isinstance(error, np.ndarray)
    centers, scales, layer_rms = source.fit_moments(raw, error)
    model = source.LocalInnovationTransport(centers, scales, layer_rms)
    artifact = torch.load(SOURCE_MODEL, map_location="cpu", weights_only=True)
    model.load_state_dict(artifact["state_dict"], strict=True)

    ranks = np.arange(source.FIT_STOP, source.VALID_STOP)
    single_reset = score_single_reset(model, source, panel, ranks)
    autonomous = source.score(model, panel, ranks, exposure=1.0)
    teacher = source.score(model, panel, ranks, exposure=0.0)
    source_receipt = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))
    expected_auto = float(source_receipt["metrics"]["validation_autonomous"]["final_mse"])
    expected_teacher = float(
        source_receipt["metrics"]["validation_same_checkpoint_teacher_forced"]["final_mse"]
    )
    if not np.isclose(float(autonomous["final_mse"]), expected_auto, rtol=0.0, atol=1e-15):
        raise AssertionError("autonomous comparator did not reproduce")
    if not np.isclose(float(teacher["final_mse"]), expected_teacher, rtol=0.0, atol=1e-15):
        raise AssertionError("teacher-forced comparator did not reproduce")

    reset_mse = float(single_reset["final_mse"])
    auto_mse = float(autonomous["final_mse"])
    teacher_mse = float(teacher["final_mse"])
    gap = auto_mse - teacher_mse
    gap_closed = (auto_mse - reset_mse) / gap if gap > 0.0 else float("nan")
    result = {
        "schema": "p2r0g1-kerdock-h7-anchor-ceiling-result-v1",
        "status": "complete",
        "anchor": {
            "layer_one_based": ANCHOR_LAYER + 1,
            "layer_zero_based": ANCHOR_LAYER,
            "location": "post-ReLU h_7",
            "resets": 1,
            "autonomous_suffix_layers": list(range(ANCHOR_LAYER + 2, source.DEPTH + 1)),
        },
        "split": {
            "role": "observer_fit",
            "development_validation": [source.FIT_STOP, source.VALID_STOP],
            "observer_selection_targets_opened": 0,
            "observer_fit_targets_at_or_above_rank_200_opened": 0,
            "deployment_or_board_truth_opened": 0,
            "network_names_opened": 0,
        },
        "metrics": {
            "autonomous_final_mse": auto_mse,
            "single_exact_h7_reset": single_reset,
            "teacher_forced_final_mse": teacher_mse,
            "autonomous_to_teacher_mse_gap": gap,
            "fraction_of_autonomous_to_teacher_mse_gap_closed": gap_closed,
            "single_reset_ratio_to_autonomous": reset_mse / auto_mse,
        },
        "gates": {
            "finite": bool(single_reset["finite_full_rollout"]),
            "single_exact_h7_reset_reaches_1e_8": reset_mse < 1e-8,
            "single_exact_h7_reset_reaches_teacher_plus_10_percent_of_gap": reset_mse <= teacher_mse + 0.1 * gap,
        },
        "interpretation_boundary": "This is an oracle sufficiency ceiling. It does not show that a finite Kerdock projection can recover the exact h_7 residual, and it is not submission-eligible.",
        "provenance": {
            "preregistration_sha256": sha256(PREREGISTRATION),
            "source_model_sha256": sha256(SOURCE_MODEL),
            "source_result_sha256": sha256(SOURCE_RESULT),
            "source_program_sha256": sha256(SOURCE_PROGRAM),
            "program_sha256": sha256(Path(__file__)),
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(RESULT), "single_reset_final_mse": reset_mse, "gates": result["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
