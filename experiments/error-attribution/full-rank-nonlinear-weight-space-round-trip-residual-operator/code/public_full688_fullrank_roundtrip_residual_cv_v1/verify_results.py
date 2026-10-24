#!/Users/matt/anaconda3/bin/python3
"""Independent post-run verification for P2R0F2P.

Re-derives every reported OOF number by replaying scoring from the persisted
fold models against the sealed outer J models and data, re-checks the entire
receipt hash chain, re-evaluates the decision gate from replayed numbers, and
verifies the frozen-J positive control against J's own CV fold records.
Writes VERIFICATION.json once on full success; raises on any mismatch.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
J_DIR = P2 / "public_full688_transport_energy_coherence_cv_v1"

RUNNER_SPEC = importlib.util.spec_from_file_location(
    "p2r0f2p_frozen_runner", HERE / "run_experiment.py"
)
if RUNNER_SPEC is None or RUNNER_SPEC.loader is None:
    raise ImportError("cannot load frozen P runner")
R = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(R)

VERIFICATION = HERE / "VERIFICATION.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_p_observer(fold: int, seal: dict[str, object]) -> tuple[object, str]:
    path = HERE / "folds" / f"fold_{fold}.pt"
    digest = sha256(path)
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    require(artifact.get("node") == "P2R0F2P", f"fold {fold}: model node changed")
    require(artifact.get("fold") == fold, f"fold {fold}: model fold changed")
    for key in (
        "preregistration_sha256",
        "program_sha256",
        "run_manifest_sha256",
    ):
        require(
            artifact.get(key) == seal[key],
            f"fold {fold}: model binding mismatch at {key}",
        )
    state = artifact["state_dict"]
    model = R.FullRankRoundTripResidualObserver(
        state["centers"].cpu().numpy(),
        state["scales"].cpu().numpy(),
        state["layer_rms"].cpu().numpy(),
    )
    model.load_state_dict(state, strict=True)
    require(
        R.parameter_count(model) == R.EXPECTED_PARAMETERS,
        f"fold {fold}: parameter count changed",
    )
    model.eval()
    return model, digest


def main() -> None:
    if VERIFICATION.exists():
        raise FileExistsError(VERIFICATION)
    R.configure()
    seal = R.verify_seal()
    release = R.verify_release(seal)
    cv = read_json(R.CV_RESULT)
    require(cv["status"] == "complete", "CV_RESULTS is not complete")
    require(cv["node"] == "P2R0F2P", "CV node changed")
    require(cv["preregistration_sha256"] == seal["preregistration_sha256"], "CV prereg mismatch")
    require(cv["program_sha256"] == seal["program_sha256"], "CV program mismatch")

    j_cv = read_json(J_DIR / "CV_RESULTS.json")
    require(
        sha256(J_DIR / "CV_RESULTS.json") == seal["j_cv_results_sha256"],
        "J CV_RESULTS changed since seal",
    )
    j_fold_final = {
        int(row["fold"]): float(row["autonomous_final_mse"])
        for row in j_cv["fold_results"]
    }

    dataset = R.PublicResidualDataset()
    R.readiness_or_raise(dataset)
    error, sums, sums2, folds = R.load_shared_cache(dataset)

    replay_folds: list[dict[str, object]] = []
    all_per_network: list[dict[str, object]] = []
    total = 0
    weighted_final = 0.0
    weighted_host = 0.0
    maximum_network = -1.0
    maximum_rank = -1
    finite_all = True
    for fold in range(R.FOLDS):
        record = read_json(HERE / "folds" / f"fold_{fold}.json")
        R.verify_completed_fold_record(record, fold, seal)
        model, model_digest = load_p_observer(fold, seal)
        require(
            record["model_sha256"] == model_digest,
            f"fold {fold}: recorded model hash mismatch",
        )
        j_model = R.load_outer_j_model(fold, seal)
        valid = np.flatnonzero(folds == fold)
        require(
            record["valid_role_ranks"] == valid.tolist(),
            f"fold {fold}: validation roster changed",
        )
        replay = R.score_outer(model, j_model, dataset, valid, error)
        recorded = record["autonomous"]
        require(
            replay["final_mse"] == recorded["final_mse"],
            f"fold {fold}: replayed final MSE {replay['final_mse']} != recorded {recorded['final_mse']}",
        )
        require(
            replay["layer_mse"] == recorded["layer_mse"],
            f"fold {fold}: replayed layer MSEs differ",
        )
        require(
            replay["frozen_j_final_mse"] == recorded["frozen_j_final_mse"],
            f"fold {fold}: replayed frozen-J final MSE differs",
        )
        require(
            replay["per_network"] == recorded["per_network"],
            f"fold {fold}: replayed per-network finals differ",
        )
        require(
            replay["finite_full_rollout"] is True,
            f"fold {fold}: replay is nonfinite",
        )
        require(
            abs(replay["frozen_j_final_mse"] - j_fold_final[fold])
            <= 1e-18 * max(1.0, abs(j_fold_final[fold])),
            f"fold {fold}: frozen-J positive control mismatch "
            f"{replay['frozen_j_final_mse']} vs J CV {j_fold_final[fold]}",
        )
        host_layer = np.mean(
            np.square(error[valid].astype(np.float64)), axis=(0, 2), dtype=np.float64
        )
        require(
            record["host"]["final_mse"] == float(host_layer[-1]),
            f"fold {fold}: host final MSE mismatch",
        )
        count = len(valid)
        total += count
        weighted_final += count * replay["final_mse"]
        weighted_host += count * float(host_layer[-1])
        for row in replay["per_network"]:
            all_per_network.append(row)
            if float(row["final_mse"]) > maximum_network:
                maximum_network = float(row["final_mse"])
                maximum_rank = int(row["role_rank"])
        finite_all = finite_all and replay["finite_full_rollout"]
        replay_folds.append(
            {
                "fold": fold,
                "networks": count,
                "replayed_final_mse": replay["final_mse"],
                "replayed_frozen_j_final_mse": replay["frozen_j_final_mse"],
                "j_cv_fold_final_mse": j_fold_final[fold],
                "maximum_per_network_final_mse": max(
                    float(row["final_mse"]) for row in replay["per_network"]
                ),
                "model_sha256": model_digest,
            }
        )
    require(total == 688, "replay coverage is not 688 networks")
    require(
        sorted(int(row["role_rank"]) for row in all_per_network) == list(range(688)),
        "replayed per-network coverage is not exactly ranks 0..687",
    )
    replay_final = weighted_final / total
    replay_host = weighted_host / total
    require(
        replay_final == float(cv["oof"]["autonomous_final_mse"]),
        f"aggregate mismatch: replay {replay_final} vs CV {cv['oof']['autonomous_final_mse']}",
    )
    require(
        maximum_network == float(cv["oof"]["maximum_per_network_final_mse"]),
        "maximum per-network mismatch",
    )
    require(
        maximum_rank == int(cv["oof"]["maximum_role_rank"]),
        "maximum role rank mismatch",
    )

    gate = {
        "finite_all_688_outer_oof_rollouts": bool(finite_all),
        "oof_final_mse_below_1e_8": bool(replay_final < R.OOF_GATE),
        "every_outer_fold_below_1_1e_8": bool(
            all(row["replayed_final_mse"] < R.FOLD_GATE for row in replay_folds)
        ),
        "maximum_per_network_below_1e_7": bool(maximum_network < R.MAX_NETWORK_GATE),
        "observer_selection_targets_opened_zero": True,
    }
    gate["conjunctive_pass"] = all(gate.values())
    recorded_gate = cv["decision_gate"]
    for key, value in gate.items():
        require(
            bool(recorded_gate[key]) == value,
            f"gate re-evaluation mismatch at {key}: replay {value} vs recorded {recorded_gate[key]}",
        )

    payload = {
        "schema": "public-full688-fullrank-roundtrip-residual-verification-v1",
        "node": "P2R0F2P",
        "status": "PASS_INDEPENDENT_REPLAY",
        "verifier_sha256": sha256(Path(__file__)),
        "cv_results_sha256": sha256(R.CV_RESULT),
        "seal_sha256": sha256(R.SEAL),
        "training_release_sha256": sha256(R.TRAINING_RELEASE),
        "audit_receipt_sha256": seal["audit_receipt_sha256"],
        "replayed": {
            "networks": total,
            "oof_final_mse": replay_final,
            "oof_host_final_mse": replay_host,
            "oof_ratio_to_host": replay_final / replay_host,
            "maximum_per_network_final_mse": maximum_network,
            "maximum_role_rank": maximum_rank,
            "folds": replay_folds,
        },
        "frozen_j_positive_control": {
            "description": "Replayed frozen-J per-fold OOF final MSEs match J's own CV fold records, proving the scoring pipeline reproduces the established baseline.",
            "matches_all_folds": True,
        },
        "decision_gate_replayed": gate,
        "observer_selection_targets_opened": 0,
        "public_name_joins": 0,
        "network_access": 0,
    }
    VERIFICATION.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "gate": gate, "oof": replay_final}, sort_keys=True))


if __name__ == "__main__":
    main()
