#!/Users/matt/anaconda3/bin/python3
"""Full-data compact local-innovation + physical-transport experiment.

The model keeps one autonomous physical residual state.  The actual layer
weight and Gaussian ReLU gate transport that state; a small shared pointwise
network predicts the missing local innovation from the target-blind raw20
features and the arriving physical message.  Network identity never enters
the model.

Protocol:
  1. four group folds over the 688 observer-fit networks (never neuron folds),
  2. one fixed architecture and fixed training schedule,
  3. refit on all 688 only after the CV estimate exists,
  4. open the separate 88-network observer-selection role exactly once.

The expensive modes refuse to start while the host scan has missing or
unrepaired tail rows.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from dataset import (
    DEPTH,
    LOCAL_FEATURES,
    WIDTH,
    PublicResidualDataset,
)


HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
CV_RESULT = HERE / "CV_RESULTS.json"
FINAL_ARTIFACT = HERE / "FINAL_MODEL.pt"
SELECTION_RESULT = HERE / "SELECTION_RESULTS.json"

SEED = 0x688_2026
FOLDS = 4
HIDDEN = 64
EPOCHS = 8
SAMPLES_PER_LAYER = 256
NETWORKS_PER_STEP = 4
LEARNING_RATE = 1.5e-3
NOISE_RMS = 0.35
FIRST_CORRECTED_LAYER = 2


def configure() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(12)
    torch.use_deterministic_algorithms(True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fold_of(mlp_id: int) -> int:
    payload = f"{SEED}:{int(mlp_id)}".encode("ascii")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "little") % FOLDS


class LocalInnovationTransport(nn.Module):
    """One exact W message plus a shared nonlinear local innovation."""

    def __init__(self, centers: np.ndarray, scales: np.ndarray, layer_rms: np.ndarray):
        super().__init__()
        # raw[:,0] is the exact depth coordinate.  The other 19 coordinates
        # use fit-only per-layer moments.  Two recurrent message coordinates
        # are appended at runtime.
        self.register_buffer("centers", torch.from_numpy(centers.astype(np.float32)))
        self.register_buffer("scales", torch.from_numpy(scales.astype(np.float32)))
        self.register_buffer("layer_rms", torch.from_numpy(layer_rms.astype(np.float32)))
        inputs = LOCAL_FEATURES + 2
        self.trunk = nn.Sequential(
            nn.Linear(inputs, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.SiLU(),
        )
        self.head_weight = nn.Parameter(torch.empty(DEPTH, HIDDEN))
        self.head_bias = nn.Parameter(torch.zeros(DEPTH))
        nn.init.normal_(self.head_weight, mean=0.0, std=2.0e-3)
        # Physical tangent gain, conservatively bounded around one.
        self.transport_delta = nn.Parameter(torch.zeros(DEPTH))

    def transport_gain(self) -> torch.Tensor:
        return 1.0 + 0.30 * torch.tanh(self.transport_delta)

    def local_features(
        self,
        raw_layer: torch.Tensor,
        sigma_layer: torch.Tensor,
        message: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        tail = torch.clamp(
            (raw_layer[..., 1:] - self.centers[layer]) / self.scales[layer],
            -8.0,
            8.0,
        )
        depth = raw_layer[..., :1]
        rms = self.layer_rms[layer]
        message_rms = torch.clamp(message / rms, -8.0, 8.0)[..., None]
        message_sigma = torch.clamp(
            message / torch.clamp(sigma_layer * rms, min=2.0**-30), -8.0, 8.0
        )[..., None]
        return torch.cat((depth, tail, message_rms, message_sigma), dim=-1)

    def innovation(
        self,
        raw_layer: torch.Tensor,
        sigma_layer: torch.Tensor,
        message: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        features = self.local_features(raw_layer, sigma_layer, message, layer)
        hidden = self.trunk(features)
        standard = torch.einsum("...h,h->...", hidden, self.head_weight[layer])
        standard = standard + self.head_bias[layer]
        return sigma_layer * self.layer_rms[layer] * standard

    def forward(
        self,
        raw: torch.Tensor,
        sigma: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        batch = raw.shape[0]
        state = torch.zeros(batch, WIDTH, dtype=raw.dtype, device=raw.device)
        outputs: list[torch.Tensor] = []
        gain = self.transport_gain()
        for layer in range(DEPTH):
            if layer < FIRST_CORRECTED_LAYER:
                state = torch.zeros_like(state)
                outputs.append(state)
                continue
            gate = raw[:, layer, :, 5]
            message = torch.bmm(state[:, None, :], weights[:, layer]).squeeze(1)
            message = gate * message
            local = self.innovation(raw[:, layer], sigma[:, layer], message, layer)
            state = gain[layer] * message + local
            outputs.append(state)
        return torch.stack(outputs, dim=1)


def _row_sufficient_statistics(dataset: PublicResidualDataset) -> tuple[np.ndarray, np.ndarray]:
    """Cache raw-feature sums/squares per network/layer; about 3.4 MiB."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / "fit_raw_sufficient_stats.npz"
    if path.is_file():
        payload = np.load(path, allow_pickle=False)
        return payload["sum"], payload["sum2"]
    sums = np.empty((dataset.total, DEPTH, LOCAL_FEATURES - 1), dtype=np.float64)
    sums2 = np.empty_like(sums)
    for rank in range(dataset.total):
        raw = np.asarray(dataset.load(rank).raw_locals[..., 1:], dtype=np.float64)
        sums[rank] = raw.sum(axis=1, dtype=np.float64)
        sums2[rank] = np.square(raw).sum(axis=1, dtype=np.float64)
        if (rank + 1) % 50 == 0:
            print(f"raw stats {rank + 1}/{dataset.total}", flush=True)
    np.savez(path, sum=sums, sum2=sums2)
    return sums, sums2


def moments_for(
    ranks: np.ndarray, sums: np.ndarray, sums2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    count = float(len(ranks) * WIDTH)
    total = sums[ranks].sum(axis=0, dtype=np.float64)
    total2 = sums2[ranks].sum(axis=0, dtype=np.float64)
    centers = total / count
    variance = np.maximum(total2 / count - centers * centers, (2.0**-20) ** 2)
    return centers, np.sqrt(variance)


def host_and_error(dataset: PublicResidualDataset) -> tuple[np.ndarray, np.ndarray]:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / "fit_patched_host.npy"
    if not path.is_file():
        mapped = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(dataset.total, DEPTH, WIDTH)
        )
        for rank in range(dataset.total):
            mapped[rank] = dataset.load(rank).host
        mapped.flush()
        del mapped
    host = np.load(path, mmap_mode="r", allow_pickle=False)
    error = np.ascontiguousarray(dataset.targets - host, dtype=np.float32)
    return host, error


def layer_rms_for(error: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    rms = np.sqrt(
        np.mean(np.square(error[ranks].astype(np.float64)), axis=(0, 2), dtype=np.float64)
    )
    return np.maximum(rms, 2.0**-20).astype(np.float32)


def teacher_messages(dataset: PublicResidualDataset, error: np.ndarray) -> np.ndarray:
    """One expensive weight pass, reused by every fold and every epoch."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / "fit_teacher_physical_messages.npy"
    if path.is_file():
        return np.load(path, mmap_mode="r", allow_pickle=False)
    mapped = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(dataset.total, DEPTH, WIDTH)
    )
    mapped[:] = 0.0
    for rank in range(dataset.total):
        row = dataset.load(rank)
        weights = np.load(row.weights_path, mmap_mode="r", allow_pickle=False)
        # Layer 2 receives an exactly zero autonomous state because L0/L1 are
        # hard-zero anchors.  Teacher transport starts one transition later.
        for layer in range(FIRST_CORRECTED_LAYER + 1, DEPTH):
            transported = error[rank, layer - 1].astype(np.float64) @ weights[layer]
            mapped[rank, layer] = (
                np.asarray(row.raw_locals[layer, :, 5], dtype=np.float64) * transported
            ).astype(np.float32)
        if (rank + 1) % 50 == 0:
            mapped.flush()
            print(f"teacher messages {rank + 1}/{dataset.total}", flush=True)
    mapped.flush()
    del mapped
    return np.load(path, mmap_mode="r", allow_pickle=False)


def sample_indices(rank: int, layer: int, epoch: int) -> np.ndarray:
    seed = SEED ^ (rank * 0x9E3779B1) ^ (layer * 0x85EBCA77) ^ (epoch * 0xC2B2AE3D)
    rng = np.random.default_rng(seed & 0xFFFF_FFFF)
    return rng.choice(WIDTH, size=SAMPLES_PER_LAYER, replace=False)


def fit_model(
    dataset: PublicResidualDataset,
    train_ranks: np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
    layer_rms: np.ndarray,
    error: np.ndarray,
    teacher: np.ndarray,
) -> LocalInnovationTransport:
    model = LocalInnovationTransport(centers, scales, layer_rms)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=2.0e-5)
    order_rng = np.random.default_rng(SEED ^ len(train_ranks))
    for epoch in range(EPOCHS):
        shuffled = order_rng.permutation(train_ranks)
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        for rank0, rank in enumerate(shuffled):
            row = dataset.load(int(rank))
            losses: list[torch.Tensor] = []
            for layer in range(FIRST_CORRECTED_LAYER, DEPTH):
                index = sample_indices(int(rank), layer, epoch)
                raw = torch.from_numpy(np.asarray(row.raw_locals[layer, index]).copy())
                sigma = torch.from_numpy(np.asarray(row.sigma[layer, index]).copy())
                base_message = np.asarray(teacher[rank, layer, index], dtype=np.float32)
                # Message jitter exposes the pointwise innovation model to the
                # state errors it must damp during autonomous deployment.
                jitter_rng = np.random.default_rng(
                    (SEED + epoch * 1000003 + int(rank) * 31 + layer) & 0xFFFF_FFFF
                )
                jitter = jitter_rng.standard_normal(len(index), dtype=np.float32)
                jitter_scale = 0.0 if layer == FIRST_CORRECTED_LAYER else NOISE_RMS
                message_np = base_message + jitter_scale * layer_rms[layer] * jitter
                message = torch.from_numpy(message_np)
                truth = torch.from_numpy(np.asarray(error[rank, layer, index]).copy())
                prediction = model.transport_gain()[layer] * message + model.innovation(
                    raw, sigma, message, layer
                )
                losses.append(torch.mean(((prediction - truth) / layer_rms[layer]).square()))
            loss = torch.stack(losses).mean() / NETWORKS_PER_STEP
            loss.backward()
            running += float(loss.detach()) * NETWORKS_PER_STEP
            pending += 1
            if pending == NETWORKS_PER_STEP or rank0 + 1 == len(shuffled):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
        print(
            f"epoch={epoch + 1}/{EPOCHS} sampled_teacher_loss={running/len(shuffled):.8f}",
            flush=True,
        )
    return model


def score_model(
    model: LocalInnovationTransport,
    dataset: PublicResidualDataset,
    ranks: np.ndarray,
    error: np.ndarray,
    batch_size: int = 2,
) -> dict[str, object]:
    model.eval()
    layer_sse = np.zeros(DEPTH, dtype=np.float64)
    per_network: list[float] = []
    with torch.no_grad():
        for offset in range(0, len(ranks), batch_size):
            group = ranks[offset : offset + batch_size]
            rows = [dataset.load(int(rank)) for rank in group]
            raw = torch.from_numpy(
                np.ascontiguousarray(np.stack([row.raw_locals for row in rows]), dtype=np.float32)
            )
            sigma = torch.from_numpy(
                np.ascontiguousarray(np.stack([row.sigma for row in rows]), dtype=np.float32)
            )
            weights = torch.from_numpy(
                np.ascontiguousarray(
                    np.stack(
                        [np.load(row.weights_path, mmap_mode="r", allow_pickle=False) for row in rows]
                    ),
                    dtype=np.float32,
                )
            )
            prediction = model(raw, sigma, weights)
            truth = torch.from_numpy(np.ascontiguousarray(error[group]))
            squared = (prediction - truth).double().square()
            layer_sse += squared.sum(dim=(0, 2)).cpu().numpy()
            per_network.extend(squared[:, -1].mean(dim=1).cpu().numpy().tolist())
            print(f"score {min(offset + batch_size, len(ranks))}/{len(ranks)}", flush=True)
    layer_mse = layer_sse / (len(ranks) * WIDTH)
    return {
        "final_mse": float(layer_mse[-1]),
        "layer_mse": layer_mse.tolist(),
        "per_network_final_mse": per_network,
    }


def model_record(model: LocalInnovationTransport) -> dict[str, object]:
    params = sum(value.numel() for value in model.parameters())
    # Shared local net plus one physical width^2 matvec per corrected layer.
    local = DEPTH * WIDTH * 2 * ((LOCAL_FEATURES + 2) * HIDDEN + HIDDEN * HIDDEN + HIDDEN)
    transport = (DEPTH - FIRST_CORRECTED_LAYER) * (2 * WIDTH * WIDTH + 2 * WIDTH)
    return {
        "parameters": params,
        "hidden": HIDDEN,
        "input_features": LOCAL_FEATURES + 2,
        "estimated_inference_flops": int(local + transport),
        "recurrence": "e_l=a_l*Phi_l*(e_{l-1}@W_l)+local_shared(raw20,message)",
    }


def prepare() -> tuple[
    PublicResidualDataset, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    dataset = PublicResidualDataset()
    readiness = dataset.readiness()
    if not readiness["ready"]:
        raise RuntimeError(f"observer-fit host is not ready: {readiness}")
    sums, sums2 = _row_sufficient_statistics(dataset)
    _, error = host_and_error(dataset)
    teacher = teacher_messages(dataset, error)
    return dataset, sums, sums2, error, teacher


def run_cv() -> None:
    if CV_RESULT.exists():
        raise FileExistsError(CV_RESULT)
    dataset, sums, sums2, error, teacher = prepare()
    folds = np.asarray([fold_of(int(value)) for value in dataset.mlp_ids], dtype=np.int64)
    records = []
    oof_sse = 0.0
    started = time.perf_counter()
    for fold in range(FOLDS):
        valid = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        centers, scales = moments_for(train, sums, sums2)
        layer_rms = layer_rms_for(error, train)
        torch.manual_seed(SEED + fold)
        model = fit_model(dataset, train, centers, scales, layer_rms, error, teacher)
        score = score_model(model, dataset, valid, error)
        oof_sse += float(score["final_mse"]) * len(valid)
        records.append(
            {
                "fold": fold,
                "train_networks": len(train),
                "valid_networks": len(valid),
                "score": score,
            }
        )
        print(f"fold={fold} final={score['final_mse']:.12e}", flush=True)
    result = {
        "schema": "full688-local-innovation-physical-transport-cv-v1",
        "observer_fit_networks": dataset.total,
        "folds": FOLDS,
        "group_key": "mlp_id",
        "observer_selection_targets_opened": 0,
        "architecture": model_record(model),
        "training": {
            "epochs": EPOCHS,
            "samples_per_network_layer_epoch": SAMPLES_PER_LAYER,
            "message_noise_rms": NOISE_RMS,
            "learning_rate": LEARNING_RATE,
        },
        "oof_final_mse": oof_sse / dataset.total,
        "fold_results": records,
        "elapsed_seconds": time.perf_counter() - started,
    }
    CV_RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"oof_final_mse": result["oof_final_mse"]}, sort_keys=True))


def fit_final() -> None:
    if not CV_RESULT.is_file():
        raise FileNotFoundError("run --cross-validate before the all-fit refit")
    if FINAL_ARTIFACT.exists():
        raise FileExistsError(FINAL_ARTIFACT)
    dataset, sums, sums2, error, teacher = prepare()
    ranks = np.arange(dataset.total)
    centers, scales = moments_for(ranks, sums, sums2)
    layer_rms = layer_rms_for(error, ranks)
    torch.manual_seed(SEED + 100)
    model = fit_model(dataset, ranks, centers, scales, layer_rms, error, teacher)
    artifact = {
        "schema": "full688-local-innovation-physical-transport-model-v1",
        "state_dict": copy.deepcopy(model.state_dict()),
        "architecture": model_record(model),
        "cv_result_sha256": sha256(CV_RESULT),
        "observer_fit_networks": dataset.total,
        "observer_selection_targets_opened": 0,
    }
    torch.save(artifact, FINAL_ARTIFACT)
    print(json.dumps({"artifact": str(FINAL_ARTIFACT), "sha256": sha256(FINAL_ARTIFACT)}))


def score_selection(args: argparse.Namespace) -> None:
    if not FINAL_ARTIFACT.is_file():
        raise FileNotFoundError(FINAL_ARTIFACT)
    if SELECTION_RESULT.exists():
        raise FileExistsError(SELECTION_RESULT)
    if args.selection_host_root is None:
        raise ValueError("--score-selection requires --selection-host-root")
    dataset = PublicResidualDataset(
        "observer_selection",
        host_root=args.selection_host_root,
        patch_root=args.selection_patch_root,
        # The selection root is the final repaired estimator itself, rather
        # than an old trajectory plus sparse replacement layer.
        tail_threshold=math.inf if args.selection_patch_root is None else 5.0e-7,
        expected_host_sha256=args.selection_host_sha256,
        expected_patch_sha256=args.selection_patch_sha256,
    )
    readiness = dataset.readiness()
    if not readiness["ready"]:
        raise RuntimeError(f"observer-selection host is not ready: {readiness}")
    artifact = torch.load(FINAL_ARTIFACT, map_location="cpu", weights_only=False)
    state = artifact["state_dict"]
    model = LocalInnovationTransport(
        state["centers"].numpy(), state["scales"].numpy(), state["layer_rms"].numpy()
    )
    model.load_state_dict(state, strict=True)
    host = np.ascontiguousarray(
        np.stack([dataset.load(rank).host for rank in range(dataset.total)]), dtype=np.float32
    )
    error = np.ascontiguousarray(dataset.targets - host, dtype=np.float32)
    score = score_model(model, dataset, np.arange(dataset.total), error)
    host_final = float(
        np.mean(np.square(error[:, -1].astype(np.float64)), dtype=np.float64)
    )
    result = {
        "schema": "full688-local-innovation-physical-transport-selection-v1",
        "selection_networks": dataset.total,
        "host_final_mse": host_final,
        "model_score": score,
        "ratio_to_host": float(score["final_mse"]) / host_final,
        "final_artifact_sha256": sha256(FINAL_ARTIFACT),
        "selection_open_count": 1,
    }
    SELECTION_RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"host": host_final, "model": score["final_mse"]}, sort_keys=True))


def smoke() -> None:
    dataset = PublicResidualDataset()
    readiness = dataset.readiness()
    loadable = [
        rank
        for rank in dataset.available_ranks()
        if not dataset.is_tail(rank) or dataset.patch_complete(rank)
    ]
    repaired = [rank for rank in dataset.tail_roster() if dataset.patch_complete(rank)]
    probes = sorted(set((0, *repaired, max(loadable))))
    loaded = []
    for rank in probes:
        row = dataset.load(rank)
        loaded.append(
            {
                "rank": rank,
                "mlp_id": row.mlp_id,
                "host_source": row.host_source,
                "host_shape": list(row.host.shape),
                "raw_shape": list(row.raw_locals.shape),
                "sigma_shape": list(row.sigma.shape),
            }
        )
    model = LocalInnovationTransport(
        np.zeros((DEPTH, LOCAL_FEATURES - 1), dtype=np.float32),
        np.ones((DEPTH, LOCAL_FEATURES - 1), dtype=np.float32),
        np.ones(DEPTH, dtype=np.float32),
    )
    forward_row = dataset.load(0)
    with torch.no_grad():
        forward = model(
            torch.from_numpy(np.asarray(forward_row.raw_locals)[None].copy()),
            torch.from_numpy(np.asarray(forward_row.sigma)[None].copy()),
            torch.from_numpy(
                np.asarray(
                    np.load(forward_row.weights_path, mmap_mode="r", allow_pickle=False)
                )[None].copy()
            ),
        )
    if tuple(forward.shape) != (1, DEPTH, WIDTH) or not bool(torch.isfinite(forward).all()):
        raise AssertionError("compact-model smoke forward failed")
    result = {
        "readiness": readiness,
        "loaded_probes": loaded,
        "fold_sizes": [
            int(np.sum([fold_of(int(value)) == fold for value in dataset.mlp_ids]))
            for fold in range(FOLDS)
        ],
        "architecture": model_record(model),
        "smoke_forward": {"shape": list(forward.shape), "finite": True},
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    configure()
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--cross-validate", action="store_true")
    mode.add_argument("--fit-final", action="store_true")
    mode.add_argument("--score-selection", action="store_true")
    parser.add_argument("--selection-host-root", type=Path)
    parser.add_argument("--selection-patch-root", type=Path)
    parser.add_argument("--selection-host-sha256")
    parser.add_argument("--selection-patch-sha256")
    args = parser.parse_args()
    if args.smoke:
        smoke()
    elif args.cross_validate:
        run_cv()
    elif args.fit_final:
        fit_final()
    else:
        score_selection(args)


if __name__ == "__main__":
    main()
