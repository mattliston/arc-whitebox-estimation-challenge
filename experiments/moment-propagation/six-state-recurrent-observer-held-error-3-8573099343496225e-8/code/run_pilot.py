#!/Users/matt/anaconda3/bin/python3
"""Compact, unnormalised physical-state residual RNN pilot.

Development is confined to public observer-fit ranks 0:25.  ``--develop``
fits on 0:20 and selects a checkpoint on 20:25.  ``--held`` opens ranks
25:35 only for the named, already-frozen checkpoint.

Each hidden coordinate has the physical units of a post-ReLU mean residual.
The coordinates are never RMS-normalised during rollout.  Four coordinates
are moved by the actual tangent operator ``Phi * W``; the remaining two are
moved by the centred, variance-normalised elementwise-square edge operator.
All recurrent inputs are predictions from the preceding step.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ROWS = (
    ROOT
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3/rows/observer_fit"
)
HOST_DIR = ROOT / "public_observer_fit_candidate_truth_score_v1/runs/raw_diag_load_rescale"
TARGET_FILE = ROOT / "data_gate/sealed_role_target_bundle/observer_fit.parquet"

DEPTH = 16
WIDTH = 1024
DEV_STOP = 20
VALID_STOP = 25
HELD_STOP = 35
SEED = 0x4D53_5251


def configure() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(12)
    torch.use_deterministic_algorithms(True)


def load_targets(stop: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    batches = pq.ParquetFile(TARGET_FILE).iter_batches(
        batch_size=1, columns=["all_layer_means"], use_threads=False
    )
    for rank, batch in enumerate(batches):
        if rank == stop:
            break
        storage = batch.column(0)
        storage = getattr(storage, "storage", storage)
        while not pa.types.is_primitive(storage.type):
            storage = storage.flatten()
        rows.append(storage.to_numpy(zero_copy_only=True).reshape(DEPTH, WIDTH).copy())
    if len(rows) != stop:
        raise RuntimeError(f"expected {stop} target rows, got {len(rows)}")
    return np.ascontiguousarray(np.stack(rows), dtype=np.float32)


def row_file(rank: int, name: str) -> Path:
    return ROWS / f"{rank:04d}" / name


def load_data(start: int, stop: int) -> dict[str, np.ndarray]:
    return {
        "host": np.ascontiguousarray(
            np.stack(
                [np.load(HOST_DIR / f"row_{rank:04d}/q24_output.npy") for rank in range(start, stop)]
            ),
            dtype=np.float32,
        ),
        "raw": np.ascontiguousarray(
            np.stack(
                [np.load(row_file(rank, "cp1792_raw_locals.npy")) for rank in range(start, stop)]
            ),
            dtype=np.float32,
        ),
        "sigma": np.ascontiguousarray(
            np.stack([np.load(row_file(rank, "cp1792_sigma.npy")) for rank in range(start, stop)]),
            dtype=np.float32,
        ),
    }


def load_weights(start: int, stop: int) -> torch.Tensor:
    # Deliberately stream only the current network mini-batch.  Stacking all
    # 25 tapes consumes 1.6 GiB before the recurrent autograd tape exists.
    array = np.ascontiguousarray(
        np.stack(
            [np.load(row_file(rank, "weights.npy"), mmap_mode="r") for rank in range(start, stop)]
        ),
        dtype=np.float32,
    )
    return torch.from_numpy(array)


def fit_standardizer(raw_dev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tail = raw_dev[..., 1:].astype(np.float64)
    centers = tail.mean(axis=(0, 2), dtype=np.float64)
    scales = np.maximum(tail.std(axis=(0, 2), dtype=np.float64), 2.0**-20)
    return centers, scales


def standardize(raw: np.ndarray, centers: np.ndarray, scales: np.ndarray) -> np.ndarray:
    tail = np.clip(
        (raw[..., 1:].astype(np.float64) - centers[None, :, None, :])
        / scales[None, :, None, :],
        -8.0,
        8.0,
    ).astype(np.float32)
    depth = np.linspace(-1.0, 1.0, DEPTH, dtype=np.float32)[None, :, None, None]
    depth = np.broadcast_to(depth, (*raw.shape[:3], 1))
    return np.ascontiguousarray(np.concatenate((depth, tail), axis=-1))


class MultiStatePhysicsRNN(nn.Module):
    """Six autonomous physical residual modes: four W and two centred-W2."""

    def __init__(self, n_features: int, hidden: int, layer_scales: np.ndarray):
        super().__init__()
        self.n_features = int(n_features)
        self.hidden = int(hidden)
        self.channels = 6
        self.w_channels = 4
        self.w2_channels = 2
        self.trunk = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.source_weight = nn.Parameter(torch.empty(DEPTH, hidden, self.channels))
        self.source_bias = nn.Parameter(torch.zeros(DEPTH, self.channels))
        nn.init.normal_(self.source_weight, mean=0.0, std=2.0e-2)

        # The W modes start near the physical Jacobian.  W2 is initially off
        # and earns its amplitude from the development loss.
        self.w_delta = nn.Parameter(torch.zeros(DEPTH, self.w_channels))
        self.w2_delta = nn.Parameter(torch.full((DEPTH, self.w2_channels), -1.5))
        self.w2_gate_weight = nn.Parameter(torch.empty(DEPTH, hidden, self.w2_channels))
        self.w2_gate_bias = nn.Parameter(torch.zeros(DEPTH, self.w2_channels))
        nn.init.normal_(self.w2_gate_weight, mean=0.0, std=1.0e-2)

        # A small depth-specific residual-basis rotation.  It is bounded and
        # initialised at zero, so coordinates retain their physical scale.
        self.mix_delta = nn.Parameter(torch.zeros(DEPTH, self.channels, self.channels))
        self.readout = nn.Parameter(torch.full((DEPTH, self.channels), 1.0 / self.channels))
        self.readout_bias = nn.Parameter(torch.zeros(DEPTH))
        self.register_buffer("layer_scales", torch.from_numpy(layer_scales.copy()))

    def forward(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
        gaussian_gate: torch.Tensor,
    ) -> torch.Tensor:
        batch = features.shape[0]
        trunk = self.trunk(features.reshape(-1, self.n_features)).reshape(
            batch, DEPTH, WIDTH, self.hidden
        )
        source_standard = torch.einsum("blwh,lhc->blwc", trunk, self.source_weight)
        source_standard = source_standard + self.source_bias[None, :, None, :]
        source = source_standard * self.layer_scales[None, :, None, None]

        state = torch.zeros(batch, WIDTH, self.channels, dtype=features.dtype)
        reported: list[torch.Tensor] = []
        identity = torch.eye(self.channels, dtype=features.dtype)
        even_center = 2.0 / WIDTH
        even_scale = math.sqrt(WIDTH / 8.0)
        for layer in range(DEPTH):
            if layer == 0:
                # L0 is the exact Gaussian anchor.
                state = torch.zeros_like(state)
                reported.append(torch.zeros(batch, WIDTH, dtype=features.dtype))
                continue

            w = weights[:, layer]
            prior_w = state[..., : self.w_channels]
            msg_w = torch.bmm(prior_w.transpose(1, 2), w).transpose(1, 2)
            msg_w = msg_w * gaussian_gate[:, layer, :, None]
            w_gain = 1.0 + 0.35 * torch.tanh(self.w_delta[layer])
            msg_w = msg_w * w_gain[None, None, :]

            prior_w2 = state[..., self.w_channels :]
            w2 = (w.square() - even_center) * even_scale
            msg_w2 = torch.bmm(prior_w2.transpose(1, 2), w2).transpose(1, 2)
            local_w2_gate = torch.tanh(
                torch.einsum("bwh,hc->bwc", trunk[:, layer], self.w2_gate_weight[layer])
                + self.w2_gate_bias[layer][None, None, :]
            )
            w2_gain = 0.5 * torch.sigmoid(self.w2_delta[layer])
            msg_w2 = msg_w2 * local_w2_gate * w2_gain[None, None, :]

            messages = torch.cat((msg_w, msg_w2), dim=-1)
            mixing = identity + 0.12 * torch.tanh(self.mix_delta[layer])
            transported = torch.einsum("bwc,cd->bwd", messages, mixing)
            state = transported + source[:, layer]

            correction = torch.einsum("bwc,c->bw", state, self.readout[layer])
            correction = correction + self.readout_bias[layer]
            if layer == 1:
                # L1 discrepancy is dominated by target Monte Carlo noise.
                # Keep a weak latent seed but never report it.
                state = 0.1 * state
                correction = torch.zeros_like(correction)
            reported.append(correction)
        return torch.stack(reported, dim=1)


class ExplicitMultiStatePhysicsRNN(nn.Module):
    """One exact tangent residual plus five additive physical latent modes."""

    def __init__(self, n_features: int, hidden: int, layer_scales: np.ndarray):
        super().__init__()
        self.n_features = int(n_features)
        self.hidden = int(hidden)
        self.latent_channels = 5
        self.latent_w_channels = 3
        self.latent_w2_channels = 2
        self.trunk = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.main_source_weight = nn.Parameter(torch.empty(DEPTH, hidden))
        self.main_source_bias = nn.Parameter(torch.zeros(DEPTH))
        self.latent_source_weight = nn.Parameter(
            torch.empty(DEPTH, hidden, self.latent_channels)
        )
        self.latent_source_bias = nn.Parameter(torch.zeros(DEPTH, self.latent_channels))
        nn.init.normal_(self.main_source_weight, mean=0.0, std=2.0e-3)
        nn.init.normal_(self.latent_source_weight, mean=0.0, std=2.0e-3)

        self.main_transport_delta = nn.Parameter(torch.zeros(DEPTH))
        self.latent_w_delta = nn.Parameter(torch.zeros(DEPTH, self.latent_w_channels))
        self.latent_w2_delta = nn.Parameter(
            torch.full((DEPTH, self.latent_w2_channels), -1.5)
        )
        self.latent_w2_gate_weight = nn.Parameter(
            torch.empty(DEPTH, hidden, self.latent_w2_channels)
        )
        self.latent_w2_gate_bias = nn.Parameter(
            torch.zeros(DEPTH, self.latent_w2_channels)
        )
        nn.init.normal_(self.latent_w2_gate_weight, mean=0.0, std=1.0e-2)
        self.latent_mix_delta = nn.Parameter(
            torch.zeros(DEPTH, self.latent_channels, self.latent_channels)
        )
        # Zero starts at the proven scalar recurrence; latent modes have to
        # improve development loss before they perturb its output.
        self.latent_readout = nn.Parameter(torch.zeros(DEPTH, self.latent_channels))
        self.register_buffer("layer_scales", torch.from_numpy(layer_scales.copy()))

    def forward(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
        gaussian_gate: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        batch = features.shape[0]
        trunk = self.trunk(features.reshape(-1, self.n_features)).reshape(
            batch, DEPTH, WIDTH, self.hidden
        )
        main_standard = torch.einsum("blwh,lh->blw", trunk, self.main_source_weight)
        main_standard = main_standard + self.main_source_bias[None, :, None]
        latent_standard = torch.einsum(
            "blwh,lhc->blwc", trunk, self.latent_source_weight
        )
        latent_standard = latent_standard + self.latent_source_bias[None, :, None, :]
        physical_scale = sigma * self.layer_scales[None, :, None]
        main_source = main_standard * physical_scale
        latent_source = latent_standard * physical_scale[..., None]

        delta = torch.zeros(batch, WIDTH, dtype=features.dtype)
        latent = torch.zeros(
            batch, WIDTH, self.latent_channels, dtype=features.dtype
        )
        reported: list[torch.Tensor] = []
        identity = torch.eye(self.latent_channels, dtype=features.dtype)
        even_center = 2.0 / WIDTH
        even_scale = math.sqrt(WIDTH / 8.0)
        for layer in range(DEPTH):
            if layer == 0:
                reported.append(torch.zeros_like(delta))
                continue

            w = weights[:, layer]
            main_message = torch.bmm(delta.unsqueeze(1), w).squeeze(1)
            main_message = main_message * gaussian_gate[:, layer]
            main_gain = 1.0 + 0.5 * torch.tanh(self.main_transport_delta[layer])

            prior_w = latent[..., : self.latent_w_channels]
            msg_w = torch.bmm(prior_w.transpose(1, 2), w).transpose(1, 2)
            msg_w = msg_w * gaussian_gate[:, layer, :, None]
            w_gain = 1.0 + 0.35 * torch.tanh(self.latent_w_delta[layer])
            msg_w = msg_w * w_gain[None, None, :]

            prior_w2 = latent[..., self.latent_w_channels :]
            w2 = (w.square() - even_center) * even_scale
            msg_w2 = torch.bmm(prior_w2.transpose(1, 2), w2).transpose(1, 2)
            local_w2_gate = torch.tanh(
                torch.einsum(
                    "bwh,hc->bwc", trunk[:, layer], self.latent_w2_gate_weight[layer]
                )
                + self.latent_w2_gate_bias[layer][None, None, :]
            )
            w2_gain = 0.5 * torch.sigmoid(self.latent_w2_delta[layer])
            msg_w2 = msg_w2 * local_w2_gate * w2_gain[None, None, :]
            messages = torch.cat((msg_w, msg_w2), dim=-1)
            mixing = identity + 0.12 * torch.tanh(self.latent_mix_delta[layer])
            latent = torch.einsum("bwc,cd->bwd", messages, mixing)
            latent = latent + latent_source[:, layer]
            auxiliary = torch.einsum(
                "bwc,c->bw", latent, self.latent_readout[layer]
            )
            delta = main_gain * main_message + main_source[:, layer] + auxiliary
            if layer == 1:
                latent = 0.1 * latent
                delta = torch.zeros_like(delta)
            reported.append(delta)
        return torch.stack(reported, dim=1)


def build_model(
    architecture: str, n_features: int, hidden: int, layer_scales: np.ndarray
) -> nn.Module:
    if architecture == "coupled6":
        return MultiStatePhysicsRNN(n_features, hidden, layer_scales)
    if architecture == "explicit6":
        return ExplicitMultiStatePhysicsRNN(n_features, hidden, layer_scales)
    raise ValueError(f"unknown architecture {architecture!r}")


def make_tensors(
    data: dict[str, np.ndarray], targets: np.ndarray, features: np.ndarray
) -> dict[str, torch.Tensor]:
    return {
        "features": torch.from_numpy(features),
        "gate": torch.from_numpy(np.ascontiguousarray(data["raw"][..., 5])),
        "sigma": torch.from_numpy(np.ascontiguousarray(data["sigma"])),
        "error": torch.from_numpy(np.ascontiguousarray(targets - data["host"])),
    }


def metric(
    model: nn.Module,
    tensors: dict[str, torch.Tensor],
    local_start: int,
    local_stop: int,
    global_offset: int,
    batch_size: int = 5,
) -> tuple[float, list[float], list[float]]:
    model.eval()
    layer_chunks: list[np.ndarray] = []
    network_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(local_start, local_stop, batch_size):
            stop = min(start + batch_size, local_stop)
            weights = load_weights(global_offset + start, global_offset + stop)
            arguments = (
                tensors["features"][start:stop], weights, tensors["gate"][start:stop]
            )
            if isinstance(model, ExplicitMultiStatePhysicsRNN):
                prediction = model(*arguments, tensors["sigma"][start:stop])
            else:
                prediction = model(*arguments)
            error = prediction - tensors["error"][start:stop]
            layer_chunks.append(torch.mean(error.double().square(), dim=2).cpu().numpy())
            network_chunks.append(torch.mean(error[:, -1].double().square(), dim=1).cpu().numpy())
    network_layer = np.concatenate(layer_chunks, axis=0)
    layer = np.mean(network_layer, axis=0, dtype=np.float64)
    per_network = np.concatenate(network_chunks, axis=0)
    return float(layer[-1]), layer.tolist(), per_network.tolist()


def estimated_flops(n_features: int, hidden: int) -> dict[str, int]:
    # Approximate challenge-style add+multiply count.  W and W2 transports
    # dominate; forming centred W2 is included.
    local = DEPTH * WIDTH * 2 * (n_features * hidden + hidden * hidden)
    w_transport = (DEPTH - 1) * self_or_one(2 * WIDTH * WIDTH * 4)
    w2_form = (DEPTH - 1) * WIDTH * WIDTH * 3
    w2_transport = (DEPTH - 1) * 2 * WIDTH * WIDTH * 2
    return {
        "local_trunk": int(local),
        "w_transport": int(w_transport),
        "w2_form": int(w2_form),
        "w2_transport": int(w2_transport),
        "total": int(local + w_transport + w2_form + w2_transport),
    }


def self_or_one(value: int) -> int:
    # Kept separate so the FLOP expression above reads like the contraction.
    return value


def develop(args: argparse.Namespace) -> None:
    configure()
    started = time.perf_counter()
    targets = load_targets(VALID_STOP)
    data = load_data(0, VALID_STOP)
    centers, scales = fit_standardizer(data["raw"][:DEV_STOP])
    features = standardize(data["raw"], centers, scales)
    residual_dev = targets[:DEV_STOP].astype(np.float64) - data["host"][:DEV_STOP].astype(np.float64)
    layer_scales = np.maximum(
        np.sqrt(np.mean(residual_dev * residual_dev, axis=(0, 2))), 2.0**-20
    ).astype(np.float32)
    tensors = make_tensors(data, targets, features)

    model = build_model(args.architecture, features.shape[-1], args.hidden, layer_scales)
    if args.init_checkpoint:
        warm = torch.load(HERE / args.init_checkpoint, map_location="cpu", weights_only=False)
        if int(warm["hidden"]) != args.hidden:
            raise ValueError("warm checkpoint hidden width mismatch")
        model.load_state_dict(warm["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2.0e-5)
    normalizer = torch.from_numpy(layer_scales)[None, :, None]
    layer_weights = (
        torch.linspace(0.5, 1.5, DEPTH)[None, :, None]
        if args.baseline_loss
        else torch.linspace(0.15, 1.0, DEPTH)[None, :, None]
    )
    layer_weights[..., :2, :] = 0.02

    best_valid = math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    trace: list[dict[str, float | int]] = []
    stale = 0
    for epoch in range(args.epochs + 1):
        if epoch % args.check_every == 0:
            train_final, _, _ = metric(model, tensors, 0, DEV_STOP, 0)
            valid_final, _, _ = metric(model, tensors, DEV_STOP, VALID_STOP, 0)
            trace.append(
                {
                    "epoch": epoch,
                    "train_final_mse": train_final,
                    "validation_final_mse": valid_final,
                    "seconds": time.perf_counter() - started,
                }
            )
            print(
                f"epoch={epoch:03d} train={train_final:.12e} valid={valid_final:.12e} "
                f"best={best_valid:.12e}@{best_epoch} elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
            if valid_final < best_valid:
                best_valid = valid_final
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= args.patience:
                break
        if epoch == args.epochs:
            break

        model.train()
        # Network mini-batches keep the unnormalised W2 recurrent tape well
        # below the host RSS limit.  The order is fixed, so the run remains
        # deterministic.
        if args.accumulate:
            optimizer.zero_grad(set_to_none=True)
        for batch_start in range(0, DEV_STOP, args.batch_size):
            batch_stop = min(batch_start + args.batch_size, DEV_STOP)
            if not args.accumulate:
                optimizer.zero_grad(set_to_none=True)
            weights = load_weights(batch_start, batch_stop)
            arguments = (
                tensors["features"][batch_start:batch_stop],
                weights,
                tensors["gate"][batch_start:batch_stop],
            )
            if isinstance(model, ExplicitMultiStatePhysicsRNN):
                prediction = model(*arguments, tensors["sigma"][batch_start:batch_stop])
            else:
                prediction = model(*arguments)
            normalized = (
                prediction - tensors["error"][batch_start:batch_stop]
            ) / normalizer
            final_loss = torch.mean(normalized[:, -1].square())
            rollout_loss = torch.mean(layer_weights * normalized.square())
            mix_parameter = (
                model.latent_mix_delta
                if isinstance(model, ExplicitMultiStatePhysicsRNN)
                else model.mix_delta
            )
            state_regularizer = 2.0e-4 * torch.mean(mix_parameter.square())
            if args.baseline_loss:
                if isinstance(model, ExplicitMultiStatePhysicsRNN):
                    near_one = 2.0e-3 * torch.mean(
                        (1.0 + 0.5 * torch.tanh(model.main_transport_delta[1:]) - 1.0)
                        .square()
                    )
                else:
                    near_one = torch.zeros((), dtype=normalized.dtype)
                loss = rollout_loss + near_one + state_regularizer
            else:
                loss = final_loss + 0.35 * rollout_loss + state_regularizer
            if args.accumulate:
                loss = loss * ((batch_stop - batch_start) / DEV_STOP)
            loss.backward()
            if not args.accumulate:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        if args.accumulate:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

    if best_state is None:
        raise RuntimeError("checkpoint selection failed")
    model.load_state_dict(best_state, strict=True)
    train_final, train_layers, _ = metric(model, tensors, 0, DEV_STOP, 0)
    valid_final, valid_layers, valid_networks = metric(model, tensors, DEV_STOP, VALID_STOP, 0)

    stem = args.name
    checkpoint = HERE / f"{stem}.pt"
    torch.save(
        {
            "state_dict": best_state,
            "centers": centers,
            "scales": scales,
            "layer_scales": layer_scales,
            "hidden": args.hidden,
            "architecture": args.architecture,
            "best_epoch": best_epoch,
            "validation_final_mse": valid_final,
        },
        checkpoint,
    )
    result = {
        "schema": "public-multistate-physics-rnn-development-v1",
        "split": {"development": [0, DEV_STOP], "validation": [DEV_STOP, VALID_STOP]},
        "architecture": {
            "channels": 6,
            "variant": args.architecture,
            "operators": ["Phi*W"] * 4 + ["tanh(local)*sqrt(n/8)*(W^2-2/n)"] * 2,
            "unnormalized_physical_state": True,
            "hidden": args.hidden,
            "parameters": sum(p.numel() for p in model.parameters()),
            "estimated_inference_flops": estimated_flops(features.shape[-1], args.hidden),
            "l0_report_gate": 0.0,
            "l1_report_gate": 0.0,
            "l1_latent_gate": 0.1,
        },
        "training": {
            "seed": SEED,
            "learning_rate": args.lr,
            "gradient_accumulation_one_step_per_epoch": args.accumulate,
            "baseline_weighted_rollout_loss": args.baseline_loss,
            "best_epoch": best_epoch,
            "trace": trace,
        },
        "train_final_mse": train_final,
        "train_layer_mse": train_layers,
        "validation_final_mse": valid_final,
        "validation_layer_mse": valid_layers,
        "validation_per_network_final_mse": valid_networks,
        "checkpoint": checkpoint.name,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (HERE / f"{stem}_DEVELOPMENT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"checkpoint": str(checkpoint), "best_epoch": best_epoch, "train": train_final, "validation": valid_final, "elapsed": result["elapsed_seconds"]}, sort_keys=True))


def held(args: argparse.Namespace) -> None:
    configure()
    started = time.perf_counter()
    checkpoint_path = HERE / f"{args.name}.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    data = load_data(VALID_STOP, HELD_STOP)
    targets = load_targets(HELD_STOP)[VALID_STOP:HELD_STOP]
    features = standardize(data["raw"], payload["centers"], payload["scales"])
    tensors = make_tensors(data, targets, features)
    model = build_model(
        str(payload.get("architecture", "coupled6")),
        features.shape[-1],
        int(payload["hidden"]),
        payload["layer_scales"],
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    held_final, held_layers, held_networks = metric(
        model, tensors, 0, HELD_STOP - VALID_STOP, VALID_STOP
    )
    host_final = float(
        np.mean(
            (targets[:, -1].astype(np.float64) - data["host"][:, -1].astype(np.float64)) ** 2,
            dtype=np.float64,
        )
    )
    result = {
        "schema": "public-multistate-physics-rnn-held-v1",
        "checkpoint": checkpoint_path.name,
        "checkpoint_validation_final_mse": float(payload["validation_final_mse"]),
        "checkpoint_best_epoch": int(payload["best_epoch"]),
        "split": {"held_opened_once_after_checkpoint_selection": [VALID_STOP, HELD_STOP]},
        "held_host_final_mse": host_final,
        "held_final_mse": held_final,
        "held_ratio_to_host": held_final / host_final,
        "held_layer_mse": held_layers,
        "held_per_network_final_mse": held_networks,
        "beats_3_82e_8": held_final < 3.82e-8,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (HERE / f"{args.name}_HELD.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--develop", action="store_true")
    mode.add_argument("--held", action="store_true")
    parser.add_argument("--name", required=True)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--architecture", choices=("coupled6", "explicit6"), default="explicit6")
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--check-every", type=int, default=4)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--accumulate", action="store_true")
    parser.add_argument("--baseline-loss", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.develop:
        develop(arguments)
    else:
        held(arguments)
