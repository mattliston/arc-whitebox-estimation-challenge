#!/usr/bin/env python3
"""Target-free Torch reference for the Phase-2 V3 residual observer.

This module contains no dataset or target loader.  It is the CPU training and
parity authority for feature tapes supplied by a separately audited, role-
scoped process.  Layer 1 is structurally host-only for every parameter value.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


WIDTH = 1024
DEPTH = 16
LOCAL_FEATURES = 20
NORMALIZED_FEATURES = 19
HIDDEN = 32
CHANNELS_PER_POWER = 8
CELL_INPUTS = 56
CELL_WIDTH = 64
INIT_SEED = 4671727249568777182
BATCH_SEED = 9059531498324691115
SOURCE_MASK_SEED = 8426681866426422326
PARITY_SEED = 501163896429966340
MESSAGE_EPS = float(2.0**-24)
NORMALIZATION_SCALE_FLOOR = float(2.0**-20)

PARAMETER_ORDER = (
    "core1.weight",
    "core1.bias",
    "core2.weight",
    "core2.bias",
    "candidate_head.weight",
    "candidate_head.bias",
    "gate_head.weight",
    "gate_head.bias",
    "mean_head.weight",
    "mean_head.bias",
    "source_gate_logits",
)

OPTIONAL_LOCAL_BLOCKS = (
    (12,),                 # x13 Hermite tail
    (1, 2, 8, 9, 10),     # x02,x03,x09:x11 CP
    (13, 14, 15),         # x14:x16 covariance
    (16, 17, 18, 19),     # x17:x20 weight summaries
)


def configure_cpu_determinism() -> dict[str, Any]:
    """Seal the deterministic CPU authority used before any target release."""

    required_environment = {
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    for name, value in required_environment.items():
        os.environ[name] = value
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting this only before inter-op work starts.  A
        # verifier records the observed value and fails if it is not already 1.
        pass
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return {
        "environment": required_environment,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
    }


def exact_depth_coordinate(layer0: int, *, device: torch.device | str = "cpu") -> Tensor:
    """The uncentered V3 depth coordinate, evaluated then rounded to float32."""

    if not 0 <= int(layer0) < DEPTH:
        raise ValueError(f"layer0 must be in [0,{DEPTH}), got {layer0}")
    return torch.tensor(2.0 * int(layer0) / 15.0 - 1.0, dtype=torch.float32, device=device)


def standardize_raw_features(raw: Tensor, centers: Tensor, scales: Tensor) -> Tensor:
    """Bypass x01 and standardize only x02:x20 in float64.

    Shapes are ``raw=(16,n,20)`` and ``centers=scales=(16,19)``.  The caller
    must treat any raised exception as a permanent observer latch-off; no
    replacement statistic is permitted.
    """

    if raw.dtype != torch.float32 or raw.ndim != 3 or raw.shape[0] != DEPTH or raw.shape[2] != LOCAL_FEATURES:
        raise ValueError("raw must be contiguous float32[16,n,20]")
    if centers.dtype != torch.float64 or scales.dtype != torch.float64:
        raise ValueError("centers/scales must be float64")
    if tuple(centers.shape) != (DEPTH, NORMALIZED_FEATURES) or tuple(scales.shape) != (DEPTH, NORMALIZED_FEATURES):
        raise ValueError("centers/scales must have shape [16,19]")
    if not bool(torch.isfinite(raw).all().item()):
        raise FloatingPointError("nonfinite raw local feature")
    if not bool(torch.isfinite(centers).all().item()) or not bool(torch.isfinite(scales).all().item()):
        raise FloatingPointError("nonfinite normalization statistic")
    if not bool((scales > 0.0).all().item()):
        raise FloatingPointError("normalization scale is not positive")

    tail64 = raw[..., 1:].to(torch.float64)
    standardized_tail = torch.clamp(
        (tail64 - centers[:, None, :]) / scales[:, None, :],
        min=-8.0,
        max=8.0,
    ).to(torch.float32)
    depth = torch.stack(
        [exact_depth_coordinate(layer0, device=raw.device) for layer0 in range(DEPTH)]
    )[:, None, None].expand(DEPTH, raw.shape[1], 1)
    result = torch.cat((depth, standardized_tail), dim=-1).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise FloatingPointError("nonfinite standardized feature")
    return result


class ResidualObserver(nn.Module):
    """Width-equivariant local or recurrent V3 observer cell."""

    def __init__(
        self,
        *,
        arm: str,
        init_seed: int = INIT_SEED,
        parity_nonzero_mean_head: bool = False,
        hidden_channels: int = HIDDEN,
        cell_width: int = CELL_WIDTH,
    ) -> None:
        super().__init__()
        if arm not in {"local_no_recurrence", "recurrent"}:
            raise ValueError("arm must be local_no_recurrence or recurrent")
        if hidden_channels <= 0 or hidden_channels % 4 != 0 or cell_width <= 0:
            raise ValueError("hidden_channels must be positive/divisible by 4 and cell_width positive")
        self.arm = arm
        self.hidden_channels = int(hidden_channels)
        self.channels_per_power = self.hidden_channels // 4
        self.cell_width = int(cell_width)
        self.cell_inputs = LOCAL_FEATURES + 4 + self.hidden_channels
        self.core1 = nn.Linear(self.cell_inputs, self.cell_width, bias=True, dtype=torch.float32)
        self.core2 = nn.Linear(self.cell_width, self.cell_width, bias=True, dtype=torch.float32)
        self.candidate_head = nn.Linear(self.cell_width, self.hidden_channels, bias=True, dtype=torch.float32)
        self.gate_head = nn.Linear(self.cell_width, self.hidden_channels, bias=True, dtype=torch.float32)
        self.mean_head = nn.Linear(self.cell_width, 1, bias=True, dtype=torch.float32)
        self.source_gate_logits = nn.Parameter(torch.zeros(8, dtype=torch.float32))
        self._initialize_exactly(int(init_seed), bool(parity_nonzero_mean_head))
        observed = {name for name, _ in self.named_parameters()}
        if observed != set(PARAMETER_ORDER):
            raise RuntimeError(f"parameter membership drift: {sorted(observed)!r}")

    def _initialize_exactly(self, seed: int, parity_nonzero_mean_head: bool) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        for module in (self.core1, self.core2, self.candidate_head, self.gate_head):
            nn.init.xavier_uniform_(module.weight, gain=1.0, generator=generator)
            nn.init.zeros_(module.bias)
        if parity_nonzero_mean_head:
            nn.init.xavier_uniform_(self.mean_head.weight, gain=1.0, generator=generator)
        else:
            nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.source_gate_logits)

    def ordered_parameters(self) -> list[nn.Parameter]:
        named = dict(self.named_parameters())
        if set(named) != set(PARAMETER_ORDER):
            raise RuntimeError("parameter membership drift")
        return [named[name] for name in PARAMETER_ORDER]

    def _gate_locals(
        self,
        local: Tensor,
        gates: Tensor,
        *,
        masked_optional_block: int | None = None,
    ) -> Tensor:
        if masked_optional_block is not None and masked_optional_block not in range(4):
            raise ValueError("masked_optional_block must be None or an integer in [0,4)")
        chunks: list[Tensor] = []
        for feature_index in range(LOCAL_FEATURES):
            block = next(
                (block_index for block_index, indices in enumerate(OPTIONAL_LOCAL_BLOCKS) if feature_index in indices),
                None,
            )
            column = local[:, feature_index : feature_index + 1]
            if block is None:
                chunks.append(column)
            elif block == masked_optional_block:
                chunks.append(torch.zeros_like(column))
            else:
                chunks.append(column * gates[block])
        return torch.cat(chunks, dim=1)

    def _normalized_messages(self, edges: Tensor, h_previous: Tensor) -> tuple[Tensor, Tensor]:
        message_groups: list[Tensor] = []
        log_rms: list[Tensor] = []
        for power in range(4):
            prior_group = h_previous[:, power * self.channels_per_power : (power + 1) * self.channels_per_power]
            transported = torch.matmul(edges[power].transpose(0, 1), prior_group).to(torch.float32)
            flat = (transported * transported).contiguous().view(-1)
            mean_square = torch.sum(flat, dtype=torch.float32) / float(flat.numel())
            rho = torch.sqrt(mean_square + MESSAGE_EPS).to(torch.float32)
            message_groups.append(torch.clamp(transported / rho, min=-5.0, max=5.0).to(torch.float32))
            log_rms.append(torch.clamp(torch.log(rho), min=-8.0, max=8.0).to(torch.float32))
        return torch.cat(message_groups, dim=1).contiguous(), torch.stack(log_rms).contiguous()

    def forward(
        self,
        standardized_locals: Tensor,
        edge_matrices: Tensor,
        host_rows: Tensor,
        sigma: Tensor,
        *,
        masked_optional_block: int | None = None,
        return_trace: bool = True,
    ) -> dict[str, Any]:
        """Run one 16-layer network; observer state never enters the host."""

        n = int(host_rows.shape[1])
        expected = {
            "standardized_locals": (DEPTH, n, LOCAL_FEATURES),
            "edge_matrices": (DEPTH, 4, n, n),
            "host_rows": (DEPTH, n),
            "sigma": (DEPTH, n),
        }
        actual = {
            "standardized_locals": tuple(standardized_locals.shape),
            "edge_matrices": tuple(edge_matrices.shape),
            "host_rows": tuple(host_rows.shape),
            "sigma": tuple(sigma.shape),
        }
        if actual != expected:
            raise ValueError(f"shape mismatch: expected={expected}, actual={actual}")
        tensors = (standardized_locals, edge_matrices, host_rows, sigma)
        if any(t.dtype != torch.float32 for t in tensors):
            raise ValueError("all learned-boundary inputs must be float32")

        h_previous = torch.zeros((n, self.hidden_channels), dtype=torch.float32, device=host_rows.device)
        active = True
        rows: list[Tensor] = []
        messages_trace: list[Tensor] = []
        log_rms_trace: list[Tensor] = []
        hidden_trace: list[Tensor] = []
        unit_trace: list[Tensor] = []
        delta_trace: list[Tensor] = []
        latch_trace: list[bool] = []
        parameter_vector = torch.cat([parameter.reshape(-1) for parameter in self.ordered_parameters()])

        for layer0 in range(DEPTH):
            host = host_rows[layer0]
            if not active:
                messages = torch.zeros((n, self.hidden_channels), dtype=torch.float32, device=host.device)
                log_rms = torch.zeros(4, dtype=torch.float32, device=host.device)
                hidden = torch.zeros_like(h_previous)
                unit = torch.zeros(n, dtype=torch.float32, device=host.device)
                delta = torch.zeros_like(unit)
                prediction = torch.clamp_min(host, 0.0)
            else:
                if layer0 == 0 or self.arm == "local_no_recurrence":
                    messages = torch.zeros((n, self.hidden_channels), dtype=torch.float32, device=host.device)
                    log_rms = torch.zeros(4, dtype=torch.float32, device=host.device)
                else:
                    messages, log_rms = self._normalized_messages(edge_matrices[layer0], h_previous)

                source_gates = torch.sigmoid(self.source_gate_logits)
                gated_local = self._gate_locals(
                    standardized_locals[layer0],
                    source_gates,
                    masked_optional_block=masked_optional_block,
                )
                gated_messages: list[Tensor] = []
                gated_log_rms: list[Tensor] = []
                for power in range(4):
                    gate = source_gates[4 + power]
                    gated_messages.append(
                        messages[:, power * self.channels_per_power : (power + 1) * self.channels_per_power] * gate
                    )
                    gated_log_rms.append(log_rms[power] * gate)
                broadcast_log = torch.stack(gated_log_rms)[None, :].expand(n, 4)
                cell_input = torch.cat((gated_local, broadcast_log, torch.cat(gated_messages, dim=1)), dim=1)
                q1 = F.silu(self.core1(cell_input))
                q2 = F.silu(self.core2(q1))
                candidate = torch.tanh(self.candidate_head(q2))
                update_gate = torch.sigmoid(self.gate_head(q2))
                hidden = torch.tanh(messages + update_gate * (candidate - messages))
                unit = torch.tanh(self.mean_head(q2)).squeeze(1)

                # Structural, not loss-induced: no parameter can change delta_1.
                if layer0 == 0:
                    delta = torch.zeros_like(unit)
                else:
                    delta = (sigma[layer0] * (1.0 / 32.0) * unit).to(torch.float32)
                prediction = torch.clamp_min((host + delta).to(torch.float32), 0.0)

                diagnostics = (standardized_locals[layer0], edge_matrices[layer0], messages, log_rms, cell_input, hidden, unit, delta, prediction, parameter_vector)
                finite = all(bool(torch.isfinite(value).all().item()) for value in diagnostics)
                positive = bool(torch.isfinite(sigma[layer0]).all().item()) and bool((sigma[layer0] > 0.0).all().item())
                if not (finite and positive):
                    active = False
                    h_previous = torch.zeros_like(h_previous)
                    hidden = torch.zeros_like(hidden)
                    unit = torch.zeros_like(unit)
                    delta = torch.zeros_like(delta)
                    prediction = torch.clamp_min(host, 0.0)

            rows.append(prediction)
            messages_trace.append(messages)
            log_rms_trace.append(log_rms)
            hidden_trace.append(hidden)
            unit_trace.append(unit)
            delta_trace.append(delta)
            latch_trace.append(active)
            if self.arm == "recurrent" and active:
                h_previous = hidden
            else:
                h_previous = torch.zeros_like(h_previous)

        result: dict[str, Any] = {
            "predictions": torch.stack(rows),
            "latch": tuple(latch_trace),
        }
        if return_trace:
            result.update(
                messages=torch.stack(messages_trace),
                log_rms=torch.stack(log_rms_trace),
                hidden=torch.stack(hidden_trace),
                unit_correction=torch.stack(unit_trace),
                physical_correction=torch.stack(delta_trace),
            )
        return result


def build_adamw(model: ResidualObserver) -> torch.optim.AdamW:
    """One parameter group, exact declared order, decay on all 11 tensors."""

    return torch.optim.AdamW(
        [{"params": model.ordered_parameters(), "weight_decay": 1.0e-5}],
        lr=1.0e-3,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        amsgrad=False,
        maximize=False,
        foreach=False,
        fused=False,
        capturable=False,
    )


def exact_training_loss(
    trace: Mapping[str, Tensor],
    targets: Tensor,
    host_rows: Tensor,
    residual_scales: Tensor,
) -> Tensor:
    """V3 objective; callers must supply only role-authorized targets."""

    del host_rows  # retained in the signature to bind the host/candidate tape pair
    predictions = trace["predictions"]
    correction = trace["physical_correction"]
    hidden = trace["hidden"]
    if targets.dtype != torch.float32 or tuple(targets.shape) != tuple(predictions.shape):
        raise ValueError("targets must be float32 and match predictions")
    if residual_scales.dtype != torch.float32 or tuple(residual_scales.shape) != (DEPTH - 1,):
        raise ValueError("residual_scales must be float32[15]")
    errors = ((predictions[1:] - targets[1:]) / residual_scales[:, None]) ** 2
    correction_energy = (correction[1:] / residual_scales[:, None]) ** 2
    final_loss = errors[-1].mean(dtype=torch.float32)
    all_loss = errors.mean(dtype=torch.float32)
    correction_loss = correction_energy.mean(dtype=torch.float32)
    hidden_loss = (hidden * hidden).mean(dtype=torch.float32)
    return final_loss + 0.25 * all_loss + 1.0e-4 * correction_loss + 1.0e-4 * hidden_loss


def exact_two_network_batch_loss(
    traces: Sequence[Mapping[str, Tensor]],
    targets: Sequence[Tensor],
    host_rows: Sequence[Tensor],
    residual_scales: Tensor,
) -> Tensor:
    """Exact batch-size-two objective in supplied canonical batch order."""

    if len(traces) != 2 or len(targets) != 2 or len(host_rows) != 2:
        raise ValueError("V3 training batches contain exactly two complete networks")
    losses = [
        exact_training_loss(trace, target, host, residual_scales)
        for trace, target, host in zip(traces, targets, host_rows, strict=True)
    ]
    return torch.stack(losses).mean(dtype=torch.float32)


def fit_epoch_order(epoch: int) -> tuple[int, ...]:
    """Return the exact immutable permutation of positions 12:700."""

    if int(epoch) != epoch or epoch < 1:
        raise ValueError("epoch must be a positive integer")
    decorated: list[tuple[bytes, int]] = []
    seen: set[bytes] = set()
    for position in range(12, 700):
        digest = hashlib.sha256(
            f"{BATCH_SEED}|epoch={epoch}|position={position}".encode("ascii")
        ).digest()
        if digest in seen:
            raise RuntimeError("batch-order digest collision")
        seen.add(digest)
        decorated.append((digest, position))
    ordered = tuple(position for _, position in sorted(decorated))
    if len(ordered) != 688 or len(set(ordered)) != 688:
        raise RuntimeError("fit permutation is not a 688-position bijection")
    return ordered


def source_mask_block(epoch: int, position: int) -> int | None:
    """Exact 1/20 trajectory mask; returns one optional block or None."""

    if int(epoch) != epoch or epoch < 1 or int(position) != position or not 12 <= position < 700:
        raise ValueError("source-mask coordinates must be epoch>=1 and fit position in [12,700)")
    digest = hashlib.sha256(
        f"{SOURCE_MASK_SEED}|epoch={epoch}|position={position}".encode("ascii")
    ).digest()
    first = int.from_bytes(digest[:8], "big", signed=False)
    # (first + 0.5) / 2^64 < 1/20, evaluated without binary64 rounding.
    if 20 * (2 * first + 1) < 2**65:
        return int.from_bytes(digest[8:16], "big", signed=False) % 4
    return None


def one_optimizer_step(model: ResidualObserver, loss: Tensor) -> float:
    """Exact V3 backward/clip/update law; returns the pre-clip norm."""

    optimizer = getattr(model, "_v3_optimizer", None)
    if optimizer is None:
        optimizer = build_adamw(model)
        model._v3_optimizer = optimizer  # type: ignore[attr-defined]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        model.ordered_parameters(),
        max_norm=1.0,
        norm_type=2.0,
        error_if_nonfinite=True,
        foreach=False,
    )
    optimizer.step()
    return float(norm.detach().cpu().item())


def tensor_bundle_sha256(named_tensors: Iterable[tuple[str, Tensor]]) -> str:
    """Stable raw-tensor hash independent of torch.save container metadata."""

    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        name_bytes = name.encode("utf-8")
        metadata = json.dumps(
            {"dtype": array.dtype.str, "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        payload = memoryview(array).cast("B")
        for part in (name_bytes, metadata, payload):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def model_sha256(model: ResidualObserver) -> str:
    named = dict(model.named_parameters())
    return tensor_bundle_sha256((name, named[name]) for name in PARAMETER_ORDER)


def optimizer_sha256(model: ResidualObserver) -> str:
    optimizer = getattr(model, "_v3_optimizer", None)
    if optimizer is None:
        raise RuntimeError("optimizer has not been created")
    named = dict(model.named_parameters())
    items: list[tuple[str, Tensor]] = []
    for name in PARAMETER_ORDER:
        state = optimizer.state[named[name]]
        for state_name in ("step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(state_name)
            if isinstance(value, Tensor):
                items.append((f"{name}/{state_name}", value))
    return tensor_bundle_sha256(items)


def numpy_array_sha256(name: str, array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()
