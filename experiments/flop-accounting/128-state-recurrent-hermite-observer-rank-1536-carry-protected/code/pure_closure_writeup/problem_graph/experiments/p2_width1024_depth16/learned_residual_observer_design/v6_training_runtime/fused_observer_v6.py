#!/usr/bin/env python3
"""Post-claim streamed Hermite transport for the unchanged V3 observer.

The local arm never accepts or touches a weight tensor.  The recurrent arm
streams one row tile of W at a time, evaluates all four frozen edge
polynomials from that tile, and accumulates the four E_p(W)^T h_p products.
The custom backward streams W again to compute E_p(W) grad_p, so no edge tape
or recurrent message is cached across optimizer steps.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import stat
import sys
import types
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F


FROZEN_OBSERVER_PATH = Path(
    "/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/"
    "p2_width1024_depth16/learned_residual_observer_design/v3/observer_model.py"
)
FROZEN_OBSERVER_SHA256 = "e70fa563bbbfe6f4e3b8952ede47eeba2c9e1ca0c2507cbc96ce541c0547bbd5"
DEFAULT_TILE_ROWS = 128


def _load_frozen_observer() -> Any:
    name = "v6_frozen_observer_model"
    if name in sys.modules:
        return sys.modules[name]
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(FROZEN_OBSERVER_PATH, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise PermissionError("frozen observer must be one regular non-hardlinked file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        source = b"".join(chunks)
    finally:
        os.close(descriptor)
    if hashlib.sha256(source).hexdigest() != FROZEN_OBSERVER_SHA256:
        raise AssertionError("frozen observer model changed")
    module = types.ModuleType(name)
    module.__file__ = str(FROZEN_OBSERVER_PATH)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(source, str(FROZEN_OBSERVER_PATH), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


OBSERVER = _load_frozen_observer()


def _edge_tile(weight_tile: Tensor, power: int, width: int) -> Tensor:
    square = weight_tile * weight_tile
    if power == 0:
        return (weight_tile * (1.0 / math.sqrt(2.0))).to(torch.float32)
    if power == 1:
        return ((square - (2.0 / width)) * math.sqrt(width / 8.0)).to(torch.float32)
    if power == 2:
        return ((square * weight_tile) * (width / math.sqrt(120.0))).to(torch.float32)
    if power == 3:
        return (
            (square * square - (12.0 / (width * width)))
            * (width**1.5 / math.sqrt(1536.0))
        ).to(torch.float32)
    raise ValueError("Hermite power must be in [0,4)")


class _StreamFourTransports(torch.autograd.Function):
    """One-pass four-polynomial forward and recomputing backward."""

    @staticmethod
    def forward(ctx: Any, weights: Tensor, hidden: Tensor, tile_rows: int) -> Tensor:
        if weights.dtype != torch.float32 or hidden.dtype != torch.float32:
            raise ValueError("streamed transport accepts float32 only")
        n = int(weights.shape[0])
        if tuple(weights.shape) != (n, n) or hidden.ndim != 2 or int(hidden.shape[0]) != n:
            raise ValueError("streamed transport shape mismatch")
        if weights.device != hidden.device:
            raise ValueError("streamed weights and hidden state must share one device")
        channels = int(hidden.shape[1])
        if channels <= 0 or channels % 4:
            raise ValueError("hidden channels must be positive and divisible by four")
        tile = int(tile_rows)
        if tile <= 0 or tile > n:
            raise ValueError("invalid streamed transport tile")
        group = channels // 4
        output = torch.zeros((n, channels), dtype=torch.float32, device=hidden.device)
        # weights are fixed host features; gradients are required only for h.
        for start in range(0, n, tile):
            stop = min(start + tile, n)
            weight_tile = weights[start:stop]
            square = weight_tile * weight_tile
            polynomials = (
                (weight_tile * (1.0 / math.sqrt(2.0))).to(torch.float32),
                ((square - (2.0 / n)) * math.sqrt(n / 8.0)).to(torch.float32),
                ((square * weight_tile) * (n / math.sqrt(120.0))).to(torch.float32),
                (
                    (square * square - (12.0 / (n * n)))
                    * (n**1.5 / math.sqrt(1536.0))
                ).to(torch.float32),
            )
            for power, edge in enumerate(polynomials):
                channel_slice = slice(power * group, (power + 1) * group)
                output[:, channel_slice].add_(
                    edge.transpose(0, 1).matmul(hidden[start:stop, channel_slice])
                )
        ctx.save_for_backward(weights)
        ctx.tile_rows = tile
        ctx.channels = channels
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[None, Tensor, None]:
        (weights,) = ctx.saved_tensors
        n = int(weights.shape[0])
        group = int(ctx.channels) // 4
        grad_hidden = torch.empty(
            (n, int(ctx.channels)), dtype=torch.float32, device=grad_output.device
        )
        for start in range(0, n, int(ctx.tile_rows)):
            stop = min(start + int(ctx.tile_rows), n)
            weight_tile = weights[start:stop]
            square = weight_tile * weight_tile
            polynomials = (
                (weight_tile * (1.0 / math.sqrt(2.0))).to(torch.float32),
                ((square - (2.0 / n)) * math.sqrt(n / 8.0)).to(torch.float32),
                ((square * weight_tile) * (n / math.sqrt(120.0))).to(torch.float32),
                (
                    (square * square - (12.0 / (n * n)))
                    * (n**1.5 / math.sqrt(1536.0))
                ).to(torch.float32),
            )
            for power, edge in enumerate(polynomials):
                channel_slice = slice(power * group, (power + 1) * group)
                grad_hidden[start:stop, channel_slice] = edge.matmul(
                    grad_output[:, channel_slice]
                )
        return None, grad_hidden, None


def stream_four_transports(weights: Tensor, hidden: Tensor, tile_rows: int) -> Tensor:
    return _StreamFourTransports.apply(weights, hidden, int(tile_rows))


class AcceleratedResidualObserver(OBSERVER.ResidualObserver):
    """Same parameters/cell/latch with weight-streamed recurrent transport."""

    def __init__(self, *, tile_rows: int = DEFAULT_TILE_ROWS, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tile_rows = int(tile_rows)
        if self.tile_rows <= 0 or self.tile_rows > 1024:
            raise ValueError("tile_rows must be in [1,1024]")

    def _streamed_normalized_messages(
        self, weights: Tensor, h_previous: Tensor
    ) -> tuple[Tensor, Tensor]:
        transported = stream_four_transports(weights, h_previous, self.tile_rows)
        groups: list[Tensor] = []
        log_rms: list[Tensor] = []
        for power in range(4):
            channel_slice = slice(
                power * self.channels_per_power,
                (power + 1) * self.channels_per_power,
            )
            value = transported[:, channel_slice]
            flat = (value * value).contiguous().view(-1)
            mean_square = torch.sum(flat, dtype=torch.float32) / float(flat.numel())
            rho = torch.sqrt(mean_square + OBSERVER.MESSAGE_EPS).to(torch.float32)
            groups.append(torch.clamp(value / rho, min=-5.0, max=5.0).to(torch.float32))
            log_rms.append(torch.clamp(torch.log(rho), min=-8.0, max=8.0).to(torch.float32))
        return torch.cat(groups, dim=1).contiguous(), torch.stack(log_rms).contiguous()

    def forward(
        self,
        standardized_locals: Tensor,
        weights: Tensor | None,
        host_rows: Tensor,
        sigma: Tensor,
        *,
        masked_optional_block: int | None = None,
        return_trace: bool = True,
        fixed_inputs_verified: bool = False,
    ) -> dict[str, Any]:
        if not fixed_inputs_verified:
            raise PermissionError("V6 requires once-verified fixed feature inputs")
        n = int(host_rows.shape[1])
        if tuple(standardized_locals.shape) != (OBSERVER.DEPTH, n, OBSERVER.LOCAL_FEATURES):
            raise ValueError("standardized-local shape changed")
        if tuple(host_rows.shape) != (OBSERVER.DEPTH, n) or tuple(sigma.shape) != (OBSERVER.DEPTH, n):
            raise ValueError("host/sigma shape changed")
        if self.arm == "local_no_recurrence":
            if weights is not None:
                raise AssertionError("local arm must bypass weights entirely")
        elif weights is None or tuple(weights.shape) != (OBSERVER.DEPTH, n, n):
            raise ValueError("recurrent arm requires float32[16,n,n] weights")
        tensors = (standardized_locals, host_rows, sigma)
        if any(value.dtype != torch.float32 for value in tensors):
            raise ValueError("V6 learned-boundary inputs must be float32")
        if weights is not None and weights.dtype != torch.float32:
            raise ValueError("V6 streamed weights must be float32")
        if any(value.device != host_rows.device for value in tensors) or (
            weights is not None and weights.device != host_rows.device
        ):
            raise ValueError("all V6 learned-boundary inputs must share one device")

        h_previous = torch.zeros(
            (n, self.hidden_channels), dtype=torch.float32, device=host_rows.device
        )
        active = True
        rows: list[Tensor] = []
        messages_trace: list[Tensor] = []
        log_rms_trace: list[Tensor] = []
        hidden_trace: list[Tensor] = []
        unit_trace: list[Tensor] = []
        delta_trace: list[Tensor] = []
        latch_trace: list[bool] = []
        parameter_vector = torch.cat(
            [parameter.reshape(-1) for parameter in self.ordered_parameters()]
        )
        for layer0 in range(OBSERVER.DEPTH):
            host = host_rows[layer0]
            if not active:
                messages = torch.zeros(
                    (n, self.hidden_channels), dtype=torch.float32, device=host.device
                )
                log_rms = torch.zeros(4, dtype=torch.float32, device=host.device)
                hidden = torch.zeros_like(h_previous)
                unit = torch.zeros(n, dtype=torch.float32, device=host.device)
                delta = torch.zeros_like(unit)
                prediction = torch.clamp_min(host, 0.0)
            else:
                if layer0 == 0 or self.arm == "local_no_recurrence":
                    messages = torch.zeros(
                        (n, self.hidden_channels), dtype=torch.float32, device=host.device
                    )
                    log_rms = torch.zeros(4, dtype=torch.float32, device=host.device)
                else:
                    assert weights is not None
                    messages, log_rms = self._streamed_normalized_messages(
                        weights[layer0], h_previous
                    )
                source_gates = torch.sigmoid(self.source_gate_logits)
                gated_local = self._gate_locals(
                    standardized_locals[layer0], source_gates,
                    masked_optional_block=masked_optional_block,
                )
                gated_messages: list[Tensor] = []
                gated_log_rms: list[Tensor] = []
                for power in range(4):
                    gate = source_gates[4 + power]
                    channel_slice = slice(
                        power * self.channels_per_power,
                        (power + 1) * self.channels_per_power,
                    )
                    gated_messages.append(messages[:, channel_slice] * gate)
                    gated_log_rms.append(log_rms[power] * gate)
                broadcast_log = torch.stack(gated_log_rms)[None, :].expand(n, 4)
                cell_input = torch.cat(
                    (gated_local, broadcast_log, torch.cat(gated_messages, dim=1)), dim=1
                )
                q1 = F.silu(self.core1(cell_input))
                q2 = F.silu(self.core2(q1))
                candidate = torch.tanh(self.candidate_head(q2))
                update_gate = torch.sigmoid(self.gate_head(q2))
                hidden = torch.tanh(messages + update_gate * (candidate - messages))
                unit = torch.tanh(self.mean_head(q2)).squeeze(1)
                if layer0 == 0:
                    delta = torch.zeros_like(unit)
                else:
                    delta = (sigma[layer0] * (1.0 / 32.0) * unit).to(torch.float32)
                prediction = torch.clamp_min((host + delta).to(torch.float32), 0.0)
                diagnostics = (
                    standardized_locals[layer0], messages, log_rms, cell_input,
                    hidden, unit, delta, prediction, parameter_vector,
                )
                finite = all(bool(torch.isfinite(value).all().item()) for value in diagnostics)
                positive = bool(torch.isfinite(sigma[layer0]).all().item()) and bool(
                    (sigma[layer0] > 0.0).all().item()
                )
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
            "predictions": torch.stack(rows), "latch": tuple(latch_trace)
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


def materialize_reference_edges(weights: Tensor) -> Tensor:
    """Target-free parity helper only; production workers never call this."""

    n = int(weights.shape[-1])
    layers = []
    for layer0 in range(int(weights.shape[0])):
        layers.append(torch.stack([_edge_tile(weights[layer0], power, n) for power in range(4)]))
    return torch.stack(layers).contiguous()
