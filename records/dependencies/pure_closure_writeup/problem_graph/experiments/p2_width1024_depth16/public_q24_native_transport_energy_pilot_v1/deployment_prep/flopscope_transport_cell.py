"""Target-free FlopScope inference for the frozen P2R0F2J cell.

This module is a deployment-preparation helper, not a submitted estimator.
It deliberately contains no Torch import.  ``transport_correction`` mirrors
the autonomous (exposure=1) forward pass of
``TransportEnergyMultiStateResidual`` and accepts an explicit, already-fitted
parameter payload.  A final package can inline this function and replace the
payload argument with literal coefficient tables.
"""

from __future__ import annotations

import flopscope.numpy as fnp


COHERENCE_FLOOR = 1.0e-4
FIRST_CORRECTED_LAYER = 2


def _silu(value):
    """SiLU using only FlopScope-array operations."""
    return value / (1.0 + fnp.exp(-value))


def materialize_parameters(payload):
    """Convert literal/NumPy payload arrays to metered float32 arrays once."""
    return {
        name: fnp.asarray(value, dtype=fnp.float32)
        for name, value in payload.items()
    }


def transport_correction(raw, sigma, weights, payload):
    """Run exact autonomous J inference on one target-free native Q24 tape.

    Args:
        raw: ``(depth, width, 20)`` native raw feature tape.
        sigma: ``(depth, width)`` native preactivation standard deviations.
        weights: ``(depth, width, width)`` MLP weight matrices.
        payload: fitted buffers and parameters, with the names documented by
            ``verify_synthetic_parity.py``.

    Returns:
        A float32 ``(depth, width)`` residual correction.  Layers zero and one
        are identically zero, matching the training source.
    """
    parameters = materialize_parameters(payload)
    raw = fnp.asarray(raw, dtype=fnp.float32)
    sigma = fnp.asarray(sigma, dtype=fnp.float32)

    depth = int(raw.shape[0])
    width = int(raw.shape[1])
    channels = int(parameters["source_bias"].shape[1])
    states = fnp.zeros((width, channels), dtype=fnp.float32)
    energies = fnp.zeros((width, channels), dtype=fnp.float32)
    identity = fnp.eye(channels, dtype=fnp.float32)
    outputs = []

    for layer in range(depth):
        if layer < FIRST_CORRECTED_LAYER:
            states = fnp.zeros((width, channels), dtype=fnp.float32)
            energies = fnp.zeros((width, channels), dtype=fnp.float32)
            outputs.append(states[:, 0])
            continue

        if layer == FIRST_CORRECTED_LAYER:
            messages = fnp.zeros((width, channels), dtype=fnp.float32)
            energies = fnp.zeros((width, channels), dtype=fnp.float32)
        else:
            # Keep the benchmark's list-of-layer-arrays representation and
            # materialize one matrix at a time.  This avoids a redundant
            # 16x1024x1024 submission-time stack/copy.
            layer_weights = fnp.asarray(weights[layer], dtype=fnp.float32)
            messages = fnp.transpose(
                fnp.transpose(states) @ layer_weights
            )
            energies = fnp.transpose(
                fnp.transpose(states * states)
                @ (layer_weights * layer_weights)
            )
            phi = raw[layer, :, 5, None]
            messages = phi * messages
            energies = (phi * phi) * energies

        tail = fnp.clip(
            (raw[layer, :, 1:] - parameters["centers"][layer])
            / parameters["scales"][layer],
            -8.0,
            8.0,
        )
        rms = parameters["layer_rms"][layer]
        rms2 = rms * rms
        message_rms = fnp.clip(messages / rms, -8.0, 8.0)
        message_sigma = fnp.clip(
            messages
            / fnp.maximum(sigma[layer, :, None] * rms, 2.0**-30),
            -8.0,
            8.0,
        )
        log_energy = fnp.clip(
            fnp.log1p(fnp.maximum(energies / rms2, 0.0)),
            0.0,
            8.0,
        )
        coherence = fnp.clip(
            messages
            / fnp.sqrt(energies + COHERENCE_FLOOR * rms2),
            -8.0,
            8.0,
        )
        features = fnp.concatenate(
            (
                raw[layer, :, :1],
                tail,
                message_rms,
                message_sigma,
                log_energy,
                coherence,
            ),
            axis=1,
        )

        hidden = _silu(
            features @ fnp.transpose(parameters["trunk0_weight"])
            + parameters["trunk0_bias"]
        )
        hidden = _silu(
            hidden @ fnp.transpose(parameters["trunk2_weight"])
            + parameters["trunk2_bias"]
        )
        source_standard = (
            hidden @ parameters["source_weight"][layer]
            + parameters["source_bias"][layer]
        )
        source = (sigma[layer] * rms)[:, None] * source_standard
        gate_standard = (
            hidden @ parameters["gate_weight"][layer]
            + parameters["gate_bias"][layer]
        )
        dynamic_gain = 1.0 + 0.25 * fnp.tanh(
            gate_standard + parameters["transport_delta"][layer]
        )
        mixing = identity + 0.12 * fnp.tanh(
            parameters["mix_delta"][layer]
        )
        states = (dynamic_gain * messages) @ mixing + source
        outputs.append(states[:, 0])

    correction = fnp.asarray(fnp.stack(outputs, axis=0), dtype=fnp.float32)
    if not bool(fnp.all(fnp.isfinite(correction))):
        raise FloatingPointError("P2R0F2J correction produced a nonfinite value")
    return correction
