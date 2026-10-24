#!/usr/bin/env python3
"""Build the D1 #328586 closure with frozen alpha=.75 J and q10:q15 head.

This is mechanical offline promotion machinery.  It opens no weights, target,
fixture, score, board resource, or network path.  The Q24/D1 closure prefix is
kept byte-identical; only the two literal model payloads and their inference
plumbing are changed.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import pprint
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
SOURCE = P2 / "p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py"
ALPHA_MODEL = P2 / "public_q24_native_j_checkpoint_interpolation_v1/INTERPOLATED_MODEL.pt"
QHEAD_MODEL = P2 / "public_q24_native_j_q10q15_residual_head_v1/Q_SUFFIX_HEAD.pt"
EXPORTER = (
    P2
    / "public_q24_native_transport_energy_pilot_v1/deployment_prep"
    / "export_literal_payload.py"
)
DESTINATION = HERE / "estimator.py"

SOURCE_SHA256 = "f2ec257c891241b57c85803c70c73f365e075711e2b2cd214d9b58250762892e"
ALPHA_MODEL_SHA256 = "fc65528720c7e1378ed7011b7b201f4086217a544b092de208ea1064656d9ef5"
QHEAD_MODEL_SHA256 = "5fb57a510d933940f19ddd5ab11650c3a966c2039c6bd12f73988457c2ce91b8"
EXPORTER_SHA256 = "1d96c49f91fedd98d66561e93c74b2eb4cbc71a902d88ccf7955b86193dc49e3"

TRANSPORT_START = "_TRANSPORT_MODEL_ARTIFACT_SHA256 = "
PAYLOAD_END = "\n}\n"
RUNTIME_START = "def transport_correction(raw, sigma, weights, payload):\n"
RUNTIME_END = "\n\n\ndef _predict_corrected(mlp: MLP):\n"
PREDICT_END = "\nclass Estimator(BaseEstimator):\n"

QHEAD_STATE_TO_PAYLOAD = {
    "centers": "centers",
    "scales": "scales",
    "layer_rms": "layer_rms",
    "trunk.0.weight": "trunk0_weight",
    "trunk.0.bias": "trunk0_bias",
    "trunk.2.weight": "trunk2_weight",
    "trunk.2.bias": "trunk2_bias",
    "head_weight": "head_weight",
    "head_bias": "head_bias",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_exporter():
    if sha256(EXPORTER) != EXPORTER_SHA256:
        raise AssertionError("audited exact-J exporter drifted")
    spec = importlib.util.spec_from_file_location("alpha075_payload_exporter", EXPORTER)
    if spec is None or spec.loader is None:
        raise ImportError(EXPORTER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise AssertionError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def transport_block(text: str) -> str:
    if text.count(TRANSPORT_START) != 1:
        raise AssertionError("transport payload marker changed")
    start = text.index(TRANSPORT_START)
    stop = text.index(PAYLOAD_END, start) + len(PAYLOAD_END)
    return text[start:stop]


def transport_fragment(exporter) -> tuple[str, dict[str, np.ndarray]]:
    payload = exporter.extract_payload(ALPHA_MODEL)
    rendered = exporter.render_literal_module(payload, ALPHA_MODEL_SHA256)
    fragment = rendered[rendered.index("MODEL_ARTIFACT_SHA256 ="):]
    fragment = fragment.replace(
        "MODEL_ARTIFACT_SHA256", "_TRANSPORT_MODEL_ARTIFACT_SHA256", 1
    )
    fragment = fragment.replace("PAYLOAD =", "_TRANSPORT_PAYLOAD =", 1)
    return fragment.rstrip() + "\n", payload


def qhead_payload() -> dict[str, np.ndarray]:
    artifact = torch.load(QHEAD_MODEL, map_location="cpu", weights_only=False)
    if artifact.get("schema") != "q24-native-j-q10q15-residual-head-checkpoint-v1":
        raise AssertionError("q-head artifact schema changed")
    state = artifact["state_dict"]
    if set(state) != set(QHEAD_STATE_TO_PAYLOAD):
        raise AssertionError("q-head state schema changed")
    payload = {
        output_name: np.ascontiguousarray(
            state[input_name].detach().cpu().numpy(), dtype=np.float32
        )
        for input_name, output_name in QHEAD_STATE_TO_PAYLOAD.items()
    }
    expected = {
        "centers": (16, 19),
        "scales": (16, 19),
        "layer_rms": (16,),
        "trunk0_weight": (64, 22),
        "trunk0_bias": (64,),
        "trunk2_weight": (64, 64),
        "trunk2_bias": (64,),
        "head_weight": (6, 64),
        "head_bias": (6,),
    }
    if {name: value.shape for name, value in payload.items()} != expected:
        raise AssertionError("q-head shape contract changed")
    if not all(np.isfinite(value).all() for value in payload.values()):
        raise FloatingPointError("q-head payload is nonfinite")
    trainable = sum(
        payload[name].size
        for name in (
            "trunk0_weight", "trunk0_bias", "trunk2_weight", "trunk2_bias",
            "head_weight", "head_bias",
        )
    )
    if trainable != 6_022:
        raise AssertionError("q-head trainable parameter count changed")
    return payload


def qhead_fragment(payload: dict[str, np.ndarray]) -> str:
    rows = [
        f'_QHEAD_MODEL_ARTIFACT_SHA256 = "{QHEAD_MODEL_SHA256}"',
        "_QHEAD_PAYLOAD = {",
    ]
    for name, value in payload.items():
        literal = pprint.pformat(
            value.tolist(), width=100, compact=True, sort_dicts=False
        )
        rows.append(f"    {name!r}: {literal},")
    rows.extend(("}", "", "QHEAD_FIRST_LAYER = 10", ""))
    return "\n".join(rows)


NEW_RUNTIME = '''def transport_correction(raw, sigma, weights, payload, qhead_payload):
    """Run frozen alpha=.75 J plus the frozen physical q10:q15 head."""
    parameters = materialize_parameters(payload)
    qparameters = materialize_parameters(qhead_payload)
    raw = fnp.asarray(raw, dtype=fnp.float32)
    sigma = fnp.asarray(sigma, dtype=fnp.float32)

    depth = int(raw.shape[0])
    width = int(raw.shape[1])
    channels = int(parameters["source_bias"].shape[1])
    states = fnp.zeros((width, channels), dtype=fnp.float32)
    energies = fnp.zeros((width, channels), dtype=fnp.float32)
    qstate = fnp.zeros(width, dtype=fnp.float32)
    identity = fnp.eye(channels, dtype=fnp.float32)
    outputs = []
    qoutputs = []

    for layer in range(depth):
        if layer < FIRST_CORRECTED_LAYER:
            states = fnp.zeros((width, channels), dtype=fnp.float32)
            energies = fnp.zeros((width, channels), dtype=fnp.float32)
            outputs.append(states[:, 0])
            qoutputs.append(qstate)
            continue

        if layer == FIRST_CORRECTED_LAYER:
            messages = fnp.zeros((width, channels), dtype=fnp.float32)
            energies = fnp.zeros((width, channels), dtype=fnp.float32)
        else:
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

        if layer >= QHEAD_FIRST_LAYER:
            qmessage = raw[layer, :, 5] * (qstate @ layer_weights)
            qrms = qparameters["layer_rms"][layer]
            qtail = fnp.clip(
                (raw[layer, :, 1:] - qparameters["centers"][layer])
                / qparameters["scales"][layer],
                -8.0,
                8.0,
            )
            base_q = states[:, 0] - messages[:, 0]
            qfeatures = fnp.concatenate(
                (
                    raw[layer, :, :1],
                    qtail,
                    fnp.clip(states[:, 0] / qrms, -8.0, 8.0)[:, None],
                    fnp.clip(
                        base_q
                        / fnp.maximum(sigma[layer] * qrms, 2.0**-30),
                        -8.0,
                        8.0,
                    )[:, None],
                ),
                axis=1,
            )
            qhidden = _silu(
                qfeatures @ fnp.transpose(qparameters["trunk0_weight"])
                + qparameters["trunk0_bias"]
            )
            qhidden = _silu(
                qhidden @ fnp.transpose(qparameters["trunk2_weight"])
                + qparameters["trunk2_bias"]
            )
            slot = layer - QHEAD_FIRST_LAYER
            qstandard = (
                qhidden @ qparameters["head_weight"][slot]
                + qparameters["head_bias"][slot]
            )
            qsource = sigma[layer] * qrms * qstandard
            qstate = qmessage + qsource

        outputs.append(states[:, 0])
        qoutputs.append(qstate)

    correction = fnp.asarray(fnp.stack(outputs, axis=0), dtype=fnp.float32)
    qcorrection = fnp.asarray(fnp.stack(qoutputs, axis=0), dtype=fnp.float32)
    if not bool(fnp.all(fnp.isfinite(correction))):
        raise FloatingPointError("alpha=.75 native-J correction is nonfinite")
    if not bool(fnp.all(fnp.isfinite(qcorrection))):
        raise FloatingPointError("q10:q15 correction is nonfinite")
    return correction, qcorrection
'''


NEW_PREDICT = '''def _predict_corrected(mlp: MLP):
    host, native_raw, native_sigma = _predict(mlp, return_native_tape=True)
    correction, qcorrection = transport_correction(
        native_raw,
        native_sigma,
        mlp.weights,
        _TRANSPORT_PAYLOAD,
        _QHEAD_PAYLOAD,
    )
    output = fnp.asarray(
        fnp.maximum(host + correction + qcorrection, 0.0), dtype=fnp.float32
    )
    if not bool(fnp.all(fnp.isfinite(output))):
        raise FloatingPointError("D1 plus alpha=.75 J/q-head produced a nonfinite mean")
    return output
'''


def main() -> None:
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    expected = {
        SOURCE: SOURCE_SHA256,
        ALPHA_MODEL: ALPHA_MODEL_SHA256,
        QHEAD_MODEL: QHEAD_MODEL_SHA256,
        EXPORTER: EXPORTER_SHA256,
    }
    for path, digest in expected.items():
        observed = sha256(path)
        if observed != digest:
            raise AssertionError((str(path), observed, digest))

    source = SOURCE.read_text(encoding="utf-8")
    exporter = load_exporter()
    new_transport, alpha_payload = transport_fragment(exporter)
    old_transport = transport_block(source)
    built = replace_once(source, old_transport, new_transport, "alpha=.75 payload")

    qpayload = qhead_payload()
    qfragment = qhead_fragment(qpayload)
    built = replace_once(
        built,
        "COHERENCE_FLOOR = 1.0e-4\n",
        qfragment + "\nCOHERENCE_FLOOR = 1.0e-4\n",
        "q-head payload insertion",
    )

    runtime_start = built.index(RUNTIME_START)
    runtime_stop = built.index(RUNTIME_END, runtime_start)
    old_runtime = built[runtime_start:runtime_stop]
    built = replace_once(built, old_runtime, NEW_RUNTIME.rstrip(), "runtime")

    predict_start = built.index("def _predict_corrected(mlp: MLP):\n")
    predict_stop = built.index(PREDICT_END, predict_start)
    old_predict = built[predict_start:predict_stop]
    built = replace_once(built, old_predict, NEW_PREDICT.rstrip() + "\n", "predict")

    compile(built, str(DESTINATION), "exec")
    if not built.startswith(source[: source.index(TRANSPORT_START)]):
        raise AssertionError("D1 closure prefix changed")
    if built.count("_TRANSPORT_MODEL_ARTIFACT_SHA256 = ") != 1:
        raise AssertionError("transport payload multiplicity changed")
    if built.count("_QHEAD_MODEL_ARTIFACT_SHA256 = ") != 1:
        raise AssertionError("q-head payload multiplicity changed")

    # Both rendered literal payloads must round-trip bit exactly.
    namespace: dict[str, object] = {}
    payload_region = built[
        built.index(TRANSPORT_START):built.index("COHERENCE_FLOOR = 1.0e-4")
    ]
    exec(payload_region, namespace)
    for name, expected_value in alpha_payload.items():
        observed = np.asarray(namespace["_TRANSPORT_PAYLOAD"][name], dtype=np.float32)
        if not np.array_equal(observed.view(np.uint32), expected_value.view(np.uint32)):
            raise AssertionError(f"transport literal mismatch: {name}")
    for name, expected_value in qpayload.items():
        observed = np.asarray(namespace["_QHEAD_PAYLOAD"][name], dtype=np.float32)
        if not np.array_equal(observed.view(np.uint32), expected_value.view(np.uint32)):
            raise AssertionError(f"q-head literal mismatch: {name}")

    temporary = DESTINATION.with_name(f".{DESTINATION.name}.tmp")
    temporary.write_text(built, encoding="utf-8")
    temporary.replace(DESTINATION)
    print(
        {
            "source_sha256": SOURCE_SHA256,
            "alpha_model_sha256": ALPHA_MODEL_SHA256,
            "qhead_model_sha256": QHEAD_MODEL_SHA256,
            "destination_sha256": sha256(DESTINATION),
            "destination_bytes": DESTINATION.stat().st_size,
            "closure_prefix_byte_identical": True,
            "alpha_literal_roundtrip": True,
            "qhead_literal_roundtrip": True,
        }
    )


if __name__ == "__main__":
    main()

