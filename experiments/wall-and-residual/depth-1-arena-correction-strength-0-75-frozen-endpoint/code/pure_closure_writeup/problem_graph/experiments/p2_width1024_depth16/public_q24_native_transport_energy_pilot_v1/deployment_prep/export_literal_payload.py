#!/Users/matt/anaconda3/bin/python3
"""Export a fitted exact-J Torch state into dependency-free Python literals.

The generated module imports nothing.  It can be pasted/inlined into the Q24
estimator and materialized through ``flopscope.numpy.asarray``.  This helper is
offline promotion machinery and is never imported by the submitted estimator.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import pprint

import numpy as np
import torch


STATE_TO_PAYLOAD = {
    "centers": "centers",
    "scales": "scales",
    "layer_rms": "layer_rms",
    "trunk.0.weight": "trunk0_weight",
    "trunk.0.bias": "trunk0_bias",
    "trunk.2.weight": "trunk2_weight",
    "trunk.2.bias": "trunk2_bias",
    "source_weight": "source_weight",
    "source_bias": "source_bias",
    "gate_weight": "gate_weight",
    "gate_bias": "gate_bias",
    "transport_delta": "transport_delta",
    "mix_delta": "mix_delta",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_payload(artifact_path: Path) -> dict[str, np.ndarray]:
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or "state_dict" not in artifact:
        raise TypeError("expected a frozen exact-J artifact with state_dict")
    state = artifact["state_dict"]
    if set(state) != set(STATE_TO_PAYLOAD):
        raise AssertionError(
            f"exact-J state schema changed: {sorted(set(state) ^ set(STATE_TO_PAYLOAD))}"
        )
    payload = {
        payload_name: np.ascontiguousarray(
            state[state_name].detach().cpu().numpy(), dtype=np.float32
        )
        for state_name, payload_name in STATE_TO_PAYLOAD.items()
    }
    if not all(np.isfinite(value).all() for value in payload.values()):
        raise FloatingPointError("fitted payload contains a nonfinite value")
    depth, local_tail = payload["centers"].shape
    hidden, input_features = payload["trunk0_weight"].shape
    channels = payload["source_bias"].shape[1]
    expected_shapes = {
        "centers": (depth, local_tail),
        "scales": (depth, local_tail),
        "layer_rms": (depth,),
        "trunk0_weight": (hidden, input_features),
        "trunk0_bias": (hidden,),
        "trunk2_weight": (hidden, hidden),
        "trunk2_bias": (hidden,),
        "source_weight": (depth, hidden, channels),
        "source_bias": (depth, channels),
        "gate_weight": (depth, hidden, channels),
        "gate_bias": (depth, channels),
        "transport_delta": (depth, channels),
        "mix_delta": (depth, channels, channels),
    }
    for name, shape in expected_shapes.items():
        if payload[name].shape != shape:
            raise AssertionError(f"payload shape changed: {name} {payload[name].shape} != {shape}")
    if (depth, local_tail, hidden, input_features, channels) != (16, 19, 96, 36, 4):
        raise AssertionError("exact deployed J architecture changed")
    if sum(value.size for name, value in payload.items() if name not in {"centers", "scales", "layer_rms"}) != 25_600:
        raise AssertionError("exact J trainable parameter count changed")
    return payload


def render_literal_module(payload: dict[str, np.ndarray], source_binding: str) -> str:
    rows = [
        '"""Generated exact-J float32 payload; no runtime dependency imports."""',
        "",
        f'MODEL_ARTIFACT_SHA256 = "{source_binding}"',
        "PAYLOAD = {",
    ]
    for name, value in payload.items():
        literal = pprint.pformat(value.tolist(), width=100, compact=True, sort_dicts=False)
        rows.append(f"    {name!r}: {literal},")
    rows.extend(("}", ""))
    return "\n".join(rows)


def write_literal_module(artifact_path: Path, output_path: Path) -> dict[str, object]:
    if output_path.exists():
        raise FileExistsError(output_path)
    payload = extract_payload(artifact_path)
    model_hash = sha256(artifact_path)
    text = render_literal_module(payload, model_hash)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(output_path)
    return {
        "model_artifact_sha256": model_hash,
        "literal_module_sha256": sha256(output_path),
        "literal_module_bytes": output_path.stat().st_size,
        "arrays": len(payload),
        "float32_values": sum(value.size for value in payload.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(write_literal_module(arguments.model.resolve(), arguments.output.resolve()))


if __name__ == "__main__":
    main()
