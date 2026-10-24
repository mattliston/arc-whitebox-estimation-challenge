#!/Users/matt/anaconda3/bin/python3
"""Replace _TRANSPORT_PAYLOAD in an already-integrated estimator, deterministically.

TAG=payload-splicer  build_integrated_estimator.py only INSERTS a payload into
a non-integrated host; the composed deployment candidate already carries the
frozen no-eigh payload, so a retrained corrector must REPLACE the block
_TRANSPORT_MODEL_ARTIFACT_SHA256 = "..." / _TRANSPORT_PAYLOAD = {...} in
place.  Extraction and literal rendering are IMPORTED from the audited
exporter (deployment_prep/export_literal_payload.py, sha-pinned), never
reimplemented.  Self-checks: the output compiles, every byte outside the
replaced block is identical, and the spliced literal round-trips bit-exact
(float32) against the torch state_dict.  Offline promotion machinery; never
imported by a submitted estimator; no targets, no store writes.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
EXPORTER_DIR = P2 / "public_q24_native_transport_energy_pilot_v1" / "deployment_prep"
EXPORTER_SHA = "1d96c49f91fedd98d66561e93c74b2eb4cbc71a902d88ccf7955b86193dc49e3"
START_MARKER = "_TRANSPORT_MODEL_ARTIFACT_SHA256 = "
END_MARKER = "\n}\n"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_fragment(exporter, model: Path) -> tuple[str, dict]:
    """Exactly build_integrated_estimator.payload_fragment, from the pinned exporter."""
    payload = exporter.extract_payload(model)
    rendered = exporter.render_literal_module(payload, exporter.sha256(model))
    fragment = rendered[rendered.index("MODEL_ARTIFACT_SHA256 ="):]
    fragment = fragment.replace(
        "MODEL_ARTIFACT_SHA256", "_TRANSPORT_MODEL_ARTIFACT_SHA256", 1
    )
    fragment = fragment.replace("PAYLOAD =", "_TRANSPORT_PAYLOAD =", 1)
    return fragment.rstrip(), payload


def spliced_block(source: str) -> str:
    count = source.count(START_MARKER)
    assert count == 1, f"payload start marker occurs {count}x, need 1"
    begin = source.index(START_MARKER)
    stop = source.index(END_MARKER, begin) + len(END_MARKER)
    return source[begin:stop]


def literal_payload(source: str) -> dict:
    block = spliced_block(source)
    dict_text = block[block.index("_TRANSPORT_PAYLOAD = {") + len("_TRANSPORT_PAYLOAD = "):]
    return ast.literal_eval(dict_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--estimator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    model = arguments.model.resolve()
    estimator = arguments.estimator.resolve()
    output = arguments.output.resolve()
    assert output != estimator, "write the spliced card to a new path"
    if output.exists():
        raise FileExistsError(output)

    exporter_path = EXPORTER_DIR / "export_literal_payload.py"
    assert sha256_path(exporter_path) == EXPORTER_SHA, "audited exporter drifted"
    sys.path.insert(0, str(EXPORTER_DIR))
    import export_literal_payload as exporter

    fragment, payload = payload_fragment(exporter, model)
    source = estimator.read_text(encoding="utf-8")
    old_block = spliced_block(source)
    new_block = fragment + "\n"
    spliced = source.replace(old_block, new_block)
    assert spliced.count(START_MARKER) == 1

    # Self-check 1: compiles.
    compile(spliced, str(output), "exec")
    # Self-check 2: every byte outside the replaced block is identical.
    prefix, suffix = source.split(old_block)
    assert spliced == prefix + new_block + suffix
    # Self-check 3: the spliced literal round-trips bit-exact against torch.
    literal = literal_payload(spliced)
    assert set(literal) == set(payload)
    for name, exported in payload.items():
        materialized = np.asarray(literal[name], dtype=np.float32)
        assert materialized.shape == exported.shape, name
        assert np.array_equal(
            materialized.view(np.uint32), exported.view(np.uint32)
        ), f"literal round-trip not bit-exact: {name}"

    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(spliced, encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({
        "schema": "transport-payload-splice-receipt-v1",
        "model_sha256": exporter.sha256(model),
        "estimator_in_sha256": sha256_path(estimator),
        "output_sha256": sha256_path(output),
        "output_bytes": output.stat().st_size,
        "replaced_block_bytes": len(old_block),
        "new_block_bytes": len(new_block),
        "literal_bit_exact_roundtrip": True,
    }, indent=1))


if __name__ == "__main__":
    main()
