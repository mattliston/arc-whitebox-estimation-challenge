#!/usr/bin/env python3
"""Summarize the already-sealed target-free aug7 correlation-max channel.

This program never loads weights, targets, names, predictions, or an estimator.
It verifies each sealed aug7 file against its row manifest, then reads channel
23 (local aug7 index 3) and prints a deterministic JSON summary to stdout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
AUG = P2 / "public_aug7_source_observable_tape_v1"
ROWS = AUG / "rows" / "observer_fit"
GENERATION = AUG / "GENERATION_RECEIPT.json"
EXPECTED_GENERATION_SHA256 = (
    "d2e36181b09c0048791f130ba75897b2962c44eeab87b37086c8f4a6fcb5bfbd"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if sha256_file(GENERATION) != EXPECTED_GENERATION_SHA256:
        raise AssertionError("sealed aug7 generation receipt changed")
    generation = json.loads(GENERATION.read_text())
    if generation.get("status") != "PASS_GENERATION_688_PARITY":
        raise AssertionError("aug7 generation is not sealed complete")
    if generation.get("observer_selection_targets_opened") != 0:
        raise AssertionError("observer-selection firewall changed")
    if generation.get("public_name_joins") != 0:
        raise AssertionError("name-join firewall changed")

    paths = sorted(ROWS.glob("*/row_manifest.json"))
    if len(paths) != 688:
        raise AssertionError(f"expected 688 sealed rows, found {len(paths)}")
    blocks: list[np.ndarray] = []
    for expected_rank, manifest_path in enumerate(paths):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("role_rank") != expected_rank:
            raise AssertionError(f"role-rank discontinuity at {expected_rank}")
        if manifest.get("status") != "SEALED_ROW_AUG7":
            raise AssertionError(f"unsealed row at {expected_rank}")
        if manifest.get("targets_read") != 0:
            raise AssertionError(f"target firewall changed at {expected_rank}")
        if manifest.get("observer_selection_opened") != 0:
            raise AssertionError(f"selection firewall changed at {expected_rank}")
        if manifest.get("parity_pass") is not True:
            raise AssertionError(f"parity failure at {expected_rank}")
        array_path = manifest_path.parent / "aug7_locals.npy"
        if sha256_file(array_path) != manifest.get("aug7_sha256"):
            raise AssertionError(f"aug7 file hash mismatch at {expected_rank}")
        array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != [16, 1024, 7] or array.dtype != np.float32:
            raise AssertionError(f"aug7 shape/dtype mismatch at {expected_rank}")
        blocks.append(np.asarray(array[:, :, 3]))

    values = np.stack(blocks)
    quantiles = (0.5, 0.9, 0.99, 0.999, 1.0)
    thresholds = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    result = {
        "schema": "p2-b2b3-existing-source-domain-summary-v1",
        "source": "public_aug7_source_observable_tape_v1 channel 23",
        "source_generation_receipt_sha256": EXPECTED_GENERATION_SHA256,
        "rows": len(paths),
        "shape": list(values.shape),
        "targets_read": 0,
        "observer_selection_targets_opened": 0,
        "public_name_joins": 0,
        "global_quantiles": {
            str(q): float(np.quantile(values, q)) for q in quantiles
        },
        "fractions_strictly_greater_than": {
            str(t): float(np.mean(values > t)) for t in thresholds
        },
        "layer_q99": [
            float(np.quantile(values[:, layer, :], 0.99))
            for layer in range(16)
        ],
        "layer_max": [
            float(np.max(values[:, layer, :])) for layer in range(16)
        ],
        "note": (
            "Each cell is the maximum absolute off-diagonal correlation in "
            "one source row; this is not an edge-count or tree-residual census."
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
