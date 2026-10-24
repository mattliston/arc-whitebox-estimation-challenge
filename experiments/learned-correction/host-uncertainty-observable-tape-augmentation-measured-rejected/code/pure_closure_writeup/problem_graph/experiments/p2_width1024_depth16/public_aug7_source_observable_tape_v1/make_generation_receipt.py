#!/Users/matt/anaconda3/bin/python3
"""Aggregate the 688 aug7 row manifests into GENERATION_RECEIPT.json."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROWS_DIR = HERE / "rows/observer_fit"
OUTPUT = HERE / "GENERATION_RECEIPT.json"
ROWS = 688
DEPTH = 16
WIDTH = 1024
AUG_CHANNELS = 7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    seal = json.loads((HERE / "PREREGISTRATION_SEAL.json").read_text())
    program_sha = sha256(HERE / "generate_aug7.py")
    prereg_sha = sha256(HERE / "PREREGISTRATION.json")
    if seal["program_sha256"] != program_sha or seal["preregistration_sha256"] != prereg_sha:
        raise AssertionError("generation package changed after seal")
    roster: list[str] = []
    parity_pass = 0
    total_wall = 0.0
    total_flops = 0
    cap_events = []
    channel_max_abs = np.zeros(AUG_CHANNELS)
    channel_nonzero_rows = np.zeros(AUG_CHANNELS, dtype=np.int64)
    for rank in range(ROWS):
        row_dir = ROWS_DIR / f"{rank:04d}"
        manifest_path = row_dir / "row_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("status") != "SEALED_ROW_AUG7"
            or manifest.get("parity_pass") is not True
            or int(manifest.get("role_rank", -1)) != rank
            or manifest.get("program_sha256") != program_sha
            or manifest.get("preregistration_sha256") != prereg_sha
        ):
            raise AssertionError(f"rank {rank}: row manifest fails receipt criteria")
        if sha256(row_dir / "aug7_locals.npy") != manifest["aug7_sha256"]:
            raise AssertionError(f"rank {rank}: payload hash mismatch")
        aug = np.load(row_dir / "aug7_locals.npy", allow_pickle=False)
        if aug.shape != (DEPTH, WIDTH, AUG_CHANNELS) or not np.isfinite(aug).all():
            raise AssertionError(f"rank {rank}: payload schema violation")
        row_max = np.abs(aug).max(axis=(0, 1))
        channel_max_abs = np.maximum(channel_max_abs, row_max)
        channel_nonzero_rows += (row_max > 0).astype(np.int64)
        roster.append(sha256(manifest_path))
        parity_pass += 1
        total_wall += float(manifest["wall_seconds"])
        total_flops += int(manifest["flops_used"])
        cap_events.append(int(manifest["cap_events"]))
    receipt = {
        "schema": "public-aug7-source-observable-generation-receipt-v1",
        "node": "P2R0F2Q",
        "stage": "generation",
        "status": "PASS_GENERATION_688_PARITY",
        "rows": ROWS,
        "parity_pass_rows": parity_pass,
        "row_manifest_sha256": roster,
        "roster_digest_sha256": hashlib.sha256(
            ("\n".join(roster) + "\n").encode("ascii")
        ).hexdigest(),
        "total_wall_seconds": total_wall,
        "mean_wall_seconds": total_wall / ROWS,
        "total_flops": total_flops,
        "cap_events_min": min(cap_events),
        "cap_events_max": max(cap_events),
        "channel_max_abs": channel_max_abs.tolist(),
        "channel_nonzero_rows": channel_nonzero_rows.tolist(),
        "channel_order": [
            "discarded_k3_signed",
            "discarded_k3_absolute",
            "covariance_series_order10_row_share",
            "max_offdiagonal_abs_correlation",
            "coincident_bond_degree",
            "rho2_neighbor_skew",
            "rho2_neighbor_alpha",
        ],
        "preregistration_sha256": prereg_sha,
        "program_sha256": program_sha,
        "seal_sha256": sha256(HERE / "PREREGISTRATION_SEAL.json"),
        "audit_receipt_sha256": sha256(HERE / "AUDIT_RECEIPT.json"),
        "observer_selection_targets_opened": 0,
        "public_name_joins": 0,
        "network_access": 0,
    }
    OUTPUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "rows": ROWS,
                "mean_wall_seconds": round(receipt["mean_wall_seconds"], 3),
                "channel_max_abs": [round(v, 4) for v in receipt["channel_max_abs"]],
                "channel_nonzero_rows": receipt["channel_nonzero_rows"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
