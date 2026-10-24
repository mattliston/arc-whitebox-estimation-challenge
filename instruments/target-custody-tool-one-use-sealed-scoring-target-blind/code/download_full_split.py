#!/usr/bin/env python3
"""Download only the 63 authority-pinned full Parquet objects.

This deliberately bypasses both the whestbench split wrapper and datasets'
``config`` resolution.  The Hugging Face client receives an explicit list of
63 paths and the immutable commit SHA, and treats every file as opaque bytes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from huggingface_hub import snapshot_download

from full_data_gate import (
    AUTHORITY,
    FULL,
    HERE,
    load_full_manifest,
    pinned_snapshot_path,
    validate_pinned_full_snapshot,
)


CACHE_DIR = HERE / ".hf-cache"
MIN_FREE_AFTER_DOWNLOAD_BYTES = 20 * (1 << 30)


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    manifest = load_full_manifest()
    paths = [str(row["path"]) for row in manifest]
    expected_bytes = sum(int(row["size"]) for row in manifest)
    free_bytes = shutil.disk_usage(HERE).free
    if free_bytes < expected_bytes + MIN_FREE_AFTER_DOWNLOAD_BYTES:
        raise RuntimeError(
            f"insufficient free disk: {free_bytes} < "
            f"{expected_bytes + MIN_FREE_AFTER_DOWNLOAD_BYTES}"
        )

    if not args.execute and not args.dry_run:
        print(
            json.dumps(
                {
                    "status": "PREFLIGHT_ONLY__PASS_EXACT_MANIFEST",
                    "repository": AUTHORITY["repository"],
                    "revision_tag": AUTHORITY["revision"],
                    "download_revision": AUTHORITY["pinned_commit_sha"],
                    "shards": len(paths),
                    "bytes": expected_bytes,
                    "gib": expected_bytes / (1 << 30),
                    "free_bytes": free_bytes,
                    "target_values_decoded": 0,
                    "execute_command": (
                        f"{HERE / '.venv/bin/python'} {Path(__file__).resolve()} "
                        "--execute --max-workers 4"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    result = snapshot_download(
        repo_id=AUTHORITY["repository"],
        repo_type="dataset",
        revision=AUTHORITY["pinned_commit_sha"],
        cache_dir=CACHE_DIR,
        allow_patterns=paths,
        max_workers=args.max_workers,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        remote = sorted(
            (
                {
                    "path": item.filename,
                    "size": item.file_size,
                    "will_download": item.will_download,
                    "will_download_bytes": item.file_size if item.will_download else 0,
                }
                for item in result
            ),
            key=lambda row: row["path"],
        )
        if [row["path"] for row in remote] != paths:
            raise AssertionError("Hub dry-run path list differs from exact manifest")
        if [row["size"] for row in remote] != [int(row["size"]) for row in manifest]:
            raise AssertionError("Hub dry-run sizes differ from exact manifest")
        print(
            json.dumps(
                {
                    "status": "PASS_REMOTE_DRY_RUN_EXACT_63_SHARDS",
                    "shards": len(remote),
                    "expected_bytes": expected_bytes,
                    "will_download_bytes": sum(
                        int(row["will_download_bytes"]) for row in remote
                    ),
                    "target_values_decoded": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    snapshot = Path(str(result)).resolve()
    if snapshot != pinned_snapshot_path().resolve():
        raise AssertionError(f"download resolved to unexpected snapshot: {snapshot}")
    report = validate_pinned_full_snapshot(hash_payloads=False)
    report["status"] = "PASS_DOWNLOAD_EXACT_63_SHARDS_POINTER_SIZE_AND_SCHEMA"
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
