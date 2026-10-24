#!/usr/bin/env python3
"""Bounded target-free generator for the 488-row all-688 complement."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
GENERATOR = HERE / "generate_all688_row.py"
PREREGISTRATION = HERE / "GENERATION_PREREGISTRATION.json"
RUN_MANIFEST = HERE / "GENERATION_RUN_MANIFEST.json"
SEAL = HERE / "GENERATION_SEAL.json"
OUTPUT_ROOT = HERE / "runs/native_rows_remaining_v1"
WORKERS = 3
EXPECTED_RANKS = tuple(range(35)) + tuple(range(235, 688))
MAX_STORAGE_BYTES = 1 << 30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_seal() -> None:
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    expected = {
        "preregistration_sha256": sha256(PREREGISTRATION),
        "run_manifest_sha256": sha256(RUN_MANIFEST),
        "program_sha256": sha256(GENERATOR),
        "orchestrator_sha256": sha256(Path(__file__)),
    }
    for key, observed in expected.items():
        if seal.get(key) != observed:
            raise AssertionError(f"final generation seal changed: {key}: {observed}")


def complete(rank: int) -> bool:
    row = OUTPUT_ROOT / f"row_{rank:04d}"
    return all(
        (row / name).is_file()
        for name in ("host.npy", "raw20.npy", "sigma.npy", "result.json")
    )


def run_one(rank: int) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(GENERATOR),
            "--rank",
            str(rank),
            "--output-root",
            str(OUTPUT_ROOT),
        ],
        cwd=HERE,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"rank {rank} failed rc={completed.returncode}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"rank {rank} produced no receipt line")
    return json.loads(lines[-1])


def main() -> None:
    verify_seal()
    started = time.perf_counter()
    already = [rank for rank in EXPECTED_RANKS if complete(rank)]
    already_set = set(already)
    pending = [rank for rank in EXPECTED_RANKS if rank not in already_set]
    done = len(already)
    print(
        json.dumps({"status": "start", "complete": done, "pending": len(pending)}),
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_one, rank): rank for rank in pending}
        for future in as_completed(futures):
            record = future.result()
            done += 1
            print(json.dumps({"complete": done, **record}, sort_keys=True), flush=True)
            if done % 100 == 0 or done == len(EXPECTED_RANKS):
                print(
                    json.dumps(
                        {
                            "status": "progress",
                            "complete": done,
                            "total": len(EXPECTED_RANKS),
                            "elapsed_seconds": time.perf_counter() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    missing = [rank for rank in EXPECTED_RANKS if not complete(rank)]
    if missing:
        raise AssertionError(f"final native generation incomplete: {missing}")
    storage = sum(path.stat().st_size for path in OUTPUT_ROOT.rglob("*") if path.is_file())
    if storage >= MAX_STORAGE_BYTES:
        raise OSError(f"final native store exceeds frozen bound: {storage}")
    print(
        json.dumps(
            {
                "status": "complete",
                "rows": len(EXPECTED_RANKS),
                "storage_bytes": storage,
                "elapsed_seconds": time.perf_counter() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
