#!/usr/bin/env python3
"""One-use launcher and whole-PGID supervisor for the frozen V7 worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from runtime_contract import (
    attest_minimal_interpreter,
    canonical_json_bytes,
    load_and_verify_source_manifest,
    sha256_file,
)


HERE = Path(__file__).resolve().parent
RELEASE_PATH = HERE / "ROOT_RELEASE.json"
RECEIPTS = HERE / "receipts"
RUNS = HERE / "runs"
WORKER = HERE / "worker.py"
AUDIT_PATH = HERE / "INDEPENDENT_LAYER3_ATTRIBUTION_RUNNER_AUDIT_V7.json"
PS_PATH = Path("/bin/ps")
PS_SHA256 = "472992c470606d28f577590decfecd7f4a20f832fd92c671bebc6d44790b5d02"
REQUIRED_RELEASE_STATUS = "ROOT_AUTHORIZED_ONE_LAYER3_ATTRIBUTION_V7_INVOCATION"
REQUIRED_AUDIT_STATUS = "PASS_LAYER3_ATTRIBUTION_V7_RUNNER_AND_ONE_USE_SCAFFOLD"
SCOPE = "target_blind_layer3_attribution_v5_science_only"
RESOURCE_CONTRACT = {
    "rss_limit_bytes_whole_worker_pgid": 2147483648,
    "wall_limit_seconds": 14400,
    "poll_interval_seconds": 0.25,
    "cpu_threads": 4,
    "worker_processes": 1,
    "devices": ["cpu"],
}
ZERO_COUNTERS = {
    "physical_snapshot_calls": 0,
    "physical_parquet_files_opened": 0,
    "physical_dataset_iterator_calls": 0,
    "physical_dataset_rows_decoded": 0,
    "teacher_artifacts_generated": 0,
    "target_values_decoded": 0,
    "target_metrics_computed": 0,
    "fits": 0,
    "scores": 0,
    "variants": 0,
    "sweeps": 0,
    "submissions": 0,
    "graph_edits": 0,
}


def exact_paths(release_id: str) -> dict[str, Path]:
    return {
        "invocation_tombstone_path": RECEIPTS / (
            f"l3-v7-{release_id}.invocation-consumed.json"
        ),
        "data_claim_path": RECEIPTS / f"l3-v7-{release_id}.data-claim.json",
        "staging_path": RUNS / f".l3-v7-{release_id}.staging.json",
        "result_path": RUNS / f"l3-v7-{release_id}.result.json",
    }


def validate_release() -> tuple[dict[str, object], str, dict[str, object]]:
    """No non-standard module or physical loader is imported here."""
    interpreter = attest_minimal_interpreter()
    if not RELEASE_PATH.is_file():
        raise PermissionError("ROOT_RELEASE.json absent; V7 execution blocked")
    release_bytes = RELEASE_PATH.read_bytes()
    release_sha = hashlib.sha256(release_bytes).hexdigest()
    release = json.loads(release_bytes)
    release_id = release.get("release_id")
    if (
        type(release_id) is not str
        or len(release_id) != 64
        or any(char not in "0123456789abcdef" for char in release_id)
    ):
        raise PermissionError("invalid V7 release id")
    manifest_sha = release.get("source_manifest_sha256")
    if type(manifest_sha) is not str or len(manifest_sha) != 64:
        raise PermissionError("release lacks exact source manifest hash")
    manifest = load_and_verify_source_manifest(manifest_sha)
    paths = exact_paths(release_id)
    if (
        release.get("schema") != "phase2-layer3-attribution-v7-root-release-v1"
        or release.get("status") != REQUIRED_RELEASE_STATUS
        or release.get("scope") != SCOPE
        or release.get("future_invocations_authorized") != 1
        or release.get("ordinals") != [0, 1]
        or release.get("resource_contract") != RESOURCE_CONTRACT
        or release.get("zero_access_counters") != ZERO_COUNTERS
        or release.get("exact_interpreter") != interpreter
    ):
        raise PermissionError("V7 root release contract mismatch")
    for label, expected in paths.items():
        if Path(str(release.get(label, ""))).resolve() != expected.resolve():
            raise PermissionError(f"release path mismatch: {label}")
    audit_binding = release.get("independent_runner_audit")
    if not isinstance(audit_binding, dict):
        raise PermissionError("release lacks independent runner audit")
    if Path(str(audit_binding.get("path", ""))).resolve() != AUDIT_PATH.resolve():
        raise PermissionError("independent audit path mismatch")
    if not AUDIT_PATH.is_file() or sha256_file(AUDIT_PATH) != audit_binding.get("sha256"):
        raise PermissionError("independent runner audit hash mismatch")
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if (
        audit.get("status") != REQUIRED_AUDIT_STATUS
        or audit.get("execution_authority") is not False
        or audit.get("source_manifest_sha256") != manifest_sha
    ):
        raise PermissionError("independent runner audit contract mismatch")
    audited_files = audit.get("frozen_sha256")
    if not isinstance(audited_files, dict):
        raise PermissionError("independent audit lacks frozen artifact hashes")
    for name in (
        "SOURCE_MANIFEST.json", "SYNTHETIC_REHEARSAL.json",
        "STATIC_VERIFICATION.json",
    ):
        if audited_files.get(name) != sha256_file(HERE / name):
            raise PermissionError(f"independent audit artifact binding drift: {name}")
    if (
        not PS_PATH.is_file()
        or not os.access(PS_PATH, os.X_OK)
        or sha256_file(PS_PATH) != PS_SHA256
    ):
        raise PermissionError("bound process-table tool unavailable")
    if not RECEIPTS.is_dir() or not RUNS.is_dir():
        raise PermissionError("frozen authority/output parents missing")
    receipt_children = sorted(
        path.name for path in RECEIPTS.iterdir() if path.name != "README.md"
    )
    run_children = sorted(path.name for path in RUNS.iterdir() if path.name != "README.md")
    if receipt_children or run_children:
        raise FileExistsError("V7 authority or output namespace already consumed")
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("release-bound V7 path already exists")
    return release, release_sha, manifest


def durable_exclusive_json(path: Path, value: object) -> str:
    payload = canonical_json_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short exclusive receipt write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return hashlib.sha256(payload).hexdigest()


def consume_invocation_tombstone(
    release: dict[str, object], release_sha: str, manifest: dict[str, object]
) -> tuple[Path, str]:
    path = Path(str(release["invocation_tombstone_path"])).resolve()
    expected = exact_paths(str(release["release_id"]))["invocation_tombstone_path"]
    if path != expected.resolve():
        raise PermissionError("invocation tombstone escaped fixed namespace")
    value = {
        "schema": "phase2-layer3-attribution-v7-invocation-consumed-v1",
        "status": "ONE_INVOCATION_IRREVERSIBLY_CONSUMED_BEFORE_WORKER_CREATION",
        "release_id": release["release_id"],
        "release_sha256": release_sha,
        "source_manifest_sha256": manifest["manifest_sha256"],
        "scope": SCOPE,
        "ordinals": [0, 1],
        "future_invocations_remaining": 0,
        "worker_created_before_tombstone": False,
        "physical_loader_imported_before_tombstone": False,
        "physical_dataset_rows_decoded_before_tombstone": 0,
        "retry_authorized": False,
        "zero_access_counters": dict(ZERO_COUNTERS),
    }
    return path, durable_exclusive_json(path, value)


def parse_ps_pgid_rss_kib(snapshot: str, pgid: int) -> tuple[int, list[int]]:
    total = 0
    pids = []
    for line in snapshot.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            pid_value, pgid_value, rss_kib = map(int, fields)
        except ValueError:
            continue
        if pgid_value == pgid:
            if rss_kib < 0:
                raise RuntimeError("negative process RSS")
            total += rss_kib
            pids.append(pid_value)
    return total, sorted(pids)


def whole_pgid_rss(pgid: int) -> tuple[int, list[int]]:
    snapshot = subprocess.check_output(
        [str(PS_PATH), "-axo", "pid=,pgid=,rss="],
        text=True,
        timeout=5.0,
    )
    kib, pids = parse_ps_pgid_rss_kib(snapshot, pgid)
    return kib * 1024, pids


def terminate_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def supervise_worker(command: list[str], environment: dict[str, str]) -> dict[str, object]:
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        child = subprocess.Popen(
            command,
            cwd=HERE,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            close_fds=True,
        )
        pgid = child.pid
        started = time.monotonic()
        peak = 0
        observed_pids: set[int] = set()
        terminal_reason = None
        while child.poll() is None:
            elapsed = time.monotonic() - started
            try:
                rss, pids = whole_pgid_rss(pgid)
            except Exception:
                terminal_reason = "PROCESS_GROUP_RSS_SUPERVISION_FAILURE"
                terminate_group(pgid)
                break
            peak = max(peak, rss)
            observed_pids.update(pids)
            if rss > RESOURCE_CONTRACT["rss_limit_bytes_whole_worker_pgid"]:
                terminal_reason = "WHOLE_PROCESS_GROUP_RSS_LIMIT_EXCEEDED"
                terminate_group(pgid)
                break
            if elapsed > RESOURCE_CONTRACT["wall_limit_seconds"]:
                terminal_reason = "WORKER_WALL_LIMIT_EXCEEDED"
                terminate_group(pgid)
                break
            time.sleep(RESOURCE_CONTRACT["poll_interval_seconds"])
        return_code = child.wait()
        stdout.seek(0)
        stderr.seek(0)
        out = stdout.read()
        err = stderr.read()
    if terminal_reason is not None:
        raise RuntimeError(
            f"{terminal_reason}; authority consumed; no retry; "
            f"return_code={return_code}; peak_pgid_rss={peak}"
        )
    if return_code != 0:
        raise RuntimeError(
            "V7 worker terminal failure; authority consumed; no retry; "
            f"return_code={return_code}; stderr={err[-4096:].decode('utf-8', 'replace')}"
        )
    return {
        "return_code": return_code,
        "stdout": out,
        "stderr": err,
        "peak_whole_pgid_rss_bytes": peak,
        "observed_worker_pids": sorted(observed_pids),
    }


def execute() -> None:
    release, release_sha, manifest = validate_release()
    tombstone_path, tombstone_sha = consume_invocation_tombstone(
        release, release_sha, manifest
    )
    environment = dict(os.environ)
    for name in (
        "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "4"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        sys.executable,
        "-B",
        str(WORKER),
        "--claimed-release-sha256",
        release_sha,
        "--invocation-tombstone-sha256",
        tombstone_sha,
    ]
    outcome = supervise_worker(command, environment)
    result_path = Path(str(release["result_path"]))
    if not result_path.is_file():
        raise RuntimeError("worker exited zero without the sole atomic result")
    if outcome["stderr"]:
        raise RuntimeError("worker emitted unexpected stderr")
    sys.stdout.buffer.write(outcome["stdout"])
    sys.stdout.buffer.flush()
    if not tombstone_path.is_file():
        raise RuntimeError("invocation tombstone vanished")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("V7 launcher has no implicit mode")
    execute()


if __name__ == "__main__":
    main()
