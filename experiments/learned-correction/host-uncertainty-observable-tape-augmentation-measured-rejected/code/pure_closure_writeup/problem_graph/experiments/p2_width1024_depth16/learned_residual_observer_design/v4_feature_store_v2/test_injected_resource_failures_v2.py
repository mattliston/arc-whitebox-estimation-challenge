#!/usr/bin/env python3
"""Target-free injected proof for all V2 predecode abort paths."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from resource_safety_v2 import (
    GroupRegistry,
    ResourceViolation,
    ResourceWatchdog,
    pid_rss_bytes,
    process_group_pids,
)


HERE = Path(__file__).resolve().parent
PHYSICAL = 48 * (1 << 30)


def _case(label: str, phase: str, injection: str | None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"v4-fs-v2-{label}-", dir="/private/tmp") as temporary:
        root = Path(temporary)
        process = subprocess.Popen(
            [sys.executable, str(HERE / "synthetic_group_tree_v2.py")],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            close_fds=True,
            start_new_session=True,
        )
        assert process.stdout is not None
        boot = json.loads(process.stdout.readline())
        registry = GroupRegistry()
        record = registry.register(process, f"synthetic-{label}")
        watchdog = ResourceWatchdog(
            registry,
            PHYSICAL,
            root / "rss.jsonl",
            synthetic_mode=True,
            injected_guards={phase: injection} if injection else {},
        )
        decoded_rows = 0
        decode_callable_invocations = 0

        def decode_one_synthetic_row() -> None:
            nonlocal decoded_rows, decode_callable_invocations
            decode_callable_invocations += 1
            decoded_rows += 1

        caught = None
        watchdog.start()
        try:
            if label == "exception":
                watchdog.guard("pre_iterator")
                raise RuntimeError("synthetic caller exception before iterator/decode")
            watchdog.guard(phase)
            decode_one_synthetic_row()  # unreachable on both injected boundaries
        except BaseException as error:
            caught = f"{type(error).__name__}: {error}"
            watchdog.abort_all(caught)
        report = watchdog.join_after_abort()
        descendant_pid = int(boot["descendant_pid"])
        leader_pid = int(boot["leader_pid"])
        if decoded_rows != 0 or decode_callable_invocations != 0:
            raise AssertionError("synthetic abort allowed a decode")
        if process_group_pids(record.pgid):
            raise AssertionError("synthetic process group is not quiescent")
        if pid_rss_bytes(leader_pid) != 0 or pid_rss_bytes(descendant_pid) != 0:
            raise AssertionError("synthetic leader or descendant remains live")
        if not report["termination"]["all_groups_quiescent"]:
            raise AssertionError("termination receipt lacks group quiescence")
        if record.pgid not in report["termination"]["sigkill_groups"]:
            raise AssertionError("stubborn descendant did not exercise group SIGKILL escalation")
        return {
            "label": label,
            "phase": phase,
            "injection": injection or "caller_exception",
            "caught": caught,
            "decoded_rows": decoded_rows,
            "decode_callable_invocations": decode_callable_invocations,
            "leader_pid": leader_pid,
            "descendant_pid": descendant_pid,
            "leader_dead": True,
            "descendant_dead": True,
            "process_group_quiescent": True,
            "term_sent_to_group": record.pgid in report["termination"]["sigterm_groups"],
            "kill_sent_to_group_after_grace": record.pgid in report["termination"]["sigkill_groups"],
            "watchdog_samples": report["samples"],
            "maximum_sample_gap_seconds": report["maximum_sample_gap_seconds"],
        }


def verify() -> dict[str, Any]:
    cases = [
        _case("system_25_percent", "pre_iterator", "system_25_percent"),
        _case("group_8gib", "pre_decode", "group_8gib"),
        _case("exception", "pre_exception", None),
    ]
    if any(case["decoded_rows"] != 0 for case in cases):
        raise AssertionError("an injected failure decoded a row")
    if not all(case["descendant_dead"] and case["process_group_quiescent"] for case in cases):
        raise AssertionError("an injected failure leaked a descendant")
    return {
        "schema": "phase2-observer-v4-feature-store-v2-injected-resource-test-v1",
        "status": "PASS_V2_INJECTED_RESOURCE_ABORTS_ZERO_DECODE_AND_GROUP_QUIESCENCE",
        "cases": cases,
        "real_full_rows_decoded": 0,
        "role_target_artifacts_opened_or_hashed": 0,
        "target_values_decoded": 0,
        "claims_created": 0,
        "releases_created": 0,
        "fits": 0,
        "scores": 0,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
