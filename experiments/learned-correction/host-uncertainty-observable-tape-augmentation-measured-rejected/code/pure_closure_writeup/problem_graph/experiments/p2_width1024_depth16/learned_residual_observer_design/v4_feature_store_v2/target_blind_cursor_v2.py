#!/usr/bin/env python3
"""Command-gated one-row cursor for the frozen target-blind full iterator.

Process startup imports no full reader and decodes zero rows.  `START` creates
the lazy iterator; each `NEXT` performs exactly one decode and then blocks.
The held 64 MiB row is written only after a separate parent `WRITE` command.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
from typing import Any


PROCESS_LIMIT_BYTES = 8 * (1 << 30)
_soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
_new_soft = PROCESS_LIMIT_BYTES if _hard == resource.RLIM_INFINITY else min(PROCESS_LIMIT_BYTES, _hard)
_rlimit_as_installed = _soft <= PROCESS_LIMIT_BYTES
if not _rlimit_as_installed:
    try:
        resource.setrlimit(resource.RLIMIT_AS, (_new_soft, _hard))
        _rlimit_as_installed = True
    except (OSError, ValueError):
        if sys.platform != "darwin":
            raise
# Darwin refuses finite RLIMIT_AS in the exact bound environment.  This cursor
# emits no data and constructs no iterator until its parent has registered its
# isolated group under the strict 8 GiB continuous RSS watchdog.

from contract_v2 import (  # noqa: E402
    DEPTH,
    FROZEN_BINDINGS,
    PER_PROCESS_LIMIT_BYTES,
    SCRATCH_PATH,
    WIDTH,
    canonical_json_bytes,
    fsync_directory,
    raw_array_sha256,
    sha256_file,
)


def _send(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _load_reader() -> Any:
    source = Path(FROZEN_BINDINGS["target_blind_full_reader"]["path"])
    if sha256_file(source) != FROZEN_BINDINGS["target_blind_full_reader"]["sha256"]:
        raise AssertionError("frozen target-blind full reader changed")
    spec = importlib.util.spec_from_file_location("v4_feature_store_v2_full_cursor_reader", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen target-blind full reader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _safe_weights_path(text: str) -> Path:
    path = Path(text).resolve()
    root = SCRATCH_PATH.resolve()
    if root not in path.parents or path.name != "weights.npy":
        raise PermissionError("cursor write path escapes exact V2 scratch root")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise PermissionError("cursor partial directory is absent or a symlink")
    if any(path.parent.iterdir()):
        raise PermissionError("cursor partial directory is not empty before weight write")
    return path


def _write_weights(row: Any, path: Path, expected_hash: str) -> dict[str, Any]:
    import numpy as np  # reader already imported after START; never at process boot

    weights = np.asarray(row.weights)
    if weights.dtype != np.float32 or weights.shape != (DEPTH, WIDTH, WIDTH):
        raise AssertionError("cursor row weight schema changed")
    digest = raw_array_sha256(weights)
    if digest != expected_hash:
        raise AssertionError("cursor row differs from frozen roster weight attestation")
    with path.open("xb") as handle:
        np.save(handle, np.ascontiguousarray(weights, dtype=np.dtype("<f4")), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    fsync_directory(path.parent)
    return {"weights_raw_sha256": digest, "weights_file_sha256": sha256_file(path), "bytes": path.stat().st_size}


def run() -> None:
    if _new_soft > PER_PROCESS_LIMIT_BYTES:
        raise AssertionError("cursor process guard changed")
    reader = None
    iterator = None
    current = None
    decoded = 0
    exhausted = False
    _send({
        "event": "BOOTED_ZERO_DECODE_CURSOR",
        "decoded_rows": 0,
        "reader_imported": False,
        "iterator_constructed": False,
        "process_limit_bytes": PER_PROCESS_LIMIT_BYTES,
        "rlimit_as_installed": _rlimit_as_installed,
        "strict_group_rss_watchdog_required": True,
    })
    for line in sys.stdin:
        command = json.loads(line)
        operation = command.get("op")
        if operation == "START":
            if reader is not None or iterator is not None or current is not None or decoded != 0:
                raise PermissionError("cursor START is one-use")
            reader = _load_reader()
            iterator = iter(reader.iter_target_blind_full())
            _send({
                "event": "ITERATOR_READY_ZERO_DECODE",
                "decoded_rows": decoded,
                "reader_imported": True,
                "iterator_constructed": True,
            })
        elif operation == "NEXT":
            if iterator is None or current is not None or exhausted:
                raise PermissionError("cursor NEXT state is invalid")
            try:
                current = next(iterator)
            except StopIteration:
                exhausted = True
                _send({"event": "END", "decoded_rows": decoded})
                continue
            decoded += 1
            _send({
                "event": "ROW_HELD",
                "decoded_rows": decoded,
                "ordinal": int(current.ordinal),
                "shard_index": int(current.shard_index),
                "row_in_shard": int(current.row_in_shard),
                "mlp_id": int(current.mlp_id),
                "mlp_name": current.mlp_name,
                "mlp_seed": int(current.mlp_seed),
                "decoded_columns": list(current.decoded_columns),
            })
        elif operation == "DISCARD":
            if current is None:
                raise PermissionError("cursor DISCARD has no held row")
            current = None
            gc.collect()
            _send({"event": "DISCARDED", "decoded_rows": decoded})
        elif operation == "WRITE":
            if current is None:
                raise PermissionError("cursor WRITE has no held row")
            for key in ("mlp_id", "mlp_seed", "ordinal"):
                if int(command[key]) != int(getattr(current, key)):
                    raise PermissionError(f"cursor WRITE row binding changed: {key}")
            path = _safe_weights_path(command["weights_path"])
            receipt = _write_weights(current, path, command["weights_raw_sha256"])
            current = None
            gc.collect()
            _send({"event": "WRITTEN", "decoded_rows": decoded, **receipt})
        elif operation == "STOP":
            if current is not None:
                raise PermissionError("cursor cannot STOP while holding a row")
            _send({"event": "STOPPED", "decoded_rows": decoded, "exhausted": exhausted})
            return
        else:
            raise PermissionError("unknown cursor operation")
    raise RuntimeError("cursor command stream closed without STOP")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
