#!/usr/bin/env python3
"""Rebuild the two designated estimators from the archived readout export inputs.

Runs only the release copy of the exporter. Estimator sources are parsed, never
imported. Every output goes to a new directory; archival inputs remain untouched.
"""
from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import datetime as dt
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import sys
import zlib

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
INPUT_SHA256 = {
    "release/common/readout/export_blocks_v1.py": "33870417dc137ddb42eee803c8def6f7cec034000da35b2bdfead48bcb0b3285",
    "release/common/readout/full90_BLOCKS_COEFFICIENTS_v1.json": "77858e6389345c1a400e82395089d899ded56be898ff85dd2fb4d87655a2d8b7",
    "release/common/parents/parent_release_f9381d09_first128_paid_nocensus_estimator.py": "f9381d0982f5d24e76d38884188ba7c998548e4a3f306825e7e80fb9fa0fc3b6",
    "release/common/parents/parent_science_49bd2ba3_candidate_first128_paid_nocensus_science_v1.py": "49bd2ba31f0bce983f44cc001746e63ffaa84e530d3b31efa6424f69a0298c80",
    "release/common/tails/sm40_cap4_size8e9_xsg_off.py": "ed3ef6e2d5446f5acb56a3a1ded656ecf23f9c334217cf487921421d90162b3a",
    "release/337619/build/BUILD_v1.json": "c42760f325d01d5b050be1341921f28b6f247efbcae2a35e79b429bb263e2027",
    "release/337641/build/BUILD_v1.json": "a1ad39416353873c945571ab9b835538ef992c7b316320cdf7b09fab9e3ae78f",
}
VARIANTS = {
    "337619": {"strength": "1.5", "sha256": "f66cb28baf0c018f25bb749d14ce9e9aedbb47b8f2acd408680cad169485d557"},
    "337641": {"strength": "1.35", "sha256": "5a4c15405c4462703aeead1bb6ccbc9b98ccf6ba2b5974ce23713303c87d594d"},
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def final_literal(text):
    values = [node.value for node in ast.parse(text).body
              if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "_GNN_LITERAL"
                      for target in node.targets)]
    require(bool(values), "No embedded learned tensors found")
    return ast.literal_eval(values[-1])


def decode(item):
    import numpy as np
    shape, encoded = item
    raw = zlib.decompress(base64.b64decode(encoded))
    values = np.frombuffer(raw, dtype="<f4").reshape(shape)
    require(bool(np.isfinite(values).all()), "Non-finite embedded tensor")
    return raw, values


def retire_governor(text):
    """The exact archived #337641 dead-code edit; no other functions change."""
    nodes = [node for node in ast.parse(text).body
             if isinstance(node, ast.FunctionDef)
             and node.name == "_fast_rescue_live_residual"]
    require(len(nodes) == 3, "Expected three retired helper definitions")
    lines = text.splitlines(keepends=True)
    replacement = ("    return None   # retired residual governor removed 2026-10-21: "
                   "never called (_FAST_RESCUE_FRAC = {} is the last line); "
                   "no meter access anywhere\n")
    for node in reversed(nodes):
        lines[node.body[0].lineno - 1:node.end_lineno] = [replacement]
    text = "".join(lines)
    old = ("# accessor (get_active_budget) first, then the remote client's _active_context. "
           "An unavailable or invalid meter now means NO rescue")
    new = ("# (comment removed 2026-10-21: described the retired residual governor, "
           "which no longer exists)")
    require(text.count(old) == 1, "Expected the archived governor comment")
    return text.replace(old, new)


def verify_tensors(parent, rebuilt, archived, coefficients, strength):
    """Check the decoded payload and independently reapply the coefficient matrix."""
    import numpy as np
    require(set(rebuilt) == set(archived), "Embedded tensor names differ")
    require(coefficients["design"] == ["hg", "fh"], "Unexpected readout design")
    require(coefficients["block_order"] == ["hg", "fh"], "Unexpected block order")
    coef = np.asarray(coefficients["coef_by_layer"], dtype="f8") * float(strength)
    require(coef.shape == (15, 961), "Unexpected coefficient shape")
    base = {"head": (0, 64), "head_b": (64, 65), "hidden_head": (65, 193),
            "first_hidden_head": (193, 321)}
    added = {"sq_hidden_head": (321, 449), "sq_first_head": (449, 577),
             "gate_first_head": (577, 705), "gate_hidden_head": (705, 833),
             "first_hidden_prod_head": (833, 961)}
    proofs = {}
    for name, item in rebuilt.items():
        raw, values = decode(item)
        archive_raw, _ = decode(archived[name])
        require(raw == archive_raw, f"Archived tensor bytes differ: {name}")
        if name in base:
            _, prior = decode(parent[name])
            lo, hi = base[name]
            expected = prior.copy()
            expected[1:] = (prior[1:].astype("f8") + coef[:, lo:hi].reshape(prior[1:].shape)).astype("f4")
            require(bool(np.array_equal(values, expected)), f"Readout coefficient mismatch: {name}")
            require(bool(np.array_equal(values[0], prior[0])), f"Row zero changed: {name}")
            role = "parent + scaled coefficients, rows 1..15"
        elif name in added:
            lo, hi = added[name]
            expected = np.zeros((16, 128), dtype="<f4")
            expected[1:] = coef[:, lo:hi].astype("f4")
            require(bool(np.array_equal(values, expected)), f"New coefficient mismatch: {name}")
            role = "scaled coefficients, row zero = 0"
        else:
            require(raw == decode(parent[name])[0], f"Unchanged parent tensor differs: {name}")
            role = "unchanged parent tensor"
        proofs[name] = {"shape": list(values.shape), "dtype": "<f4", "sha256": sha256(raw),
                        "matches_archive": True, "construction": role}
    return proofs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "_build/reproduced",
                        help="new output directory (default: _build/reproduced)")
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), f"Output already exists; choose a new directory: {output}")
    import numpy as np
    for relative, expected in INPUT_SHA256.items():
        require(sha256((ROOT / relative).read_bytes()) == expected, f"Input hash mismatch: {relative}")
    for sid, variant in VARIANTS.items():
        archived = ROOT / "estimator" / f"submission_{sid}.py"
        require(sha256(archived.read_bytes()) == variant["sha256"], f"Archived source hash mismatch: {sid}")
    exporter_path = ROOT / "release/common/readout/export_blocks_v1.py"
    spec = importlib.util.spec_from_file_location("archived_readout_exporter", exporter_path)
    require(spec is not None and spec.loader is not None, "Cannot load archived exporter")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    exporter.PARENTS = {
        "science": ROOT / "release/common/parents/parent_science_49bd2ba3_candidate_first128_paid_nocensus_science_v1.py",
        "release": ROOT / "release/common/parents/parent_release_f9381d09_first128_paid_nocensus_estimator.py",
    }
    # Only the export record's timestamp uses this helper, not the generated source.
    exporter.utc = lambda: dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    coefficient_path = ROOT / "release/common/readout/full90_BLOCKS_COEFFICIENTS_v1.json"
    coefficients = json.loads(coefficient_path.read_text())
    parent = final_literal(exporter.PARENTS["release"].read_text())
    tail = (ROOT / "release/common/tails/sm40_cap4_size8e9_xsg_off.py").read_text()
    output.mkdir(parents=True)
    report = {"passed": False, "python": platform.python_version(), "numpy": np.__version__,
              "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION, "inputs": INPUT_SHA256, "variants": {}}
    try:
        for sid, variant in VARIANTS.items():
            target_dir = output / sid
            target_dir.mkdir()
            exported_dir = target_dir / "readout_export"
            previous_argv = sys.argv
            try:
                sys.argv = [str(exporter_path), str(coefficient_path), str(exported_dir), variant["strength"]]
                with contextlib.redirect_stdout(io.StringIO()):
                    exporter.main()
            finally:
                sys.argv = previous_argv
            sealed = (exported_dir / "candidate_first128_onpolicy_blocks_release_v1.py").read_text()
            build = json.loads((ROOT / f"release/{sid}/build/BUILD_v1.json").read_text())
            require(build["tail"] == tail, f"Build tail mismatch: {sid}")
            require(sealed.endswith("_FAST_RESCUE_FRAC = {}\n"), "Unexpected export suffix")
            assembled = (sealed.removesuffix("_FAST_RESCUE_FRAC = {}\n")
                         + "\n# fable 2026-10-21 stack " + build["name"] + "\n"
                         + tail + "_FAST_RESCUE_FRAC = {}\n")
            before_strip_hash = sha256(assembled.encode("utf-8"))
            require(before_strip_hash == build["files"]["release"]["sha256"],
                    f"Assembly hash mismatch: {sid}; check Python/NumPy/zlib versions")
            if sid == "337641":
                assembled = retire_governor(assembled)
            source_bytes = assembled.encode("utf-8")
            source_hash = sha256(source_bytes)
            archived_bytes = (ROOT / f"estimator/submission_{sid}.py").read_bytes()
            tensors = verify_tensors(parent, final_literal(assembled),
                                     final_literal(archived_bytes.decode("utf-8")), coefficients,
                                     variant["strength"])
            record = {"path": f"{sid}/estimator.py", "sha256": source_hash,
                      "expected_sha256": variant["sha256"], "bytes": len(source_bytes),
                      "strength": float(variant["strength"]), "before_governor_strip_sha256": before_strip_hash,
                      "source_matches_archive": source_bytes == archived_bytes, "tensors": tensors}
            report["variants"][sid] = record
            require(source_hash == variant["sha256"] and source_bytes == archived_bytes,
                    f"Final source mismatch: {sid}; decoded tensor checks do not imply source-byte identity")
            (target_dir / "estimator.py").write_bytes(source_bytes)
            print(f"{sid}: exact source and all {len(tensors)} decoded tensors verified ({source_hash})", flush=True)
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verification: {output / 'verification.json'}")


if __name__ == "__main__":
    main()
