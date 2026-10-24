"""Exact, target-free runtime and immutable-source attestation for V7."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys


HERE = Path(__file__).resolve().parent
P2 = HERE.parents[2]
REPO = HERE.parents[6]
MANIFEST_PATH = HERE / "SOURCE_MANIFEST.json"

EXPECTED_SYS_EXECUTABLE = str(P2 / "data_gate/.venv/bin/python")
EXPECTED_REALPATH = "/Users/matt/anaconda3/bin/python3.11"
EXPECTED_INTERPRETER_SHA256 = (
    "6116d7bcb852d61f4384daa1c43488771e6794636d959f2a7065b54c92a070a1"
)
EXPECTED_PYVENV_SHA256 = (
    "7513b2eb1c0531598377bcf1677fda972031bf33a852d00863364a4bc33b68bc"
)
EXPECTED_PACKAGES = {
    "numpy": "2.4.6",
    "pyarrow": "25.0.1",
    "flopscope": "0.12.0",
    "whestbench": "0.16.0",
}
EXPECTED_RUNTIME_HASHES = {
    "numpy_init": "92a46f791e453926d3292af2582b89995a289475f0eaaea71a949823200b838a",
    "numpy_multiarray": "2a5f2e415c5e582109b015eafd0da1ce887b3e3006969369fe2e5e5b27720acd",
    "pyarrow_init": "04ef85e330eb3cda057b5f4f8e9c06891536bf06d7b26341d392153ecaa861e2",
    "pyarrow_runtime": "dcfdfed0ae94edd01f0eef7a3dcd79751d276a61f3cb1688972e658a3a631679",
    "flopscope_init": "cd3d8b325339717390cee951c5cbeee58c9521aa0e8d642ae62f63c39a2ccdcb",
    "whestbench_init": "eab9716010f7314f5d088430456061b941d377994f9d66e6478df415cff05914",
}
THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def attest_minimal_interpreter() -> dict[str, str]:
    """Standard-library-only identity check safe before invocation consumption."""
    if sys.executable != EXPECTED_SYS_EXECUTABLE:
        raise RuntimeError("wrong V7 interpreter path")
    if os.path.realpath(sys.executable) != EXPECTED_REALPATH:
        raise RuntimeError("wrong V7 interpreter realpath")
    if sha256_file(Path(sys.executable).resolve()) != EXPECTED_INTERPRETER_SHA256:
        raise RuntimeError("V7 interpreter bytes changed")
    return {
        "sys_executable": sys.executable,
        "realpath": EXPECTED_REALPATH,
        "sha256": EXPECTED_INTERPRETER_SHA256,
    }


def load_and_verify_source_manifest(
    expected_manifest_sha256: str | None = None,
) -> dict[str, object]:
    if not MANIFEST_PATH.is_file():
        raise RuntimeError("SOURCE_MANIFEST.json is absent")
    observed_manifest_sha = sha256_file(MANIFEST_PATH)
    if expected_manifest_sha256 is not None and observed_manifest_sha != expected_manifest_sha256:
        raise RuntimeError("source manifest hash differs from release")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("schema")
        != "phase2-layer3-attribution-v7-authoritative-source-manifest-v1"
        or manifest.get("status") != "FROZEN_V7_RUNNER_SOURCES_NO_EXECUTION_AUTHORITY"
        or manifest.get("execution_authority") is not False
    ):
        raise RuntimeError("inadmissible V7 source manifest")
    for section in ("local_source_sha256", "frozen_dependency_sha256"):
        bindings = manifest.get(section)
        if not isinstance(bindings, dict) or not bindings:
            raise RuntimeError(f"missing manifest section: {section}")
        for relative, expected in bindings.items():
            path = (REPO / relative).resolve()
            try:
                path.relative_to(REPO.resolve())
            except ValueError as error:
                raise RuntimeError("manifest path escapes repository") from error
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"manifest binding drift: {relative}")
    return {
        "manifest_sha256": observed_manifest_sha,
        "manifest": manifest,
    }


def attest_exact_environment(*, require_thread_contract: bool = True) -> dict[str, object]:
    """Attest the known-working V5 interpreter and all imported runtimes."""
    minimal = attest_minimal_interpreter()
    pyvenv = P2 / "data_gate/.venv/pyvenv.cfg"
    if sha256_file(pyvenv) != EXPECTED_PYVENV_SHA256:
        raise RuntimeError("pyvenv.cfg changed")
    if platform.python_version() != "3.11.5":
        raise RuntimeError("Python version changed")
    if require_thread_contract:
        bad = {name: os.environ.get(name) for name in THREAD_VARIABLES
               if os.environ.get(name) != "4"}
        if bad:
            raise RuntimeError(f"four-thread environment not bound: {bad}")
    packages = {name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES}
    if packages != EXPECTED_PACKAGES:
        raise RuntimeError("package version drift")
    if importlib.util.find_spec("scipy") is not None:
        raise RuntimeError("SciPy is forbidden in V7")
    if importlib.util.find_spec("torch") is not None:
        raise RuntimeError("Torch is forbidden in V7")

    import flopscope
    import numpy as np
    import pyarrow as pa
    import whestbench

    observed_hashes = {
        "numpy_init": sha256_file(Path(np.__file__).resolve()),
        "numpy_multiarray": sha256_file(Path(np._core._multiarray_umath.__file__).resolve()),
        "pyarrow_init": sha256_file(Path(pa.__file__).resolve()),
        "pyarrow_runtime": sha256_file(Path(pa.lib.__file__).resolve()),
        "flopscope_init": sha256_file(Path(flopscope.__file__).resolve()),
        "whestbench_init": sha256_file(Path(whestbench.__file__).resolve()),
    }
    if observed_hashes != EXPECTED_RUNTIME_HASHES:
        raise RuntimeError("runtime module bytes changed")
    return {
        "status": "PASS_V7_EXACT_KNOWN_WORKING_ENVIRONMENT",
        "interpreter": minimal,
        "python": platform.python_version(),
        "pyvenv_cfg_sha256": EXPECTED_PYVENV_SHA256,
        "packages": packages,
        "runtime_module_sha256": observed_hashes,
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "scipy_present": False,
        "torch_present": False,
        "physical_dataset_iterator_calls": 0,
        "physical_dataset_rows_decoded": 0,
        "target_values_decoded": 0,
    }
