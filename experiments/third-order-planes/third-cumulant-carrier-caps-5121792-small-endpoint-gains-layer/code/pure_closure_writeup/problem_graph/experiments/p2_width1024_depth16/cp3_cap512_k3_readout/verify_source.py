#!/usr/bin/env python3
"""Dataset-free algebra and source-surface verification for CP3 cap 512."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "estimator.py"


def _dense_cp(lam, p, q):
    return (
        np.einsum("r,ir,jr,kr->ijk", lam, p, p, q)
        + np.einsum("r,ir,jr,kr->ijk", lam, p, q, p)
        + np.einsum("r,ir,jr,kr->ijk", lam, q, p, p)
    )


def verify() -> dict[str, object]:
    rng = np.random.default_rng(20260823)
    width, rank = 7, 5
    lam = rng.normal(size=rank)
    p = rng.normal(size=(width, rank))
    q = rng.normal(size=(width, rank))
    weight = rng.normal(size=(width, width))

    dense = _dense_cp(lam, p, q)
    diagonal = dense[np.arange(width), np.arange(width), np.arange(width)]
    factor_diagonal = 3.0 * ((p * p * q) @ lam)

    pushed_dense = np.einsum(
        "ia,jb,kc,ijk->abc",
        weight,
        weight,
        weight,
        dense,
    )
    pushed_p = weight.T @ p
    pushed_q = weight.T @ q
    factor_pushed_dense = _dense_cp(lam, pushed_p, pushed_q)

    component_energy = []
    formula_energy = []
    for component in range(rank):
        one = _dense_cp(
            lam[component : component + 1],
            p[:, component : component + 1],
            q[:, component : component + 1],
        )
        component_energy.append(float(np.sum(one * one)))
        p2 = float(p[:, component] @ p[:, component])
        q2 = float(q[:, component] @ q[:, component])
        pq = float(p[:, component] @ q[:, component])
        formula_energy.append(
            float(lam[component] ** 2 * (3.0 * p2 * p2 * q2 + 6.0 * p2 * pq * pq))
        )

    source_text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    imports = sorted(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    forbidden_import_prefixes = (
        "numpy",
        "scipy",
        "torch",
        "sklearn",
        "autoresearch",
    )
    forbidden_imports = [
        name for name in imports if name.startswith(forbidden_import_prefixes)
    ]
    forbidden_tokens = [
        token
        for token in (
            "all_layer_means",
            "final_means",
            "target",
            "teacher",
            "checkpoint",
            "K31",
            "K22",
            "cholesky",
            "eigh",
        )
        if re.search(rf"\b{re.escape(token)}\b", source_text)
    ]

    result = {
        "schema": "phase2-cp3-cap512-source-verification-v1",
        "estimator_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "checks": {
            "cp_diagonal_max_abs_error": float(
                np.max(np.abs(diagonal - factor_diagonal))
            ),
            "cp_affine_push_max_abs_error": float(
                np.max(np.abs(pushed_dense - factor_pushed_dense))
            ),
            "cp_component_energy_max_abs_error": float(
                np.max(np.abs(np.asarray(component_energy) - np.asarray(formula_energy)))
            ),
            "forbidden_imports": forbidden_imports,
            "forbidden_tokens": forbidden_tokens,
        },
    }
    tolerance = 2.0e-10
    if (
        result["checks"]["cp_diagonal_max_abs_error"] > tolerance
        or result["checks"]["cp_affine_push_max_abs_error"] > tolerance
        or result["checks"]["cp_component_energy_max_abs_error"] > tolerance
        or forbidden_imports
        or forbidden_tokens
    ):
        result["status"] = "FAIL"
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    result["status"] = "PASS_TARGET_BLIND_SOURCE_ALGEBRA"
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
