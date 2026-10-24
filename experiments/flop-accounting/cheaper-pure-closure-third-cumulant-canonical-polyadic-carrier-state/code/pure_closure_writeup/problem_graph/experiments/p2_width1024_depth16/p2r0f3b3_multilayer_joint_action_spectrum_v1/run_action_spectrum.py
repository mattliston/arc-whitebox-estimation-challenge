#!/usr/bin/env python3
"""Target-free actual-network immediate-action rank discriminator.

This program reads only sealed observer_fit weights and target-blind host
features.  It never imports an estimator, data loader, fixture, target, score,
observer_selection artifact, or submission code.

For a post-ReLU source h_l and the immediately scored next mean m_(l+1), the
linear cumulant action in output coordinate a is a scalar response coefficient
times a tensor power of column a of W_l.  The row Gram of this operator can be
formed exactly from power Grams of W_l.  Its top-r eigenspace is the optimal
unconstrained r-query endpoint representation.  It is therefore an upper
bound on every common hidden subspace and every r-term factorized action-query
list.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
import scipy.linalg


HERE = Path(__file__).resolve().parent
ROOT = next(parent for parent in HERE.parents if (parent / "pure_closure_writeup").is_dir())
PREREG = HERE / "PREREGISTRATION.json"
RESULT = HERE / "RESULT.json"
BASIS = HERE / "OPTIMAL_ENDPOINT_BASIS_R13.npy"

WIDTH = 1024
DEPTH = 16
SOURCE_LAYERS_ONE_BASED = (2, 3, 4, 5, 6, 7)
RANKS = (8, 13)
SPECTRUM_PREFIX = 64
COMPLETENESS_GATE = 0.99
OBJECTIVE_RELEVANCE_GATE = 0.70
KERDOCK_PATHS = 65_536
PHASE2_BUDGET = 2**41
D1_WORST_BILL = 1_612_602_817_032

STORE = (
    ROOT
    / "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16"
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3/rows/observer_fit"
)

PANEL = {
    "0000": {
        "row_manifest.json": "28e7c706c092fca59b419d88aff5893cb72cb17e5f3a2858341641de8222a6aa",
        "weights.npy": "399488100ef907d6a76fa27e911285809bf5d53405bc59290c84396a7e4c8c97",
        "cp1792_raw_locals.npy": "e7745da464dd37f75c8e33ff864002004b25bf7683a57790610605828bd6d497",
        "cp1792_sigma.npy": "3f1097d8e41cb18ba0c48cc86be567172f6128e938b52a673a72efc424f3ad35",
    },
    "0343": {
        "row_manifest.json": "9a0c78f12d4c6fcbfb8a82b8657e171bb86579e580563c1159937d0be2a34e46",
        "weights.npy": "d27e82ad20dab3bc971a1f0740735008df82532c7af4462bf1f2957f4bfada20",
        "cp1792_raw_locals.npy": "c8bd5a02e1cbe373af0c55f745151bb4efe89dac97bfb0c8ed3504974839283b",
        "cp1792_sigma.npy": "3d8ffc65ee9dd288eab448a47291ac959785771dab52e6bb2edd2ecf497507a9",
    },
    "0687": {
        "row_manifest.json": "056548b93191a4e3e0b213bb2d65938e95fbea1f65d79c0d04bb2e79303e19bc",
        "weights.npy": "c30185f0d882efb4be47ad842f8969d99f5123cccd6deed383ec9ef70045b5ca",
        "cp1792_raw_locals.npy": "1a5a494a5451222f53e09fbde9856c0184036d707025adf990a50e49914fa0f4",
        "cp1792_sigma.npy": "ca283baeef8125fb987bf274c4fef05085646ecea936e9770c211d17c3295598",
    },
}

# Every coordinate-collision sector through q=5 is reported.  The existing
# host already represents some marginal/bivariate sectors, but retaining them
# makes marginal-versus-joint diffusion visible.  The mixed grade-2/3 shells
# are tested in the more favorable unrestricted full tensor space.
BLOCKS = (
    ("q2_[2]", 2, "marginal", "partition", (2,)),
    ("q2_[11]", 2, "joint", "partition", (1, 1)),
    ("q3_[3]", 3, "marginal", "partition", (3,)),
    ("q3_[21]", 3, "joint", "partition", (2, 1)),
    ("q3_[111]", 3, "joint", "partition", (1, 1, 1)),
    ("q4_[4]", 4, "marginal", "partition", (4,)),
    ("q4_[31]", 4, "joint", "partition", (3, 1)),
    ("q4_[22]", 4, "joint", "partition", (2, 2)),
    ("q4_[211]", 4, "joint", "partition", (2, 1, 1)),
    ("q4_[1111]", 4, "joint", "partition", (1, 1, 1, 1)),
    ("q5_[5]", 5, "marginal", "partition", (5,)),
    ("q5_[41]", 5, "joint", "partition", (4, 1)),
    ("q5_[32]", 5, "joint", "partition", (3, 2)),
    ("q5_[311]", 5, "joint", "partition", (3, 1, 1)),
    ("q5_[221]", 5, "joint", "partition", (2, 2, 1)),
    ("q5_[2111]", 5, "joint", "partition", (2, 1, 1, 1)),
    ("q5_[11111]", 5, "joint", "partition", (1, 1, 1, 1, 1)),
    ("q6_k3_squared", 6, "mixed", "full", None),
    ("q7_k3_k4", 7, "mixed", "full", None),
    ("q9_k3_cubed", 9, "mixed", "full", None),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def hermite_probabilists(order: int, x: np.ndarray) -> np.ndarray:
    if order == 0:
        return np.ones_like(x)
    hm1 = np.ones_like(x)
    h0 = x.copy()
    for degree in range(1, order):
        hp1 = x * h0 - degree * hm1
        hm1, h0 = h0, hp1
    return h0


def response_coefficient(order: int, alpha: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Immediate ReLU mean response in the frozen Hermite convention."""
    phi = np.exp(-0.5 * alpha * alpha) / math.sqrt(2.0 * math.pi)
    if order == 2:
        denominator = 2.0
    elif order == 3:
        denominator = 6.0
    elif order == 4:
        denominator = 24.0
    elif order == 5:
        denominator = 120.0
    elif order == 6:
        denominator = 72.0
    elif order == 7:
        denominator = 144.0
    elif order == 9:
        denominator = 1296.0
    else:
        raise ValueError(order)
    return (
        phi
        * hermite_probabilists(order - 2, -alpha)
        / (denominator * sigma ** (order - 1))
    )


def partition_kernel(partition: tuple[int, ...], p: dict[int, np.ndarray]) -> np.ndarray:
    """Canonical distinct-index monomial Gram, sans constant multiplicity.

    Positive multinomial multiplicities are common to every row of one sector,
    so suppressing their square leaves spectra and retained-energy ratios
    exactly unchanged.
    """
    key = tuple(partition)
    p1, p2 = p[1], p[2]
    if key == (2,):
        return p2
    if key == (1, 1):
        return 0.5 * (p1 * p1 - p2)
    if key == (3,):
        return p[3]
    if key == (2, 1):
        return p2 * p1 - p[3]
    if key == (1, 1, 1):
        return (p1**3 - 3.0 * p1 * p2 + 2.0 * p[3]) / 6.0
    if key == (4,):
        return p[4]
    if key == (3, 1):
        return p[3] * p1 - p[4]
    if key == (2, 2):
        return 0.5 * (p2 * p2 - p[4])
    if key == (2, 1, 1):
        return 0.5 * ((p1 * p1 - p2) * p2 - 2.0 * p1 * p[3] + 2.0 * p[4])
    if key == (1, 1, 1, 1):
        return (
            p1**4
            - 6.0 * p1 * p1 * p2
            + 3.0 * p2 * p2
            + 8.0 * p1 * p[3]
            - 6.0 * p[4]
        ) / 24.0
    if key == (5,):
        return p[5]
    if key == (4, 1):
        return p[4] * p1 - p[5]
    if key == (3, 2):
        return p[3] * p2 - p[5]
    if key == (3, 1, 1):
        return 0.5 * ((p1 * p1 - p2) * p[3] - 2.0 * p1 * p[4] + 2.0 * p[5])
    if key == (2, 2, 1):
        return 0.5 * p1 * (p2 * p2 - p[4]) - p2 * p[3] + p[5]
    if key == (2, 1, 1, 1):
        e3 = (p1**3 - 3.0 * p1 * p2 + 2.0 * p[3]) / 6.0
        return p2 * e3 - 0.5 * (
            (p1 * p1 - p2) * p[3] - 2.0 * p1 * p[4] + 2.0 * p[5]
        )
    if key == (1, 1, 1, 1, 1):
        return (
            p1**5
            - 10.0 * p1**3 * p2
            + 15.0 * p1 * p2 * p2
            + 20.0 * p1 * p1 * p[3]
            - 20.0 * p2 * p[3]
            - 30.0 * p1 * p[4]
            + 24.0 * p[5]
        ) / 120.0
    raise ValueError(partition)


def exact_small_n_algebra_check() -> dict[str, float]:
    rng = np.random.default_rng(2026082603)
    left = rng.normal(size=(7, 5))
    p = {power: (left**power).T @ (left**power) for power in range(1, 6)}
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for _name, _order, _group, kind, partition in BLOCKS:
        if kind != "partition":
            continue
        assert partition is not None
        length = len(partition)
        repetition = 1
        for exponent in set(partition):
            repetition *= math.factorial(partition.count(exponent))
        explicit = np.zeros((left.shape[1], left.shape[1]), dtype=np.float64)
        for indices in itertools.permutations(range(left.shape[0]), length):
            feature = np.ones(left.shape[1], dtype=np.float64)
            for index, exponent in zip(indices, partition):
                feature *= left[index] ** exponent
            explicit += np.outer(feature, feature) / repetition
        formula = partition_kernel(partition, p)
        absolute = float(np.max(np.abs(formula - explicit)))
        relative = absolute / max(float(np.max(np.abs(explicit))), 1.0e-300)
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, relative)
    # The largest formula has several cancellation terms; float64 evaluation
    # of the independently enumerated and power-sum spellings agrees well
    # inside a 1e-12 relative arithmetic gate.
    if maximum_relative > 1.0e-12:
        raise RuntimeError(("partition algebra mismatch", maximum_absolute, maximum_relative))
    return {
        "maximum_absolute_error": maximum_absolute,
        "maximum_relative_error": maximum_relative,
    }


def canonicalize_basis(vectors: np.ndarray) -> np.ndarray:
    value = vectors.copy()
    for column in range(value.shape[1]):
        pivot = int(np.argmax(np.abs(value[:, column])))
        if value[pivot, column] < 0.0:
            value[:, column] *= -1.0
    return value


def validate_sources(prereg: dict) -> list[dict]:
    if sha256_file(Path(__file__).resolve()) != prereg["runner_sha256"]:
        raise RuntimeError("runner hash differs from preregistration")
    records = []
    for rank, files in PANEL.items():
        row = STORE / rank
        for name, expected in files.items():
            actual = sha256_file(row / name)
            if actual != expected:
                raise RuntimeError((rank, name, expected, actual))
        manifest = json.loads((row / "row_manifest.json").read_text())
        if manifest["role"] != "observer_fit" or manifest["role_rank"] != int(rank):
            raise RuntimeError((rank, manifest["role"], manifest["role_rank"]))
        if manifest["decoded_columns"] != ["mlp_id", "mlp_name", "mlp_seed", "weights"]:
            raise RuntimeError((rank, manifest["decoded_columns"]))
        if manifest["fits"] != 0 or manifest["scores"] != 0:
            raise RuntimeError((rank, manifest["fits"], manifest["scores"]))
        records.append(
            {
                "role_rank": int(rank),
                "mlp_id": int(manifest["mlp_id"]),
                "mlp_seed": int(manifest["mlp_seed"]),
                "source_ordinal": int(manifest["source_ordinal"]),
                "file_sha256": files,
            }
        )
    return records


def summarize(records: list[dict], group: str, rank_key: str) -> dict:
    values = [
        float(record[rank_key])
        for record in records
        if record["group"] == group
    ]
    return {
        "count": len(values),
        "minimum": min(values),
        "median": float(np.median(values)),
        "maximum": max(values),
    }


def main() -> None:
    if RESULT.exists() or BASIS.exists():
        raise RuntimeError("one-shot outputs already exist")
    prereg = json.loads(PREREG.read_text())
    source_records = validate_sources(prereg)
    algebra = exact_small_n_algebra_check()
    started = time.time()

    basis = np.empty(
        (
            len(PANEL),
            len(SOURCE_LAYERS_ONE_BASED),
            len(BLOCKS),
            RANKS[-1],
            WIDTH,
        ),
        dtype=np.float32,
    )
    records: list[dict] = []
    for network_index, rank in enumerate(PANEL):
        row = STORE / rank
        weights = np.load(row / "weights.npy", mmap_mode="r", allow_pickle=False)
        raw = np.load(row / "cp1792_raw_locals.npy", mmap_mode="r", allow_pickle=False)
        sigma_all = np.load(row / "cp1792_sigma.npy", mmap_mode="r", allow_pickle=False)
        if weights.shape != (DEPTH, WIDTH, WIDTH) or weights.dtype != np.float32:
            raise RuntimeError((rank, weights.shape, weights.dtype))
        if raw.shape != (DEPTH, WIDTH, 20) or raw.dtype != np.float32:
            raise RuntimeError((rank, raw.shape, raw.dtype))
        if sigma_all.shape != (DEPTH, WIDTH) or sigma_all.dtype != np.float32:
            raise RuntimeError((rank, sigma_all.shape, sigma_all.dtype))

        for layer_index, source_layer in enumerate(SOURCE_LAYERS_ONE_BASED):
            # h_l -> z_(l+1) uses zero-based weight index l.  The target-blind
            # raw/sigma row l describes z_(l+1); raw field 3 is alpha=mu/sigma.
            weight = np.asarray(weights[source_layer], dtype=np.float64)
            alpha = np.asarray(raw[source_layer, :, 3], dtype=np.float64)
            sigma = np.asarray(sigma_all[source_layer], dtype=np.float64)
            if not (
                np.all(np.isfinite(weight))
                and np.all(np.isfinite(alpha))
                and np.all(np.isfinite(sigma))
                and np.all(sigma > 0.0)
            ):
                raise RuntimeError((rank, source_layer, "nonfinite source"))

            powers = {1: weight}
            for power in range(2, 6):
                powers[power] = powers[power - 1] * weight
            p = {power: powers[power].T @ powers[power] for power in range(1, 6)}
            coefficients = {
                order: response_coefficient(order, alpha, sigma)
                for order in (2, 3, 4, 5, 6, 7, 9)
            }

            for block_index, (name, order, group, kind, partition) in enumerate(BLOCKS):
                if kind == "partition":
                    assert partition is not None
                    kernel = partition_kernel(partition, p)
                else:
                    kernel = p[1] ** order
                coefficient = coefficients[order]
                gram = coefficient[:, None] * kernel * coefficient[None, :]
                gram = 0.5 * (gram + gram.T)
                trace = float(np.trace(gram))
                if not np.isfinite(trace) or trace <= 0.0:
                    raise RuntimeError((rank, source_layer, name, "bad trace", trace))
                gram /= trace
                diagonal_minimum = float(np.min(np.diag(gram)))
                if diagonal_minimum < -1.0e-14 or not np.all(np.isfinite(gram)):
                    raise RuntimeError((rank, source_layer, name, "bad gram"))

                values, vectors = scipy.linalg.eigh(
                    gram,
                    subset_by_index=(WIDTH - SPECTRUM_PREFIX, WIDTH - 1),
                    check_finite=False,
                    overwrite_a=True,
                    driver="evr",
                )
                values = np.maximum(values, 0.0)
                vectors = canonicalize_basis(vectors)
                retained8 = float(np.sum(values[-RANKS[0] :]))
                retained13 = float(np.sum(values[-RANKS[1] :]))
                cumulative = np.cumsum(values[::-1])
                r70_hit = np.flatnonzero(cumulative >= OBJECTIVE_RELEVANCE_GATE)
                r99_hit = np.flatnonzero(cumulative >= COMPLETENESS_GATE)
                basis[network_index, layer_index, block_index] = (
                    vectors[:, -RANKS[1] :][:, ::-1].T.astype(np.float32)
                )
                records.append(
                    {
                        "role_rank": int(rank),
                        "source_layer_h_one_based": source_layer,
                        "immediate_output_layer_h_one_based": source_layer + 1,
                        "block": name,
                        "order": order,
                        "group": group,
                        "retained_energy_rank8": retained8,
                        "retained_energy_rank13": retained13,
                        "relative_residual_rank8": math.sqrt(max(0.0, 1.0 - retained8)),
                        "relative_residual_rank13": math.sqrt(max(0.0, 1.0 - retained13)),
                        "rank_for_70pct": (
                            int(r70_hit[0] + 1) if r70_hit.size else f">{SPECTRUM_PREFIX}"
                        ),
                        "rank_for_99pct": (
                            int(r99_hit[0] + 1) if r99_hit.size else f">{SPECTRUM_PREFIX}"
                        ),
                        "trace_before_sector_normalization": trace,
                        "top_eigenvalue_fraction": float(values[-1]),
                    }
                )
            print(
                f"observer_fit/{rank} h{source_layer}->h{source_layer + 1} complete "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    np.save(BASIS, basis, allow_pickle=False)
    basis_file_sha = sha256_file(BASIS)
    basis_array_sha = sha256_array(basis)
    marginal13 = summarize(records, "marginal", "retained_energy_rank13")
    joint13 = summarize(records, "joint", "retained_energy_rank13")
    mixed13 = summarize(records, "mixed", "retained_energy_rank13")
    marginal8 = summarize(records, "marginal", "retained_energy_rank8")
    joint8 = summarize(records, "joint", "retained_energy_rank8")
    mixed8 = summarize(records, "mixed", "retained_energy_rank8")

    joint_records = [record for record in records if record["group"] in ("joint", "mixed")]
    weak_pass = all(
        record["retained_energy_rank13"] >= OBJECTIVE_RELEVANCE_GATE
        for record in joint_records
    )
    complete_pass = all(
        record["retained_energy_rank13"] >= COMPLETENESS_GATE
        for record in joint_records
    )
    n = WIDTH
    power_gram_flops_per_layer = 5 * n * n * (2 * n - 1) + 4 * n * n
    projection_cost_rank8 = (
        len(SOURCE_LAYERS_ONE_BASED) * KERDOCK_PATHS * n * (2 * 8 - 1)
    )
    projection_cost_rank13 = (
        len(SOURCE_LAYERS_ONE_BASED) * KERDOCK_PATHS * n * (2 * 13 - 1)
    )
    result = {
        "schema": "p2r0f3b3-multilayer-joint-action-spectrum-result-v1",
        "status": "PASS_ACTION_RANK13" if complete_pass else "FAIL_TERMINAL_ACTION_DIFFUSE",
        "decision": {
            "rank13_completeness_99pct_all_joint_and_mixed": complete_pass,
            "rank13_objective_relevance_70pct_all_joint_and_mixed": weak_pass,
            "meaning": (
                "The saved U13 is the optimal unconstrained endpoint basis. Failure is "
                "an upper-bound failure for every <=13 common-V or factorized-query spelling."
            ),
            "marginal_all_column_escape": (
                "Informational only: Kerdock paths already expose all 1024 scalar column "
                "actions, so marginal rank failure does not close that cheap sidecar."
            ),
        },
        "scope": {
            "source_layers_h_one_based": list(SOURCE_LAYERS_ONE_BASED),
            "immediate_scored_output_layers_h_one_based": [
                layer + 1 for layer in SOURCE_LAYERS_ONE_BASED
            ],
            "panel_selection": (
                "observer_fit role ranks floor(j*687/2), j=0,1,2; fixed stratification "
                "independent of weights, targets, residuals, and scores"
            ),
            "panel_role_ranks": [int(rank) for rank in PANEL],
            "target_values_decoded": 0,
            "observer_fit_target_or_error_arrays_opened_at_or_above_rank_200": 0,
            "observer_selection_files_opened": 0,
            "deployment_or_board_truth_opened": 0,
            "estimator_or_fixture_calls": 0,
        },
        "operator": {
            "definition": (
                "immediate h_l cumulant/covariance innovation -> scored mean h_(l+1), "
                "with covariance-resummed scalar Hermite coefficients"
            ),
            "partition_multinomial_constants": (
                "suppressed per sector because their positive square is a common Gram "
                "scale and cancels retained-energy ratios and eigenvectors"
            ),
            "mixed_shells": {
                "q6": "kappa3^2 coefficient c6/(72 sigma^6)",
                "q7": "kappa3*kappa4 coefficient c7/(144 sigma^7)",
                "q9": "kappa3^3 coefficient c9/(1296 sigma^9)",
            },
            "basis_semantics": (
                "OPTIMAL_ENDPOINT_BASIS_R13.npy axes are network, source layer, block, "
                "descending basis vector, immediate output coordinate"
            ),
            "basis_is_not_hidden_V": (
                "U13 permits arbitrary tensor queries and is strictly more favorable than "
                "a hidden V or factor tuples. A hidden V is emitted only after U13 passes."
            ),
        },
        "gates": {
            "completeness_retained_energy": COMPLETENESS_GATE,
            "objective_relevance_retained_energy": OBJECTIVE_RELEVANCE_GATE,
            "hard_rank_cap": 13,
            "reported_lower_rank": 8,
            "every_network_layer_sector_required": True,
            "no_pooling_can_rescue_failure": True,
        },
        "summary": {
            "rank8": {"marginal": marginal8, "joint": joint8, "mixed": mixed8},
            "rank13": {"marginal": marginal13, "joint": joint13, "mixed": mixed13},
            "worst_rank13_record": min(records, key=lambda item: item["retained_energy_rank13"]),
            "best_rank13_joint_or_mixed_record": max(
                joint_records, key=lambda item: item["retained_energy_rank13"]
            ),
        },
        "records": records,
        "cost": {
            "phase2_budget": PHASE2_BUDGET,
            "d1_worst_bill": D1_WORST_BILL,
            "d1_worst_headroom": PHASE2_BUDGET - D1_WORST_BILL,
            "kerdock_n65536_six_layer_common_v_projection_rank8": projection_cost_rank8,
            "kerdock_n65536_six_layer_common_v_projection_rank13": projection_cost_rank13,
            "source_only_diagnostic_power_gram_flops_per_network_layer": power_gram_flops_per_layer,
            "source_only_diagnostic_power_gram_flops_total": (
                power_gram_flops_per_layer * len(PANEL) * len(SOURCE_LAYERS_ONE_BASED)
            ),
            "dense_eigen_diagnostics": len(records),
            "deployment_cost_admission": (
                "Not reached on rank failure; diagnostic eigensolves are offline and are "
                "not hidden inside a candidate."
            ),
        },
        "integrity": {
            "algebra_self_test": algebra,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "preregistration_sha256": sha256_file(PREREG),
            "basis_file_sha256": basis_file_sha,
            "basis_array_sha256": basis_array_sha,
            "basis_shape": list(basis.shape),
            "source_records": source_records,
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "platform": platform.platform(),
            },
            "elapsed_seconds": time.time() - started,
        },
        "cross_lane_constraints_consumed": {
            "analytic": (
                "complete q2 covariance; q3-q5 collision sectors; mixed q6/q7/q9; "
                "rank<=13; 99% completeness and 70% relevance; immediate action is a "
                "necessary block of any nonlinear suffix"
            ),
            "kerdock": (
                "N=65536 remains viable with recursive Winograd; old scalar h2:h7 reset "
                "is forgotten after the last reset, so persistent Duhamel/action queries are required"
            ),
        },
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
