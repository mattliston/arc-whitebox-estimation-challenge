"""Target-blind K42+K51 source/action observation over frozen public D4.

This module is deliberately not an estimator and has no autonomous reset.
It observes two oriented sixth-order pair sectors on the unmodified D4-Q7
trajectory, consumes each plane exactly once, and exposes only their shared
raw marginal-kappa6 action vector to a caller-supplied sink.  Importing this
module performs no numerical call and opens no row, name, target, or score.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
BASE_SOURCE = HERE.parent / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py"
BASE_SHA256 = "6aedce54f2006efcb9d87eeb41aae6fd785d7344534634371c48a7979158342e"
PRICING_SOURCE = (
    HERE.parent
    / "p2r0c1d29_k42_k51_pair_inheritance_static_pricing_v1"
    / "K42_K51_STATIC_DESIGN_PRICING_RECEIPT.json"
)
PRICING_SHA256 = "bb8a55e3224da56139dc79fa2b65acc4cf8320f0991c7d77da5ec0892ad3b6cc"
ACCOUNTING_ERRATUM = HERE / "K42_K51_STATIC_ACCOUNTING_ERRATUM_V1.json"
ACCOUNTING_ERRATUM_SHA256 = "63fbf1da297b292b6340c688da3e20b16a850c296e165e05f111b0847534b8c3"

LEFT_DERIVATIVES = (0, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6)
RIGHT_DERIVATIVES = (0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0)

# This is the literal compact-q12, fully batched, coordinate-only spelling
# under FlopScope 0.12.  Binary64 operations meter at rate two, advanced
# float64 gathers at eight units per output element, and mixed f64/f32
# matmul resolves as f64.  The production L6 endpoint is absent.
OPERATION_LEDGER = {
    "width": 1024,
    "source_layers": 15,
    "transitions": 15,
    "initial_batched_source_einsum_calls": 1,
    "later_batched_kernel_einsum_calls": 14,
    "later_batched_reduction_einsum_calls": 14,
    "total_source_einsum_calls": 29,
    "batched_dense_matmul_calls": 15,
    "logical_dense_matmuls": 30,
    "syntactic_fill_diagonal_sites": 2,
    "runtime_fill_diagonal_calls": 30,
    "binary64_operation_rate": 2,
    "advanced_float64_gather_units_per_output": 8,
    "shared_coefficient_and_context_formula": "15*3078*n + 14*3808*n",
    "shared_coefficient_and_context_flops": 101_869_568,
    "source_kernel_and_reduction_formula": "80*n^2 + 28*n + 14*1506*n^2",
    "source_kernel_and_reduction_flops": 22_192_091_136,
    "connected_and_diagonal_formula": "15*(44*n^2 + 20*n)",
    "connected_and_diagonal_flops": 692_367_360,
    "actions_and_shared_sum_formula": "15*(4*n^2*(2*n-1) + 16*n^2 + 2*n)",
    "actions_and_shared_sum_flops": 129_037_793_280,
    "production_endpoint_flops": 0,
    "future_separately_licensed_shared_endpoint_formula": "15*3*n + 14*(n + 2*n^2)",
    "future_separately_licensed_shared_endpoint_flops": 29_420_544,
    "future_shared_endpoint_excluded_from_current_bill": True,
    "incremental_flops": 152_024_121_344,
    "incremental_flops_semantics": "standalone dense diagnostic, not runtime equality",
    "predicted_integrated_incremental_flops": 151_825_424_384,
    "hard_incremental_flop_cap": 152_200_000_000,
    "official_operation_call_deltas": {
        "empty": 16,
        "add": 165,
        "concatenate": 30,
        "copy": 15,
        "getitem": 43,
        "multiply": 1020,
        "reshape": 15,
        "stack": 45,
        "subtract": 210,
        "einsum": 29,
        "diagonal": 15,
        "fill_diagonal": 30,
        "transpose": 15,
        "copyto": 30,
        "matmul": 15,
        "sum": 15,
    },
    "remote_incremental_scratch_ceiling_bytes": 309_092_352,
    "legacy_D4_plus_remote_scientific_scratch_projection_excluding_client_diagnostics_bytes": 7_043_162_112,
    "client_diagnostic_working_ceiling_bytes": 25_165_824,
    "conservative_projected_peak_rss_including_client_diagnostics_bytes": 7_068_327_936,
    "largest_new_array_bytes": 285_212_672,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_base(tag: str) -> Any:
    if _sha256(BASE_SOURCE) != BASE_SHA256:
        raise RuntimeError("frozen D4 source hash mismatch before import")
    if _sha256(PRICING_SOURCE) != PRICING_SHA256:
        raise RuntimeError("frozen D29 pricing hash mismatch before import")
    if _sha256(ACCOUNTING_ERRATUM) != ACCOUNTING_ERRATUM_SHA256:
        raise RuntimeError("corrected K42+K51 accounting authority mismatch")
    spec = importlib.util.spec_from_file_location(tag, BASE_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen D4 source {BASE_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[tag] = module
    spec.loader.exec_module(module)
    return module


def _install(base: Any, sink: Any) -> tuple[dict[str, Any], Any]:
    """Install read-only association wrappers and return state plus restore."""

    fnp = base.fnp
    originals = {
        "q24": base._q24_context,
        "connected": base._connected_planes,
        "initialize": base._initialize_h1,
        "reset": base._full_relu_reset,
        "cp_push": base._cp_push,
        "predict": base._predict,
        "transport": base.transport_correction,
    }
    state: dict[str, Any] = {
        "q24_capture": None,
        "connected_capture": None,
        "pair_planes": None,
        "pair_layer": None,
        "q24_calls": 0,
        "connected_calls": 0,
        "source_calls": 0,
        "source_einsum_calls": 0,
        "push_calls": 0,
        "action_calls": 0,
        "batched_matmul_calls": 0,
        "logical_matmuls": 0,
        "fill_diagonal_calls": 0,
        "reset_calls": 0,
        "public_tape": None,
        "public_correction": None,
    }

    def q24_with_capture(
        coefficients: Any,
        sigma: Any,
        covariance: Any,
        fusion_context: Any,
        inverse_context: Any = None,
    ) -> Any:
        if inverse_context is None:
            inverse_context = base._inverse_sigma_context(sigma)
        answer = originals["q24"](
            coefficients,
            sigma,
            covariance,
            fusion_context,
            inverse_context,
        )
        if state["q24_capture"] is not None:
            raise RuntimeError("unconsumed Q24 context before pair source")
        state["q24_capture"] = (answer, inverse_context)
        state["q24_calls"] += 1
        return answer

    def connected_with_capture(*args: Any, **kwargs: Any) -> Any:
        answer = originals["connected"](*args, **kwargs)
        if state["connected_capture"] is not None:
            raise RuntimeError("unconsumed connected state before pair source")
        mean = args[0] if args else kwargs["mean"]
        covariance = args[1] if len(args) > 1 else kwargs["covariance"]
        variance = args[9] if len(args) > 9 else kwargs["variance"]
        k21, k4, k31, k22, covariance2, variance_outer = answer
        state["connected_capture"] = {
            "mean": mean,
            "covariance": covariance,
            "variance": variance,
            "k21": k21,
            "k4": k4,
            "k31": k31,
            "k22": k22,
            "covariance2": covariance2,
            "variance_outer": variance_outer,
        }
        state["connected_calls"] += 1
        return answer

    def centered_rows_q12(
        coefficients: Any,
        gaussian_raw: Any,
        sigma: Any,
        alpha: Any,
        corrected_mean: Any,
    ) -> Any:
        """Return d4,d2,d5,d1 on exactly q=0..12 in raw units."""

        c1 = coefficients[0, :13]
        c2 = coefficients[1, :13]
        c3 = coefficients[2, :13]
        _mass, raw1, raw2, raw3, raw4 = gaussian_raw

        sigma2 = sigma * sigma
        sigma3 = sigma2 * sigma
        sigma4 = sigma2 * sigma2
        pre_mean = alpha * sigma
        raw5 = pre_mean * raw4 + (4.0 * sigma2) * raw3

        four_sigma = 4.0 * sigma
        c4 = fnp.concatenate(
            (
                fnp.stack(
                    (
                        raw4,
                        four_sigma * raw3,
                        (12.0 * sigma2) * raw2,
                        (24.0 * sigma3) * raw1,
                    ),
                    axis=0,
                ),
                four_sigma[None, :] * c3[3:12],
            ),
            axis=0,
        )
        five_sigma = 5.0 * sigma
        c5 = fnp.concatenate(
            (
                fnp.stack(
                    (
                        raw5,
                        five_sigma * raw4,
                        (20.0 * sigma2) * raw3,
                        (60.0 * sigma3) * raw2,
                        (120.0 * sigma4) * raw1,
                    ),
                    axis=0,
                ),
                five_sigma[None, :] * c4[4:12],
            ),
            axis=0,
        )

        mu2 = corrected_mean * corrected_mean
        mu3 = mu2 * corrected_mean
        mu4 = mu2 * mu2
        mu5 = mu4 * corrected_mean

        d1 = fnp.copy(c1)
        fnp.subtract(d1[0], corrected_mean, out=d1[0])

        two_mu = 2.0 * corrected_mean
        d2 = c2 - c1 * two_mu[None, :]
        fnp.add(d2[0], mu2, out=d2[0])

        four_mu = 4.0 * corrected_mean
        six_mu2 = 6.0 * mu2
        four_mu3 = 4.0 * mu3
        d4 = c4 - c3 * four_mu[None, :]
        d4 = d4 + c2 * six_mu2[None, :]
        d4 = d4 - c1 * four_mu3[None, :]
        fnp.add(d4[0], mu4, out=d4[0])

        five_mu = 5.0 * corrected_mean
        ten_mu2 = 10.0 * mu2
        ten_mu3 = 10.0 * mu3
        five_mu4 = 5.0 * mu4
        d5 = c5 - c4 * five_mu[None, :]
        d5 = d5 + c3 * ten_mu2[None, :]
        d5 = d5 - c2 * ten_mu3[None, :]
        d5 = d5 + c1 * five_mu4[None, :]
        fnp.subtract(d5[0], mu5, out=d5[0])
        return fnp.stack((d4, d2, d5, d1), axis=0)

    def assemble_connected_pairs(
        layer: int,
        central_moments: Any,
        kernel_scratch: Any,
    ) -> Any:
        connected = state["connected_capture"]
        if connected is None:
            raise RuntimeError(f"missing connected state for h{layer}")
        state["connected_capture"] = None

        planes = central_moments
        k42 = planes[0]
        k51 = planes[1]
        temporary = kernel_scratch[0]
        secondary = kernel_scratch[1]
        covariance = connected["covariance"]
        variance = connected["variance"]
        k21 = connected["k21"]
        k4 = connected["k4"]
        k31 = connected["k31"]
        k22 = connected["k22"]
        k3 = fnp.diagonal(k21)

        # K42=C42-vj*K4i-3*vi^2*vj-8*C*K31-12*vi*C^2
        #     -4*K12*k3i-6*vi*K22-6*K21^2.
        variance2 = variance * variance
        diagonal_prefactor = k4 + 3.0 * variance2
        fnp.multiply(diagonal_prefactor[:, None], variance[None, :], out=temporary)
        fnp.subtract(k42, temporary, out=k42)
        fnp.multiply(variance[:, None], covariance, out=temporary)
        fnp.multiply(12.0, temporary, out=temporary)
        fnp.multiply(8.0, k31, out=secondary)
        fnp.add(temporary, secondary, out=temporary)
        fnp.multiply(temporary, covariance, out=temporary)
        fnp.subtract(k42, temporary, out=k42)
        fnp.multiply(fnp.transpose(k21), k3[:, None], out=temporary)
        fnp.multiply(4.0, temporary, out=temporary)
        fnp.subtract(k42, temporary, out=k42)
        fnp.multiply(variance[:, None], k22, out=temporary)
        fnp.multiply(k21, k21, out=secondary)
        fnp.add(temporary, secondary, out=temporary)
        fnp.multiply(6.0, temporary, out=temporary)
        fnp.subtract(k42, temporary, out=k42)
        fnp.fill_diagonal(k42, 0.0)
        state["fill_diagonal_calls"] += 1

        # K51=C51-C*(5*K4i+15*vi^2)-10*vi*K31-10*K21*k3i.
        diagonal_prefactor = 5.0 * k4 + 15.0 * variance2
        fnp.multiply(covariance, diagonal_prefactor[:, None], out=temporary)
        fnp.subtract(k51, temporary, out=k51)
        ten_variance = 10.0 * variance
        fnp.multiply(k31, ten_variance[:, None], out=temporary)
        fnp.subtract(k51, temporary, out=k51)
        ten_k3 = 10.0 * k3
        fnp.multiply(k21, ten_k3[:, None], out=temporary)
        fnp.subtract(k51, temporary, out=k51)
        fnp.fill_diagonal(k51, 0.0)
        state["fill_diagonal_calls"] += 1

        sink.capture_source(layer, k42, k51)
        return planes

    def birth_source(
        layer: int,
        coefficients: Any,
        gaussian_raw: Any,
        sigma: Any,
        alpha: Any,
        corrected_mean: Any,
        weights: Any,
        initial_gaussian: bool,
    ) -> None:
        if state["pair_planes"] is not None:
            raise RuntimeError("previous pair planes were not consumed")
        capture = state["q24_capture"]
        if capture is None:
            raise RuntimeError(f"missing Q24 context for h{layer} pair source")
        q24_context, inverse_context = capture
        state["q24_capture"] = None

        centered = centered_rows_q12(
            coefficients, gaussian_raw, sigma, alpha, corrected_mean
        )
        centered_grid = fnp.reshape(
            centered[:, base._Q24_ORDER_GRID, :]
            * inverse_context[0][None, :, None, :],
            (2, 2, 7, 7, int(sigma.shape[0])),
        )
        _features, _left, _right, rho_powers, inverse_factorial = q24_context
        sector_left = centered_grid[:, 0]
        sector_right = centered_grid[:, 1]
        if initial_gaussian:
            central_moments = fnp.einsum(
                "sti,stj,tij,t->sij",
                sector_left[:, 0],
                sector_right[:, 0],
                rho_powers,
                inverse_factorial,
                optimize=[(0, 3), (0, 1, 2)],
            )
            state["source_einsum_calls"] += 1
            kernel_scratch = fnp.empty(
                (2, int(sigma.shape[0]), int(sigma.shape[0])),
                dtype=fnp.float64,
            )
        else:
            left = sector_left[:, LEFT_DERIVATIVES, :, :]
            right = sector_right[:, RIGHT_DERIVATIVES, :, :]
            kernels = fnp.einsum(
                "sdti,sdtj,tij,t->sdij",
                left,
                right,
                rho_powers,
                inverse_factorial,
                optimize=[(0, 3), (0, 1, 2)],
            )
            central_moments = fnp.einsum(
                "dij,sdij->sij", weights, kernels, optimize=[(0, 1)]
            )
            state["source_einsum_calls"] += 2
            kernel_scratch = kernels[0, :2]
        state["pair_planes"] = assemble_connected_pairs(
            layer, central_moments, kernel_scratch
        )
        state["pair_layer"] = layer
        state["source_calls"] += 1

    def initialize_with_source(*args: Any, **kwargs: Any) -> Any:
        answer = originals["initialize"](*args, **kwargs)
        coefficients, gaussian_raw, sigma, alpha, _pdf = answer[10]
        birth_source(
            1,
            coefficients,
            gaussian_raw,
            sigma,
            alpha,
            answer[0],
            None,
            True,
        )
        return answer

    def reset_with_source(*args: Any, **kwargs: Any) -> Any:
        answer = originals["reset"](*args, **kwargs)
        layer = 2 + int(state["reset_calls"])
        coefficients = args[6] if len(args) > 6 else kwargs["coefficients"]
        gaussian_raw = args[7] if len(args) > 7 else kwargs["gaussian_raw"]
        sigma = args[8] if len(args) > 8 else kwargs["sigma"]
        alpha = args[11] if len(args) > 11 else kwargs["alpha"]
        fusion_context = args[14] if len(args) > 14 else kwargs["fusion_context"]
        birth_source(
            layer,
            coefficients,
            gaussian_raw,
            sigma,
            alpha,
            answer[0],
            fusion_context[16],
            False,
        )
        state["reset_calls"] += 1
        return answer

    def cp_push_with_action(
        cp_lambda: Any,
        cp_p: Any,
        cp_q: Any,
        weight_context: Any,
        output_p: Any,
        output_q: Any,
        arena_views: Any,
    ) -> Any:
        answer = originals["cp_push"](
            cp_lambda,
            cp_p,
            cp_q,
            weight_context,
            output_p,
            output_q,
            arena_views,
        )
        layer = 1 + int(state["push_calls"])
        planes = state["pair_planes"]
        if planes is None or state["pair_layer"] != layer:
            raise RuntimeError(f"missing one-transition pair source at h{layer}")

        width = int(planes.shape[1])
        scratch = fnp.empty((6, width, width), dtype=fnp.float64)
        left_powers = scratch[0:2]
        right_powers = scratch[2:4]
        pushed = scratch[4:6]
        # weight_context=(W,W.T,W2,W2.T,W3,W3.T).  K42 uses W4/W2;
        # K51 uses W5/W.  One batched matmul dispatches both orientations.
        fnp.multiply(weight_context[2], weight_context[2], out=left_powers[0])
        fnp.multiply(weight_context[4], weight_context[2], out=left_powers[1])
        fnp.copyto(right_powers[0], weight_context[2])
        fnp.copyto(right_powers[1], weight_context[0])
        fnp.matmul(planes, right_powers, out=pushed)
        state["batched_matmul_calls"] += 1
        state["logical_matmuls"] += 2
        fnp.multiply(left_powers, pushed, out=pushed)
        raw_by_sector = fnp.sum(pushed, axis=1)
        fnp.multiply(15.0, raw_by_sector[0], out=raw_by_sector[0])
        fnp.multiply(6.0, raw_by_sector[1], out=raw_by_sector[1])
        raw_total = raw_by_sector[0] + raw_by_sector[1]
        sink.capture_action(layer, raw_by_sector, raw_total)

        state["pair_planes"] = None
        state["pair_layer"] = None
        state["push_calls"] += 1
        state["action_calls"] += 1
        return answer

    base._q24_context = q24_with_capture
    base._connected_planes = connected_with_capture
    base._initialize_h1 = initialize_with_source
    base._full_relu_reset = reset_with_source
    base._cp_push = cp_push_with_action

    def predict_with_tape_capture(*args: Any, **kwargs: Any) -> Any:
        answer = originals["predict"](*args, **kwargs)
        if state["public_tape"] is not None:
            raise RuntimeError("public D4 internal path executed more than once")
        if not isinstance(answer, tuple) or len(answer) != 3:
            raise RuntimeError("public D4 internal path did not return native tape")
        state["public_tape"] = answer
        return answer

    def transport_with_capture(*args: Any, **kwargs: Any) -> Any:
        answer = originals["transport"](*args, **kwargs)
        if state["public_correction"] is not None:
            raise RuntimeError("native-J correction executed more than once")
        state["public_correction"] = answer
        return answer

    base._predict = predict_with_tape_capture
    base.transport_correction = transport_with_capture

    def restore() -> None:
        base._q24_context = originals["q24"]
        base._connected_planes = originals["connected"]
        base._initialize_h1 = originals["initialize"]
        base._full_relu_reset = originals["reset"]
        base._cp_push = originals["cp_push"]
        base._predict = originals["predict"]
        base.transport_correction = originals["transport"]

    return state, restore


def predict_with_pair_tape(mlp: Any, sink: Any) -> tuple[Any, dict[str, int]]:
    """Observe one full public-D4 call; never feed either sector back."""

    tag = f"k42_k51_frozen_d4_{id(mlp)}"
    base = _load_base(tag)
    state, restore = _install(base, sink)
    try:
        final_output = base.Estimator().predict(mlp, base._PHASE2_FLOP_BUDGET)
    finally:
        restore()
    expected = {
        "q24_calls": 15,
        "connected_calls": 15,
        "source_calls": 15,
        "source_einsum_calls": 29,
        "push_calls": 15,
        "action_calls": 15,
        "batched_matmul_calls": 15,
        "logical_matmuls": 30,
        "fill_diagonal_calls": 30,
        "reset_calls": 14,
    }
    observed = {key: int(state[key]) for key in expected}
    if observed != expected:
        raise RuntimeError(f"K42+K51 source/action count mismatch: {observed}")
    if (
        state["q24_capture"] is not None
        or state["connected_capture"] is not None
        or state["pair_planes"] is not None
        or state["public_tape"] is None
        or state["public_correction"] is None
    ):
        raise RuntimeError("K42+K51 terminal association state is inconsistent")
    host, native_raw, native_sigma = state["public_tape"]
    output = (
        host,
        native_raw,
        native_sigma,
        state["public_correction"],
        final_output,
    )
    sys.modules.pop(tag, None)
    return output, observed
