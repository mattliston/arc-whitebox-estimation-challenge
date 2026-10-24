"""Target-blind raw-K33 instrumentation over the frozen D4 estimator.

This module never injects K33-derived L6 content into the host recurrence.
It computes the preregistered local source/action and a diagnostic mean-only
tangent while returning the frozen D4 native tape byte for byte.  The caller
must provide a streaming sink: no full-plane capture is retained after its
one-layer action.
"""

from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
BASE_SOURCE = HERE.parent / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py"
BASE_SHA256 = "6aedce54f2006efcb9d87eeb41aae6fd785d7344534634371c48a7979158342e"
LEFT_DERIVATIVES = (0, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6)
RIGHT_DERIVATIVES = (0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0)


def _load_base(tag: str) -> Any:
    digest = hashlib.sha256()
    with BASE_SOURCE.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != BASE_SHA256:
        raise RuntimeError("frozen D4 source hash mismatch before import")
    spec = importlib.util.spec_from_file_location(tag, BASE_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen D4 source {BASE_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[tag] = module
    spec.loader.exec_module(module)
    return module


def _install(base: Any, sink: Any) -> tuple[dict[str, Any], Any]:
    """Install association-preserving wrappers and return state plus restore."""

    fnp = base.fnp
    originals = {
        "q24": base._q24_context,
        "connected": base._connected_planes,
        "initialize": base._initialize_h1,
        "reset": base._full_relu_reset,
        "cp_push": base._cp_push,
        "univariate": base._univariate_edgeworth,
        "predict": base._predict,
        "transport": base.transport_correction,
    }
    state: dict[str, Any] = {
        "q24_capture": None,
        "connected_capture": None,
        "k33": None,
        "k33_layer": None,
        "pending_raw_k6": None,
        "current_weight_context": None,
        "tangent": None,
        "q24_calls": 0,
        "connected_calls": 0,
        "source_calls": 0,
        "push_calls": 0,
        "action_calls": 0,
        "reset_calls": 0,
        "univariate_calls": 0,
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
            raise RuntimeError("unconsumed Q24 context before K33 source")
        state["q24_capture"] = (answer, inverse_context)
        state["q24_calls"] += 1
        return answer

    def connected_with_capture(
        mean: Any,
        covariance: Any,
        m11: Any,
        m21: Any,
        m31: Any,
        m22: Any,
        raw2: Any,
        raw3: Any,
        raw4: Any,
        variance: Any,
        diagonal_context: Any,
        fusion_context: Any,
    ) -> Any:
        """Literal frozen D4 connected reset with one captured existing term."""

        mean_i = mean[:, None]
        mean_j = mean[None, :]
        raw2_i = raw2[:, None]
        raw2_j = raw2[None, :]
        raw3_i = raw3[:, None]

        arena = fusion_context[17]
        k21 = arena[8]
        k31 = arena[9]
        k22 = arena[10]
        covariance2 = arena[11]
        variance_outer = arena[12]
        temporary = arena[13]

        fnp.copyto(k21, m21)
        fnp.multiply(raw2_i, mean_j, out=temporary)
        fnp.subtract(k21, temporary, out=k21)
        two_mean_i = 2.0 * mean_i
        fnp.multiply(two_mean_i, m11, out=temporary)
        fnp.subtract(k21, temporary, out=k21)
        two_mean_i2 = two_mean_i * mean_i
        fnp.multiply(two_mean_i2, mean_j, out=temporary)
        fnp.add(k21, temporary, out=k21)

        fnp.copyto(k31, m31)
        fnp.multiply(mean_j, raw3_i, out=temporary)
        fnp.subtract(k31, temporary, out=k31)
        three_mean_i = 3.0 * mean_i
        fnp.multiply(three_mean_i, m21, out=temporary)
        fnp.subtract(k31, temporary, out=k31)
        fnp.multiply(three_mean_i, mean_j, out=temporary)
        fnp.multiply(temporary, raw2_i, out=temporary)
        fnp.add(k31, temporary, out=k31)
        three_mean_i2 = three_mean_i * mean_i
        fnp.multiply(three_mean_i2, m11, out=temporary)
        fnp.add(k31, temporary, out=k31)
        three_mean_i3 = three_mean_i2 * mean_i
        fnp.multiply(three_mean_i3, mean_j, out=temporary)
        fnp.subtract(k31, temporary, out=k31)
        three_variance_i = 3.0 * variance[:, None]
        fnp.multiply(three_variance_i, covariance, out=temporary)
        fnp.subtract(k31, temporary, out=k31)
        fnp.copyto(k31, 0.0, where=diagonal_context)

        m12 = fnp.transpose(m21)
        fnp.copyto(k22, m22)
        two_mean_j = 2.0 * mean_j
        fnp.multiply(two_mean_j, m21, out=temporary)
        fnp.subtract(k22, temporary, out=k22)
        fnp.multiply(two_mean_i, m12, out=temporary)
        fnp.subtract(k22, temporary, out=k22)
        mean_j2 = mean_j * mean_j
        fnp.multiply(mean_j2, raw2_i, out=temporary)
        fnp.add(k22, temporary, out=k22)
        mean_i2 = mean_i * mean_i
        fnp.multiply(mean_i2, raw2_j, out=temporary)
        fnp.add(k22, temporary, out=k22)
        four_mean_i = 4.0 * mean_i
        fnp.multiply(four_mean_i, mean_j, out=temporary)
        fnp.multiply(temporary, m11, out=temporary)
        fnp.add(k22, temporary, out=k22)
        fnp.multiply(three_mean_i2, mean_j, out=temporary)
        fnp.multiply(temporary, mean_j, out=temporary)
        fnp.subtract(k22, temporary, out=k22)

        fnp.multiply(covariance, covariance, out=covariance2)
        fnp.multiply(variance[:, None], variance[None, :], out=variance_outer)
        fnp.subtract(k22, variance_outer, out=k22)
        fnp.multiply(2.0, covariance2, out=temporary)
        fnp.subtract(k22, temporary, out=k22)
        fnp.copyto(k22, 0.0, where=diagonal_context)

        k4 = (
            raw4
            - 4.0 * mean * raw3
            - 3.0 * raw2 * raw2
            + 12.0 * mean * mean * raw2
            - 6.0 * mean * mean * mean * mean
        )
        answer = (k21, k4, k31, k22, covariance2, variance_outer)
        if state["connected_capture"] is not None:
            raise RuntimeError("unconsumed connected reset before K33 source")
        state["connected_capture"] = {
            "mean": mean,
            "covariance": covariance,
            "variance": variance,
            "k21": k21,
            "k31": k31,
            "k22": k22,
            "covariance2": covariance2,
            "variance_outer": variance_outer,
            "three_variance_i": three_variance_i,
        }
        state["connected_calls"] += 1
        return answer

    def centered_c33(
        layer: int,
        coefficients: Any,
        mean: Any,
        weights: Any,
        initial_gaussian: bool,
    ) -> tuple[Any, Any]:
        capture = state["q24_capture"]
        if capture is None:
            raise RuntimeError(f"missing Q24 context for h{layer} K33")
        q24_context, inverse_context = capture
        state["q24_capture"] = None

        mu2 = mean * mean
        mu3 = mu2 * mean
        three_mu = 3.0 * mean
        three_mu2 = 3.0 * mu2
        d3 = coefficients[2] - coefficients[1] * three_mu[None, :]
        d3 = d3 + coefficients[0] * three_mu2[None, :]
        fnp.subtract(d3[0], mu3, out=d3[0])

        centered_grid = (
            d3[base._Q24_ORDER_GRID, :]
            * inverse_context[0][:, None, :]
        )
        _features, _left, _right, rho_powers, inverse_factorial = q24_context
        if initial_gaussian:
            c33 = fnp.einsum(
                "ti,tj,tij,t->ij",
                centered_grid[0],
                centered_grid[0],
                rho_powers,
                inverse_factorial,
                optimize=[(0, 3), (0, 1, 2)],
            )
            temporary = fnp.empty_like(c33)
        else:
            left = centered_grid[LEFT_DERIVATIVES, :, :]
            right = centered_grid[RIGHT_DERIVATIVES, :, :]
            kernels = fnp.einsum(
                "dti,dtj,tij,t->dij",
                left,
                right,
                rho_powers,
                inverse_factorial,
                optimize=[(0, 3), (0, 1, 2)],
            )
            c33 = fnp.einsum(
                "dij,dij->ij", weights, kernels, optimize=[(0, 1)]
            )
            temporary = kernels[0]
        sink.capture_c33(layer, c33)
        return c33, temporary

    def assemble_k33(layer: int, c33: Any, temporary: Any) -> Any:
        connected = state["connected_capture"]
        if connected is None:
            raise RuntimeError(f"missing connected state for h{layer} K33")
        state["connected_capture"] = None
        k33 = c33
        covariance = connected["covariance"]
        k21 = connected["k21"]
        k31 = connected["k31"]

        fnp.multiply(
            connected["three_variance_i"], fnp.transpose(k31), out=temporary
        )
        fnp.subtract(k33, temporary, out=k33)
        fnp.multiply(
            fnp.transpose(connected["three_variance_i"]), k31, out=temporary
        )
        fnp.subtract(k33, temporary, out=k33)
        fnp.multiply(covariance, connected["k22"], out=temporary)
        fnp.multiply(9.0, temporary, out=temporary)
        fnp.subtract(k33, temporary, out=k33)
        k3 = fnp.diag(k21)
        fnp.multiply(k3[:, None], k3[None, :], out=temporary)
        fnp.subtract(k33, temporary, out=k33)
        fnp.multiply(k21, fnp.transpose(k21), out=temporary)
        fnp.multiply(9.0, temporary, out=temporary)
        fnp.subtract(k33, temporary, out=k33)
        fnp.multiply(connected["variance_outer"], covariance, out=temporary)
        fnp.multiply(9.0, temporary, out=temporary)
        fnp.subtract(k33, temporary, out=k33)
        fnp.multiply(connected["covariance2"], covariance, out=temporary)
        fnp.multiply(6.0, temporary, out=temporary)
        fnp.subtract(k33, temporary, out=k33)

        sink.capture_pre_symmetry(layer, k33)
        fnp.add(k33, fnp.transpose(k33), out=temporary)
        fnp.multiply(0.5, temporary, out=k33)
        fnp.fill_diagonal(k33, 0.0)
        return k33

    def birth_source(
        layer: int,
        coefficients: Any,
        mean: Any,
        weights: Any,
        initial_gaussian: bool,
    ) -> None:
        if state["k33"] is not None:
            raise RuntimeError("previous K33 plane was not consumed")
        c33, temporary = centered_c33(
            layer, coefficients, mean, weights, initial_gaussian
        )
        state["k33"] = assemble_k33(layer, c33, temporary)
        state["k33_layer"] = layer
        state["source_calls"] += 1

    def initialize_with_source(*args: Any, **kwargs: Any) -> Any:
        answer = originals["initialize"](*args, **kwargs)
        coefficients = answer[10][0]
        birth_source(1, coefficients, answer[0], None, True)
        return answer

    def reset_with_source(*args: Any, **kwargs: Any) -> Any:
        answer = originals["reset"](*args, **kwargs)
        layer = 2 + int(state["reset_calls"])
        fusion_context = args[14] if len(args) > 14 else kwargs["fusion_context"]
        coefficients = args[6] if len(args) > 6 else kwargs["coefficients"]
        birth_source(layer, coefficients, answer[0], fusion_context[16], False)
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
        if state["k33"] is None or state["k33_layer"] != layer:
            raise RuntimeError(f"missing one-layer K33 source at h{layer}")
        pushed = state["k33"] @ weight_context[4]
        fnp.multiply(weight_context[4], pushed, out=pushed)
        raw_k6 = 10.0 * fnp.sum(pushed, axis=0)
        sink.capture_action(layer, state["k33"], raw_k6)
        state["k33"] = None
        state["k33_layer"] = None
        state["pending_raw_k6"] = raw_k6
        state["current_weight_context"] = weight_context
        state["push_calls"] += 1
        state["action_calls"] += 1
        return answer

    def univariate_with_tangent(
        coefficients: Any,
        raw: Any,
        sigma: Any,
        alpha: Any,
        pdf: Any,
        k3: Any,
        k4: Any,
        sigma2: Any,
        sigma3: Any,
        inverse_context: Any = None,
    ) -> Any:
        if inverse_context is None:
            inverse_context = base._inverse_sigma_context(sigma)
        answer = originals["univariate"](
            coefficients,
            raw,
            sigma,
            alpha,
            pdf,
            k3,
            k4,
            sigma2,
            sigma3,
            inverse_context,
        )
        target_layer = 2 + int(state["univariate_calls"])
        raw_k6 = state["pending_raw_k6"]
        if raw_k6 is None:
            raise RuntimeError(f"missing incoming raw K6 action at h{target_layer}")
        birth = (raw_k6 * inverse_context[5]) * coefficients[0, 6]
        birth = birth / 720.0
        if target_layer == 2:
            tangent = birth
        else:
            weight_context = state["current_weight_context"]
            if weight_context is None or state["tangent"] is None:
                raise RuntimeError("missing weight/tangent association")
            transported = weight_context[1] @ state["tangent"]
            tangent = birth + raw[0] * transported
        sink.capture_birth(
            target_layer,
            raw_k6,
            coefficients[0, 3],
            coefficients[0, 4],
            coefficients[0, 6],
            sigma,
            k3,
            k4,
            birth,
            tangent,
        )
        state["tangent"] = tangent
        state["pending_raw_k6"] = None
        state["univariate_calls"] += 1
        return answer

    base._q24_context = q24_with_capture
    base._connected_planes = connected_with_capture
    base._initialize_h1 = initialize_with_source
    base._full_relu_reset = reset_with_source
    base._cp_push = cp_push_with_action
    base._univariate_edgeworth = univariate_with_tangent

    def predict_with_tape_capture(*args: Any, **kwargs: Any) -> Any:
        answer = originals["predict"](*args, **kwargs)
        if state["public_tape"] is not None:
            raise RuntimeError("public D4 path executed more than once")
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
        base._univariate_edgeworth = originals["univariate"]
        base._predict = originals["predict"]
        base.transport_correction = originals["transport"]

    return state, restore


def predict_with_k33_tape(mlp: Any, sink: Any) -> tuple[Any, dict[str, int]]:
    """Run one full public-D4, zero-feedback K33 source/action observation."""

    tag = f"k33_frozen_d4_{id(mlp)}"
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
        "push_calls": 15,
        "action_calls": 15,
        "reset_calls": 14,
        "univariate_calls": 15,
    }
    observed = {key: int(state[key]) for key in expected}
    if observed != expected:
        raise RuntimeError(f"K33 source/action count mismatch: {observed}")
    if (
        state["q24_capture"] is not None
        or state["connected_capture"] is not None
        or state["k33"] is not None
        or state["pending_raw_k6"] is not None
        or state["tangent"] is None
        or state["public_tape"] is None
        or state["public_correction"] is None
    ):
        raise RuntimeError("K33 source/action terminal state is inconsistent")
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
