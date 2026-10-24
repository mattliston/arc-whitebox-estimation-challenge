"""Inert, input-only second location jet of the current response-P GNN.

This is a NumPy reference, NOT a participant implementation or native bill.
There is no file, payload, target, model, or network loader and no CLI run.
The caller supplies and authenticates every input.  No caller array is changed.

Definition of the continuous extension
------------------------------------
* The original FP32 prior state, message parameters, and W powers generate
  the six fixed message contractions, in the current 409f expression order.
  Their FP32 results (including the three log-energy messages) are constants.
* Raw x, its diagonal virtual-location derivatives dx/d2x, sigma, and the
  original FP32 head parameters are promoted to FP64.  Normalization,
  clipping, the 126->128->128->64 encoder, and the layer head run in FP64.
* Row i of dx/d2x means partial_mu_i x_i / partial_mu_i^2 x_i.  Previous
  state and messages have ZERO derivatives.  There is no same-layer mixing
  of newly encoded states, so these rowwise jets give diagonal responses.
* Both derivatives of clip are zero outside AND at exactly +/-8.  Inside,
  both input jets pass through.  Endpoint and nonfinite counts are returned;
  endpoints are a convention, not a claim of classical differentiability.
* The result is the learned correction delta, NOT host_mean+delta and NOT
  the final served maximum/cast.  The caller owns host derivatives.  These
  are not literal derivatives of the discontinuous served FP32 program.

Preparation is paid.  prepare_fixed_messages may be reused only for its
same, immutable W/prior-state/message-parameter owners; prepare_head may be
reused only for its same immutable payload owners.  step reports when a
preparation was supplied rather than performed.  Do not silently omit its
cost from a complete acquisition.  Fused reuse of already-computed current
messages is a separate implementation/provenance obligation.

CostLedger lists every numerical reference operation, including casts,
copies, concatenation, comparisons, clipping, and diagnostic scans.  It
uses ordinary dense FNP.12 tariffs, no symmetry/zero discounts: FP64 rate2,
FP32/bool rate1, exp/log1p/tanh/power weight16, where weight4, two-bound clip
2 comparisons per output, matmul m*n*(2*k-1), and casts at the heavier rate.
Basic views cost0; zero creation costs0.  Checks based only on shapes/dtypes
and Python integer metadata are not numerical array operations.  This
analytical ledger is independently testable, NOT a FlopScope qualification.
Gross output bytes are traffic/creation bookkeeping, NOT peak live RSS.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


CURRENT_SOURCE_SHA256 = "409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f"
FNP_TARIFF_SHA256 = "f729bc47f07c080b8c7b9d7b045e5663f0af313669b4c2a962b5a9044a94cb7a"
DEPTH, FEATURES, MESSAGE_CHANNELS, HIDDEN, STATE = 16, 30, 16, 128, 64
ENCODER_INPUT = FEATURES + 6 * MESSAGE_CHANNELS
_HEAD_SHAPES = {
    "centers": (DEPTH, FEATURES), "scales": (DEPTH, FEATURES),
    "rms": (DEPTH,), "enc0_w": (HIDDEN, ENCODER_INPUT),
    "enc0_b": (HIDDEN,), "enc2_w": (HIDDEN, HIDDEN),
    "enc2_b": (HIDDEN,), "enc4_w": (STATE, HIDDEN),
    "enc4_b": (STATE,), "head": (DEPTH, STATE),
    "head_b": (DEPTH,), "gain": (DEPTH,),
}
_WEIGHTS = {name: 1 for name in (
    "add", "subtract", "multiply", "divide", "negative", "maximum",
    "greater", "less", "less_equal", "equal", "logical_and", "logical_or",
    "isfinite", "count_nonzero", "astype", "concatenate", "clip", "matmul",
)}
_WEIGHTS.update(exp=16, log1p=16, tanh=16, power=16, where=4,
                zeros=0, view=0)


def _array(value, name, shape=None, dtype=None):
    if type(value) is not np.ndarray:
        raise TypeError(f"{name} must be an ordinary NumPy ndarray, not a subclass")
    if value.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError(f"{name} must have float32 or float64 dtype")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name}: shape {value.shape}, expected {shape}")
    if dtype is not None and value.dtype != np.dtype(dtype):
        raise TypeError(f"{name}: dtype {value.dtype}, expected {dtype}")
    return value


def _rate(dtype):
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        return 1
    if dtype.kind == "f" and dtype.itemsize in (4, 8):
        return dtype.itemsize // 4
    raise TypeError(f"Unsupported billing dtype {dtype}")


@dataclass
class CostLedger:
    records: list[dict[str, Any]] = field(default_factory=list)

    def record(self, op, label, result, base_flops, billing_dtype, phase):
        shape = tuple(np.shape(result))
        dtype = np.dtype(billing_dtype)
        cost = int(base_flops) * _rate(dtype) * _WEIGHTS[op]
        self.records.append(dict(
            operation=op, label=label, phase=phase, shape=list(shape),
            billing_dtype=str(dtype), base_flops=int(base_flops),
            dtype_rate=_rate(dtype), operation_weight=_WEIGHTS[op], flops=cost,
            gross_output_bytes=0 if op == "view" else int(np.size(result))*np.asarray(result).dtype.itemsize,
        ))
        return result

    def report(self, start=0):
        records = self.records[start:]
        phases = {}
        for item in records:
            phases[item["phase"]] = phases.get(item["phase"], 0) + item["flops"]
        return dict(
            total_flops=sum(item["flops"] for item in records),
            by_phase=phases, operations=[dict(item) for item in records],
            tariff_sha256=FNP_TARIFF_SHA256,
            qualification="Analytical ordinary-dense reference ledger; not a native or complete-estimator bill.",
        )


class _Ops:
    def __init__(self, ledger, phase):
        self.ledger, self.phase = ledger, phase

    def unary(self, op, x, label):
        with np.errstate(all="ignore"):
            result = getattr(np, op)(x)
        return self.ledger.record(op, label, result, np.size(result), np.asarray(x).dtype, self.phase)

    def binary(self, op, a, b, label):
        dtype = np.result_type(a, b)
        with np.errstate(all="ignore"):
            result = getattr(np, op)(a, b)
        return self.ledger.record(op, label, result, np.size(result), dtype, self.phase)

    def cast(self, x, dtype, label):
        dtype = np.dtype(dtype)
        billed = dtype if _rate(dtype) >= _rate(x.dtype) else x.dtype
        result = x.astype(dtype, copy=True)
        return self.ledger.record("astype", label, result, x.size, billed, self.phase)

    def view(self, x, key, label):
        result = x[key]
        return self.ledger.record("view", label, result, 0, x.dtype, self.phase)

    def transpose(self, x, label):
        return self.ledger.record("view", label, x.T, 0, x.dtype, self.phase)

    def zeros(self, shape, dtype, label):
        result = np.zeros(shape, dtype=dtype)
        return self.ledger.record("zeros", label, result, 0, result.dtype, self.phase)

    def concat(self, arrays, label):
        result = np.concatenate(arrays, axis=1)
        return self.ledger.record("concatenate", label, result, result.size, result.dtype, self.phase)

    def matmul(self, a, b, label):
        if a.ndim != 2 or b.ndim not in (1, 2) or a.shape[1] != b.shape[0]:
            raise ValueError("Only compatible ordinary matrix/matrix or matrix/vector products are supported")
        with np.errstate(all="ignore"):
            result = a @ b
        cost = a.shape[0] * (1 if b.ndim == 1 else b.shape[1]) * (2*a.shape[1]-1)
        return self.ledger.record("matmul", label, result, cost, np.result_type(a, b), self.phase)

    def clip(self, x, label):
        result = np.clip(x, -8.0, 8.0)
        return self.ledger.record("clip", label, result, 2*x.size, x.dtype, self.phase)

    def where_zero(self, condition, x, label):
        result = np.where(condition, x, 0.0)
        return self.ledger.record("where", label, result, result.size, x.dtype, self.phase)

    def count(self, x, label):
        result = int(np.count_nonzero(x))
        return self.ledger.record("count_nonzero", label, result, x.size, x.dtype, self.phase)


def _nonfinite(ledger, named):
    ops = _Ops(ledger, "diagnostics")
    counts = {}
    for name, value in named.items():
        finite = ops.unary("isfinite", value, name+".isfinite")
        counts[name] = int(np.size(value)) - ops.count(finite, name+".finite_count")
    return counts


@dataclass(frozen=True)
class FixedMessages:
    values: tuple[np.ndarray, ...]  # m1,m2,m3,log_energy1,log_energy2,log_energy3
    owners: tuple[np.ndarray, ...]  # W, prior state, original msg_w, original msg_b
    cost: dict[str, Any]
    nonfinite: dict[str, int]


@dataclass(frozen=True)
class PreparedHead:
    values: Mapping[str, np.ndarray]
    owners: tuple[np.ndarray, ...]
    cost: dict[str, Any]
    nonfinite: dict[str, int]
    nonpositive_scales: int


@dataclass(frozen=True)
class SecondJet:
    delta: np.ndarray
    state: np.ndarray
    ddelta: np.ndarray
    dstate: np.ndarray
    d2delta: np.ndarray
    d2state: np.ndarray
    diagnostics: dict[str, Any]
    cost: dict[str, Any]
    preparation: dict[str, Any]


def prepare_fixed_messages(W, prior_state, payload, *, ledger=None):
    """Pay the original FP32 message circuit once for this fixed state/W."""
    ledger = CostLedger() if ledger is None else ledger
    start = len(ledger.records)
    n = _array(W, "W").shape[0]
    _array(W, "W", (n, n))
    _array(prior_state, "prior_state", (n, STATE), "float32")
    mw = _array(payload["msg_w"], "msg_w", (3*MESSAGE_CHANNELS, STATE), "float32")
    mb = _array(payload["msg_b"], "msg_b", (3*MESSAGE_CHANNELS,), "float32")
    ops = _Ops(ledger, "message_preparation")
    src = ops.binary("add", ops.matmul(prior_state, ops.transpose(mw, "msg_w.T"), "state@msg_w.T"), mb, "source_bias")
    w = ops.cast(W, "float32", "W.astype(float32)")
    w2 = ops.binary("multiply", w, w, "W2")
    w3 = ops.binary("multiply", w2, w, "W3")
    parts = tuple(ops.view(src, (slice(None), slice(j*MESSAGE_CHANNELS, (j+1)*MESSAGE_CHANNELS)), f"source{j+1}") for j in range(3))
    powers = (w, w2, w3)
    messages = tuple(ops.matmul(ops.transpose(power, f"W{j+1}.T"), part, f"message{j+1}") for j, (power, part) in enumerate(zip(powers, parts)))
    energy1 = ops.matmul(ops.transpose(w2, "energy_W2.T"), ops.binary("power", parts[0], 2, "source1**2"), "energy1")
    w4 = ops.binary("multiply", w2, w2, "W4")
    energy2 = ops.matmul(ops.transpose(w4, "energy_W4.T"), ops.binary("power", parts[1], 2, "source2**2"), "energy2")
    w6 = ops.binary("multiply", w3, w3, "W6")
    energy3 = ops.matmul(ops.transpose(w6, "energy_W6.T"), ops.binary("power", parts[2], 2, "source3**2"), "energy3")
    energies = (energy1, energy2, energy3)
    logs = tuple(ops.unary("log1p", ops.binary("maximum", e, 0, f"energy{j+1}.max0"), f"energy{j+1}.log1p") for j, e in enumerate(energies))
    counts = _nonfinite(ledger, {"W": W, "prior_state": prior_state, "msg_w": mw, "msg_b": mb,
                                 **{f"fixed_message{j}": value for j, value in enumerate(messages+logs)},
                                 **{f"raw_energy{j}": value for j, value in enumerate(energies)}})
    return FixedMessages(messages+logs, (W, prior_state, mw, mb), ledger.report(start), counts)


def prepare_head(payload, *, ledger=None):
    """Pay all FP32->FP64 head parameter casts once; do not load a payload."""
    ledger = CostLedger() if ledger is None else ledger
    start = len(ledger.records)
    originals = tuple(_array(payload[name], name, shape, "float32") for name, shape in _HEAD_SHAPES.items())
    ops = _Ops(ledger, "head_preparation")
    values = {name: ops.cast(value, "float64", name+".astype(float64)") for name, value in zip(_HEAD_SHAPES, originals)}
    values["gain_multiplier"] = ops.binary("add", 1.0, ops.binary("multiply", .25, ops.unary("tanh", values["gain"], "gain.tanh"), "gain.quarter"), "gain.plus1")
    counts = _nonfinite(ledger, values)
    diag = _Ops(ledger, "diagnostics")
    nonpositive = diag.count(diag.binary("less_equal", values["scales"], 0.0, "scales<=0"), "nonpositive_scale_count")
    return PreparedHead(values, originals, ledger.report(start), counts, nonpositive)


def _clip_jet(value, first, second, ledger, label, diagnostics):
    ops, diag = _Ops(ledger, "jet"), _Ops(ledger, "diagnostics")
    interior = ops.binary("logical_and", ops.binary("greater", value, -8.0, label+">-8"), ops.binary("less", value, 8.0, label+"<8"), label+".interior")
    endpoint = diag.binary("logical_or", diag.binary("equal", value, -8.0, label+"==-8"), diag.binary("equal", value, 8.0, label+"==8"), label+".endpoint")
    diagnostics["clip_endpoint_counts"][label] = diag.count(endpoint, label+".endpoint_count")
    diagnostics["nonfinite"].update(_nonfinite(ledger, {label+".value": value, label+".first": first, label+".second": second}))
    return ops.clip(value, label+".clip"), ops.where_zero(interior, first, label+".first_mask"), ops.where_zero(interior, second, label+".second_mask")


def _silu_jet(value, first, second, ledger, label, diagnostics):
    ops = _Ops(ledger, "jet")
    denominator = ops.binary("add", 1.0, ops.unary("exp", ops.unary("negative", value, label+".negative"), label+".exp"), label+".denominator")
    output = ops.binary("divide", value, denominator, label+".value")
    sigmoid = ops.binary("divide", 1.0, denominator, label+".sigmoid")
    q = ops.binary("multiply", sigmoid, ops.binary("subtract", 1.0, sigmoid, label+".one_minus_sigmoid"), label+".sigmoid_slope")
    fp = ops.binary("add", sigmoid, ops.binary("multiply", value, q, label+".tq"), label+".first_response")
    fpp = ops.binary("multiply", q, ops.binary("add", 2.0, ops.binary("multiply", value, ops.binary("subtract", 1.0, ops.binary("multiply", 2.0, sigmoid, label+".two_sigmoid"), label+".one_minus_two_sigmoid"), label+".curvature_inner"), label+".curvature_plus2"), label+".second_response")
    doutput = ops.binary("multiply", fp, first, label+".first")
    d2output = ops.binary("add", ops.binary("multiply", fpp, ops.binary("multiply", first, first, label+".first_squared"), label+".quadratic_chain"), ops.binary("multiply", fp, second, label+".linear_chain"), label+".second")
    diagnostics["nonfinite"].update(_nonfinite(ledger, {
        label+".input_value": value, label+".input_first": first,
        label+".input_second": second, label+".denominator": denominator,
        label+".sigmoid": sigmoid, label+".value": output,
        label+".first": doutput, label+".second": d2output,
    }))
    return output, doutput, d2output


def step(layer, x, dx, d2x, sigma, W, prior_state, payload, *,
         messages=None, head=None, ledger=None):
    """Return FP64 learned correction/state and their first/second jets.

    A supplied preparation incurs no new preparation operations in this call;
    its separately returned receipt MUST remain in the complete caller bill.
    Input and prepared owners must remain immutable throughout their lifetime.
    Nonfinite values are reported, never clipped/repaired or silently accepted.
    """
    if type(layer) is not int or not 0 <= layer < DEPTH:
        raise ValueError("layer must be a Python integer in0..15")
    ledger = CostLedger() if ledger is None else ledger
    start = len(ledger.records)
    n = _array(x, "x").shape[0]
    for name, value in (("x", x), ("dx", dx), ("d2x", d2x)):
        _array(value, name, (n, FEATURES))
    _array(sigma, "sigma", (n,))
    _array(W, "W", (n, n))
    _array(prior_state, "prior_state", (n, STATE), "float32")
    diagnostics = dict(clip_endpoint_counts={}, nonfinite={}, zero_clip_endpoint_derivatives=True,
                       interpretation="FP64 continuous extension with frozen FP32 message constants; not a served-FP32 derivative.")
    ops = _Ops(ledger, "jet")
    if layer == 0:
        zero = ops.zeros((n,), "float64", "layer0.zero_delta")
        state = ops.binary("multiply", ops.cast(prior_state, "float64", "layer0.state_cast"), 0.0, "layer0.state_zero")
        zs = ops.zeros((n, STATE), "float64", "layer0.zero_state_jets")
        diagnostics["nonfinite"] = _nonfinite(ledger, {"delta": zero, "state": state})
        diagnostics.update(all_finite=not any(diagnostics["nonfinite"].values()),
                           message_preparation_nonfinite={}, head_preparation_nonfinite={},
                           nonpositive_head_scales=0)
        preparation = dict(messages_reused=False, head_reused=False,
                           messages_cost=None, head_cost=None,
                           qualification="Layer0 does not use messages or head parameters.")
        return SecondJet(zero, state, zero, zs, zero, zs, diagnostics, ledger.report(start), preparation)
    message_reused, head_reused = messages is not None, head is not None
    if messages is None:
        messages = prepare_fixed_messages(W, prior_state, payload, ledger=ledger)
    if head is None:
        head = prepare_head(payload, ledger=ledger)
    if not isinstance(messages, FixedMessages) or not isinstance(head, PreparedHead):
        raise TypeError("Use this module's fixed-message/head preparation APIs")
    expected_owners = (W, prior_state, payload["msg_w"], payload["msg_b"])
    if len(messages.owners) != len(expected_owners) or any(a is not b for a, b in zip(messages.owners, expected_owners)):
        raise ValueError("Fixed messages belong to different input owners")
    if len(head.owners) != len(_HEAD_SHAPES) or any(a is not payload[name] for name, a in zip(_HEAD_SHAPES, head.owners)):
        raise ValueError("Prepared head belongs to a different payload owner")
    if len(messages.values) != 6:
        raise ValueError("Expected six fixed messages")
    for j, message in enumerate(messages.values):
        _array(message, f"fixed_message{j}", (n, MESSAGE_CHANNELS), "float32")
    P = head.values
    x64, dx64, d2x64 = (ops.cast(value, "float64", name+".astype(float64)") for name, value in (("x", x), ("dx", dx), ("d2x", d2x)))
    scales = ops.view(P["scales"], layer, "layer.scales")
    centers = ops.view(P["centers"], layer, "layer.centers")
    z = ops.binary("divide", ops.binary("subtract", x64, centers, "feature.center"), scales, "feature.normalize")
    dz = ops.binary("divide", dx64, scales, "feature.first_normalize")
    d2z = ops.binary("divide", d2x64, scales, "feature.second_normalize")
    z, dz, d2z = _clip_jet(z, dz, d2z, ledger, "features", diagnostics)
    g, dg, d2g = (ops.view(value, (slice(None), slice(5, 6)), label) for value, label in ((z, "g"), (dz, "dg"), (d2z, "d2g")))
    mean_messages, log_energies = messages.values[:3], messages.values[3:]
    mv = ops.concat([ops.binary("multiply", g, value, f"g_message{j}") for j, value in enumerate(mean_messages)] + list(log_energies), "message.value_concat")
    energy_zero = ops.zeros((n, MESSAGE_CHANNELS), "float64", "constant_energy_zero_jet")
    dm = ops.concat([ops.binary("multiply", dg, value, f"dg_message{j}") for j, value in enumerate(mean_messages)] + [energy_zero]*3, "message.first_concat")
    d2m = ops.concat([ops.binary("multiply", d2g, value, f"d2g_message{j}") for j, value in enumerate(mean_messages)] + [energy_zero]*3, "message.second_concat")
    mv, dm, d2m = _clip_jet(mv, dm, d2m, ledger, "messages", diagnostics)
    h, dh, d2h = (ops.concat(pair, label) for pair, label in (((z, mv), "encoder.value_concat"), ((dz, dm), "encoder.first_concat"), ((d2z, d2m), "encoder.second_concat")))
    for index in (0, 2):
        weight = ops.transpose(P[f"enc{index}_w"], f"enc{index}.weight.T")
        t = ops.binary("add", ops.matmul(h, weight, f"enc{index}.value_matmul"), P[f"enc{index}_b"], f"enc{index}.bias")
        dt = ops.matmul(dh, weight, f"enc{index}.first_matmul")
        d2t = ops.matmul(d2h, weight, f"enc{index}.second_matmul")
        h, dh, d2h = _silu_jet(t, dt, d2t, ledger, f"enc{index}.silu", diagnostics)
    weight = ops.transpose(P["enc4_w"], "enc4.weight.T")
    state = ops.binary("add", ops.matmul(h, weight, "state.value_matmul"), P["enc4_b"], "state.bias")
    dstate = ops.matmul(dh, weight, "state.first_matmul")
    d2state = ops.matmul(d2h, weight, "state.second_matmul")
    scale = ops.binary("multiply", ops.cast(sigma, "float64", "sigma.astype(float64)"), ops.view(P["rms"], layer, "layer.rms"), "head.scale")
    gain = ops.view(P["gain_multiplier"], layer, "layer.gain")
    head_vector = ops.view(P["head"], layer, "layer.head")
    response = ops.binary("add", ops.matmul(state, head_vector, "head.value_matvec"), ops.view(P["head_b"], layer, "layer.head_bias"), "head.bias")
    dresponse = ops.matmul(dstate, head_vector, "head.first_matvec")
    d2response = ops.matmul(d2state, head_vector, "head.second_matvec")
    delta, ddelta, d2delta = (ops.binary("multiply", ops.binary("multiply", scale, value, label+".scale"), gain, label+".gain") for value, label in ((response, "delta"), (dresponse, "ddelta"), (d2response, "d2delta")))
    diagnostics["nonfinite"].update(_nonfinite(ledger, dict(delta=delta, state=state, ddelta=ddelta, dstate=dstate, d2delta=d2delta, d2state=d2state)))
    diagnostics["message_preparation_nonfinite"] = dict(messages.nonfinite)
    diagnostics["head_preparation_nonfinite"] = dict(head.nonfinite)
    diagnostics["nonpositive_head_scales"] = head.nonpositive_scales
    diagnostics["all_finite"] = not any(diagnostics["nonfinite"].values()) and not any(messages.nonfinite.values()) and not any(head.nonfinite.values())
    preparation = dict(messages_reused=message_reused, head_reused=head_reused,
                       messages_cost=messages.cost, head_cost=head.cost,
                       qualification="Supplied preparation costs are excluded from this call's ledger; new preparations are included. Account each acquisition exactly once.")
    return SecondJet(delta, state, ddelta, dstate, d2delta, d2state, diagnostics, ledger.report(start), preparation)


def linear_subtotals(width):
    """Pure metadata arithmetic, excluding every listed non-GEMM obligation."""
    if type(width) is not int or width <= 0:
        raise ValueError("width must be a positive Python integer")
    one32 = width * (HIDDEN*(2*ENCODER_INPUT-1) + HIDDEN*(2*HIDDEN-1)
                     + STATE*(2*HIDDEN-1) + (2*STATE-1))
    return dict(value_first_second_linear_fp64=6*one32,
                derivative_only_linear_fp64=4*one32,
                fixed_six_message_products_fp32=6*width*MESSAGE_CHANNELS*(2*width-1),
                qualification="Dense linear subtotals only; preparation/source products, nonlinearities, copies, diagnostics and pair/feature jets are additional.")
