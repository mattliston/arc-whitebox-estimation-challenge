"""Inert native-point BLOCK lift and two diagonal input jets of current409f P.

This is an input-only NumPy diagnostic, not a participant or a new estimator.
No payload, network, capture, target, or model is loaded here.  Import performs
no array arithmetic.  The sole source import, on an explicit preparation,
authenticates the old fixed-FP32-message preparer and its analytical ledger.
The caller must authenticate supplied data and impose runtime/resource caps.

API
---
prepare(layer, sigma, W, prior_state, payload, *, messages=None, ledger=None)
    Snapshots the active layer's FP32 head parameters and fixed constants.
    Layers1..15 are supported; the special zero P head at layer0 is outside
    this API.  ``messages`` may only come from prepare_fixed_messages below.
step(prepared, x, dx, d2x, *, ledger=None)
    x is the actual FP32 feature matrix.  Row i of dx/d2x gives the first/
    second derivative of that row with respect to its own virtual mu_i.
    Returns native FP32 delta/state, FP64 jets, diagnostics, cost, and lift.
result.lift(x64, *, ledger=None)
    Evaluates the SAME saved-base extension on a supplied FP64 feature
    matrix; returns LiftEvaluation(delta64, state64, cost, diagnostics).

Uniform block definition
------------------------
The native primal preserves the current FP32 expression order: feature
cast, subtract then divide; clip; three gated fixed messages plus three
fixed log energies; clip/concatenate; two affine+bias / SiLU stages;
state affine+bias; and (sigma32*rms32)*(state@head+bias)*gain32.

For each affine block, with native input x0 and output y0, the lift is
    y0 + A64 (x-x0).
Normalization is ONE affine block with the original FP32 scales; gated
message products are affine with the original FP32 fixed messages.  Each
encoder matmul+bias is one affine block.  The final affine head includes
its bias and the FIXED native FP32 sigma*rms and gain constants.  Constants
are not silently recomputed in FP64 to redefine the native base operator.

SiLU is explicitly an ATOMIC COMPOSITE block, not operation-level AD:
    y0 + (silu64(t)-silu64(t0)),
with ideal SiLU derivatives evaluated at the NATIVE preactivation t0.
Clip is the anchored mathematical clip of its native-point input.  Its
first/second jets are zero outside AND at +/-8; exact boundary counts are
reported.  A boundary is NOT a claim of classical differentiability.

All block bases remain fixed in lift evaluations.  Fresh perturbed FP32
executions are NOT finite differences of this lift.  It is neither the old
global FP64 shadow nor a literal derivative of the rounded served program.
Every anchor uses the same paid zero-increment selection to preserve the
saved base's signed-zero bits.  This is the identical real-valued map, not
a feature-dependent change or numerical repair of nonzero increments.
There are no per-feature selections, offsets fitted to targets, or altered
head parameters.  delta is the learned correction ONLY; the caller owns
physical-host jets, mean addition, and any final served maximum/cast.

Preparation is paid and reusable only for its declared fixed frame.  The
primal/jet call excludes preparation; supplied message preparation retains
its own mandatory receipt.  The ledger prices every numerical reference
operation, including snapshot casts, saved bases, clips, diagnostics and
lift evaluations, using the pinned FNP.12 ordinary-dense tariffs.  It is
an analytical diagnostic ledger, NOT a complete native/deployed bill.
Gross allocation/output bytes are not peak live memory.  Source hashing
and Python shape/ownership metadata have no array-arithmetic tariff.

Prepared arrays, fixed messages, and saved lift bases are caller-immutable
for their lifetime.  Original caller arrays are never modified.  Returned
native FP32 endpoint bytes still require independent current-source tests.
"""

from dataclasses import dataclass, field
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np


CURRENT_SOURCE_SHA256 = "409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f"
MESSAGE_SOURCE = (Path(__file__).resolve().parent.parent /
                  "current_response_quadratic_v1/gnn_second_jet_reference.py")
MESSAGE_SOURCE_SHA256 = "2ce74c478040d456078091751520d3a845b2fc8ca47acd6269d8fd8d255d8b21"
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
_SHARED = None


def _shared():
    """Import definitions only, after authenticating the exact reused source."""
    global _SHARED
    if _SHARED is None:
        if hashlib.sha256(MESSAGE_SOURCE.read_bytes()).hexdigest() != MESSAGE_SOURCE_SHA256:
            raise RuntimeError("Pinned FP32-message/ledger source changed")
        name = "_current_rounding_pinned_message_reference_v1"
        spec = importlib.util.spec_from_file_location(name, MESSAGE_SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        if module._HEAD_SHAPES != _HEAD_SHAPES or module.FNP_TARIFF_SHA256 != FNP_TARIFF_SHA256:
            raise RuntimeError("Pinned message reference metadata changed")
        _SHARED = module
    return _SHARED


def _ledger(value):
    return _shared().CostLedger() if value is None else value


def prepare_fixed_messages(W, prior_state, payload, *, ledger=None):
    """The exact pinned FP32 preparation; its receipt is never free reuse."""
    return _shared().prepare_fixed_messages(W, prior_state, payload, ledger=ledger)


@dataclass(frozen=True)
class Prepared:
    layer: int
    width: int
    native: dict[str, Any] = field(repr=False)
    constants64: dict[str, Any] = field(repr=False)
    messages: Any = field(repr=False)
    cost: dict[str, Any] = field(repr=False)
    preparation: dict[str, Any]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _ClipBase:
    before64: np.ndarray = field(repr=False)
    after64: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class _AffineBase:
    before64: np.ndarray = field(repr=False)
    after64: np.ndarray = field(repr=False)
    weight64: np.ndarray = field(repr=False)  # row-vector orientation, already transposed


@dataclass(frozen=True)
class _SiluBase:
    before64: np.ndarray = field(repr=False)
    after64: np.ndarray = field(repr=False)
    ideal_before64: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class LiftEvaluation:
    delta: np.ndarray = field(repr=False)
    state: np.ndarray = field(repr=False)
    cost: dict[str, Any] = field(repr=False)
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class NativeJet:
    delta: np.ndarray = field(repr=False)
    state: np.ndarray = field(repr=False)
    ddelta: np.ndarray = field(repr=False)
    dstate: np.ndarray = field(repr=False)
    d2delta: np.ndarray = field(repr=False)
    d2state: np.ndarray = field(repr=False)
    diagnostics: dict[str, Any]
    cost: dict[str, Any] = field(repr=False)
    preparation: dict[str, Any]
    lift: Any = field(repr=False)


def prepare(layer, sigma, W, prior_state, payload, *, messages=None, ledger=None):
    """Prepare one immutable native frame; do not load or infer input data.

    Reused messages must be created by this module's forwarding helper and
    retain the identical W/state/message-parameter owners.  A future fused
    reuse of internal participant messages is NOT implemented or credited.
    Head parameter snapshots are charged even though they preserve all bits.
    """
    if type(layer) is not int or not 1 <= layer < DEPTH:
        raise ValueError("This active-head API supports Python integer layers1..15")
    old = _shared()
    ledger = _ledger(ledger)
    start = len(ledger.records)
    n = old._array(W, "W").shape[0]
    old._array(W, "W", (n, n))
    old._array(sigma, "sigma", (n,))
    old._array(prior_state, "prior_state", (n, STATE), "float32")
    originals = {name: old._array(payload[name], name, shape, "float32")
                 for name, shape in _HEAD_SHAPES.items()}
    old._array(payload["msg_w"], "msg_w", (3*MESSAGE_CHANNELS, STATE), "float32")
    old._array(payload["msg_b"], "msg_b", (3*MESSAGE_CHANNELS,), "float32")
    reused = messages is not None
    if messages is None:
        messages = prepare_fixed_messages(W, prior_state, payload, ledger=ledger)
    if not isinstance(messages, old.FixedMessages):
        raise TypeError("Use this module's pinned prepare_fixed_messages wrapper")
    owners = (W, prior_state, payload["msg_w"], payload["msg_b"])
    if len(messages.owners) != len(owners) or any(a is not b for a, b in zip(messages.owners, owners)):
        raise ValueError("Fixed messages belong to a different native frame")
    if len(messages.values) != 6:
        raise ValueError("Six fixed FP32 message arrays are required")
    for i, value in enumerate(messages.values):
        old._array(value, f"message{i}", (n, MESSAGE_CHANNELS), "float32")

    ops = old._Ops(ledger, "native_preparation")
    p = {name: ops.cast(value, "float32", name+".native_snapshot")
         for name, value in originals.items()}
    sigma32 = ops.cast(sigma, "float32", "sigma.astype(float32)")
    scale32 = ops.binary("multiply", sigma32, ops.view(p["rms"], layer, "layer.rms"), "native.sigma_times_rms")
    gain32 = ops.binary("add", 1, ops.binary("multiply", .25,
        ops.unary("tanh", ops.view(p["gain"], layer, "layer.gain"), "native.gain_tanh"),
        "native.gain_quarter"), "native.gain_plus1")
    if scale32.dtype != np.dtype("float32") or gain32.dtype != np.dtype("float32"):
        raise TypeError("Native sigma/rms/gain arithmetic must remain FP32")
    native = dict(payload=p, sigma=sigma32, scale=scale32, gain=gain32)

    wide = old._Ops(ledger, "constant_fp64_preparation")
    constants = {
        "scales": wide.cast(wide.view(p["scales"], layer, "layer.scales"), "float64", "layer.scales64"),
        "scale": wide.cast(scale32, "float64", "native.scale64"),
        "gain": wide.cast(gain32, "float64", "native.gain64"),
        "head": wide.cast(wide.view(p["head"], layer, "layer.head"), "float64", "layer.head64"),
    }
    for index in (0, 2, 4):
        weight = wide.cast(p[f"enc{index}_w"], "float64", f"enc{index}.weight64")
        constants[f"weight{index}"] = wide.transpose(weight, f"enc{index}.weight64.T")
    constants["mean_messages"] = tuple(wide.cast(value, "float64", f"fixed_message{i}.64")
                                       for i, value in enumerate(messages.values[:3]))
    nonfinite = old._nonfinite(ledger, {
        **{"native."+k: v for k, v in p.items()},
        "sigma": sigma32, "scale": scale32, "gain": gain32,
        **{"constant."+k: v for k, v in constants.items() if k != "mean_messages"},
        **{f"message64.{i}": v for i, v in enumerate(constants["mean_messages"])},
    })
    diag = old._Ops(ledger, "diagnostics")
    nonpositive = diag.count(diag.binary("less_equal", constants["scales"], 0.0,
                                        "layer.scales<=0"), "nonpositive_layer_scale_count")
    diagnostics = dict(nonfinite=nonfinite, message_nonfinite=dict(messages.nonfinite),
                       nonpositive_layer_scales=nonpositive,
                       all_finite=not any(nonfinite.values()) and not any(messages.nonfinite.values()))
    preparation = dict(
        current_source_sha256=CURRENT_SOURCE_SHA256,
        message_preparer_source=dict(path=str(MESSAGE_SOURCE), sha256=MESSAGE_SOURCE_SHA256,
                                     function="prepare_fixed_messages"),
        tariff_sha256=FNP_TARIFF_SHA256, numpy_version=np.__version__,
        native_messages_reused=reused, message_preparation_cost=messages.cost,
        message_cost_in_this_preparation=not reused,
        head_parameters_snapshotted=True, fixed_sigma_rms_gain_native_fp32=True,
        same_message_input_owners_verified=True,
        caller_input_authentication_required=True,
        qualification=("Include this preparation once. If messages were supplied, include their "
                       "separate preparation receipt once too; no fused participant reuse is claimed."),
    )
    return Prepared(layer, n, native, constants, messages, ledger.report(start), preparation, diagnostics)


def _clip_native(value32, first64, second64, ledger, label, diagnostics):
    old = _shared()
    primal = old._Ops(ledger, "native_primal")
    tangent = old._Ops(ledger, "jet")
    diag = old._Ops(ledger, "diagnostics")
    inside = diag.binary("logical_and", diag.binary("greater", value32, -8, label+">-8"),
                         diag.binary("less", value32, 8, label+"<8"), label+".interior")
    endpoints = diag.binary("logical_or", diag.binary("equal", value32, -8, label+"==-8"),
                            diag.binary("equal", value32, 8, label+"==8"), label+".endpoints")
    diagnostics["clip_endpoint_counts"][label] = diag.count(endpoints, label+".endpoint_count")
    output = primal.clip(value32, label+".clip32")
    d1 = tangent.where_zero(inside, first64, label+".first_mask")
    d2 = tangent.where_zero(inside, second64, label+".second_mask")
    save = old._Ops(ledger, "saved_lift_bases")
    base = _ClipBase(save.cast(value32, "float64", label+".before64"),
                     save.cast(output, "float64", label+".after64"))
    diagnostics["nonfinite"].update(old._nonfinite(ledger, {
        label+".before": value32, label+".first": first64, label+".second": second64,
        label+".after": output,
    }))
    return output, d1, d2, base


def _ideal_silu(value64, ops, label, *, derivatives):
    denominator = ops.binary("add", 1, ops.unary("exp", ops.unary("negative", value64,
        label+".negative"), label+".exp"), label+".denominator")
    value = ops.binary("divide", value64, denominator, label+".value")
    if not derivatives:
        return value, dict(denominator=denominator)
    sigmoid = ops.binary("divide", 1.0, denominator, label+".sigmoid")
    q = ops.binary("multiply", sigmoid, ops.binary("subtract", 1.0, sigmoid,
                   label+".one_minus_sigmoid"), label+".q")
    first = ops.binary("add", sigmoid, ops.binary("multiply", value64, q,
                       label+".tq"), label+".first_response")
    second = ops.binary("multiply", q, ops.binary("add", 2.0,
        ops.binary("multiply", value64, ops.binary("subtract", 1.0,
            ops.binary("multiply", 2.0, sigmoid, label+".two_sigmoid"),
            label+".one_minus_two_sigmoid"), label+".curvature_inner"),
        label+".curvature_plus2"), label+".second_response")
    return value, first, second, dict(denominator=denominator, sigmoid=sigmoid)


def _silu_native(value32, first64, second64, before64, ledger, label, diagnostics):
    old = _shared()
    primal = old._Ops(ledger, "native_primal")
    denominator32 = primal.binary("add", 1, primal.unary("exp",
        primal.unary("negative", value32, label+".native_negative"),
        label+".native_exp"), label+".native_denominator")
    output = primal.binary("divide", value32, denominator32, label+".native_value")
    tangent = old._Ops(ledger, "jet")
    ideal, fp, fpp, internals = _ideal_silu(before64, tangent, label+".atomic64", derivatives=True)
    d1 = tangent.binary("multiply", fp, first64, label+".first")
    d2 = tangent.binary("add", tangent.binary("multiply", fpp,
        tangent.binary("multiply", first64, first64, label+".first_square"),
        label+".quadratic_chain"), tangent.binary("multiply", fp, second64,
        label+".linear_chain"), label+".second")
    save = old._Ops(ledger, "saved_lift_bases")
    base = _SiluBase(before64, save.cast(output, "float64", label+".native_after64"), ideal)
    diagnostics["nonfinite"].update(old._nonfinite(ledger, {
        label+".native_input": value32, label+".native_denominator": denominator32,
        label+".native_output": output, label+".ideal_value": ideal,
        label+".first_response": fp, label+".second_response": fpp,
        **{label+".ideal_"+k: v for k, v in internals.items()},
        label+".first": d1, label+".second": d2,
    }))
    return output, d1, d2, base


def _anchor(ops, base, increment, label):
    """Uniform base+increment, preserving native signed zeros at zero change."""
    summed = ops.binary("add", base, increment, label+".add")
    zero = ops.binary("equal", increment, 0.0, label+".zero_increment")
    value = np.where(zero, base, summed)
    return ops.ledger.record("where", label+".preserve_zero_base", value,
                             value.size, value.dtype, ops.phase)


@dataclass(frozen=True)
class SavedLift:
    prepared: Prepared = field(repr=False)
    input64: np.ndarray = field(repr=False)
    normalization: _ClipBase = field(repr=False)
    message_clip: _ClipBase = field(repr=False)
    affine: tuple[_AffineBase, ...] = field(repr=False)
    silu: tuple[_SiluBase, ...] = field(repr=False)
    delta64: np.ndarray = field(repr=False)
    native_boundary_counts: dict[str, int]

    def __call__(self, x64, *, ledger=None):
        """Evaluate frozen native block bases, never a fresh FP32 perturbation."""
        old = _shared()
        old._array(x64, "lift.x64", (self.prepared.width, FEATURES), "float64")
        ledger = _ledger(ledger)
        start = len(ledger.records)
        ops = old._Ops(ledger, "frozen_block_lift")
        c = self.prepared.constants64
        nonfinite = old._nonfinite(ledger, {"lift.input": x64})

        def clip(value, base, label):
            # The native clip base is itself exact at FP32 +/-8.  Written as
            # an anchored block so the same convention is explicit throughout.
            current = ops.clip(value, label+".clip")
            reference = ops.clip(base.before64, label+".base_clip")
            return _anchor(ops, base.after64, ops.binary("subtract", current,
                           reference, label+".increment"), label+".anchored")

        z = _anchor(ops, self.normalization.before64,
            ops.binary("divide", ops.binary("subtract", x64, self.input64,
                       "normalization.input_increment"), c["scales"],
                       "normalization.affine_increment"), "normalization.anchored")
        z = clip(z, self.normalization, "features")
        g = ops.view(z, (slice(None), slice(5, 6)), "current.g")
        g0 = ops.view(self.normalization.after64, (slice(None), slice(5, 6)), "native.g")
        dg = ops.binary("subtract", g, g0, "g.increment")
        zero = ops.zeros((self.prepared.width, MESSAGE_CHANNELS), "float64", "fixed_energy.zero_increment")
        increment = ops.concat([ops.binary("multiply", dg, m, f"message{i}.increment")
                                for i, m in enumerate(c["mean_messages"])] + [zero]*3,
                                "message.increment_concat")
        msg = _anchor(ops, self.message_clip.before64, increment, "message.affine_anchor")
        msg = clip(msg, self.message_clip, "messages")
        h = ops.concat([z, msg], "encoder.input")
        for i, block in enumerate(self.affine):
            difference = ops.binary("subtract", h, block.before64, f"affine{i}.input_increment")
            h = _anchor(ops, block.after64, ops.matmul(difference, block.weight64,
                        f"affine{i}.increment_product"), f"affine{i}.anchor")
            nonfinite.update(old._nonfinite(ledger, {f"lift.affine{i}": h}))
            if i < len(self.silu):
                sb = self.silu[i]
                value, intermediate = _ideal_silu(h, ops, f"silu{i}.current", derivatives=False)
                h = _anchor(ops, sb.after64, ops.binary("subtract", value,
                    sb.ideal_before64, f"silu{i}.ideal_increment"), f"silu{i}.anchor")
                nonfinite.update(old._nonfinite(ledger, {f"lift.silu{i}": h,
                    **{f"lift.silu{i}."+k: v for k, v in intermediate.items()}}))
        state = h
        difference = ops.binary("subtract", state, self.affine[-1].after64, "head.state_increment")
        increment = ops.matmul(difference, c["head"], "head.increment_matvec")
        increment = ops.binary("multiply", ops.binary("multiply", c["scale"], increment,
                               "head.increment_scale"), c["gain"], "head.increment_gain")
        delta = _anchor(ops, self.delta64, increment, "head.anchor")
        nonfinite.update(old._nonfinite(ledger, {"lift.delta": delta, "lift.state": state}))
        diagnostics = dict(nonfinite=nonfinite, all_finite=not any(nonfinite.values()),
            native_clip_endpoint_counts=dict(self.native_boundary_counts),
            interpretation="Saved native-point block lift; bases never recomputed at perturbed x.")
        return LiftEvaluation(delta, state, ledger.report(start), diagnostics)


def step(prepared, x, dx, d2x, *, ledger=None):
    """Reproduce native active-P primals and return two FP64 diagonal jets.

    The preparation and this step are separate paid acquisitions.  The
    returned lift uses copied native bases, not old FP64-shadow values.
    No test, finite difference, payload decoding, or target scoring occurs.
    """
    if not isinstance(prepared, Prepared):
        raise TypeError("Use this module's prepare API")
    old = _shared()
    n, layer = prepared.width, prepared.layer
    old._array(x, "x", (n, FEATURES), "float32")
    old._array(dx, "dx", (n, FEATURES))
    old._array(d2x, "d2x", (n, FEATURES))
    ledger = _ledger(ledger)
    start = len(ledger.records)
    native = old._Ops(ledger, "native_primal")
    jet = old._Ops(ledger, "jet")
    save = old._Ops(ledger, "saved_lift_bases")
    p, c = prepared.native["payload"], prepared.constants64
    diagnostics = dict(clip_endpoint_counts={}, nonfinite={},
        zero_endpoint_derivative_convention=True, silu_granularity="atomic_composite",
        interpretation="Native-point block lift; not operation-level or literal rounded-program AD.")

    x32 = native.cast(x, "float32", "x.astype(float32)")
    x0 = save.cast(x32, "float64", "input.native_base64")
    dx64, d2x64 = (jet.cast(value, "float64", name+".astype(float64)")
                   for name, value in (("dx", dx), ("d2x", d2x)))
    z32 = native.binary("divide", native.binary("subtract", x32,
        native.view(p["centers"], layer, "layer.centers"), "feature.center32"),
        native.view(p["scales"], layer, "layer.scales"), "feature.normalize32")
    dz = jet.binary("divide", dx64, c["scales"], "feature.first_normalization")
    d2z = jet.binary("divide", d2x64, c["scales"], "feature.second_normalization")
    z32, dz, d2z, normalization = _clip_native(z32, dz, d2z, ledger, "features", diagnostics)
    g32 = native.view(z32, (slice(None), slice(5, 6)), "native.g")
    dg = jet.view(dz, (slice(None), slice(5, 6)), "first.g")
    d2g = jet.view(d2z, (slice(None), slice(5, 6)), "second.g")
    msg32 = native.concat([native.binary("multiply", g32, m, f"g_message{i}.32")
                          for i, m in enumerate(prepared.messages.values[:3])]
                         + list(prepared.messages.values[3:]), "native.message_concat")
    zero = jet.zeros((n, MESSAGE_CHANNELS), "float64", "fixed_energy.zero_jets")
    dmsg = jet.concat([jet.binary("multiply", dg, m, f"first.message{i}")
                      for i, m in enumerate(c["mean_messages"])] + [zero]*3,
                      "first.message_concat")
    d2msg = jet.concat([jet.binary("multiply", d2g, m, f"second.message{i}")
                       for i, m in enumerate(c["mean_messages"])] + [zero]*3,
                       "second.message_concat")
    msg32, dmsg, d2msg, message_clip = _clip_native(msg32, dmsg, d2msg, ledger,
                                                  "messages", diagnostics)
    h32 = native.concat([z32, msg32], "native.encoder_concat")
    dh = jet.concat([dz, dmsg], "first.encoder_concat")
    d2h = jet.concat([d2z, d2msg], "second.encoder_concat")
    base_input64 = save.cast(h32, "float64", "encoder.native_input64")
    affines, silus = [], []
    for index in (0, 2, 4):
        t32 = native.binary("add", native.matmul(h32,
            native.transpose(p[f"enc{index}_w"], f"native.enc{index}.weight.T"),
            f"native.enc{index}.matmul"), p[f"enc{index}_b"], f"native.enc{index}.bias")
        dt = jet.matmul(dh, c[f"weight{index}"], f"first.enc{index}.matmul")
        d2t = jet.matmul(d2h, c[f"weight{index}"], f"second.enc{index}.matmul")
        t0 = save.cast(t32, "float64", f"enc{index}.native_output64")
        affines.append(_AffineBase(base_input64, t0, c[f"weight{index}"]))
        diagnostics["nonfinite"].update(old._nonfinite(ledger,
            {f"enc{index}.native_affine": t32, f"enc{index}.first_affine": dt,
             f"enc{index}.second_affine": d2t}))
        if index != 4:
            h32, dh, d2h, sb = _silu_native(t32, dt, d2t, t0, ledger,
                                           f"enc{index}.silu", diagnostics)
            silus.append(sb)
            base_input64 = sb.after64
        else:
            h32, dh, d2h = t32, dt, d2t
    state32, dstate, d2state = h32, dh, d2h
    response32 = native.binary("add", native.matmul(state32,
        native.view(p["head"], layer, "native.layer.head"), "native.head.matvec"),
        native.view(p["head_b"], layer, "native.layer.head_bias"), "native.head.bias")
    delta32 = native.binary("multiply", native.binary("multiply", prepared.native["scale"],
        response32, "native.head.scale"), prepared.native["gain"], "native.head.gain")
    first = jet.matmul(dstate, c["head"], "first.head.matvec")
    second = jet.matmul(d2state, c["head"], "second.head.matvec")
    ddelta = jet.binary("multiply", jet.binary("multiply", c["scale"], first,
                        "first.head.scale"), c["gain"], "first.head.gain")
    d2delta = jet.binary("multiply", jet.binary("multiply", c["scale"], second,
                         "second.head.scale"), c["gain"], "second.head.gain")
    old._array(delta32, "native.delta", (n,), "float32")
    old._array(state32, "native.state", (n, STATE), "float32")
    for name, value in (("ddelta", ddelta), ("d2delta", d2delta)):
        old._array(value, name, (n,), "float64")
    for name, value in (("dstate", dstate), ("d2state", d2state)):
        old._array(value, name, (n, STATE), "float64")
    diagnostics["nonfinite"].update(old._nonfinite(ledger, dict(delta=delta32, state=state32,
        ddelta=ddelta, dstate=dstate, d2delta=d2delta, d2state=d2state)))
    diagnostics["preparation"] = prepared.diagnostics
    diagnostics["all_finite"] = not any(diagnostics["nonfinite"].values()) and prepared.diagnostics["all_finite"]
    diagnostics["nonpositive_layer_scales"] = prepared.diagnostics["nonpositive_layer_scales"]
    lift = SavedLift(prepared, x0, normalization, message_clip, tuple(affines), tuple(silus),
        save.cast(delta32, "float64", "native.delta_base64"), dict(diagnostics["clip_endpoint_counts"]))
    preparation = dict(prepared.preparation, preparation_cost=prepared.cost,
        preparation_in_step_cost=False,
        lift_call_cost_in_step=False,
        qualification=("This step includes native head replay, two jets and saved bases; "
                       "preparation and every lift evaluation have separate paid receipts."))
    return NativeJet(delta32, state32, ddelta, dstate, d2delta, d2state,
                     diagnostics, ledger.report(start), preparation, lift)


def linear_subtotals(width):
    """Pure source arithmetic only; not a complete native/diagnostic price."""
    if type(width) is not int or width <= 0:
        raise ValueError("width must be a positive Python integer")
    one32 = width * (HIDDEN*(2*ENCODER_INPUT-1) + HIDDEN*(2*HIDDEN-1)
                     + STATE*(2*HIDDEN-1) + (2*STATE-1))
    return dict(native_head_linear=one32, two_fp64_linear_jets=4*one32,
        native_plus_two_jet_linear=5*one32, one_frozen_lift_linear=2*one32,
        fixed_six_message_products_fp32=6*width*MESSAGE_CHANNELS*(2*width-1),
        qualification=("Linear subtotals only. Message-source projection, bias, powers, logs, "
                       "native and ideal activations, clipping, normalization, snapshots, "
                       "saved bases, all diagnostics and feature/pair construction are additional."))
