"""Inert, target-free direction for the fixed native entry8 -> mean9 test.

The only changed parameter is the existing shared enc4_w.  The fixed entry8
is not differentiated through its prefix.  Applying this same P direction at
steps8 and9 tests a local two-step Jacobian, not a full-rollout gradient.

No numerical library is imported.  Native operations run only in perturb(),
through the caller's already-metered public FNP surface.  Torch construction
runs only in the explicitly caller-owned adjoint scope.  Nothing is cached.
"""
import hashlib
import struct


PARAMETER_KEY = 'enc4_w'
PARAMETER_SHAPE = (64, 128)
LAYERS = (8, 9)
ARMS = ('N', 'plus', 'minus')
EPSILON = 1.0 / 8.0
U = tuple((1.0 if (i // 4) % 2 == 0 else -1.0) / 8.0 for i in range(64))
V = tuple((1.0 if (j // 16) % 2 == 0 else -1.0) / 2.0 for j in range(128))
DIRECTION = tuple(tuple(u * v for v in V) for u in U)
COTANGENT_SHAPE = (1024,)
COTANGENT_VALUE = 1.0 / 32.0
COTANGENT = (COTANGENT_VALUE,) * 1024
# Exact binary-rational source identities, not measured estimates.
DIRECTION_FROBENIUS_SQUARED = 32.0
COTANGENT_FROBENIUS_SQUARED = 1.0
REQUESTED_INCREMENT_ABS = 1.0 / 128.0

# Prospective diagnostic constants.  The owning worker must record every
# predicate, rather than converting a selector/noise failure into agreement.
SECANT_ABSOLUTE_TOLERANCE = 2e-6
SECANT_RELATIVE_TOLERANCE = 0.05
MINIMUM_SECANT_MAGNITUDE = 4e-5
MINIMUM_CARRIED_FRACTION = 0.10
MAXIMUM_ONE_SIDED_CURVATURE_RATIO = 0.10


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def source_proof():
    """Stdlib scalar/constant metadata only; no source or data execution."""
    flat = tuple(x for row in DIRECTION for x in row)
    return dict(schema='current_native_trajectory_fixed_direction_v1',
        parameter_key=PARAMETER_KEY, parameter_shape=list(PARAMETER_SHAPE),
        layers=list(LAYERS), epsilon=EPSILON, arms=list(ARMS),
        factor_u=list(U), factor_v=list(V), exact_rank_one=True,
        both_factors_balanced=True, direction_frobenius_squared=32.0,
        direction_fp32_sha256=_sha(struct.pack('<8192f', *flat)),
        requested_increment_abs=REQUESTED_INCREMENT_ABS,
        cotangent_shape=list(COTANGENT_SHAPE), cotangent_value=COTANGENT_VALUE,
        cotangent_frobenius_squared=1.0,
        cotangent_fp64_sha256=_sha(struct.pack('<1024d', *COTANGENT)),
        changed_keys=[PARAMETER_KEY], unchanged_bias_head_gain=True,
        unchanged_NODE_A_and_cstar=True, fixed_entry8=True,
        full_prefix_derivative_qualified=False, no_adaptive_step_or_direction=True,
        native_parameter_addition_is_FP32=True,
        real_algebra_derivative_not_derivative_of_rounding=True,
        complete_public_FNP_accounting_required=True,
        absolute_tolerance=SECANT_ABSOLUTE_TOLERANCE,
        relative_tolerance=SECANT_RELATIVE_TOLERANCE,
        minimum_secant_magnitude=MINIMUM_SECANT_MAGNITUDE,
        minimum_carried_fraction=MINIMUM_CARRIED_FRACTION,
        maximum_one_sided_curvature_ratio=MAXIMUM_ONE_SIDED_CURVATURE_RATIO)


def perturb(P, fnp, arm):
    """Mutate only P['enc4_w']; return array-free provenance metadata.

    The caller owns a fresh P dictionary, fresh native module and metered
    context for each arm.  N leaves every parameter object and byte intact.
    +/- use a typed length-one FP32 step, never an RPC weak Python scalar.
    Copies/constant ingress, checks, arithmetic and byte exports are public
    FNP operations inside that same complete context.  No array escapes in
    the returned record.  All temporary arrays die when this call returns.
    """
    require(arm in ARMS and type(P) is dict, 'Declared arm and fresh P mapping')
    require(PARAMETER_KEY in P, 'Existing current encoder matrix')
    old = P[PARAMETER_KEY]
    require(tuple(old.shape) == PARAMETER_SHAPE and old.dtype == fnp.float32,
        'Exact current FP32 enc4_w geometry')
    identities = {key: id(value) for key, value in P.items()}
    require(bool(fnp.all(fnp.isfinite(old))), 'Finite current encoder')
    before = old.tobytes()
    direction = step = delta = changed = None
    try:
        if arm != 'N':
            direction = fnp.array(DIRECTION, dtype=fnp.float32)
            step = fnp.array([EPSILON if arm == 'plus' else -EPSILON], dtype=fnp.float32)
            delta = direction * step
            changed = old + delta
            require(tuple(changed.shape) == PARAMETER_SHAPE and changed.dtype == fnp.float32
                and bool(fnp.all(fnp.isfinite(changed))), 'Finite same-shape FP32 perturbation')
            P[PARAMETER_KEY] = changed
        after = P[PARAMETER_KEY].tobytes()
        require(set(P) == set(identities) and all(id(P[key]) == value
            for key, value in identities.items() if key != PARAMETER_KEY),
            'All other original parameter owners retained exactly')
        require((arm != 'N' and after != before) or
            (arm == 'N' and id(P[PARAMETER_KEY]) == identities[PARAMETER_KEY] and after == before),
            'Nonzero declared perturbation or exact N identity')
        return dict(arm=arm, parameter_key=PARAMETER_KEY,
            parameter_shape=list(PARAMETER_SHAPE), epsilon=EPSILON,
            signed_epsilon=0.0 if arm == 'N' else (EPSILON if arm == 'plus' else -EPSILON),
            before_sha256=_sha(before), after_sha256=_sha(after),
            requested_direction_sha256=source_proof()['direction_fp32_sha256'],
            other_parameter_owners_unchanged=True, no_array_retained=True,
            actual_complete_context_owns_all_operations=True,
            unchanged_identity=(arm == 'N'), target_accessed=False)
    finally:
        direction = step = delta = changed = old = None


def torch_direction(torch, P):
    """Same exact D, with zero directions for every other supplied P key."""
    require(type(P) is dict and PARAMETER_KEY in P, 'Explicit admitted Torch P')
    old = P[PARAMETER_KEY]
    require(tuple(old.shape) == PARAMETER_SHAPE and old.dtype == torch.float32
        and old.device.type == 'cpu', 'Production FP32 CPU parameter geometry')
    result = {key: torch.zeros_like(value, requires_grad=False) for key, value in P.items()}
    result[PARAMETER_KEY] = torch.tensor(DIRECTION, dtype=torch.float32, device=old.device)
    return result


def cotangent(torch):
    """Fixed mean9 cotangent; source-exact FP64, unit Euclidean norm."""
    return torch.full(COTANGENT_SHAPE, COTANGENT_VALUE, dtype=torch.float64, device='cpu')
