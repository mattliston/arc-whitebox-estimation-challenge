"""Inert v29 feature-capture source adapter; no estimator runs on import.

Public source: 504aldo, commit 18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45.
MIT License, Copyright (c) 2026 504aldo. The complete notice is in PROTOCOL.md
and the authenticated repository/LICENSE; retain it with any derived source.
This file implements no learned observer, fitting, target reader or stage.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import types

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
PUBLIC = NEW / 'public_504aldo_20260912_v1/repository'
OWNER = PUBLIC / 'estimators/estimator_v29.py'
OWNER_SHA = '86d9ca9b28e6fe2b6c74750a0b6ae4bba4c14f742ddc3f5f56bc7fb4ec9d8e27'
LICENSE = PUBLIC / 'LICENSE'
LICENSE_SHA = 'd2c9a83a86ed95c8d279219cd36d5a025c0372dee57d072fe4718850e226e740'
ROSTER_OWNER = NEW / 'current_comp_joint_p_node_v1/TRAIN_PROTOCOL.md'
ROSTER_SHA = '53cb4c490f0a474e12fb35d18f1ecfa59f229ea0cb9c0ef8442a2ddf7d3bee16'
TARIFF_OWNER = NEW.parent / 'data_gate/.venv/lib/python3.11/site-packages/flopscope/data/default_weights.json'
TARIFF_SHA = 'f729bc47f07c080b8c7b9d7b045e5663f0af313669b4c2a962b5a9044a94cb7a'
TRAIN = tuple(r for r in range(35, 59) if r not in (37, 39, 42, 48))
CHECK = (37, 39, 42, 48)

# Every environment read in the pinned owner is resolved to one reviewed value.
# FIRST=5 is the sole non-default choice; LEVELS=5 and NO_CORR=1 are explicit.
FIXED_ENV = {
    'V17_NO_FEED': '0', 'V17_NO_WK431': '0', 'V17_NO_REGEN': '0',
    'V18_NO_FB': '0', 'V19_FULL_LAST': '0', 'V21_NO_CONFINE': '0',
    'V21_QPASS': '1', 'V19_NO_SRC_LAST': '0', 'V17_NO_CORR': '1',
    'V17_LAM_SCALE': '0.95', 'V25_BETA': '1.0', 'V26_STRASSEN': '5',
    'V26_STRASSEN_HUB': '5', 'V26_STRASSEN_NEW': '0', 'V29_CPRE_LEV': '5',
    'V28_STRASSEN_SB': '5', 'V26_STRASSEN_MIN': '32',
    'V28_STRASSEN_FIRST': '5', 'V28_STRASSEN_FUSE_P': '343',
    'V26_WARM': '1', 'V21_AGE_OLD': '4', 'V21_R_OLD': '384',
    'V24_AGE_OLD2': '7', 'V24_R_OLD2': '224', 'V24_QPASS2': '2',
    'V17_DEBUG': '0',
}

FEATURE_NAMES = (
    'layer_symmetric', 'layer_fraction', 'one', 'alpha', 'centered_log_sigma',
    'gate', 'pdf', 'gaussian_mean_over_sigma', 'host_over_sigma',
    'host_minus_gaussian_over_sigma', 'standardized_k3', 'standardized_k4',
    'absolute_alpha', 'rho_row_mean', 'rho_row_rms', 'rho_row_absolute_mean',
    'weight_sum_standardized', 'weight_square_sum_standardized',
    'weight_cube_sum_standardized', 'weight_fourth_sum_standardized',
    'skew_plane_row_norm', 'skew_plane_column_norm',
    'skew_plane_row_sum_over_sqrt_n', 'skew_plane_column_sum_over_sqrt_n',
    'rho_alpha_action_over_n', 'skew_plane_row_absolute_mean',
    'skew_plane_column_absolute_mean', 'log1p_absolute_standardized_k3',
    'log1p_absolute_standardized_k4', 'network_mean_host_over_sigma',
)
TERMINAL_ZERO_SLOTS = (13, 14, 15, 20, 21, 22, 23, 24, 25, 26)
SKEW_ZERO_FIRST_SLOTS = (10, 11, 20, 21, 22, 23, 25, 26, 27, 28)

ANCHOR = b'            pk1v, pk2v, pk3v, pk4v = PK1[0], PK1[1], PK1[2], PK1[3]\n'
INSERT = b'''
            # Read-only capture: no original state/view is handed to the sink.
            _v29_sink = getattr(self, "_v29_temporal_sink", None)
            if _v29_sink is not None:
                _v29_temporal_emit(_v29_sink, li, last, mu, var, sigma, alpha,
                                   phi, Phi, D3, g4row, pk1v, C_off, D21, w32)
            del _v29_sink
'''


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def authenticate():
    """Read only four source/document files; never load weights or targets."""
    pins = {OWNER: OWNER_SHA, LICENSE: LICENSE_SHA,
            ROSTER_OWNER: ROSTER_SHA, TARIFF_OWNER: TARIFF_SHA}
    for path, expected in pins.items():
        require(not path.is_symlink() and digest(path.read_bytes()) == expected,
                f'Changed source owner: {path}')
    require(len(TRAIN) == 20 and not set(TRAIN).intersection(CHECK), 'Fixed20/4 roster')
    return {str(path): value for path, value in pins.items()}


def _environment_get(node):
    f = node.func if isinstance(node, ast.Call) else None
    return (isinstance(f, ast.Attribute) and f.attr == 'get'
            and isinstance(f.value, ast.Attribute) and f.value.attr == 'environ'
            and isinstance(f.value.value, ast.Name) and f.value.value.id == '_os')


def derive_source():
    """Return source and a byte-inverse certificate. Does not compile/import it."""
    pins = authenticate()
    original = OWNER.read_bytes()
    tree = ast.parse(original.decode('utf-8'), filename=str(OWNER))
    lines = original.splitlines(keepends=True)
    starts, count = [], 0
    for line in lines:
        starts.append(count)
        count += len(line)
    env_nodes = []
    keys = set()
    for node in ast.walk(tree):
        if not _environment_get(node):
            continue
        require(len(node.args) == 2 and not node.keywords
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in FIXED_ENV, 'Unowned environment expression')
        key = node.args[0].value
        keys.add(key)
        a = starts[node.lineno - 1] + node.col_offset
        b = starts[node.end_lineno - 1] + node.end_col_offset
        env_nodes.append((a, b, key))
    require(keys == set(FIXED_ENV), 'Complete environment-read inventory')
    # A nested getenv (CPRE fallback) is owned by its enclosing getenv span.
    outer = [x for x in env_nodes if not any(
        y[0] <= x[0] and x[1] <= y[1] and x != y for y in env_nodes)]
    edits = [(a, b, original[a:b], repr(FIXED_ENV[key]).encode(), 'env:' + key)
             for a, b, key in outer]
    require(original.count(ANCHOR) == 1, 'Unique pre-update feature boundary')
    pos = original.index(ANCHOR) + len(ANCHOR)
    edits.append((pos, pos, b'', INSERT, 'readonly_hook'))
    edits.sort(key=lambda item: item[0])
    for left, right in zip(edits, edits[1:]):
        require(left[1] <= right[0], 'Disjoint AST/hook edits')
    derived = original
    for a, b, old, new, label in reversed(edits):
        require(derived[a:b] == old, 'Forward source span: ' + label)
        derived = derived[:a] + new + derived[b:]
    positions, offset = [], 0
    for a, b, old, new, label in edits:
        positions.append((a + offset, old, new, label))
        offset += len(new) - len(old)
    inverse = derived
    for pos, old, new, label in reversed(positions):
        require(inverse[pos:pos + len(new)] == new, 'Inverse source span: ' + label)
        inverse = inverse[:pos] + old + inverse[pos + len(new):]
    require(inverse == original and digest(inverse) == OWNER_SHA, 'Whole public byte inverse')
    changed = ast.parse(derived.decode('utf-8'), filename=str(OWNER))
    require(not any(_environment_get(node) for node in ast.walk(changed)),
            'No remaining environment-dependent numerical configuration')
    proof = dict(schema='current_v29_temporal_source_proof_v1', source_bindings=pins,
                 commit='18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45',
                 original_sha256=OWNER_SHA, derived_sha256=digest(derived),
                 inverse_sha256=digest(inverse), whole_source_inverse=True,
                 fixed_environment=dict(FIXED_ENV), physical_source_changes=False,
                 hook_count=1, numerical_imports=False, estimator_executed=False,
                 edits=[dict(owner=label, old_sha256=digest(old), new_sha256=digest(new))
                        for _, old, new, label in positions])
    return derived.decode('utf-8'), proof


def feature_ceiling(n=1024):
    """Conservative complete FP32 feature tariff, NOT an actual bill."""
    return 64 * n * n + 4096 * n


def feature_packet(fnp, layer, last, mu, var, sigma, alpha, phi, gate,
                   k3, k4, host, covariance_off, skew_plane, weights):
    """All numerical operations use the caller's counted FNP. Never mutate inputs.

    Only newly allocated feature/host/scale arrays leave this function. Public
    pooled arrays may be overwritten immediately after the synchronous callback.
    """
    n = 1024
    require(0 <= layer < 16 and last == (layer == 15), 'Fixed16 trajectory')
    for value in (mu, var, sigma, alpha, phi, gate, host):
        require(value.shape == (n,) and value.dtype == fnp.float32, 'FP32 scalar inputs')
    require(weights.shape == (n, n) and weights.dtype == fnp.float32, 'Canonical weight axes')
    require((covariance_off is None) == last, 'Terminal covariance remains trimmed')
    require((skew_plane is None) == (layer == 0 or last), 'Actual D21 availability')
    require((k3 is None) == (layer == 0) and (k4 is None) == (layer == 0), 'Actual scalar availability')
    for value in (covariance_off, skew_plane):
        if value is not None:
            require(value.shape == (n, n) and value.dtype == fnp.float32, 'FP32 pair inputs')
    for value in (k3, k4):
        if value is not None:
            require(value.shape == (n,) and value.dtype == fnp.float32, 'FP32 cumulant inputs')
    one = fnp.ones(n, dtype=fnp.float32)
    zero = fnp.zeros(n, dtype=fnp.float32)
    logs = fnp.log(sigma)
    mean_over_s = host / sigma
    gaussian_over_s = phi + alpha * gate
    sk3 = zero if k3 is None else k3 / (var * sigma)
    sk4 = zero if k4 is None else k4 / (var * var)
    # Original w32 has input rows/output columns; no additional dense rotation.
    w2 = weights * weights
    w3 = w2 * weights
    w4 = w2 * w2
    weight_features = (
        fnp.sum(weights, axis=0) * 0.7071067811865475,
        fnp.sum(w2 - 0.001953125, axis=0) * 11.313708498984761,
        fnp.sum(w3, axis=0) * 93.47798314754834,
        fnp.sum(w4 - 1.1444091796875e-05, axis=0) * 836.0924988699916,
    )
    rho_mean = rho_rms = rho_abs = rho_alpha = zero
    if covariance_off is not None:
        rho = covariance_off / (sigma[:, None] * sigma[None, :])
        rho_mean = fnp.mean(rho, axis=1)
        rho_rms = fnp.sqrt(fnp.mean(rho * rho, axis=1))
        rho_abs = fnp.mean(fnp.abs(rho), axis=1)
        rho_alpha = (rho @ alpha) / n
    row_norm = col_norm = row_sum = col_sum = row_abs = col_abs = zero
    if skew_plane is not None:
        sk = skew_plane / (var[:, None] * sigma[None, :])
        sk2 = sk * sk
        abs_sk = fnp.abs(sk)
        row_norm = fnp.sqrt(fnp.sum(sk2, axis=1))
        col_norm = fnp.sqrt(fnp.sum(sk2, axis=0))
        row_sum = fnp.sum(sk, axis=1) / 32.0
        col_sum = fnp.sum(sk, axis=0) / 32.0
        row_abs = fnp.mean(abs_sk, axis=1)
        col_abs = fnp.mean(abs_sk, axis=0)
    values = (
        one * (2 * layer / 15 - 1), one * (layer / 16), one, alpha,
        logs - fnp.mean(logs), gate, phi, gaussian_over_s, mean_over_s,
        mean_over_s - gaussian_over_s, sk3, sk4, fnp.abs(alpha),
        rho_mean, rho_rms, rho_abs, *weight_features, row_norm, col_norm,
        row_sum, col_sum, rho_alpha, row_abs, col_abs,
        fnp.log1p(fnp.abs(sk3)), fnp.log1p(fnp.abs(sk4)),
        one * fnp.mean(mean_over_s),
    )
    require(len(values) == len(FEATURE_NAMES) == 30, 'Fixed30 features')
    features = fnp.stack(values, axis=1)
    host_copy, sigma_copy = fnp.copy(host), fnp.copy(sigma)
    require(features.shape == (n, 30) and features.dtype == fnp.float32,
            'Fresh FP32 feature matrix')
    for value in (features, host_copy, sigma_copy):
        require(bool(fnp.all(fnp.isfinite(value))), 'Finite captured features/host/scale')
    return dict(schema='current_v29_temporal_feature_packet_v1', layer=layer,
                terminal=last, covariance_available=not last,
                skew_plane_available=(layer not in (0, 15)),
                features=features, host=host_copy, sigma=sigma_copy)


def load_runtime(*, allow_numerical_imports=False):
    """Explicit future caller only. No setup/predict, file output or target access."""
    require(allow_numerical_imports is True, 'Numerical runtime requires separate launch authority')
    source, proof = derive_source()
    module = types.ModuleType('qualified_v29_temporal_capture')
    module.__file__ = str(OWNER)

    def emit(binding, *args):
        sink, context = binding
        start_f, start_ops = int(context.flops_used), len(context.op_log)
        packet = feature_packet(module.fnp, *args)
        delta = int(context.flops_used) - start_f
        operations = len(context.op_log) - start_ops
        require(0 < delta <= feature_ceiling() and operations > 0, 'Actual complete feature tariff')
        packet.update(feature_flops=delta, feature_operations=operations)
        # The sink may serialize these fresh arrays, not run another FNP operator.
        before_sink = (int(context.flops_used), len(context.op_log))
        sink(packet)
        require((int(context.flops_used), len(context.op_log)) == before_sink,
                'Capture sink must not perform numerical operations')

    module._v29_temporal_emit = emit
    exec(compile(source, str(OWNER), 'exec'), module.__dict__)
    return module, proof


def attach_capture(estimator, sink, context):
    """Metadata attachment; sink gets no original state or pooled-array aliases."""
    require(callable(sink) and hasattr(context, 'flops_used') and hasattr(context, 'op_log'),
            'Explicit sink and owning actual counted context')
    require(not hasattr(estimator, '_v29_temporal_sink'), 'One capture attachment per estimator')
    estimator._v29_temporal_sink = (sink, context)


def detach_capture(estimator):
    require(hasattr(estimator, '_v29_temporal_sink'), 'Existing capture attachment')
    del estimator._v29_temporal_sink


if __name__ == '__main__':
    raise SystemExit('Source-only adapter: no CLI stage, qualification, fit, or prediction is implemented.')
