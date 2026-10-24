"""Inert output-only nonnegative projection of the frozen v29 capture source.

Public MIT attribution and the complete original notice remain in PROTOCOL.md.
No physical recurrence, feature, parameter, setup or import-time array changes.
"""
import ast
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE/'adapter.py'
BASE_SHA = 'cab844cd7981d31c7e44f4797ad60490729bdfadb8ce46655cdd5966ae9f36b7'
BASE_PROTOCOL = HERE/'PROTOCOL.md'
BASE_PROTOCOL_SHA = '2a93907e69427a55c9a11eef7bd2143d9f71835d62f5eb4323700c1a8bd545ba'
OLD_RETURN = '        return fnp.stack(rows, axis=0)'
NEW_RETURN = '        return fnp.maximum(fnp.stack(rows, axis=0), 0.0)'
PROJECTION_FLOPS = 16*1024
_BASE = None


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def _base():
    global _BASE
    if _BASE is None:
        require(not BASE.is_symlink() and hashlib.sha256(BASE.read_bytes()).hexdigest() == BASE_SHA,
                'Frozen original capture adapter')
        spec = importlib.util.spec_from_file_location('v29_projection_original_adapter', BASE)
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        _BASE = value
    return _BASE


def authenticate():
    require(not BASE_PROTOCOL.is_symlink()
            and hashlib.sha256(BASE_PROTOCOL.read_bytes()).hexdigest() == BASE_PROTOCOL_SHA,
            'Frozen original feature specification and MIT attribution')
    return {str(BASE):BASE_SHA,str(BASE_PROTOCOL):BASE_PROTOCOL_SHA,**_base().authenticate()}


def derive_source():
    """One reversible final-return edit; no compilation or numerical import."""
    pins = authenticate()
    original, parent = _base().derive_source()
    require(original.count(OLD_RETURN) == 1 and NEW_RETURN not in original,
            'Unique physical core final stack return')
    changed = original.replace(OLD_RETURN, NEW_RETURN)
    require(changed.count(NEW_RETURN) == 1 and changed.replace(NEW_RETURN, OLD_RETURN) == original,
            'Whole projected-source inverse to frozen feature-adapted public source')
    tree = ast.parse(changed)
    classes = [n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == 'Estimator']
    core = [n for n in classes[0].body if isinstance(n,ast.FunctionDef) and n.name == '_predict_core']
    require(len(classes) == len(core) == 1 and isinstance(core[0].body[-1],ast.Return),
            'Only the final returned value, after the complete unchanged physical loop')
    require(ast.unparse(core[0].body[-1].value) == 'fnp.maximum(fnp.stack(rows, axis=0), 0.0)',
            'Exact FP32 maximum, not clip, nan replacement, feedback or an in-place write')
    proof = dict(schema='current_v29_output_projection_source_v1',parent=parent,source_bindings=pins,
        original_sha256=parent['original_sha256'],unprojected_sha256=parent['derived_sha256'],
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),whole_source_inverse=True,
        final_output_only=True,physical_trajectory_unchanged=True,raw_feature_hosts_unchanged=True,
        projection_expression=NEW_RETURN.strip(),projection_flops_planning=PROJECTION_FLOPS,
        projection_extra_output_bytes=65536,head_attached=False,numerical_imports=False,
        estimator_executed=False,original_negative_acquisition_not_requalified=True)
    return changed, proof


def load_runtime(*, allow_numerical_imports=False):
    """Reuse the exact old loader/emitter with only its source provider replaced."""
    require(allow_numerical_imports is True, 'Separate explicit numerical launch authority')
    base = _base()
    source = BASE.read_text()
    nodes = [n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name == 'load_runtime']
    require(len(nodes) == 1, 'Exact original paid feature loader/emitter')
    node = nodes[0]
    selected = ''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    scope = dict(vars(base),derive_source=derive_source)
    exec(compile(selected,str(BASE)+':unchanged-loader','exec'),scope)
    return scope['load_runtime'](allow_numerical_imports=True)


def feature_ceiling(n=1024):
    return _base().feature_ceiling(n)


def attach_capture(estimator, sink, context):
    return _base().attach_capture(estimator,sink,context)


def detach_capture(estimator):
    return _base().detach_capture(estimator)


# These are metadata only; do not import or instantiate the public estimator.
OWNER = HERE.parent/'public_504aldo_20260912_v1/repository/estimators/estimator_v29.py'
TERMINAL_ZERO_SLOTS = (13,14,15,20,21,22,23,24,25,26)
SKEW_ZERO_FIRST_SLOTS = (10,11,20,21,22,23,25,26,27,28)
