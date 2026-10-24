"""DRAFT inert BOX-to-lazy-factor source composer; no writer or model entry.

Root may call authenticate(tick) and derive_source(tick) in metadata admission.
The returned estimator text is never executed, imported or materialized here.
"""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOX = HERE.parent/'current_comp_push_frontier_boxes_v1'
PARENT = BOX/'rpc37_v1/estimator.py'
PARENT_SHA = '378092f76102b83ddcff445adaf27309ae8136d1df6844599dad59b69cd75dfa'
UTILITY = BOX/'compose.py'
UTILITY_SHA = '9783b429379f699a86b64d40baa4623c9320e0728c6bf72f1c7de47ec00e8038'
FRAGMENT = HERE/'fnp.py'
FRAGMENT_SHA = 'e96ee6b660a688f767abd2b8300883e52920eace477e35e7b5edaae759ac6e92'
PINS = {str(PARENT):PARENT_SHA, str(UTILITY):UTILITY_SHA,
        str(FRAGMENT):FRAGMENT_SHA}
INITIALIZATION_FLOPS = 3286988570
EXTRA_COLD_BINDING_FLOPS = 0
_UTILITY = None


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def _utility(tick=None):
    global _UTILITY
    if tick is not None:
        tick()
    require(UTILITY.is_file() and not UTILITY.is_symlink(), 'Exact source helper')
    before = UTILITY.stat()
    require(0 < before.st_size < 1024**2, 'Bounded source helper')
    raw = UTILITY.read_bytes()
    after = UTILITY.stat()
    ident = lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    require(ident(before)==ident(after) and len(raw)==before.st_size
            and hashlib.sha256(raw).hexdigest()==UTILITY_SHA, 'Frozen stdlib helper')
    if _UTILITY is None:
        namespace = dict(__name__='_lazy_source_utility', __file__=str(UTILITY))
        exec(compile(raw, str(UTILITY), 'exec'), namespace)
        _UTILITY = namespace
    if tick is not None:
        tick()
    return _UTILITY


def authenticate(tick=None):
    helper = _utility(tick)
    for path, digest in PINS.items():
        helper['_read'](Path(path), digest, tick)
    return dict(PINS)


def _splice(body, before, after):
    require(before != after and body.count(before)==1, 'Unique original source span')
    result = body.replace(before, after)
    require(result.count(after)==1 and result.replace(after,before)==body,
            'Complete inverse of each prospective numerical source edit')
    return result


def derive_source(tick=None):
    sources = authenticate(tick)
    h = _utility(tick)
    original = h['_read'](PARENT, PARENT_SHA, tick)
    fragment = h['_read'](FRAGMENT, FRAGMENT_SHA, tick)
    require(b'_slr_' not in original and b'_SLR_' not in original,
            'Fresh bounded private representation namespace')
    require(fragment.isascii(), 'ASCII numerical appendix')
    fragment_tree = ast.parse(fragment.decode('ascii'))
    require(all(isinstance(node,(ast.FunctionDef,ast.Assign))
                or (isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant)
                    and isinstance(node.value.value,str)) for node in fragment_tree.body)
            and not any(isinstance(node,(ast.Import,ast.ImportFrom))
                        for node in ast.walk(fragment_tree)),
            'Numerical fragment has no imports or module-entry tensor work')
    constants = {node.targets[0].id:ast.literal_eval(node.value)
                 for node in fragment_tree.body if isinstance(node,ast.Assign)
                 and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)}
    require(constants=={'_SLR_RANK':384,'_SLR_LAZY_LAYERS':(8,10,12,14),
                        '_SLR_CONSOLIDATE_LAYERS':(9,11,13)}
            and all(not node.decorator_list and not node.args.defaults
                    and not any(node.args.kw_defaults)
                    for node in fragment_tree.body if isinstance(node,ast.FunctionDef)),
            'Only exact host constants and undecorated inert functions at entry')
    tree, offsets = h['_index'](original, tick)
    changes = {
        '_shared_old384_state': [
            (b'Q=None, F=None, core=None, pre_basis=None)',
             b'Q=None, F=None, core=None, pre_basis=None, lazy=None)')],
        '_shared_old384_push': [
            (b"            rotated = _shared_old384_mm(overlap, state['F'])\n"
             b"            core = _shared_old384_mm(overlap, _shared_old384_mm(state['core'], overlap.T))\n"
             b"            factors = fnp.concatenate([rotated, Fj], axis=1)\n"
             b"            core = core + joined_core\n"
             b"            del overlap, rotated",
             b"            factors = _slr_join_factors(layer, overlap, state, Fj)\n"
             b"            core = _shared_old384_mm(overlap, _shared_old384_mm(state['core'], overlap.T))\n"
             b"            core = core + joined_core\n"
             b"            del overlap"),
            (b"        state['F'] = _shared_old384_finite(factors, 'old coefficient factors')",
             b"        _slr_finish_factors(state, factors)"),
            (b"        if tuple(state['F'].shape) != (rank, old_columns):\n"
             b"            raise RuntimeError('Shared-old384 factor columns')",
             b"        _slr_layout(state, old_columns)"),
            (b"    old = _shared_old384_mm(U, state['F'])",
             b"    old = _slr_reconstruct(U, state)")],
        '_shared_old384_read': [
            (b"    weighted_factors = state['F'] * lam[:old_columns][None, :]\n"
             b"    inner = _shared_old384_mm(old_left * old_left, weighted_factors.T)",
             b"    inner = _slr_inner(old_left, lam[:old_columns], state)")]
    }
    edits, function_proofs = [], {}
    for name, splices in changes.items():
        start, stop, before = h['_span'](original, offsets, h['_last'](tree,name))
        after = before
        for old, new in splices:
            after = _splice(after, old, new)
        inverse = after
        for old, new in reversed(splices):
            require(inverse.count(new)==1, 'Unique inverse span in '+name)
            inverse = inverse.replace(new,old)
        require(inverse==before, 'Entire original function inverse: '+name)
        edits.append((start,stop,before,after))
        function_proofs[name] = dict(before_sha256=hashlib.sha256(before).hexdigest(),
            after_sha256=hashlib.sha256(after).hexdigest(), splices=len(splices))
    changed, applied = original, []
    for start, stop, before, after in sorted(edits,reverse=True):
        require(changed[start:stop]==before, 'Exact disjoint original function')
        changed = changed[:start]+after+changed[stop:]
        applied.append((start,before,after))
    appendix = b'\n\n# BEGIN FIXED LAZY ROTATION V1\n'+fragment+b'\n# END FIXED LAZY ROTATION V1\n'
    changed += appendix
    inverse = changed[:-len(appendix)]
    for start,before,after in reversed(applied):
        require(inverse[start:start+len(after)]==after, 'Exact positional whole inverse')
        inverse = inverse[:start]+before+inverse[start+len(after):]
    require(inverse==original, 'Full parent restored byte-for-byte')
    final, final_offsets = h['_index'](changed,tick)
    unchanged = {}
    for name in ('_predict_binary','_shared_old384_mm','_shared_old384_finite',
                 '_shared_old384_commit','_push_bank_shared_newborn','_mm',
                 '_packed_product_4','_p1_d6_832_product','_fbx_product','_fbx_bind'):
        a = h['_span'](original,offsets,h['_last'](tree,name))[2]
        b = h['_span'](changed,final_offsets,h['_last'](final,name))[2]
        require(a==b, 'Unchanged original physical/helper owner: '+name)
        unchanged[name] = hashlib.sha256(a).hexdigest()
    require(len(changed)<3*1024**2, 'Bounded complete source')
    proof = dict(schema='current_comp_shared_lazy_rotation_composition_v1',
        parent_sha256=PARENT_SHA, candidate_sha256=hashlib.sha256(changed).hexdigest(),
        source_bytes=len(changed), source_bindings=sources,
        full_source_inverse_verified=True, changed_original_functions=function_proofs,
        original_source_splices=6, appendix_sha256=hashlib.sha256(appendix).hexdigest(),
        unchanged_helper_sha256=unchanged, lazy_layers=[8,10,12,14],
        consolidate_layers=[9,11,13], final_map_used_at_layer=15,
        maximum_retained_factor_groups=2, unchanged_rank=384,
        all_original_QR_and_canonical_core_updates_retained=True,
        all_1024_row_reconstruction_and_original_consumers_retained=True,
        full_stored_factor_finite_scans_retained=True, no_coordinate_inverse_or_solve=True,
        virtual_rotation_overflow_envelope_paid=True,
        individual_map_and_prefix_overflow_envelopes_paid=True,
        every_original_binding_and_arena_retained=True, extra_cold_binding_flops=0,
        planned_initialization_flops=INITIALIZATION_FLOPS,
        original_arena_bytes=1168900096, additional_backing_arenas=0,
        real_arithmetic_factor_identity=True, FP32_byte_identity_claimed=False,
        model_calls=0, numerical_work_executed=False, candidate_materialized=False,
        truth_accessed=False, runtime_qualified=False, submission_authorized=False)
    return changed.decode('utf-8'), json.loads(json.dumps(proof,sort_keys=True,allow_nan=False))


if __name__ == '__main__':
    print('Inert source component: no estimator, array, build or stage is executed.')
