"""DRAFT, inert: current c2 -> original banked D4 PUSH fallback; no writer/run."""
import ast
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
PARENT = NEW/'current_comp_shared_gain_slice_v1/build_v1/estimator.py'
PARENT_SHA = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
D4_OWNER = NEW/'birth_direction_fit_v1/release_confirmation_v1/package_stage/estimator.py'
D4_OWNER_SHA = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'
PINS = {PARENT: PARENT_SHA, D4_OWNER: D4_OWNER_SHA}
MAX_SOURCE_BYTES = 3*1024**2
REMOVED_COLD_BINDING_FLOPS = 278566656+282489344
PLANNED_INITIALIZATION_FLOPS = 1944970298-REMOVED_COLD_BINDING_FLOPS


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def _read(path, digest, tick=None):
    if tick is not None:
        tick()
    require(not path.is_symlink() and path.is_file(), 'Exact regular source owner')
    before = path.stat()
    require(0 < before.st_size <= MAX_SOURCE_BYTES, 'Bounded source size')
    with path.open('rb') as stream:
        raw = stream.read(MAX_SOURCE_BYTES+1)
    after = path.stat()
    identity = lambda s: (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    require(identity(before) == identity(after) and len(raw) == before.st_size
        and hashlib.sha256(raw).hexdigest() == digest, 'Unchanged exact source bytes')
    raw.decode('utf-8')
    if tick is not None:
        tick()
    return raw


def authenticate(tick=None):
    """Source pins only: not admission, a build receipt, or numerical authority."""
    for path,digest in PINS.items():
        _read(path,digest,tick)
    return {str(path):digest for path,digest in PINS.items()}


def _index(raw, tick=None):
    if tick is not None:
        tick()
    tree = ast.parse(raw.decode('utf-8'))
    offsets,position = [],0
    for line in raw.splitlines(keepends=True):
        offsets.append(position)
        position += len(line)
    require(position == len(raw), 'Exact UTF8 byte line index')
    if tick is not None:
        tick()
    return tree,offsets


def _span(raw, offsets, node):
    start = offsets[node.lineno-1]+node.col_offset
    stop = offsets[node.end_lineno-1]+node.end_col_offset
    return start,stop,raw[start:stop]


def _last(tree, name):
    nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == name]
    require(bool(nodes), 'Observed final function: '+name)
    return nodes[-1]


def derive_source(tick=None):
    """Return (full UTF8 text, proof), without import/compile/run of an estimator."""
    parent = _read(PARENT,PARENT_SHA,tick)
    bank = _read(D4_OWNER,D4_OWNER_SHA,tick)
    pt,pi = _index(parent,tick)
    bt,bi = _index(bank,tick)
    helper_hashes = {}
    for name in ('_packed_frame','_packed_product_cached','_mm'):
        current = _span(parent,pi,_last(pt,name))[2]
        original = _span(bank,bi,_last(bt,name))[2]
        require(current == original, 'Current exact banked D4 owner: '+name)
        helper_hashes[name] = hashlib.sha256(current).hexdigest()

    target = _last(pt,'_packed_product_4')
    start,stop,old_dispatch = _span(parent,pi,target)
    bank_dispatch = _span(bank,bi,_last(bt,'_packed_product_4'))[2]
    require(bank_dispatch == b'def _packed_product_4(a,b,out,arena_views):\n'
        b'    return _packed_product_cached(a,b,out,arena_views,4)',
        'Exact already-banked fallback, no new kernel arithmetic')
    require(b'return _p1_d6_832_product(a,b,out,_P1_D6_832_BINDINGS[0])' in old_dispatch
        and old_dispatch.endswith(b'return _packed_product_cached(a,b,out,arena_views,4)'),
        'Only current broad PUSH override precedes original fallback')
    predictions = _last(pt,'_predict_binary')
    calls = [n for n in ast.walk(pt) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Name) and n.func.id == '_p1_d6_832_bind']
    statements = [n for n in ast.walk(predictions) if isinstance(n,ast.Expr)
        and isinstance(n.value,ast.Call) and n.value in calls]
    require(len(calls) == len(statements) == 1, 'One original paid cold binding statement')
    a,b,old_binding = _span(parent,pi,statements[0])
    require(old_binding == b'_p1_d6_832_bind(_SCRATCH[0])', 'Exact unused D6 binding owner')
    new_binding = b'pass  # PUSH runtime repair: no unused D6 view bindings.'
    edits = sorted(((start,stop,old_dispatch,bank_dispatch),(a,b,old_binding,new_binding)),reverse=True)
    changed,applied = parent,[]
    for begin,end,before,after in edits:
        require(changed[begin:end] == before, 'Exact disjoint source splice')
        changed = changed[:begin]+after+changed[end:]
        applied.append((begin,before,after))
    inverse = changed
    for begin,before,after in reversed(applied):
        require(inverse[begin:begin+len(after)] == after, 'Exact positional inverse span')
        inverse = inverse[:begin]+before+inverse[begin+len(after):]
    require(inverse == parent, 'Entire c2 source restored byte-for-byte')
    final_tree,_ = _index(changed,tick)
    require(not [n for n in ast.walk(final_tree) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Name) and n.func.id == '_p1_d6_832_bind'],
        'No D6 source-binding invocation remains')
    proof = dict(schema='current_comp_push_runtime_repair_composition_v1',
        parent_sha256=PARENT_SHA,banked_D4_source_sha256=D4_OWNER_SHA,
        candidate_sha256=hashlib.sha256(changed).hexdigest(),source_bytes=len(changed),
        full_source_inverse_verified=True,exact_banked_kernel_helpers=helper_hashes,
        splice_count=2,only_push_override_and_unused_cold_binding_changed=True,
        original_D6_definitions_and_layout_literals_preserved=True,
        all_other_source_and_learned_literals_byte_exact=True,
        square_read_tiling_and_all_consumers_unchanged=True,
        original_arena_bytes=1168900096,additional_backing_arenas=0,
        removed_cold_binding_flops=REMOVED_COLD_BINDING_FLOPS,
        planned_initialization_flops=PLANNED_INITIALIZATION_FLOPS,
        initialization_requires_actual_measurement=True,
        first5_c2_byte_parity_inherited=False,raw_fidelity_measured=False,
        runtime_measured=False,actual_complete_bills_measured=False,
        model_calls=0,parameter_updates=0,truth_accessed=False,
        pair256_composed=False,submission_authorized=False,
        source_bindings={str(path):digest for path,digest in PINS.items()})
    return changed.decode('utf-8'),proof


if __name__ == '__main__':
    print('Inert source-only draft. No candidate is written and no stage is run.')
