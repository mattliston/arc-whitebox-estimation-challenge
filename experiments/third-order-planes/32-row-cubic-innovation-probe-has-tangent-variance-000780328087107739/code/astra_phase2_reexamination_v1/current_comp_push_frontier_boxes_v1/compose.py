"""DRAFT: inert full-source inverse, current c2 plus two fixed frontier boxes.

No writer, build, model import, array, target, stage or numerical CLI exists.
Only root's separately admitted supervisor may materialize the returned text.
"""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent/'current_comp_shared_gain_slice_v1/build_v1/estimator.py'
PARENT_SHA = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
KERNEL = HERE/'kernel.py'
KERNEL_SHA = '7002c3c970fc5fac86d64714d55130291a0ffac872a9d85a15d22147a8974403'
EXTRA_COLD_FLOPS = 1342018272
PARENT_INITIALIZATION_FLOPS = 1944970298
PLANNED_INITIALIZATION_FLOPS = 3286988570
PARENT_COLD37_FLOPS = 406399683439
PARENT_WARM37_FLOPS = 404454713141
PLANNED_COLD37_FLOPS = 407741701711
PLANNED_WARM37_FLOPS = 404454713141
PINS = {str(PARENT):PARENT_SHA,str(KERNEL):KERNEL_SHA}
_KERNEL = None


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def _read(path,digest,tick=None):
    if tick is not None:
        tick()
    path = Path(path)
    require(path.is_file() and not path.is_symlink(),'Exact regular source owner')
    before = path.stat()
    require(0<before.st_size<=3*1024**2,'Bounded source owner')
    with path.open('rb') as stream:
        raw = stream.read(3*1024**2+1)
    after = path.stat()
    identity = lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    require(identity(before)==identity(after) and len(raw)==before.st_size
        and hashlib.sha256(raw).hexdigest()==digest,'Exact unchanged source bytes')
    raw.decode('utf-8')
    if tick is not None:
        tick()
    return raw


def _kernel(tick=None):
    global _KERNEL
    raw = _read(KERNEL,KERNEL_SHA,tick)
    if _KERNEL is None:
        namespace = dict(__name__='_reviewed_two_box_source_owner',__file__=str(KERNEL))
        exec(compile(raw,str(KERNEL),'exec'),namespace)
        _KERNEL = namespace
    return _KERNEL


def authenticate(tick=None):
    """Flat direct source pins only; no historical eligibility is inferred."""
    for name,digest in PINS.items():
        _read(name,digest,tick)
    pins = _kernel(tick)['authenticate'](tick)
    for name,digest in PINS.items():
        require(name not in pins or pins[name]==digest,'Same direct owner')
        pins[name] = digest
    return pins


def _index(raw,tick=None):
    if tick is not None:
        tick()
    tree = ast.parse(raw.decode('utf-8'))
    offsets,position = [],0
    for line in raw.splitlines(keepends=True):
        offsets.append(position)
        position += len(line)
    require(position==len(raw),'Single-pass exact UTF8 AST byte offsets')
    if tick is not None:
        tick()
    return tree,offsets


def _span(raw,offsets,node):
    start = offsets[node.lineno-1]+node.col_offset
    stop = offsets[node.end_lineno-1]+node.end_col_offset
    require(0<=start<stop<=len(raw),'Valid exact source span')
    return start,stop,raw[start:stop]


def _last(tree,name):
    nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
    require(bool(nodes),'Observed last-effective source definition: '+name)
    return nodes[-1]


def derive_source(tick=None):
    """Return full candidate text and proof; do not write, compile or run it."""
    sources = authenticate(tick)
    original = _read(PARENT,PARENT_SHA,tick)
    require(b'_fbx_' not in original and b'_FBX_' not in original,'Fresh private namespace')
    tree,offsets = _index(original,tick)
    appendix,kernel_proof = _kernel(tick)['source_and_proof'](tick)
    require(appendix.isascii(),'Entire added numerical source is ASCII')
    start,stop,dispatch = _span(original,offsets,_last(tree,'_packed_product_4'))
    old = b'        return _p1_d6_832_product(a,b,out,_P1_D6_832_BINDINGS[0])'
    new = (b'        if int(a.shape[1]) == 384:\n'
           b'            return _fbx_product(a,b,out,_FBX_BINDINGS[0])\n'
           b'        if int(b.shape[1]) <= 6144:\n'
           b'            return _fbx_product(a,b,out,_FBX_BINDINGS[1])\n'+old)
    require(dispatch.count(old)==1 and dispatch.endswith(
        b'    return _packed_product_cached(a,b,out,arena_views,4)'),
        'Exact current original broad PUSH guard and final fallback')
    replacement = dispatch.replace(old,new)
    require(replacement.count(new)==1 and replacement.replace(new,old)==dispatch,
        'Entire original dispatcher restored, including every guard and fallback')
    predictions = _last(tree,'_predict_binary')
    calls = [node for node in ast.walk(tree) if isinstance(node,ast.Call)
        and isinstance(node.func,ast.Name) and node.func.id=='_p1_d6_832_bind']
    statements = [node for node in ast.walk(predictions) if isinstance(node,ast.Expr)
        and isinstance(node.value,ast.Call) and node.value in calls]
    require(len(calls)==len(statements)==1,'One original cold paid binding statement')
    a,b,binding = _span(original,offsets,statements[0])
    require(binding==b'_p1_d6_832_bind(_SCRATCH[0])','Exact retained cold binding')
    added_binding = binding+b'\n        _fbx_bind(_SCRATCH[0])'
    edits = sorted(((start,stop,dispatch,replacement),(a,b,binding,added_binding)),reverse=True)
    changed,applied = original,[]
    for begin,end,before,after in edits:
        require(changed[begin:end]==before,'Exact disjoint source edit')
        changed = changed[:begin]+after+changed[end:]
        applied.append((begin,before,after))
    core = changed
    changed += appendix.encode('ascii')
    require(changed.endswith(appendix.encode('ascii')),'Exact appended inert helper block')
    inverse = changed[:-len(appendix.encode('ascii'))]
    for begin,before,after in reversed(applied):
        require(inverse[begin:begin+len(after)]==after,'Exact positional complete inverse')
        inverse = inverse[:begin]+before+inverse[begin+len(after):]
    require(inverse==original,'Whole parent restored byte-for-byte, not AST equivalence')
    require(len(changed)<=3*1024**2,'Bounded complete candidate source')
    final,final_offsets = _index(changed,tick)
    unchanged_helpers = {}
    for name in ('_cp3_bind_arena_views','_packed_frame','_packed_product_cached',
                 '_p1_d6_832_bind','_p1_d6_832_product','_mm','_shared_old384_push',
                 '_shared_old384_read','_shared_old384_commit'):
        before = _span(original,offsets,_last(tree,name))[2]
        after = _span(changed,final_offsets,_last(final,name))[2]
        require(before==after,'Original numerical/binding owner remains byte-exact: '+name)
        unchanged_helpers[name] = hashlib.sha256(before).hexdigest()
    root_calls = [node for node in ast.walk(_last(final,'_predict_binary'))
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
        and node.func.id in ('_p1_d6_832_bind','_fbx_bind')]
    require([node.func.id for node in root_calls]==['_p1_d6_832_bind','_fbx_bind'],
        'Original binding first, both new binding sets second, inside cold prediction')
    require(PARENT_COLD37_FLOPS+EXTRA_COLD_FLOPS==PLANNED_COLD37_FLOPS
        and PLANNED_COLD37_FLOPS-PLANNED_WARM37_FLOPS==PLANNED_INITIALIZATION_FLOPS,
        'Own complete cold bill and initialization; no inherited setup credit')
    proof = dict(schema='current_comp_push_frontier_boxes_composition_v1',
        source_bindings=sources,parent_sha256=PARENT_SHA,
        candidate_sha256=hashlib.sha256(changed).hexdigest(),source_bytes=len(changed),
        full_source_inverse_verified=True,original_source_edits=2,appendix_blocks=1,
        appendix_sha256=hashlib.sha256(appendix.encode('ascii')).hexdigest(),
        changed_core_sha256=hashlib.sha256(core).hexdigest(),kernel_proof=kernel_proof,
        unchanged_helper_sha256=unchanged_helpers,
        original_dispatch_guard_byte_exact=True,first_box_k384_n_le8192=True,
        second_box_k_le1024_n_le6144=True,original_general_serial_fallback_retained=True,
        every_original_paid_binding_and_layout_literal_retained=True,
        all_other_source_and_learned_literals_byte_exact=True,
        square_READ_tiling_and_all_physical_consumers_unchanged=True,
        original_arena_bytes=1168900096,additional_backing_arenas=0,
        additional_cold_binding_flops=EXTRA_COLD_FLOPS,
        planned_initialization_flops=PLANNED_INITIALIZATION_FLOPS,
        planned_cold37_complete_flops=PLANNED_COLD37_FLOPS,
        planned_warm37_complete_flops=PLANNED_WARM37_FLOPS,
        model_calls=0,parameter_updates=0,truth_accessed=False,
        numerical_work_executed=False,runtime_measured=False,
        full_candidate_byte_parity_measured=False,actual_complete_bills_measured=False,
        candidate_materialized=False,submission_authorized=False)
    return changed.decode('utf-8'),json.loads(json.dumps(proof,sort_keys=True,allow_nan=False))


if __name__ == '__main__':
    print('Inert source draft: no candidate, array, build or stage is created.')
