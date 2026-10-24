"""Inert exact-geometry scratch proposal; original three-root D5 arithmetic.

No numerical imports, arrays or test runs at import. The inherited source
provider authenticates actual BOX definitions and their executed schedules.
"""
import ast
import hashlib
import importlib.util
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER = HERE.parent/'current_comp_root_batch_gram_v2/kernel.py'
OWNER_SHA = '6ee41c26c1156bdd308b34ee96977f91624c3a099dd24b324b5ee92a89650f02'
ARENA_ELEMENTS = 292225024
WARM_FLOPS = 955654144
NAMES = ('_cp3_bind_arena_views','_packed_frame','_packed_product_cached',
         '_packed_product_5','_square_product','_resident_gram512')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


require(not OWNER.is_symlink() and hashlib.sha256(OWNER.read_bytes()).hexdigest() == OWNER_SHA,
        'Exact executed schedule/primitive metadata owner')
_spec = importlib.util.spec_from_file_location('compact_scratch_primitive_provider',OWNER)
_BASE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_BASE)
PINS = {**_BASE.PINS,OWNER:OWNER_SHA}


def compact_specs(specs):
    """Shrink each interval prefix; no two original intervals are relocated."""
    m,k,n,d = 512,1024,512,5
    require(len(specs) == 6*d, 'Original transformed/dependency/leaf/unwind roster')
    axes = []
    for level in range(1,d+1):
        axes.extend(((m//2**level,k//2**level),(k//2**level,n//2**level)))
    axes.extend((m//2**level,k//2**level) for level in range(1,d+1))
    axes.extend((k//2**level,n//2**level) for level in range(1,d+1))
    axes.append((m//2**d,n//2**d))
    axes.extend((m//2**(level-1),n//2**(level-1)) for level in range(d,1,-1))
    axes.extend((m//2**level,n//2**level) for level in range(d,0,-1))
    require(len(axes) == len(specs), 'One exact dimension pair for every bound view')
    compact,intervals = [],[]
    for ordinal,((start,stop,old_shape),(rows,columns)) in enumerate(zip(specs,axes)):
        require(rows <= old_shape[-2] and columns <= old_shape[-1], 'Only shrink original capacity')
        shape = (*old_shape[:-2],rows,columns)
        end = start+math.prod(shape)
        require(0 <= start < end <= stop <= ARENA_ELEMENTS, 'Original interval-prefix ownership')
        compact.append((start,end,shape))
        intervals.append(dict(ordinal=ordinal,original_start=start,original_stop=stop,
            compact_stop=end,original_shape=list(old_shape),compact_shape=list(shape),
            interval_subset=True,compact_matrix_row_stride=columns))
    return tuple(compact),intervals


def source_bundle(tick=None):
    emitted,original_specs,unused_batch_specs,parent_proof = _BASE.source_bundle(tick)
    tree = ast.parse(emitted)
    functions = {}
    for name in NAMES:
        nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == name]
        require(len(nodes) == 1, 'Only the authenticated original helper body: '+name)
        source = ast.get_source_segment(emitted,nodes[0])
        require(hashlib.sha256(source.encode()).hexdigest() == parent_proof['exact_original_functions'][name],
                'Byte-unchanged original arithmetic, not the failed batched implementation')
        functions[name] = source
    source = '\n\n'.join(functions.values())+'\n'
    compact,intervals = compact_specs(original_specs)
    original_bind = sum(math.prod(shape) for _,_,shape in original_specs)
    compact_bind = sum(math.prod(shape) for _,_,shape in compact)
    require(original_bind == 172392448, 'Original full-capacity binding price')
    fixed = original_bind+compact_bind+4*WARM_FLOPS+4292870144
    require(fixed < 10_000_000_000, 'Complete unit has positive ancillary budget')
    proof = dict(parent=parent_proof,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        original_functions_unchanged=True,no_root_batching=True,no_interval_relocation=True,
        interval_subsets=intervals,geometry=[512,1024,512],depth=5,
        original_bind_flops=original_bind,compact_bind_flops=compact_bind,
        combined_bind_flops=original_bind+compact_bind,warm_helper_flops=WARM_FLOPS,
        fixed_core_reference_bindings_flops=fixed,
        ancillary_allowance_flops=10_000_000_000-fixed,
        original_arena_bytes=4*ARENA_ELEMENTS,
        compact_leaf_a_shape=[7]*5+[16,32],compact_leaf_b_shape=[7]*5+[32,16],
        full_estimator_cost_measured=False)
    return source,original_specs,compact,proof


def _namespace(fnp,source,specs):
    env = dict(fnp=fnp,_PACKED_FRAME_CACHE={},_SCRATCH=None,_PACKED_SQUARE5=specs)
    exec(compile(source,'compact-scratch:original-D5-arithmetic','exec'),env)
    def schedule(m,k,n):
        require((m,k,n) == (512,1024,512), 'Only the fixed original root geometry')
        return 5,(m,k,n)
    def mm(a,b):
        require(str(a.dtype) == str(b.dtype) == 'float32', 'Original FP32 operands')
        return env['_square_product'](a,b,int(a.shape[0]),int(a.shape[1]),int(b.shape[1]))
    env['_square_schedule'],env['_mm'] = schedule,mm
    return env


def namespace(fnp):
    source,specs,compact,proof = source_bundle()
    env = _namespace(fnp,source,specs)
    second = _namespace(fnp,source,compact)
    # Two independent frame dictionaries keep the actual first/warm checks.
    env['_ROOT_FRAME_CACHE'] = second['_PACKED_FRAME_CACHE']
    env['_COMPACT_ENV'] = second
    return env,proof


def setup(fnp,env):
    arena = fnp.empty(ARENA_ELEMENTS,dtype=fnp.float32)
    original = env['_cp3_bind_arena_views'](arena,env['_PACKED_SQUARE5'])
    second = env['_COMPACT_ENV']
    compact = second['_cp3_bind_arena_views'](arena,second['_PACKED_SQUARE5'])
    env['_SCRATCH'] = [arena,None,None,None,None,original,None]
    second['_SCRATCH'] = [arena,None,None,None,None,compact,None]
    return arena,original,compact


def compact_gram(fnp,env,left,right):
    return env['_COMPACT_ENV']['_resident_gram512'](left,right)


def clear(env):
    env['_PACKED_FRAME_CACHE'].clear()
    env['_ROOT_FRAME_CACHE'].clear()
    env['_SCRATCH'] = None
    env['_COMPACT_ENV']['_SCRATCH'] = None
