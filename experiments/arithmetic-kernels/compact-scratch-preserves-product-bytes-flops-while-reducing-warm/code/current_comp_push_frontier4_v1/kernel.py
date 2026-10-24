"""DRAFT: fixed D6 breadth-four frontier with depth-two packets800,800,801.

Inert stdlib source/shape algebra only at import. No estimator, fixture, stage,
numerical library, or candidate is imported or executed. Numerical entry points
require the caller's already-qualified FNP namespace and paid context.
"""
import ast
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent/'current_comp_shared_gain_slice_v1/build_v1/estimator.py'
PARENT_SHA = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
MICRO = HERE.parent/'current_comp_push_microbatch3_v1/kernel.py'
MICRO_SHA = '90abc9bcd8ef543c5e3d4306b540af0a72a7210f0333e30e869e52a3b258a6fb'
PINS = {str(PARENT):PARENT_SHA, str(MICRO):MICRO_SHA}
SHAPE = (1024,384,8192)
CAPACITY = (1024,1024,8192)  # Historical controls only; frontier is SHAPE-only.
ARENA_ELEMENTS = 292225024
ARENA_BYTES = 1168900096
PACKETS = ((0,800),(800,1600),(1600,2401))
BATCH_CAPACITY = 801
ROLES = ('d4','serial_d6','microbatch3','frontier4')
COUNTS = {
    'd4':dict(add=28,subtract=32,stack=8,copyto=4,matmul=1),
    'serial_d6':dict(add=252,subtract=288,stack=72,copyto=36,matmul=7),
    'microbatch3':dict(add=112,subtract=128,stack=32,copyto=16,matmul=3),
    'frontier4':dict(add=70,subtract=80,stack=20,copyto=10,matmul=3),
}
SETUP_RESHAPE_CALLS = dict(d4=48,serial_d6=116,microbatch3=116,frontier4=109)
COLD_UNSTACK_CALLS = dict(d4=16,serial_d6=24,microbatch3=44,frontier4=32)
_OWNER = None


def require(value, message):
    if not value:
        raise RuntimeError(message)


def authenticate(tick=None):
    for name, expected in PINS.items():
        if tick is not None:
            tick()
        require(hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,
                'Exact direct kernel source owner: '+name)
    if tick is not None:
        tick()
    return dict(PINS)


def _owner():
    global _OWNER
    if _OWNER is None:
        authenticate()
        namespace = dict(__file__=str(MICRO),__name__='_reviewed_micro3_kernel_owner')
        exec(compile(MICRO.read_text(),str(MICRO),'exec'),namespace)
        _OWNER = namespace
    return _OWNER


def source_bundle():
    return _owner()['source_bundle']()


def _functions(source):
    lines = source.splitlines(keepends=True)
    offsets, position = [], 0
    for line in lines:
        offsets.append(position)
        position += len(line)
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node,ast.FunctionDef):
            require(source.isascii(),'These extracted kernel spans must be ASCII')
            start = offsets[node.lineno-1]+node.col_offset
            stop = offsets[node.end_lineno-1]+node.end_col_offset
            result[node.name] = source[start:stop]
    return result


def child_source(bundle):
    """Whole-span inverse to micro3, whose exact parent inverse is rerun."""
    owner = _owner()
    original = owner['child_source'](bundle)
    functions = _functions(original)
    frame = owner['_adapt'](functions['_micro3_frame'],(
        ('def _micro3_frame(', 'def _frontier2_frame('),
    ))
    child = owner['_adapt'](functions['_micro3_child'],(
        ('def _micro3_child(', 'def _frontier2_child('),
        ('assert batch in (1,3) and int(b.shape[0])==batch',
         'assert batch in (800,801) and int(b.shape[0])==batch'),
        ('_micro3_frame(levels,m,k,n,views,batch)',
         '_frontier2_frame(levels,m,k,n,views,batch)'),
    ))
    return frame+'\n\n'+child+'\n'


def frontier_source(bundle):
    """Exact original four-level transforms/unwind; replace ONLY the leaf."""
    original = _functions(bundle)['_packed_product_cached']
    return _owner()['_adapt'](original,(
        ('def _packed_product_cached(a,b,out,views,levels):',
         'def _frontier4_body(a,b,out,binding):\n'
         '    assert binding.busy\n'
         '    levels = 4'),
        ('forward,leaf_a,leaf_b,leaf,backward = _packed_frame(levels,m,k,n,views)',
         'forward,leaf_a,leaf_b,leaf,backward = binding.frames["upper"]'),
        ('    fnp.matmul(leaf_a,leaf_b,out=leaf)',
         '    for left,right,destination in binding.groups["packets"]:\n'
         '        _frontier2_child(left,right,destination,binding.child_views,2)'),
    ))+'\n'


def layout_metadata():
    """Integer-only actual view tariff and conservative simultaneous-value proof.

    All named views are bound once. Overlapping aliases do NOT imply live
    values overlap: every phase below lists the complete arena value set.
    Input/output fixture owners and the one original arena are separate.
    """
    m,k,n = SHAPE
    q = Fraction(7,4)
    H = {j:int((m*k+k*n)*q**j) for j in range(1,5)}
    P = {j:int(m*n*q**j) for j in range(1,5)}
    require(H=={1:6193152,2:10838016,3:18966528,4:33191424}
        and P=={1:14680064,2:25690112,3:44957696,4:78675968},
        'Fixed frontier dimensions, not a shape optimizer')
    h4, child_start = H[4], H[4]+P[4]
    child_elements = BATCH_CAPACITY*200704
    required = child_start+child_elements
    require(required==272631296 and required<=ARENA_ELEMENTS,'Original arena cap')
    names, specs, regions = [], [], {}

    def add(name,start,shape):
        stop = start+math.prod(shape)
        require(name not in regions and 0<=start<stop<=ARENA_ELEMENTS,
                'Unique bounded actual view '+name)
        names.append(name); specs.append((start,stop,tuple(shape)))
        regions[name] = (start,stop)

    # Upper forward buffers alternate. Level4 must finish in [0,H4).
    for j in range(1,5):
        start = 0 if j%2==0 else h4
        ashape = (7,)*j+(m//2**j,k//2**j)
        bshape = (7,)*j+(k//2**j,n//2**j)
        add('a'+str(j),start,ashape)
        add('b'+str(j),start+math.prod(ashape),bshape)
    # Upper output frontiers alternate only AFTER all three child packets.
    for j in range(4,0,-1):
        start = h4 if j%2==0 else child_start
        add('p'+str(j),start,(7,)*j+(m//2**j,n//2**j))
    for j in range(4,0,-1):
        start = (h4 if (j-1)%2==0 else child_start)+P[j-1] if j>1 else h4
        add('s'+str(j),start,(3,)+(7,)*(j-1)+(m//2**j,n//2**j))
    # These three paid aliases merge ONLY the already contiguous prefix axes.
    add('flat_a',regions['a4'][0],(2401,64,24))
    add('flat_b',regions['b4'][0],(2401,24,512))
    add('flat_p',regions['p4'][0],(2401,64,512))

    # One reusable depth-two packet region. AB2 is dead before P1/S2 reuse it.
    leaf_elements = BATCH_CAPACITY*100352
    first_output_elements = BATCH_CAPACITY*57344
    for j in (1,2):
        start = child_start if j==1 else child_start+leaf_elements
        ashape = (7,)*j+(BATCH_CAPACITY,64//2**j,24//2**j)
        bshape = (7,)*j+(BATCH_CAPACITY,24//2**j,512//2**j)
        add('ca'+str(j),start,ashape)
        add('cb'+str(j),start+math.prod(ashape),bshape)
    add('cp2',child_start,(7,7,BATCH_CAPACITY,16,128))
    add('cp1',child_start+leaf_elements,(7,BATCH_CAPACITY,32,256))
    add('cs2',child_start+leaf_elements+first_output_elements,(3,7,BATCH_CAPACITY,16,128))
    add('cs1',child_start,(3,BATCH_CAPACITY,32,256))
    require(len(specs)==27 and sum(stop-start for start,stop,shape in specs)==649085920,
            'All27 bindings, including three duplicate-view charges')

    phases = []
    for j in range(1,5):
        live = ['a'+str(j),'b'+str(j)]
        if j>1:
            live += ['a'+str(j-1),'b'+str(j-1)]
        phases.append(('upper_forward'+str(j),live))
    retained = ['a4','b4','p4']
    for packet in range(3):
        for label,live in (
            ('forward1',['ca1','cb1']),
            ('forward2',['ca1','cb1','ca2','cb2']),
            ('leaf',['ca2','cb2','cp2']),
            ('unwind2',['cp2','cp1','cs2']),
            ('unwind1',['cp1','cs1']),
        ):
            phases.append(('packet'+str(packet)+'_'+label,retained+live))
    for j in range(4,0,-1):
        live = ['p'+str(j),'s'+str(j)]
        if j>1:
            live += ['p'+str(j-1)]
        phases.append(('upper_unwind'+str(j),live))
    for label,live in phases:
        for index,name in enumerate(live):
            start,stop = regions[name]
            for other in live[index+1:]:
                a,b = regions[other]
                require(stop<=a or b<=start,'Live values alias: '+label+'/'+name+'/'+other)
    require(max(stop for start,stop in regions.values())==required,
            'Reported high water includes every bound view')
    return dict(names=names,specs=tuple(specs),regions=regions,phases=phases,
        capacity=SHAPE,packets=PACKETS,required_elements=required,
        scratch_bytes=4*required,original_arena_bytes=ARENA_BYTES,
        binding_flops=649085920,binding_reshape_calls=27,
        additional_cold_binding_flops=370519264,
        full_card_extra_if_general_serial_binding_retained=649085920,
        exact_simultaneous_value_intervals_disjoint=True,
        upper_frontier_elements=h4+P[4],child_elements=child_elements)


def _legacy_views(bound,prefix,levels):
    # The original cached helper SKIPS its two unused dependency lists. None
    # preserves their tuple positions without inventing new numerical views.
    views = tuple(bound[prefix+side+str(j)] for j in range(1,levels+1) for side in ('a','b'))
    views += (None,)*(2*levels)
    views += tuple(bound[prefix+'p'+str(j)] for j in range(levels,0,-1))
    views += tuple(bound[prefix+'s'+str(j)] for j in range(levels,0,-1))
    require(len(views)==6*levels,'Exact unchanged cached-frame view slots')
    return views


class _FrontierBinding:
    def __init__(self,bound):
        self.upper_views = _legacy_views(bound,'',4)
        self.child_views = _legacy_views(bound,'c',2)
        self.flat_a,self.flat_b,self.flat_p = (bound[name] for name in ('flat_a','flat_b','flat_p'))
        self.frames,self.groups = {},{}
        self.busy = False


def source_proof():
    """The same complete, target-free proof returned by make_namespace."""
    owner = _owner()
    bundle,literals,proof = source_bundle()
    micro_source = owner['child_source'](bundle)
    child,frontier = child_source(bundle),frontier_source(bundle)
    metadata = layout_metadata()
    proof.update(frontier_child_source_sha256=hashlib.sha256(child.encode()).hexdigest(),
        frontier_source_sha256=hashlib.sha256(frontier.encode()).hexdigest(),
        frontier_source_bindings=authenticate(),frontier_layout=metadata,
        control_kernel_proof=dict(
            batch_source_sha256=hashlib.sha256(micro_source.encode()).hexdigest(),
            original_complete_child_and_top_inverses=True,
            roles=['d4','serial_d6','microbatch3']),
        original_complete_child_and_upper_inverses=True,
        fixed_four_level_frontier_then_three_depth2_packets=True,
        unchanged_six_level_scalar_tree_and_leaf_dimensions=True,
        physical_terms_or_heads_changed=False,no_depth_or_shape_policy_search=True,
        frontier_output_byte_identity_is_a_runtime_gate=True)
    return literals,json.loads(json.dumps(proof,sort_keys=True,allow_nan=False))


def make_namespace(fnp):
    owner = _owner()
    namespace,literals,control_proof = owner['make_namespace'](fnp)
    bundle,unused,original_proof = source_bundle()
    child,frontier = child_source(bundle),frontier_source(bundle)
    exec(compile(child,'<exact-parent-inverse-frontier-depth2>','exec'),namespace)
    exec(compile(frontier,'<exact-parent-inverse-frontier-upper4>','exec'),namespace)
    checked,proof = source_proof()
    require(checked==literals,'Same authenticated control literals')
    namespace['_FRONTIER_METADATA'] = layout_metadata()
    return namespace,literals,proof


def setup(fnp,ns,literals,role,left,right):
    require(role in ROLES,'Only four prospectively fixed roles')
    if role!='frontier4':
        require(_OWNER is not None,'Control owner must be admitted before paid setup')
        return _OWNER['setup'](fnp,ns,literals,role,left,right)
    require(tuple(left.shape)==SHAPE[:2] and tuple(right.shape)==SHAPE[1:]
        and str(left.dtype)==str(right.dtype)=='float32','Only fixed synthetic inputs')
    arena = fnp.empty((ARENA_ELEMENTS,),dtype=fnp.float32)
    bind = ns['_cp3_bind_arena_views']
    old_push = bind(arena,literals['_PACKED_PUSH'])
    old_read = bind(arena,literals['_PACKED_READ'])
    metadata = ns['_FRONTIER_METADATA']
    values = bind(arena,metadata['specs'])
    bound = dict(zip(metadata['names'],values))
    state = _FrontierBinding(bound)
    extra_read = bind(arena,literals['_P1_D6_832_READ_SPECS'])
    output = fnp.empty((SHAPE[0],SHAPE[2]),dtype=fnp.float32)
    return dict(arena=arena,old_push=old_push,old_read=old_read,extra_read=extra_read,
                state=state,output=output)


def product(ns,role,left,right,state):
    require(role in ROLES,'Only four fixed roles')
    if role!='frontier4':
        require(_OWNER is not None,'Control owner must be admitted before timed product')
        return _OWNER['product'](ns,role,left,right,state)
    require(tuple(left.shape)==SHAPE[:2] and tuple(right.shape)==SHAPE[1:]
        and tuple(state['output'].shape)==(SHAPE[0],SHAPE[2])
        and str(left.dtype)==str(right.dtype)==str(state['output'].dtype)=='float32',
        'Exact fixed typed input and caller-owned output')
    binding = state['state']
    require(isinstance(binding,_FrontierBinding) and not binding.busy,
            'Single serial arena owner; no reentrancy or nested scratch consumer')
    binding.busy = True
    try:
        if not binding.frames:
            binding.frames['upper'] = ns['_packed_frame'](4,*SHAPE,binding.upper_views)
            binding.groups['packets'] = tuple(
                (binding.flat_a[start:stop],binding.flat_b[start:stop],binding.flat_p[start:stop])
                for start,stop in PACKETS)
        return ns['_frontier4_body'](left,right,state['output'],binding)
    finally:
        binding.busy = False


def price(role):
    require(role in ROLES,'Fixed role tariff')
    if role!='frontier4':
        return _owner()['price'](role)
    m,k,n = SHAPE
    q = Fraction(7,4)
    total = 7**6*(m//64)*(n//64)*(2*k//64-1)+q*(m*k+k*n+m*n)*sum(q**j for j in range(6))
    require(total==3421951456 and sum(COUNTS[role].values())==183,
            'Full leaf/arithmetic/copy tariff, not just nonzero multiply count')
    return int(total)


def setup_price(literals,role):
    require(role in ROLES,'Fixed setup role')
    if role!='frontier4':
        return _owner()['setup_price'](literals,role)
    base = sum(stop-start for key in ('_PACKED_PUSH','_PACKED_READ')
               for start,stop,shape in literals[key])
    read = literals['_P1_D6_832_READ_INFO']['binding_flops']
    require(base==1068953600 and read==282489344,'Actual unchanged original bindings')
    total = base+read+layout_metadata()['binding_flops']
    require(total==2000528864,'Complete isolated-role setup F')
    return total


def setup_operation_counts(role):
    require(role in ROLES,'Fixed setup operation-count owner')
    return dict(empty=2,reshape=SETUP_RESHAPE_CALLS[role])


def cold_extra_operation_counts(role):
    require(role in ROLES,'Fixed first-product view-count owner')
    return dict(unstack=COLD_UNSTACK_CALLS[role])


def initial_operation_counts(role):
    require(role in ROLES,'Fixed first-product complete operation counts')
    return dict(COUNTS[role],**cold_extra_operation_counts(role))
