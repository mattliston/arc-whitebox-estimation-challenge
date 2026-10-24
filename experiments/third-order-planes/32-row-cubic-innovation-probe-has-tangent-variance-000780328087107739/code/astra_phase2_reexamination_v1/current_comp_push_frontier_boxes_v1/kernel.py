"""DRAFT, source only: two fixed capacity boxes for the tested frontier4 circuit.

Only stdlib metadata is evaluated here. source_and_proof() returns an ASCII
appendix; it never imports a candidate, numerical package, fixture or target,
creates an array, or writes a build/stage. The appendix runs only in the paid
candidate after separate source review, admission and explicit authorization.
"""
import ast
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
PARENT = NEW/'current_comp_shared_gain_slice_v1/build_v1/estimator.py'
PARENT_SHA = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
FRONT = NEW/'current_comp_push_frontier4_v1/kernel.py'
FRONT_SHA = '16e121a0f557bd1705ca5c620e1125ed157b463c1421f5bfad8b162d8fba2c46'
FRONT_PROTOCOL = FRONT.parent/'PROTOCOL.md'
FRONT_PROTOCOL_SHA = '56321ed32ce1927ff0530e9e1e62ebddeee9d970551422a0a09c9aa2b7400256'
MICRO = NEW/'current_comp_push_microbatch3_v1/kernel.py'
MICRO_SHA = '90abc9bcd8ef543c5e3d4306b540af0a72a7210f0333e30e869e52a3b258a6fb'
PINS = {str(PARENT):PARENT_SHA, str(FRONT):FRONT_SHA,
        str(FRONT_PROTOCOL):FRONT_PROTOCOL_SHA, str(MICRO):MICRO_SHA}
CAPACITIES = ((1024,384,8192),(1024,1024,6144))
PACKETS = ((0,800),(800,1600),(1600,2401))
ARENA_ELEMENTS = 292225024
ARENA_BYTES = 1168900096
BATCH_CAPACITY = 801
EXTRA_COLD_FLOPS = 1342018272
PARENT_INITIALIZATION_FLOPS = 1944970298
PLANNED_INITIALIZATION_FLOPS = 3286988570
COUNTS = dict(add=70,subtract=80,stack=20,copyto=10,matmul=3)
_OWNER = None


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read_source(path, digest, tick=None):
    path = Path(path)
    if tick is not None:
        tick()
    require(path.is_file() and not path.is_symlink(),'Exact regular source owner')
    before = path.stat()
    require(0 < before.st_size <= 3*1024**2,'Bounded source owner')
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


def authenticate(tick=None):
    for path,digest in PINS.items():
        read_source(path,digest,tick)
    return dict(PINS)


def _owner(tick=None):
    global _OWNER
    authenticate(tick)
    if _OWNER is None:
        namespace = dict(__name__='_reviewed_fixed_frontier_source_owner',__file__=str(FRONT))
        exec(compile(read_source(FRONT,FRONT_SHA,tick),str(FRONT),'exec'),namespace)
        _OWNER = namespace
    return _OWNER


def _adapt(source, edits):
    changed,positions = source,[]
    for before,after in edits:
        require(before and after and changed.count(before)==1,'Unique nonempty source splice')
        at = changed.index(before)
        positions.append((at,before,after))
        changed = changed[:at]+after+changed[at+len(before):]
    inverse = changed
    for at,before,after in reversed(positions):
        require(inverse[at:at+len(after)]==after,'Exact positional source inverse')
        inverse = inverse[:at]+before+inverse[at+len(after):]
    require(inverse==source,'Entire tested source span restored byte-exactly')
    return changed


def layout_metadata(capacity):
    """Two fixed integer layouts; no search, arrays, input geometry or padding."""
    capacity = tuple(capacity)
    require(capacity in CAPACITIES,'Only the two prospective capacity boxes')
    m,k,n = capacity
    q = Fraction(7,4)
    H = {j:int((m*k+k*n)*q**j) for j in range(1,5)}
    P = {j:int(m*n*q**j) for j in range(1,5)}
    cm,ck,cn = m//16,k//16,n//16
    leaf_per_lane = 49*(cm//4)*(cn//4)
    first_output_per_lane = 7*(cm//2)*(cn//2)
    second_AB_per_lane = 49*((cm//4)*(ck//4)+(ck//4)*(cn//4))
    h4,child_start = H[4],H[4]+P[4]
    # At box A the two-output unwind is largest; at box B the leaf+AB2 is.
    child_elements = BATCH_CAPACITY*max(2*leaf_per_lane,leaf_per_lane+second_AB_per_lane)
    required = child_start+child_elements
    expected = {CAPACITIES[0]:(33191424,78675968,160763904,272631296,649085920),
                CAPACITIES[1]:(68841472,59006976,130620672,258469120,692932352)}[capacity]
    require((h4,P[4],child_elements,required)==expected[:4]
        and required<=ARENA_ELEMENTS,'Fixed capacity and original arena ceiling')
    names,specs,regions = [],[],{}

    def add(name,start,shape):
        stop = start+math.prod(shape)
        require(name not in regions and 0<=start<stop<=ARENA_ELEMENTS,'Unique bounded view')
        names.append(name); specs.append((start,stop,tuple(shape)))
        regions[name] = (start,stop)

    for j in range(1,5):
        start = 0 if j%2==0 else h4
        ashape,bshape = (7,)*j+(m//2**j,k//2**j),(7,)*j+(k//2**j,n//2**j)
        add('a'+str(j),start,ashape)
        add('b'+str(j),start+math.prod(ashape),bshape)
    for j in range(4,0,-1):
        add('p'+str(j),h4 if j%2==0 else child_start,(7,)*j+(m//2**j,n//2**j))
    for j in range(4,0,-1):
        start = (h4 if (j-1)%2==0 else child_start)+P[j-1] if j>1 else h4
        add('s'+str(j),start,(3,)+(7,)*(j-1)+(m//2**j,n//2**j))
    # Bind full contiguous capacity aliases ONCE, then slice actual dimensions.
    # Never reshape a narrowed matrix whose capacity row strides contain holes.
    add('flat_a',regions['a4'][0],(2401,cm,ck))
    add('flat_b',regions['b4'][0],(2401,ck,cn))
    add('flat_p',regions['p4'][0],(2401,cm,cn))
    leaf_elements = BATCH_CAPACITY*leaf_per_lane
    first_output_elements = BATCH_CAPACITY*first_output_per_lane
    for j in (1,2):
        start = child_start if j==1 else child_start+leaf_elements
        ashape = (7,)*j+(BATCH_CAPACITY,cm//2**j,ck//2**j)
        bshape = (7,)*j+(BATCH_CAPACITY,ck//2**j,cn//2**j)
        add('ca'+str(j),start,ashape)
        add('cb'+str(j),start+math.prod(ashape),bshape)
    add('cp2',child_start,(7,7,BATCH_CAPACITY,cm//4,cn//4))
    add('cp1',child_start+leaf_elements,(7,BATCH_CAPACITY,cm//2,cn//2))
    add('cs2',child_start+leaf_elements+first_output_elements,(3,7,BATCH_CAPACITY,cm//4,cn//4))
    add('cs1',child_start,(3,BATCH_CAPACITY,cm//2,cn//2))
    tariff = sum(stop-start for start,stop,shape in specs)
    require(len(specs)==27 and tariff==expected[4],'Every paid alias is charged')

    phases = []
    for j in range(1,5):
        live = ['a'+str(j),'b'+str(j)]
        if j>1:
            live += ['a'+str(j-1),'b'+str(j-1)]
        phases.append(('upper_forward'+str(j),live))
    retained = ['a4','b4','p4']
    for packet in range(3):
        for label,live in (('forward1',['ca1','cb1']),
            ('forward2',['ca1','cb1','ca2','cb2']),('leaf',['ca2','cb2','cp2']),
            ('unwind2',['cp2','cp1','cs2']),('unwind1',['cp1','cs1'])):
            phases.append(('packet'+str(packet)+'_'+label,retained+live))
    for j in range(4,0,-1):
        live = ['p'+str(j),'s'+str(j)]
        if j>1:
            live += ['p'+str(j-1)]
        phases.append(('upper_unwind'+str(j),live))
    require(len(phases)==23,'All four forward, fifteen child and four unwind phases')
    for label,live in phases:
        for index,name in enumerate(live):
            start,stop = regions[name]
            for other in live[index+1:]:
                a,b = regions[other]
                require(stop<=a or b<=start,'Live values overlap: '+label+'/'+name+'/'+other)
    require(max(stop for start,stop in regions.values())==required,'Full high water')
    return dict(names=names,specs=tuple(specs),regions=regions,phases=phases,
        capacity=capacity,packets=PACKETS,required_elements=required,
        scratch_bytes=4*required,original_arena_bytes=ARENA_BYTES,
        binding_flops=tariff,binding_reshape_calls=27,
        exact_simultaneous_value_intervals_disjoint=True,
        upper_frontier_elements=child_start,child_elements=child_elements)


def price(m,k,n):
    require(m==1024 and 0<k<=1024 and 1024<n<=8192
        and k%64==n%64==0 and (k==384 or n<=6144),'Covered original PUSH dispatch only')
    q = Fraction(7,4)
    value = 7**6*(m//64)*(n//64)*(2*k//64-1)+q*(m*k+k*n+m*n)*sum(q**j for j in range(6))
    require(value.denominator==1,'Complete integer D6 transform/copy/leaf/unwind F')
    return value.numerator


# This appended runtime glue has no import or numerical work at module scope.
# The only persistent owners are basic views of the already existing arena.
RUNTIME_GLUE = '''
_FBX_BINDINGS = None
_FBX_BUSY = False


def _fbx_views(bound,prefix,levels):
    views = tuple(bound[prefix+side+str(j)] for j in range(1,levels+1) for side in ('a','b'))
    views += (None,)*(2*levels)
    views += tuple(bound[prefix+'p'+str(j)] for j in range(levels,0,-1))
    views += tuple(bound[prefix+'s'+str(j)] for j in range(levels,0,-1))
    assert len(views)==6*levels
    return views


class _FBX_Binding:
    def __init__(self,arena,capacity,bound):
        self.arena,self.capacity = arena,capacity
        self.upper_views = _fbx_views(bound,'',4)
        self.child_views = _fbx_views(bound,'c',2)
        self.flat_a,self.flat_b,self.flat_p = (bound[name] for name in ('flat_a','flat_b','flat_p'))
        self.frames,self.groups = {},{}
        self.busy = False


def _fbx_bind(arena):
    global _FBX_BINDINGS
    assert _FBX_BINDINGS is None and not _FBX_BUSY
    assert tuple(arena.shape)==(292225024,) and str(arena.dtype)=='float32'
    assert _SCRATCH is not None and arena is _SCRATCH[0]
    assert _P1_D6_832_BINDINGS is not None
    bindings = []
    for capacity,names,specs in _FBX_LAYOUTS:
        values = _cp3_bind_arena_views(arena,specs)
        bindings.append(_FBX_Binding(arena,capacity,dict(zip(names,values))))
    _FBX_BINDINGS = tuple(bindings)


def _fbx_product(a,b,out,binding):
    global _FBX_BUSY
    assert _FBX_BINDINGS is not None and any(binding is item for item in _FBX_BINDINGS)
    assert not _FBX_BUSY and not binding.busy
    assert _P1_D6_832_BINDINGS is not None and not any(item.busy for item in _P1_D6_832_BINDINGS)
    assert _SCRATCH is not None and binding.arena is _SCRATCH[0]
    assert len(a.shape)==len(b.shape)==len(out.shape)==2
    assert str(a.dtype)==str(b.dtype)==str(out.dtype)=='float32'
    m,k,n = int(a.shape[0]),int(a.shape[1]),int(b.shape[1])
    assert int(b.shape[0])==k and tuple(out.shape)==(m,n)
    assert m==1024 and 0<k<=1024 and 1024<n<=8192 and k%64==n%64==0
    assert all(value<=limit for value,limit in zip((m,k,n),binding.capacity))
    assert (binding is _FBX_BINDINGS[0] and k==384) or (binding is _FBX_BINDINGS[1] and k!=384 and n<=6144)
    _FBX_BUSY = binding.busy = True
    try:
        key = (m,k,n)
        if key not in binding.frames:
            if len(binding.frames)>=128:
                oldest = next(iter(binding.frames))
                binding.frames.pop(oldest)
                binding.groups.pop(oldest)
            binding.frames[key] = _packed_frame(4,m,k,n,binding.upper_views)
            binding.groups[key] = tuple(
                (binding.flat_a[start:stop,:m//16,:k//16],
                 binding.flat_b[start:stop,:k//16,:n//16],
                 binding.flat_p[start:stop,:m//16,:n//16])
                for start,stop in ((0,800),(800,1600),(1600,2401)))
        return _fbx_body(a,b,out,binding)
    finally:
        binding.busy = _FBX_BUSY = False
'''


def source_and_proof(tick=None):
    """Return (ASCII runtime appendix, JSON-native source/layout proof)."""
    owner = _owner(tick)
    if tick is not None:
        tick()
    bundle,literals,original_proof = owner['source_bundle']()
    child = owner['child_source'](bundle)
    functions = owner['_functions'](child)
    frame = _adapt(functions['_frontier2_frame'],(
        ('def _frontier2_frame(', 'def _fbx_child_frame('),))
    product = _adapt(functions['_frontier2_child'],(
        ('def _frontier2_child(', 'def _fbx_child('),
        ('_frontier2_frame(levels,m,k,n,views,batch)', '_fbx_child_frame(levels,m,k,n,views,batch)'),))
    original_upper = owner['frontier_source'](bundle)
    upper = _adapt(original_upper,(
        ('def _frontier4_body(', 'def _fbx_body('),
        ('binding.frames["upper"]', 'binding.frames[(m,k,n)]'),
        ('binding.groups["packets"]', 'binding.groups[(m,k,n)]'),
        ('_frontier2_child(left,right,destination,binding.child_views,2)',
         '_fbx_child(left,right,destination,binding.child_views,2)'),))
    layouts = [layout_metadata(capacity) for capacity in CAPACITIES]
    tested_layout = owner['layout_metadata']()
    for key,value in layouts[0].items():
        require(tested_layout[key]==value,'First box is the exact tested layout: '+key)
    require(sum(x['binding_flops'] for x in layouts)==EXTRA_COLD_FLOPS
        and PARENT_INITIALIZATION_FLOPS+EXTRA_COLD_FLOPS==PLANNED_INITIALIZATION_FLOPS,
        'All retained old bindings plus both new complete cold tariffs')
    require(price(*CAPACITIES[0])==3421951456 and price(*CAPACITIES[1])==6483751680
        and sum(COUNTS.values())==183,'Actual complete shape tariffs and all numerical calls')
    literal = tuple((x['capacity'],tuple(x['names']),x['specs']) for x in layouts)
    appendix = '\n\n# BEGIN FIXED TWO-BOX FRONTIER4 SOURCE APPENDIX\n'
    appendix += '_FBX_LAYOUTS = '+repr(literal)+'\n\n'+frame+'\n\n'+product+'\n\n'+upper+RUNTIME_GLUE
    appendix += '\n# END FIXED TWO-BOX FRONTIER4 SOURCE APPENDIX\n'
    require(appendix.isascii(),'Entire numerical appendix is ASCII')
    tree = ast.parse(appendix)
    require(not any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(tree)),
        'No new numerical/runtime/source imports')
    require(all(isinstance(n,(ast.Assign,ast.FunctionDef,ast.ClassDef)) for n in tree.body),
        'Only inert literals and definitions at candidate import')
    proof = dict(schema='current_comp_push_frontier_boxes_kernel_v1',
        parent_sha256=PARENT_SHA,tested_frontier_source_sha256=FRONT_SHA,
        source_bindings=authenticate(tick),original_kernel_extraction=original_proof,
        tested_child_source_sha256=hashlib.sha256(child.encode()).hexdigest(),
        tested_upper_source_sha256=hashlib.sha256(original_upper.encode()).hexdigest(),
        child_frame_source_sha256=hashlib.sha256(frame.encode()).hexdigest(),
        child_source_sha256=hashlib.sha256(product.encode()).hexdigest(),
        upper_source_sha256=hashlib.sha256(upper.encode()).hexdigest(),
        appendix_sha256=hashlib.sha256(appendix.encode()).hexdigest(),
        whole_tested_child_and_upper_inverses=True,first_capacity_layout_exact_tested=True,
        both_capacity_live_intervals_checked=True,layouts=layouts,
        full_capacity_flat_aliases_before_actual_shape_slices=True,
        actual_shape_upper_and_child_frames=True,packet_sizes=[800,800,801],
        unchanged_six_level_scalar_tree_and_actual_leaf_dimensions=True,
        same_original_arena=True,original_arena_bytes=ARENA_BYTES,
        added_backing_arenas=0,new_paid_binding_reshape_calls=54,
        additional_cold_binding_flops=EXTRA_COLD_FLOPS,
        planned_initialization_flops=PLANNED_INITIALIZATION_FLOPS,
        warm_primitive_counts=COUNTS,warm_primitive_calls=183,
        first_unstack_calls_per_new_shape_with_empty_child_cache=32,
        bounded_shape_caches=128,retained_inputs_outputs=False,
        physical_terms_or_heads_changed=False,model_calls=0,parameter_updates=0,
        truth_accessed=False,runtime_measured=False,full_candidate_byte_parity_measured=False)
    return appendix,json.loads(json.dumps(proof,sort_keys=True,allow_nan=False))


if __name__ == '__main__':
    print('Inert source draft: no candidate, array, build or stage is created.')
