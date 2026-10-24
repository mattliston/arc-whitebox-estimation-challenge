"""Inert fixed three-WHOLE-root D5 Gram unit; no candidate or array at import.

The original eleven forward primitives and seven unwind primitives are taken
byte-for-byte from BOX. Only stages after each root's first transform batch
the three roots. The root axis follows all recursion axes.
"""
import ast
from fractions import Fraction
import hashlib
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
BOX = NEW/'current_comp_push_frontier_boxes_v1/rpc37_v1/estimator.py'
BOX_SHA = '378092f76102b83ddcff445adaf27309ae8136d1df6844599dad59b69cd75dfa'
GRAM = NEW/'resident_gram512_v1/gram.py'
GRAM_SHA = 'ffe6800e527f9dc44984d87ff48cea988b26cf8d3b8c1ebe6ed73074dd05fc39'
SYNTHETIC = GRAM.with_name('check_synthetic.py')
SYNTHETIC_SHA = 'b23a39cfc9aa06ed475120aaa4d6dabe9c93a48067143f1aae9864d918ce5e3b'
DESIGN = NEW/'current_comp_box_rpc_profile_v1/ROOT_BATCH_ATTRIBUTION_AND_UNIT_DESIGN_20260913_UTC.md'
DESIGN_SHA = 'd64e5d16893037cb54ffed17aa1875379da044343de65aa36664cba6541d2cf6'
PINS = {BOX:BOX_SHA,GRAM:GRAM_SHA,SYNTHETIC:SYNTHETIC_SHA,DESIGN:DESIGN_SHA}
ARENA_ELEMENTS = 292225024
WARM_FLOPS = 955654144
SERIAL_BIND_FLOPS, BATCH_BIND_FLOPS = 172392448, 517177344


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def _edit(source, before, after, changes):
    require(source.count(before)==1, 'Unique whole-root source splice: '+before[:90])
    at = source.index(before)
    changes.append((at,before,after))
    return source[:at]+after+source[at+len(before):]


def _inverse(source, original, changes):
    for at,before,after in reversed(changes):
        require(source[at:at+len(after)]==after, 'Exact positional source inverse')
        source = source[:at]+before+source[at+len(after):]
    require(source==original, 'Entire original function restored')


def source_bundle(tick=None):
    for path,digest in PINS.items():
        require(not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest()==digest,
                'Frozen primitive/prospective owner: '+str(path))
        if tick:
            tick()
    raw = BOX.read_bytes()
    tree = ast.parse(raw.decode())
    offsets = [0]
    for line in raw.splitlines(keepends=True):
        offsets.append(offsets[-1]+len(line))
    def span(node):
        return raw[offsets[node.lineno-1]+node.col_offset:
                   offsets[node.end_lineno-1]+node.end_col_offset].decode()
    def function(name):
        nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
        require(nodes, 'Exact named primitive owner: '+name)
        return span(nodes[-1])
    def literal(name):
        nodes = [n for n in tree.body if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
        require(len(nodes)==1, 'One fixed layout/schedule literal: '+name)
        return ast.literal_eval(nodes[0].value)
    names = ('_cp3_bind_arena_views','_packed_frame','_packed_product_cached',
             '_packed_product_5','_square_product','_resident_gram512')
    originals = {name:function(name) for name in names}
    gram_text=GRAM.read_text()
    gram_nodes=[n for n in ast.parse(gram_text).body if isinstance(n,ast.FunctionDef)
                and n.name=='_resident_gram512']
    require(len(gram_nodes)==1,'One original resident helper, excluding its module docstring')
    gram_node=gram_nodes[0]
    gram_function=''.join(gram_text.splitlines(keepends=True)[gram_node.lineno-1:gram_node.end_lineno])
    require(originals['_resident_gram512'].strip()==gram_function.strip(),
            'Actual current resident helper equals its qualified source owner')
    require(literal('_SQUARE_EQUAL_DEPTHS')[15][31]==5,
            'Actual 512x1024x512 current schedule is D5 without padding')
    specs = literal('_PACKED_SQUARE5')
    batch_specs = tuple((3*start,3*stop,(*shape[:-2],3,*shape[-2:]))
                        for start,stop,shape in specs)
    require(sum(math.prod(shape) for _,_,shape in specs)==SERIAL_BIND_FLOPS
            and sum(math.prod(shape) for _,_,shape in batch_specs)==BATCH_BIND_FLOPS
            and max(stop for _,stop,_ in batch_specs)*4==767090688
            and max(stop for _,stop,_ in batch_specs)<=ARENA_ELEMENTS,
            'Paid original/new view bindings and one existing arena')
    for start,stop,shape in batch_specs:
        require(stop-start==math.prod(shape), 'Exact batch layout span, no hidden storage')

    original = originals['_packed_frame']
    frame,changes = original,[]
    frame = _edit(frame,'def _packed_frame(', 'def _root_frame(',changes)
    # Cache owner only; all forward/downstream interval geometry is unchanged.
    positions = []
    while '_PACKED_FRAME_CACHE' in frame:
        at = frame.index('_PACKED_FRAME_CACHE')
        positions.append((at,'_PACKED_FRAME_CACHE','_ROOT_FRAME_CACHE'))
        frame = frame[:at]+'_ROOT_FRAME_CACHE'+frame[at+len('_PACKED_FRAME_CACHE'):]
    changes.extend(positions)
    frame = _edit(frame,'    frame = (forward, old_a, old_b, leaf, backward)',
        '    first = forward[0]\n'
        '    root_first = tuple((tuple(x[root] for x in first[2]),\n'
        '                        tuple(x[root] for x in first[3]),\n'
        '                        first[4][:,root],first[5][:,root]) for root in range(3))\n'
        '    frame = (forward, old_a, old_b, leaf, backward, root_first)',changes)
    _inverse(frame,original,changes)

    original = originals['_packed_product_cached']
    product,changes = original,[]
    start = original.index('    for index,(aq,bq,aa,bb,acopies,bcopies)')
    end = original.index('    fnp.matmul(leaf_a,leaf_b,out=leaf)')
    transform = original[original.index('        fnp.add(a21,a22,out=aa[4])'):end]
    require(len(transform.splitlines())==11, 'Exactly eleven actual first-forward primitives')
    quarters = original[original.index('            m2,k2,n2 = m//2,k//2,n//2'):
                        original.index('        else:\n            a11,a12,a21,a22=aq')]
    first = ('    pairs=((left[:512,:],right[:,:512]),\n'
             '           (left[:512,:],right[:,512:]),\n'
             '           (left[512:,:],right[:,512:]))\n'
             '    for root,(a,b) in enumerate(pairs):\n'
             '        aa,bb,acopies,bcopies=root_first[root]\n')
    first += ''.join(line[4:]+'\n' for line in quarters.splitlines())
    first += transform
    first += ('    del pairs,a,b,aa,bb,acopies,bcopies\n'
              '    for aq,bq,aa,bb,acopies,bcopies in forward[1:]:\n'
              '        a11,a12,a21,a22=aq\n'
              '        b11,b12,b21,b22=bq\n')+transform
    product = _edit(product,original[start:end],first,changes)
    product = _edit(product,'def _packed_product_cached(a,b,out,views,levels):',
                    'def _root_product(left,right,out,views):',changes)
    product = _edit(product,
        '    m,k,n = int(a.shape[-2]),int(a.shape[-1]),int(b.shape[-1])\n'
        '    assert int(b.shape[-2])==k and tuple(out.shape)==(m,n)\n'
        '    assert m%2**levels==k%2**levels==n%2**levels==0\n'
        '    forward,leaf_a,leaf_b,leaf,backward = _packed_frame(levels,m,k,n,views)',
        '    m,k,n,levels=512,1024,512,5\n'
        '    assert tuple(left.shape)==tuple(right.shape)==(1024,1024)\n'
        '    assert tuple(out.shape)==(3,m,n)\n'
        '    assert str(left.dtype)==str(right.dtype)==str(out.dtype)=="float32"\n'
        '    forward,leaf_a,leaf_b,leaf,backward,root_first = _root_frame(levels,m,k,n,views)',changes)
    product = _edit(product,
        'corners=(out[:rows,:columns],out[:rows,columns:],out[rows:,:columns],out[rows:,columns:])',
        'corners=(out[...,:rows,:columns],out[...,:rows,columns:],out[...,rows:,:columns],out[...,rows:,columns:])',changes)
    _inverse(product,original,changes)
    source = '\n\n'.join(originals.values())+'\n\n'+frame+'\n\n'+product+'\n'
    certificate = cost_certificate()
    proof = dict(parent_sha256=BOX_SHA,resident_owner_sha256=GRAM_SHA,
        exact_original_functions={name:hashlib.sha256(text.encode()).hexdigest()
                                  for name,text in originals.items()},
        whole_frame_inverse=True,whole_product_inverse=True,first_forward_ops_per_root=11,
        original_scalar_tree_and_unwinds=True,root_axis_after_recursion_axes=True,
        no_raw_input_stack=True,no_estimator_module_executed=True,
        derived_source_sha256=hashlib.sha256(source.encode()).hexdigest(),cost=certificate)
    return source,specs,batch_specs,proof


def cost_certificate():
    m,k,n,d=512,1024,512,5
    total = 7**d*(m//2**d)*(n//2**d)*(2*k//2**d-1)
    total += Fraction(7,4)*(m*k+k*n+m*n)*sum(Fraction(7,4)**i for i in range(d))
    require(total.denominator==1 and 3*total+1024**2==WARM_FLOPS,
            'Exact unchanged scalar tariff including four assembly copies')
    fixed = 4*WARM_FLOPS+2*1024**2*(2*1024-1)+SERIAL_BIND_FLOPS+BATCH_BIND_FLOPS
    require(fixed==8805056512, 'Complete four cores, dense FP64 reference and bindings')
    return dict(geometry=[512,1024,512],depth=5,roots=3,seed=938248,
        warm_helper_flops=WARM_FLOPS,serial_bind_flops=SERIAL_BIND_FLOPS,
        batch_bind_flops=BATCH_BIND_FLOPS,fixed_core_reference_binding_flops=fixed,
        remaining_input_export_guard_allowance_flops=10_000_000_000-fixed,
        complete_context_cap_flops=10_000_000_000,
        original_arena_bytes=ARENA_ELEMENTS*4,batch_live_scratch_bytes=767090688,
        serial_core_arithmetic_calls=273,batch_core_arithmetic_calls=113,
        serial_noncache_helper_calls=281,batch_noncache_helper_calls=119)


def namespace(fnp):
    source,specs,batch_specs,proof = source_bundle()
    env = dict(fnp=fnp,_PACKED_FRAME_CACHE={},_ROOT_FRAME_CACHE={},_SCRATCH=None,
               _PACKED_SQUARE5=specs,_ROOT_SPECS=batch_specs,_ROOT_VIEWS=None)
    exec(compile(source,'root-batch:exact-current-D5-primitives','exec'),env)
    def schedule(m,k,n):
        require((m,k,n)==(512,1024,512), 'Only the fixed original D5 Gram geometry')
        return 5,(m,k,n)
    def mm(a,b):
        require(str(a.dtype)==str(b.dtype)=='float32', 'Original FP32 helper operands')
        return env['_square_product'](a,b,int(a.shape[0]),int(a.shape[1]),int(b.shape[1]))
    env['_square_schedule'],env['_mm']=schedule,mm
    return env,proof


def setup(fnp,env):
    arena=fnp.empty(ARENA_ELEMENTS,dtype=fnp.float32)
    serial=env['_cp3_bind_arena_views'](arena,env['_PACKED_SQUARE5'])
    batch=env['_cp3_bind_arena_views'](arena,env['_ROOT_SPECS'])
    env['_SCRATCH']=[arena,None,None,None,None,serial,None]
    env['_ROOT_VIEWS']=batch
    return arena,serial,batch


def batched_gram(fnp,env,left,right):
    parts=fnp.empty((3,512,512),dtype=fnp.float32)
    env['_root_product'](left,right,parts,env['_ROOT_VIEWS'])
    out=fnp.empty((1024,1024),dtype=fnp.float32)
    fnp.copyto(out[:512,:512],parts[0])
    fnp.copyto(out[:512,512:],parts[1])
    fnp.copyto(out[512:,:512],parts[1].T)
    fnp.copyto(out[512:,512:],parts[2])
    del parts
    return out


def clear(env):
    env['_PACKED_FRAME_CACHE'].clear()
    env['_ROOT_FRAME_CACHE'].clear()
    env['_SCRATCH']=env['_ROOT_VIEWS']=None
