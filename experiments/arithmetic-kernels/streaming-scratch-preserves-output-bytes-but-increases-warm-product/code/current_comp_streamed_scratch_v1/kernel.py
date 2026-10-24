"""Inert fixed D4 scratch-lifetime experiment, not a whole estimator.

The original numerical statements are extracted unchanged. Only storage,
basic views, iteration ownership and explicit retirement are new.
"""
import ast
import hashlib
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
NEW=HERE.parent
BOX=NEW/'current_comp_push_frontier_boxes_v1/rpc37_v1/estimator.py'
BOX_SHA='378092f76102b83ddcff445adaf27309ae8136d1df6844599dad59b69cd75dfa'
COMPACT=NEW/'current_comp_compact_boxes_v2/rpc37_v1/estimator.py'
COMPACT_SHA='b279c9b723273a71fe4d6a2f600367e24465a746f7a8b055a8b9f78a5df80857'
PINS={BOX:BOX_SHA,COMPACT:COMPACT_SHA}
GEOMETRY=(384,384,8192)
CAPACITY=(512,512,8192)
DEPTH=4
ARENA_ELEMENTS=292225024
BASE_BIND_FLOPS=254059520
WARM_FLOPS=1512549696
REFERENCE_FLOPS=4825546752
SCRATCH_PEAK_BYTES=241559808
INPUT_OUTPUT_BYTES=25755648


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def _read(path,digest,tick):
    if tick: tick()
    require(path.is_file() and not path.is_symlink(),'Actual regular source owner')
    raw=path.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==digest,'Exact source owner '+str(path))
    if tick: tick()
    return raw.decode()


def _functions(source,names):
    tree=ast.parse(source)
    lines=source.splitlines(keepends=True)
    result={}
    for name in names:
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
        require(len(nodes)==1,'One effective source function '+name)
        node=nodes[0]
        result[name]=''.join(lines[node.lineno-1:node.end_lineno]).rstrip('\n')
    return result


STREAM_HEADER='''def _ss_streamed_product(a,b,out,levels,own):
    m,k,n=int(a.shape[-2]),int(a.shape[-1]),int(b.shape[-1])
    assert (m,k,n,levels)==(384,384,8192,4)
    assert tuple(a.shape)==(m,k) and tuple(b.shape)==(k,n)
    assert tuple(out.shape)==(m,n)
    assert str(a.dtype)==str(b.dtype)==str(out.dtype)=='float32'
    def alloc(shape):
        return own(fnp.empty(shape,dtype=fnp.float32))
    left,right=a,b
    for index in range(levels):
        level=index+1
        m2,k2,n2=m//2**level,k//2**level,n//2**level
        na=alloc((7,)*level+(m2,k2))
        nb=alloc((7,)*level+(k2,n2))
        aa=fnp.unstack(na,axis=0)
        bb=fnp.unstack(nb,axis=0)
        acopies,bcopies=na[:2],nb[:3]
        a11,a12,a21,a22=(left[..., :m2,:k2],left[..., :m2,k2:2*k2],
                        left[..., m2:2*m2,:k2],left[..., m2:2*m2,k2:2*k2])
        b11,b12,b21,b22=(right[..., :k2,:n2],right[..., :k2,n2:2*n2],
                        right[..., k2:2*k2,:n2],right[..., k2:2*k2,n2:2*n2])
'''
STREAM_MIDDLE='''
        # No quadrant or unstack proxy from the previous generation survives
        # the next allocation. Public RemoteArray finalizers queue real frees.
        del aa,bb,acopies,bcopies,a11,a12,a21,a22,b11,b12,b21,b22
        left,right=na,nb
        del na,nb
    leaf=alloc((7,)*levels+(m//2**levels,n//2**levels))
    fnp.matmul(left,right,out=leaf)
    del left,right
    current=leaf
    del leaf
    for level in range(levels,0,-1):
        rows,columns=m//2**level,n//2**level
        products=fnp.unstack(current,axis=0)
        scratch_array=alloc((3,)+(7,)*(level-1)+(rows,columns))
        scratch=fnp.unstack(scratch_array,axis=0)
        dest=alloc((7,)*(level-1)+(2*rows,2*columns)) if level>1 else out
        corners=(dest[..., :rows,:columns],dest[..., :rows,columns:2*columns],
                 dest[..., rows:2*rows,:columns],dest[..., rows:2*rows,columns:2*columns])
'''
STREAM_TAIL='''
        del products,scratch,scratch_array,corners,p0,p1,p2,p3,p4,p5,p6,s0,s1,s2,tl,tr,bl,br
        current=dest
        del dest
    return out
'''


def cost_certificate():
    m,k,n=GEOMETRY
    d=DEPTH
    # All ratios below are exact integers for this declared geometry.
    scale=sum(7**j*4**(d-1-j) for j in range(d))
    transform=7*(m*k+k*n+m*n)*scale//4**d
    leaf=7**d*(m//2**d)*(n//2**d)*(2*k//2**d-1)
    require(transform+leaf==WARM_FLOPS,'Exact unchanged full product tariff')
    forward=max(4*(m*k+k*n)*(7**j*4+7**(j-1)*4**2)//4**(j+1)
                for j in range(1,d+1))
    leaf_bytes=4*7**d*(m*k+k*n+m*n)//4**d
    backward=max(8*7**j*m*n//4**j for j in range(1,d+1))
    require((forward,leaf_bytes,backward)==(194143488,241559808,236027904),
            'Exact two-generation source lifetime envelope')
    require(4*(m*k+k*n+m*n)==INPUT_OUTPUT_BYTES,'Same inputs and one output')
    require(2*m*n*(2*k-1)==REFERENCE_FLOPS,'Exact stored-operand FP64 reference')
    return dict(geometry=list(GEOMETRY),capacity=list(CAPACITY),depth=d,
        arithmetic_primitives=18*d+1,helper_flops=WARM_FLOPS,
        canonical_binding_flops=BASE_BIND_FLOPS,streamed_binding_flops=0,
        original_arena_bytes=4*ARENA_ELEMENTS,input_and_one_output_bytes=INPUT_OUTPUT_BYTES,
        forward_scratch_peak_bytes=forward,leaf_scratch_peak_bytes=leaf_bytes,
        backward_scratch_peak_bytes=backward,maximum_logical_scratch_bytes=SCRATCH_PEAK_BYTES,
        streamed_new_backing_allocations=2*d+1+d+(d-1),
        reference_flops=REFERENCE_FLOPS,
        fixed_bindings_four_helpers_reference=BASE_BIND_FLOPS+4*WARM_FLOPS+REFERENCE_FLOPS,
        bound_requires_actual_owner_retirement=True,server_physical_peak_measured=False,
        full_card_cost_or_runtime_measured=False)


def source_bundle(tick=None):
    box=_read(BOX,BOX_SHA,tick)
    compact=_read(COMPACT,COMPACT_SHA,tick)
    names=('_cp3_bind_arena_views','_packed_frame','_packed_product_cached')
    original=_functions(box,names)
    current=_functions(compact,(*names,'_cs_shape_axes','_cs_compact_specs'))
    for name in ('_cp3_bind_arena_views','_packed_product_cached'):
        require(current[name]==original[name],'Exact current arithmetic '+name)
    hook='    views = _cs_compact_views(levels,m,k,n,views)\n'
    require(current['_packed_frame'].count(hook)==1
        and current['_packed_frame'].replace(hook,'')==original['_packed_frame'],
        'Original frame exactly equals current prebound canonical frame')
    tree=ast.parse(compact)
    assignments=[node for node in tree.body if isinstance(node,ast.Assign)
        and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
        and node.targets[0].id=='_PACKED_PUSH']
    require([n.lineno for n in assignments]==[325,1635,1819],
        'Actual final-executed regular PUSH layout, not its overwritten ancestor')
    specs=ast.literal_eval(assignments[-1].value)
    metadata={}
    exec(compile(current['_cs_shape_axes']+'\n\n'+current['_cs_compact_specs'],
                 'streamed-scratch:actual-scalar-layout','exec'),metadata)
    canonical,cost,changed=metadata['_cs_compact_specs'](specs,DEPTH,*CAPACITY)
    require(changed and cost==BASE_BIND_FLOPS and len(canonical)==24,
        'Actual canonical capacity binding and complete paid reshape tariff')
    product=original['_packed_product_cached']
    begin=product.index('        fnp.add(a21,a22,out=aa[4])')
    end=product.index('    fnp.matmul(leaf_a,leaf_b,out=leaf)')
    forward=product[begin:end].rstrip('\n')+'\n'
    begin=product.index('        p0,p1,p2,p3,p4,p5,p6=products')
    backward=product[begin:product.index('    return out',begin)].rstrip('\n')+'\n'
    require(forward.count('fnp.')==11 and backward.count('fnp.')==7,
        'Original eleven forward/seven backward arithmetic statements')
    stream=STREAM_HEADER+forward+STREAM_MIDDLE+backward+STREAM_TAIL
    source='\n\n'.join(original[name] for name in names)+'\n\n'+stream
    compile(source,'streamed-scratch:exact-arithmetic-new-lifetimes','exec')
    proof=dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        source_bindings={str(p):d for p,d in PINS.items()},
        canonical_frame_whole_source_inverse=True,
        original_functions={n:hashlib.sha256(original[n].encode()).hexdigest() for n in names},
        forward_arithmetic_sha256=hashlib.sha256(forward.encode()).hexdigest(),
        backward_arithmetic_sha256=hashlib.sha256(backward.encode()).hexdigest(),
        forward_statements_byte_exact=True,backward_statements_byte_exact=True,
        leaf_matmul_same_dtype_geometry_and_order=True,
        new_storage_not_covered_by_arithmetic_inverse=True,
        no_persistent_streamed_array_or_frame_cache=True,
        no_dependency_buffers_or_additional_numerical_copies=True,
        no_private_allocation_or_free_API=True,cost_certificate=cost_certificate(),
        candidate_or_head_executed=False,numerical_execution=False)
    if tick: tick()
    return source,canonical,proof


def namespace(fnp):
    source,specs,proof=source_bundle()
    env=dict(fnp=fnp,_PACKED_FRAME_CACHE={},_STREAM_ARENA=None,_STREAM_VIEWS=None,
             _STREAM_CANONICAL_SPECS=specs)
    exec(compile(source,'streamed-scratch:unit-only-namespace','exec'),env)
    return env,proof


def setup(fnp,env):
    arena=fnp.empty(ARENA_ELEMENTS,dtype=fnp.float32)
    views=env['_cp3_bind_arena_views'](arena,env['_STREAM_CANONICAL_SPECS'])
    env['_STREAM_ARENA'],env['_STREAM_VIEWS']=arena,views
    return arena,views


def canonical_product(fnp,env,left,right,own=lambda x:x):
    require(tuple(left.shape)==GEOMETRY[:2] and tuple(right.shape)==GEOMETRY[1:],
        'Only fixed current regular factor-rotation geometry')
    out=own(fnp.empty((GEOMETRY[0],GEOMETRY[2]),dtype=fnp.float32))
    return env['_packed_product_cached'](left,right,out,env['_STREAM_VIEWS'],DEPTH)


def streamed_product(fnp,env,left,right,own=lambda x:x):
    require(tuple(left.shape)==GEOMETRY[:2] and tuple(right.shape)==GEOMETRY[1:],
        'Only fixed current regular factor-rotation geometry')
    out=own(fnp.empty((GEOMETRY[0],GEOMETRY[2]),dtype=fnp.float32))
    return env['_ss_streamed_product'](left,right,out,DEPTH,own)


def clear(env):
    env['_PACKED_FRAME_CACHE'].clear()
    env['_STREAM_VIEWS']=None
    env['_STREAM_ARENA']=None

