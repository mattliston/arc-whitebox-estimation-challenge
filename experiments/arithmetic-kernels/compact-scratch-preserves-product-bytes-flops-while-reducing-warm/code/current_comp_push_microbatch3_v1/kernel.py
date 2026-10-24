"""DRAFT, inert: current-c2 PUSH D6 with fixed independent-child batches3+3+1.

No numerical import, caller arrays, estimator module, or stage at import.
make_namespace compiles only authenticated kernel definitions. Both controls
and the new child adapter use the exact current depth/layout/arithmetic owners.
"""
import ast
from fractions import Fraction
import hashlib
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent/'current_comp_shared_gain_slice_v1/build_v1/estimator.py'
PARENT_SHA = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
SHAPE = (1024,384,8192)
CAPACITY = (1024,1024,8192)
ARENA_ELEMENTS = 292225024
ARENA_BYTES = 1168900096
ROLES = ('d4','serial_d6','microbatch3')
COUNTS = {
    'd4':dict(add=28,subtract=32,stack=8,copyto=4,matmul=1),
    'serial_d6':dict(add=252,subtract=288,stack=72,copyto=36,matmul=7),
    'microbatch3':dict(add=112,subtract=128,stack=32,copyto=16,matmul=3),
}
FUNCTIONS = ('_cp3_bind_arena_views','_packed_frame','_packed_product_cached','_p1_d6_832_product')
LITERALS = ('_PACKED_PUSH','_PACKED_READ','_P1_D6_832_PUSH_SPECS','_P1_D6_832_READ_SPECS',
            '_P1_D6_832_PUSH_INFO','_P1_D6_832_READ_INFO')


def require(value,message):
    if not value:
        raise RuntimeError(message)


def source_bundle():
    raw=PARENT.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==PARENT_SHA,'Exact current-c2 source, never imported as an estimator')
    tree=ast.parse(raw.decode('utf-8')); nodes={}; literals={}
    names=FUNCTIONS+('_P1_D6_832_Binding',)
    offsets,position=[],0
    for line in raw.splitlines(keepends=True):
        offsets.append(position);position+=len(line)
    require(position==len(raw),'Exact single-pass UTF8 byte line index')
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:
            nodes[node.name]=node
        elif isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in LITERALS:
                    literals[target.id]=ast.literal_eval(node.value)
    require(set(literals)==set(LITERALS),'Actual last-effective current layout literals')
    require(set(nodes)==set(names),'Only the five exact last-effective requested definitions')
    functions={}
    for name in names:
        node=nodes[name]
        start=offsets[node.lineno-1]+node.col_offset
        stop=offsets[node.end_lineno-1]+node.end_col_offset
        require(0<=start<stop<=len(raw),'Exact AST UTF8 byte span')
        functions[name]=raw[start:stop].decode('utf-8')
    text='\n\n'.join(functions[k] for k in names)+'\n'
    return text,literals,dict(parent_sha256=PARENT_SHA,exact_current_function_names=list(names),
        extracted_sha256=hashlib.sha256(text.encode()).hexdigest(),estimator_module_executed=False)


def _adapt(text,edits):
    source,spans=text,[]
    for before,after in edits:
        require(source.count(before)==1,'Unique kernel-only batch splice: '+before[:90])
        at=source.index(before);spans.append((at,before,after))
        source=source[:at]+after+source[at+len(before):]
    inverse=source
    for at,before,after in reversed(spans):
        require(inverse[at:at+len(after)]==after,'Positional complete kernel inverse')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    require(inverse==text,'Entire original c2 kernel definitions restored')
    return source


def child_source(bundle):
    nodes={n.name:ast.get_source_segment(bundle,n) for n in ast.parse(bundle).body
           if isinstance(n,ast.FunctionDef)}
    frame=_adapt(nodes['_packed_frame'],(
        ('def _packed_frame(levels, m, k, n, views):','def _micro3_frame(levels, m, k, n, views, batch):'),
        ('key = (id(views), levels, m, k, n)','key = (id(views), levels, m, k, n, batch)'),
        ('na = av[index][..., :m2, :k2]','na = av[index][..., :batch, :m2, :k2]'),
        ('nb = bv[index][..., :k2, :n2]','nb = bv[index][..., :batch, :k2, :n2]'),
        ('leaf = leaf_view[..., :m//2**levels, :n//2**levels]',
         'leaf = leaf_view[..., :batch, :m//2**levels, :n//2**levels]'),
        ('scratch = u_scratch[unwind][..., :rows, :columns]',
         'scratch = u_scratch[unwind][..., :batch, :rows, :columns]'),
        ('dest = intermediates[unwind][..., :2*rows, :2*columns]',
         'dest = intermediates[unwind][..., :batch, :2*rows, :2*columns]'),
    ))
    product=_adapt(nodes['_packed_product_cached'],(
        ('def _packed_product_cached(a,b,out,views,levels):','def _micro3_child(a,b,out,views,levels):'),
        ('assert int(b.shape[-2])==k and tuple(out.shape)==(m,n)',
         'batch = int(a.shape[0])\n    assert len(a.shape)==len(b.shape)==len(out.shape)==3\n'
         '    assert batch in (1,3) and int(b.shape[0])==batch\n'
         '    assert int(b.shape[-2])==k and tuple(out.shape)==(batch,m,n)'),
        ('_packed_frame(levels,m,k,n,views)','_micro3_frame(levels,m,k,n,views,batch)'),
        ('a[:m2,:k2],a[:m2,k2:],a[m2:,:k2],a[m2:,k2:]',
         'a[..., :m2,:k2],a[..., :m2,k2:],a[..., m2:,:k2],a[..., m2:,k2:]'),
        ('b[:k2,:n2],b[:k2,n2:],b[k2:,:n2],b[k2:,n2:]',
         'b[..., :k2,:n2],b[..., :k2,n2:],b[..., k2:,:n2],b[..., k2:,n2:]'),
        ('out[:rows,:columns],out[:rows,columns:],out[rows:,:columns],out[rows:,columns:]',
         'out[..., :rows,:columns],out[..., :rows,columns:],out[..., rows:,:columns],out[..., rows:,columns:]'),
    ))
    push=_adapt(nodes['_p1_d6_832_product'],(
        ('def _p1_d6_832_product(a,b,out,binding):','def _micro3_push(a,b,out,binding):'),
        ('child = binding.namespace["_packed_product_cached"]\n        for j in range(7):\n'
         '            child(aa[j],bb[j],products[j],binding.child_views,5)',
         'child = binding.namespace["_micro3_child"]\n'
         '        for left,right,destination in _micro3_groups(binding,m,k,n):\n'
         '            child(left,right,destination,binding.child_views,5)'),
    ))
    return frame+'\n\n'+product+'\n\n'+push+'\n'


def micro_metadata(literals):
    original=literals['_P1_D6_832_PUSH_INFO']
    require(tuple(original['capacity'])==CAPACITY and original['top_elements']==37486592
        and original['child_elements']==80519936 and original['binding_flops']==278566656,
        'Actual current maximum PUSH layout, not a measured-array assumption')
    top=original['top_elements']; child=[]
    for start,stop,shape in original['child_specs']:
        shape=tuple(shape); new_shape=shape[:-2]+(3,)+shape[-2:]
        child.append((top+3*(start-top),top+3*(stop-top),new_shape))
    specs=tuple(original['top_specs'])+tuple(child)
    required=top+3*original['child_elements']
    require(required==279046400 and required<=ARENA_ELEMENTS and len(specs)==34
        and all(stop-start==math.prod(shape) for start,stop,shape in specs)
        and all(top<=start<stop<=required for start,stop,shape in child),
        'Three disjoint child lanes inside the SAME original arena')
    tariff=sum(stop-start for start,stop,shape in specs)
    require(tariff==760726784 and tariff-original['binding_flops']==482160128,
        'Complete replacement PUSH reshape tariff, not merely new live elements')
    return dict(specs=specs,required_elements=required,scratch_bytes=4*required,
        original_arena_bytes=ARENA_BYTES,binding_flops=tariff,binding_reshape_calls=len(specs),
        additional_cold_binding_flops=482160128,batches=(3,3,1),read_binding_unchanged=True)


def _micro3_groups(binding,m,k,n):
    key=(m,k,n)
    found=binding.groups.get(key)
    if found is None:
        a,b,p,unused=binding.top_views
        found=tuple((a[start:stop,:m//2,:k//2],b[start:stop,:k//2,:n//2],p[start:stop,:m//2,:n//2])
                    for start,stop in ((0,3),(3,6),(6,7)))
        binding.groups[key]=found
    return found


def make_namespace(fnp):
    bundle,literals,proof=source_bundle(); adapted=child_source(bundle)
    ns=dict(fnp=fnp,_PACKED_FRAME_CACHE={},_micro3_groups=_micro3_groups)
    exec(compile(bundle,'<exact-current-c2-kernels>','exec'),ns)
    exec(compile(adapted,'<positional-inverse-microbatch3-kernels>','exec'),ns)
    proof.update(batch_source_sha256=hashlib.sha256(adapted.encode()).hexdigest(),
        original_complete_child_and_top_inverses=True,physical_terms_or_heads_changed=False,
        no_depth_or_shape_policy_search=True,original_READ_kernel_unchanged=True)
    return ns,literals,proof


def setup(fnp,ns,literals,role,left,right):
    require(role in ROLES and tuple(left.shape)==SHAPE[:2] and tuple(right.shape)==SHAPE[1:]
        and str(left.dtype)==str(right.dtype)=='float32','Only the fixed synthetic kernel inputs')
    arena=fnp.empty((ARENA_ELEMENTS,),dtype=fnp.float32)
    bind=ns['_cp3_bind_arena_views']
    old_push=bind(arena,literals['_PACKED_PUSH'])
    old_read=bind(arena,literals['_PACKED_READ'])
    state=None; extra_read=None
    if role!='d4':
        specs=literals['_P1_D6_832_PUSH_SPECS'] if role=='serial_d6' else micro_metadata(literals)['specs']
        views=bind(arena,specs)
        extra_read=bind(arena,literals['_P1_D6_832_READ_SPECS'])
        info=literals['_P1_D6_832_PUSH_INFO'] if role=='serial_d6' else micro_metadata(literals)
        state=ns['_P1_D6_832_Binding'](fnp,arena,CAPACITY,ns,views,info)
        state.groups={}
    output=fnp.empty((SHAPE[0],SHAPE[2]),dtype=fnp.float32)
    return dict(arena=arena,old_push=old_push,old_read=old_read,extra_read=extra_read,state=state,output=output)


def product(ns,role,left,right,state):
    require(role in ROLES and tuple(left.shape)==SHAPE[:2] and tuple(right.shape)==SHAPE[1:]
        and tuple(state['output'].shape)==(SHAPE[0],SHAPE[2])
        and str(left.dtype)==str(right.dtype)==str(state['output'].dtype)=='float32',
        'Only one fixed geometry, original typed operands and caller-owned output; no depth argument')
    if role=='d4':
        return ns['_packed_product_cached'](left,right,state['output'],state['old_push'],4)
    name='_p1_d6_832_product' if role=='serial_d6' else '_micro3_push'
    return ns[name](left,right,state['output'],state['state'])


def price(role):
    require(role in ROLES,'Fixed price role')
    m,k,n=SHAPE; d=4 if role=='d4' else 6; q=Fraction(7,4)
    total=7**d*(m//2**d)*(n//2**d)*(2*k//2**d-1)+q*(m*k+k*n+m*n)*sum(q**j for j in range(d))
    require(total.denominator==1,'Integer complete arithmetic/copy kernel tariff')
    expected=3930963456 if role=='d4' else 3421951456
    require(total.numerator==expected,'Independently recorded fixed-shape tariff')
    return expected


def setup_price(literals,role):
    base=sum(stop-start for key in ('_PACKED_PUSH','_PACKED_READ')
             for start,stop,shape in literals[key])
    require(base==1068953600,'Both exact last-effective original D4 binding tariffs')
    if role=='d4':
        return base
    push=literals['_P1_D6_832_PUSH_INFO']['binding_flops'] if role=='serial_d6' else micro_metadata(literals)['binding_flops']
    return base+push+literals['_P1_D6_832_READ_INFO']['binding_flops']
