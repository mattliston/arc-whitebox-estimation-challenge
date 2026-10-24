"""Unqualified serial-top D6, using only public FNP operations.

This module is inert: no FNP import, arrays, filesystem reads, or numerical
calls occur on import.  It contains no estimator, learned payload, or fixture.

Reference helpers are literal source copies from the pinned d50 estimator.
audit_sources() authenticates that fact, the child layout, and the 18 top
arithmetic statements without executing numerical code.  A remote harness
may ship this compact module after that audit instead of the full estimator.

Numerical API (all creation and calls belong inside the caller's budget):
    ns = make_namespace(fnp, d50_source_text)  # source text optional remotely
    state = bind(fnp, existing_float32_arena, READ_CAPACITY, ns)
    product(a, b, out, state)

a, b and out must be separately owned from the arena and mutually compatible;
out must not overlap either input.  No padding, casting, concatenation, or
output allocation is performed here.  Dimensions must be positive multiples
of 64 within the chosen maximum capacity.  All bindings sharing an arena
must execute serially, including other current-bank square/rectangle kernels.

The seven top operands, seven top products, three reconstruction buffers and
one max-capacity D5 child are disjoint.  Child outputs go directly into the
top product slots.  Top and child frame caches retain arena views only, never
caller inputs or outputs.  Exact-real equivalence is the design; FP32 byte
equivalence to breadth D6 and all timing/RSS/residual gates are UNQUALIFIED.
"""

import ast
import copy
from fractions import Fraction
import hashlib
from math import prod


PARENT_SHA256 = "d50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451"
LAYOUT_SHA256 = "4e4a3dea533c222f34ba3a08538245414e149dbc0f918fd097267b0f869fb27e"
PARENT_RELATIVE_PATH = "birth_direction_fit_v1/build_terminal_v1/estimator.py"
ORIGINAL_ARENA_ELEMENTS = 292225024
ORIGINAL_ARENA_BYTES = 1168900096
READ_CAPACITY = (1024, 8192, 1024)
PUSH_CAPACITY = (1024, 1024, 8192)
HELPER_NAMES = ("_cp3_bind_arena_views", "_packed_frame", "_packed_product_cached")
WARM_PRIMITIVE_CALLS = dict(add=252, subtract=288, stack=72, copyto=36, matmul=7)


_HELPER_SOURCE = r'''def _cp3_bind_arena_views(arena, view_specs):
    """Bind the sealed D1 arena views once on the first prediction."""
    return tuple((fnp.reshape(arena[start:stop], shape) for start, stop, shape in view_specs))

def _packed_frame(levels, m, k, n, views):
    key = (id(views), levels, m, k, n)
    found = _PACKED_FRAME_CACHE.get(key)
    if found is not None:
        return found
    cursor = 0
    av = views[cursor:cursor+2*levels:2]
    bv = views[cursor+1:cursor+2*levels:2]
    cursor += 4*levels  # transformed arrays plus the two dependency lists
    leaf_view = views[cursor]; cursor += 1
    intermediates = views[cursor:cursor+levels-1]; cursor += levels-1
    u_scratch = views[cursor:cursor+levels]
    assert cursor+levels == len(views)
    forward = []
    old_a = old_b = None
    for index in range(levels):
        level = index+1
        m2, k2, n2 = m//2**level, k//2**level, n//2**level
        na = av[index][..., :m2, :k2]
        nb = bv[index][..., :k2, :n2]
        aa = fnp.unstack(na, axis=0)
        bb = fnp.unstack(nb, axis=0)
        if index:
            aq = (old_a[..., :m2, :k2], old_a[..., :m2, k2:2*k2],
                  old_a[..., m2:2*m2, :k2], old_a[..., m2:2*m2, k2:2*k2])
            bq = (old_b[..., :k2, :n2], old_b[..., :k2, n2:2*n2],
                  old_b[..., k2:2*k2, :n2], old_b[..., k2:2*k2, n2:2*n2])
        else:
            aq = bq = None
        forward.append((aq, bq, aa, bb, na[:2], nb[:3]))
        old_a, old_b = na, nb
    leaf = leaf_view[..., :m//2**levels, :n//2**levels]
    current = leaf
    backward = []
    for unwind, level in enumerate(range(levels,0,-1)):
        rows, columns = m//2**level, n//2**level
        products = fnp.unstack(current, axis=0)
        scratch = u_scratch[unwind][..., :rows, :columns]
        scratch_parts = fnp.unstack(scratch, axis=0)
        if level>1:
            dest = intermediates[unwind][..., :2*rows, :2*columns]
            corners = (dest[..., :rows, :columns], dest[..., :rows, columns:2*columns],
                       dest[..., rows:2*rows, :columns], dest[..., rows:2*rows, columns:2*columns])
            current = dest
        else:
            corners = None
        backward.append((products,scratch_parts,corners))
    frame = (forward, old_a, old_b, leaf, backward)
    if len(_PACKED_FRAME_CACHE)>=128:
        _PACKED_FRAME_CACHE.pop(next(iter(_PACKED_FRAME_CACHE)))
    _PACKED_FRAME_CACHE[key] = frame
    return frame

def _packed_product_cached(a,b,out,views,levels):
    m,k,n = int(a.shape[-2]),int(a.shape[-1]),int(b.shape[-1])
    assert int(b.shape[-2])==k and tuple(out.shape)==(m,n)
    assert m%2**levels==k%2**levels==n%2**levels==0
    forward,leaf_a,leaf_b,leaf,backward = _packed_frame(levels,m,k,n,views)
    for index,(aq,bq,aa,bb,acopies,bcopies) in enumerate(forward):
        if index==0:
            m2,k2,n2 = m//2,k//2,n//2
            a11,a12,a21,a22 = a[:m2,:k2],a[:m2,k2:],a[m2:,:k2],a[m2:,k2:]
            b11,b12,b21,b22 = b[:k2,:n2],b[:k2,n2:],b[k2:,:n2],b[k2:,n2:]
        else:
            a11,a12,a21,a22=aq
            b11,b12,b21,b22=bq
        fnp.add(a21,a22,out=aa[4])
        fnp.subtract(aa[4],a11,out=aa[5])
        fnp.subtract(a11,a21,out=aa[6])
        fnp.subtract(a12,aa[5],out=aa[2])
        fnp.stack((a11,a12),axis=0,out=acopies)
        fnp.copyto(aa[3],a22)
        fnp.subtract(b12,b11,out=bb[4])
        fnp.subtract(b22,bb[4],out=bb[5])
        fnp.subtract(b22,b12,out=bb[6])
        fnp.subtract(bb[5],b21,out=bb[3])
        fnp.stack((b11,b21,b22),axis=0,out=bcopies)
    fnp.matmul(leaf_a,leaf_b,out=leaf)
    for products,scratch,corners in backward:
        if corners is None:
            rows,columns=m//2,n//2
            corners=(out[:rows,:columns],out[:rows,columns:],out[rows:,:columns],out[rows:,columns:])
        p0,p1,p2,p3,p4,p5,p6=products
        s0,s1,s2=scratch
        tl,tr,bl,br=corners
        fnp.add(p0,p1,out=tl)
        fnp.add(p0,p5,out=s0)
        fnp.add(s0,p6,out=s1)
        fnp.add(s0,p4,out=s2)
        fnp.add(s2,p2,out=tr)
        fnp.subtract(s1,p3,out=bl)
        fnp.add(s1,p4,out=br)
    return out
'''


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _functions(source):
    # Last effective definition, exactly as Python's module binding behaves.
    return {node.name: node for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)}


def extract_helpers(d50_source_text):
    """Authenticate/extract a compact reference bundle; stdlib only."""
    assert _sha(d50_source_text) == PARENT_SHA256
    parent, embedded = _functions(d50_source_text), _functions(_HELPER_SOURCE)
    for name in HELPER_NAMES:
        assert ast.get_source_segment(d50_source_text, parent[name]) == ast.get_source_segment(
            _HELPER_SOURCE, embedded[name]), name
    return dict(source=_HELPER_SOURCE, sha256=_sha(_HELPER_SOURCE),
                parent_sha256=PARENT_SHA256, helper_source_byte_exact=True)


def make_namespace(fnp, d50_source_text=None):
    """Compile only reference AST functions, with an isolated view cache.

Supplying the full source authenticates it here.  Omitting it is intended for
a remote package whose frozen kernel/helper bytes were authenticated before
launch; provenance explicitly records that the full parent was not reread.
No numerical operation is performed by this function.
"""
    if d50_source_text is not None:
        bundle = extract_helpers(d50_source_text)
        functions = _functions(d50_source_text)
    else:
        bundle = dict(sha256=_sha(_HELPER_SOURCE), parent_sha256=PARENT_SHA256)
        functions = _functions(_HELPER_SOURCE)
    ns = dict(fnp=fnp, _PACKED_FRAME_CACHE={})
    for name in HELPER_NAMES:
        unit = ast.Module(body=[functions[name]], type_ignores=[])
        exec(compile(unit, PARENT_RELATIVE_PATH + ":reference", "exec"), ns)
    ns["provenance"] = dict(parent_sha256=PARENT_SHA256,
        helpers_sha256=bundle["sha256"], parent_verified_here=d50_source_text is not None,
        compiled_functions=list(HELPER_NAMES))
    return ns


def _child_layout(m,k,n,levels):
    assert all(v%2**levels==0 for v in (m,k,n))
    q=Fraction(7,4);a=m*k;b=k*n;c=m*n
    def exact(x):
        assert x.denominator==1
        return x.numerator
    last_a=exact(q**levels*a);last_b=exact(q**levels*b)
    prev_a=exact(q**(levels-1)*a);prev_b=exact(q**(levels-1)*b)
    dep=exact(q**(levels-1)*max(a,b));leaf=exact(q**levels*c);product=exact(q**(levels-1)*c)
    intervals={};cursor=0
    for name,size in (("last_a",last_a),("last_b",last_b),("prev_a",prev_a),("prev_b",prev_b),("dep",dep)):
        intervals[name]=(cursor,cursor+size);cursor+=size
    final_stop=last_a+last_b;leaf_start=cursor-leaf
    assert leaf<=prev_a+prev_b+dep and final_stop<=leaf_start and 2*product<=leaf_start
    intervals.update(leaf=(leaf_start,cursor),product0=(0,product),u=(product,2*product),product1=(leaf_start,leaf_start+product))
    specs=[]
    def view(owner,shape):
        start,limit=intervals[owner];stop=start+prod(shape)
        assert stop<=limit,(owner,shape,stop,limit)
        specs.append((start,stop,tuple(shape)))
    for l in range(1,levels+1):
        parity="last" if l%2==levels%2 else "prev"
        view(parity+"_a",(7,)*l+(m//2**l,k//2**l))
        view(parity+"_b",(7,)*l+(k//2**l,n//2**l))
    for x,y in ((m,k),(k,n)):
        for l in range(1,levels+1):view("dep",(4,)+(7,)*(l-1)+(x//2**l,y//2**l))
    view("leaf",(7,)*levels+(m//2**levels,n//2**levels))
    for unwind,l in enumerate(range(levels-1,0,-1)):
        view("product0" if unwind%2==0 else "product1",(7,)*l+(m//2**l,n//2**l))
    for l in range(levels,0,-1):view("u",(3,)+(7,)*(l-1)+(m//2**l,n//2**l))
    assert len(specs)==6*levels
    return cursor,tuple(specs),intervals


def metadata(capacity=READ_CAPACITY):
    """Disjoint owner/reshape census; maximum layouts, not exact-aspect ones."""
    capacity = tuple(capacity)
    assert capacity in (READ_CAPACITY, PUSH_CAPACITY)
    m,k,n = capacity
    cursor = 0
    top_specs = []
    for shape in ((7,m//2,k//2), (7,k//2,n//2),
                  (7,m//2,n//2), (3,m//2,n//2)):
        stop = cursor + prod(shape)
        top_specs.append((cursor,stop,shape))
        cursor = stop
    top_elements = cursor
    child_size, child_specs, child_intervals = _child_layout(m//2,k//2,n//2,5)
    shifted = tuple((start+cursor,stop+cursor,shape) for start,stop,shape in child_specs)
    specs = tuple(top_specs)+shifted
    required = cursor+child_size
    assert len(specs) == 34 and required <= ORIGINAL_ARENA_ELEMENTS
    assert all(top_elements <= start <= stop <= required for start,stop,_ in shifted)
    assert all(stop-start == prod(shape) for start,stop,shape in specs)
    return dict(capacity=capacity, top_specs=tuple(top_specs), child_specs=shifted,
        specs=specs, top_elements=top_elements, child_elements=child_size,
        child_intervals_relative=child_intervals, required_elements=required,
        scratch_bytes=4*required, original_arena_bytes=ORIGINAL_ARENA_BYTES,
        binding_flops=sum(stop-start for start,stop,_ in specs),
        binding_reshape_calls=34, depth=6, child_depth=5, child_calls=7,
        warm_primitive_calls=dict(WARM_PRIMITIVE_CALLS),
        outside_owner="caller-owned input A, input B and output; no padding included")


def binding_flops(capacity=READ_CAPACITY):
    return metadata(capacity)["binding_flops"]


def price(m,k,n,depth=6):
    """Warm FP32 product bill only, including arithmetic and bulk copies.

No initialization/bindings, ingress, padding, output assembly, consumer,
whole-estimator work or timing is included.  All seven D5 child calls are
charged; child outputs introduce no additional matrix copy.
"""
    assert all(isinstance(v,int) and v>0 for v in (m,k,n,depth))
    assert all(v%2**depth==0 for v in (m,k,n))
    q = Fraction(7,4)
    value = 7**depth*(m//2**depth)*(n//2**depth)*(2*k//2**depth-1)
    value += q*(m*k+k*n+m*n)*sum(q**j for j in range(depth))
    assert value.denominator == 1
    return value.numerator


class _Binding:
    def __init__(self, fnp, arena, capacity, namespace, views, info):
        self.fnp, self.arena, self.capacity = fnp, arena, capacity
        self.namespace, self.info = namespace, info
        self.top_views, self.child_views = views[:4], views[4:]
        self.frames = {}
        self.busy = False

    def frame(self,m,k,n):
        key = (m,k,n)
        found = self.frames.get(key)
        if found is not None:
            return found
        fnp = self.fnp
        m2,k2,n2 = m//2,k//2,n//2
        av,bv,pv,sv = self.top_views
        na,nb = av[..., :m2, :k2],bv[..., :k2, :n2]
        aa,bb = fnp.unstack(na,axis=0),fnp.unstack(nb,axis=0)
        products = fnp.unstack(pv[..., :m2, :n2],axis=0)
        scratch = fnp.unstack(sv[..., :m2, :n2],axis=0)
        found = (aa,bb,na[:2],nb[:3],products,scratch)
        if len(self.frames)>=128:
            self.frames.pop(next(iter(self.frames)))
        self.frames[key] = found
        return found


def bind(fnp, arena, capacity=READ_CAPACITY, namespace=None):
    """Create 34 paid reshape views; never allocate a second backing owner."""
    info = metadata(capacity)
    assert len(arena.shape)==1 and str(arena.dtype)=="float32"
    assert info["required_elements"] <= int(arena.shape[0]) <= ORIGINAL_ARENA_ELEMENTS
    if namespace is None:
        namespace = make_namespace(fnp)
    assert namespace["fnp"] is fnp
    views = namespace["_cp3_bind_arena_views"](arena,info["specs"])
    return _Binding(fnp,arena,tuple(capacity),namespace,views,info)


def product(a,b,out,binding):
    """Fixed serial-top D6, then the exact current top reconstruction."""
    assert len(a.shape)==len(b.shape)==len(out.shape)==2
    assert str(a.dtype)==str(b.dtype)==str(out.dtype)=="float32"
    m,k,n = int(a.shape[0]),int(a.shape[1]),int(b.shape[1])
    assert int(b.shape[0])==k and tuple(out.shape)==(m,n)
    assert all(v>0 and v%64==0 for v in (m,k,n))
    assert all(v<=limit for v,limit in zip((m,k,n),binding.capacity))
    assert not binding.busy
    binding.busy = True
    try:
        fnp = binding.fnp
        aa,bb,acopies,bcopies,products,scratch = binding.frame(m,k,n)
        m2,k2,n2 = m//2,k//2,n//2
        a11,a12,a21,a22 = a[:m2,:k2],a[:m2,k2:],a[m2:,:k2],a[m2:,k2:]
        b11,b12,b21,b22 = b[:k2,:n2],b[:k2,n2:],b[k2:,:n2],b[k2:,n2:]
        fnp.add(a21,a22,out=aa[4])
        fnp.subtract(aa[4],a11,out=aa[5])
        fnp.subtract(a11,a21,out=aa[6])
        fnp.subtract(a12,aa[5],out=aa[2])
        fnp.stack((a11,a12),axis=0,out=acopies)
        fnp.copyto(aa[3],a22)
        fnp.subtract(b12,b11,out=bb[4])
        fnp.subtract(b22,bb[4],out=bb[5])
        fnp.subtract(b22,b12,out=bb[6])
        fnp.subtract(bb[5],b21,out=bb[3])
        fnp.stack((b11,b21,b22),axis=0,out=bcopies)
        child = binding.namespace["_packed_product_cached"]
        for j in range(7):
            child(aa[j],bb[j],products[j],binding.child_views,5)
        p0,p1,p2,p3,p4,p5,p6=products
        s0,s1,s2=scratch
        tl,tr,bl,br=out[:m2,:n2],out[:m2,n2:],out[m2:,:n2],out[m2:,n2:]
        fnp.add(p0,p1,out=tl)
        fnp.add(p0,p5,out=s0)
        fnp.add(s0,p6,out=s1)
        fnp.add(s0,p4,out=s2)
        fnp.add(s2,p2,out=tr)
        fnp.subtract(s1,p3,out=bl)
        fnp.add(s1,p4,out=br)
        return out
    finally:
        binding.busy = False


def _arithmetic_statements(node):
    result = []
    class Visitor(ast.NodeVisitor):
        def visit_Expr(self, item):
            call = item.value
            if (isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
                    and isinstance(call.func.value,ast.Name) and call.func.value.id=="fnp"
                    and call.func.attr in ("add","subtract","stack","copyto")):
                result.append(ast.dump(item,include_attributes=False))
            self.generic_visit(item)
    Visitor().visit(node)
    return result


def audit_sources(d50_source_text, layout_source_text, module_source_text):
    """Source-only identity proof; not numerical qualification or execution.

The child allocator differs from the pinned allocator only in its name.
The top forward/reconstruction arithmetic statements are byte-AST identical
and in the same order.  Their surrounding scheduling intentionally changes.
"""
    bundle = extract_helpers(d50_source_text)
    assert _sha(layout_source_text)==LAYOUT_SHA256
    own = _functions(module_source_text)
    old_layout = _functions(layout_source_text)["layout"]
    restored = copy.deepcopy(own["_child_layout"])
    restored.name = "layout"
    assert ast.dump(restored,include_attributes=False)==ast.dump(old_layout,include_attributes=False)
    original = _functions(d50_source_text)["_packed_product_cached"]
    expected = _arithmetic_statements(original)
    actual = _arithmetic_statements(own["product"])
    assert len(actual)==18 and actual==expected
    info = [metadata(PUSH_CAPACITY),metadata(READ_CAPACITY)]
    assert [x["scratch_bytes"] for x in info]==[472026112,639320064]
    assert [x["binding_flops"] for x in info]==[278566656,282489344]
    assert sum(x["binding_flops"] for x in info)==561056000
    assert sum(WARM_PRIMITIVE_CALLS.values())==655
    return dict(parent_sha256=PARENT_SHA256,layout_sha256=LAYOUT_SHA256,
        kernel_sha256=_sha(module_source_text),helpers_sha256=bundle["sha256"],
        helper_source_byte_exact=True,child_layout_inverse_name_only=True,
        top_arithmetic_ast_equal=True,top_arithmetic_statements=18,
        max_scratch_bytes=639320064,additional_both_binding_flops=561056000,
        numerical_execution=False,numerical_equivalence_qualified=False,
        runtime_qualified=False)
