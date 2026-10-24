"""Bounded shape-only views of an existing arena; arithmetic is unchanged.

Only views into the fixed scratch arena are retained. Input/output matrices
are never cached. This removes repeated remote slicing, not billed matrix
arithmetic. Twelve frames bound server handles across successive MLPs.
"""
import flopscope.numpy as fnp

_PACKED_FRAME_CACHE = {}


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
        aa = tuple(na[j] for j in range(7))
        bb = tuple(nb[j] for j in range(7))
        if index:
            aq = (old_a[..., :m2, :k2], old_a[..., :m2, k2:2*k2],
                  old_a[..., m2:2*m2, :k2], old_a[..., m2:2*m2, k2:2*k2])
            bq = (old_b[..., :k2, :n2], old_b[..., :k2, n2:2*n2],
                  old_b[..., k2:2*k2, :n2], old_b[..., k2:2*k2, n2:2*n2])
        else:
            aq = bq = None
        forward.append((aq, bq, aa, bb))
        old_a, old_b = na, nb
    leaf = leaf_view[..., :m//2**levels, :n//2**levels]
    current = leaf
    backward = []
    for unwind, level in enumerate(range(levels,0,-1)):
        rows, columns = m//2**level, n//2**level
        products = tuple(current[j] for j in range(7))
        scratch = u_scratch[unwind][..., :rows, :columns]
        scratch_parts = tuple(scratch[j] for j in range(3))
        if level>1:
            dest = intermediates[unwind][..., :2*rows, :2*columns]
            corners = (dest[..., :rows, :columns], dest[..., :rows, columns:2*columns],
                       dest[..., rows:2*rows, :columns], dest[..., rows:2*rows, columns:2*columns])
            current = dest
        else:
            corners = None
        backward.append((products,scratch_parts,corners))
    frame = (forward, old_a, old_b, leaf, backward)
    if len(_PACKED_FRAME_CACHE)>=12:
        _PACKED_FRAME_CACHE.pop(next(iter(_PACKED_FRAME_CACHE)))
    _PACKED_FRAME_CACHE[key] = frame
    return frame


def _packed_product_cached(a,b,out,views,levels):
    m,k,n = int(a.shape[-2]),int(a.shape[-1]),int(b.shape[-1])
    assert int(b.shape[-2])==k and tuple(out.shape)==(m,n)
    assert m%2**levels==k%2**levels==n%2**levels==0
    forward,leaf_a,leaf_b,leaf,backward = _packed_frame(levels,m,k,n,views)
    for index,(aq,bq,aa,bb) in enumerate(forward):
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
        fnp.copyto(aa[0],a11)
        fnp.copyto(aa[1],a12)
        fnp.copyto(aa[3],a22)
        fnp.subtract(b12,b11,out=bb[4])
        fnp.subtract(b22,bb[4],out=bb[5])
        fnp.subtract(b22,b12,out=bb[6])
        fnp.subtract(bb[5],b21,out=bb[3])
        fnp.copyto(bb[0],b11)
        fnp.copyto(bb[1],b21)
        fnp.copyto(bb[2],b22)
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
