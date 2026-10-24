

# BEGIN FABLE-ENGINEER HOLD-ONLY KERNEL (E-22, 2026-09-22): the bank's own kernel bodies (_packed_product_cached, _read5_leaf_packets,
# _packed_read5, _w5_port_product_body), unchanged except that every discarded out= alias and per-call quadrant view is kept in
# one list and released at the end of the call, so the client frees them in one round trip instead of one per op. Same arithmetic,
# same order, same bill (bit-exact); fewer round trips only.
_BH_HOLD = []


def _packed_product_cached(a,b,out,views,levels):
    m,k,n = int(a.shape[-2]),int(a.shape[-1]),int(b.shape[-1])
    assert int(b.shape[-2])==k and tuple(out.shape)==(m,n)
    assert m%2**levels==k%2**levels==n%2**levels==0
    forward,leaf_a,leaf_b,leaf,backward = _packed_frame(levels,m,k,n,views)
    for index,(aq,bq,aa,bb,acopies,bcopies) in enumerate(forward):
        if index==0:
            m2,k2,n2 = m//2,k//2,n//2
            a11,a12,a21,a22 = a[:m2,:k2],a[:m2,k2:],a[m2:,:k2],a[m2:,k2:]
            _BH_HOLD.append((a11,a12,a21,a22))
            b11,b12,b21,b22 = b[:k2,:n2],b[:k2,n2:],b[k2:,:n2],b[k2:,n2:]
            _BH_HOLD.append((b11,b12,b21,b22))
        else:
            a11,a12,a21,a22=aq
            b11,b12,b21,b22=bq
        _BH_HOLD.append(fnp.add(a21,a22,out=aa[4]))
        _BH_HOLD.append(fnp.subtract(aa[4],a11,out=aa[5]))
        _BH_HOLD.append(fnp.subtract(a11,a21,out=aa[6]))
        _BH_HOLD.append(fnp.subtract(a12,aa[5],out=aa[2]))
        _BH_HOLD.append(fnp.stack((a11,a12),axis=0,out=acopies))
        _BH_HOLD.append(fnp.copyto(aa[3],a22))
        _BH_HOLD.append(fnp.subtract(b12,b11,out=bb[4]))
        _BH_HOLD.append(fnp.subtract(b22,bb[4],out=bb[5]))
        _BH_HOLD.append(fnp.subtract(b22,b12,out=bb[6]))
        _BH_HOLD.append(fnp.subtract(bb[5],b21,out=bb[3]))
        _BH_HOLD.append(fnp.stack((b11,b21,b22),axis=0,out=bcopies))
    _BH_HOLD.append(fnp.matmul(leaf_a,leaf_b,out=leaf))
    for products,scratch,corners in backward:
        if corners is None:
            rows,columns=m//2,n//2
            corners=(out[:rows,:columns],out[:rows,columns:],out[rows:,:columns],out[rows:,columns:])
            _BH_HOLD.append(corners)
        p0,p1,p2,p3,p4,p5,p6=products
        s0,s1,s2=scratch
        tl,tr,bl,br=corners
        _BH_HOLD.append(fnp.add(p0,p1,out=tl))
        _BH_HOLD.append(fnp.add(p0,p5,out=s0))
        _BH_HOLD.append(fnp.add(s0,p6,out=s1))
        _BH_HOLD.append(fnp.add(s0,p4,out=s2))
        _BH_HOLD.append(fnp.add(s2,p2,out=tr))
        _BH_HOLD.append(fnp.subtract(s1,p3,out=bl))
        _BH_HOLD.append(fnp.add(s1,p4,out=br))
    _BH_HOLD.clear()
    return out


def _read5_leaf_packets(m, k, n, views, leaf_a, leaf_b, leaf):
    for aq, bq, corners, child in _read5_packet_frames(m, k, n, views, leaf_a, leaf_b, leaf):
        ca, cb, cc, aa, bb, acopies, bcopies, products, scratch = child
        a11, a12, a21, a22 = aq
        b11, b12, b21, b22 = bq
        _BH_HOLD.append(fnp.add(a21, a22, out=aa[4]))
        _BH_HOLD.append(fnp.subtract(aa[4], a11, out=aa[5]))
        _BH_HOLD.append(fnp.subtract(a11, a21, out=aa[6]))
        _BH_HOLD.append(fnp.subtract(a12, aa[5], out=aa[2]))
        _BH_HOLD.append(fnp.stack((a11, a12), axis=0, out=acopies))
        _BH_HOLD.append(fnp.copyto(aa[3], a22))
        _BH_HOLD.append(fnp.subtract(b12, b11, out=bb[4]))
        _BH_HOLD.append(fnp.subtract(b22, bb[4], out=bb[5]))
        _BH_HOLD.append(fnp.subtract(b22, b12, out=bb[6]))
        _BH_HOLD.append(fnp.subtract(bb[5], b21, out=bb[3]))
        _BH_HOLD.append(fnp.stack((b11, b21, b22), axis=0, out=bcopies))
        _BH_HOLD.append(fnp.matmul(ca, cb, out=cc))
        p0, p1, p2, p3, p4, p5, p6 = products
        s0, s1, s2 = scratch
        tl, tr, bl, br = corners
        _BH_HOLD.append(fnp.add(p0, p1, out=tl))
        _BH_HOLD.append(fnp.add(p0, p5, out=s0))
        _BH_HOLD.append(fnp.add(s0, p6, out=s1))
        _BH_HOLD.append(fnp.add(s0, p4, out=s2))
        _BH_HOLD.append(fnp.add(s2, p2, out=tr))
        _BH_HOLD.append(fnp.subtract(s1, p3, out=bl))
        _BH_HOLD.append(fnp.add(s1, p4, out=br))


def _packed_read5(a,b,out,views):
    levels = 4
    m,k,n = int(a.shape[-2]),int(a.shape[-1]),int(b.shape[-1])
    assert int(b.shape[-2])==k and tuple(out.shape)==(m,n)
    assert m%2**levels==k%2**levels==n%2**levels==0
    forward,leaf_a,leaf_b,leaf,backward = _packed_frame(levels,m,k,n,views)
    for index,(aq,bq,aa,bb,acopies,bcopies) in enumerate(forward):
        if index==0:
            m2,k2,n2 = m//2,k//2,n//2
            a11,a12,a21,a22 = a[:m2,:k2],a[:m2,k2:],a[m2:,:k2],a[m2:,k2:]
            _BH_HOLD.append((a11,a12,a21,a22))
            b11,b12,b21,b22 = b[:k2,:n2],b[:k2,n2:],b[k2:,:n2],b[k2:,n2:]
            _BH_HOLD.append((b11,b12,b21,b22))
        else:
            a11,a12,a21,a22=aq
            b11,b12,b21,b22=bq
        _BH_HOLD.append(fnp.add(a21,a22,out=aa[4]))
        _BH_HOLD.append(fnp.subtract(aa[4],a11,out=aa[5]))
        _BH_HOLD.append(fnp.subtract(a11,a21,out=aa[6]))
        _BH_HOLD.append(fnp.subtract(a12,aa[5],out=aa[2]))
        _BH_HOLD.append(fnp.stack((a11,a12),axis=0,out=acopies))
        _BH_HOLD.append(fnp.copyto(aa[3],a22))
        _BH_HOLD.append(fnp.subtract(b12,b11,out=bb[4]))
        _BH_HOLD.append(fnp.subtract(b22,bb[4],out=bb[5]))
        _BH_HOLD.append(fnp.subtract(b22,b12,out=bb[6]))
        _BH_HOLD.append(fnp.subtract(bb[5],b21,out=bb[3]))
        _BH_HOLD.append(fnp.stack((b11,b21,b22),axis=0,out=bcopies))
    _read5_leaf_packets(m,k,n,views,leaf_a,leaf_b,leaf)
    for products,scratch,corners in backward:
        if corners is None:
            rows,columns=m//2,n//2
            corners=(out[:rows,:columns],out[:rows,columns:],out[rows:,:columns],out[rows:,columns:])
            _BH_HOLD.append(corners)
        p0,p1,p2,p3,p4,p5,p6=products
        s0,s1,s2=scratch
        tl,tr,bl,br=corners
        _BH_HOLD.append(fnp.add(p0,p1,out=tl))
        _BH_HOLD.append(fnp.add(p0,p5,out=s0))
        _BH_HOLD.append(fnp.add(s0,p6,out=s1))
        _BH_HOLD.append(fnp.add(s0,p4,out=s2))
        _BH_HOLD.append(fnp.add(s2,p2,out=tr))
        _BH_HOLD.append(fnp.subtract(s1,p3,out=bl))
        _BH_HOLD.append(fnp.add(s1,p4,out=br))
    _BH_HOLD.clear()
    return out


def _w5_port_product_body(a,b,out,views,levels,cache):
    m,k,n = int(a.shape[-2]),int(a.shape[-1]),int(b.shape[-1])
    assert int(b.shape[-2])==k and tuple(out.shape)==(m,n)
    assert m%2**levels==k%2**levels==n%2**levels==0
    forward,leaf_a,leaf_b,leaf,backward = _packed_frame(levels,m,k,n,views)
    for index,(aq,bq,aa,bb,acopies,bcopies) in enumerate(forward):
        if index==0:
            m2,k2,n2 = m//2,k//2,n//2
            a11,a12,a21,a22 = a[:m2,:k2],a[:m2,k2:],a[m2:,:k2],a[m2:,k2:]
            _BH_HOLD.append((a11,a12,a21,a22))
            b11,b12,b21,b22 = b[:k2,:n2],b[:k2,n2:],b[k2:,:n2],b[k2:,n2:]
            _BH_HOLD.append((b11,b12,b21,b22))
        else:
            a11,a12,a21,a22=aq
            b11,b12,b21,b22=bq
        if cache._leaf is None:
            _BH_HOLD.append(fnp.add(a21,a22,out=aa[4]))
            _BH_HOLD.append(fnp.subtract(aa[4],a11,out=aa[5]))
            _BH_HOLD.append(fnp.subtract(a11,a21,out=aa[6]))
            _BH_HOLD.append(fnp.subtract(a12,aa[5],out=aa[2]))
            _BH_HOLD.append(fnp.stack((a11,a12),axis=0,out=acopies))
            _BH_HOLD.append(fnp.copyto(aa[3],a22))
        _BH_HOLD.append(fnp.subtract(b12,b11,out=bb[4]))
        _BH_HOLD.append(fnp.subtract(b22,bb[4],out=bb[5]))
        _BH_HOLD.append(fnp.subtract(b22,b12,out=bb[6]))
        _BH_HOLD.append(fnp.subtract(bb[5],b21,out=bb[3]))
        _BH_HOLD.append(fnp.stack((b11,b21,b22),axis=0,out=bcopies))
    if cache._leaf is None:
        cached = fnp.empty(cache._shape,dtype=fnp.float32)
        cache._own_leaf(cached,cached)
        _BH_HOLD.append(fnp.copyto(cached,leaf_a))
    else:
        leaf_a = cache._leaf
    _BH_HOLD.append(fnp.matmul(leaf_a,leaf_b,out=leaf))
    for products,scratch,corners in backward:
        if corners is None:
            rows,columns=m//2,n//2
            corners=(out[:rows,:columns],out[:rows,columns:],out[rows:,:columns],out[rows:,columns:])
            _BH_HOLD.append(corners)
        p0,p1,p2,p3,p4,p5,p6=products
        s0,s1,s2=scratch
        tl,tr,bl,br=corners
        _BH_HOLD.append(fnp.add(p0,p1,out=tl))
        _BH_HOLD.append(fnp.add(p0,p5,out=s0))
        _BH_HOLD.append(fnp.add(s0,p6,out=s1))
        _BH_HOLD.append(fnp.add(s0,p4,out=s2))
        _BH_HOLD.append(fnp.add(s2,p2,out=tr))
        _BH_HOLD.append(fnp.subtract(s1,p3,out=bl))
        _BH_HOLD.append(fnp.add(s1,p4,out=br))
    _BH_HOLD.clear()
    return out
# END FABLE-ENGINEER HOLD-ONLY KERNEL
