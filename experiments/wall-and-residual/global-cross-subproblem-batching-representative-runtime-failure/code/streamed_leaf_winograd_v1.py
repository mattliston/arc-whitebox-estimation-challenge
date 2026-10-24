"""Metered rank-seven final level without materializing a seven-way batch.

The four A/B additions and seven output additions retain the established
Winograd arithmetic. Three identity copies per input disappear. Products
are issued separately over the previous five batch levels. All numerical
operations are FNP, with explicit, disjoint output/work views.
"""
import flopscope.numpy as fnp


def streamed_leaf(a, b, out, wa, wb, wp):
    m2 = int(a.shape[-2])//2
    k2 = int(a.shape[-1])//2
    n2 = int(b.shape[-1])//2
    a11 = a[..., :m2, :k2]
    a12 = a[..., :m2, k2:2*k2]
    a21 = a[..., m2:2*m2, :k2]
    a22 = a[..., m2:2*m2, k2:2*k2]
    b11 = b[..., :k2, :n2]
    b12 = b[..., :k2, n2:2*n2]
    b21 = b[..., k2:2*k2, :n2]
    b22 = b[..., k2:2*k2, n2:2*n2]
    c11 = out[..., :m2, :n2]
    c12 = out[..., :m2, n2:2*n2]
    c21 = out[..., m2:2*m2, :n2]
    c22 = out[..., m2:2*m2, n2:2*n2]
    wa = wa[..., :m2, :k2]
    wb = wb[..., :k2, :n2]
    wp = wp[..., :m2, :n2]
    fnp.matmul(a11, b11, out=wp[0])
    fnp.matmul(a12, b21, out=c11)
    fnp.add(wp[0], c11, out=c11)
    fnp.add(a21, a22, out=wa[0])
    fnp.subtract(wa[0], a11, out=wa[1])
    fnp.subtract(b12, b11, out=wb[0])
    fnp.subtract(b22, wb[0], out=wb[1])
    fnp.matmul(wa[1], wb[1], out=wp[1])
    fnp.add(wp[0], wp[1], out=wp[1])
    fnp.subtract(a11, a21, out=wa[2])
    fnp.subtract(b22, b12, out=wb[2])
    fnp.matmul(wa[2], wb[2], out=wp[2])
    fnp.add(wp[1], wp[2], out=wp[2])
    fnp.matmul(wa[0], wb[0], out=c22)
    fnp.add(wp[1], c22, out=c12)
    fnp.add(wp[2], c22, out=c22)
    fnp.subtract(a12, wa[1], out=wa[0])
    fnp.matmul(wa[0], b22, out=wp[0])
    fnp.add(c12, wp[0], out=c12)
    fnp.subtract(wb[1], b21, out=wb[0])
    fnp.matmul(a22, wb[0], out=wp[1])
    fnp.subtract(wp[2], wp[1], out=c21)
    return out

