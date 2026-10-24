"""Offline lifetime proof and code transform for the streamed D6 leaf."""
from fractions import Fraction
from math import prod
from packed_arena_generic import layout


def streamed_layout(m, k, n):
    size, base, regions = layout(m, k, n, 6)
    batch = 7**5
    product = batch*(m//32)*(n//32)
    work_p = 3*batch*(m//64)*(n//64)
    work_a = 3*batch*(m//64)*(k//64)
    work_b = 3*batch*(k//64)*(n//64)
    # Current D5 operands reside in prev_a/prev_b. Final-even operands are
    # dead because their D6 transforms are never created. Product + p work
    # use the beginning of that dead region; a work follows them. The old
    # dependency region, unused by the direct-write descent, holds b work.
    a_start = product+work_p
    a_stop = a_start+work_a
    b_start = regions["dep"][0]
    b_stop = b_start+work_b
    assert regions["product0"][0] == 0
    assert regions["u"][0] == product
    assert a_stop <= regions["prev_a"][0]
    assert b_stop <= regions["dep"][1]
    assert product+work_p <= a_start
    assert regions["prev_b"][1] <= b_start
    extra = ((a_start,a_stop,(3,)+(7,)*5+(m//64,k//64)),
             (b_start,b_stop,(3,)+(7,)*5+(k//64,n//64)))
    assert all(stop-start==prod(shape) for start,stop,shape in extra)
    return size, base+extra, dict(regions, streamed_a=(a_start,a_stop),streamed_b=(b_start,b_stop))


def transform_kernel(source):
    edits = {
        "_split_views_6(arena_views)": "_split_views_6(arena_views[:36])",
        "for index in range(levels):": "for index in range(levels-1):",
        "    current = leaf_view[..., :m // 2 ** levels, :n // 2 ** levels]\n    fnp.matmul(current_a, current_b, out=current)":
            "    current = intermediates[0][..., :m//32, :n//32]\n    streamed_leaf(current_a,current_b,current,arena_views[36],arena_views[37],u_scratch[0])",
        "enumerate(range(levels, 0, -1))": "enumerate(range(levels-1, 0, -1),1)",
    }
    for old,new in edits.items():
        assert source.count(old)==1,old
        source=source.replace(old,new)
    return source

