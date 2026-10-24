#!/usr/bin/env python3
"""TAG fable-wall-repair (W-I): variantI_rule<target>.py = variantH_splitframes.py with a DETERMINISTIC size rule for the Winograd
depth of every packed product (never deeper than c2's own schedule): plain matmul for the smallest products, then depth 3/4/5 (squares),
4/6 (pushes), 4 (reads) above naive-flop thresholds fitted to the platform-cost greedy (bench_depths.json + the request/compute model).
usage: build_variantI.py <target_s> <sq thresholds d3,d4,d5> <push thresholds d4,d6> <read threshold d4>"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
target = sys.argv[1]; sq = [float(x) for x in sys.argv[2].split(",")]; push = [float(x) for x in sys.argv[3].split(",")]; read = float(sys.argv[4])
src = (HERE / "variantH_splitframes.py").read_text()

def rep(text, old, new, count=1):
    assert text.count(old) == count, (text.count(old), old[:90])
    return text.replace(old, new)

rule = f"""
# W-I: deterministic depth rule (platform wall repair): the maximum Winograd depth by product size (naive flops 2mkn), fitted for a
# {target} s platform estimate; products never go deeper than c2's own schedule. Below the first threshold: a plain matmul.
_RULE_SQ = (({sq[0]:.3e}, 3), ({sq[1]:.3e}, 4), ({sq[2]:.3e}, 5))
_RULE_PUSH = (({push[0]:.3e}, 4), ({push[1]:.3e}, 6))
_RULE_READ = (({read:.3e}, 4),)

def _rule_depth(kind, m, k, n):
    f = 2.0 * m * k * n; d = 0
    for t, hi in {{'sq': _RULE_SQ, 'push': _RULE_PUSH, 'read': _RULE_READ}}[kind]:
        if f >= t:
            d = hi
    while d and (m % 2 ** d or k % 2 ** d or n % 2 ** d):
        d -= 1
    return d

"""
# insert the rule before the ACTIVE _square_schedule_reference definition (the second block of square schedule code)
anchor = "def _square_schedule_reference(m,k,n):\n"
assert src.count(anchor) == 1
src = src.replace(anchor, rule + anchor)

_i = src.index("    depth=min(depth,_SQUARE_DEPTH_CAP)   # W-A2")            # the ACTIVE (last) _square_schedule carries the W-A2 cap line
_start = src.rindex("def _square_schedule(m,k,n):\n", 0, _i); _end = src.index("def _square_product(a,b,m,k,n):", _i)
old_sched = src[_start:_end]
assert src.count(old_sched) == 1, src.count(old_sched)
new_sched = """def _square_schedule(m,k,n):
    if m==n:
        depth=_SQUARE_EQUAL_DEPTHS[m//32-1][k//32-1]
    elif (m==k and n==1024) or (k==n and m==1024):
        depth=_SQUARE_CROSS_DEPTHS[k//32-1]
    else:
        depth,_p=_square_schedule_reference(m,k,n)
    depth=min(depth,_SQUARE_DEPTH_CAP,_rule_depth('sq',m,k,n))   # W-I: size rule caps the depth (0 = plain matmul)
    unit=2**depth
    return depth,((m+unit-1)//unit*unit,(k+unit-1)//unit*unit,(n+unit-1)//unit*unit)


"""
src = src.replace(old_sched, new_sched)

# rectangle path of the ACTIVE _mm (the last definition): rule-selected kernel
old_rect = """    if max(m,k,n)<=1024:
        return _square_product(a,b,m,k,n)
    mp = (m + 31) // 32 * 32
    kp = (k + 31) // 32 * 32
    npad = (n + 31) // 32 * 32
    if mp != m or kp != k:
        aa = fnp.zeros((mp, kp), dtype=fnp.float32)
        fnp.copyto(aa[:m, :k], a)
    else:
        aa = a
    if kp != k or npad != n:
        bb = fnp.zeros((kp, npad), dtype=fnp.float32)
        fnp.copyto(bb[:k, :n], b)
    else:
        bb = b
    out = fnp.empty((mp, npad), dtype=fnp.float32)
    out = _packed_product_4(aa, bb, out, _SCRATCH[1] if n > 1024 else _SCRATCH[2])
    return out if (mp == m and npad == n) else out[:m, :n]   # W-H
"""
new_rect = """    if max(m,k,n)<=1024:
        return _square_product(a,b,m,k,n)
    rd = _rule_depth('push' if n > 1024 else 'read', m, k, n)   # W-I
    if rd == 0:
        return a @ b
    mp = (m + 31) // 32 * 32
    kp = (k + 31) // 32 * 32
    npad = (n + 31) // 32 * 32
    if mp != m or kp != k:
        aa = fnp.zeros((mp, kp), dtype=fnp.float32)
        fnp.copyto(aa[:m, :k], a)
    else:
        aa = a
    if kp != k or npad != n:
        bb = fnp.zeros((kp, npad), dtype=fnp.float32)
        fnp.copyto(bb[:k, :n], b)
    else:
        bb = b
    out = fnp.empty((mp, npad), dtype=fnp.float32)
    if n > 1024:
        out = _packed_product_4(aa, bb, out, _SCRATCH[1]) if rd == 6 else _packed_product_cached(aa, bb, out, _SCRATCH[1], 4)   # W-I
    else:
        out = _packed_product_cached(aa, bb, out, _SCRATCH[2], 4)
    return out if (mp == m and npad == n) else out[:m, :n]   # W-H
"""
src = rep(src, old_rect, new_rect)
out = HERE / f"variantI_rule{target}.py"; out.write_text(src); print("wrote", out, len(src.splitlines()), "lines")
