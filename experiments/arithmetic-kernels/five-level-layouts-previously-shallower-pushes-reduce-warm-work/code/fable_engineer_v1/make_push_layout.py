#!/usr/bin/env python3
"""TAG fable-engineer (E-28): batched Winograd arena layout for PUSH products at any depth L (capacity M x K x N), in the spec format
consumed by _packed_frame / _cp3_bind_arena_views (interleaved a_j/b_j for j=1..L, 2L unused quadrant placeholders, leaf,
inter_{L-1..1}, u_{L..1}). Same placement and lifetime check as fable_wall_repair_v1/make_push6_layout.py (W-F, frozen; this file
generalizes L, capacity and arena size). usage: make_push_layout.py <L> <M> <K> <N> <arena floats> <python name>"""
import sys
L, M, K, N, ARENA = (int(x) for x in sys.argv[1:6]); NAME = sys.argv[6]
def size(shape):
    s = 1
    for x in shape: s *= x
    return s
shapes = {}
for j in range(1, L + 1):
    shapes[f"a{j}"] = (7,) * j + (M >> j, K >> j); shapes[f"b{j}"] = (7,) * j + (K >> j, N >> j)
shapes["leaf"] = (7,) * L + (M >> L, N >> L)
for l in range(1, L):
    shapes[f"inter{l}"] = (7,) * l + (M >> l, N >> l)
for l in range(1, L + 1):
    shapes[f"u{l}"] = (3,) + (7,) * (l - 1) + (M >> l, N >> l)
P = 0; Q = size(shapes[f"a{L}"]) + size(shapes[f"b{L}"])
start = {}
for j in range(1, L + 1):
    base = P if (L - j) % 2 == 0 else Q      # the last forward level sits at P (the leaf follows it at Q); levels alternate below it
    start[f"a{j}"] = base; start[f"b{j}"] = base + size(shapes[f"a{j}"])
start["leaf"] = Q
bank = P
for l in range(L, 0, -1):
    if l > 1:
        start[f"inter{l-1}"] = bank; start[f"u{l}"] = bank + size(shapes[f"inter{l-1}"])
    else:
        start["u1"] = bank
    bank = Q if bank == P else P
spec = {k: (start[k], start[k] + size(shapes[k]), shapes[k]) for k in shapes}
extent = max(s[1] for s in spec.values())
assert extent <= ARENA, (extent, ARENA)
def iv(k): return (spec[k][0], spec[k][1])
def overlap(x, y): return x[0] < y[1] and y[0] < x[1]
steps = []
for j in range(1, L + 1):
    steps.append((f"forward {j}", [f"a{j-1}", f"b{j-1}"] if j > 1 else [], [f"a{j}", f"b{j}"]))
steps.append(("leaf", [f"a{L}", f"b{L}"], ["leaf"]))
for l in range(L, 0, -1):
    cur = "leaf" if l == L else f"inter{l}"
    steps.append((f"unwind {l}", [cur], [f"u{l}"] + ([f"inter{l-1}"] if l > 1 else [])))
for name, reads, writes in steps:
    for i, w in enumerate(writes):
        for w2 in writes[i + 1:]:
            assert not overlap(iv(w), iv(w2)), (name, w, w2)
        for r in reads:
            assert not overlap(iv(w), iv(r)), (name, w, r)
order = []
for j in range(1, L + 1):
    order += [spec[f"a{j}"], spec[f"b{j}"]]
for j in range(1, L + 1):
    sh = (4,) + (7,) * (j - 1) + (M >> j, K >> j); order.append((0, size(sh), sh))
for j in range(1, L + 1):
    sh = (4,) + (7,) * (j - 1) + (K >> j, N >> j); order.append((0, size(sh), sh))
order.append(spec["leaf"])
for l in range(L - 1, 0, -1):
    order.append(spec[f"inter{l}"])
for l in range(L, 0, -1):
    order.append(spec[f"u{l}"])
assert len(order) == 6 * L
for s in order:
    assert s[1] - s[0] == size(s[2]) and s[1] <= ARENA
print(f"# E-28 batched D{L} push layout: capacity {(M, K, N)}, levels {L}, arena extent {extent} of {ARENA} ({100*extent/ARENA:.1f}%), lifetime check passed")
print(f"{NAME} = " + repr(tuple(order)))
