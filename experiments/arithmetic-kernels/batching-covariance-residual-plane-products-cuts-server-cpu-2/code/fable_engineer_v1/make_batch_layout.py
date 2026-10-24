#!/usr/bin/env python3
"""TAG fable-engineer (E-35): arena layout for P independent Winograd products batched through one kernel call (capacity P x M x K x N,
depth L), in the spec format of _packed_frame / _cp3_bind_arena_views (a_j/b_j interleaved for j=1..L, 2L unused placeholders,
leaf, inter_{L-1..1}, u_{L..1}). Every array carries the batch axis P right before its two trailing dims, so the bank frame
builders (_a_frame, _b_frame) and the per-level ops batch the P products with no code change. Levels alternate between bank P0
(level L, L-2, ...) and bank Q (level L-1, ...); the LEAF sits after everything else, disjoint from the level-(L-1) and level-L
arrays, so the last-level copy elimination (leaf products read level L-1 quadrants directly) is always valid. A lifetime
simulation (including those elimination reads) checks every step. usage: make_batch_layout.py <P> <L> <M> <K> <N> <arena> <name>"""
import sys
P, L, M, K, N, ARENA = (int(x) for x in sys.argv[1:7]); NAME = sys.argv[7]
def size(shape):
    s = 1
    for x in shape: s *= x
    return s
shapes = {}
for j in range(1, L + 1):
    shapes[f"a{j}"] = (7,) * j + (P, M >> j, K >> j); shapes[f"b{j}"] = (7,) * j + (P, K >> j, N >> j)
shapes["leaf"] = (7,) * L + (P, M >> L, N >> L)
for l in range(1, L):
    shapes[f"inter{l}"] = (7,) * l + (P, M >> l, N >> l)
for l in range(1, L + 1):
    shapes[f"u{l}"] = (3,) + (7,) * (l - 1) + (P, M >> l, N >> l)
B0 = 0; Q = size(shapes[f"a{L}"]) + size(shapes[f"b{L}"])
start = {}
for j in range(1, L + 1):
    base = B0 if (L - j) % 2 == 0 else Q
    start[f"a{j}"] = base; start[f"b{j}"] = base + size(shapes[f"a{j}"])
bank = B0
for l in range(L, 0, -1):
    if l > 1:
        start[f"inter{l-1}"] = bank; start[f"u{l}"] = bank + size(shapes[f"inter{l-1}"])
    else:
        start["u1"] = bank
    bank = Q if bank == B0 else B0
R = max(start[k] + size(shapes[k]) for k in start)
start["leaf"] = R
spec = {k: (start[k], start[k] + size(shapes[k]), shapes[k]) for k in shapes}
extent = max(s[1] for s in spec.values())
assert extent <= ARENA, (extent, ARENA)
def iv(k): return (spec[k][0], spec[k][1])
def overlap(x, y): return x[0] < y[1] and y[0] < x[1]
steps = []
for j in range(1, L + 1):
    steps.append((f"forward {j}", [f"a{j-1}", f"b{j-1}"] if j > 1 else [], [f"a{j}", f"b{j}"]))
steps.append(("leaf", [f"a{L}", f"b{L}", f"a{L-1}", f"b{L-1}"], ["leaf"]))
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
    sh = (4,) + (7,) * (j - 1) + (P, M >> j, K >> j); order.append((0, size(sh), sh))
for j in range(1, L + 1):
    sh = (4,) + (7,) * (j - 1) + (P, K >> j, N >> j); order.append((0, size(sh), sh))
order.append(spec["leaf"])
for l in range(L - 1, 0, -1):
    order.append(spec[f"inter{l}"])
for l in range(L, 0, -1):
    order.append(spec[f"u{l}"])
assert len(order) == 6 * L
for s in order:
    assert s[1] - s[0] == size(s[2]) and s[1] <= ARENA
print(f"# E-35 batched layout: P={P}, capacity {(M, K, N)}, levels {L}, extent {extent} of {ARENA} ({100*extent/ARENA:.1f}%), leaf disjoint, lifetime check passed")
print(f"{NAME} = " + repr(tuple(order)))
