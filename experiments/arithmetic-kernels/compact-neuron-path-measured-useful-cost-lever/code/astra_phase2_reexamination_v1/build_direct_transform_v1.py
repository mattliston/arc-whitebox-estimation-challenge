"""Remove redundant Winograd transform copies without changing arithmetic.

Each operand needs four sums and three copies, not four sums plus seven
copies. The transform's child buffers are disjoint from its input parity,
so intermediate sums can be written directly into their final child slots.
The existing lifetime arena and unwind are unchanged for this certificate.
"""
import argparse
import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
target = HERE / args.out
if target.exists():
    raise FileExistsError(target)
source = (HERE / args.source).read_text()
old_a = '''        scratch_a = dependency_a[index][..., :m2, :k2]
        fnp.add(a21, a22, out=scratch_a[0])
        fnp.subtract(scratch_a[0], a11, out=scratch_a[1])
        fnp.subtract(a11, a21, out=scratch_a[2])
        fnp.subtract(a12, scratch_a[1], out=scratch_a[3])
        next_a = a_views[index][..., :m2, :k2]
        for destination, source in zip(
            next_a,
            (a11, a12, scratch_a[3], a22, scratch_a[0], scratch_a[1], scratch_a[2]),
        ):
            fnp.copyto(destination, source)'''
new_a = '''        next_a = a_views[index][..., :m2, :k2]
        fnp.add(a21, a22, out=next_a[4])
        fnp.subtract(next_a[4], a11, out=next_a[5])
        fnp.subtract(a11, a21, out=next_a[6])
        fnp.subtract(a12, next_a[5], out=next_a[2])
        fnp.copyto(next_a[0], a11)
        fnp.copyto(next_a[1], a12)
        fnp.copyto(next_a[3], a22)'''
old_b = '''        scratch_b = dependency_b[index][..., :k2, :n2]
        fnp.subtract(b12, b11, out=scratch_b[0])
        fnp.subtract(b22, scratch_b[0], out=scratch_b[1])
        fnp.subtract(b22, b12, out=scratch_b[2])
        fnp.subtract(scratch_b[1], b21, out=scratch_b[3])
        next_b = b_views[index][..., :k2, :n2]
        for destination, source in zip(
            next_b,
            (b11, b21, b22, scratch_b[3], scratch_b[0], scratch_b[1], scratch_b[2]),
        ):
            fnp.copyto(destination, source)'''
new_b = '''        next_b = b_views[index][..., :k2, :n2]
        fnp.subtract(b12, b11, out=next_b[4])
        fnp.subtract(b22, next_b[4], out=next_b[5])
        fnp.subtract(b22, b12, out=next_b[6])
        fnp.subtract(next_b[5], b21, out=next_b[3])
        fnp.copyto(next_b[0], b11)
        fnp.copyto(next_b[1], b21)
        fnp.copyto(next_b[2], b22)'''
assert source.count(old_a) == source.count(old_b) == 2
source = source.replace(old_a, new_a).replace(old_b, new_b)
ast.parse(source)
target.write_text(source)
print(target)
