"""Append only the fixed bounded Read6 kernel and shape choice to Weight5."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'weight5_reuse_v1/candidate.py'
EXPECTED = 'a432b94336a1ec45f9140d2e0a7ed808a8a7826deb72034a7ca185d331c823b9'
source = PARENT.read_text()
assert hashlib.sha256(source.encode()).hexdigest() == EXPECTED
tree = ast.parse(source)


def last(name):
    return ast.get_source_segment(source, [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name][-1])


def once(s, a, b):
    assert s.count(a) == 1, a
    return s.replace(a, b, 1)


packed = once(last('_packed_product_cached'), 'def _packed_product_cached(a,b,out,views,levels):',
              'def _packed_read6(a,b,out,views):\n    levels = 4')
packed = once(packed, 'fnp.matmul(leaf_a,leaf_b,out=leaf)',
              '_read6_leaf_packets(m,k,n,views,leaf_a,leaf_b,leaf)')
rule = once(last('_rule_depth'), "((0, 4, 5) if kind == 'read' else (0, 4))",
             "((0, 4, 5, 6) if kind == 'read' else (0, 4))")
needle = "        value = bill + 750000*calls"
rule = once(rule, needle, "        if kind == 'read' and depth == 6:\n            calls = 233  # D4's86-1 plus four two-level child products of37 operations.\n" + needle)
mm = last('_mm')
needle = "out = (_packed_read5(aa, bb, out, _SCRATCH[2]) if rd == 5\n               else _packed_product_cached(aa, bb, out, _SCRATCH[2], 4))"
replacement = "out = (_packed_read6(aa, bb, out, _SCRATCH[2]) if rd == 6\n               else (_packed_read5(aa, bb, out, _SCRATCH[2]) if rd == 5\n                     else _packed_product_cached(aa, bb, out, _SCRATCH[2], 4)))"
mm = once(mm, needle, replacement)
extension = '\n\n# Fixed four-packet D6 read extension, 2026-09-19.\n' + (HERE / 'kernel.py').read_text()
extension += '\n\n' + '\n\n'.join([packed, rule, mm]) + '\n'
ast.parse(extension)
candidate = source + extension
assert candidate[:-len(extension)] == source
with (HERE / 'candidate.py').open('x') as stream:
    stream.write(candidate)
receipt = dict(parent_sha256=EXPECTED, candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
               kernel_sha256=hashlib.sha256((HERE / 'kernel.py').read_bytes()).hexdigest(),
               builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               parent_byte_prefix_exact=True, arena_bytes=392150528 * 4,
               extra_arena_bytes=(392150528 - 292576256) * 4, truth_accessed=False)
with (HERE / 'build.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps(receipt))
