"""Build a reviewable, invertible append-only kernel/dispatcher extension."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'cost_rule75_scalar.py'
EXPECTED = '5f66f94afdd2909584527a299c3989377d374289b8f5e1f379047b4176c7521c'
source = PARENT.read_text()
assert hashlib.sha256(source.encode()).hexdigest() == EXPECTED
tree = ast.parse(source)


def last_function(name):
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    return ast.get_source_segment(source, nodes[-1])


packed = last_function('_packed_product_cached')
assert packed.count('fnp.matmul(leaf_a,leaf_b,out=leaf)') == 1
packed = packed.replace('def _packed_product_cached(a,b,out,views,levels):',
                        'def _packed_read5(a,b,out,views):\n    levels = 4')
packed = packed.replace('fnp.matmul(leaf_a,leaf_b,out=leaf)',
                        '_read5_leaf_packets(m,k,n,views,leaf_a,leaf_b,leaf)')
rule = last_function('_rule_depth')
needle = "options = (0, 3, 4, 5) if kind == 'sq' else (0, 4)"
assert rule.count(needle) == 1
rule = rule.replace(needle, "options = (0, 3, 4, 5) if kind == 'sq' else ((0, 4, 5) if kind == 'read' else (0, 4))")
needle = 'calls = (18*depth+14)*len(packets)'
assert rule.count(needle) == 1
rule = rule.replace(needle, needle + "\n        if kind == 'read' and depth == 5:\n            calls = 142  # Original D4's86 calls, replacing one matmul by3x19 primitives.")
mm = last_function('_mm')
needle = 'out = _packed_product_cached(aa, bb, out, _SCRATCH[2], 4)'
assert mm.count(needle) == 1
mm = mm.replace(needle, 'out = (_packed_read5(aa, bb, out, _SCRATCH[2]) if rd == 5\n               else _packed_product_cached(aa, bb, out, _SCRATCH[2], 4))')
extension = '\n\n# Fixed three-packet D5 read extension, 2026-09-19.\n' + (HERE / 'kernel.py').read_text()
extension += '\n\n' + packed + '\n\n' + rule + '\n\n' + mm + '\n'
candidate = source + extension
ast.parse(candidate)
assert candidate[:-len(extension)] == source
out = HERE / 'candidate.py'
with out.open('x') as stream:
    stream.write(candidate)
receipt = dict(parent_sha256=EXPECTED, candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
               kernel_sha256=hashlib.sha256((HERE / 'kernel.py').read_bytes()).hexdigest(),
               builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               parent_byte_prefix_exact=True, source_bytes=len(candidate.encode()),
               arena_bytes=292576256 * 4, added_arena_bytes=(292576256 - 292225024) * 4,
               changes=['read5 schedule', 'read leaf location', 'read-only D5 shape option'])
with (HERE / 'build.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps(receipt))
