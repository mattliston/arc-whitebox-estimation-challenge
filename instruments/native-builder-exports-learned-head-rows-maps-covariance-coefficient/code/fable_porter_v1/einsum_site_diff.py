#!/usr/bin/env python3
"""TAG fable-porter (PT-13): pre-fire einsum gate (lead's crash law 2026-09-23: no inferred-symmetric f32 einsum ships). Lists every
einsum call site of <cand.py> and <ref.py> (the last platform-proven file), with hold wrappers _e2_H(x) -> x stripped, and diffs them as
multisets. Flags every site whose operands repeat (flopscope infers a symmetric output from operand identity and VALIDATES it at runtime,
raising SymmetryError at atol 1e-6) and every symmetric-tag call (as_symmetric, SymmetricTensor, symmetrize). Exit 1 if cand has any
repeated-operand or symmetric-tag site that ref does not. usage: einsum_site_diff.py <cand.py> <ref.py>"""
import ast, sys
from collections import Counter
class Unhold(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == '_e2_H' and len(node.args) == 1 and not node.keywords:
            return node.args[0]
        return node
def name(f): return f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ''
def sites(path):
    tree = Unhold().visit(ast.parse(open(path).read())); es, tags = [], []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            nm = name(n.func)
            if nm == 'einsum':
                ops = [ast.dump(a) for a in n.args[1:]]
                rep = len(ops) != len(set(ops))
                es.append((ast.unparse(n), rep, n.lineno))
            elif nm in ('as_symmetric', 'SymmetricTensor', 'symmetrize'):
                tags.append((ast.unparse(n), n.lineno))
    return es, tags
(ce, ct), (re_, rt) = sites(sys.argv[1]), sites(sys.argv[2])
cc, rc = Counter(s for s, _, _ in ce), Counter(s for s, _, _ in re_)
print(f"einsum sites: cand {len(ce)} ({len(cc)} distinct), ref {len(re_)} ({len(rc)} distinct)")
for s in sorted(set(cc) | set(rc)):
    if cc[s] != rc[s]:
        print(f"  {'ADDED' if cc[s] > rc[s] else 'REMOVED'} x{abs(cc[s] - rc[s])}: {s[:180]}")
rep_c = Counter(s for s, r, _ in ce if r); rep_r = Counter(s for s, r, _ in re_ if r)
print(f"repeated-operand einsum sites (inferred symmetric, runtime-validated): cand {sum(rep_c.values())}, ref {sum(rep_r.values())}")
new_rep = [s for s in rep_c if rep_c[s] > rep_r[s]]
for s in sorted(rep_c): print(f"  {'NEW ' if s in new_rep else 'old '}x{rep_c[s]}: {s[:180]}  (lines {[l for t, r, l in ce if t == s]})")
tc, tr = Counter(s for s, _ in ct), Counter(s for s, _ in rt); new_tag = [s for s in tc if tc[s] > tr[s]]
print(f"symmetric-tag calls: cand {sum(tc.values())}, ref {sum(tr.values())}" + ''.join(f"\n  NEW x{tc[s] - tr[s]}: {s[:180]}" for s in new_tag))
verdict = 'FAIL' if new_rep or new_tag else 'PASS'
print(f"VERDICT {verdict}: {len(new_rep)} new repeated-operand einsum site(s), {len(new_tag)} new symmetric-tag call(s) vs ref")
sys.exit(1 if verdict == 'FAIL' else 0)
