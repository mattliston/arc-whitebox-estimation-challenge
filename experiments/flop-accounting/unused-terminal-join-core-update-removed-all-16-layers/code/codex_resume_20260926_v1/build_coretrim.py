"""Remove the unused terminal old-tier core; optionally exploit Gram symmetry.

The core is read only during a later join. All factors, the lazy map, and every
prediction consumer remain intact. Historical candidate sources are immutable.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
base = Path(sys.argv[1]) if len(sys.argv) > 1 else P2/'fable_fire_v1/fire_candS29/candS29_S28_d7cp_d7sq_mem.py'
tag = sys.argv[2] if len(sys.argv) > 2 else 'candidate_coretrim_v1'
sym = len(sys.argv) > 3 and sys.argv[3] == 'sym'
src = base.read_text()
tree = ast.parse(src)
node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_shared_old384_push'][-1]
fn = '\n'.join(src.splitlines()[node.lineno - 1:node.end_lineno]) + '\n'
old = "        joined_core = _shared_old384_mm(Fj, _e2_H(_e2_H(weight[:, None]) * _e2_H(Fj.T)))"
assert fn.count(old) == 1
mm = '_sym_upper' if sym else '_shared_old384_mm'
new = "        _need_core = layer < max(_JOIN_SCHEDULE)\n"
new += f"        joined_core = {mm}(Fj, _e2_H(_e2_H(weight[:, None]) * _e2_H(Fj.T))) if _need_core else None"
fn = fn.replace(old, new)
old = "            core = _shared_old384_mm(overlap, _e2_H(_shared_old384_mm(state['core'], _e2_H(overlap.T))))\n            core = core + joined_core"
assert fn.count(old) == 1
fn = fn.replace(old, "            core = None\n            if _need_core:\n                core = _shared_old384_mm(overlap, _e2_H(_shared_old384_mm(state['core'], _e2_H(overlap.T))))\n                core = core + joined_core")
old = "        state['core'] = _shared_old384_finite(core, 'weighted Gram core')"
assert fn.count(old) == 1
fn = fn.replace(old, "        state['core'] = _shared_old384_finite(core, 'weighted Gram core') if _need_core else None")
fragment = '\n\n# Terminal core removal, 2026-09-26. See build_coretrim.py.\n' + fn
out = src + fragment
ast.parse(out)
for path, value in [(HERE/(tag+'.py'), out), (HERE/(tag+'_fragment.py'), fragment)]:
    with path.open('x') as f: f.write(value)
manifest = dict(base=str(base.resolve()), base_sha256=hashlib.sha256(src.encode()).hexdigest(),
                candidate_sha256=hashlib.sha256(out.encode()).hexdigest(), nonterminal_symmetric=sym,
                invariant='state.core is accessed only at joins; last join has no future core reader')
with (HERE/(tag+'_manifest.json')).open('x') as f: json.dump(manifest, f, indent=2)
print(json.dumps(manifest))
