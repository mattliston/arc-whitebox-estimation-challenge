"""Diagnostic-only adapter of the frozen capacity-v7 native client."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
PORT = HERE.parents[2] / 'fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7'
sys.path.insert(0, str(PORT))
from common_linux_v7 import sha, verified
BASE = PORT/'client_linux_v7.py'
assert sha(BASE) == 'afbf67e08e0741694c9089c797651965eab33cbec277b5bb5effb9dc9a9940d8'
recipe_path = Path(os.environ['ARC_BIRTH_CONSUMER_RECIPE'])
assert sha(recipe_path) == os.environ['ARC_BIRTH_CONSUMER_RECIPE_SHA']
recipe = json.loads(recipe_path.read_text())

def install_observer(mod, out, root):
    assert sha(mod.__file__) == recipe['teacher']['effective_sha256']
    path = verified(root, recipe['hook'])
    spec = importlib.util.spec_from_file_location('arc_birth_consumer_hook', path)
    hook = importlib.util.module_from_spec(spec); spec.loader.exec_module(hook)
    return hook.install(mod, out, recipe)

source = BASE.read_text()
fragment = HERE/'fragment_mark_complete_line_v1.py'
assert sha(fragment) == 'ca5f41cad45a5a380b2782d4436d2149a9af50955c2af7bfc27746494e63b684'
mark_node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'mark')
lines = source.splitlines(keepends=True)
source = ''.join(lines[:mark_node.lineno-1]) + fragment.read_text() + ''.join(lines[mark_node.end_lineno:])
patches = [
    ('assert mod._FAST_RESCUE_FRAC == {}\n\n# Same metadata-only',
     'assert mod._FAST_RESCUE_FRAC == {}\n_student_observer = install_observer(mod, out, a.root)\n\n# Same metadata-only'),
    ('def evaluate(rank, phase):\n', 'def evaluate(rank, phase):\n    _student_observer.begin(rank, phase)\n'),
    ("print('ALL_NETWORKS_COMPLETE', flush=True)", "_student_observer.finish()\nprint('ALL_NETWORKS_COMPLETE', flush=True)"),
    ("out / 'rows.jsonl'", "out / 'diagnostic_rows.jsonl'"),
    ("f'pred_{rank}.bin'", "f'diagnostic_pred_{rank}.bin'"),
    ('harness_sha256=sha(__file__),', 'harness_sha256=sha(__file__), diagnostic_only=True,'),
]
for old, new in patches:
    assert source.count(old) == 1, old
    source = source.replace(old, new)
exec(compile(source, str(BASE)+':birth_consumer_capture', 'exec'), globals())
