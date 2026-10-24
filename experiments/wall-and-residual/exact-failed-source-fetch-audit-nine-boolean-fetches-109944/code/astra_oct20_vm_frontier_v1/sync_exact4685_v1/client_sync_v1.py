"""Diagnostic-only exact4685 adapter; original v7 numerical boundary unchanged."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[1]
PORT = RUNS / 'fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7'
sys.path.insert(0, str(PORT))
from common_linux_v7 import sha, verified

BASE = PORT / 'client_linux_v7.py'
assert sha(BASE) == 'afbf67e08e0741694c9089c797651965eab33cbec277b5bb5effb9dc9a9940d8'
MODE = os.environ['ARC_EXACT4685_SYNC_MODE']
assert MODE in ('control', 'observe')
RECIPE_PATH = HERE / 'RECIPE_v1.json'
assert sha(RECIPE_PATH) == os.environ['ARC_EXACT4685_SYNC_RECIPE_SHA']
RECIPE = json.loads(RECIPE_PATH.read_text())


def install_sync_observer(candidate, root):
    assert sha(candidate) == RECIPE['candidate']['effective_sha256']
    observer_path = verified(root, RECIPE['observer'])
    spec = importlib.util.spec_from_file_location('exact4685_sync_observer', observer_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    observer = helper.SyncObserver(str(candidate))
    observer.install()
    return observer


def adapted_source():
    source = BASE.read_text()
    fragment = RUNS / 'astra_oct20_bill_v1/edge_prepost_capture_v1/fragment_mark_complete_line_v1.py'
    assert sha(fragment) == 'ca5f41cad45a5a380b2782d4436d2149a9af50955c2af7bfc27746494e63b684'
    mark_node = next(n for n in ast.parse(source).body
                     if isinstance(n, ast.FunctionDef) and n.name == 'mark')
    lines = source.splitlines(keepends=True)
    source = ''.join(lines[:mark_node.lineno - 1]) + fragment.read_text() + ''.join(lines[mark_node.end_lineno:])
    patches = [
        ('assert sha(a.candidate) == a.candidate_sha',
         "assert sha(a.candidate) == a.candidate_sha == RECIPE['candidate']['effective_sha256']\nassert a.ranks == '35' and not a.warmup_each and a.setup == 'paced'"),
        ("out / 'rows.jsonl'", "out / 'diagnostic_rows.jsonl'"),
        ("f'pred_{rank}.bin'", "f'diagnostic_pred_{rank}.bin'"),
        ('harness_sha256=sha(__file__),',
         "harness_sha256=sha(__file__), diagnostic_only=True, sync_mode=MODE, sync_recipe_sha256=sha(RECIPE_PATH),"),
    ]
    if MODE == 'observe':
        patches += [
            ("start = time.monotonic()\nmod = types.ModuleType('release_candidate')",
             "sync_observer = install_sync_observer(a.candidate, a.root)\nsync_observer.phase('candidate_import')\nstart = time.monotonic()\nmod = types.ModuleType('release_candidate')"),
            ('import_s = time.monotonic() - start\n',
             "import_s = time.monotonic() - start\nsync_observer.phase('harness')\n"),
            ('est.setup(types.SimpleNamespace(',
             "sync_observer.phase('setup')\nest.setup(types.SimpleNamespace("),
            ('setup_s = time.monotonic() - start; m1 = mark()',
             "setup_s = time.monotonic() - start; sync_observer.phase('harness'); m1 = mark()"),
            ('        prediction = est.predict(MLP(weights, rank), 2**41)\n',
             "        sync_observer.phase('predict', rank)\n        prediction = est.predict(MLP(weights, rank), 2**41)\n        sync_observer.phase('required_final_fetch', rank)\n"),
            ('        wall = time.monotonic() - start\n',
             "        wall = time.monotonic() - start\n        sync_observer.phase('harness', rank)\n"),
        ]
    for old, new in patches:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    return source


if __name__ == '__main__':
    try:
        exec(compile(adapted_source(), str(BASE) + ':exact4685_sync_' + MODE, 'exec'), globals())
    finally:
        observer = globals().get('sync_observer')
        if observer is not None:
            observer.write(out / 'sync_diagnostic_v1.json')
            observer.uninstall()
