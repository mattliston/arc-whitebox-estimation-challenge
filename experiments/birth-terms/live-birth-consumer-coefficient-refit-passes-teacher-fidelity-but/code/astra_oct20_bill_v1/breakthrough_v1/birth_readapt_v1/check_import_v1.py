"""Full NumPy-free source import and exact birth-literal-only proof."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

assert importlib.util.find_spec('numpy') is None
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
p = argparse.ArgumentParser()
p.add_argument('--build', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
build = json.loads(a.build.read_text())
stub = types.ModuleType('whestbench')
stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
suffixes, cases = [], []
for role, record in build['outputs'].items():
    path, parent = Path(record['path']), Path(record['parent_path'])
    assert sha(path) == record['sha256'] and sha(parent) == record['parent_sha256']
    source, original = path.read_text(), parent.read_text()
    assert source.startswith(original)
    tail = source[len(original):]
    suffixes.append(tail)
    nodes = ast.parse(tail).body
    assert len(nodes) == 3 and isinstance(nodes[0], ast.Assert)
    assert isinstance(nodes[1], ast.Assign) and nodes[1].targets[0].id == '_BDNH_A_LITERAL'
    assert ast.unparse(nodes[2]) == '_FAST_RESCUE_FRAC = {}'
    values = ast.literal_eval(nodes[1].value)
    assert len(values) == 64 and all(len(row) == 4 for row in values)
    mod = types.ModuleType('birth_readapt_' + role)
    mod.__file__ = str(path)
    sys.modules[mod.__name__] = mod
    exec(compile(source, str(path), 'exec'), mod.__dict__)
    assert mod._BDNH_A_LITERAL == values and mod._BDNH_PARAMETERS is None
    assert mod._GNN_PARAMS is None and not mod._E2_GNN_STATE and not mod._E2_BDNH
    assert mod._FAST_RESCUE_FRAC == {} and mod._VMFR_CM == 8
    assert mod._gn_message_edge_source_base is mod._g5_message_edge_base
    assert mod._O20_FLATEXTRA_ON is False and mod._hzd_load is mod._O20_FLATEXTRA_HZD_PARENT
    assert mod._rs_load_b_owned_ql_v1 is mod._O20_FLATEXTRA_OWNED_PARENT
    cases.append(dict(role=role, source_sha256=sha(path), birth_A_only=True,
                      cold_caches_empty=True, loader_binding_preserved=True))
assert suffixes[0] == suffixes[1]
result = dict(utc=subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(),
              successful=True, numpy_free=True, no_setup=True, no_prediction=True, cases=cases,
              script_sha256=sha(__file__))
with a.out.open('x') as f:
    json.dump(result, f, indent=2)
    f.write('\n')
print(json.dumps(dict(successful=True, numpy_free=True)))
