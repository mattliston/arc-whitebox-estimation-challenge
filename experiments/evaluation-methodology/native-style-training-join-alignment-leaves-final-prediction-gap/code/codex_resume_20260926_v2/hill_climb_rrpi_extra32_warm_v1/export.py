"""Export the one completed warm endpoint onto exact RRPI; no native execution."""
from pathlib import Path
import ast
import base64
import copy
import hashlib
import json
import math
import sys
import zlib

HERE = Path(__file__).resolve().parent
H = HERE.parent
sys.path.insert(0, str(HERE))
from native_parameters import SOURCE, SOURCE_SHA, FROZEN_P, LITERALS, load_s33_arrays, _source_values, _nest

def pin(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))

protocol_path = HERE / 'protocol.json'
assert len(sys.argv) == 2 and pin(protocol_path)['sha256'] == sys.argv[1]
protocol = json.loads(protocol_path.read_text())
assert protocol['root_admitted'] is True
for record in protocol['pins']:
    assert pin(record['path']) == record, record['path']
out = HERE / 'candidate.py'
assert not out.exists() and not (HERE / 'build.json').exists()
done_path = HERE / 'training/completed.json'
done = json.loads(done_path.read_text())
assert done['completed'] is True and done['optimizer_updates'] == 8 and done['checkpoints_saved'] == 1
assert done['training_ranks'] == protocol['training_ranks'] == json.loads((HERE / 'data_roster.json').read_text())['training_ranks']
assert done['baseline'] == pin(SOURCE) == protocol['baseline'] and done['baseline']['sha256'] == SOURCE_SHA
assert done['trainer'] == pin(HERE / 'train.py') and done['protocol'] == pin(protocol_path)
for key in ('endpoint', 'dials'):
    assert done[key] == pin(done[key]['path'])
assert done['endpoint']['path'] == str(HERE / 'endpoint.npz')
assert done['dials']['path'] == str(HERE / 'endpoint_dials.json')
guard = json.loads((HERE / 'training_guard.json').read_text())
assert guard['completed'] is True and guard['exit_code'] == 0 and guard['error'] is None
assert guard['whole_group_retired'] is True and guard['remaining_rows'] == []
assert guard['command'] == [protocol['torch_runtime'], '-B', str(HERE / 'train.py'), sys.argv[1]]
assert 0 < guard['elapsed_s'] <= 1200 and 0 < guard['max_rss_bytes'] < guard['rss_trigger_bytes'] == 7500000000
assert guard['max_swap_growth_bytes'] == 0
import numpy as np

base, base_dials = load_s33_arrays(np)
with np.load(done['endpoint']['path'], allow_pickle=False) as saved:
    arrays = {key: saved[key].copy() for key in saved.files}
assert set(arrays) == set(base)
for key, value in arrays.items():
    assert value.dtype == np.float32 and value.shape == base[key].shape and np.isfinite(value).all(), key
for key in FROZEN_P:
    assert arrays['p.' + key].tobytes() == base['p.' + key].tobytes()
for key in ('p.head', 'p.head_b'):
    assert arrays[key][[0, 15]].tobytes() == base[key][[0, 15]].tobytes()
assert arrays['c'][:, [1, 3]].tobytes() == base['c'][:, [1, 3]].tobytes()
dials = json.loads(Path(done['dials']['path']).read_text())
assert set(dials) == set(base_dials) and all(math.isfinite(value) for value in dials.values())
assert all(dials[key] == base_dials[key] for key in ('X3', 'c1'))
literal, nodes = _source_values()
packed = {}
for key in literal['_GNN_LITERAL']:
    value = np.ascontiguousarray(arrays['p.' + key], dtype=np.float32)
    packed[key] = (tuple(value.shape), base64.b64encode(zlib.compress(value.tobytes())).decode())
replacements = dict(_GNN_LITERAL=packed,
    _BDNH_A_LITERAL=_nest(literal['_BDNH_A_LITERAL'], arrays['a'].tolist()),
    _BD_COEFFICIENT_LITERAL=_nest(literal['_BD_COEFFICIENT_LITERAL'], arrays['c'].tolist()),
    _EDGE_LITERAL={key: _nest(value, arrays['e.' + key].tolist()) for key, value in literal['_EDGE_LITERAL'].items()},
    _MESSAGE_EDGE_LITERAL={key: _nest(value, arrays['e.context_' + key].tolist()) for key, value in literal['_MESSAGE_EDGE_LITERAL'].items()},
    _BD_J1_PHYS=dials['X1'], _S19_BD_ADD=(dials['X0'], 0., dials['X2'], dials['X3']),
    _S19_BSCALE=dials['bscale'], _S19_LAM=(dials['s_carry'], dials['s_new']),
    _S19_K4BAND=(dials['k4a'], dials['k4b'], dials['k4c']))
source = SOURCE.read_text()
old_tree = ast.parse(source)
edits = {}
for name, value in replacements.items():
    node = nodes[name][-1]
    if name in LITERALS:
        assert len(nodes[name]) == 1
    edits[node.lineno] = (node.end_lineno, f'{name} = {value!r}')
table = copy.deepcopy(nodes['_K31R_TABLE'][-1])
assert isinstance(table.value, ast.DictComp)
original_generators = ast.dump(table.value.generators[0])
original_key = ast.dump(table.value.key)
table.value.value = ast.Tuple(elts=[ast.Constant(dials['c0']), ast.Constant(dials['c1'])], ctx=ast.Load())
edits[table.lineno] = (table.end_lineno, ast.unparse(table))
lines = source.splitlines(keepends=True)
for start, (end, text) in sorted(edits.items(), reverse=True):
    lines[start - 1:end] = [text + '\n']
candidate = ''.join(lines)
new_tree = ast.parse(candidate)
assert len(new_tree.body) == len(old_tree.body)
allowed = set(replacements) | {'_K31R_TABLE'}
changed = []
for old, new in zip(old_tree.body, new_tree.body):
    if ast.dump(old) != ast.dump(new):
        assert isinstance(old, ast.Assign) and isinstance(old.targets[0], ast.Name)
        name = old.targets[0].id
        assert name in allowed and old.lineno in edits and new.targets[0].id == name
        changed.append(name)
new_nodes = {}
for node in new_tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        new_nodes.setdefault(node.targets[0].id, []).append(node)
for key, value in replacements.items():
    assert ast.literal_eval(new_nodes[key][-1].value) == value
new_table = new_nodes['_K31R_TABLE'][-1].value
assert ast.dump(new_table.key) == original_key and ast.dump(new_table.generators[0]) == original_generators
assert ast.literal_eval(new_table.value) == (dials['c0'], dials['c1'])
# Undo the 11 replacements and require exact original source bytes.
old_lines = source.splitlines(keepends=True)
undo = candidate.splitlines(keepends=True)
for new, old in sorted(((new_nodes[name][-1], nodes[name][-1]) for name in allowed),
                       key=lambda pair: -pair[0].lineno):
    undo[new.lineno - 1:new.end_lineno] = old_lines[old.lineno - 1:old.end_lineno]
assert ''.join(undo) == source
with out.open('x') as stream:
    stream.write(candidate)
record = dict(candidate=pin(out), parent=pin(SOURCE), exporter=pin(__file__), protocol=pin(protocol_path),
              training_completion=pin(done_path), training_guard=pin(HERE / 'training_guard.json'),
              endpoint=done['endpoint'], dials=done['dials'], changed_statements=changed,
              only_five_learned_and_six_dial_assignments=True, unrelated_statements_ast_exact=True,
              frozen_parameter_bytes_exact=True, k31_table_keys_unchanged=True,
              learned_arrays=28, optimizer_updates=8, native_executed=False,
              source_inverse_exact=True, serving_qualified=False, submitted=False)
with (HERE / 'build.json').open('x') as stream:
    json.dump(record, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps(dict(candidate=record['candidate'], changed_statements=changed)), flush=True)
