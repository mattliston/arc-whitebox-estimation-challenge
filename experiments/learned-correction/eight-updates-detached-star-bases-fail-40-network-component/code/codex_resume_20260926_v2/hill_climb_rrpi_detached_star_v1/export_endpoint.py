"""Export only fixed update8 after both2 clean guards; root executes after training."""
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
assert len(sys.argv) == 2 and len(sys.argv[1]) == 64
PROTOCOL_SHA = sys.argv[1]

def pin(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=path.stat().st_size)

protocol_path = HERE / 'protocol.json'
assert len(sys.argv) == 2 and sys.argv[1] == PROTOCOL_SHA
protocol_pin = pin(protocol_path)
assert protocol_pin['sha256'] == PROTOCOL_SHA
protocol = json.loads(protocol_path.read_text())
assert protocol['root_admitted'] is True
assert protocol['updates'] == 8 and protocol['batch'] == 4 and protocol['schedule_steps'] == 120
assert protocol['segment_limits'] == dict(updates=4, seconds=600, rss_bytes=7500000000,
                                        added_swap_bytes=0, threads=1, serial_only=True)
for record in protocol['pins']:
    assert pin(record['path']) == record, record['path']
source_review = json.loads((HERE / 'source_review.json').read_text())
assert source_review['passed'] is True and source_review['protocol'] == protocol_pin
export_review = json.loads((HERE / 'source_review_exporter.json').read_text())
assert export_review['passed'] is True and export_review['protocol'] == protocol_pin
assert export_review['exporter'] == pin(__file__)
out = HERE / 'candidate.py'
assert not out.exists() and not (HERE / 'build.json').exists()
assert not (HERE / 'export_manifest.json').exists()
# All completion/resource checks occur BEFORE importing NumPy or opening endpoint arrays.
done_path = HERE / 'training_complete.json'
done = json.loads(done_path.read_text())
assert done['completed'] is True and done['updates'] == 8 and done['selected_endpoint'] == 8
assert done['protocol'] == protocol_pin and done['validation_read'] is False
assert len(done['segments']) == 2
status = json.loads((HERE / 'controller_status.json').read_text())
assert status['state'] == 'completed' and status['completed_updates'] == 8
trainer_pin = pin(HERE / 'train_segment.py')
guard_pins, completion_pins, checkpoint_pins, recorded_batches = [], [], [], []
previous_checkpoint = None
ranks = protocol['training_ranks']
assert len(ranks) == len(set(ranks)) == 160
for segment in range(2):
    guard_path = HERE / ('segment_%02d_guard.json' % segment)
    segment_path = HERE / ('segment_%02d' % segment) / 'completed.json'
    guard_pin = pin(guard_path)
    assert done['segments'][segment] == guard_pin
    guard = json.loads(guard_path.read_text())
    assert guard['completed'] is True and guard['exit_code'] == 0 and guard['error'] is None
    assert guard['whole_group_retired'] is True and guard['remaining_rows'] == []
    assert guard['command'] == [protocol['torch_runtime'], '-B',
                                str(HERE / 'train_segment.py'), PROTOCOL_SHA, str(segment)]
    assert 0 < guard['elapsed_s'] <= 600
    assert 0 < guard['max_rss_bytes'] < guard['rss_trigger_bytes'] == 7500000000
    assert guard['max_swap_growth_bytes'] == 0
    completed = json.loads(segment_path.read_text())
    assert completed['completed'] is True and completed['start'] == 4 * segment
    assert completed['optimizer_updates'] == 4 * (segment + 1)
    assert completed['protocol'] == protocol_pin and completed['trainer'] == trainer_pin
    assert completed['baseline'] == protocol['baseline']
    assert completed['training_ranks'] == ranks
    assert completed['resumed_from'] == previous_checkpoint
    checkpoint = HERE / ('resume_after_%03d.pt' % (4 * (segment + 1)))
    assert completed['checkpoint'] == pin(checkpoint)
    previous_checkpoint = completed['checkpoint']
    batches = completed['segment_ranks']
    assert len(batches) == 4 and all(len(batch) == 4 for batch in batches)
    assert all(rank in ranks for batch in batches for rank in batch)
    recorded_batches.extend(batches)
    assert completed['endpoint_selected'] is (segment == 1)
    if segment < 1:
        assert completed['endpoint'] is None and completed['dials'] is None
    else:
        assert completed['endpoint'] == done['endpoint'] and completed['dials'] == done['dials']
    guard_pins.append(guard_pin)
    completion_pins.append(pin(segment_path))
    checkpoint_pins.append(completed['checkpoint'])
for key, name in [('endpoint', 'endpoint.npz'), ('dials', 'endpoint_dials.json')]:
    assert done[key]['path'] == str(HERE / name)
    assert done[key] == pin(HERE / name)
assert len(recorded_batches) == 8
assert len({rank for batch in recorded_batches for rank in batch}) == 32
sys.path.insert(0, str(HERE))
from native_parameters import SOURCE, SOURCE_SHA, FROZEN_P, LITERALS, load_s33_arrays, _source_values, _nest
assert pin(SOURCE) == protocol['baseline'] and SOURCE_SHA == protocol['baseline']['sha256']
assert FROZEN_P == ('centers', 'scales', 'rms', 'gain', 'first')
import numpy as np

rng = np.random.default_rng(0)
expected_order = rng.permutation(ranks).tolist()[:32]
assert recorded_batches == [expected_order[i:i+4] for i in range(0, 32, 4)]
base, base_dials = load_s33_arrays(np)
with np.load(done['endpoint']['path'], allow_pickle=False) as saved:
    arrays = {key: saved[key].copy() for key in saved.files}
assert set(arrays) == set(base)
for key, value in arrays.items():
    assert value.dtype == np.float32 and value.shape == base[key].shape and np.isfinite(value).all(), key
for key in FROZEN_P:
    assert arrays['p.' + key].tobytes() == base['p.' + key].tobytes()
# Detached-star8 follows original full-group training: no head15 or coefficient-column freeze.
dials = json.loads(Path(done['dials']['path']).read_text())
assert set(dials) == set(base_dials) and all(type(value) in (int, float) and math.isfinite(value) for value in dials.values())
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
assert len(allowed) == 11 and set(LITERALS).issubset(allowed)
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
assert ''.join(undo).encode() == SOURCE.read_bytes()
compile(candidate, str(out), 'exec')
with out.open('x') as stream:
    stream.write(candidate)
record = dict(passed=True, candidate=pin(out), baseline=pin(SOURCE), parent=pin(SOURCE),
              exporter=pin(__file__), protocol=pin(protocol_path),
              training_completion=pin(done_path), training_guards=guard_pins,
              segment_completions=completion_pins, resumption_checkpoints=checkpoint_pins,
              training_guards_checked=2, training_complete8_verified=True,
              source_review=pin(HERE / 'source_review.json'),
              exporter_source_review=pin(HERE / 'source_review_exporter.json'),
              endpoint=done['endpoint'], dials=done['dials'], changed_statements=changed,
              only_five_learned_and_six_dial_assignments=True, unrelated_statements_ast_exact=True,
              frozen_parameter_bytes_exact=True, k31_table_keys_unchanged=True,
              learned_arrays=28, optimizer_updates=8, selected_endpoint=8,
              head_and_coefficient_columns_permitted_to_train=True,
              frozen_groups=list(FROZEN_P), frozen_dials=['X3', 'c1'],
              native_executed=False,
              source_inverse_exact=True, serving_qualified=False, submitted=False)
with (HERE / 'build.json').open('x') as stream:
    json.dump(record, stream, indent=2, allow_nan=False)
    stream.write('\n')
with (HERE / 'export_manifest.json').open('x') as stream:
    json.dump(record, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps(dict(passed=True, candidate=record['candidate'],
                     export_manifest=pin(HERE / 'export_manifest.json'),
                     changed_statements=changed)), flush=True)
