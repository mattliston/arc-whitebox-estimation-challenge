from endpoint_common import *
endpoint_verify()
fit_success()
fit = read(F / 'fit_result.json')
assert fit['completed'] and fit['passed'] and fit['model_fitted']
assert fit['training_graphs'] == 160 and fit['updates'] == 40 and fit['epoch'] == 1
assert fit['fixed_parameters_unchanged'] and fit['inactive_parameters_unchanged']
assert not fit['validation_targets_accessed'] and not fit['checkpoint_selection']
assert not fit['native_score_measured'] and not fit['candidate_created']
check(fit['terminal_payload'])
check(fit['training_order'])
check(fit['source_seal'])
assert fit['source_seal'] == pin(F / 'sources_sealed.json')
order = read(fit['training_order']['path'])['ranks']
assert [r['rank'] for r in fit['records']] == order
assert [r['index'] for r in fit['records']] == list(range(160))
for row in fit['records']:
    check(row['record'])
    graph = read(row['record']['path'])
    assert graph['card_retired'] and graph['fixed_parameters_unchanged'] and not graph['validation_targets_accessed']
    check(graph['prediction'])
for i, update in enumerate(fit['update_receipts']):
    check(update)
    r = read(update['path'])
    assert r['update'] == i + 1 and r['ranks'] == order[4 * i:4 * i + 4]
import numpy as np
_, _, initial_literal = literal_span(ZERO.read_text())
bank_values = decode(np, initial_literal)
with np.load(fit['terminal_payload']['path'], allow_pickle=False) as z:
    trained = {k: z[k].copy() for k in z.files}
assert set(trained) == set(bank_values)
assert all(trained[k].shape == bank_values[k].shape and trained[k].dtype == bank_values[k].dtype for k in trained)
assert all(np.isfinite(v).all() for v in trained.values())
assert all(trained[k].tobytes() == bank_values[k].tobytes() for k in trained if k not in PKEYS)
assert all(trained[k][0].tobytes() == bank_values[k][0].tobytes() for k in INACTIVE)
assert sum(trained[k].size for k in PKEYS) == 51344
changed = [k for k in PKEYS if trained[k].tobytes() != bank_values[k].tobytes()]
assert changed == fit['changed_trainable_keys']
assert 'ungated_w' in changed
source = BANK.read_text()
start, end, literal = literal_span(source)
assert exact(decode(np, literal), bank_values)
with (HERE / 'bank_literal.txt').open('x') as f:
    f.write(literal)
exports = {}
for name, values in (('null', bank_values), ('trained', trained)):
    encoded = serialize(np, values)
    assert exact(decode(np, encoded), values)
    changed_source = source[:start] + encoded + source[end:]
    a, b, actual_literal = literal_span(changed_source)
    assert actual_literal == encoded and changed_source[:a] + literal + changed_source[b:] == source
    ast.parse(changed_source)
    path = HERE / ('null_candidate.py' if name == 'null' else 'candidate.py')
    with path.open('x') as f:
        f.write(changed_source)
    with (HERE / f'{name}_literal.txt').open('x') as f:
        f.write(encoded)
    exports[name] = dict(source=pin(path), literal=pin(HERE / f'{name}_literal.txt'),
        decoded_arrays_byte_exact=True, inverse_source_byte_exact=True)
write(HERE / 'build.json', dict(completed=True, passed=True, candidate=exports['trained']['source'],
    null_candidate=exports['null']['source'], exports=exports, parent=pin(PARENT), initial_source=pin(ZERO),
    bank_literal=pin(HERE / 'bank_literal.txt'), fit=pin(F / 'fit_result.json'),
    fit_resources=pin(F / 'fit_resource_summary.json'), terminal_payload=fit['terminal_payload'],
    only_gnn_literal_changed=True, fixed_arrays_byte_exact=True, inactive_coordinates_byte_exact=True,
    allocated_trainable_parameters=51344, active_trainable_parameters=51278,
    changed_trainable_keys=changed, validation_targets_accessed=False))
print(dict(completed=True, candidate=pin(HERE / 'candidate.py'), changed_keys=changed), flush=True)
