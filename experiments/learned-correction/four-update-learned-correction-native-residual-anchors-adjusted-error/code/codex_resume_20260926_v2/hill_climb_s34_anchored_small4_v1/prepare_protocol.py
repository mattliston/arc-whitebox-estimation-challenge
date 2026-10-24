"""Bind completed FIT16 native controls into an inactive training protocol; stdlib only."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
H = HERE.parent
P2 = H.parent
W = H / 's33_two_update_warm_proposal_v1'
RANKS = [38, 40, 41, 43, 44, 46, 47, 49, 50, 51, 53, 54, 55, 56, 57, 60]

def pin(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=path.stat().st_size)

admission_path = HERE / 'native_baseline_root_admission.json'
assert pin(admission_path)['sha256'] == '7004b867a50a7dcb339cbec5781bac9ba0d6dab64d4ba3f212ced0064b73a6a6'
admission = json.loads(admission_path.read_text())
assert admission['root_admitted'] is True and admission['ranks'] == RANKS
for key in ('baseline', 'bank', 'driver'):
    assert admission[key] == pin(admission[key]['path'])
assert admission['baseline']['sha256'] == '42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
native = H / admission['tag']
guard_path = Path(admission['guard_path'])
guard = json.loads(guard_path.read_text())
assert guard['completed'] is True and guard['exit_code'] == 0 and guard['error'] is None
assert guard['whole_group_retired'] is True and guard['remaining_rows'] == []
assert 0 < guard['elapsed_s'] <= 600
assert 0 < guard['max_rss_bytes'] < guard['rss_trigger_bytes'] == 7_500_000_000
assert guard['max_swap_growth_bytes'] == 0
manifest = json.loads((native / 'manifest.json').read_text())
assert manifest['candidate'] == admission['baseline']['path']
assert manifest['sha256'] == admission['baseline']['sha256'] and manifest['ranks'] == RANKS
assert all(value == '1' for value in manifest['thread_env'].values())
assert guard['command'] == [manifest['runtime'], '-B', admission['driver']['path'],
    admission['baseline']['path'], ','.join(map(str, RANKS)), admission['tag']]
rows = [json.loads(line) for line in (native / 'rows.jsonl').read_text().splitlines()]
assert [row['rank'] for row in rows] == RANKS
old = json.loads((W / 'protocol.json').read_text())
parts_path = P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json'
parts = json.loads(parts_path.read_text())
assert set(RANKS) <= set(parts['train']) and not set(RANKS) & set(parts['valid'])
assert not set(RANKS) & {35, 100, 36, 45, 52, 58, 64, 70}
pins = [record for record in old['pins']
        if str(H / 's33_two_update_warm_proposal_v1') not in record['path']
        and str(H / 's33_joint_forward_qualification_v1') not in record['path']
        and 'hill_climb_precision_g3_symdiag128_v1/candidate.py' not in record['path']]
for record in pins:
    assert pin(record['path']) == record
pins += [pin(path) for path in (
    HERE / 'train.py', HERE / 'export.py', HERE / 'native_parameters.py',
    HERE / 'offline_guard.py', HERE / 'prepare_protocol.py', HERE / 'PROTOCOL.md',
    admission_path, admission['baseline']['path'], admission['bank']['path'],
    admission['driver']['path'], guard_path, native / 'manifest.json', native / 'rows.jsonl')]
data = []
for rank in RANKS:
    weights = P2 / ('learned_residual_observer_design/v4_feature_store_v3/'
        f'sealed_target_blind_feature_store_v3/rows/observer_fit/{rank:04d}/weights.npy')
    truth = P2 / f'fable_refit_v1/truth688/truthpost_rank{rank:04d}.npz'
    data.append(dict(rank=rank, weights=pin(weights), truth=pin(truth),
                     native_prediction=pin(native / f'pred_{rank}.npy')))
optimizer = dict(old['optimizer'])
optimizer.pop('trunk_lr')
optimizer.pop('trunk_warmup')
protocol = dict(schema='s34-anchored-small4-v1', root_admitted=False,
    baseline=admission['baseline'], official_bank=admission['bank'],
    training_ranks=RANKS, passes=1, updates=4, batch=4, schedule_steps=120,
    torch_runtime=old['torch_runtime'], optimizer=optimizer,
    trainable_main_scalars=3437, trainable_dials=10, frozen_trunk_scalars=44144,
    frozen_normalizers=list(('centers', 'scales', 'rms', 'gain', 'first')),
    frozen_head_rows=[0, 15], frozen_coefficient_columns=[1, 3], frozen_dials=['X3', 'c1'],
    anchor='truth - S34_native_initial + Torch_initial',
    all_16_torch_initial_anchors_before_update_zero=True,
    endpoint_count=1, checkpoint_or_coefficient_selection=False,
    native_gradient_certified=False,
    anchor_limitation='Corrects only initial residual mismatch; Torch/native Jacobians remain uncertified.',
    prior_closed_experiments=['s33_two_update_warm_proposal_v1', 'same_layer_edge_messages_v1'],
    remaining_structural_gaps=old['remaining_structural_gaps'] + [
        'Torch does not include S34 young-negative trace or D5 boundary-side structural changes.'],
    resource_limits=dict(seconds=900, rss_bytes=9_500_000_000, added_swap_bytes=0,
                         threads=1, serial_only=True),
    native_anchor_limits=admission['limits'], native_serving_rss_max=7_500_000_000,
    native2_gates=old['native2_gates'], native8_gates=old['native8_gates'],
    native2_roster=[35, 100], native6_roster=[36, 45, 52, 58, 64, 70],
    native_anchor=dict(manifest=pin(native / 'manifest.json'), rows=pin(native / 'rows.jsonl'),
                       guard=pin(guard_path), root_admission=pin(admission_path)),
    no_validation_targets=True, no_numerical_execution_by_preparer=True,
    serving_qualified=False, submission_authorized=False, pins=pins, data=data)
with (HERE / 'protocol_draft.json').open('x') as stream:
    json.dump(protocol, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps(dict(protocol=pin(HERE / 'protocol_draft.json'), root_admitted=False)))
