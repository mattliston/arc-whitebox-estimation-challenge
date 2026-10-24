"""Payload-only terminal cap builder; inert without --build.

No estimator/Torch import, network execution or truth access. Read only sealed
training metadata and payload arrays, change one AST assignment, prove the
inverse, and exclusively create build_terminal_v1/{estimator.py,build.json}.
The helper admission functions are JSON/hash-only for native40.py reuse.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from common import HERE, NEW, bindings, completed, sha, unchanged, write

CAP_SHA = 'd6dc6bb47d0c1f047ae34c1fb37ff624ccb957e20074447725abe7b7c21fcbf4'
TRAINER_SHA = '6780e3e8f9f7ed3e6fddd0c347f6024f756972cf29bf3987e3622b08529cd81c'
PARITY_SHA = '10b0cde3244a7e94cfb324ad6ae86fa01a755347b768b09815d642674ef46eeb'
NATIVE_SHA = 'fc542f5604fe2d17afc6d83fa2f3d16ff37f9cfc3b21afa8d1d144b6ac511841'
PAYLOAD_SHA = '6b55c0d6071b2350e53e243bcd4137651e2c80a2cd55d04467bf194d0ea54a1c'
DATA_SHA = '22fff8465a13e412c68c1e5c204d7bfa0ce3d7c1cb9b99e1e5378098a9fd8b6b'
PROTOCOL_SHA = '851c15aae9c0343af232145646bb8084f82618818fe2dc8e10d014ca6903de58'


def definitions():
    path = NEW/'joint_prefix_final_data_v1.py'
    assert sha(path) == DATA_SHA
    tree = ast.parse(path.read_bytes())
    def literal(name):
        found = [n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
        assert len(found) == 1
        return ast.literal_eval(found[0].value)
    valid, keys = literal('VALID'), literal('PKEYS')
    train = tuple(r for r in range(35, 235) if r not in valid)
    assert len(valid) == 40 and len(train) == 160 and len(set(valid)) == 40
    assert len(keys) == len(set(keys)) == 11
    return train, valid, keys


def current_readiness():
    """Current null readiness only; deliberately independent of training_v1."""
    pins = bindings((__file__, HERE/'common.py', HERE/'estimator.py', HERE/'build.json',
                     HERE/'PROTOCOL.md', NEW/'joint_prefix_final_data_v1.py'))
    assert sha(HERE/'estimator.py') == CAP_SHA
    assert sha(HERE/'PROTOCOL.md') == PROTOCOL_SHA
    assert sha(NEW/'joint_prefix_final_data_v1.py') == DATA_SHA
    ready = completed(HERE/'parity_v4', PARITY_SHA)
    assert ready['schema'] == 'current_paircap4096_e2e_wide_v4'
    assert ready['fidelity_and_cost_passed'] and not ready['training_executed']
    assert ready['native_result_sha256'] == NATIVE_SHA
    native = completed(HERE/'native', NATIVE_SHA)
    assert native['candidate_sha256'] == CAP_SHA and native['cost_gate_passed']
    assert not native['truth_accessed']
    assert [t['rank'] for t in native['trials']] == [35, 100, 35]
    assert all(t['flops'] <= 360000000000 for t in native['trials'][:2])
    assert native['trials'][0]['output_bytes_sha256'] == native['trials'][2]['output_bytes_sha256']
    old_build = json.loads((HERE/'build.json').read_text())
    assert old_build['passed'] and old_build['inverse_source_byte_exact']
    assert old_build['candidate_sha256'] == CAP_SHA and old_build['all16_P_unchanged']
    pins.update(ready['source_bindings']); pins.update(native['source_bindings'])
    for directory in ('parity_v4', 'native'):
        for name in ('result.json', 'completion.json'):
            pins[str(HERE/directory/name)] = sha(HERE/directory/name)
    unchanged(pins)
    return pins


def training_admission(digest):
    """No array decode and no traversal of recorded target-file bindings."""
    assert sha(HERE/'train.py') == TRAINER_SHA
    record = completed(HERE/'training_v1', digest)
    assert record['schema'] == 'current_paircap4096_e2e_training_v1'
    assert record['completed'] and record['epochs'] == 6 and record['updates'] == 240
    assert record['seed'] == 9309092 and record['batch_nets'] == 4
    assert record['allocated_parameters'] == 45200 and record['structurally_active_parameters'] == 45134
    assert not record['validation_targets_accessed'] and not record['checkpoint_selection']
    assert record['unused_layer0_parameters_unchanged'] and record['hashes_unchanged']
    assert record['parity_result_sha256'] == PARITY_SHA and record['native_result_sha256'] == NATIVE_SHA
    assert record['source_bindings'][str(HERE/'train.py')] == TRAINER_SHA
    train, valid, keys = definitions()
    assert record['train_ranks'] == list(train) and record['validation_ranks'] == list(valid)
    assert record['trainable_keys'] == list(keys)
    assert not record['Q_used'] and not record['birth_intervention_used']
    terminal = record['terminal_payload']
    assert Path(terminal['path']).resolve() == HERE/'training_v1/terminal_payload.npz'
    assert sha(terminal['path']) == terminal['sha256']
    assert record['artifacts'][terminal['path']] == terminal['sha256']
    seal_path = HERE/'training_v1/terminal_seal.json'
    predictions_path = HERE/'training_v1/validation_predictions_sealed.json'
    assert sha(seal_path) == record['terminal_seal_sha256']
    assert sha(predictions_path) == record['validation_predictions_sealed_sha256']
    seal, predictions = json.loads(seal_path.read_text()), json.loads(predictions_path.read_text())
    assert seal['terminal_epoch'] == 6 and seal['updates'] == 240
    assert seal['terminal_payload'] == terminal and not seal['validation_targets_accessed']
    assert not seal['checkpoint_selection']
    assert predictions['terminal_payload'] == terminal and predictions['all_40_complete']
    assert predictions['terminal_seal_sha256'] == record['terminal_seal_sha256']
    assert not predictions['validation_targets_accessed'] and not predictions['native_score_measured']
    assert predictions['validation_outputs'] == record['validation_outputs']
    assert set(record['validation_outputs']) == {str(r) for r in valid}
    for rank in valid:
        item = record['validation_outputs'][str(rank)]
        assert Path(item['path']).resolve() == HERE/'training_v1'/f'validation_{rank:04d}.npy'
        assert sha(item['path']) == item['sha256'] == record['artifacts'][item['path']]
    checkpoints = record['audit_checkpoints']
    assert [c['epoch'] for c in checkpoints] == list(range(1, 7))
    assert all(not c['checkpoint_selection'] for c in checkpoints)
    assert checkpoints[-1]['arrays'] == terminal['arrays'], 'Terminal is epoch6, not a selected checkpoint'
    pins = dict(record['source_bindings'])
    for path in (HERE/'training_v1/result.json', HERE/'training_v1/completion.json',
                 seal_path, predictions_path, Path(terminal['path'])):
        pins[str(path)] = sha(path)
    return record, pins


def assignment_span(source):
    """UTF8 byte offsets, not character offsets; no arbitrary source execution."""
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == '_GNN_LITERAL' for t in n.targets)]
    assert len(nodes) == 1 and len(nodes[0].targets) == 1
    node = nodes[0]
    lines = source.splitlines(keepends=True)
    start = sum(map(len, lines[:node.lineno-1]))+node.col_offset
    stop = sum(map(len, lines[:node.end_lineno-1]))+node.end_col_offset
    literal = ast.literal_eval(node.value)
    assert isinstance(literal, dict)
    return start, stop, literal


def inverse_check(parent, candidate):
    pstart, pstop, _ = assignment_span(parent)
    cstart, cstop, _ = assignment_span(candidate)
    assert pstart == cstart
    assert candidate[:cstart]+parent[pstart:pstop]+candidate[cstop:] == parent
    return dict(inverse_source_byte_exact=True, all_other_parent_source_byte_exact=True,
        modified_assignment_only='_GNN_LITERAL',
        old_assignment_sha256=hashlib.sha256(parent[pstart:pstop]).hexdigest(),
        new_assignment_sha256=hashlib.sha256(candidate[cstart:cstop]).hexdigest())


def build(training_digest, source_digest):
    assert sha(__file__) == source_digest, 'Reviewed builder SHA required'
    os.chdir(NEW)
    out = HERE/'build_terminal_v1'
    assert not out.exists(), 'Exclusive terminal build; no overwrite or checkpoint choice'
    pins = current_readiness()
    training, training_pins = training_admission(training_digest)
    pins.update(training_pins)
    original_path = NEW/'payload_response_rows_v1.npz'
    assert sha(original_path) == PAYLOAD_SHA
    pins[str(original_path)] = PAYLOAD_SHA
    import numpy as np
    with np.load(original_path, allow_pickle=False) as source:
        original = {k: source[k].copy() for k in source.files}
    with np.load(training['terminal_payload']['path'], allow_pickle=False) as source:
        terminal = {k: source[k].copy() for k in source.files}
    parent = (HERE/'estimator.py').read_bytes()
    start, stop, baseline_literal = assignment_span(parent)
    _, _, keys = definitions()
    assert set(original) == set(terminal) == set(baseline_literal)
    assert sum(original[k].size for k in keys) == 45200
    for key in original:
        old, new = original[key], terminal[key]
        assert old.dtype == new.dtype and old.shape == new.shape
        assert np.isfinite(old).all() and np.isfinite(new).all()
        decoded = np.asarray(baseline_literal[key], dtype=old.dtype)
        assert decoded.shape == old.shape and decoded.tobytes() == old.tobytes(), key
        meta = training['terminal_payload']['arrays'][key]
        assert meta == dict(shape=list(new.shape), dtype=str(new.dtype),
            raw_sha256=hashlib.sha256(new.tobytes()).hexdigest())
        if key not in keys:
            assert old.tobytes() == new.tobytes(), key
    for key in ('head', 'head_b', 'gain'):
        assert terminal[key][0].tobytes() == original[key][0].tobytes(), key
    replacement = ('_GNN_LITERAL = '+repr({k: terminal[k].tolist() for k in baseline_literal})).encode()
    candidate = parent[:start]+replacement+parent[stop:]
    proof = inverse_check(parent, candidate)
    _, _, replay = assignment_span(candidate)
    assert set(replay) == set(terminal)
    for key, value in terminal.items():
        decoded = np.asarray(replay[key], dtype=value.dtype)
        assert decoded.shape == value.shape and decoded.tobytes() == value.tobytes(), key
    unchanged(pins)
    receipt = dict(passed=True, schema='current_paircap4096_e2e_terminal_build_v1',
        source_bindings=pins, parent_sha256=CAP_SHA,
        candidate=str(out/'estimator.py'), candidate_sha256=hashlib.sha256(candidate).hexdigest(),
        training_result_sha256=training_digest, trainer_sha256=TRAINER_SHA,
        terminal_payload=training['terminal_payload'], terminal_epoch=6, updates=240,
        all_40_validation_predictions_sealed=True, validation_targets_accessed=False,
        original_literal_matches_original_payload=True, terminal_literal_roundtrip_byte_exact=True,
        non_PKEYS_byte_exact=True, unused_layer0_byte_exact=True,
        trainable_keys=list(keys), signed_pair_cap=4096, physical_column_cap=8192,
        estimator_imported=False, numerical_network_executed=False,
        complete_cost_measured=False, created_utc=datetime.now(timezone.utc).isoformat(), **proof)
    out.mkdir()
    with (out/'estimator.py').open('xb') as stream:
        stream.write(candidate)
    assert sha(out/'estimator.py') == receipt['candidate_sha256']
    write(out/'build.json', receipt)
    print(json.dumps(dict(candidate_sha256=receipt['candidate_sha256'],
        build_sha256=sha(out/'build.json'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--training-sha256')
    parser.add_argument('--source-sha256')
    args = parser.parse_args()
    if args.build:
        build(args.training_sha256, args.source_sha256)
    else:
        parser.print_help()
