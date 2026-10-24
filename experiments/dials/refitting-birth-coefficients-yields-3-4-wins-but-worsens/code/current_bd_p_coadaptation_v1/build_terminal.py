"""Build only the sealed epoch6 same-shaped P replacement of d50.

No card, Torch, training or target imports/execution. The matched null is the
exact parent source. Array decoding is limited to original and terminal P.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
PARENT = NEW/'birth_direction_fit_v1/build_terminal_v1/estimator.py'
PARENT_SHA = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'
ORIGINAL = NEW/'payload_response_rows_v1.npz'
ORIGINAL_SHA = '6b55c0d6071b2350e53e243bcd4137651e2c80a2cd55d04467bf194d0ea54a1c'
DATA = NEW/'joint_prefix_final_data_v1.py'
DATA_SHA = '22fff8465a13e412c68c1e5c204d7bfa0ce3d7c1cb9b99e1e5378098a9fd8b6b'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unchanged(pins):
    for path, digest in pins.items():
        if sha(path) != digest:
            raise RuntimeError('Authenticated builder input changed: ' + path)


def write(path, record):
    with Path(path).open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def definitions():
    assert sha(DATA) == DATA_SHA
    tree = ast.parse(DATA.read_bytes())
    def literal(name):
        nodes = [n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
        assert len(nodes) == 1
        return ast.literal_eval(nodes[0].value)
    valid, keys = literal('VALID'), literal('PKEYS')
    train = tuple(r for r in range(35, 235) if r not in valid)
    assert len(valid) == 40 and len(train) == 160 and len(set(valid)) == 40
    assert len(keys) == len(set(keys)) == 11
    return train, valid, keys


def completed(folder, expected):
    folder = Path(folder)
    rp, cp = folder/'result.json', folder/'completion.json'
    assert sha(rp) == expected
    record, seal = json.loads(rp.read_text()), json.loads(cp.read_text())
    assert record['passed'] and record['completed'] and seal['passed'] and seal['completed']
    assert seal['result_sha256'] == expected
    unchanged(record.get('source_bindings', {}))
    unchanged(record.get('artifacts', {}))
    return record


def training_admission(digest):
    record = completed(HERE/'training_v1', digest)
    assert record['schema'] == 'current_d50_p_coadaptation_training_v1'
    assert record['seed'] == 9309096 and record['epochs'] == 6 and record['updates'] == 240
    assert record['batch_nets'] == 4 and record['allocated_parameters'] == 45200
    assert record['active_parameters'] == 45134 and not record['validation_targets_accessed']
    assert not record['Q_used'] and record['birth_coefficients_frozen']
    assert record['full_autonomous_birth_gradients'] and not record['checkpoint_selection']
    assert record['unused_layer0_parameters_unchanged'] and record['hashes_unchanged']
    assert record['source_bindings'][str(HERE/'train.py')] == sha(HERE/'train.py')
    train, valid, keys = definitions()
    assert record['train_ranks'] == list(train) and record['validation_ranks'] == list(valid)
    assert record['trainable_keys'] == list(keys)
    watched = completed(HERE/'watch_train_v1', sha(HERE/'watch_train_v1/result.json'))
    assert watched['child_result_sha256'] == digest
    terminal = record['terminal_payload']
    assert Path(terminal['path']).resolve() == HERE/'training_v1/terminal_payload.npz'
    assert sha(terminal['path']) == terminal['sha256'] == record['artifacts'][terminal['path']]
    sp, pp = HERE/'training_v1/terminal_seal.json', HERE/'training_v1/validation_predictions_sealed.json'
    assert sha(sp) == record['terminal_seal_sha256']
    assert sha(pp) == record['validation_predictions_sealed_sha256']
    seal, predictions = json.loads(sp.read_text()), json.loads(pp.read_text())
    assert seal['terminal_epoch'] == 6 and seal['updates'] == 240
    assert seal['terminal_payload'] == terminal and seal['cstar_raw_sha256'] == record['cstar_raw_sha256']
    assert not seal['validation_targets_accessed'] and not seal['checkpoint_selection']
    assert predictions['terminal_payload'] == terminal and predictions['all_40_complete']
    assert predictions['terminal_seal_sha256'] == record['terminal_seal_sha256']
    assert not predictions['validation_targets_accessed'] and not predictions['native_score_measured']
    assert predictions['validation_outputs'] == record['validation_outputs']
    assert set(record['validation_outputs']) == {str(r) for r in valid}
    for rank in valid:
        item = record['validation_outputs'][str(rank)]
        assert Path(item['path']).resolve() == HERE/'training_v1'/f'validation_{rank:04d}.npy'
        assert sha(item['path']) == item['sha256'] == record['artifacts'][item['path']]
    assert [c['epoch'] for c in record['audit_checkpoints']] == list(range(1, 7))
    assert all(not c['checkpoint_selection'] for c in record['audit_checkpoints'])
    assert record['audit_checkpoints'][-1]['arrays'] == terminal['arrays']
    pins = dict(record['source_bindings'])
    for path in (HERE/'training_v1/result.json', HERE/'training_v1/completion.json', sp, pp,
                 Path(terminal['path']), HERE/'watch_train_v1/result.json', HERE/'watch_train_v1/completion.json'):
        pins[str(path)] = sha(path)
    return record, pins


def assignment_span(source, name='_GNN_LITERAL'):
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(nodes) == 1 and len(nodes[0].targets) == 1
    node = nodes[0]
    lines = source.splitlines(keepends=True)
    start = sum(map(len, lines[:node.lineno-1])) + node.col_offset
    stop = sum(map(len, lines[:node.end_lineno-1])) + node.end_col_offset
    return start, stop, ast.literal_eval(node.value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--source-sha256')
    parser.add_argument('--training-sha256')
    args = parser.parse_args()
    if not args.build:
        parser.print_help()
        return 0
    assert sha(__file__) == args.source_sha256
    assert sha(PARENT) == PARENT_SHA and sha(ORIGINAL) == ORIGINAL_SHA
    training, pins = training_admission(args.training_sha256)
    pins.update({str(Path(__file__).resolve()): args.source_sha256, str(PARENT): PARENT_SHA,
                 str(ORIGINAL): ORIGINAL_SHA, str(DATA): DATA_SHA})
    out = HERE/'build_terminal_v1'
    assert not out.exists(), 'Exclusive terminal build; no overwrite or checkpoint choice'
    import numpy as np
    with np.load(ORIGINAL, allow_pickle=False) as source:
        original = {k: source[k].copy() for k in source.files}
    with np.load(training['terminal_payload']['path'], allow_pickle=False) as source:
        terminal = {k: source[k].copy() for k in source.files}
    parent = PARENT.read_bytes()
    start, stop, literal = assignment_span(parent)
    _, _, keys = definitions()
    assert set(original) == set(terminal) == set(literal)
    assert sum(original[k].size for k in keys) == 45200
    for key, old in original.items():
        new = terminal[key]
        assert old.dtype == new.dtype and old.shape == new.shape
        assert np.isfinite(old).all() and np.isfinite(new).all()
        decoded = np.asarray(literal[key], dtype=old.dtype)
        assert decoded.shape == old.shape and decoded.tobytes() == old.tobytes(), key
        assert training['terminal_payload']['arrays'][key] == dict(shape=list(new.shape), dtype=str(new.dtype),
            raw_sha256=hashlib.sha256(new.tobytes()).hexdigest())
        if key not in keys:
            assert old.tobytes() == new.tobytes(), key
    for key in ('head', 'head_b', 'gain'):
        assert terminal[key][0].tobytes() == original[key][0].tobytes(), key
    replacement = ('_GNN_LITERAL = ' + repr({key: terminal[key].tolist() for key in literal})).encode()
    candidate = parent[:start]+replacement+parent[stop:]
    cstart, cstop, replay = assignment_span(candidate)
    assert cstart == start and candidate[:cstart]+parent[start:stop]+candidate[cstop:] == parent
    for key, value in terminal.items():
        decoded = np.asarray(replay[key], dtype=value.dtype)
        assert decoded.shape == value.shape and decoded.tobytes() == value.tobytes(), key
    pbd, cbd = assignment_span(parent, '_BD_COEFFICIENT_LITERAL'), assignment_span(candidate, '_BD_COEFFICIENT_LITERAL')
    assert parent[pbd[0]:pbd[1]] == candidate[cbd[0]:cbd[1]]
    assert hashlib.sha256(np.asarray(pbd[2], dtype=np.float32).tobytes()).hexdigest() == training['cstar_raw_sha256']
    unchanged(pins)
    receipt = dict(schema='current_d50_p_coadaptation_terminal_build_v1', passed=True,
        parent_sha256=PARENT_SHA, null_sha256=PARENT_SHA, candidate_sha256=hashlib.sha256(candidate).hexdigest(),
        source_bindings=pins, training_result_sha256=args.training_sha256,
        terminal_payload=training['terminal_payload'], terminal_epoch=6, updates=240,
        null_source_byte_exact=True, inverse_source_byte_exact=True,
        all_other_parent_source_byte_exact=True, modified_assignment_only='_GNN_LITERAL',
        original_literal_matches_original_payload=True, terminal_literal_roundtrip_byte_exact=True,
        non_PKEYS_byte_exact=True, unused_layer0_byte_exact=True, cstar_byte_exact=True,
        all_40_validation_predictions_sealed=True, validation_targets_accessed=False,
        estimator_imported=False, numerical_network_executed=False, complete_cost_measured=False,
        created_utc=datetime.now(timezone.utc).isoformat())
    out.mkdir()
    for name, data in (('estimator.py', candidate), ('null_estimator.py', parent)):
        with (out/name).open('xb') as stream:
            stream.write(data)
    assert sha(out/'estimator.py') == receipt['candidate_sha256']
    assert sha(out/'null_estimator.py') == PARENT_SHA
    write(out/'build.json', receipt)
    print(json.dumps(dict(candidate_sha256=receipt['candidate_sha256'],
                          build_sha256=sha(out/'build.json'))), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
