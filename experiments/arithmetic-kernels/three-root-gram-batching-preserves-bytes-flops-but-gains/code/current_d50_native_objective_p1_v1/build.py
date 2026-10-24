"""Inert, literal-only build of the single completed one-pass proposal.

No training, estimator execution, Torch or target access. Numerical array
decoding is confined to the original and sole terminal parameter payloads.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
PARENT = NEW/'birth_direction_fit_v1/build_terminal_v1/estimator.py'
PARENT_SHA = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'
ORIGINAL = NEW/'payload_response_rows_v1.npz'
ORIGINAL_SHA = '6b55c0d6071b2350e53e243bcd4137651e2c80a2cd55d04467bf194d0ea54a1c'
HELPER = NEW/'current_bd_p_coadaptation_v1/build_terminal.py'
HELPER_SHA = '8b621d1713841c6271f048504205030ec7bc78df1842be4a3071e00ecf01d084'
PROTO_SHA = '2141e2d4f025522a7cbea55fbd5ff10a66c313feacb1b4bd49a13f75a4b9af35'
SCHEMA = 'current_d50_native_objective_p1_v1'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    path = Path(path).resolve()
    require('truthpost' not in path.parts and not
            (path.suffix in ('.npz', '.npy') and 'truth' in path.name.lower()),
            'Builder never opens target arrays')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def bind(mapping):
    for path, digest in mapping.items():
        require(sha(path) == digest, 'Immutable source/artifact: '+path)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def sealed(folder, expected):
    require(sha(folder/'result.json') == expected, 'Exact completed result')
    record, end = read(folder/'result.json'), read(folder/'completion.json')
    require(all(record.get(k) is True for k in
                ('passed', 'completed', 'hashes_unchanged', 'sources_unchanged'))
            and end['passed'] and end['completed'] and end['result_sha256'] == expected,
            'Completed successful immutable stage')
    for key in ('source_bindings', 'input_bindings', 'weight_bindings', 'artifacts'):
        bind(record.get(key, {}))
    return record


def admission(args):
    sources = {str(Path(__file__).resolve()): args.source_sha256,
               str(HERE/'PROTOCOL.md'): PROTO_SHA,
               str(HERE/'worker.py'): args.worker_sha256,
               str(HERE/'common.py'): args.common_sha256,
               str(HERE/'watch.py'): args.watcher_sha256,
               str(PARENT): PARENT_SHA, str(ORIGINAL): ORIGINAL_SHA,
               str(HELPER): HELPER_SHA}
    bind(sources)
    spec = importlib.util.spec_from_file_location('p1_literal_build_helpers', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    require(not any(k in sys.modules for k in ('numpy', 'torch', 'flopscope')),
            'Inert source admission')
    train, valid, keys = helper.definitions()
    record = sealed(HERE/'training_v1', args.training_sha256)
    require(record['schema'] == SCHEMA and record['stage'] == 'train'
            and record['epochs'] == 1 and record['updates'] == record['planned_updates'] == 40
            and record['graphs_completed'] == 200 and record['checkpoint_selection'] is False
            and record['validation_targets_accessed'] is False
            and record['no_native_gradient_claim'] is True and record['native_score_measured'] is False
            and record['unused_layer0_parameters_unchanged'] is True
            and record['allocated_parameters'] == 45200 and record['active_parameters'] == 45134
            and record['trainable_keys'] == list(keys), 'The one fixed terminal proposal, not a selected checkpoint')
    for path in (HERE/'PROTOCOL.md', HERE/'worker.py', HERE/'common.py', HERE/'watch.py'):
        require(record['source_bindings'][str(path)] == sources[str(path)], 'Executed reviewed proposal sources')
    watch = sealed(HERE/'training_watch_v1', args.training_watch_sha256)
    require(watch['schema'] == 'current_d50_native_objective_p1_watch_v1'
            and watch['stage'] == 'train' and watch['child_result_sha256'] == args.training_sha256
            and watch['child_completion_sha256'] == sha(HERE/'training_v1/completion.json')
            and watch['whole_owned_tree_retired'] and watch['child_execution_completed'],
            'Completed independent proposal resource supervision')
    for path in (HERE/'PROTOCOL.md', HERE/'worker.py', HERE/'common.py', HERE/'watch.py'):
        require(watch['source_bindings'][str(path)] == sources[str(path)], 'Watched reviewed proposal sources')
    terminal = record['terminal_payload']
    require(Path(terminal['path']).resolve() == HERE/'training_v1/terminal_payload.npz'
            and sha(terminal['path']) == terminal['sha256'] == record['artifacts'][terminal['path']],
            'Only the sole terminal parameter payload')
    sp, vp = HERE/'training_v1/terminal_seal.json', HERE/'training_v1/validation_predictions_sealed.json'
    require(sha(sp) == record['terminal_seal_sha256'] and
            sha(vp) == record['validation_predictions_sealed_sha256'], 'Payload and all40 prediction seals')
    seal, predictions = read(sp), read(vp)
    require(seal['terminal_payload'] == predictions['terminal_payload'] == terminal
            and not seal['validation_targets_accessed'] and not predictions['validation_targets_accessed']
            and predictions['validation_outputs'] == record['validation_outputs']
            and set(record['validation_outputs']) == {str(r) for r in valid},
            'All40 surrogate predictions sealed, not scored')
    for rank, item in record['validation_outputs'].items():
        graph_index = 160+list(valid).index(int(rank))
        require(Path(item['path']).resolve() == HERE/'training_v1'/f'graph_{graph_index:04d}_served.npy'
                and sha(item['path']) == item['sha256'] == record['artifacts'][item['path']],
                'Exact fixed prediction artifact')
    inputs = {}
    for folder in (HERE/'training_v1', HERE/'training_watch_v1'):
        for name in ('result.json', 'completion.json'):
            inputs[str(folder/name)] = sha(folder/name)
    inputs.update({str(sp): sha(sp), str(vp): sha(vp), terminal['path']: terminal['sha256']})
    return helper, record, keys, sources, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    for name in ('source', 'worker', 'common', 'watcher', 'training', 'training-watch'):
        parser.add_argument('--'+name+'-sha256')
    args = parser.parse_args()
    if not args.build:
        parser.print_help(); return 0
    require(sys.flags.optimize == 0 and all(isinstance(v, str) and len(v) == 64
            and set(v) <= set('0123456789abcdef') for k, v in vars(args).items() if k != 'build'),
            'Assertions and every reviewed digest are required')
    began = time.monotonic()
    def expired(_signum, _frame):
        raise TimeoutError('Fixed60s literal-only builder deadline')
    signal.signal(signal.SIGALRM, expired); signal.alarm(60)
    helper, training, keys, sources, inputs = admission(args)
    out = HERE/'build_v1'
    require(not out.exists(), 'Exclusive terminal build; no overwrite')
    import numpy as np
    require(np.__version__ == '2.4.6', 'Builder uses fixed OFFPY NumPy')
    with np.load(ORIGINAL, allow_pickle=False) as f:
        original = {k: f[k].copy() for k in f.files}
    with np.load(training['terminal_payload']['path'], allow_pickle=False) as f:
        terminal = {k: f[k].copy() for k in f.files}
    parent = PARENT.read_bytes()
    start, stop, literal = helper.assignment_span(parent)
    require(set(original) == set(terminal) == set(literal), 'Complete same-field payload')
    for key, old in original.items():
        new = terminal[key]
        require(old.dtype == new.dtype and old.shape == new.shape
                and np.isfinite(new).all() and np.isfinite(old).all(), 'Finite same-shaped parameter '+key)
        require(np.asarray(literal[key], dtype=old.dtype).tobytes() == old.tobytes(), 'Original deployed P bytes')
        metadata = dict(shape=list(new.shape), dtype=str(new.dtype),
                        raw_sha256=hashlib.sha256(new.tobytes()).hexdigest())
        require(metadata == training['terminal_payload']['arrays'][key], 'Sealed parameter bytes')
        require(key in keys or new.tobytes() == old.tobytes(), 'Frozen non-P field')
    for key in ('head', 'head_b', 'gain'):
        require(terminal[key][0].tobytes() == original[key][0].tobytes(), 'Frozen inactive layer0')
    replacement = ('_GNN_LITERAL = '+repr({k: terminal[k].tolist() for k in literal})).encode()
    candidate = parent[:start]+replacement+parent[stop:]
    a, b, replay = helper.assignment_span(candidate)
    require(candidate[:a]+parent[start:stop]+candidate[b:] == parent, 'Exact source inverse outside P literal')
    require(all(np.asarray(replay[k], dtype=v.dtype).tobytes() == v.tobytes() for k, v in terminal.items()),
            'Native terminal-literal roundtrip')
    pbd, cbd = (helper.assignment_span(x, '_BD_COEFFICIENT_LITERAL') for x in (parent, candidate))
    require(parent[pbd[0]:pbd[1]] == candidate[cbd[0]:cbd[1]] and
            hashlib.sha256(np.asarray(pbd[2], dtype=np.float32).tobytes()).hexdigest() ==
            training['cstar_raw_sha256'], 'Exact frozen cstar')
    bind(sources); bind(inputs)
    require(time.monotonic()-began < 60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2_000_000_000,
            'Literal-only60s/2GB resource gate')
    result = dict(schema=SCHEMA, stage='build', passed=True, completed=True,
        hashes_unchanged=True, sources_unchanged=True, source_bindings=sources, input_bindings=inputs,
        candidate_sha256=hashlib.sha256(candidate).hexdigest(), null_sha256=PARENT_SHA,
        parent_sha256=PARENT_SHA, training_result_sha256=args.training_sha256,
        training_watch_sha256=args.training_watch_sha256, terminal_payload=training['terminal_payload'],
        terminal_epoch=1, updates=40, modified_assignment_only='_GNN_LITERAL',
        inverse_source_byte_exact=True, null_source_byte_exact=True, non_PKEYS_byte_exact=True,
        unused_layer0_byte_exact=True, cstar_byte_exact=True, terminal_literal_roundtrip_byte_exact=True,
        validation_targets_accessed=False, estimator_imported=False, native_score_measured=False,
        no_native_gradient_claim=True, official_score=False, production_qualified=False,
        wall_s=time.monotonic()-began, peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        created_utc=datetime.now(timezone.utc).isoformat())
    out.mkdir(exist_ok=False)
    for name, data in (('estimator.py', candidate), ('null_estimator.py', parent)):
        with (out/name).open('xb') as f:
            f.write(data)
    result['artifacts'] = {str(out/name): sha(out/name) for name in ('estimator.py', 'null_estimator.py')}
    write(out/'result.json', result)
    write(out/'completion.json', dict(passed=True, completed=True, result_sha256=sha(out/'result.json')))
    signal.alarm(0)
    print(json.dumps({k: result[k] for k in ('passed', 'candidate_sha256', 'wall_s', 'peak_rss_bytes')}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
