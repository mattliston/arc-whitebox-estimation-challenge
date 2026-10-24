"""Two-value literal build for one fully sealed matched P/NODE arm."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
COMP = NEW/'current_p1_node_composition_v1'
PARENT = COMP/'build_v1/terminal_estimator.py'
PARENT_SHA = '2e13d40a21e62691155918719f8ec088f581d495ae17800c5503c3b56504918a'
MODEL_SHA = 'cb49d7d598a76e430f388e6f3dc84b6b79d89eaf8a0e12cd3c19f8caca61cc86'
SCHEMA = 'current_comp_joint_p_node_build_v1'
TRAIN_SCHEMA = 'current_comp_joint_p_node_training_v1'
TRAIN = tuple(r for r in range(35, 59) if r not in (37, 39, 42, 48))
VALID = (37, 39, 42, 48)
PKEYS = ('msg_w', 'msg_b', 'enc0_w', 'enc0_b', 'enc2_w', 'enc2_b',
         'enc4_w', 'enc4_b', 'head', 'head_b', 'gain')
INACTIVE = ('head', 'head_b', 'gain')
PINS = {
    PARENT: PARENT_SHA,
    HERE/'model.py': MODEL_SHA,
    HERE/'PROTOCOL.md': '330f744e26f5672748ce94fc20f6e0f617358f83883d6a6c0c4b25c185d11830',
    HERE/'qualification.py': '6d161c22229016f524eeea28313af996de309e43e7585c5f101fe7ce82218e04',
    HERE/'QUALIFICATION_PROTOCOL.md': 'd5bd03fed976479e74580c7f2d0f7f2727c23b629225a94799af9f9cc4f0460e',
    HERE/'watch_qualification.py': '096102b0f1fe3f4618af79c035d3d283d9a98e00281ca487b372741f1f36e6f4',
    HERE/'WATCH_SPEC.md': 'dce270b1177217dab42602332337bba340523c3a5a6c160926f02d70c99c1890',
    HERE/'qualification_v1/result.json': '25d1beb88adf81264705c15b287b676f1bb15ba8f24b0515bad58eeff7d04469',
    HERE/'qualification_v1/completion.json': '542378dc2b7cc51ad17a078a1e67d0cb0b415df713c61fe965ce8fdd17c1d721',
    HERE/'qualification_v1/finalization.json': '857cb560c1c93170726ee8f1b943ab754607e3dcc3686570d1ad1090e802e084',
    HERE/'qualification_watch_v1/result.json': '9413e57e3f7c7f011f577db6f480bced6b7aeb01ba8406ebfc062ead42eda493',
    HERE/'qualification_watch_v1/completion.json': 'c2fad0df75545d29417b1b356b772363b45d01503ff21c25e93de1c85be68cd4',
    HERE/'qualification_watch_v1/finalization.json': 'f9db978814559ad1c3675ccbfd2fec8d3da337bb8bda3b4e290a9bef3648dda0',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    path = Path(path).resolve()
    require('truthpost' not in str(path), 'Builder never hashes or reads target arrays')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def value_span(source, name):
    """AST columns are UTF8 byte offsets; surrounding source is untouched."""
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)
             and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name]
    require(len(nodes) == 1, 'Exactly one module literal owner: '+name)
    node = nodes[0].value; lines = source.splitlines(keepends=True)
    start = sum(map(len, lines[:node.lineno-1]))+node.col_offset
    end = sum(map(len, lines[:node.end_lineno-1]))+node.end_col_offset
    return start, end, source[start:end], ast.literal_eval(node)


def replace_values(source, replacements):
    edits = [(value_span(source, key), value) for key, value in replacements.items()]
    for (start, end, _, _), replacement in sorted(edits, key=lambda item: item[0][0], reverse=True):
        source = source[:start]+replacement+source[end:]
    ast.parse(source)
    return source


def inverse(candidate, parent):
    original = {name: value_span(parent, name)[2] for name in ('_GNN_LITERAL', '_BDNH_A_LITERAL')}
    require(replace_values(candidate, original) == parent, 'Only the two exact literal VALUE spans may change')
    require(value_span(candidate, '_BD_COEFFICIENT_LITERAL')[2]
        == value_span(parent, '_BD_COEFFICIENT_LITERAL')[2], 'Original cstar source byte exact')


def chain(folder, digest, pin, *, worker, limit):
    rp, ep, fp = (folder/name for name in ('result.json', 'completion.json', 'finalization.json'))
    pin(rp, digest); end_sha, final_sha = sha(ep), sha(fp)
    pin(ep, end_sha); pin(fp, final_sha)
    record, end, final = read(rp), read(ep), read(fp)
    require(record['passed'] and end['passed'] and final['passed'] and end['result_sha256'] == digest
        and final['completion_sha256'] == end_sha, 'Passed three-seal chain: '+str(folder))
    if worker:
        require(record['completed'] and end['completed'] and final['completed']
            and final['result_sha256'] == digest and final['elapsed_s'] < limit, 'Complete worker finalization')
    else:
        require(record['whole_owned_tree_retired'] and record['cleanup_passed'] and record['sources_unchanged']
            and record['elapsed_s'] < limit and record['maximum_interval_s'] <= 1
            and record['maximum_combined_rss_bytes'] < 12_000_000_000
            and end['elapsed_through_result_hash_s'] < limit
            and final['elapsed_through_completion_io_s'] < limit, 'Independent resource and cleanup seal')
    return record, end_sha, final_sha


def admission(args, record, guard):
    def pin(path, digest):
        path = Path(path).resolve()
        require(path.is_relative_to(P2) and re.fullmatch('[0-9a-f]{64}', digest or '') and sha(path) == digest,
                'Exact source/non-target binding: '+str(path))
        group = 'source_bindings' if path.suffix in ('.py', '.md') else 'input_bindings'
        prior = record[group].get(str(path)); require(prior is None or prior == digest, 'Conflicting source/input binding')
        record[group][str(path)] = digest; guard()
    def inventory(item):
        for group in ('source_bindings', 'input_bindings', 'weight_bindings', 'artifacts'):
            for path, digest in item.get(group, {}).items():
                pin(path, digest)
    for path, digest in {**PINS, Path(__file__).resolve(): args.source_sha256,
            HERE/'BUILD_PROTOCOL.md': args.protocol_sha256, HERE/'train.py': args.training_source_sha256,
            HERE/'TRAIN_PROTOCOL.md': args.training_protocol_sha256,
            HERE/'watch_train.py': args.training_watcher_sha256}.items():
        pin(path, digest)
    q, qe, qf = chain(HERE/'qualification_v1', PINS[HERE/'qualification_v1/result.json'], pin, worker=True, limit=180)
    qw, _, _ = chain(HERE/'qualification_watch_v1', PINS[HERE/'qualification_watch_v1/result.json'], pin, worker=False, limit=180)
    require(q['execution_integrity_passed'] and q['hashes_unchanged'] and q['sources_unchanged']
        and q['full_model_calls'] == q['retired_call_count'] == 5 and not q['native_gradient_qualified']
        and not q['training_targets_accessed'] and not q['validation_targets_accessed']
        and qw['retired_calls'] == 5 and qw['child_result_sha256'] == PINS[HERE/'qualification_v1/result.json']
        and qw['child_completion_sha256'] == qe and qw['child_finalization_sha256'] == qf, 'Exact five-call qualification')
    for item in (q, qw):
        require(item['source_bindings'][str(HERE/'model.py')] == MODEL_SHA, 'Same qualified joint model')
        inventory(item)
    folder = HERE/('train_'+args.arm+'_v1')
    train, te, tf = chain(folder, args.training_result_sha256, pin, worker=True, limit=600)
    watch, _, _ = chain(HERE/('train_'+args.arm+'_watch_v1'), args.training_watch_sha256, pin, worker=False, limit=600)
    require(train['schema'] == TRAIN_SCHEMA and train['arm'] == args.arm
        and train['hashes_unchanged'] and train['sources_unchanged'] and train['execution_integrity_passed']
        and train['full_model_calls'] == train['retired_call_count'] == 64 and train['training_graphs'] == 60
        and train['validation_graphs'] == 4 and train['epochs'] == 3 and train['updates'] == 15
        and train['all_graphs_retired'] and not train['validation_targets_accessed']
        and not train['scientific_score_computed'] and not train['checkpoint_selection']
        and train['no_native_gradient_claim'] and not train['native_gradient_qualified']
        and watch['retired_calls'] == 64 and watch['child_result_sha256'] == args.training_result_sha256
        and watch['child_completion_sha256'] == te and watch['child_finalization_sha256'] == tf, 'Complete matched-arm training and watch')
    reviewed = {str(HERE/'train.py'): args.training_source_sha256, str(HERE/'model.py'): MODEL_SHA,
        str(HERE/'TRAIN_PROTOCOL.md'): args.training_protocol_sha256, str(HERE/'watch_train.py'): args.training_watcher_sha256}
    for item in (train, watch):
        require(all(item['source_bindings'].get(path) == digest for path, digest in reviewed.items()),
                'Reviewed training source authority is not inferred from receipts')
        inventory(item)
    prerequisites = dict(qualification_result_sha256=PINS[HERE/'qualification_v1/result.json'],
                         qualification_watch_sha256=PINS[HERE/'qualification_watch_v1/result.json'])
    require(train['prerequisites'] == prerequisites
        and watch['source_bindings'][str(HERE/'qualification_v1/result.json')] == prerequisites['qualification_result_sha256']
        and watch['source_bindings'][str(HERE/'qualification_watch_v1/result.json')] == prerequisites['qualification_watch_sha256'],
        'Same exact completed qualification in both arm owners')
    def artifact(item, name):
        path = folder/name
        require(item['path'] == str(path) and train['artifacts'][str(path)] == item['sha256'], 'Owned sealed artifact: '+name)
        pin(path, item['sha256']); return path
    roster = read(artifact(train['model_call_roster'], 'model_call_roster.json'))
    require(roster['schema'] == TRAIN_SCHEMA and roster['arm'] == args.arm and roster['seed'] == train['seed'] == 9309093
        and roster['generator'] == 'PCG64DXSM' and len(roster['orders']) == 3
        and all(len(order) == 20 and set(order) == set(TRAIN) for order in roster['orders'])
        and train['train_ranks'] == list(TRAIN) and train['validation_ranks'] == list(VALID), 'Fixed complete train/VALID frame')
    expected = [dict(graph_index=i, rank=rank, kind='train_backward', update=i//4, epoch=i//20+1)
        for i, rank in enumerate(r for order in roster['orders'] for r in order)]
    expected += [dict(graph_index=60+i, rank=rank, kind='validation_forward', update=None, epoch=None) for i, rank in enumerate(VALID)]
    require(roster['entries'] == expected and len(train['graph_records']) == 64 and len(train['update_records']) == 15,
            'Exact64 graph/15 update roster')
    terminal = read(artifact(train['terminal_seal'], 'terminal_seal.json'))
    validation = read(artifact(train['validation_seal'], 'validation_predictions_sealed.json'))
    require(terminal['schema'] == validation['schema'] == TRAIN_SCHEMA and terminal['arm'] == validation['arm'] == args.arm
        and terminal['terminal_epoch'] == 3 and terminal['updates'] == 15 and terminal['training_graphs'] == 60
        and not terminal['validation_predictions_started'] and not terminal['validation_targets_accessed']
        and terminal['terminal_payload'] == train['terminal_payload'] and terminal['terminal_A'] == train['terminal_A']
        and terminal['source_bindings'] == train['source_bindings'] and terminal['frozen_nonP_and_inactive66_byte_exact']
        and terminal['P_only_A_byte_exact'] == (args.arm == 'p_only')
        and validation['terminal_seal_sha256'] == train['terminal_seal']['sha256'] and validation['all_four_complete']
        and not validation['validation_targets_accessed'] and validation['outputs'] == train['validation_outputs']
        and set(validation['outputs']) == {str(r) for r in VALID}, 'Sole terminal and all-four pretruth prediction seals')
    artifact(train['terminal_payload'], 'terminal_P.npz'); artifact(train['terminal_A'], 'terminal_A.npy')
    for index, item in enumerate(train['graph_records']):
        graph = read(artifact(item, f'graph_{index:04d}_complete.json'))
        require(all(graph[key] == value for key, value in expected[index].items())
            and graph['passed'] and graph['completed'] and graph['graph_retired']
            and graph['all16_domains_present'] and graph['all47_selectors_present'] and graph['frozen_cstar']
            and graph['parameter_bytes_unchanged_during_call'] and graph['observed_checkpoint_replay_byte_exact'],
            'Every actual graph completed and retired')
        artifact(graph['prediction'], f'graph_{index:04d}_served.npy')
        if index >= 60:
            require(graph['prediction'] == validation['outputs'][str(expected[index]['rank'])], 'Sealed final four prediction ownership')
    for index, item in enumerate(train['update_records'], 1):
        update = read(artifact(item, f'update_{index:02d}.json'))
        require(update['update'] == index and update['ranks'] == [x['rank'] for x in expected[4*(index-1):4*index]],
                'All15 original updates completed in roster order')
    record.update(arm=args.arm, training_result_sha256=args.training_result_sha256,
        training_watch_sha256=args.training_watch_sha256, prerequisites=prerequisites,
        terminal_payload=train['terminal_payload'], terminal_A=train['terminal_A'], terminal_seal=train['terminal_seal'],
        validation_seal=train['validation_seal'], validation_outputs=train['validation_outputs'])
    return train, pin


def array_meta(value):
    return dict(dtype=str(value.dtype), shape=list(value.shape), raw_sha256=hashlib.sha256(value.tobytes()).hexdigest())


def verify_build(arm, result_sha, candidate_sha, source_sha, protocol_sha):
    """Strictly stdlib: hashes and lexical proof only, never array decoding."""
    require(arm in ('p_only', 'joint') and sha(__file__) == source_sha
        and sha(HERE/'BUILD_PROTOCOL.md') == protocol_sha, 'Reviewed builder source/protocol')
    folder = HERE/('build_'+arm+'_v1'); pins = {}
    def pin(path, digest):
        require(sha(path) == digest, 'Complete build binding: '+str(path)); pins[str(Path(path).resolve())] = digest
    result, _, _ = chain(folder, result_sha, pin, worker=True, limit=60)
    require(result['schema'] == SCHEMA and result['arm'] == arm and result['hashes_unchanged'] and result['sources_unchanged']
        and result['source_bindings'][str(Path(__file__).resolve())] == source_sha
        and result['source_bindings'][str(HERE/'BUILD_PROTOCOL.md')] == protocol_sha
        and all(result['proof'][key] for key in ('literal_inverse_byte_exact', 'literal_roundtrip_byte_exact',
            'frozen_nonP_and_inactive66_byte_exact', 'cstar_source_byte_exact', 'full_shape_FP32_finite')),
        'Completed source-only candidate build')
    for group in ('source_bindings', 'input_bindings', 'artifacts'):
        for path, digest in result[group].items():
            pin(path, digest)
    candidate = result['candidate']; parent = result['parent']
    require(candidate == dict(path=str(folder/'estimator.py'), sha256=candidate_sha)
        and parent == dict(path=str(PARENT), sha256=PARENT_SHA), 'Exact per-arm candidate and COMP parent')
    pin(candidate['path'], candidate_sha); pin(PARENT, PARENT_SHA)
    inverse(Path(candidate['path']).read_bytes(), PARENT.read_bytes())
    return dict(build=result, candidate=candidate, parent=parent, source_bindings=dict(result['source_bindings']),
        input_bindings={**result['input_bindings'], **pins})


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--build', action='store_true')
    parser.add_argument('--arm', choices=('p_only', 'joint'))
    for key in ('source', 'protocol', 'training-source', 'training-protocol', 'training-watcher', 'training-result', 'training-watch'):
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.build:
        parser.print_help(); return 0
    require(args.arm in ('p_only', 'joint'), 'Explicit one-arm build')
    began = time.monotonic()
    def expired(*_):
        raise TimeoutError('Fixed60s literal build')
    signal.signal(signal.SIGALRM, expired); signal.alarm(60)
    folder = HERE/('build_'+args.arm+'_v1'); folder.mkdir(exist_ok=False)
    record = dict(schema=SCHEMA, arm=args.arm, passed=False, completed=False, source_bindings={}, input_bindings={}, artifacts={},
        estimator_executed=False, truth_arrays_accessed=False, refit_performed=False, native_gradient_qualified=False,
        native_score_measured=False, parameter_updates=0, hashes_unchanged=False, sources_unchanged=False)
    def guard():
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != 'darwin':
            rss *= 1024
        require(time.monotonic()-began < 60 and rss < 2_000_000_000, 'Complete60s/2GB build bound')
        return rss
    try:
        require(sys.flags.optimize == 0 and os.getpriority(os.PRIO_PROCESS, 0) >= 10
            and sys.executable == '/Users/matt/anaconda3/bin/python'
            and all(os.environ.get(k) == '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')), 'AP/nice10/BLAS1 build')
        train, pin = admission(args, record, guard)
        require(not any(k in sys.modules for k in ('numpy', 'torch', 'flopscope', 'threadpoolctl')), 'Metadata admission before NumPy only')
        import numpy as np
        require(np.__version__ == '1.26.4', 'Original FP32 literal conversion runtime')
        parent = PARENT.read_bytes()
        p0 = value_span(parent, '_GNN_LITERAL')[3]; a0 = value_span(parent, '_BDNH_A_LITERAL')[3]
        original = {key: np.asarray(value, dtype=np.float32) for key, value in p0.items()}
        original_A = np.asarray(a0, dtype=np.float32)
        require(train['initial_P'] == {key: array_meta(value) for key, value in original.items()}
            and train['initial_A'] == array_meta(original_A), 'Training began from these exact COMP literal bytes')
        pm, am = train['terminal_payload'], train['terminal_A']
        with np.load(pm['path'], allow_pickle=False) as archive:
            terminal = {key: archive[key].copy() for key in archive.files}
        A = np.load(am['path'], allow_pickle=False)
        require(set(terminal) == set(original) == set(pm['arrays']) and A.dtype == np.float32
            and A.shape == original_A.shape == (64, 4) and np.isfinite(A).all()
            and array_meta(A) == {k: am[k] for k in ('dtype', 'shape', 'raw_sha256')}, 'Complete finite terminal A and P inventory')
        for key, value in terminal.items():
            require(value.dtype == np.float32 and value.shape == original[key].shape and np.isfinite(value).all()
                and array_meta(value) == pm['arrays'][key], 'Terminal P shape/bytes: '+key)
            if key not in PKEYS:
                require(value.tobytes() == original[key].tobytes(), 'Frozen non-P value: '+key)
            if key in INACTIVE:
                require(value[0].tobytes() == original[key][0].tobytes(), 'Inactive66 bytes: '+key)
        require(sum(terminal[k].size for k in PKEYS) == 45200 and sum(terminal[k][0].size for k in INACTIVE) == 66,
                'Same complete P architecture')
        if args.arm == 'p_only':
            require(A.tobytes() == original_A.tobytes(), 'P-only retains exact original NODE A')
        cstar = np.asarray(value_span(parent, '_BD_COEFFICIENT_LITERAL')[3], dtype=np.float32)
        seal = read(train['terminal_seal']['path'])
        require(cstar.shape == (15, 4) and train['cstar_raw_sha256'] == seal['cstar_raw_sha256']
            == array_meta(cstar)['raw_sha256'], 'Untouched original cstar bytes')
        for rank in VALID:
            item = train['validation_outputs'][str(rank)]; value = np.load(item['path'], allow_pickle=False)
            require(value.dtype == np.float32 and value.shape == (16, 1024) and np.isfinite(value).all()
                and (value >= 0).all() and array_meta(value) == {k: item[k] for k in ('dtype', 'shape', 'raw_sha256')},
                'Actual sealed target-blind terminal prediction')
        replacements = {'_GNN_LITERAL': repr({key: terminal[key].tolist() for key in p0}).encode(),
                        '_BDNH_A_LITERAL': repr(A.tolist()).encode()}
        candidate = replace_values(parent, replacements); inverse(candidate, parent)
        restored_P, restored_A = value_span(candidate, '_GNN_LITERAL')[3], value_span(candidate, '_BDNH_A_LITERAL')[3]
        require(set(restored_P) == set(terminal) and all(np.asarray(restored_P[key], dtype=np.float32).tobytes()
            == terminal[key].tobytes() for key in terminal)
            and np.asarray(restored_A, dtype=np.float32).tobytes() == A.tobytes(), 'Every terminal literal byte round-trips')
        for group in ('source_bindings', 'input_bindings'):
            for path, digest in record[group].items():
                require(sha(path) == digest, 'Prewrite fixed binding'); guard()
        target = folder/'estimator.py'
        with target.open('xb') as stream:
            stream.write(candidate)
        digest = sha(target); record['artifacts'][str(target)] = digest
        record['candidate'] = dict(path=str(target), sha256=digest)
        record['parent'] = dict(path=str(PARENT), sha256=PARENT_SHA)
        record['proof'] = dict(literal_inverse_byte_exact=True, literal_roundtrip_byte_exact=True,
            frozen_nonP_and_inactive66_byte_exact=True, cstar_source_byte_exact=True,
            full_shape_FP32_finite=True, P_only_A_byte_exact=args.arm == 'p_only',
            changed_value_names=['_GNN_LITERAL', '_BDNH_A_LITERAL'])
        record.update(passed=True, completed=True, hashes_unchanged=True, sources_unchanged=True, peak_rss_bytes=guard())
    except BaseException:
        record.update(passed=False, completed=False, error=traceback.format_exc())
    record['elapsed_s'] = time.monotonic()-began
    write(folder/'result.json', record); digest = sha(folder/'result.json')
    write(folder/'completion.json', dict(passed=record['passed'], completed=record['completed'], result_sha256=digest))
    end = sha(folder/'completion.json'); elapsed = time.monotonic()-began; ok = record['passed'] and elapsed < 60
    write(folder/'finalization.json', dict(passed=ok, completed=ok, result_sha256=digest, completion_sha256=end, elapsed_s=elapsed))
    signal.alarm(0)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
