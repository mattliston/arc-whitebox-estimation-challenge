"""Qualified zero or sole terminal P/Q literal; no estimator execution."""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import struct
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
JOINT = NEW/'current_comp_joint_p_node_v1'
RD = NEW/'current_comp_receiver_directions_v1'
PARENT = NEW/'current_p1_node_composition_v1/build_v1/terminal_estimator.py'
PARENT_SHA = '2e13d40a21e62691155918719f8ec088f581d495ae17800c5503c3b56504918a'
PORT = NEW/'port_joint_p_covariance_working_v1.py'
SCHEMA = 'current_comp_joint_p_covariance_build_v1'
TRAIN_SCHEMA = 'current_comp_joint_p_covariance_training_v1'
TRAIN = tuple(r for r in range(35, 59) if r not in (37, 39, 42, 48))
VALID = (37, 39, 42, 48)
PKEYS = ('msg_w', 'msg_b', 'enc0_w', 'enc0_b', 'enc2_w', 'enc2_b',
         'enc4_w', 'enc4_b', 'head', 'head_b', 'gain')
INACTIVE = ('head', 'head_b', 'gain')
COMMON_ARGS = ('source', 'protocol', 'model', 'model_protocol', 'qualification_source',
    'qualification_protocol', 'qualification_watcher', 'qualification_watch_spec',
    'qualification_result', 'qualification_watch')
TRAIN_ARGS = ('training_source', 'training_protocol', 'training_watcher', 'training_result', 'training_watch')
PINS = {
    PARENT: PARENT_SHA,
    PORT: 'bafcc96ebd7fdb1856cb4a73be6f6c975c58c206f0179b0fa55f2ad4e35ed50c',
    HERE/'fnp.py': '89be5cf89be6e4f479b13539502f5abbd1b78cd80b5936726f980986dd0af581',
    HERE/'DECISION.md': '814ac1f31121db92f3165ecfa0cb3253e8b66617e3e62d17316f2f288e71be98',
    JOINT/'build.py': '11e7f7588e692587c11480f6a09bd7cfb9492d02851381755821b7dbf01a9ed4',
    RD/'build.py': 'c844c0526e1d98d7138a8958c9a9501e9a794274222294cbc6e30dbf4073aa39',
    RD/'build_v2.py': '7d4c6ee2b5cf7ab9990569e579862a4c3648d00b36cec4d0e5d5e18ad0ebb89f',
    RD/'BUILD_PROTOCOL.md': '767e8f39000facef13a17da5636c8750229e9bdda1f4550ed8581471ad605ca5',
    RD/'BUILD_PROTOCOL_V2.md': 'db2304a1dde8d07407d513474718038eef90c4dfd6d3897eb0641b5763b01aaf',
}
_HELPER_PROOF = None


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    path = Path(path).resolve()
    require(path.is_relative_to(P2) and 'truthpost' not in path.parts, 'Only project-local non-target bindings')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    path = Path(path).resolve()
    require(path.is_relative_to(P2) and path.suffix == '.json' and 'truthpost' not in path.parts,
            'Only project-local metadata JSON')
    return json.loads(path.read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def helpers():
    """Original JOINT checks and the exact repaired RD literal parser only."""
    global _HELPER_PROOF
    for path in (JOINT/'build.py', RD/'build.py', RD/'build_v2.py'):
        require(sha(path) == PINS[path], 'Frozen literal/seal helper source')
    if _HELPER_PROOF is None:
        path = JOINT/'build.py'; source = path.read_text(); tree = ast.parse(source)
        names = ('value_span', 'replace_values', 'chain', 'array_meta')
        nodes = [node for name in names for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
        require(tuple(node.name for node in nodes) == names, 'Unique exact JOINT helper inventory')
        lines = source.splitlines(keepends=True)
        text = '\n\n'.join(''.join(lines[n.lineno-1:n.end_lineno]) for n in nodes)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path)+':covariance-build-checks', 'exec'), globals())
        begin = '        for key, value in terminal.items():\n'
        end = "        if args.arm == 'p_only':\n"
        require(source.count(begin) == source.count(end) == 1, 'Unique original strict P validation span')
        guards = source[source.index(begin):source.index(end)]
        wrapper = ('def validate_payload(np, terminal, original, metadata):\n'
                   '    pm = {"arrays": metadata}\n'
                   '    require(set(terminal) == set(original) == set(metadata), "Complete original P inventory")\n'
                   '    if True:\n'+guards)
        exec(compile(wrapper, str(path)+':unchanged-P-payload-guards', 'exec'), globals())
        owner = RD/'build.py'; old = owner.read_text(); rows = old.splitlines(keepends=True)
        chosen = [node for node in ast.parse(old).body if isinstance(node, ast.FunctionDef) and node.name == 'literal_metadata']
        require(len(chosen) == 1, 'Unique original RD exact-FP32 parser')
        node = chosen[0]; literal = ''.join(rows[node.lineno-1:node.end_lineno])
        before, after = 'isinstance(item, list)', 'isinstance(item, (list, tuple))'
        require(literal.count(before) == 1, 'Sole original list-container check')
        repaired = literal.replace(before, after)
        require(repaired.count(after) == 1 and repaired.replace(after, before) == literal,
                'Exact unchanged RD-v2 tuple-container repair')
        exec(compile(repaired, str(owner)+':reviewed-tuple-parser', 'exec'), globals())
        _HELPER_PROOF = dict(owner=str(path), owner_sha256=PINS[path], functions=list(names),
            extracted_source_sha256=hashlib.sha256(text.encode()).hexdigest(), unchanged_source=True,
            exact_P_guard_sha256=hashlib.sha256(guards.encode()).hexdigest(), P_guards_unchanged=True,
            literal_owner=str(RD/'build_v2.py'), literal_owner_sha256=PINS[RD/'build_v2.py'],
            literal_parser_sha256=hashlib.sha256(repaired.encode()).hexdigest(), tuple_repair_inverse_byte_exact=True)
    return dict(_HELPER_PROOF)


def composer():
    require(sha(PORT) == PINS[PORT] and sha(HERE/'fnp.py') == PINS[HERE/'fnp.py'], 'Pinned source-only port and FNP fragment')
    spec = importlib.util.spec_from_file_location('covariance_build_exact_port', PORT)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    require(value.PARENT == PARENT and value.PARENT_SHA == PARENT_SHA and value.EXPERIMENT == HERE,
            'Exact fixed COMP source and experiment destination')
    return value


def P_metadata(literal, original, *, unchanged):
    require(isinstance(literal, dict) and tuple(literal) == tuple(original), 'Exact ordered full P dictionary')
    metadata = {}
    for key, value in literal.items():
        meta, raw = literal_metadata(value); initial, initial_raw = literal_metadata(original[key])
        require(meta['dtype'] == initial['dtype'] and meta['shape'] == initial['shape'], 'Original P shape and precision')
        if unchanged or key not in PKEYS:
            require(raw == initial_raw, 'Exact frozen P value: '+key)
        if key in INACTIVE:
            require(literal_metadata(value[0])[1] == literal_metadata(original[key][0])[1], 'Exact inactive66 P bytes')
        metadata[key] = meta
    require(sum(math.prod(metadata[key]['shape']) for key in PKEYS) == 45200
        and sum(math.prod(literal_metadata(literal[key][0])[0]['shape']) for key in INACTIVE) == 66,
        'Original allocated P and inactive architecture')
    return metadata


def source_candidate(role, mean_literal, covariance_literal):
    helpers(); parent = PARENT.read_bytes(); original_P = value_span(parent, '_GNN_LITERAL')[3]
    Pmeta = P_metadata(mean_literal, original_P, unchanged=role == 'zero')
    Qmeta, Qraw = literal_metadata(covariance_literal)
    require(role in ('zero', 'terminal') and Qmeta['shape'] == [14, 65], 'Sole910-coordinate covariance literal')
    if role == 'zero':
        require(Qraw == bytes(3640), 'Exact positive FP32 zero covariance literal')
    port = composer()
    text, proof = port.compose(covariance_literal, None if role == 'zero' else mean_literal)
    candidate = text.encode('utf-8')
    zero_text, _ = port.compose([[0.0]*65 for _ in range(14)])
    zero = zero_text.encode('utf-8')
    require(replace_values(candidate, {name: value_span(zero, name)[2]
        for name in ('_GNN_LITERAL', '_PCC_LITERAL')}) == zero, 'Zero/terminal differ only at P and Q VALUE spans')
    restored_P = value_span(candidate, '_GNN_LITERAL')[3]
    restored_Q, restored_raw = literal_metadata(value_span(candidate, '_PCC_LITERAL')[3])
    require(P_metadata(restored_P, original_P, unchanged=role == 'zero') == Pmeta
        and restored_Q == Qmeta and restored_raw == Qraw, 'Every P/Q FP32 literal byte round-trips')
    for name in ('_BDNH_A_LITERAL', '_BD_COEFFICIENT_LITERAL'):
        require(value_span(candidate, name)[2] == value_span(parent, name)[2], 'Untouched COMP A/cstar VALUE source')
    if role == 'zero':
        require(value_span(candidate, '_GNN_LITERAL')[2] == value_span(parent, '_GNN_LITERAL')[2], 'Zero P source is original verbatim')
    require(proof['full_source_inverse_verified'] and proof['physical_function_inverse_verified']
        and proof['original_stored_pair_repair_once'] and proof['live_zero_covariance_action']
        and proof['original_NODE_A_cstar_literals_unchanged'] and proof['original_new_radial_unchanged'],
        'Whole parent inverse and original birth/radial ownership')
    return candidate, Pmeta, Qmeta, dict(proof, literal_roundtrip_byte_exact=True,
        zero_terminal_P_Q_literal_only=True, frozen_nonP_and_inactive66_byte_exact=True,
        zero_parameters_not_bypassed=proof['live_zero_covariance_action'],
        original_A_cstar_literals_unchanged=proof['original_NODE_A_cstar_literals_unchanged'],
        full_shape_FP32_finite=True, zero_source_sha256=hashlib.sha256(zero).hexdigest())


def admission(args, record, guard):
    def pin(path, digest):
        path = Path(path).resolve()
        require(re.fullmatch('[0-9a-f]{64}', digest or '') and sha(path) == digest, 'Exact non-target binding: '+str(path))
        group = 'source_bindings' if path.suffix in ('.py', '.md') else 'input_bindings'
        require(record[group].get(str(path), digest) == digest, 'No conflicting source/input pin')
        record[group][str(path)] = digest; guard()
    def inventory(item):
        targets = set(item.get('target_bindings', {}))
        for group in ('source_bindings', 'input_bindings', 'weight_bindings', 'artifacts'):
            for path, digest in item.get(group, {}).items():
                require(path not in targets, 'Target arrays never become builder inputs')
                pin(path, digest)
    owners = {
        HERE/'model.py': args.model_sha256, HERE/'MODEL_PROTOCOL.md': args.model_protocol_sha256,
        HERE/'qualification.py': args.qualification_source_sha256,
        HERE/'QUALIFICATION_PROTOCOL.md': args.qualification_protocol_sha256,
        HERE/'watch_qualification.py': args.qualification_watcher_sha256,
        HERE/'WATCH_SPEC.md': args.qualification_watch_spec_sha256,
    }
    for path, digest in {**PINS, **owners, Path(__file__).resolve(): args.source_sha256,
                         HERE/'BUILD_PROTOCOL.md': args.protocol_sha256}.items():
        pin(path, digest)
    record['helper_proof'] = helpers()
    q, qe, qf = chain(HERE/'qualification_v1', args.qualification_result_sha256, pin, worker=True, limit=180)
    qw, _, _ = chain(HERE/'qualification_watch_v1', args.qualification_watch_sha256, pin, worker=False, limit=180)
    require(q['schema'] == 'current_comp_joint_p_covariance_qualification_v1'
        and q['execution_integrity_passed'] and q['hashes_unchanged'] and q['sources_unchanged']
        and q['full_model_calls'] == q['retired_call_count'] == 5 and q['null_forwards'] == 2
        and q['joint_backwards'] == q['zero_lr_Adam_calls'] == 3
        and q['all_graphs_retired'] and q['all_parameter_owners_retired'] and q['repeat_gradient_byte_exact']
        and q['all16_domains_qualified'] and q['all47_selectors_qualified']
        and not q['native_gradient_qualified'] and not q['training_targets_accessed'] and not q['validation_targets_accessed']
        and q['tiny']['passed'] and q['tiny']['no_card_calls'] and q['tiny']['null_Q_gradient_connected']
        and q['tiny']['endogenous_H_gradient_connected'] and q['tiny']['live_C_gradient_connected']
        and q['tiny']['explicit_congruence_checked']
        and qw['retired_calls'] == 5 and qw['child_result_sha256'] == args.qualification_result_sha256
        and qw['child_completion_sha256'] == qe and qw['child_finalization_sha256'] == qf,
        'Complete fresh P/Q qualification, not a historical substitute')
    for item in (q, qw):
        require(all(item['source_bindings'].get(str(path)) == digest for path, digest in owners.items()),
                'Supplied reviewed qualification/model source authority')
        inventory(item)
    prerequisites = dict(qualification_result_sha256=args.qualification_result_sha256,
        qualification_watch_sha256=args.qualification_watch_sha256,
        qualification_source_sha256=args.qualification_source_sha256,
        qualification_protocol_sha256=args.qualification_protocol_sha256,
        qualification_watcher_sha256=args.qualification_watcher_sha256,
        qualification_watch_spec_sha256=args.qualification_watch_spec_sha256)
    record.update(prerequisites=prerequisites, qualification_owner_pins={str(p): h for p, h in owners.items()},
                  training_provenance_used=args.role == 'terminal')
    parent = PARENT.read_bytes(); P0 = value_span(parent, '_GNN_LITERAL')[3]
    initial_P = P_metadata(P0, P0, unchanged=True)
    initial_A = literal_metadata(value_span(parent, '_BDNH_A_LITERAL')[3])[0]
    cstar = literal_metadata(value_span(parent, '_BD_COEFFICIENT_LITERAL')[3])[0]
    require(initial_A['shape'] == [64, 4] and cstar['shape'] == [15, 4], 'Exact COMP stored dimensions')
    require([call['rank'] for call in q['calls']] == [35, 100, 35, 100, 35], 'Fixed current-COMP qualification roster')
    for call in q['calls']:
        frozen = call['frozen_hashes']
        require(all(frozen[key] == value['raw_sha256'] for key, value in initial_P.items())
            and frozen['NODE_A'] == initial_A['raw_sha256'] and frozen['cstar'] == cstar['raw_sha256'],
            'All qualification calls use these exact COMP parent literals')
    record.update(initial_P=initial_P, initial_A=initial_A, cstar_raw_sha256=cstar['raw_sha256'])
    if args.role == 'zero':
        return None, pin
    reviewed = {HERE/'train.py': args.training_source_sha256,
        HERE/'TRAIN_PROTOCOL.md': args.training_protocol_sha256, HERE/'watch_train.py': args.training_watcher_sha256}
    for path, digest in reviewed.items():
        pin(path, digest)
    train, te, tf = chain(HERE/'train_v1', args.training_result_sha256, pin, worker=True, limit=600)
    watch, _, _ = chain(HERE/'train_watch_v1', args.training_watch_sha256, pin, worker=False, limit=600)
    require(train['schema'] == TRAIN_SCHEMA and train['arm'] == 'p_covariance'
        and train['hashes_unchanged'] and train['sources_unchanged'] and train['execution_integrity_passed']
        and train['full_model_calls'] == train['retired_call_count'] == 64 and train['training_graphs'] == 60
        and train['validation_graphs'] == 4 and train['epochs'] == 3
        and train['updates'] == train['P_optimizer_steps'] == train['Q_optimizer_steps'] == 15
        and train['trainable_keys'] == list(PKEYS)+['covariance_Q'] and train['trainable_parameter_count'] == 46110
        and train['all_graphs_retired'] and train['frozen_A_cstar'] and train['frozen_nonP_and_inactive66_byte_exact']
        and not train['validation_targets_accessed'] and not train['scientific_score_computed']
        and not train['checkpoint_selection'] and train['no_native_gradient_claim'] and not train['native_gradient_qualified']
        and train['prerequisites'] == prerequisites
        and watch['retired_calls'] == 64 and watch['child_result_sha256'] == args.training_result_sha256
        and watch['child_completion_sha256'] == te and watch['child_finalization_sha256'] == tf,
        'One complete fixed60+4/15-update P/Q training and independent watch')
    for item in (train, watch):
        require(all(item['source_bindings'].get(str(path)) == digest for path, digest in {**owners, **reviewed}.items()),
                'Same reviewed model, qualification and trainer sources')
        inventory(item)
    require(train['initial_P'] == initial_P and train['initial_A'] == initial_A
        and train['cstar_raw_sha256'] == cstar['raw_sha256']
        and train['initial_Q'] == dict(dtype='float32', shape=[14, 65], raw_sha256=hashlib.sha256(bytes(3640)).hexdigest()),
        'Exact COMP P/A/cstar and positive-zero Q initialization')
    folder = HERE/'train_v1'
    def artifact(item, name):
        path = folder/name
        require(item['path'] == str(path) and train['artifacts'].get(str(path)) == item['sha256'], 'Exclusive sealed training artifact')
        pin(path, item['sha256']); return path
    roster = read(artifact(train['model_call_roster'], 'model_call_roster.json'))
    require(roster['schema'] == TRAIN_SCHEMA and roster['arm'] == 'p_covariance' and roster['seed'] == train['seed'] == 9309093
        and roster['generator'] == 'PCG64DXSM' and len(roster['orders']) == 3
        and all(len(order) == 20 and set(order) == set(TRAIN) for order in roster['orders'])
        and train['train_ranks'] == list(TRAIN) and train['validation_ranks'] == list(VALID), 'Fixed20/4 data and three-order frame')
    expected = [dict(graph_index=i, rank=rank, kind='train_backward', update=i//4, epoch=i//20+1)
        for i, rank in enumerate(rank for order in roster['orders'] for rank in order)]
    expected += [dict(graph_index=60+i, rank=rank, kind='validation_forward', update=None, epoch=None) for i, rank in enumerate(VALID)]
    require(roster['entries'] == expected and len(train['graph_records']) == 64 and len(train['update_records']) == 15,
            'Exact complete64/15 roster, never a partial endpoint')
    terminal = read(artifact(train['terminal_seal'], 'terminal_seal.json'))
    validation = read(artifact(train['validation_seal'], 'validation_predictions_sealed.json'))
    require(terminal['schema'] == validation['schema'] == TRAIN_SCHEMA
        and terminal['arm'] == validation['arm'] == 'p_covariance' and terminal['terminal_epoch'] == 3
        and terminal['updates'] == 15 and terminal['training_graphs'] == 60
        and not terminal['validation_predictions_started'] and not terminal['validation_targets_accessed']
        and terminal['terminal_payload'] == train['terminal_payload'] and terminal['terminal_Q'] == train['terminal_Q']
        and terminal['initial_A'] == initial_A and terminal['cstar_raw_sha256'] == cstar['raw_sha256']
        and terminal['source_bindings'] == train['source_bindings'] and terminal['frozen_A_cstar']
        and terminal['frozen_nonP_and_inactive66_byte_exact']
        and validation['terminal_seal_sha256'] == train['terminal_seal']['sha256'] and validation['all_four_complete']
        and not validation['validation_targets_accessed'] and validation['outputs'] == train['validation_outputs']
        and set(validation['outputs']) == {str(rank) for rank in VALID}, 'Sole terminal and all-four pretruth seals')
    artifact(train['terminal_payload'], 'terminal_P.npz'); artifact(train['terminal_Q'], 'terminal_Q.npy')
    for index, item in enumerate(train['graph_records']):
        graph = read(artifact(item, f'graph_{index:04d}_complete.json'))
        require(all(graph[key] == value for key, value in expected[index].items())
            and graph['passed'] and graph['completed'] and graph['graph_retired'] and graph['all16_domains_present']
            and graph['all47_selectors_present'] and graph['frozen_cstar']
            and graph['parameter_bytes_unchanged_during_call'] and graph['observed_checkpoint_replay_byte_exact']
            and set(graph['parameter_before']) == set(initial_P)
            and all(graph['parameter_before'][key] == value['raw_sha256'] for key, value in initial_P.items() if key not in PKEYS)
            and graph['A_before_sha256'] == initial_A['raw_sha256'], 'Every graph retired with exact frozen non-P/A bytes')
        artifact(graph['prediction'], f'graph_{index:04d}_served.npy')
        if index < 4:
            require(graph['parameter_before'] == {key: value['raw_sha256'] for key, value in initial_P.items()}
                and graph['Q_before_sha256'] == train['initial_Q']['raw_sha256'], 'First batch starts from exact COMP P/zero Q')
        if index >= 60:
            require(graph['prediction'] == validation['outputs'][str(expected[index]['rank'])]
                and graph['parameter_before'] == {key: value['raw_sha256'] for key, value in train['terminal_payload']['arrays'].items()}
                and graph['Q_before_sha256'] == train['terminal_Q']['raw_sha256'], 'Four predictions use sole sealed P/Q')
    for index, item in enumerate(train['update_records'], 1):
        update = read(artifact(item, f'update_{index:02d}.json'))
        cosine = .5*(1+math.cos(math.pi*(index-1)/14))
        require(update['update'] == index and update['ranks'] == [row['rank'] for row in expected[4*(index-1):4*index]]
            and update['learning_rates'] == [5e-5+(5e-4-5e-5)*cosine, .001+(.01-.001)*cosine]
            and len(update['pre_clip_gradient_norms']) == 2 and all(math.isfinite(x) for x in update['pre_clip_gradient_norms']),
            'All15 two-group updates in prescribed order/schedule')
    record.update(training_result_sha256=args.training_result_sha256, training_watch_sha256=args.training_watch_sha256,
        training_owner_pins={str(path): digest for path, digest in reviewed.items()}, terminal_payload=train['terminal_payload'],
        terminal_Q=train['terminal_Q'], terminal_seal=train['terminal_seal'], validation_seal=train['validation_seal'],
        validation_outputs=train['validation_outputs'])
    return train, pin


def verify_build(role, result_sha, candidate_sha, source_sha, protocol_sha):
    """Strict stdlib revalidation; no model, array decoder or target access."""
    require(role in ('zero', 'terminal') and sha(__file__) == source_sha
        and sha(HERE/'BUILD_PROTOCOL.md') == protocol_sha, 'Reviewed builder role/source/protocol')
    helpers(); folder = HERE/('build_'+role+'_v1'); pins = {}
    def pin(path, digest):
        require(re.fullmatch('[0-9a-f]{64}', digest or '') and sha(path) == digest, 'Build-owned source/input bytes')
        pins[str(Path(path).resolve())] = digest
    result, _, _ = chain(folder, result_sha, pin, worker=True, limit=60)
    require(result['schema'] == SCHEMA and result['role'] == role and result['hashes_unchanged'] and result['sources_unchanged']
        and result['source_bindings'].get(str(Path(__file__).resolve())) == source_sha
        and result['source_bindings'].get(str(HERE/'BUILD_PROTOCOL.md')) == protocol_sha
        and not result['estimator_executed'] and not result['truth_arrays_accessed'] and not result['refit_performed']
        and result['parameter_updates'] == 0 and result['training_provenance_used'] == (role == 'terminal'),
        'Complete qualified source-only build scope')
    for group in ('source_bindings', 'input_bindings', 'artifacts'):
        for path, digest in result[group].items():
            pin(path, digest)
    for path, digest in PINS.items():
        require(result['source_bindings'].get(str(path), result['input_bindings'].get(str(path))) == digest,
                'Fixed parent/helper authority')
    candidate, parent = result['candidate'], result['parent']
    require(candidate == dict(path=str(folder/'estimator.py'), sha256=candidate_sha)
        and parent == dict(path=str(PARENT), sha256=PARENT_SHA), 'Exact role path and original COMP parent')
    pin(candidate['path'], candidate_sha); pin(PARENT, PARENT_SHA)
    actual = Path(candidate['path']).read_bytes()
    expected, Pmeta, Qmeta, proof = source_candidate(role, value_span(actual, '_GNN_LITERAL')[3],
                                                  value_span(actual, '_PCC_LITERAL')[3])
    require(actual == expected and result['P_literal'] == Pmeta and result['covariance_literal'] == Qmeta
        and result['proof'] == proof, 'Actual candidate finite P/Q and complete source inverse reconstructed')
    if role == 'terminal':
        require(Pmeta == result['terminal_payload']['arrays']
            and Qmeta == {key: result['terminal_Q'][key] for key in ('dtype', 'shape', 'raw_sha256')},
            'Candidate P/Q are the sole sealed terminal, not another payload')
    return dict(build=result, candidate=candidate, parent=parent, source_bindings=dict(result['source_bindings']),
        input_bindings={**result['input_bindings'], **pins})


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--build', action='store_true')
    parser.add_argument('--role', choices=('zero', 'terminal'))
    for key in COMMON_ARGS+TRAIN_ARGS:
        parser.add_argument('--'+key.replace('_', '-')+'-sha256')
    args = parser.parse_args()
    if not args.build:
        parser.print_help(); return 0
    require(args.role in ('zero', 'terminal'), 'One explicit build role')
    for key in COMMON_ARGS+(TRAIN_ARGS if args.role == 'terminal' else ()):
        require(re.fullmatch('[0-9a-f]{64}', getattr(args, key+'_sha256') or ''), 'Explicit reviewed/future SHA: '+key)
    began = time.monotonic()
    def expired(*_):
        raise TimeoutError('Complete60s P/Q literal build')
    signal.signal(signal.SIGALRM, expired); signal.alarm(60)
    folder = HERE/('build_'+args.role+'_v1'); folder.mkdir(exist_ok=False)
    record = dict(schema=SCHEMA, role=args.role, passed=False, completed=False, source_bindings={}, input_bindings={}, artifacts={},
        estimator_executed=False, truth_arrays_accessed=False, refit_performed=False, native_gradient_qualified=False,
        native_score_measured=False, parameter_updates=0, hashes_unchanged=False, sources_unchanged=False)
    def guard():
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != 'darwin':
            rss *= 1024
        require(time.monotonic()-began < 60 and rss < 2_000_000_000, 'Whole60s/2GB literal build')
        return rss
    try:
        require(sys.flags.optimize == 0 and os.getpriority(os.PRIO_PROCESS, 0) >= 10
            and sys.executable == '/Users/matt/anaconda3/bin/python'
            and all(os.environ.get(key) == '1' for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')), 'AP/nice10/BLAS1 builder')
        train, pin = admission(args, record, guard)
        require(not any(key in sys.modules for key in ('numpy', 'torch', 'flopscope', 'threadpoolctl')),
                'All qualification/training/source admission precedes NumPy-only decoding')
        import numpy as np
        require(np.__version__ == '1.26.4', 'Fixed FP32 literal conversion runtime')
        parent = PARENT.read_bytes(); p0 = value_span(parent, '_GNN_LITERAL')[3]
        original = {key: np.asarray(value, dtype=np.float32) for key, value in p0.items()}
        if args.role == 'zero':
            terminal = {key: value.copy() for key, value in original.items()}
            metadata = record['initial_P']; Q = np.zeros((14, 65), dtype=np.float32)
        else:
            pm, qm = train['terminal_payload'], train['terminal_Q']
            with np.load(pm['path'], allow_pickle=False) as archive:
                terminal = {key: archive[key].copy() for key in archive.files}
            metadata = pm['arrays']; Q = np.load(qm['path'], allow_pickle=False)
            require(array_meta(Q) == {key: qm[key] for key in ('dtype', 'shape', 'raw_sha256')}, 'Saved terminal Q bytes')
            for rank in VALID:
                item = train['validation_outputs'][str(rank)]
                value = np.load(item['path'], allow_pickle=False)
                require(value.dtype == np.float32 and value.shape == (16, 1024) and np.isfinite(value).all()
                    and (value >= 0).all() and array_meta(value) == {key: item[key] for key in ('dtype', 'shape', 'raw_sha256')},
                    'Each actual sealed target-blind CHECK prediction')
        validate_payload(np, terminal, original, metadata)
        require(Q.dtype == np.float32 and Q.shape == (14, 65) and np.isfinite(Q).all(), 'Finite exact covariance surface')
        mean_literal = {key: terminal[key].tolist() for key in p0}
        candidate, Pmeta, Qmeta, proof = source_candidate(args.role, mean_literal, Q.tolist())
        require(Pmeta == {key: array_meta(value) for key, value in terminal.items()} and Qmeta == array_meta(Q),
                'NumPy/Python-literal FP32 bytes match exactly')
        for group in ('source_bindings', 'input_bindings'):
            for path, digest in record[group].items():
                require(sha(path) == digest, 'Prewrite admitted bytes unchanged'); guard()
        target = folder/'estimator.py'
        with target.open('xb') as stream:
            stream.write(candidate)
        digest = sha(target); record['artifacts'][str(target)] = digest
        record.update(candidate=dict(path=str(target), sha256=digest), parent=dict(path=str(PARENT), sha256=PARENT_SHA),
            P_literal=Pmeta, covariance_literal=Qmeta, proof=proof, peak_rss_bytes=guard(), passed=True, completed=True,
            hashes_unchanged=True, sources_unchanged=True)
    except BaseException:
        record.update(passed=False, completed=False, error=traceback.format_exc())
    record['elapsed_s'] = time.monotonic()-began
    write(folder/'result.json', record); digest = sha(folder/'result.json')
    write(folder/'completion.json', dict(passed=record['passed'], completed=record['completed'], result_sha256=digest))
    end = sha(folder/'completion.json'); elapsed = time.monotonic()-began; ok = record['passed'] and elapsed < 60
    write(folder/'finalization.json', dict(passed=ok, completed=ok, result_sha256=digest,
        completion_sha256=end, elapsed_s=elapsed))
    signal.alarm(0)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
