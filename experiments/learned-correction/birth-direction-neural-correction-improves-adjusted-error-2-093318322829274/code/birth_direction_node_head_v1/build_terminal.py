"""Stdlib-only terminal A builder and admission; inert without --build.

Requires the completed fixed6-epoch/30-update training receipt and all four
sealed prediction files. No model, NumPy/FNP, fit, target array or network
execution. Only _BDNH_A_LITERAL changes in the qualified zero prototype.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
BD = NEW / 'birth_direction_fit_v1'
TRAIN_DIR, OUT = HERE / 'training_v1', HERE / 'build_terminal_v1'
VALID = (37, 39, 42, 48)
TRAIN = tuple(rank for rank in range(35, 59) if rank not in VALID)
TRAIN_SHA = 'f2982348c5a56676443c3ab2cc24370e14fcf1c06150ea2ff09d285d39a9949f'
PROTOTYPE_SHA = '42bef1c5c8aeaf0b60ae96961bb7c29d20535a555bbbfc554a61c5ec0aecb926'
WIDE_NATIVE_SHA = '70506904442e58f0931aa254a41cc7be25522b73363db2118eefec6761cdf2e0'
WIDE_SOURCE_SHA = '428cb85613e3bf4b0baec7812f408760171f586a51bb5d7d7b554c1166420a5e'
WIDE_RESULT_SHA = '66ddcb87f1abeb45438da7598422aee373a376e66997e6a5a33dace07eb1f434'
PARENT_SHA = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'
ZERO_SHA = '1a99b49fcdc5cbfdaa03a40e676c0a910c5cf4e2d001965bf9e15e18678f1f05'
ZERO_BUILD_SHA = 'a560235a6cbee805b2d37f50b34583996d93e013fafcbb4c2b6243c408aa7f48'
INITIALIZATION = {'parent': 1383913786, 'zero': 1383914298, 'terminal': 1383914298}
ROLE_PATHS = {'parent': BD / 'build_terminal_v1/estimator.py',
              'zero': HERE / 'build_zero_v1/estimator.py', 'terminal': OUT / 'estimator.py'}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def path(value):
    value = Path(value).resolve()
    require(value.parent.name != 'truthpost' and not value.name.startswith('mps_truth_')
            and not value.name.endswith('_truth4M.npz'), 'No truth-array access in terminal source admission')
    return value


def sha(value):
    digest = hashlib.sha256()
    with path(value).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(value):
    return json.loads(path(value).read_text())


def pin(pins, value, expected=None):
    value = path(value)
    digest = sha(value)
    require(expected is None or digest == expected, 'Input SHA: ' + str(value))
    require(str(value) not in pins or pins[str(value)] == digest, 'Conflicting input pin')
    pins[str(value)] = digest
    return digest


def absorb(items, pins):
    for key, item in items.items():
        if isinstance(item, str):
            pin(pins, key, item)
        else:
            value = path(item['path'])
            pin(pins, value, item['sha256'])
            stat = value.stat()
            require('bytes' not in item or stat.st_size == item['bytes'], 'Binding size changed')
            require('mtime_ns' not in item or stat.st_mtime_ns == item['mtime_ns'], 'Binding timestamp changed')


def unchanged(pins):
    require(all(sha(value) == digest for value, digest in pins.items()), 'Sealed input changed')


def write(value, record):
    with Path(value).open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return sha(value)


def module(value, name):
    spec = importlib.util.spec_from_file_location(name, value)
    require(spec is not None and spec.loader is not None, 'Module specification')
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def prediction_bytes(item, expected_path, pins):
    """Authenticate a synthetic/output NPY container without numeric decoding."""
    require(path(item['path']) == expected_path.resolve(), 'Fixed validation output path')
    absorb({str(expected_path): item}, pins)
    blob = expected_path.read_bytes()
    require(blob[:6] == b'\x93NUMPY', 'NPY output required')
    version = tuple(blob[6:8])
    require(version in ((1, 0), (2, 0), (3, 0)), 'Known NPY header version')
    size = 2 if version == (1, 0) else 4
    length = int.from_bytes(blob[8:8+size], 'little')
    offset = 8 + size + length
    header = ast.literal_eval(blob[8+size:offset].decode('utf8' if version == (3, 0) else 'latin1'))
    require(header == dict(descr='<f4', fortran_order=False, shape=(16, 1024)), 'Exact served NPY frame')
    require(len(blob) - offset == 16 * 1024 * 4, 'Concrete complete validation output')
    digest = hashlib.sha256(blob[offset:]).hexdigest()
    require(item['shape'] == [16, 1024] and item['dtype'] == 'float32'
            and digest == item['raw_sha256'] == item['output_bytes_sha256'], 'Validation payload byte seal')


def training_admission(training_sha):
    """Return pins/training/prototype/A; no target or numeric-library read."""
    pins = {}
    for value, digest in ((HERE / 'train.py', TRAIN_SHA), (HERE / 'build.py', PROTOTYPE_SHA),
                          (HERE / 'native_wide.py', WIDE_NATIVE_SHA),
                          (ROLE_PATHS['parent'], PARENT_SHA), (ROLE_PATHS['zero'], ZERO_SHA),
                          (HERE / 'build_zero_v1/build.json', ZERO_BUILD_SHA)):
        pin(pins, value, digest)
    pin(pins, __file__)
    require(isinstance(training_sha, str) and len(training_sha) == 64, 'Explicit terminal training result SHA')
    pin(pins, TRAIN_DIR / 'result.json', training_sha)
    pin(pins, TRAIN_DIR / 'completion.json')
    training, completion = read(TRAIN_DIR / 'result.json'), read(TRAIN_DIR / 'completion.json')
    require(training['passed'] and training['completed'] and training['hashes_unchanged']
            and completion['passed'] and completion['completed']
            and completion['result_sha256'] == training_sha, 'Completed fixed training required')
    require(training['schema'] == 'birth_direction_node_head_training_v1'
            and training['seed'] == 9309093 and training['epochs'] == 6
            and training['updates'] == training['updates_completed'] == 30
            and training['planned_updates'] == 30 and training['batch_nets'] == 4
            and training['allocated_parameters'] == 256
            and training['train_ranks'] == list(TRAIN) and training['validation_ranks'] == list(VALID),
            'Exact20/4 six-epoch30-update pilot')
    require(training['frozen_P'] and training['frozen_cstar'] and training['terminal_restore_byte_exact']
            and training['training_targets_accessed'] and not training['validation_targets_accessed']
            and not training['validation_score_measured'] and not training['native_complete_bill_measured']
            and not training['checkpoint_selection'] and not training['parameter_checkpoints']
            and not training['automatic_advancement'], 'No selection/refit/validation-score admission')
    require(training['source_bindings'][str(HERE / 'train.py')] == TRAIN_SHA, 'Actual trainer source')
    require(training['wide_source_sha256'] == WIDE_SOURCE_SHA
            and training['wide_judge_sha256'] == WIDE_RESULT_SHA, 'Exact completed fixed fidelity gate')
    absorb(training['source_bindings'], pins)
    absorb(training['source_binding_metadata'], pins)
    absorb(training['artifacts'], pins)
    absorb(training['artifact_metadata'], pins)
    names = {'config.json', 'weight_bindings.json', 'training_targets.json', 'input_seal.json',
             'fixed_orders.json', 'terminal_A.json', 'terminal_seal.json',
             'validation_predictions_sealed.json', 'progress.jsonl'}
    names.update(f'validation_{rank:04d}.npy' for rank in VALID)
    require(set(training['artifacts']) == {str(TRAIN_DIR / name) for name in names}, 'Exact no-checkpoint artifact set')
    trainer = module(HERE / 'train.py', 'bdnh_terminal_inert_training_contract')
    require(training['config'] == read(TRAIN_DIR / 'config.json') == trainer.CONFIG, 'Frozen fit configuration')
    wide_native = module(HERE / 'native_wide.py', 'bdnh_terminal_inert_prototype_contract')
    absorb(wide_native.provenance(), pins)
    # Reuse only the frozen trainer's hash/JSON admission, not main, targets,
    # model, environment setup or any numerical code.
    replay = dict(source_bindings={}, environment=training['environment'])
    prototype, cstar = trainer.admission(SimpleNamespace(source_sha256=TRAIN_SHA,
        wide_source_sha256=training['wide_source_sha256'],
        wide_judge_sha256=training['wide_judge_sha256']), replay)
    absorb(replay['source_bindings'], pins)
    require(replay['fidelity_stage_seals'] == training['fidelity_stage_seals'], 'Completed same wide gate')
    cstar_sha = hashlib.sha256(prototype.matrix_bytes(cstar, 15, 4)).hexdigest()
    require(cstar_sha == training['frozen_cstar_bytes_sha256'], 'Exact original cstar payload')
    terminal_path = TRAIN_DIR / 'terminal_A.json'
    terminal = training['terminal_A']
    require(path(terminal['path']) == terminal_path, 'Only the terminal parameter file')
    absorb({str(terminal_path): terminal}, pins)
    payload = read(terminal_path)
    encoded = prototype.matrix_bytes(payload['A'], 64, 4)
    raw_sha = hashlib.sha256(encoded).hexdigest()
    require(payload['schema'] == 'birth_direction_node_head_terminal_A_v1'
            and payload['shape'] == terminal['shape'] == [64, 4]
            and payload['dtype'] == terminal['dtype'] == 'float32' and payload['byte_order'] == 'little'
            and payload['terminal_epoch'] == 6 and payload['updates'] == 30
            and payload['learned'] and not payload['checkpoint_selection']
            and raw_sha == payload['raw_sha256'] == payload['A_bytes_sha256']
            == terminal['raw_sha256'] == terminal['A_bytes_sha256'], 'Exact learned FP32 terminal A')
    terminal_seal_path = TRAIN_DIR / 'terminal_seal.json'
    validation_seal_path = TRAIN_DIR / 'validation_predictions_sealed.json'
    pin(pins, terminal_seal_path, training['terminal_seal_sha256'])
    pin(pins, validation_seal_path, training['validation_predictions_sealed_sha256'])
    terminal_seal, sealed = read(terminal_seal_path), read(validation_seal_path)
    require(terminal_seal['schema'] == 'birth_direction_node_head_terminal_seal_v1'
            and terminal_seal['terminal_A'] == terminal and terminal_seal['epochs'] == 6
            and terminal_seal['updates'] == 30 and terminal_seal['seed'] == 9309093
            and not terminal_seal['validation_predictions_started']
            and not terminal_seal['validation_targets_accessed'] and not terminal_seal['checkpoint_selection']
            and terminal_seal['frozen_cstar_bytes_sha256'] == cstar_sha, 'Terminal-before-validation ownership')
    require(terminal_seal['source_bindings'] == training['source_bindings'], 'Same terminal/training source frame')
    for name, field in (('config.json', 'config_sha256'), ('input_seal.json', 'input_seal_sha256'),
                        ('fixed_orders.json', 'fixed_orders_sha256'),
                        ('weight_bindings.json', 'weight_bindings_sha256'),
                        ('training_targets.json', 'training_targets_sha256')):
        pin(pins, TRAIN_DIR / name, terminal_seal[field])
    require(sealed['schema'] == 'birth_direction_node_head_validation_seal_v1'
            and sealed['terminal_A'] == terminal and sealed['terminal_seal_sha256'] == training['terminal_seal_sha256']
            and sealed['validation_outputs'] == training['validation_outputs'] and sealed['all_4_complete']
            and sealed['terminal_restore_byte_exact'] and not sealed['validation_targets_accessed']
            and not sealed['native_complete_bill_measured'] and not sealed['raw_or_adjusted_score_measured'],
            'All4 fixed terminal predictions sealed without targets')
    require(set(training['validation_outputs']) == {str(rank) for rank in VALID}, 'Exact VALID output set')
    for rank in VALID:
        output_path = TRAIN_DIR / f'validation_{rank:04d}.npy'
        prediction_bytes(training['validation_outputs'][str(rank)], output_path, pins)
        require(terminal_path.stat().st_mtime_ns <= terminal_seal_path.stat().st_mtime_ns
                <= output_path.stat().st_mtime_ns <= validation_seal_path.stat().st_mtime_ns
                <= (TRAIN_DIR / 'result.json').stat().st_mtime_ns, 'Terminal/output/result chronology')
    # These metadata indices contain label hashes; the label arrays themselves
    # remain outside builder/native admission and are never opened here.
    target_meta = read(TRAIN_DIR / 'training_targets.json')
    require(target_meta['train_ranks'] == list(TRAIN) and target_meta['all_authenticated_before_decode']
            and not target_meta['validation_targets_accessed'] and len(target_meta['bindings']) == 20,
            'TRAIN-only target metadata')
    require(not any(name in sys.modules for name in ('numpy', 'torch', 'flopscope')), 'Source-only builder imported numerics')
    unchanged(pins)
    return dict(pins=pins, training=training, prototype=prototype, A=tuple(tuple(row) for row in payload['A']),
                terminal_A=terminal, training_result_sha256=training_sha)


def construct(zero, literal, prototype):
    require(hashlib.sha256(zero.encode()).hexdigest() == ZERO_SHA, 'Exact qualified zero source')
    prototype.matrix_bytes(literal, 64, 4)
    nodes = [node for node in ast.parse(zero).body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == '_BDNH_A_LITERAL' for target in node.targets)]
    require(len(nodes) == 1 and len(nodes[0].targets) == 1, 'Unique literal-only source boundary')
    node, lines = nodes[0], zero.splitlines(keepends=True)
    start, end = sum(map(len, lines[:node.lineno-1])), sum(map(len, lines[:node.end_lineno]))
    before = zero[start:end]
    after = '_BDNH_A_LITERAL = ' + repr(literal) + '\n'
    candidate = zero[:start] + after + zero[end:]
    require(candidate[:start] + before + candidate[start+len(after):] == zero, 'Whole-source byte inverse')
    parsed = ast.parse(candidate)
    imports = lambda tree: [ast.dump(n, include_attributes=False) for n in tree.body
                            if isinstance(n, (ast.Import, ast.ImportFrom))]
    require(imports(parsed) == imports(ast.parse(zero)), 'No import change')
    require(prototype.matrix_bytes(prototype.assignment_literal(candidate, '_BDNH_A_LITERAL'), 64, 4)
            == prototype.matrix_bytes(literal, 64, 4), 'Literal parameter bytes')
    for name in ('_bd_update', '_bdnh_parameters', '_bdnh_coefficients', 'gnn_step',
                 '_predict_binary', '_predict_learned_compact'):
        require(prototype.last_function(candidate, name)[2] == prototype.last_function(zero, name)[2],
                'An original mathematical/runtime function changed')
    require(prototype.assignment_literal(candidate, '_BD_COEFFICIENT_LITERAL')
            == prototype.assignment_literal(zero, '_BD_COEFFICIENT_LITERAL'), 'Original cstar unchanged')
    return candidate


def verify_build(training_sha, build_sha):
    context = training_admission(training_sha)
    pins = context['pins']
    pin(pins, OUT / 'build.json', build_sha)
    pin(pins, OUT / 'completion.json')
    record, completion = read(OUT / 'build.json'), read(OUT / 'completion.json')
    require(record['passed'] and record['completed'] and not record['executed']
            and completion['passed'] and completion['completed'] and completion['source_only']
            and completion['build_sha256'] == build_sha, 'Complete source-only terminal build')
    require(record['schema'] == 'birth_direction_node_head_terminal_build_v1'
            and record['training_result_sha256'] == training_sha and record['zero_candidate_sha256'] == ZERO_SHA
            and record['parent_sha256'] == PARENT_SHA and record['terminal_A'] == context['terminal_A'], 'Terminal build lineage')
    for flag in ('inverse_source_byte_exact', 'only_A_literal_changed', 'all_other_source_byte_exact',
                 'imports_and_P_payloads_unchanged', 'frozen_cstar_unchanged', 'no_new_CP_columns'):
        require(record[flag], flag)
    absorb(record['source_bindings'], pins)
    require(set(record['artifacts']) == {str(OUT / 'estimator.py'), str(OUT / 'A.json')}, 'Exact source build artifacts')
    absorb(record['artifacts'], pins)
    expected = construct(ROLE_PATHS['zero'].read_text(), context['A'], context['prototype'])
    require((OUT / 'estimator.py').read_bytes() == expected.encode()
            and sha(OUT / 'estimator.py') == record['candidate_sha256'], 'Exact terminal participant bytes')
    require((OUT / 'A.json').read_bytes() == (TRAIN_DIR / 'terminal_A.json').read_bytes(), 'Original terminal payload copy')
    context.update(build=record, build_sha256=build_sha, candidates={
        'parent': dict(path=str(ROLE_PATHS['parent']), sha256=PARENT_SHA),
        'zero': dict(path=str(ROLE_PATHS['zero']), sha256=ZERO_SHA),
        'terminal': dict(path=str(ROLE_PATHS['terminal']), sha256=record['candidate_sha256'])})
    unchanged(pins)
    return context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--source-sha256')
    parser.add_argument('--training-sha256')
    args = parser.parse_args()
    if not args.build:
        parser.print_help()
        return 0
    require(sha(__file__) == args.source_sha256, 'Reviewed terminal builder source pin')
    require(not OUT.exists(), 'Exclusive terminal build; never overwrite/resume')
    context = training_admission(args.training_sha256)
    candidate = construct(ROLE_PATHS['zero'].read_text(), context['A'], context['prototype'])
    unchanged(context['pins'])
    OUT.mkdir(exist_ok=False)
    record = dict(schema='birth_direction_node_head_terminal_build_v1', passed=False, completed=False,
        executed=False, source_bindings=context['pins'], training_result_sha256=args.training_sha256,
        terminal_A=context['terminal_A'], zero_candidate_sha256=ZERO_SHA, parent_sha256=PARENT_SHA,
        candidate=str(OUT / 'estimator.py'), candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
        inverse_source_byte_exact=True, only_A_literal_changed=True, all_other_source_byte_exact=True,
        imports_and_P_payloads_unchanged=True, frozen_cstar_unchanged=True, no_new_CP_columns=True,
        numerical_or_score_qualification=False, artifacts={}, utc=datetime.now(timezone.utc).isoformat())
    try:
        with (OUT / 'estimator.py').open('xb') as stream:
            stream.write(candidate.encode())
        with (OUT / 'A.json').open('xb') as stream:
            stream.write((TRAIN_DIR / 'terminal_A.json').read_bytes())
        record['artifacts'] = {str(OUT / name): sha(OUT / name) for name in ('estimator.py', 'A.json')}
        unchanged(context['pins'])
        record.update(passed=True, completed=True, sources_unchanged=True)
    except BaseException:
        record.update(passed=False, completed=False, error=traceback.format_exc())
    digest = write(OUT / 'build.json', record)
    write(OUT / 'completion.json', dict(passed=record['passed'], completed=record['completed'],
        source_only=True, build_sha256=digest))
    print(json.dumps(dict(passed=record['passed'], candidate_sha256=record['candidate_sha256'],
                         build_sha256=digest)), flush=True)
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
