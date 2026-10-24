"""Inert fixed20/4 node-head terminal pilot judge; no estimator or fitting.

Only the epoch-six/30-update terminal A and its four already-sealed MPS
outputs may enter. Fresh zero/terminal native4+repeat own their complete
bills. Parent d50 reuses the EXACT first-four prefix37/39/42/48 of its
authenticated official-input native40 run, with its original cold37 bill.
All41 parent calls/lifecycles/repeat are checked; no costs are prorated.

All native and MPS outputs/bills are sealed before any VALID truth file
is hashed or decoded. Score is OBSERVED1e9 final MSE times each network's
own max(.1,F/2**41). Pilot advancement needs mean ratio<=.97 against EACH
control and at least3 SAME ranks beating BOTH controls. Numerical fidelity
also requires all16 terminal native/MPS max difference<=1e-5 and final-MSE
ratio within1%. A completed valid score has passed=True even if the separate
pilot_gate_passed=False. Neither result automatically admits another fit,
an independent panel, a production release or a submission.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
BD = NEW / 'birth_direction_fit_v1'
ENV = P2 / 'data_gate/.venv'
TRUTH = P2 / 'public_g22_operator_features_v1/truthpost'
TRUTH_INDEX = NEW / 'joint_prefix_final_judge_cached_v2/receipt.json'
TRUTH_INDEX_SHA = '4968610553b95011cabdc651dd65d4c9057054f9d2d2bf73711426fff695893c'
VALID = (37, 39, 42, 48)
BUDGET = 2**41
PARENT = BD / 'build_terminal_v1/estimator.py'
PARENT_SHA = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'
PARENT_FOLDER = BD / 'validation_terminal_v2'
PARENT_RESULT_SHA = 'ef7f4fcd8c6bc9275e434560ac5f514fd5f99d94077441d757ad34fb3a8df17d'
ZERO_SHA = '1a99b49fcdc5cbfdaa03a40e676c0a910c5cf4e2d001965bf9e15e18678f1f05'
TRAINER_SHA = 'f2982348c5a56676443c3ab2cc24370e14fcf1c06150ea2ff09d285d39a9949f'
WIDE_SHA = '428cb85613e3bf4b0baec7812f408760171f586a51bb5d7d7b554c1166420a5e'
WIDE_RESULT_SHA = '66ddcb87f1abeb45438da7598422aee373a376e66997e6a5a33dace07eb1f434'
VERIFIER_SHA = '5528601148c98384de72af4ea7f7f39eed1d4657c51617910cd2be466299a608'
NATIVE_BASE_SHA = '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
INITIALIZATION = {'parent': 1383913786, 'zero': 1383914298, 'terminal': 1383914298}
VERSIONS = {'numpy': '2.4.6', 'flopscope': '0.12.0', 'whestbench': '0.16.0', 'threadpoolctl': '3.6.0'}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': repr(value)}
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    return value


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Module specification')
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def unchanged(pins):
    require(all(sha(path) == digest for path, digest in pins.items()), 'Sealed source/artifact changed')


def native_verifier(candidates, result):
    """Only identity/init edits; exact inverse of the qualified five-file verifier."""
    path = BD / 'qualify_native_v2.py'
    require(sha(path) == VERIFIER_SHA, 'Qualified native-v2 verifier source')
    text = path.read_text()
    node = next(x for x in ast.parse(text).body if isinstance(x, ast.FunctionDef) and x.name == 'verify_native')
    original = ''.join(text.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    edits = (
        ("    expected_sha = BANK_SHA if source == BANK else {HERE / 'build_zero_v1/estimator.py': ZERO_SHA,\n"
         "                       HERE / 'build_terminal_v1/estimator.py': TERMINAL_SHA}.get(source)",
         '    expected_sha = NODE_CANDIDATES.get(source)'),
        ('    expected_init = 1383913666 + (0 if source == BANK else 120)',
         '    expected_init = NODE_INITIALIZATION[source]'),
    )
    derived = original
    for old, new in edits:
        require(derived.count(old) == 1, 'Unique native-verifier source edit')
        derived = derived.replace(old, new)
    inverse = derived
    for old, new in reversed(edits):
        require(inverse.count(new) == 1, 'Unique native-verifier inverse')
        inverse = inverse.replace(new, old)
    require(inverse == original, 'All five-artifact/resource/input/reset/repeat checks preserved')
    base = module(path, 'bdnh_terminal_preserved_native_verifier')
    mapping = {Path(item['path']).resolve(): item['sha256'] for item in candidates.values()}
    costs = {Path(item['path']).resolve(): INITIALIZATION[role] for role, item in candidates.items()}
    namespace = dict(vars(base), NODE_CANDIDATES=mapping, NODE_INITIALIZATION=costs)
    exec(compile(derived, str(__file__) + ':native_verifier', 'exec'), namespace)
    result['native_verifier_inverse_byte_exact'] = True
    return base, namespace['verify_native']


def judge(args, out, result):
    pins, truth_pins = {}, {}
    result['source_bindings'], result['truth_bindings'] = pins, truth_pins
    def pin(path, expected=None):
        path = Path(path)
        path = (path if path.is_absolute() else NEW / path).resolve()
        require(not path.is_relative_to(TRUTH.resolve()), 'No truth file may enter pre-score source/artifact bindings')
        actual = sha(path)
        require(expected is None or actual == expected, 'Source/artifact SHA mismatch: ' + str(path))
        require(str(path) not in pins or pins[str(path)] == actual, 'Conflicting or changed source binding')
        pins[str(path)] = actual
        return actual
    def bind(items):
        for name, item in items.items():
            if isinstance(item, dict):
                pin(item['path'], item['sha256'])
            else:
                pin(name, item)

    pin(__file__, args.source_sha256)
    pin(HERE / 'train.py', TRAINER_SHA)
    pin(HERE / 'check_wide.py', WIDE_SHA)
    pin(HERE / 'build_terminal.py', args.builder_sha256)
    pin(HERE / 'native_validation.py', args.native_source_sha256)
    pin(BD / 'qualify_native_v2.py', VERIFIER_SHA)
    pin(BD / 'native_v2.py', NATIVE_BASE_SHA)
    pin(TRUTH_INDEX, TRUTH_INDEX_SHA)
    builder = module(HERE / 'build_terminal.py', 'bdnh_terminal_inert_build_verifier')
    context = builder.verify_build(args.training_sha256, args.build_sha256)
    bind(context['pins'])
    training, build, candidates = context['training'], context['build'], context['candidates']
    require(set(candidates) == {'parent', 'zero', 'terminal'} and builder.VALID == VALID
            and builder.INITIALIZATION == INITIALIZATION, 'Fixed role/split/initialization interface')
    require(candidates['parent'] == {'path': str(PARENT), 'sha256': PARENT_SHA}
            and candidates['zero'] == {'path': str(HERE / 'build_zero_v1/estimator.py'), 'sha256': ZERO_SHA}
            and candidates['terminal']['path'] == str(HERE / 'build_terminal_v1/estimator.py')
            and candidates['terminal']['sha256'] == build['candidate_sha256'], 'Exact parent/zero/terminal candidates')
    require(training['passed'] and training['completed'] and training['hashes_unchanged']
            and training['epochs'] == 6 and training['updates'] == training['updates_completed'] == 30
            and training['validation_ranks'] == list(VALID) and len(training['train_ranks']) == 20
            and not set(training['train_ranks']) & set(VALID) and training['allocated_parameters'] == 256
            and training['frozen_P'] and training['frozen_cstar']
            and not training['validation_targets_accessed'] and not training['validation_score_measured']
            and not training['checkpoint_selection'] and not training['parameter_checkpoints']
            and training['terminal_restore_byte_exact'], 'Only the fixed terminal six-epoch A fit')
    require(training['source_bindings'][str(HERE / 'train.py')] == TRAINER_SHA
            and training['wide_source_sha256'] == WIDE_SHA
            and training['wide_judge_sha256'] == WIDE_RESULT_SHA, 'Exact trainer/fidelity source lineage')
    require(set(training['validation_outputs']) == set(map(str, VALID)), 'Exactly four sealed MPS predictions')
    seal_path = HERE / 'training_v1/validation_predictions_sealed.json'
    pin(seal_path, training['validation_predictions_sealed_sha256'])
    terminal_seal_path = HERE / 'training_v1/terminal_seal.json'
    pin(terminal_seal_path, training['terminal_seal_sha256'])
    seal, terminal_seal = read(seal_path), read(terminal_seal_path)
    require(seal['all_4_complete'] and seal['terminal_restore_byte_exact']
            and not seal['validation_targets_accessed'] and not seal['raw_or_adjusted_score_measured']
            and seal['validation_outputs'] == training['validation_outputs']
            and seal['terminal_A'] == training['terminal_A']
            and seal['terminal_seal_sha256'] == training['terminal_seal_sha256'], 'Pre-truth validation seal')
    require(terminal_seal['epochs'] == 6 and terminal_seal['updates'] == 30
            and not terminal_seal['validation_predictions_started']
            and not terminal_seal['validation_targets_accessed'] and not terminal_seal['checkpoint_selection'],
            'Terminal payload frozen before validation inference')
    base, verify = native_verifier(candidates, result)
    all_parent_ranks = tuple(base.VALID)
    require(all_parent_ranks[:4] == VALID and len(all_parent_ranks) == 40, 'Exact archived parent first-four prefix')
    definitions = {
        'parent': (PARENT_FOLDER, PARENT_RESULT_SHA, all_parent_ranks),
        'zero': (HERE / 'native_validation_zero_v1', args.zero_sha256, VALID),
        'terminal': (HERE / 'native_validation_terminal_v1', args.terminal_sha256, VALID),
    }
    acquisitions, selected_bytes, selected_trials = {}, {}, {}
    for role, (folder, digest, ranks) in definitions.items():
        pin(folder / 'result.json', digest)
        pin(folder / 'completion.json')
        acq, completion = read(folder / 'result.json'), read(folder / 'completion.json')
        require(acq['passed'] and completion['passed'] and completion['result_sha256'] == digest
                and Path(acq['source']).resolve() == Path(candidates[role]['path'])
                and acq['candidate_sha256'] == candidates[role]['sha256'], 'Complete candidate acquisition role')
        if role != 'parent':
            require(acq['schema'] == 'birth_direction_node_head_native4_v1' and acq['role'] == role
                    and acq['completed'] and acq['own_complete_bills'] and acq['frozen_P'] and acq['frozen_cstar']
                    and acq['training_result_sha256'] == args.training_sha256
                    and acq['terminal_build_sha256'] == args.build_sha256
                    and acq['source_bindings'][str(HERE / 'native_validation.py')] == args.native_source_sha256,
                    'Fresh node native4 source/build/training ownership')
        raw = verify(acq, folder, ranks, pin)
        require(len(raw) == len(ranks)+1, 'Every native call and repeat verified')
        acquisitions[role], selected_bytes[role], selected_trials[role] = acq, raw[:4], acq['trials'][:4]
    # Backend and accounting implementation are matched, not merely similar
    # score summaries from the old NumPy-input frame.
    pools = acquisitions['parent']['threadpools']
    require(pools and all(x['num_threads'] == 1 for x in pools if x.get('user_api') == 'blas'),
            'Recorded parent BLAS1 backend')
    require(all(acquisitions[role]['threadpools'] == pools for role in ('zero', 'terminal')),
            'Identical recorded native numerical backend')
    require(selected_bytes['zero'] == selected_bytes['parent'], 'Zero node all16 output bytes equal parent on all4')
    require(all(a[:4096] == b[:4096] for a, b in zip(selected_bytes['terminal'], selected_bytes['parent'])),
            'Terminal Gaussian layer0 is unchanged')
    result['parent_reuse'] = dict(result_sha256=PARENT_RESULT_SHA, verified_calls=41,
        scored_call_indices=[0, 1, 2, 3], prior_repeat_call=40, ranks=list(VALID),
        initialization_flops=INITIALIZATION['parent'], initialization_included_once=True,
        costs_prorated=False, exact_original_prefix=True, newly_measured_parent=False,
        scope='Actual d50 official-input first-four prefix; full archived40+repeat integrity verified')
    result['zero_all16_parent_bytes_exact'] = True
    require('torch' not in sys.modules and 'flopscope' not in sys.modules, 'No model/FNP/Torch scorer import')
    import numpy as np  # Only sealed predictions and, after the seal, fixed score labels.
    predictions = {role: [np.frombuffer(data, dtype='<f4').reshape(16, 1024) for data in arrays]
                   for role, arrays in selected_bytes.items()}
    mps, drift = [], []
    result['mps_native_drift'] = drift
    for call, rank in enumerate(VALID):
        item = training['validation_outputs'][str(rank)]
        path = HERE / 'training_v1' / f'validation_{rank:04d}.npy'
        require(Path(item['path']) == path, 'Exact terminal-only MPS output path')
        pin(path, item['sha256'])
        value = np.load(path, allow_pickle=False)
        require(type(value) is np.ndarray and value.shape == (16, 1024) and value.dtype == np.float32
                and np.isfinite(value).all() and (value >= 0).all()
                and hashlib.sha256(value.tobytes()).hexdigest() == item['output_bytes_sha256'] == item['raw_sha256'],
                'Finite, nonnegative, byte-authenticated terminal MPS prediction')
        difference = value.astype(np.float64)-predictions['terminal'][call].astype(np.float64)
        maximum = float(np.max(np.abs(difference)))
        drift.append(dict(rank=rank, max_absolute=maximum,
                          final_rms=float(np.sqrt(np.mean(difference[-1]**2)))))
        require(maximum <= 1e-5, 'Terminal native/MPS all16 means within1e-5')
        mps.append(value)
    unchanged(pins)
    before_truth = dict(utc=utc(), source_bindings=dict(pins), requested_seals=vars(args),
        truth_accessed=False, terminal_A=training['terminal_A'], validation_outputs=training['validation_outputs'],
        native_trials=selected_trials, parent_reuse=result['parent_reuse'], mps_native_drift=drift,
        all4_predictions_and_own_bills=True, initialization_once=True, repeats_excluded=True)
    write(out / 'predictions_and_bills_sealed.json', before_truth)
    seal_sha = sha(out / 'predictions_and_bills_sealed.json')
    result['artifacts'] = {str(out / 'predictions_and_bills_sealed.json'): seal_sha}
    result.update(predictions_and_bills_seal_sha256=seal_sha,
                  all_predictions_verified_before_truth=True, mps_native_drift=drift)
    records = read(TRUTH_INDEX)['records']
    require(len(records) == 40 and set(VALID).issubset({r['rank'] for r in records}), 'Fixed existing1e9 VALID index')
    expected = {r['rank']: r['target_sha256'] for r in records}
    result['records'] = []
    for call, rank in enumerate(VALID):
        path = TRUTH / f'truthpost_rank{rank:04d}.npz'
        result['truth_accessed'] = True
        require(sha(path) == expected[rank], 'Fixed VALID truth SHA')
        truth_pins[str(path)] = expected[rank]
        with np.load(path, allow_pickle=False) as source:
            target = source['mc_post'].astype(np.float64)
        require(target.shape == (16, 1024) and np.isfinite(target).all(), 'Finite existing1e9 mean target')
        row = dict(rank=rank, per_role={}, adjusted_wins={}, raw_wins={}, truth_sha256=expected[rank])
        for role in ('parent', 'zero', 'terminal'):
            per_layer = np.mean((predictions[role][call].astype(np.float64)-target)**2, axis=1)
            raw, flops = float(per_layer[-1]), selected_trials[role][call]['flops']
            multiplier = max(.1, flops/BUDGET)
            require(math.isfinite(raw) and raw >= 0 and type(flops) is int and 0 < flops <= BUDGET,
                    'Finite actual final observed raw and complete bill')
            row['per_role'][role] = dict(raw_mse=raw, adjusted=raw*multiplier,
                flops=flops, multiplier=multiplier, layer_mse=per_layer.tolist())
        require(row['per_role']['zero']['raw_mse'] == row['per_role']['parent']['raw_mse'],
                'Byte-equal zero/parent must yield identical raw')
        native_raw = row['per_role']['terminal']['raw_mse']
        mps_raw = float(np.mean((mps[call][-1].astype(np.float64)-target[-1])**2))
        ratio = mps_raw/native_raw if native_raw > 0 else None
        row.update(mps_final_mse=mps_raw, mps_native_mse_ratio=ratio)
        result['records'].append(row)
        require(native_raw > 0 and math.isfinite(mps_raw) and .99 <= ratio <= 1.01,
                'Terminal native/MPS final-MSE fidelity within1%')
        for control in ('parent', 'zero'):
            row['adjusted_wins'][control] = row['per_role']['terminal']['adjusted'] < row['per_role'][control]['adjusted']
            row['raw_wins'][control] = native_raw < row['per_role'][control]['raw_mse']
        row['beats_both_controls'] = all(row['adjusted_wins'].values())
        print(json.dumps(dict(rank=rank, per_role=row['per_role'], beats_both=row['beats_both_controls'])), flush=True)
    rows = result['records']
    summaries = {role: dict(mean_raw_mse=math.fsum(r['per_role'][role]['raw_mse'] for r in rows)/4,
        mean_adjusted=math.fsum(r['per_role'][role]['adjusted'] for r in rows)/4,
        mean_complete_flops=sum(r['per_role'][role]['flops'] for r in rows)/4,
        initialization_flops=INITIALIZATION[role]) for role in ('parent', 'zero', 'terminal')}
    comparisons = {}
    for control in ('parent', 'zero'):
        require(summaries[control]['mean_adjusted'] > 0 and summaries[control]['mean_raw_mse'] > 0,
                'Positive control objective for ratio')
        ratio = summaries['terminal']['mean_adjusted']/summaries[control]['mean_adjusted']
        comparisons[control] = dict(mean_adjusted_ratio=ratio, mean_adjusted_improvement=1-ratio,
            mean_raw_ratio=summaries['terminal']['mean_raw_mse']/summaries[control]['mean_raw_mse'],
            adjusted_wins=sum(r['adjusted_wins'][control] for r in rows),
            raw_wins=sum(r['raw_wins'][control] for r in rows), mean_gain_at_least3pct=ratio <= .97)
    joint_wins = sum(r['beats_both_controls'] for r in rows)
    gate = all(x['mean_gain_at_least3pct'] for x in comparisons.values()) and joint_wins >= 3
    unchanged(pins)
    unchanged(truth_pins)
    require(sha(out / 'predictions_and_bills_sealed.json') == seal_sha, 'Pre-truth seal unchanged')
    result.update(passed=True, completed=True, integrity_passed=True, numerical_fidelity_passed=True,
        summary=summaries, comparisons=comparisons, joint_adjusted_wins=joint_wins,
        score_thresholds_passed=gate, pilot_gate_passed=gate, eligible_for_separate_larger_fit_review=gate,
        automatic_advancement=False, hashes_unchanged=True,
        aggregation='Mean of per-net observed1e9 final MSE * max(.1, own complete F/2^41); cold initialization once; repeats excluded',
        frame='Four reused-development VALID37/39/42/48; not NC14, official score or production qualification')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--judge', action='store_true')
    for name in ('source', 'builder', 'native-source', 'training', 'build', 'zero', 'terminal'):
        parser.add_argument('--' + name + '-sha256')
    args = parser.parse_args()
    if not args.judge:
        parser.print_help()
        return 0
    require(all(isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v)
                for k, v in vars(args).items() if k != 'judge'), 'All reviewed source/result SHA values required')
    require(Path(sys.prefix).resolve() == ENV.resolve()
            and {k: importlib.metadata.version(k) for k in VERSIONS} == VERSIONS, 'Pinned OFFPY environment')
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10 and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
            and all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')),
            'nice10, no bytecode, CPU BLAS1')
    require('torch' not in sys.modules and 'flopscope' not in sys.modules, 'Fresh non-model scoring process')
    os.chdir(NEW)
    out = HERE / 'terminal_score_v1'
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    result = dict(schema='birth_direction_node_head_terminal_native4_v1', passed=False, completed=False,
        integrity_passed=False, truth_accessed=False, requested_seals=vars(args), started_utc=utc(),
        official_score=False, production_qualified=False, refit_performed=False, automatic_advancement=False)
    write(out / 'started.json', result)
    def timeout(_signum, _frame):
        raise TimeoutError('Fixed300-second pilot-score watchdog')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(295)
    try:
        judge(args, out, result)
    except BaseException:
        result.update(passed=False, completed=False, error=traceback.format_exc())
    finally:
        signal.alarm(0)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = int(rss if sys.platform == 'darwin' else rss * 1024)
    elapsed = time.monotonic()-start
    if rss >= 2e9 or elapsed >= 300:
        result.update(passed=False, completed=False, resource_failure=True)
    if not result['passed']:
        result.update(pilot_gate_passed=False, eligible_for_separate_larger_fit_review=False)
    result.update(finished_utc=utc(), elapsed_s=elapsed, peak_rss_bytes=rss)
    write(out / 'result.json', result)
    write(out / 'completion.json', dict(passed=result['passed'], completed=result['completed'],
        result_sha256=sha(out / 'result.json')))
    print(json.dumps(safe({k: result.get(k) for k in ('passed', 'pilot_gate_passed', 'joint_adjusted_wins',
                                                    'summary', 'comparisons', 'error')})), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
