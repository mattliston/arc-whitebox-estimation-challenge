"""Inert fixed-40 native terminal judge; run only with --judge and four seals.

This is reused public development evidence, NOT the independent 14-net NC
panel. No estimator, Torch, FNP, weights array, or fitting code is imported.
All training/native predictions and complete bills are authenticated and
sealed before opening any of the 40 validation truth files. The first cold
call is charged once; only call 40 (the repeated first net) is excluded.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import signal
import struct
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
P2 = NEW.parent
OFFPY = P2 / 'data_gate/.venv'
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
BUDGET = 2**41
VALID = (37, 39, 42, 48, 59, 63, 66, 67, 71, 73, 75, 79, 93, 101,
         122, 130, 132, 135, 139, 145, 146, 149, 150, 151, 155, 159,
         163, 169, 173, 175, 181, 192, 198, 199, 204, 219, 224, 227, 233, 234)
TRAIN = tuple(r for r in range(35, 235) if r not in VALID)
BANK = NEW / 'unused_materializations_v1/estimator.py'
BANK_SHA = '409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f'
TRUTH_RECEIPT = NEW / 'joint_prefix_final_judge_cached_v2/receipt.json'
TRUTH_RECEIPT_SHA = '4968610553b95011cabdc651dd65d4c9057054f9d2d2bf73711426fff695893c'
FIXED = {
    HERE / 'PROTOCOL.md': '9638ccee8e75e830f205b5cc8304d7333c7b202ae5441db3a71f610645f94212',
    HERE / 'NATIVE_SCOPE_CLARIFICATION.md': 'b43aa00adcc30f434f5e301abcd490deb220020096ea47daebcf908f420bd333',
    HERE / 'native.py': '7718591dd4324956fdcad52721009e7a9f7fb01c6969710b622376633ffbe075',
    HERE / 'train.py': 'b0c3c6ba6eadc5a333661a51b06e1636d6e2ee385ea096cc38f6833a204f91ab',
    HERE / 'model.py': 'cea1ea473db821cea6b4b9d34aeabf06f43ded8c5d77ee4037dd3f3d269df613',
    HERE / 'compare_ready.py': '7a29e743b815787382b9d04708fa9aba2f93b3a756f4ed9b09babf132181caf6',
    HERE / 'fnp/build_v2.py': 'cdbb295a2e598506a15778dbfeec9f6033c4ad23aaa442881da743c7a8b06365',
    HERE / 'fnp/core_v2.py': 'c31670320840545051ff7dc1031d71442cec99313423cff1e86ac62444b296f0',
    NEW / 'joint_prefix_final_data_v1.py': '22fff8465a13e412c68c1e5c204d7bfa0ce3d7c1cb9b99e1e5378098a9fd8b6b',
    NEW / 'payload_response_rows_v1.npz': '6b55c0d6071b2350e53e243bcd4137651e2c80a2cd55d04467bf194d0ea54a1c',
    BANK: BANK_SHA,
    TRUTH_RECEIPT: TRUTH_RECEIPT_SHA,
}
TIMINGS = ('wall_s', 'context_wall_s', 'flopscope_backend_s',
           'flopscope_overhead_s', 'native_residual_s')
COEFFICIENT_RULE = 'offline float32(math.tanh(float64(theta)))'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def float32_rows(value):
    require(isinstance(value, (list, tuple)) and len(value) == 15, 'Coefficient rows')
    result = []
    for row in value:
        require(isinstance(row, (list, tuple)) and len(row) == 4, 'Coefficient columns')
        for x in row:
            require(type(x) in (float, int) and math.isfinite(x), 'Finite real coefficient')
            result.append(struct.pack('<f', x))
    return b''.join(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--judge', action='store_true')
    for role in ('training', 'bank', 'zero', 'terminal'):
        parser.add_argument('--' + role + '-sha256', help=role + ' result.json SHA256')
    args = parser.parse_args()
    if not args.judge:
        parser.print_help()
        return 0
    seals = {role: getattr(args, role + '_sha256')
             for role in ('training', 'bank', 'zero', 'terminal')}
    require(all(isinstance(x, str) and len(x) == 64 and
                all(c in '0123456789abcdef' for c in x) for x in seals.values()),
            '--judge requires the four completed result SHA256 values')
    require(Path(sys.prefix).resolve() == OFFPY.resolve(), 'OFFPY only')
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'nice >= 10')
    require(all(os.environ.get(k) == '1' for k in
                ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')),
            'One-thread environment')
    require(importlib.metadata.version('numpy') == '2.4.6', 'Pinned NumPy')
    out = HERE / 'terminal_score_v1'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    result = dict(schema='birth_direction_terminal_native40_v1', passed=False,
                  completed=False, integrity_passed=False, started_utc=utc(),
                  requested_seals=seals, validation_ranks=list(VALID), records=[],
                  truth_accessed=False, truth_arrays_opened=False, independent_panel_qualified=False,
                  production_qualified=False, training_or_refit_performed=False)
    write(out / 'started.json', result)
    pins = {}
    truth_pins = {}
    truthpaths = {rank: P2 / 'public_g22_operator_features_v1/truthpost' /
                  f'truthpost_rank{rank:04d}.npz' for rank in VALID}
    validation_paths = {str(p.resolve()) for p in truthpaths.values()}

    def pin(path, expected=None):
        path = str(Path(path).resolve())
        require(path not in validation_paths, 'Validation truth in pre-score source bindings')
        digest = pins.get(path)
        if digest is None:
            digest = sha(path)
            pins[path] = digest
        require(expected is None or digest == expected, 'Hash mismatch: ' + path)
        return digest

    def bind(mapping):
        for path, value in mapping.items():
            if isinstance(value, dict):
                path, value = value['path'], value['sha256']
            pin(path, value)

    def resources():
        require(time.monotonic() - started < 300, 'Judge 300-second watchdog')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2e9, 'Judge RSS < 2GB')

    def alarm(_signal, _frame):
        raise TimeoutError('Judge hard 300-second watchdog')

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(300)
    try:
        import numpy as np
        for path, digest in FIXED.items():
            pin(path, digest)
        pin(Path(__file__))
        training_dir = HERE / 'training_v1'
        training_path = training_dir / 'result.json'
        pin(training_path, seals['training'])
        training = read(training_path)
        require(training['completed'] and training['hashes_unchanged'] and
                training['validation_targets_accessed'] is False, 'Completed target-blind terminal training')
        require(training['validation_ranks'] == list(VALID) and training['train_ranks'] == list(TRAIN),
                'Fixed TRAIN/VALID split')
        require((training['seed'], training['epochs'], training['updates'], training['batch_nets'],
                 training['allocated_parameters'], training['structurally_active_parameters']) ==
                (9309081, 6, 240, 4, 60, 56), 'Fixed fit settings')
        bind(training['sources'])
        for path in (HERE / 'train.py', HERE / 'model.py', HERE / 'PROTOCOL.md', HERE / 'ready_v1.json'):
            require(training['sources'].get(str(path)) == pin(path), 'Training source ownership: ' + str(path))
        ready = read(HERE / 'ready_v1.json')
        require(ready['passed'], 'Pre-fit numerical readiness')
        bind(ready['source_bindings'])
        terminal_path = training_dir / 'terminal_theta.json'
        pin(terminal_path, training['terminal_sha256'])
        terminal = read(terminal_path)
        require(terminal['terminal_epoch'] == 6 and terminal['coefficient_rule'] == COEFFICIENT_RULE,
                'Terminal coefficient convention')
        float32_rows(terminal['theta'])
        coefficients = [[math.tanh(float(x)) for x in row] for row in terminal['theta']]
        coefficient_bytes = float32_rows(coefficients)
        require(coefficient_bytes == float32_rows(terminal['coefficients']), 'Rounded deployment coefficients')
        require(all(x == 0 for x in terminal['theta'][0]), 'Structurally inactive first layer')
        pin(training_dir / 'epoch_06_audit_only.json')
        require(read(training_dir / 'epoch_06_audit_only.json')['theta'] == terminal['theta'],
                'Terminal is epoch 6, not checkpoint-selected')
        pin(training_dir / 'progress.jsonl')
        progress = [json.loads(line) for line in (training_dir / 'progress.jsonl').read_text().splitlines()]
        stages = [x for x in progress if x['stage'] in ('terminal_sealed', 'validation_prediction_only')]
        require(len(stages) == 41 and stages[0]['stage'] == 'terminal_sealed' and
                stages[0]['theta_sha256'] == training['terminal_sha256'] and
                [x['rank'] for x in stages[1:]] == list(VALID), 'Terminal precedes all fixed validation predictions')
        require(set(training['validation_outputs']) == {str(rank) for rank in VALID}, 'All 40 training predictions sealed')
        for rank in VALID:
            item = training['validation_outputs'][str(rank)]
            require(Path(item['path']).resolve() == training_dir / f'validation_{rank:04d}.npy', 'Training prediction path')
            pin(item['path'], item['sha256'])

        # Canonical weights are hashed, never decoded. All three roles must bind
        # the identical canonical row/weight files used by their actual runs.
        manifest_path = STORE / 'manifest.json'
        pin(manifest_path)
        manifest = read(manifest_path)['roles']['observer_fit']['row_manifests']
        fixtures = {}
        for rank in VALID:
            matches = [x for x in manifest if x['role_rank'] == rank]
            require(len(matches) == 1, 'Unique canonical row')
            rp = STORE / matches[0]['path']
            pin(rp, matches[0]['sha256'])
            row = read(rp)
            require(row['role_rank'] == rank and row['role'] == 'observer_fit', 'Canonical row identity')
            wp = rp.parent / 'weights.npy'
            spec = row['files']['weights.npy']
            require(spec['shape'] == [16, 1024, 1024] and spec['dtype'] == '<f4', 'Canonical weight contract')
            pin(wp, spec['sha256'])
            fixtures[str(rank)] = dict(row_path=str(rp), row_sha256=pin(rp),
                                      weights_path=str(wp), weights_sha256=pin(wp),
                                      mlp_seed=row['mlp_seed'])
        pin(training_dir / 'weight_bindings.json')
        trained_weights = read(training_dir / 'weight_bindings.json')
        require(len(trained_weights) == 200, 'All TRAIN/VALID training weight bindings')
        for fixture in fixtures.values():
            require(trained_weights.get(fixture['weights_path']) == fixture['weights_sha256'],
                    'Training/native validation weight identity')
        acquisitions = {}
        for role in ('bank', 'zero', 'terminal'):
            folder = HERE / f'native_validation_{role}_v1'
            rp = folder / 'result.json'
            pin(rp, seals[role])
            pin(folder / 'completion.json')
            end, acq = read(folder / 'completion.json'), read(rp)
            require(end['passed'] and end['result_sha256'] == seals[role] and
                    acq['passed'] and acq['hashes_unchanged'], 'Native completion: ' + role)
            require(acq['split'] == 'validation' and acq['ranks'] == list(VALID) and
                    acq['truth_accessed'] is False and acq['initialization_once'] and acq['repeat_excluded'],
                    'Native fixed acquisition contract: ' + role)
            source = BANK if role == 'bank' else HERE / f'build_{role}_v1/estimator.py'
            require(Path(acq['source']).resolve() == source, 'Native candidate path: ' + role)
            pin(source, acq['candidate_sha256'])
            bind(acq['source_bindings'])
            for p in (source, HERE / 'native.py', HERE / 'PROTOCOL.md',
                      HERE / 'NATIVE_SCOPE_CLARIFICATION.md', manifest_path):
                require(acq['source_bindings'].get(str(p)) == pin(p), 'Native source ownership')
            for fixture in fixtures.values():
                for key in ('row', 'weights'):
                    require(acq['source_bindings'].get(fixture[key + '_path']) == fixture[key + '_sha256'],
                            'Native canonical fixture ownership')
            if role != 'bank':
                bp = source.with_name('build.json')
                pin(bp)
                require(acq['source_bindings'].get(str(bp)) == pin(bp), 'Native build ownership')
                build = read(bp)
                require(build['passed'] and build['inverse_source_byte_exact'] and
                        build['all_other_parent_source_byte_exact'] and
                        build['imports_and_learned_P_payloads_unchanged'] and
                        build['parent_sha256'] == BANK_SHA and
                        build['candidate_sha256'] == acq['candidate_sha256'], 'Frozen-source build proof')
                bind(build['source_bindings'])
                require(Path(build['source_bindings']['core']['path']).resolve() == HERE / 'fnp/core_v2.py',
                        'FNP-left core version')
                literals = {node.targets[0].id: ast.literal_eval(node.value)
                            for node in ast.parse(source.read_text()).body
                            if isinstance(node, ast.Assign) and len(node.targets) == 1 and
                            isinstance(node.targets[0], ast.Name) and node.targets[0].id in
                            ('_BD_THETA_LITERAL', '_BD_COEFFICIENT_LITERAL')}
                expected = b'\0' * (15 * 4 * 4) if role == 'zero' else coefficient_bytes
                require(float32_rows(build['coefficient_literal']) == expected and
                        float32_rows(literals['_BD_COEFFICIENT_LITERAL']) == expected, 'Actual deployed coefficient bytes')
                require(build['coefficient_rule'] == COEFFICIENT_RULE, 'Build coefficient convention')
                if role == 'terminal':
                    require(build['theta_literal'] == terminal['theta'] and
                            [list(row) for row in literals['_BD_THETA_LITERAL']] == terminal['theta'] and
                            Path(build['source_bindings']['theta']['path']).resolve() == terminal_path and
                            build['source_bindings']['theta']['sha256'] == training['terminal_sha256'],
                            'Actual native terminal owns the sealed final training checkpoint')
            trials = acq['trials']
            require(len(trials) == 41 and [t['rank'] for t in trials] == list(VALID) + [VALID[0]], '40 plus fixed repeat')
            expected_artifacts = {}
            for call, trial in enumerate(trials):
                output = folder / f"call_{call:02d}_rank{trial['rank']}.npy"
                require(trial['call'] == call and Path(trial['output']).resolve() == output and
                        trial['includes_initialization'] is (call == 0), 'Call order/output/init ownership')
                require(type(trial['flops']) is int and 0 < trial['flops'] <= BUDGET and
                        trial['multiplier'] == max(.1, trial['flops'] / BUDGET), 'Complete native bill')
                require(all(math.isfinite(trial[k]) and trial[k] >= 0 for k in TIMINGS) and
                        trial['wall_s'] < 120 and 0 < trial['peak_rss_bytes'] < 8e9, 'Native resource gates')
                pin(output, trial['output_sha256'])
                expected_artifacts[str(output)] = trial['output_sha256']
                bill = folder / f'call_{call:02d}_bill.json'
                pin(bill)
                require(read(bill) == trial, 'Per-call bill/result identity')
            require({str(Path(p).resolve()): h for p, h in acq['artifacts'].items()} == expected_artifacts,
                    'Exact native artifact set')
            require(type(acq['initialization_flops']) is int and acq['initialization_flops'] >= 0 and
                    trials[0]['flops'] - trials[-1]['flops'] == acq['initialization_flops'] and
                    trials[0]['output_bytes_sha256'] == trials[-1]['output_bytes_sha256'], 'Initialization/repeat identity')
            acquisitions[role] = acq
            resources()

        def output(path, bytes_sha=None):
            array = np.load(path, allow_pickle=False)
            require(type(array) is np.ndarray and array.dtype == np.float32 and array.shape == (16, 1024) and
                    np.isfinite(array).all() and (array >= 0).all(), 'Concrete unchanged served-output contract')
            require(bytes_sha is None or hashlib.sha256(array.tobytes()).hexdigest() == bytes_sha,
                    'Served-byte hash')
            return array

        predictions = {role: [] for role in acquisitions}
        mps = []
        drift = []
        for call, rank in enumerate(VALID):
            for role, acq in acquisitions.items():
                trial = acq['trials'][call]
                predictions[role].append(output(trial['output'], trial['output_bytes_sha256']))
            require(predictions['zero'][-1].tobytes() == predictions['bank'][-1].tobytes(), 'Null/current all16 byte identity')
            require(predictions['terminal'][-1][0].tobytes() == predictions['bank'][-1][0].tobytes(), 'Exact unchanged layer0')
            mps.append(output(training['validation_outputs'][str(rank)]['path']))
            difference = mps[-1].astype(np.float64) - predictions['terminal'][-1].astype(np.float64)
            drift.append(dict(rank=rank, max_absolute=float(np.max(np.abs(difference))),
                              final_rms=float(np.sqrt(np.mean(difference[-1] ** 2)))))
        for role, acq in acquisitions.items():
            last = acq['trials'][-1]
            require(output(last['output'], last['output_bytes_sha256']).tobytes() == predictions[role][0].tobytes(),
                    'Loaded final repeat byte identity')
        require(all(sha(path) == digest for path, digest in pins.items()), 'All prediction/source seals before truth')
        seal = dict(utc=utc(), truth_accessed=False, source_bindings=dict(pins),
                    requested_result_seals=seals, terminal_sha256=training['terminal_sha256'],
                    fixture_bindings=fixtures, native_trials={role: acq['trials'] for role, acq in acquisitions.items()},
                    training_validation_outputs=training['validation_outputs'],
                    mps_native_drift_diagnostic=drift, initialization_once=True, repeat_excluded=True)
        write(out / 'predictions_and_bills_sealed.json', seal)
        result['predictions_and_bills_seal_sha256'] = sha(out / 'predictions_and_bills_sealed.json')
        result['all_predictions_verified_before_truth'] = True
        result['source_bindings'] = dict(pins)
        result['fixture_bindings'] = fixtures
        result['mps_native_drift_diagnostic'] = drift

        # The old judge's negative scientific verdict is not reused. Only its
        # already sealed 40 target hashes identify the unchanged 1e9 MC means.
        old_records = read(TRUTH_RECEIPT)['records']
        require(len(old_records) == 40 and {x['rank'] for x in old_records} == set(VALID), 'Fixed validation truth index')
        expected_truths = {x['rank']: x['target_sha256'] for x in old_records}
        result['truth_accessed'] = True  # Hashing starts only after the prediction/bill seal.
        for rank in VALID:
            path = truthpaths[rank]
            require(sha(path) == expected_truths[rank], 'Preexisting 1e9 truth hash: ' + str(rank))
            truth_pins[str(path)] = expected_truths[rank]
        write(out / 'truth_bindings.json', truth_pins)
        result['truth_arrays_opened'] = True
        for call, rank in enumerate(VALID):
            with np.load(truthpaths[rank], allow_pickle=False) as src:
                truth = src['mc_post'].copy()
            require(truth.shape == (16, 1024) and truth.dtype.kind == 'f' and np.isfinite(truth).all(), 'Truth mean contract')
            target = truth.astype(np.float64)
            row = dict(rank=rank, per_role={}, raw_ratios={}, adjusted_ratios={}, raw_wins={}, adjusted_wins={})
            for role, acq in acquisitions.items():
                layer_mse = np.mean((predictions[role][call].astype(np.float64) - target) ** 2, axis=1)
                raw = float(layer_mse[-1])
                multiplier = acq['trials'][call]['multiplier']
                require(math.isfinite(raw) and raw >= 0, 'Finite raw MSE')
                row['per_role'][role] = dict(raw_mse=raw, adjusted=raw * multiplier,
                                            flops=acq['trials'][call]['flops'], multiplier=multiplier,
                                            layer_mse=layer_mse.tolist())
            for comparator in ('bank', 'zero'):
                a, b = row['per_role']['terminal'], row['per_role'][comparator]
                require(b['raw_mse'] > 0 and b['adjusted'] > 0, 'Nonzero comparator denominator')
                row['raw_ratios'][comparator] = a['raw_mse'] / b['raw_mse']
                row['adjusted_ratios'][comparator] = a['adjusted'] / b['adjusted']
                row['raw_wins'][comparator] = a['raw_mse'] < b['raw_mse']
                row['adjusted_wins'][comparator] = a['adjusted'] < b['adjusted']
            row['mps_final_mse_diagnostic'] = float(np.mean((mps[call][-1].astype(np.float64) - target[-1]) ** 2))
            row['truth_sha256'] = expected_truths[rank]
            result['records'].append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
            resources()
        summary = {role: dict(
            mean_raw_mse=math.fsum(x['per_role'][role]['raw_mse'] for x in result['records']) / 40,
            mean_adjusted=math.fsum(x['per_role'][role]['adjusted'] for x in result['records']) / 40,
            mean_complete_flops=sum(x['per_role'][role]['flops'] for x in result['records']) / 40,
            initialization_flops=acq['initialization_flops']) for role, acq in acquisitions.items()}
        comparisons = {}
        for role in ('bank', 'zero'):
            wins = sum(x['adjusted_wins'][role] for x in result['records'])
            ratio = summary['terminal']['mean_adjusted'] / summary[role]['mean_adjusted']
            comparisons[role] = dict(mean_adjusted_ratio=ratio,
                                     mean_raw_ratio=summary['terminal']['mean_raw_mse'] / summary[role]['mean_raw_mse'],
                                     adjusted_wins=wins, raw_wins=sum(x['raw_wins'][role] for x in result['records']),
                                     objective_improved=ratio < 1, wins_gate=wins >= 24,
                                     passed=ratio < 1 and wins >= 24)
        require(all(sha(path) == digest for path, digest in pins.items()), 'Source/artifact end seals')
        require(all(sha(path) == digest for path, digest in truth_pins.items()), 'Truth end seals')
        require(sha(out / 'predictions_and_bills_sealed.json') == result['predictions_and_bills_seal_sha256'], 'Pre-truth seal unchanged')
        resources()
        advances = all(x['passed'] for x in comparisons.values())
        result.update(completed=True, integrity_passed=True, passed=advances,
                      advances_to_independent14=advances, summary=summary, comparisons=comparisons,
                      source_and_artifact_hashes_unchanged=True, truth_bindings=truth_pins,
                      rule='Both comparators: strict mean of per-net observed final MSE * max(.1, complete F/2**41), and >=24/40 adjusted wins',
                      truth_scope='Reused public VALID40 existing 1e9 mc_post means; observed MSE, no noise correction. Not the final independent14 NC panel.',
                      deployment_drift_scope='MPS/native drift reported only; native outputs and own bills define this actual-objective judgment.')
    except BaseException:
        result['error'] = traceback.format_exc()
        result['source_bindings'] = dict(pins)
        result['truth_bindings'] = dict(truth_pins)
    finally:
        signal.alarm(0)
        result.update(finished_utc=utc(), wall_seconds=time.monotonic() - started,
                      peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        write(out / 'result.json', result)
        write(out / 'completion.json', dict(completed=result['completed'], passed=result['passed'],
                                            result_sha256=sha(out / 'result.json')))
    print(json.dumps({k: result.get(k) for k in ('completed', 'passed', 'comparisons', 'error')}, allow_nan=False), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
