"""Frozen CAP native40 objective only; inert unless --run is explicit.

No estimator, Torch, FlopScope, subprocess, network, fitting or new predictions.
The failed native/MPS qualification stays failed. All native ownership checks
remain hard gates; only the native estimator's actual objective is adjudicated.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
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
NEW = HERE.parent.parent
P2 = NEW.parent
CAP = NEW / 'current_paircap4096_e2e_v1'
BD = NEW / 'birth_direction_fit_v1'
OFFPY = P2 / 'data_gate/.venv'
TRUTH_DIR = P2 / 'public_g22_operator_features_v1/truthpost'
TRUTH_INDEX = NEW / 'joint_prefix_final_judge_cached_v2/receipt.json'
TRUTH_INDEX_SHA = '4968610553b95011cabdc651dd65d4c9057054f9d2d2bf73711426fff695893c'
BUDGET = 2**41
VALID = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
         139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,
         204,219,224,227,233,234)
TRAIN_SHA = '5499433f54ed648c66716cfd9044fa7908c6024f9bb20012f6c091d56f8500d8'
BUILD_SHA = '027186a362bb487a76f993b7c9b9a9c68c4be1157f7f05cd1aa8e403219ff4a1'
NULL_SHA = 'd6dc6bb47d0c1f047ae34c1fb37ff624ccb957e20074447725abe7b7c21fcbf4'
TERMINAL_SHA = '604d23a2118cfd7781ff3a51b4adc9d9125c586b051a9cc797c23a0573735a6a'
D50_SHA = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'
ROLES = {
    'd50': (BD/'validation_terminal_v2',
        'ef7f4fcd8c6bc9275e434560ac5f514fd5f99d94077441d757ad34fb3a8df17d',
        BD/'build_terminal_v1/estimator.py', D50_SHA, 1383913786),
    'null': (CAP/'native_validation_null_v1',
        '3672e8ac2d628b8981c0bdbbdd5d55f2db49df21a4dfba68139fec8ad1e51f01',
        CAP/'estimator.py', NULL_SHA, 1383913666),
    'terminal': (CAP/'native_validation_terminal_v1',
        'ad5107badc63f81630595db4f5d6058f1ee44cc2c6be44b3af38147308f1c598',
        CAP/'build_terminal_v1/estimator.py', TERMINAL_SHA, 1383913666),
}
FIXED = {
    CAP/'common.py': '4716f96444ec81ad3e3463b670b2ceacad6ec6c93ecd8fec92af42d1a6ee5242',
    CAP/'build_terminal.py': '5a6a38b0c42a77431302c185b77e359db35c8c597c22564bddba15916a07fc0d',
    CAP/'native40.py': '253c45d7e7e9329d5e0abcc0732857e75f2010d9bb182fd75545f0ae0d897d9d',
    CAP/'judge_terminal.py': '3f5c7538c0fe18b4a1ccad0cf7081e3965ce2b1c806a8cdda41a7f1056ada796',
    CAP/'build_terminal_v1/build.json': BUILD_SHA,
    CAP/'training_v1/result.json': TRAIN_SHA,
    CAP/'training_v1/completion.json': '0d136d0a2d6195b651418b2cb665615b488102f15e5a31317de6367cb5eda8a4',
    CAP/'terminal_score_v1/result.json': '7f07b494eed79a512e716cfdd3a4134fb1845ec29bba242899056b843730abce',
    CAP/'terminal_score_v1/completion.json': '7a632c54f5fc685cb5b45d64815071bec92eba3e93c806a343f71a973dec352f',
    CAP/'native_validation_null_v1/completion.json': 'd7a82c4c55a5e4f435ade9f2be8eee36dedf9d6a3591999e5b67dfb7fb29cd42',
    CAP/'native_validation_terminal_v1/completion.json': '34ba49df6ca51392306f12c7fc4b2353739e60ef8ce278c25acb53fb33bf92bc',
    BD/'validation_terminal_v2/completion.json': 'fbbe37f97f39210f9e60a9dc3eeaa2d647529cfef875f72fb31942a7e724a079',
    BD/'qualify_native_v2.py': '5528601148c98384de72af4ea7f7f39eed1d4657c51617910cd2be466299a608',
    BD/'native_v2.py': '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b',
    BD/'build_terminal_v1/build.json': '4889c6246e8e0dd420cdb8fbca3ccbce721a163befd226339ebe6d0023cc070f',
    TRUTH_INDEX: TRUTH_INDEX_SHA,
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def guard(start):
    require(time.monotonic()-start < 60, 'Fixed offline wall <60s')
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2_000_000_000,
            'Offline RSS <2GB; external watcher also bounds the complete tree')


def module(path, name):
    """Only the explicitly pinned, standard-library-only metadata modules."""
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def bind(mapping, pin):
    for path, value in mapping.items():
        if isinstance(value, dict):
            path, value = value['path'], value['sha256']
        pin(path, value)


def replace_exact(source, edits):
    changed = source
    for before, after in edits:
        require(changed.count(before) == 1, 'Unique admission edit')
        changed = changed.replace(before, after)
    inverse = changed
    for before, after in reversed(edits):
        require(inverse.count(after) == 1, 'Unique admission inverse')
        inverse = inverse.replace(after, before)
    require(inverse == source, 'All other native-verifier source byte exact')
    return changed


def verifier(v2):
    """Keep every native check; change only its candidate and init mappings."""
    source = (BD/'qualify_native_v2.py').read_text()
    nodes = [n for n in ast.parse(source).body
             if isinstance(n, ast.FunctionDef) and n.name == 'verify_native']
    require(len(nodes) == 1, 'Unique preserved verifier')
    node = nodes[0]
    original = ''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    changed = replace_exact(original, [
        ("    expected_sha = BANK_SHA if source == BANK else {HERE / 'build_zero_v1/estimator.py': ZERO_SHA,\n"
         "                       HERE / 'build_terminal_v1/estimator.py': TERMINAL_SHA}.get(source)",
         '    expected_sha = CANDIDATES.get(source)'),
        ('    expected_init = 1383913666 + (0 if source == BANK else 120)',
         '    expected_init = INITIALIZATIONS[source]'),
    ])
    env = dict(vars(v2), CANDIDATES={v[2]: v[3] for v in ROLES.values()},
               INITIALIZATIONS={v[2]: v[4] for v in ROLES.values()})
    exec(compile(changed, str(__file__)+':two_admission_edits', 'exec'), env)
    return env['verify_native'], dict(source_inverse_byte_exact=True, admission_edits=2,
        original_function_sha256=hashlib.sha256(original.encode()).hexdigest(),
        adapted_function_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        numeric_input_bill_reset_checks_unchanged=True)


def admissions(pin):
    for path, digest in FIXED.items():
        pin(path, digest)
    for folder, digest, source, source_sha, _ in ROLES.values():
        pin(folder/'result.json', digest)
        pin(source, source_sha)

    # Resolve the builder's one absolute `from common` import to its authentic
    # sibling, never a same-named unrelated module. Only its metadata APIs run.
    common = module(CAP/'common.py', 'cap_native_objective_frozen_common')
    common.sha = pin  # Streaming, target-blocked, memoized hashes; full rehash at end.
    sentinel = object()
    previous = sys.modules.get('common', sentinel)
    sys.modules['common'] = common
    try:
        builder = module(CAP/'build_terminal.py', 'cap_native_objective_frozen_builder')
    finally:
        if previous is sentinel:
            del sys.modules['common']
        else:
            sys.modules['common'] = previous
    bind(builder.current_readiness(), pin)
    training, training_pins = builder.training_admission(TRAIN_SHA)
    bind(training_pins, pin)
    require(builder.definitions()[1] == VALID, 'Unchanged exact VALID40')
    require(training['completed'] and training['hashes_unchanged'] and
            not training['validation_targets_accessed'] and not training['checkpoint_selection'],
            'Frozen epoch-six training and pre-target terminal seals')
    build = read(CAP/'build_terminal_v1/build.json')
    require(build['passed'] and build['schema'] == 'current_paircap4096_e2e_terminal_build_v1'
            and build['parent_sha256'] == NULL_SHA and build['candidate_sha256'] == TERMINAL_SHA
            and Path(build['candidate']).resolve() == ROLES['terminal'][2]
            and build['training_result_sha256'] == TRAIN_SHA and build['trainer_sha256'] == builder.TRAINER_SHA
            and build['terminal_payload'] == training['terminal_payload']
            and build['terminal_epoch'] == 6 and build['updates'] == 240
            and build['all_40_validation_predictions_sealed'] and not build['validation_targets_accessed'],
            'Exact frozen CAP terminal build')
    bind(build['source_bindings'], pin)
    for key in ('inverse_source_byte_exact', 'all_other_parent_source_byte_exact',
                'original_literal_matches_original_payload', 'terminal_literal_roundtrip_byte_exact',
                'non_PKEYS_byte_exact', 'unused_layer0_byte_exact'):
        require(build[key] is True, 'Recorded terminal source proof: '+key)
    proof = builder.inverse_check(ROLES['null'][2].read_bytes(), ROLES['terminal'][2].read_bytes())
    require(all(build[key] == value for key, value in proof.items()), 'Recomputed literal-only source inverse')
    d50_build = read(BD/'build_terminal_v1/build.json')
    require(d50_build['passed'] and d50_build['candidate_sha256'] == D50_SHA
            and d50_build['inverse_source_byte_exact'] and d50_build['all_other_parent_source_byte_exact']
            and d50_build['imports_and_learned_P_payloads_unchanged'], 'Unchanged frozen d50 build')
    bind(d50_build['source_bindings'], pin)

    failed = read(CAP/'terminal_score_v1/result.json')
    failed_end = read(CAP/'terminal_score_v1/completion.json')
    require(failed['passed'] is False and failed['completed'] is False and not failed['truth_accessed']
            and failed_end['passed'] is False
            and failed_end['result_sha256'] == FIXED[CAP/'terminal_score_v1/result.json']
            and 'Terminal native/MPS full-size fidelity' in failed['error'],
            'Preserve original failed pre-truth numerical-bridge verdict')
    for key, expected in (('training_sha256', TRAIN_SHA), ('build_sha256', BUILD_SHA),
                          ('null_sha256', ROLES['null'][1]), ('terminal_sha256', ROLES['terminal'][1]),
                          ('source_sha256', FIXED[CAP/'judge_terminal.py'])):
        require(failed['requested_seals'][key] == expected, 'Original failure belongs to these frozen inputs')

    v2 = module(BD/'qualify_native_v2.py', 'cap_native_objective_preserved_verifier')
    require(v2.VALID == VALID and v2.BUDGET == BUDGET, 'Original ranks and complete budget')
    v2.verify_meter(pin)
    verify, verifier_proof = verifier(v2)
    return builder, training, failed, verify, verifier_proof


def analyze(args, out, result, start):
    pins, truth_pins = {}, {}
    result['source_bindings'] = pins
    result['truth_bindings'] = truth_pins

    def pin(path, expected=None):
        path = Path(path).resolve()
        require(TRUTH_DIR not in path.parents and '/truthpost/' not in str(path)
                and 'mps_truth_' not in path.name, 'No target access through source/artifact admission')
        key = str(path)
        if key not in pins:
            pins[key] = sha(path)
        require(expected is None or pins[key] == expected, 'Pin mismatch: '+key)
        guard(start)
        return pins[key]

    pin(__file__, args.source_sha256)
    pin(HERE/'PROTOCOL.md', args.protocol_sha256)
    builder, training, failed, verify, proof = admissions(pin)
    result['verifier_source_proof'] = proof
    acquisitions, native_bytes = {}, {}
    for role, (folder, digest, source, source_sha, _) in ROLES.items():
        acq, completion = read(folder/'result.json'), read(folder/'completion.json')
        require(acq['passed'] and completion['passed'] and completion['result_sha256'] == digest
                and Path(acq['source']).resolve() == source and acq['candidate_sha256'] == source_sha,
                'Completed sealed native role: '+role)
        if role != 'd50':
            require(acq['schema'] == 'current_paircap4096_e2e_native40_v1' and acq['role'] == role
                    and acq['own_complete_bills'] and acq['all_outputs_and_bills_sealed_before_scoring']
                    and not acq['state_instrumentation'] and not acq['score_measured']
                    and acq['source_bindings'][str(CAP/'native40.py')] == FIXED[CAP/'native40.py'],
                    'Frozen complete CAP acquisition')
            pre = acq['prerequisites']
            require(pre['fixed_validation_ranks'] == list(VALID) and pre['role'] == role
                    and pre['parity_result_sha256'] == builder.PARITY_SHA
                    and pre['initial_native_result_sha256'] == builder.NATIVE_SHA
                    and pre['truth_accessed'] is False, 'Original fixed CAP admission')
            if role == 'terminal':
                require(pre['completed_fit_required'] and pre['training_result_sha256'] == TRAIN_SHA
                        and pre['build_sha256'] == BUILD_SHA
                        and pre['terminal_payload_sha256'] == training['terminal_payload']['sha256'],
                        'Actual terminal acquisition owns the exact frozen fit')
            else:
                require(pre['completed_fit_required'] is False, 'Original no-fit null admission')
        native_bytes[role] = verify(acq, folder, VALID, pin)
        acquisitions[role] = acq
    pools = [acq['threadpools'] for acq in acquisitions.values()]
    require(all(isinstance(p, list) for p in pools) and all(p == pools[0] for p in pools)
            and all(p['num_threads'] == 1 for p in pools[0] if p.get('user_api') == 'blas'),
            'Same reported native thread frame; empty inventory is not measured BLAS attestation')
    require(all(len(values) == 41 for values in native_bytes.values()), 'All123 calls independently verified')
    result['native_ownership_verified'] = True

    import numpy as np  # Only tiny frozen outputs/payloads/truths, no model import.
    require(np.__version__ == '2.4.6', 'Frozen OFFPY offline scoring NumPy')
    result['environment']['numpy'] = np.__version__

    def output(path, meta):
        pin(path, meta['sha256'])
        value = np.load(path, allow_pickle=False)
        require(type(value) is np.ndarray and value.dtype == np.float32 and value.shape == (16,1024)
                and np.isfinite(value).all() and (value >= 0).all()
                and hashlib.sha256(value.tobytes()).hexdigest() == meta['raw_sha256'],
                'Frozen finite all16 FP32 shadow output')
        return value

    # Recheck the literal against exact epoch-six arrays; do not execute it.
    _, _, literal = builder.assignment_span(ROLES['terminal'][2].read_bytes())
    audit_path = CAP/'training_v1/epoch_06_audit_only.npz'
    pin(audit_path, training['audit_checkpoints'][-1]['sha256'])
    payload = training['terminal_payload']
    with np.load(payload['path'], allow_pickle=False) as terminal, np.load(audit_path, allow_pickle=False) as audit:
        require(set(terminal.files) == set(audit.files) == set(literal) == set(payload['arrays']), 'Full terminal key set')
        for key in terminal.files:
            value, old = terminal[key], audit[key]
            meta = payload['arrays'][key]
            decoded = np.asarray(literal[key], dtype=value.dtype)
            require(str(value.dtype) == meta['dtype'] and list(value.shape) == meta['shape']
                    and np.isfinite(value).all() and hashlib.sha256(value.tobytes()).hexdigest() == meta['raw_sha256']
                    and value.dtype == old.dtype and value.shape == old.shape and value.tobytes() == old.tobytes()
                    and decoded.shape == value.shape and decoded.tobytes() == value.tobytes(),
                    'Exact terminal source/payload/epoch-six bytes: '+key)
    del literal

    drift = []
    for call, rank in enumerate(VALID):
        preds = {role: np.frombuffer(values[call], dtype='<f4').reshape(16,1024)
                 for role, values in native_bytes.items()}
        require(all(preds[role][0].tobytes() == preds['d50'][0].tobytes() for role in ('null','terminal')),
                'All roles preserve Gaussian layer0 bytes')
        shadow = output(training['validation_outputs'][str(rank)]['path'], training['validation_outputs'][str(rank)])
        delta = shadow.astype(np.float64)-preds['terminal'].astype(np.float64)
        maximum = float(np.max(np.abs(delta)))
        drift.append(dict(rank=rank, max_absolute=maximum,
            final_rms=float(np.sqrt(np.mean(delta[-1]**2))), all16_gate_passed=maximum <= 1e-5))
    require(not all(row['all16_gate_passed'] for row in drift), 'Reproduce, do not relabel, failed output-fidelity diagnostic')
    result['mps_fidelity'] = dict(original_result_sha256=FIXED[CAP/'terminal_score_v1/result.json'],
        original_completed=False, original_passed=False, original_truth_accessed=False,
        original_output_fidelity_status='FAILED', original_mse_fidelity_status='NOT_REACHED',
        all16_gate=1e-5, output_comparisons=drift, output_fidelity_passed=False,
        derivative_qualified=False, diagnostic_only_in_this_native_objective_audit=True)

    # Every prediction, payload, source, weight and bill is pinned before any
    # numerical truth read. Targets cannot pass through pin() above.
    for path, digest in pins.items():
        require(sha(path) == digest, 'Pre-truth source/artifact unchanged: '+path)
        guard(start)
    seal_path = out/'predictions_and_bills_sealed.json'
    write(seal_path, dict(created_utc=utc(), source_bindings=dict(pins), truth_accessed=False,
        roles={role: dict(result_sha256=ROLES[role][1], candidate_sha256=ROLES[role][3],
            trials=acq['trials'], initialization_flops=acq['initialization_flops'])
            for role, acq in acquisitions.items()},
        terminal_payload=payload, validation_outputs=training['validation_outputs'],
        verifier_source_proof=proof, mps_fidelity=result['mps_fidelity'],
        fixed_valid=list(VALID), initialization_once=True, repeat_excluded=True))
    pretruth_sha = sha(seal_path)
    result.update(all_predictions_verified_before_truth=True, predictions_and_bills_seal_sha256=pretruth_sha,
                  artifacts={str(seal_path): pretruth_sha})

    records = read(TRUTH_INDEX)['records']
    require(len(records) == 40 and {row['rank'] for row in records} == set(VALID), 'Fixed1e9 VALID40 truth index')
    expected = {row['rank']: row['target_sha256'] for row in records}
    for rank in VALID:
        path = TRUTH_DIR/f'truthpost_rank{rank:04d}.npz'
        require(sha(path) == expected[rank], 'Authenticated fixed truth bytes')
        truth_pins[str(path)] = expected[rank]
        guard(start)
    write(out/'truth_bindings.json', truth_pins)
    result['artifacts'][str(out/'truth_bindings.json')] = sha(out/'truth_bindings.json')
    rows = []
    result['records'] = rows  # Preserve every completed row even if a later guard fails.
    for call, rank in enumerate(VALID):
        result['truth_accessed'] = True
        with np.load(TRUTH_DIR/f'truthpost_rank{rank:04d}.npz', allow_pickle=False) as source:
            target = source['mc_post'].astype(np.float64)
        require(target.shape == (16,1024) and np.isfinite(target).all(), 'Finite unchanged mc_post target')
        row = dict(rank=rank, call=call, truth_sha256=expected[rank], per_role={},
                   adjusted_wins={}, raw_wins={}, adjusted_ratios={}, raw_ratios={})
        for role, acq in acquisitions.items():
            pred = np.frombuffer(native_bytes[role][call], dtype='<f4').reshape(16,1024)
            layer_mse = np.mean((pred.astype(np.float64)-target)**2, axis=1)
            raw, flops = float(layer_mse[-1]), acq['trials'][call]['flops']
            factor = max(.1, flops/BUDGET)
            require(math.isfinite(raw) and raw >= 0, 'Finite actual native objective')
            row['per_role'][role] = dict(raw_mse=raw, adjusted=raw*factor, flops=flops,
                multiplier=factor, includes_initialization=(call == 0), layer_mse=layer_mse.tolist())
        native_raw = row['per_role']['terminal']['raw_mse']
        shadow = output(training['validation_outputs'][str(rank)]['path'], training['validation_outputs'][str(rank)])
        mps_raw = float(np.mean((shadow[-1].astype(np.float64)-target[-1])**2))
        require(math.isfinite(mps_raw) and mps_raw >= 0, 'Finite diagnostic MPS raw')
        mse_ratio = mps_raw/native_raw if native_raw > 0 else None
        row.update(mps_raw_mse=mps_raw, mps_native_mse_ratio=mse_ratio,
                   mps_mse_fidelity_passed=(mse_ratio is not None and .99 <= mse_ratio <= 1.01))
        for control in ('d50','null'):
            base = row['per_role'][control]
            require(base['raw_mse'] > 0 and base['adjusted'] > 0, 'Positive comparator for paired ratios')
            row['adjusted_wins'][control] = row['per_role']['terminal']['adjusted'] < base['adjusted']
            row['raw_wins'][control] = native_raw < base['raw_mse']
            row['adjusted_ratios'][control] = row['per_role']['terminal']['adjusted']/base['adjusted']
            row['raw_ratios'][control] = native_raw/base['raw_mse']
        rows.append(row)
        guard(start)
        print(json.dumps(dict(rank=rank, per_role={role: {k: value[k] for k in ('raw_mse','adjusted','flops')}
                         for role, value in row['per_role'].items()})), flush=True)
    summary = {role: dict(mean_raw_mse=math.fsum(r['per_role'][role]['raw_mse'] for r in rows)/40,
        mean_adjusted=math.fsum(r['per_role'][role]['adjusted'] for r in rows)/40,
        mean_complete_flops=sum(r['per_role'][role]['flops'] for r in rows)/40,
        initialization_flops=acquisitions[role]['initialization_flops'], scored_primary_calls=40,
        actual_total_flops_including_repeat=sum(t['flops'] for t in acquisitions[role]['trials']))
        for role in ROLES}
    comparisons = {}
    for control in ('d50','null'):
        ratio = summary['terminal']['mean_adjusted']/summary[control]['mean_adjusted']
        wins = sum(row['adjusted_wins'][control] for row in rows)
        comparisons[control] = dict(mean_adjusted_ratio=ratio,
            mean_raw_ratio=summary['terminal']['mean_raw_mse']/summary[control]['mean_raw_mse'],
            adjusted_wins=wins, raw_wins=sum(row['raw_wins'][control] for row in rows),
            mean_adjusted_improves=ratio < 1, passes_24_wins=wins >= 24,
            objective_and_wins_passed=ratio < 1 and wins >= 24)
    for mapping in (pins, truth_pins, result['artifacts']):
        for path, digest in mapping.items():
            require(sha(path) == digest, 'Final authenticated bytes unchanged: '+path)
            guard(start)
    require(len(rows) == 40 and sha(seal_path) == pretruth_sha, 'Complete fixed judgment and unchanged pre-truth seal')
    require(not any(name in sys.modules for name in ('torch','flopscope','whestbench')), 'No model/backend imports')
    result['mps_fidelity'].update(mse_gate_relative=.01,
        mse_fidelity_passed=all(row['mps_mse_fidelity_passed'] for row in rows),
        mse_diagnostic_was_previously_unreached=True)
    result.update(summary=summary, comparisons=comparisons, integrity_passed=True, hashes_unchanged=True,
        advances_to_independent14=comparisons['d50']['objective_and_wins_passed'],
        smaller_gain_component_only=(comparisons['d50']['mean_adjusted_improves']
                                     and not comparisons['d50']['objective_and_wins_passed']),
        aggregation='Mean per-net final observed MSE * max(.1, own complete F/2^41); actual init once; repeat excluded',
        frame='Reused development VALID40, observed1e9 mc_post; not independent14 or official leaderboard',
        actual_native_calls_verified=123, new_native_calls=0, passed=True, completed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--source-sha256')
    parser.add_argument('--protocol-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    require(__debug__, 'Assertions in frozen metadata verifiers must remain enabled')
    require(all(isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v)
                for v in (args.source_sha256, args.protocol_sha256)), 'Reviewed source/protocol pins required')
    require(sys.platform == 'darwin' and Path(sys.prefix).resolve() == OFFPY.resolve()
            and os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'OFFPY nice10; macOS RSS byte convention')
    env = {key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}
    require(all(value == '1' for value in env.values()), 'All three launch BLAS controls are one')
    require(not any(name in sys.modules for name in ('numpy','torch','flopscope','whestbench')), 'Authenticate before numerical imports')
    sys.dont_write_bytecode = True  # Metadata imports must not write into frozen directories.
    os.chdir(NEW)  # Preserved readiness artifacts are relative to this exact base.
    out = HERE/'analysis_v2'
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    result = dict(schema='frozen_cap_native_objective_v2', passed=False, completed=False,
        hashes_unchanged=False, integrity_passed=False, truth_accessed=False, refit_performed=False,
        training_shadow_qualified=False, production_qualified=False, official_score=False,
        source_sha256=args.source_sha256, protocol_sha256=args.protocol_sha256,
        environment=dict(sys_prefix=sys.prefix, nice=os.getpriority(os.PRIO_PROCESS, 0), launch_env=env),
        started_utc=utc(), fixed_valid=list(VALID))
    write(out/'started.json', result)

    def expired(_signum, _frame):
        raise TimeoutError('Fixed offline objective audit 60-second alarm')

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(60)
    try:
        analyze(args, out, result, start)
        guard(start)
    except BaseException:
        result.update(passed=False, completed=False, hashes_unchanged=False, integrity_passed=False,
                      advances_to_independent14=False, error=traceback.format_exc())
    finally:
        signal.alarm(0)
        result.update(finished_utc=utc(), wall_s=time.monotonic()-start,
                      peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        write(out/'result.json', result)
        write(out/'completion.json', dict(passed=result['passed'], completed=result['completed'],
            hashes_unchanged=result['hashes_unchanged'], result_sha256=sha(out/'result.json')))
    print(json.dumps({key: result.get(key) for key in ('passed','summary','comparisons','advances_to_independent14','error')}), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
