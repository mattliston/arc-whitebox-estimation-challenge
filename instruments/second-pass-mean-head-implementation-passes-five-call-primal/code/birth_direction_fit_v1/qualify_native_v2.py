"""Inert metadata/served-byte qualification of the two native-v2 repairs.

No model, FNP, NumPy, weights array, truth, or fitting code is imported.
Use --qualify plus both completed development result SHA256 values. Fixed
bank/zero35,100,35 must reproduce old bytes and measured +179306496 F/call.
The separate 16777216-F official-input copy is NOT participant work. This
constant is a qualification assertion, never a substitute for a fresh bill.
Reusable verifiers also authenticate all v2 lifecycle artifacts and only
the recorded qualified overlaps of the two preserved failed v1 acquisitions.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
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
NEW, P2 = HERE.parent, HERE.parent.parent
WORKSPACE = P2.parents[2]
OFFPY = P2 / 'data_gate/.venv'
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
BANK = NEW / 'unused_materializations_v1/estimator.py'
BANK_SHA = '409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f'
ZERO_SHA = '300dbc660b3f2334235b523dc493452b1062f9dab4f80bf00f011acf892faff2'
TERMINAL_SHA = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'
NATIVE_SHA = '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
DELTA_F, INPUT_F, BUDGET = 179306496, 16777216, 2**41
TIMINGS = ('wall_s', 'context_wall_s', 'flopscope_backend_s',
           'flopscope_overhead_s', 'native_residual_s')
VALID = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
         139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,
         204,219,224,227,233,234)
FIXED = {
    HERE / 'native_v2.py': NATIVE_SHA,
    HERE / 'native.py': '7718591dd4324956fdcad52721009e7a9f7fb01c6969710b622376633ffbe075',
    HERE / 'judge_terminal.py': '4f4ccb45d75f52c36530cc1266847334c84014ea60eb25d933d03aa933420c0b',
    HERE / 'PROTOCOL.md': '9638ccee8e75e830f205b5cc8304d7333c7b202ae5441db3a71f610645f94212',
    HERE / 'NATIVE_INPUT_AND_LOG_LIFECYCLE_V2.md': 'c08620d45fbcd93385f26c882fbc6c56e988673a19fe89484bc66dcdd2ba5dfa',
    HERE / 'NATIVE_SCOPE_CLARIFICATION.md': 'b43aa00adcc30f434f5e301abcd490deb220020096ea47daebcf908f420bd333',
    HERE / 'check_meter_lifecycle_v2.py': '7211c965b7e63535689da03cf0d1ba57a428bf0f07458b9be887e4fe3d0d65c9',
    HERE / 'meter_lifecycle_v2/result.json': '2298d30b0002df4d60b935bbbb11b9079b17c913fcc6eac7e713f2614e0bf206',
    HERE / 'meter_lifecycle_v2/completion.json': 'f2aac909bea91cb33c096c4944f40197c752e467b80a8572157968dea049d003',
    HERE / 'native_zero_v1/result.json': 'ab32be9a25cd23506a627ac594b0ac9f0395ababc3179f711ba4d729c67d55eb',
    NEW / 'unused_materializations_v1/native/result.json': 'a9ccecaa3c0ca79f070af3c0961a6a952f08f4d9712e82a3d6af4d9a0d1d0951',
    BANK: BANK_SHA,
    HERE / 'build_zero_v1/estimator.py': ZERO_SHA,
    HERE / 'build_terminal_v1/estimator.py': TERMINAL_SHA,
    STORE / 'manifest.json': '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50',
}
FAILED_V1 = {
    'bank': ('163e6ef4e1669860629a111ad64ac3486d48b35542e72b559f24df33aa87f6b2', 25),
    'terminal': ('9de3efa4e4141efe6ce44f978842c2e267b7f74c83856036b76e02e499a39764', 28),
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def path(value):
    p = Path(value)
    if p.is_absolute():
        return p.resolve()
    if p.parts[0] == 'problem_graph':
        return (WORKSPACE / p).resolve()
    if p.parts[0] == HERE.name:
        return (NEW / p).resolve()
    found = {(root / p).resolve() for root in (HERE, NEW, WORKSPACE) if (root / p).exists()}
    require(len(found) == 1, 'Unambiguous preserved relative path: ' + str(value))
    return found.pop()


def sha(p):
    h = hashlib.sha256()
    with path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(path(p).read_text())


def write(p, value):
    with Path(p).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def bind(mapping, pin):
    for p, value in mapping.items():
        if isinstance(value, dict):
            p, value = value['path'], value['sha256']
        pin(path(p), value)


def sealed(folder, expected, pin, passed=True):
    rp, cp = folder / 'result.json', folder / 'completion.json'
    pin(rp, expected)
    pin(cp)
    result, completion = read(rp), read(cp)
    require(completion['result_sha256'] == expected
            and result['passed'] is passed and completion['passed'] is passed,
            'Exact completion/result identity: ' + str(folder))
    return result


def payload(p, expected, pin):
    """Read only the tiny served-output file; no NumPy array decoder/import."""
    p = path(p)
    pin(p)
    raw = p.read_bytes()
    require(raw[:6] == b'\x93NUMPY' and raw[6:8] in (b'\x01\x00', b'\x02\x00'), 'NPY output header')
    size = 2 if raw[6] == 1 else 4
    length = int.from_bytes(raw[8:8 + size], 'little')
    require(length < 4096, 'Bounded scalar output header')
    start = 8 + size
    header = ast.literal_eval(raw[start:start + length].decode('latin1').strip())
    require(header == dict(descr='<f4', fortran_order=False, shape=(16, 1024)), 'Concrete served FP32 shape')
    data = raw[start + length:]
    require(len(data) == 16 * 1024 * 4 and hashlib.sha256(data).hexdigest() == expected,
            'Exact recorded served-byte digest')
    require(all(math.isfinite(x) and x >= 0 for (x,) in struct.iter_unpack('<f', data)),
            'Finite nonnegative served output')
    return data


def guard(t):
    require(type(t['flops']) is int and 0 < t['flops'] <= BUDGET, 'Actual closed complete bill')
    require(all(type(t[k]) in (float, int) and math.isfinite(t[k]) and t[k] >= 0 for k in TIMINGS)
            and t['wall_s'] < 120 and 0 < t['peak_rss_bytes'] < 8e9, 'Unchanged native resources')


def verify_reset(p, bill_path, bill, pin, call, rank):
    pin(p)
    r = read(p)
    require(path(r['saved_bill']) == bill_path and r['saved_bill_sha256'] == pin(bill_path),
            'Reset follows authenticated saved bill')
    require(r['call'] == call and r['rank'] == rank and r['reset_api'] == 'flopscope.budget_reset'
            and r['after_context_exit'] and r['complete'] and r['context_released']
            and r['closed_context_bill_unchanged_by_reset'] and r['candidate_state_reset'] is False,
            'Harness-only, closed-context, released lifecycle')
    expected = dict(flops=bill['flops'], operation_count=bill['operation_count'],
                    context_wall_s=bill['context_wall_s'], backend_s=bill['flopscope_backend_s'],
                    overhead_s=bill['flopscope_overhead_s'], residual_s=bill['native_residual_s'])
    require(r['closed_metrics'] == expected, 'Retained exact complete metrics across reset')
    return datetime.fromisoformat(r['finished_utc'])


def verify_meter(pin):
    for p, digest in FIXED.items():
        pin(p, digest)
    pin(Path(__file__))
    folder = HERE / 'meter_lifecycle_v2'
    r = sealed(folder, FIXED[folder / 'result.json'], pin)
    bind(r['source_bindings'], pin)
    bind(r['artifacts'], pin)
    require(r['completed'] and r['sources_and_bills_unchanged']
            and all(r[k] is False for k in ('model_accessed', 'weights_accessed', 'truth_accessed', 'private_backend_mutated'))
            and r['closed_context_bill_unchanged_by_reset'] and r['final_context_released'],
            'Fixed target-free meter certificate')
    require((r['records_before_reset'], r['operations_before_reset'], r['records_after_reset'],
             r['operations_after_reset'], r['session_flops_after_reset']) == (4, 16388, 0, 0, 0)
            and r['elapsed_s'] < 60 and r['peak_rss_bytes'] < 1e9, 'Fixed meter results/resources')


def verify_native(acq, folder, ranks, pin):
    """Complete v2 acquisition and exact five-artifact/call lifecycle inventory."""
    require(acq['passed'] and acq['hashes_unchanged'] and acq['ranks'] == list(ranks)
            and acq['initialization_once'] and acq['repeat_excluded'] and not acq['truth_accessed'],
            'Complete, target-free fixed sequence')
    require(acq['input_preparation_in_participant_bill'] is False and acq['candidate_state_reset'] is False
            and acq['input_representation'] == 'owned FNP float32 copies, official loader convention',
            'Official inputs, participant bill and reset ownership')
    bind(acq['source_bindings'], pin)
    bindings = {str(path(p)): v for p, v in acq['source_bindings'].items()}
    require(bindings[str(HERE / 'native_v2.py')] == NATIVE_SHA, 'Actual v2 runner source')
    source = path(acq['source'])
    expected_sha = BANK_SHA if source == BANK else {HERE / 'build_zero_v1/estimator.py': ZERO_SHA,
                       HERE / 'build_terminal_v1/estimator.py': TERMINAL_SHA}.get(source)
    require(expected_sha is not None and acq['candidate_sha256'] == expected_sha,
            'Unchanged bank/null/terminal candidate')
    pin(source, expected_sha)
    require(acq['split'] == ('development' if tuple(ranks) == (35, 100) else 'validation'), 'Fixed split')
    rows = read(STORE / 'manifest.json')['roles']['observer_fit']['row_manifests']
    weight_hashes = {}
    for rank in ranks:
        row_entry = next(x for x in rows if x['role_rank'] == rank)
        rp = STORE / row_entry['path']
        require(bindings[str(rp)] == pin(rp, row_entry['sha256']), 'Canonical row ownership')
        row = read(rp)
        wp = rp.parent / 'weights.npy'
        require(row['role_rank'] == rank and row['role'] == 'observer_fit', 'Canonical rank')
        weight_hashes[rank] = row['files']['weights.npy']['sha256']
        require(bindings[str(wp)] == pin(wp, weight_hashes[rank]), 'Canonical weight ownership')
    trials, preps = acq['trials'], acq['input_preparation']
    require(len(trials) == len(preps) == len(ranks) + 1
            and [t['rank'] for t in trials] == list(ranks) + [ranks[0]], 'All calls including repeat')
    artifacts, outputs, last_finished = {}, [], None
    for call, (t, prep) in enumerate(zip(trials, preps)):
        rank = t['rank']
        require(t['call'] == call and t['includes_initialization'] is (call == 0), 'Cold once, ordered calls')
        guard(t)
        require(t['multiplier'] == max(.1, t['flops'] / BUDGET), 'Actual per-call score factor')
        output, bill = folder / f'call_{call:02d}_rank{rank}.npy', folder / f'call_{call:02d}_bill.json'
        ib, ir, lr = (folder / f'call_{call:02d}_{name}.json' for name in ('input_bill', 'input_reset', 'lifecycle'))
        require(path(t['output']) == output and path(t['input_bill']) == ib
                and path(t['input_reset_receipt']) == ir and path(t['lifecycle_receipt']) == lr,
                'Exact output/bill/lifecycle paths')
        pin(output, t['output_sha256'])
        outputs.append(payload(output, t['output_bytes_sha256'], pin))
        require(read(bill) == t and read(ib) == prep, 'Immutable call and input bills')
        pin(ib, t['input_bill_sha256'])
        pin(ir, t['input_reset_sha256'])
        guard(prep)
        require(prep['call'] == call and prep['rank'] == rank and prep['flops'] == INPUT_F
                and prep['layers'] == 16 and prep['context_closed'] and prep['charged_to_participant'] is False
                and prep['official_fnp_type'] and prep['exact_bytes'] and prep['independent_weight_buffers']
                and prep['weight_file_sha256'] == weight_hashes[rank], 'Exact separately billed official input')
        input_finished = verify_reset(ir, ib, prep, pin, call, rank)
        prediction_finished = verify_reset(lr, bill, t, pin, call, rank)
        require(input_finished <= prediction_finished
                and (last_finished is None or last_finished <= input_finished), 'Causal lifecycle chronology')
        last_finished = prediction_finished
        for p in (output, bill, ib, ir, lr):
            artifacts[str(p)] = pin(p)
    require({str(path(p)): digest for p, digest in acq['artifacts'].items()} == artifacts,
            'Exact complete five-artifact/call set')
    expected_init = 1383913666 + (0 if source == BANK else 120)
    require(outputs[0] == outputs[-1] and acq['initialization_flops'] == expected_init
            and trials[0]['flops'] - trials[-1]['flops'] == expected_init, 'Exact repeat and original init cost')
    return outputs


def verify_development(pin, expected):
    require(isinstance(expected, str) and len(expected) == 64, 'Development proof SHA required')
    r = sealed(HERE / 'native_v2_qualification', expected, pin)
    require(r['completed'] and r['hashes_unchanged'] and r['truth_accessed'] is False
            and r['all_old_output_bytes_exact'] and r['all_actual_bill_deltas_exact']
            and r['delta_flops'] == DELTA_F and len(r['records']) == 6, 'Complete fixed development qualification')
    bind(r['source_bindings'], pin)
    require(r['source_bindings'][str(HERE / 'qualify_native_v2.py')] == pin(HERE / 'qualify_native_v2.py'),
            'Exact development verifier')
    verify_meter(pin)
    return r


def verify_v1_overlaps(acquisitions, pin):
    """Comparison only: failed/partial v1 receipts never enter any score."""
    records = []
    for role, (digest, count) in FAILED_V1.items():
        folder = HERE / f'native_validation_{role}_v1'
        old = sealed(folder, digest, pin, passed=False)
        require(len(old['trials']) == count and old['ranks'] == list(VALID)
                and old['split'] == 'validation' and old['truth_accessed'] is False,
                'Preserved failed v1 exact qualified prefix')
        bind(old['source_bindings'], pin)
        bind(old['artifacts'], pin)  # Includes unqualified saved output; never scored.
        new = acquisitions[role]
        require(old['candidate_sha256'] == new['candidate_sha256']
                and path(old['source']) == path(new['source']), 'Unchanged overlap candidate')
        for call, previous in enumerate(old['trials']):
            guard(previous)
            current = new['trials'][call]
            require(previous['call'] == call and previous['rank'] == current['rank']
                    and previous['includes_initialization'] is (call == 0), 'Qualified overlap call ownership')
            bp = folder / f'call_{call:02d}_bill.json'
            pin(bp)
            require(read(bp) == previous, 'Preserved qualified old bill')
            pin(path(previous['output']), previous['output_sha256'])
            require(payload(previous['output'], previous['output_bytes_sha256'], pin)
                    == payload(current['output'], current['output_bytes_sha256'], pin), 'All old qualified output bytes preserved')
            require(current['flops'] - previous['flops'] == DELTA_F, 'Actual measured overlap accounting delta')
            records.append(dict(role=role, rank=current['rank'], call=call,
                                old_flops=previous['flops'], new_flops=current['flops'],
                                delta_flops=DELTA_F, output_bytes_exact=True, used_for_scoring=False))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualify', action='store_true')
    parser.add_argument('--bank-sha256')
    parser.add_argument('--zero-sha256')
    args = parser.parse_args()
    if not args.qualify:
        parser.print_help()
        return 0
    seals = dict(bank=args.bank_sha256, zero=args.zero_sha256)
    require(all(isinstance(v, str) and len(v) == 64 for v in seals.values()), 'Both development result pins required')
    require(Path(sys.prefix).resolve() == OFFPY.resolve() and os.getpriority(os.PRIO_PROCESS, 0) >= 10,
            'OFFPY, nice10 metadata worker')
    out = HERE / 'native_v2_qualification'
    out.mkdir(exist_ok=False)
    start, pins = time.monotonic(), {}
    result = dict(passed=False, completed=False, records=[], source_bindings=pins,
                  repaired_result_seals=seals, delta_flops=DELTA_F, truth_accessed=False,
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  score_claim=False, initialization_once=True, repeat_excluded_from_any_future_score=True)
    write(out / 'started.json', result)

    def pin(p, expected=None):
        p = path(p)
        require('/truthpost/' not in str(p) and 'mps_truth_' not in p.name, 'No truth access in qualification')
        value = pins.get(str(p))
        if value is None:
            value = sha(p)
            pins[str(p)] = value
        require(expected is None or value == expected, 'Hash mismatch: ' + str(p))
        return value

    def expired(_sig, _frame):
        raise TimeoutError('Fixed metadata qualification 120-second watchdog')

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(120)
    try:
        verify_meter(pin)
        fixed = {}
        for role in ('bank', 'zero'):
            folder = HERE / f'native_development_{role}_v2'
            acq = sealed(folder, seals[role], pin)
            arrays = verify_native(acq, folder, (35, 100), pin)
            require(path(acq['source']) == (BANK if role == 'bank' else HERE / 'build_zero_v1/estimator.py'),
                    'Exact development role')
            fixed[role] = (acq, arrays)
        old_bank_folder = NEW / 'unused_materializations_v1/native'
        old_bank = sealed(old_bank_folder, FIXED[old_bank_folder / 'result.json'], pin)
        require(old_bank['candidate_sha256'] == BANK_SHA and len(old_bank['trials']) == 15, 'Qualified old bank14 plus repeat')
        bind(old_bank['source_bindings'], pin)
        old_zero_folder = HERE / 'native_zero_v1'
        old_zero = sealed(old_zero_folder, FIXED[old_zero_folder / 'result.json'], pin)
        bind(old_zero['source_bindings'], pin)
        bind(old_zero['artifacts'], pin)
        require(old_zero['hashes_unchanged'] and old_zero['candidate_sha256'] == ZERO_SHA
                and len(old_zero['trials']) == 3, 'Exact old null development')
        for role, old, indices, folder in (('bank', old_bank, (0, 1, 14), old_bank_folder),
                                            ('zero', old_zero, (0, 1, 2), old_zero_folder)):
            for call, index in enumerate(indices):
                t, current = old['trials'][index], fixed[role][0]['trials'][call]
                if role == 'bank':
                    op = folder / t['output_artifact']
                    pin(op, old['artifacts'][op.name]['sha256'])
                    for key in ('row_' + str(t['rank']), 'weights_' + str(t['rank'])):
                        entry = old['fixture_bindings'][key]
                        pin(entry['path'], entry['sha256'])
                        require(fixed[role][0]['source_bindings'][entry['path']] == entry['sha256'], 'Old/new canonical fixture')
                else:
                    op = path(t['output'])
                    pin(op, t['output_sha256'])
                bp = folder / f'call_{index:02d}_bill.json'
                pin(bp)
                require(read(bp) == t and t['rank'] == current['rank']
                        and t['includes_initialization'] is (call == 0), 'Matched old call/bill/init')
                require(payload(op, t['output_bytes_sha256'], pin) == fixed[role][1][call], 'Development old/current byte identity')
                require(current['flops'] - t['flops'] == DELTA_F, 'Exactly observed weight-arithmetic delta')
                result['records'].append(dict(role=role, rank=t['rank'], call=call,
                    old_call=index, old_flops=t['flops'], new_flops=current['flops'],
                    delta_flops=DELTA_F, separate_input_copy_flops=INPUT_F, output_bytes_exact=True))
        require(fixed['bank'][1] == fixed['zero'][1], 'Null/current all16 bytes on all development calls')
        require(all(sha(p) == digest for p, digest in pins.items()), 'All qualification seals unchanged')
        require(time.monotonic() - start < 120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 1e9,
                'Metadata qualification resources')
        result.update(passed=True, completed=True, hashes_unchanged=True,
                      all_old_output_bytes_exact=True, all_actual_bill_deltas_exact=True,
                      initialization_flops={r: a[0]['initialization_flops'] for r, a in fixed.items()},
                      input_copy_excluded_from_participant=True, old_partial_scores_used=False)
    except BaseException:
        result['error'] = traceback.format_exc()
    finally:
        signal.alarm(0)
        result.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_s=time.monotonic() - start,
                      peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        write(out / 'result.json', result)
        write(out / 'completion.json', dict(passed=result['passed'], completed=result['completed'],
                                            result_sha256=sha(out / 'result.json')))
    print(json.dumps({k: result.get(k) for k in ('passed', 'records', 'error')}), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
