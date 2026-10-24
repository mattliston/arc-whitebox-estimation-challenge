"""Read-only Oct18 release reliability gate; no SDK, model, upload, or credentials.

Science coverage is exactly the seven ranks authorized by Matt on 2026-10-18.
Runtime/setup calibration is graded336935, with unchanged336854 runtime limits.
PASS is local qualification, not a hosted guarantee or automatic submission.
"""
import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent
LEAD = RUNS / 'astra_oct18_lead_v1'
ACC = RUNS / 'astra_oct18_accuracy_v1'
O = RUNS.parents[1]
BASE_SHA = '95c8717670a5b4fbc2dcf57e289a5a0f5a8c6f9c656bee3a638655deb94b0cdf'
CANDIDATE_SHA = '13330d6320c764109aaca98a98d3aa5f380c3463d7df079ecfba44d1f0d4f57c'
BASE_SCORE = 1.4468565569762618e-9
OLIVER_SCORE = 1.4423218171396587e-9
GROUPS = [[35, 150, 200], [250, 259], [277, 287]]
RANKS = [r for g in GROUPS for r in g]
CONTROL = LEAD / 'compact10_repair_qualification_v1'
SETUP_LIMIT = 4.60
AUTHORIZATION = dict(user='Matt', date='2026-10-18', coverage='exact seven fixed ranks',
    instruction='Fire after shorter test if it clears; preserve runtime, setup, Stage A, prefire, seal and release checks.',
    science_threshold=OLIVER_SCORE, full32_required=False,
    provenance='Root relayed explicit user authorization; also recorded by frozen seven runner in PLAN and SHORT_SOURCE7_GATE.')
PINS = {
    LEAD / 'runtime_preliminary_336935_v2.py': 'b2fd404aa53974b4e860e582d881f6f91b232a5703043df2206e76ea7a322495',
    ACC / 'runtime_gate_thread_v6.py': 'e8119395c599d48a1033d8107fe37cc7732e8b6403cdd9bba2806cc7dacd93cc',
    ACC / 'runtime_gate_current_v4.py': '353e25df2e595649508671903b189245fff196bc4ccfa6507ee7fbc0cdf985c6',
    ACC / 'run_shared32_continuation_v1.py': '0e4bbfa74d8eba17b2787f70c6014a4816345982aa27758ba18082e26f9a3ce6',
    RUNS / 'astra_oct18_small_stack_qualification_v1/run_current7_authorized_v1.py': '7d29d60daa1639f60bcd46979e3121a011ef188ad552982c10907c07aa1dc72e',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def require(condition, message):
    if not condition:
        raise ValueError(message)

def modules():
    for path, digest in PINS.items():
        require(sha(path) == digest, 'Frozen validator changed: ' + str(path))
    sys.path[:0] = [str(LEAD), str(ACC)]
    import runtime_preliminary_336935_v2 as preliminary
    import runtime_gate_thread_v6 as thread
    import runtime_gate_current_v4 as setup_validator
    require(all(thread.LIMITS[k] == v for k, v in preliminary.LIMITS.items()), 'Original runtime limits changed')
    spec = importlib.util.spec_from_file_location('authorized7_source', next(p for p in PINS if p.name == 'run_current7_authorized_v1.py'))
    science = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(science)
    return preliminary, thread, setup_validator, science

def guard_clean(guard):
    require(guard['completed'] and guard['whole_group_retired'] and guard['exit_code'] == 0
            and guard['error'] is None and not guard.get('remaining_rows'), 'Guard unfinished or not cleanly retired')
    require(guard['min_pressure_free_pct'] >= 50 and
            max(guard['max_rss_bytes'], guard['max_footprint_bytes']) <= 7_500_000_000,
            'Unchanged7.5GB/free50 guard exceeded')

def science_reference(directory):
    """Revalidate and freeze only the seven completed source-specific predictions."""
    _, _, _, science = modules()
    directory = Path(directory).resolve()
    plan = read(directory / 'PLAN_v1.json')
    result = read(directory / 'SHORT_SOURCE7_GATE_v1.json')
    require(plan['sha256'] == result['candidate_sha256'] == CANDIDATE_SHA == sha(plan['candidate']), 'Science source mismatch')
    require(plan['baseline_submission'] == 336935 and plan['control_sha256'] == result['baseline_sha256'] == BASE_SHA, 'Science control mismatch')
    require(plan['baseline_score'] == BASE_SCORE and plan['groups'][:3] == GROUPS, 'Science group or baseline mismatch')
    require('2026-10-18' in plan['rule'] and 'No32 required' in plan['rule'], 'Missing explicit short-test authorization')
    require(result['user_authorization'] == 'fire after shorter test if clears; 2026-10-18', 'Authorization metadata mismatch')
    require(result['all_groups_retired'] and result['passes_short_science'], 'Seven science failed or groups not retired')
    require(sorted(p.name for p in directory.glob('candidate_part*') if p.is_dir()) ==
            [f'candidate_part{i:02d}' for i in range(1, 4)], 'Require exactly three completed science groups')
    for path, digest in plan['source_pins'].items():
        require(sha(path) == digest, 'Science source pin changed: ' + path)
    references = []
    for index, ranks in enumerate(GROUPS, 1):
        part = directory / f'candidate_part{index:02d}'
        science.verify_part(part, ranks, CANDIDATE_SHA)
        guard_clean(read(part / 'GUARD_v1.json'))
        controls = CONTROL / 'science32' / f'candidate_part{index:02d}'
        science.verify_part(controls, ranks, BASE_SHA)
        for rank in ranks:
            path = (part / f'pred_{rank}.bin').resolve()
            require(path.stat().st_size == 65536, 'Invalid science prediction size')
            require(sha(directory / 'candidate7' / path.name) == sha(path), 'Merged candidate byte mismatch')
            require(sha(directory / 'control7' / path.name) == sha(controls / path.name), 'Merged control byte mismatch')
            references.append(dict(rank=rank, path=str(path), bytes=65536, sha256=sha(path)))
    grade_path = directory / 'candidate7/GRADE_STORE_v1.json'
    grade = read(grade_path)
    require(sorted(r['rank'] for r in grade) == sorted(RANKS), 'Grade is not exactly seven unique ranks')
    measured = science.summarize(grade)
    require(measured == result['source7'] == read(directory / 'CURRENT_SOURCE7_v1.json'), 'Science summary differs from grade')
    require(measured['n'] == 7 and measured['projected_adjusted_score'] < OLIVER_SCORE, 'Science projection does not clear Oliver')
    return dict(source_sha256=CANDIDATE_SHA, candidate=plan['candidate'], science_n=7,
        science_directory=str(directory), authorization=AUTHORIZATION, source7=measured, rows=references,
        science_result_path=str(directory / 'SHORT_SOURCE7_GATE_v1.json'),
        science_result_sha256=sha(directory / 'SHORT_SOURCE7_GATE_v1.json'),
        science_grade_path=str(grade_path), science_grade_sha256=sha(grade_path))

def setup_current(candidate_directory, candidate_guard, preliminary, validator):
    pins = read(HERE / 'SETUP_CONTROL_PINS_336935_v1.json')
    require(sha(HERE / 'SETUP_CONTROL_PINS_336935_v1.json') == SETUP_PINS_SHA, 'Setup pin ledger changed')
    for relative, digest in pins.items():
        require(sha(CONTROL / relative) == digest, 'Frozen current setup reference changed')
    candidate = validator.completed_five(candidate_directory, candidate_guard)
    reference = validator.completed_five(CONTROL / 'setup5', CONTROL / 'SETUP_GUARD_v1.json')
    for record in (candidate, reference):
        guard_clean(read(record['guard']))
    require(candidate['sha256'] == CANDIDATE_SHA and reference['sha256'] == BASE_SHA, 'Setup source mismatch')
    _, telemetry_path = preliminary.telemetry()
    hosted_max = max(r['wall_s'] for r in read(telemetry_path)['evaluation']['results']['setup_executions'])
    ratios = {key: max(r[key] for r in candidate['values']) / min(r[key] for r in reference['values'])
              for key in ('bound_s', 'envelope_s')}
    projected = {key: hosted_max * value for key, value in ratios.items()}
    return dict(status='PASS' if all(v <= SETUP_LIMIT for v in projected.values()) else 'HOLD',
        control_sid=336935, candidate=candidate, reference=reference, projected=projected,
        max_candidate_over_min_reference=ratios, hosted_setup_max_s=hosted_max, limit_s=SETUP_LIMIT)

def stage_a(path):
    path = Path(path).resolve()
    result = read(path)
    required = ('contract', 'package', 'validate_package', 'archive_contents', 'no_numpy_import',
                'einsum_signatures_proven', 'einsum_no_new_repeated_operand')
    require(result['stage'] == 'A' and result['passed'] and result['uploaded'] is False, 'Stage A failed or unexpected upload')
    require(all(result['gates'].get(k) is True for k in required) and all(result['gates'].values()), 'Stage A gate not passed')
    require(result['candidate_sha256'] == CANDIDATE_SHA == sha(result['candidate']), 'Stage A source mismatch')
    for name in ('validate.json', 'package.json', 'validate_package.json'):
        require(read(path.parent / name)['returncode'] == 0, 'Stage A subprocess failed')
    archive = Path(result['archive']).resolve()
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        require(sorted(m.name for m in members) == ['estimator.py', 'manifest.json'] and all(m.isfile() for m in members), 'Archive contents invalid')
        require(hashlib.sha256(tar.extractfile('estimator.py').read()).hexdigest() == CANDIDATE_SHA, 'Archive estimator differs')
        require(json.loads(tar.extractfile('manifest.json').read())['entrypoint'] == dict(module='estimator', **{'class': 'Estimator'}), 'Archive entrypoint invalid')
    return dict(status='PASS', path=str(path), sha256=sha(path), archive=str(archive), archive_sha256=sha(archive))

def evaluate(args):
    preliminary, thread, validator, _ = modules()
    frozen = read(args.byte_references)
    fresh = science_reference(frozen['science_directory'])
    require(frozen == fresh, 'Frozen authorized-seven references changed')
    modes = {mode: preliminary.evaluate(directory, CANDIDATE_SHA, mode)
             for mode, directory in [('normal', args.normal), ('slow13', args.slow)]}
    candidate = [thread.group(args.normal, CANDIDATE_SHA, 1.), thread.group(args.slow, CANDIDATE_SHA, 1.3)]
    controls = [thread.group(CONTROL / ('candidate_' + mode), BASE_SHA, pace)
                for mode, pace in [('normal', 1.), ('slow13', 1.3)]]
    repeatability = dict(candidate=thread.repeatability(*candidate), control=thread.repeatability(*controls))
    cold = thread.group(args.cold259, CANDIDATE_SHA, 1., ranks=(259,))
    hashes = {r['rank']: r['sha256'] for r in frozen['rows']}
    for group in candidate + controls + [cold]:
        guard_clean(group['guard'])
        require(group['manifest']['dual_counter']['thread_counter_scope'] == 'measurement thread only', 'Incorrect counter scope')
        require(str(Path(group['manifest']['candidate']).resolve()) in group['guard']['command'] and
                group['directory'] in group['guard']['command'], 'Runtime guard path mismatch')
    for group in candidate + [cold]:
        for rank in group['rows']:
            path = Path(group['directory']) / f'pred_{rank}.bin'
            require(path.stat().st_size == 65536 and sha(path) == hashes[rank], 'Runtime differs from own-seven bytes')
    setup = setup_current(args.setup, args.setup_guard, preliminary, validator)
    packaged = stage_a(args.stage_a)
    for record in modes.values():
        record['limitation'] = 'Current336935 paired projection. Exact authorized-seven bytes, setup5, cold259 and Stage A are checked by this final gate; no32 claim.'
    passed = all(x['status'] == 'PASS' for x in list(modes.values()) + [setup, packaged])
    return dict(status='PASS' if passed else 'HOLD', base_sid=336935, candidate_sha256=CANDIDATE_SHA,
        authorization=AUTHORIZATION, science_n=7, science=frozen, modes=modes, repeatability=repeatability,
        setup=setup, stage_a=packaged, cold259=dict(directory=cold['directory'], own_seven_bytes_equal=True),
        limits=dict(preliminary.LIMITS, setup_max_s=SETUP_LIMIT), submission_authorized=False,
        remaining='Root-owned prefire, sealed release confirmation and fire; this gate never submits.',
        limitation='Archived matched controls and local projections cannot guarantee hosted behavior. No heldout25 or32 coverage is claimed.')

def self_test():
    preliminary, _, _, _ = modules()
    ts, _ = preliminary.telemetry()
    projected = dict(residual_median_s=statistics.median(t['residual_wall_time_s'] for t in ts),
        residual_max_s=max(t['residual_wall_time_s'] for t in ts), roundtrip_max_s=max(t['participant_roundtrip_wall_s'] for t in ts))
    decision = 'PASS' if all(projected[k] <= v for k, v in preliminary.LIMITS.items()) else 'HOLD'
    require(decision == 'HOLD', 'All1 ratios must not pass current hosted residual cushion')
    require(projected['residual_median_s'] > preliminary.LIMITS['residual_median_s'] and
            projected['residual_max_s'] > preliminary.LIMITS['residual_max_s'], 'Expected both residual limits to fail')
    return dict(status='PASS', self_test='all1 ratios correctly HOLD', hypothetical_release_decision=decision,
        projected=projected, limits=preliminary.LIMITS, models=0, sdk_calls=0)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freeze = commands.add_parser('freeze-seven'); freeze.add_argument('science', type=Path)
    full = commands.add_parser('evaluate')
    for name in ('normal', 'slow', 'setup', 'setup_guard', 'byte_references', 'cold259', 'stage_a'):
        full.add_argument(name, type=Path)
    test = commands.add_parser('self-test')
    for subparser in (freeze, full, test): subparser.add_argument('output', type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'Never overwrite a receipt')
    try:
        result = science_reference(args.science) if args.command == 'freeze-seven' else self_test() if args.command == 'self-test' else evaluate(args)
        status = result.get('status', 'PASS')
    except Exception as exc:
        result = dict(status='FAIL', reason=repr(exc), candidate_sha256=CANDIDATE_SHA)
        status = 'FAIL'
    if args.command != 'freeze-seven':
        result['gate_source_sha256'] = sha(__file__)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(dict(status=status, output=str(args.output), gate_sha256=sha(__file__))))
    return 0 if status == 'PASS' else 1

SETUP_PINS_SHA = 'e01efba29a430e30a1ae9607c26536816c53c50df6f11037c4450870d6be4407'

if __name__ == '__main__':
    raise SystemExit(main())
