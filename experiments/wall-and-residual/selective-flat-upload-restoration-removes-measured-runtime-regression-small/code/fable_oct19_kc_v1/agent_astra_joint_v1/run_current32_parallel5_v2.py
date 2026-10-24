"""Local full32 qualification: five independent original guarded groups.

Source/receipt checks are the default. --run is for the lead after isolating
the local lane queue. No process pausing, guard changes, retries, or submission.
"""
import argparse
import ast
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

A = Path(__file__).resolve().parent
K = A.parent
R = K.parent
ORIGINAL = R / 'astra_oct18_small_stack_qualification_v1/run_current32_m15c_fable_v1.py'
ORIGINAL_SHA = '8c520910f57280e0deb34cc2889f24824ca0d5513e607674983ed03c5980becb'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(ORIGINAL) == ORIGINAL_SHA
spec = importlib.util.spec_from_file_location('frozen_original_current32', ORIGINAL)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)  # Definitions only; guarded main is not called.
from_types = ('verify_part', 'merge', 'read', 'write', 'CONTROL', 'CONTROL_SHA',
              'CONTROL_RUN', 'ALL_GROUPS', 'RUNNER', 'GRADE', 'summarize')
for name in from_types:
    globals()[name] = getattr(original, name)

sys.path.insert(0, str(K))
from compare_best_v1 import report as compare_best

FIXED_PINS = {
    ORIGINAL: ORIGINAL_SHA,
    original.HELPER: '0e4bbfa74d8eba17b2787f70c6014a4816345982aa27758ba18082e26f9a3ce6',
    RUNNER: '27f8eb8a1571207e7611e2cec4baddda01fc893a55da625e9d2f53eefd573f0d',
    RUNNER.parent / 'boundary_client_product_census_v5.py': 'a7545a6f0de0a923805e9743a86be1232cea63cc6dc30740c4345305bf9a93d7',
    RUNNER.parent / 'boundary_run_product_census_v5.py': 'e0b929cdec8ea29a49d25f611bd453c3c2e5ee09f1cde8b4af96dfb7584192fe',
    R / 'astra_crunch_physics_v1/guard_memory15_sci_v2.py': 'b61828a6a4dcfbec2e4e34316067bcb365c588ee0bc9aee35a1421b17b002b40',
    GRADE: 'f7d76fa9804798d8aa05a6b1f4d1aa5d93506539c6c6b769ae1e9c505c2d1e30',
    K / 'compare_best_v1.py': '26f519604895cb039f68c018d1d9f07524fd241183067652e0fbf1a9b62d75d2',
    K / 'compare_relative_v1.py': 'dead1805bfc9dfe381cb740a270b7b41a0642af845cc5ae90b46f052b3997144',
    CONTROL: CONTROL_SHA,
}
# Preserve every original science-runner/boundary pin as well as the extra
# offline grading and comparison pins above.
FIXED_PINS.update({p: digest for p, digest in original.PINS.items()
                   if p.parent == RUNNER.parent and p.name.startswith(('run_guarded_', 'boundary_'))})
J2W_SCORE = 1.416013781559781e-9


def utc():
    return subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def verify_pins(pins):
    for path, digest in pins.items():
        assert sha(path) == digest, ('source changed', str(path))


def assert_no_governor(candidate):
    tree = ast.parse(candidate.read_text())
    final = tree.body[-1]
    assert isinstance(final, ast.Assign) and len(final.targets) == 1
    assert isinstance(final.targets[0], ast.Name) and final.targets[0].id == '_FAST_RESCUE_FRAC'
    assert isinstance(final.value, ast.Dict) and not final.value.keys, 'Permanent no-governor assignment must be last'


def command(candidate, ranks, part):
    return [sys.executable, '-B', str(RUNNER), str(candidate), ','.join(map(str, ranks)), str(part)]


def active_native_rows(raw, own_pid):
    parsed = [line.strip().split(None, 3) for line in raw.splitlines()]
    parents = {int(f[0]): int(f[1]) for f in parsed if len(f) == 4}
    ancestors = {own_pid}
    while own_pid in parents and parents[own_pid] not in ancestors:
        own_pid = parents[own_pid]
        ancestors.add(own_pid)
    active = []
    pattern = re.compile(r'/(?:guard_memory\w*\.py|boundary_run_product_census_v\w*\.py|'
                         r'boundary_run_dual_v\w*\.py|eval_fresh_arm_v\d+\.py)(?:\s|$)')
    dispatcher = re.compile(r'/run_current32_\w+\.py(?:\s|$)')
    for line in raw.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) == 4 and int(fields[0]) not in ancestors:
            if pattern.search(fields[3]) or (dispatcher.search(fields[3]) and 'T' not in fields[2]):
                active.append(line.strip())
    return active


def require_drained_native_slots():
    raw = subprocess.check_output(['/bin/ps', '-axo', 'pid=,ppid=,stat=,command='], text=True)
    active = active_native_rows(raw, os.getpid())
    assert not active, ('Lead must drain all active native/auxiliary workers first', active)


def merge_checked(output, sources):
    if not output.exists():
        return merge(output, sources, census=True)
    expected = [dict(path=str(p), ranks=g) for p, g in sources]
    assert read(output / 'MERGE_PROVENANCE_v1.json') == expected
    expected_rows = []
    for path, ranks in sources:
        expected_rows.extend(original._H.rows(path, ranks, census=True))
        for rank in ranks:
            assert (output / f'pred_{rank}.bin').read_bytes() == (path / f'pred_{rank}.bin').read_bytes()
    assert [json.loads(x) for x in (output / 'rows.jsonl').read_text().splitlines()] == expected_rows


def write_same(path, value):
    if path.exists():
        assert read(path) == value, ('existing receipt differs', path)
    else:
        write(path, value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate', type=Path)
    p.add_argument('sha256')
    p.add_argument('output', type=Path)
    p.add_argument('--run', action='store_true')
    p.add_argument('--resume', action='store_true', help='Reuse verified whole groups only; partial groups require manual review.')
    a = p.parse_args()
    candidate, output = a.candidate.resolve(), a.output.resolve()
    pins = dict(FIXED_PINS)
    pins[candidate] = a.sha256
    pins[Path(__file__).resolve()] = sha(__file__)
    best_ref = K / 'BEST_GRADED_REFERENCE_v1.json'
    pins[best_ref] = sha(best_ref)
    verify_pins(pins)
    assert_no_governor(candidate)
    assert read(CONTROL_RUN / 'PLAN_v1.json')['groups'] == ALL_GROUPS
    assert read(CONTROL_RUN / 'PLAN_v1.json')['sha256'] == CONTROL_SHA
    controls = []
    for index, ranks in enumerate(ALL_GROUPS, 1):
        part = CONTROL_RUN / f'candidate_part{index:02d}'
        verify_part(part, ranks, CONTROL_SHA)
        controls.append((part, ranks))
    baseline = read(best_ref)
    plan = dict(candidate=str(candidate), sha256=a.sha256, baseline_submission=336935,
                baseline_score=original.BASE_SCORE, control_sha256=CONTROL_SHA,
                groups=ALL_GROUPS, source_pins={str(p): h for p, h in pins.items()},
                run_kind='PARALLEL5_ORIGINAL_CURRENT32_v2', max_concurrent_original_groups=5,
                group_order='Original cold/warm order preserved within each of the 12 independent groups.',
                comparison_reference=baseline, j2w_score=J2W_SCORE,
                rule='All32 and held25 required. Any next fire must verify projection below J2W and pass original release gates. No seven-rank or prior sub-bar authority.',
                caveat='Concurrent census wall times are diagnostic only; this helper neither isolates queues nor submits.')
    if not a.run:
        print(json.dumps(dict(utc=utc(), status='SOURCE_AND_CONTROL_CHECK_PASS_NO_LAUNCH', plan=plan,
            commands=[command(candidate, ranks, output / f'candidate_part{i:02d}') for i, ranks in enumerate(ALL_GROUPS, 1)]), indent=2))
        return 0
    lock = (A / 'LOCAL_PARALLEL32_v1.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require_drained_native_slots()
    if output.exists():
        assert a.resume, 'Existing output requires --resume'
        assert read(output / 'PLAN_v1.json') == plan
    else:
        output.mkdir(exist_ok=False)
        write(output / 'PLAN_v1.json', plan)
    parts, pending = [], []
    for i, ranks in enumerate(ALL_GROUPS, 1):
        part = output / f'candidate_part{i:02d}'
        parts.append((part, ranks))
        if part.exists():
            assert a.resume
            verify_part(part, ranks, a.sha256)
        else:
            pending.append((i, part, ranks))
    active, failures, completed = {}, [], len(parts) - len(pending)
    started = utc()
    while pending or active:
        while pending and len(active) < 5 and not failures:
            verify_pins(pins)
            i, part, ranks = pending.pop(0)
            log = (output / f'group{i:02d}_controller.log').open('a')
            proc = subprocess.Popen(command(candidate, ranks, part), stdout=log, stderr=subprocess.STDOUT)
            active[proc.pid] = (proc, log, i, part, ranks)
            print(json.dumps(dict(utc=utc(), event='group_started', group=i, ranks=ranks, worker_pid=proc.pid)), flush=True)
        for pid, (proc, log, i, part, ranks) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            del active[pid]
            try:
                assert code == 0, ('worker_exit', code)
                verify_part(part, ranks, a.sha256)
            except Exception as error:
                failures.append(dict(group=i, exit_code=code, error=repr(error)))
            else:
                completed += 1
            print(json.dumps(dict(utc=utc(), event='group_retired', group=i, completed_groups=completed,
                                  exit_code=code, failures=failures)), flush=True)
        if failures and not active:
            write(output / 'FAILED_v1.json', dict(utc=utc(), failures=failures, completed_groups=completed,
                                                remaining_groups=[i for i, _, _ in pending]))
            return 1
        if active:
            time.sleep(.2)
    verify_pins(pins)
    for part, ranks in parts:
        verify_part(part, ranks, a.sha256)
    summaries = {}
    for name, aa, bb in [('seven', parts[:3], controls[:3]), ('heldout25', parts[3:], controls[3:]), ('all32', parts, controls)]:
        cn = 'candidate7' if name == 'seven' else 'candidate_' + name
        bn = 'control7' if name == 'seven' else 'control_' + name
        cp, bp = output / cn, output / bn
        merge_checked(cp, aa)
        merge_checked(bp, bb)
        if not (cp / 'GRADE_STORE_v1.json').exists():
            subprocess.run([sys.executable, '-B', str(GRADE), str(cp), str(bp), '--layers'], check=True)
        summaries[name] = summarize(read(cp / 'GRADE_STORE_v1.json'))
    write_same(output / 'CURRENT_SOURCE7_v1.json', summaries['seven'])
    result = dict(candidate_sha256=a.sha256, baseline_sha256=CONTROL_SHA, baseline_submission=336935,
                  all_groups_retired=True, source7=summaries['seven'], heldout25=summaries['heldout25'],
                  all32=summaries['all32'], submission_authorized=False)
    write_same(output / 'RESULT32_v1.json', result)
    vs_best = compare_best(output)
    write_same(output / 'VS_BEST_GRADED_v1.json', vs_best)
    projected = baseline['platform_score'] * (1 + vs_best['all32']['adjusted_pct'] / 100)
    receipt = dict(utc=utc(), started_utc=started, candidate_sha256=a.sha256,
                   max_concurrent_groups=5, all_groups_retired=True, best_reference=baseline,
                   all32=vs_best['all32'], heldout25=vs_best['heldout25'],
                   projected_adjusted_score=projected, j2w_score=J2W_SCORE,
                   projects_below_j2w=projected < J2W_SCORE and vs_best['heldout25']['adjusted_pct'] < 0,
                   projection_caveat='Paired local32 projection, not a measured platform score.',
                   original_release_gates_required=True, submission_authorized=False)
    write_same(output / 'PARALLEL32_COMPLETE_v1.json', receipt)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
