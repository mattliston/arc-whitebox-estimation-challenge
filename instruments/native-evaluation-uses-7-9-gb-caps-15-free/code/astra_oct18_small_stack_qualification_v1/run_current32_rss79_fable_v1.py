"""Frozen current-95c paired science on ALL 32 store ranks (fable-lead copy of run_current7_authorized_v1 without the seven-rank early return; 7.9 GB RSS backstop science runner); never submits."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
R = HERE.parent
ACC = R / 'astra_oct18_accuracy_v1'
sys.path.insert(0, str(ACC))
HELPER = ACC / 'run_shared32_continuation_v1.py'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == '0e4bbfa74d8eba17b2787f70c6014a4816345982aa27758ba18082e26f9a3ce6'
from run_shared32_continuation_v1 import checked_part, merge, read, write, PINS, RUNNER, GRADE, SEVEN, FRESH, GROUPS
import run_shared32_continuation_v1 as _H   # [fable-lead 2026-10-19] science judge with the 7.9 GB backstop runner (value-only arms)
RUNNER = _H.RUNNER = RUNNER.parent / 'run_guarded_product_census_v5_rss79.py'
PINS[RUNNER] = '3b1ff6806c23459e6f4c47d5ad49d2348e6204a4b58b99b807c8ab2dc260b4d3'

CONTROL = ACC / 'setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py'
CONTROL_SHA = '95c8717670a5b4fbc2dcf57e289a5a0f5a8c6f9c656bee3a638655deb94b0cdf'
CONTROL_RUN = R / 'astra_oct18_lead_v1/compact10_repair_qualification_v1/science32'
BASE_SCORE = 1.4468565569762618e-9
ALL_GROUPS = [[35, 150, 200], [250, 259], [277, 287]] + GROUPS

def verify_part(path, ranks, digest):
    result = checked_part(path, ranks, digest, census=True)
    guard = read(path / 'GUARD_v1.json')
    assert guard['completed'] and guard['error'] is None and guard['whole_group_retired']
    assert not guard.get('remaining_rows')
    assert guard['min_pressure_free_pct'] >= 50
    assert max(guard['max_rss_bytes'], guard['max_footprint_bytes']) <= 7_900_000_000   # [fable-lead] science copy: 7.9 GB (release files use the 7.5 GB judge)
    return result

def summarize(grade):
    values = [row['adjusted_pct'] for row in grade]
    mean = statistics.mean(values)
    return dict(n=len(values), adjusted_pct=mean,
                projected_adjusted_score=BASE_SCORE * (1 + mean / 100),
                adjusted_se=statistics.stdev(values) / math.sqrt(len(values)),
                lower=sum(value < 0 for value in values),
                raw_pct=statistics.mean(row['raw_pct'] for row in grade),
                bill_pct=statistics.mean(row['bill_pct'] for row in grade))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('sha256')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    candidate, output = args.candidate.resolve(), args.output.resolve()
    pins = {p: digest for p, digest in PINS.items()
            if p.parent == RUNNER.parent and p.name.startswith(('run_guarded_', 'boundary_'))}
    pins.update({CONTROL: CONTROL_SHA, candidate: args.sha256,
                 HELPER: '0e4bbfa74d8eba17b2787f70c6014a4816345982aa27758ba18082e26f9a3ce6'})

    def verify_sources():
        for path, digest in pins.items():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path

    verify_sources()
    plan = read(CONTROL_RUN / 'PLAN_v1.json')
    assert plan['sha256'] == CONTROL_SHA and plan['groups'] == ALL_GROUPS
    controls = []
    for index, ranks in enumerate(ALL_GROUPS, 1):
        path = CONTROL_RUN / f'candidate_part{index:02d}'
        verify_part(path, ranks, CONTROL_SHA)
        controls.append((path, ranks))
    output.mkdir(exist_ok=False)
    write(output / 'PLAN_v1.json', dict(candidate=str(candidate), sha256=args.sha256,
          baseline_submission=336935, baseline_score=BASE_SCORE, control_sha256=CONTROL_SHA,
          groups=ALL_GROUPS, source_pins={str(p): digest for p, digest in pins.items()},
          rule='Matt explicitly authorized firing after shorter test on2026-10-18. Frozen7 science gate: projection belowOliver336858=1.4423218171396587e-9; preserve separate runtime/setup/stageA/prefire/release requirements. No32 required by updated user instruction.',
          caveat='Paired frozen existing32 store ranks. No submission authority. Census timing is diagnostic.'))
    parts = []
    for index, ranks in enumerate(ALL_GROUPS, 1):
        verify_sources()
        part = output / f'candidate_part{index:02d}'
        subprocess.run([sys.executable, '-B', str(RUNNER), str(candidate),
                        ','.join(map(str, ranks)), str(part)], check=True)
        verify_part(part, ranks, args.sha256)
        parts.append((part, ranks))
        print(json.dumps(dict(group=index, complete=sum(len(g) for _, g in parts), ranks=ranks)), flush=True)
        if index == 3:
            cp, bp = output / 'candidate7', output / 'control7'
            merge(cp, parts, census=True); merge(bp, controls[:3], census=True)
            subprocess.run([sys.executable, '-B', str(GRADE), str(cp), str(bp), '--layers'], check=True)
            seven = summarize(read(cp / 'GRADE_STORE_v1.json'))
            write(output / 'CURRENT_SOURCE7_v1.json', seven)
            print(json.dumps(dict(seven=seven)), flush=True)
            short_gate = dict(candidate_sha256=args.sha256, baseline_sha256=CONTROL_SHA,
                              source7=seven, all_groups_retired=True,
                              passes_short_science=seven['projected_adjusted_score'] < 1.4423218171396587e-9,
                              user_authorization='fire after shorter test if clears; 2026-10-18',
                              pending='runtime, setup, stageA, prefire, seal and release confirmation',
                              submission_authorized=False)
            write(output / 'SHORT_SOURCE7_GATE_v1.json', short_gate)
            print(json.dumps(short_gate), flush=True)
            # [fable-lead 2026-10-19] run_current32_fable_v1: no early return and no -0.20 continuation gate: always grade the 25 held-out ranks and all 32
    result = dict(candidate_sha256=args.sha256, baseline_sha256=CONTROL_SHA, baseline_submission=336935,
                  all_groups_retired=True, source7=seven, submission_authorized=False)
    for name, a, b in [('heldout25', parts[3:], controls[3:]), ('all32', parts, controls)]:
        cp, bp = output / ('candidate_' + name), output / ('control_' + name)
        merge(cp, a, census=True); merge(bp, b, census=True)
        subprocess.run([sys.executable, '-B', str(GRADE), str(cp), str(bp), '--layers'], check=True)
        result[name] = summarize(read(cp / 'GRADE_STORE_v1.json'))
    result['passes_next_bank_science'] = (result['all32']['projected_adjusted_score'] <= 1.445e-9
                                        and result['heldout25']['adjusted_pct'] < 0)
    result['projects_below_oliver'] = result['all32']['projected_adjusted_score'] < 1.4423218171396587e-9
    write(output / 'RESULT32_v1.json', result)
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
