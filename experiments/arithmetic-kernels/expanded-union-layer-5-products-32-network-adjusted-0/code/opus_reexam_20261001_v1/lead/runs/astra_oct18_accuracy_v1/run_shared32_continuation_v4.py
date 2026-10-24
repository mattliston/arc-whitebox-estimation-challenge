"""HELD: run all32 on one final source, collecting its own prebind census.

Usage: .venv/bin/python -B THIS CANDIDATE EXPECTED_SHA COMPLETED_CONTROL25 NEW_OUTPUT
The lead must grant the exclusive native slot. No old candidate predictions or
bills are reused. Controls are the unchanged current v61j lineage. No retries.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
HELPER = HERE/'run_shared32_continuation_v1.py'
HELPER_SHA = '0e4bbfa74d8eba17b2787f70c6014a4816345982aa27758ba18082e26f9a3ce6'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == HELPER_SHA
from run_shared32_continuation_v1 import (
    R, L, B, CONTROL, CONTROL_SHA, SEVEN, FRESH, GROUPS, CONTROL7,
    RUNNER, GRADE, PINS as ORIGINAL_PINS,
    read, write, rows, checked_part, merge, summary,
)

_PARENT_CHECKED_PART = checked_part
def checked_part(path, ranks, digest, census=False):
    got = _PARENT_CHECKED_PART(path, ranks, digest, census)
    guard = read(path/'GUARD_v1.json')
    assert guard['completed'] and guard['error'] is None and guard['whole_group_retired']
    assert guard['min_pressure_free_pct'] >= 50
    assert guard['max_rss_bytes'] <= 7_500_000_000
    assert guard['max_footprint_bytes'] <= 7_500_000_000
    return got

ALL_GROUPS = [[35,150,200], [250,259], [277,287]] + GROUPS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('expected_sha')
    parser.add_argument('controls25', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    candidate, controls25, output = (p.resolve() for p in (args.candidate, args.controls25, args.output))
    assert len(args.expected_sha) == 64 and all(c in '0123456789abcdef' for c in args.expected_sha)
    pins = {path:digest for path,digest in ORIGINAL_PINS.items()
            if path.parent == B and path.name.startswith(('run_guarded_', 'boundary_'))}
    pins.update({candidate:args.expected_sha, CONTROL:CONTROL_SHA, HELPER:HELPER_SHA})

    def check_pins():
        for path, digest in pins.items():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path

    check_pins()
    complete = read(controls25/'COMPLETE_v1.json')
    assert complete['ranks'] == FRESH and complete['clean'] == 25 and complete['sha256'] == CONTROL_SHA
    for index, group in enumerate(GROUPS, 1):
        checked_part(controls25/f'part{index:02d}', group, CONTROL_SHA, census=True)
    rows(controls25, FRESH, census=True)
    rows(CONTROL7, SEVEN)
    output.mkdir(exist_ok=False)
    write(output/'PLAN_v1.json', dict(status='FINAL_SOURCE_ALL32', candidate=str(candidate),
        sha256=args.expected_sha, groups=ALL_GROUPS, control_source_sha256=CONTROL_SHA,
        controls7=str(CONTROL7), controls25=str(controls25), new_candidate_models=32,
        reused_candidate_models=0, source_pins={str(k):v for k,v in pins.items()},
        note='All32 candidate rows use this one source and matched cold/warm groups. Census timing is diagnostic only.'))
    parts = []
    census_dir = output/'prebind_census'
    census_dir.mkdir()
    census_paths = []
    for index, group in enumerate(ALL_GROUPS, 1):
        check_pins()
        part = output/f'candidate_part{index:02d}'
        subprocess.run([sys.executable, '-B', str(RUNNER), str(candidate),
                        ','.join(map(str, group)), str(part)], check=True)
        got = checked_part(part, group, args.expected_sha, census=True)
        parts.append((part, group))
        # One file per worker session preserves first/later positions in the
        # prebind_targeted_v2 request-budget simulation; its schema is {'rows':...}.
        census_path = census_dir/f'part{index:02d}.json'
        write(census_path, dict(candidate=str(candidate), sha256=args.expected_sha,
            rows=[dict(rank=r['rank'], key_uses=r['key_uses'], bind_list=r['bind_list']) for r in got]))
        census_paths.append(str(census_path))
        print(json.dumps(dict(group=index, complete=sum(len(g) for _,g in parts), ranks=group)), flush=True)
        if index == 3:
            pilot7 = output/'pilot7'
            merge(pilot7, parts, census=True)
            subprocess.run([sys.executable, '-B', str(GRADE), str(pilot7), str(CONTROL7), '--layers'], check=True)
            early = summary(read(pilot7/'GRADE_STORE_v1.json'))
            write(output/'CURRENT_SOURCE7_v1.json', early)
            print(json.dumps(dict(current_source7=early)), flush=True)
            if early['adjusted_pct'] >= 0:
                raise SystemExit('STOP: current-source seven does not improve adjusted score; no remaining25 launched')

    candidate32, control32 = output/'candidate32', output/'control32'
    allrows = merge(candidate32, parts, census=True)
    merge(control32, [(CONTROL7, SEVEN), (controls25, FRESH)])
    candidate7, candidate25 = output/'candidate7', output/'candidate25'
    merge(candidate7, parts[:3], census=True)
    merge(candidate25, parts[3:], census=True)
    keys, binds = Counter(), Counter()
    for row in allrows:
        for *key, count in row['key_uses']:
            keys[tuple(key)] += count
        for start, shape, _repeated in row['bind_list']:
            binds[(start, tuple(shape))] += 1
    write(output/'OWN_PREBIND_CENSUS_v2.json', dict(candidate=str(candidate), sha256=args.expected_sha,
        rows=[dict(rank=r['rank'], key_uses=r['key_uses'], bind_list=r['bind_list']) for r in allrows],
        session_files=census_paths, key_uses=[list(k)+[v] for k,v in sorted(keys.items())],
        bind_uses=[[start,list(shape),count] for (start,shape),count in sorted(binds.items())],
        prebind_input_note='Pass session_files individually to prebind_targeted_v2; retain actual session grouping.',
        setup_budget_note='Use this source own setup trace and measured normal/slow budgets, not historical defaults.'))
    for cand, ctrl in [(candidate7, CONTROL7), (candidate25, controls25), (candidate32, control32)]:
        subprocess.run([sys.executable, '-B', str(GRADE), str(cand), str(ctrl), '--layers'], check=True)
    result = dict(status='COMPLETE_NO_SUBMISSION_AUTHORITY', candidate_sha256=args.expected_sha,
        current_source7=summary(read(candidate7/'GRADE_STORE_v1.json')),
        heldout25=summary(read(candidate25/'GRADE_STORE_v1.json')),
        all32=summary(read(candidate32/'GRADE_STORE_v1.json')),
        prebind_session_files=census_paths, reused_candidate_rows=0, all_groups_retired=True)
    write(output/'RESULT32_v2.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
