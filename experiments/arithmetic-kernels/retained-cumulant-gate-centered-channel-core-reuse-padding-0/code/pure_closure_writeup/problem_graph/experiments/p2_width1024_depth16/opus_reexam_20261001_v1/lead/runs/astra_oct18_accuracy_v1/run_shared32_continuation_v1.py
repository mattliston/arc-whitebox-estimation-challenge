"""HELD: fixed shared-v2 continuation; run only after lead grants the sole slot.

Usage: .venv/bin/python -B THIS NEW_OUTPUT [--controls25 COMPLETED_CONTROL25]
Reuses the existing seven paired models and first three fresh controls. Default
work is 22 current-v61j controls plus 25 unchanged-candidate models. Groups of
three preserve the same cold/warm positions. No source edits or retry loop.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

D = Path(__file__).resolve().parent
R = D.parent
L = R/'astra_oct18_lead_v1'
B = R/'astra_oct18_bill_v1'
CANDIDATE = B/'candidate_v61j_union_shared_HELD_v2.py'
CANDIDATE_SHA = 'ddbb8bb782fa35d751dc2e853948da89c4f756d4e9ba515b9108b93fc6f75269'
CONTROL = R/'opus_fix336687_v1/candidate_v61j.py'
CONTROL_SHA = 'ac93b10f2c8fa541b94cf0787b7f573d36cf1233df219d00a22335c4f9a47129'
SEVEN = [35, 150, 200, 250, 259, 277, 287]
FRESH = [16, 293, 300, 310, 312, 316, 320, 328, 333, 378, 390, 400,
         424, 459, 473, 474, 500, 530, 548, 550, 557, 623, 641, 651, 672]
GROUPS = [FRESH[i:i+3] for i in range(0, 25, 3)]
BASE_SCORE = 1.4515052706908603e-9
CANDIDATE7 = L/'union_shared_7_v2'
CONTROL7 = D/'control_current_carrier_7_v1'
FIRST_CONTROL3 = L/'control_v61j_fresh3_v1'
CONTROLLER = L/'run_current_controls25_v2.py'
RUNNER = B/'run_guarded_product_census_v5.py'
GRADE = R/'opus_fix336687_v1/grade_store_v1.py'
PINS = {
    CANDIDATE: CANDIDATE_SHA,
    CONTROL: CONTROL_SHA,
    CONTROLLER: '21585030a9ffbb6577b6039920adc3105b339246432f80046d7f46b420e788e4',
    B/'run_guarded_product_census_v4.py': 'cbfb375b7b3ab1e4330a009526ee8345149fb67473c223bf39edbdb90cb4221e',
    RUNNER: '9994afd0e2240f0469645da8325df1b2f9e8f8e0ce703b77778344adb4572210',
    B/'boundary_client_product_census_v5.py': 'a7545a6f0de0a923805e9743a86be1232cea63cc6dc30740c4345305bf9a93d7',
    B/'boundary_run_product_census_v5.py': 'e0b929cdec8ea29a49d25f611bd453c3c2e5ee09f1cde8b4af96dfb7584192fe',
}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def pins():
    for path, digest in PINS.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path


def rows(path, ranks, census=False):
    got = [json.loads(line) for line in (path/'rows.jsonl').read_text().splitlines()]
    assert [r['rank'] for r in got] == ranks, path
    for row in got:
        assert row['nonfinite'] == 0 and row['n_out'] == 16384
        assert row['fast_rescue_stats']['calls'] == 0
        assert row['fast_rescue_control']['layer'] is None
        assert (path/f"pred_{row['rank']}.bin").stat().st_size == 16*1024*4
        if census:
            assert row['census_instrumented'] and 'key_uses' in row and 'bind_list' in row
    return got


def checked_part(path, ranks, digest, census=False):
    p, g = read(path/'PLAN_v1.json'), read(path/'GUARD_v1.json')
    assert p['sha256'] == digest and p['ranks'] == ','.join(map(str, ranks))
    assert g['whole_group_retired'] and g['exit_code'] == 0
    assert all(v['version'].split('+')[0] == '0.12.3' for v in p['runtime'].values())
    return rows(path, ranks, census)


def merge(output, sources, census=False):
    output.mkdir(exist_ok=False)
    merged = []
    with (output/'rows.jsonl').open('x') as f:
        for source, ranks in sources:
            got = rows(source, ranks, census)
            for row in got:
                rank = row['rank']
                assert rank not in [r['rank'] for r in merged]
                (output/f'pred_{rank}.bin').symlink_to(source/f'pred_{rank}.bin')
                f.write(json.dumps(row)+'\n')
                merged.append(row)
    write(output/'MERGE_PROVENANCE_v1.json', [dict(path=str(p), ranks=r) for p, r in sources])
    return merged


def summary(graded):
    a = [x['adjusted_pct'] for x in graded]
    mean = statistics.mean(a)
    return dict(n=len(a), adjusted_pct=mean, projected_adjusted_score=BASE_SCORE*(1+mean/100),
                adjusted_se=statistics.stdev(a)/math.sqrt(len(a)), lower=sum(x < 0 for x in a),
                raw_pct=statistics.mean(x['raw_pct'] for x in graded),
                bill_pct=statistics.mean(x['bill_pct'] for x in graded))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--controls25', type=Path, help='Reuse an already completed identical current-control cohort')
    args = parser.parse_args()
    pins()
    p7 = read(CANDIDATE7/'PLAN_v1.json')
    assert p7['sha256'] == CANDIDATE_SHA
    assert p7['groups'] == [[35, 150, 200], [250, 259], [277, 287]]
    assert p7['runner_sha256'] == PINS[RUNNER]
    for index, group in enumerate(p7['groups'], 1):
        checked_part(CANDIDATE7/f'part{index}', group, CANDIDATE_SHA, census=True)
    pilot_grade = read(CANDIDATE7/'GRADE_STORE_v1.json')
    assert sorted(x['rank'] for x in pilot_grade) == sorted(SEVEN)
    rows(CONTROL7, SEVEN)
    checked_part(FIRST_CONTROL3, FRESH[:3], CONTROL_SHA, census=True)
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    write(output/'PLAN_v1.json', dict(status='FIXED_CONTINUATION', candidate=str(CANDIDATE),
        candidate_sha256=CANDIDATE_SHA, control_sha256=CONTROL_SHA, fresh_groups=GROUPS,
        existing_seven=str(CANDIDATE7), existing_first_control_three=str(FIRST_CONTROL3),
        new_models=25 if args.controls25 else 47, source_pins={str(k):v for k,v in PINS.items()},
        runtime_note='Census timings are diagnostic; final runtime gates remain separate and uninstrumented.'))
    controls25 = args.controls25.resolve() if args.controls25 else output/'control25'
    if not args.controls25:
        subprocess.run([sys.executable, '-B', str(CONTROLLER), str(controls25), str(FIRST_CONTROL3)], check=True)
    complete = read(controls25/'COMPLETE_v1.json')
    assert complete['ranks'] == FRESH and complete['clean'] == 25 and complete['sha256'] == CONTROL_SHA
    for index, group in enumerate(GROUPS, 1):
        checked_part(controls25/f'part{index:02d}', group, CONTROL_SHA, census=True)
    candidate_parts = []
    for index, group in enumerate(GROUPS, 1):
        pins()
        part = output/f'candidate_part{index:02d}'
        subprocess.run([sys.executable, '-B', str(RUNNER), str(CANDIDATE), ','.join(map(str, group)), str(part)], check=True)
        checked_part(part, group, CANDIDATE_SHA, census=True)
        candidate_parts.append((part, group))
        print(json.dumps(dict(candidate_group=index, complete=sum(len(g) for _,g in candidate_parts), ranks=group)), flush=True)
    candidate25 = output/'candidate25'
    merge(candidate25, candidate_parts, census=True)
    control32, candidate32 = output/'control32', output/'candidate32'
    merge(control32, [(CONTROL7, SEVEN), (controls25, FRESH)])
    allrows = merge(candidate32, [(CANDIDATE7, SEVEN), (candidate25, FRESH)], census=True)
    key_counts, bind_counts = Counter(), Counter()
    for row in allrows:
        for *key, count in row['key_uses']:
            key_counts[tuple(key)] += count
        for start, shape, repeated in row['bind_list']:
            bind_counts[(start, tuple(shape))] += 1
    write(output/'OWN_PREBIND_CENSUS_v1.json', dict(candidate_sha256=CANDIDATE_SHA, ranks=SEVEN+FRESH,
        key_uses=[list(k)+[v] for k,v in sorted(key_counts.items())],
        bind_uses=[[start,list(shape),count] for (start,shape),count in sorted(bind_counts.items())],
        sources=[dict(path=str(p), ranks=g) for p,g in [(CANDIDATE7,SEVEN)]+candidate_parts],
        note='Current candidate prediction-time census, not a prebind policy change or runtime approval.'))
    for candidate, control in [(candidate25, controls25), (candidate32, control32)]:
        subprocess.run([sys.executable, '-B', str(GRADE), str(candidate), str(control), '--layers'], check=True)
    result = dict(status='COMPLETE_NO_SUBMISSION_AUTHORITY', source_sha256=CANDIDATE_SHA,
                  fixed7=summary(pilot_grade), heldout25=summary(read(candidate25/'GRADE_STORE_v1.json')),
                  all32=summary(read(candidate32/'GRADE_STORE_v1.json')), all_groups_retired=True)
    write(output/'RESULT32_v1.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
