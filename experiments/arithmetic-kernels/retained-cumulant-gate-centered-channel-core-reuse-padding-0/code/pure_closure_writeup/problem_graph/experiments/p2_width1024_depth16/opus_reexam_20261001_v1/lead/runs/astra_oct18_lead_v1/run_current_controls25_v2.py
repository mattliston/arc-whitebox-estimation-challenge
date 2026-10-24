"""Fixed current-lineage controls and public geometry; serial groups of three."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

D = Path(__file__).resolve().parent
R = D.parent
source = R/'opus_fix336687_v1/candidate_v61j.py'
expected = 'ac93b10f2c8fa541b94cf0787b7f573d36cf1233df219d00a22335c4f9a47129'
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
pilot = D/'control_v61j_v4_parity35_v1/PARITY_v1.json'
assert json.loads(pilot.read_text())['passed']
ranks = [16,293,300,310,312,316,320,328,333,378,390,400,424,459,473,474,500,530,548,550,557,623,641,651,672]
groups = [ranks[i:i+3] for i in range(0, len(ranks), 3)]
out = Path(sys.argv[1]).resolve()
out.mkdir(exist_ok=False)
first_part = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
plan = dict(candidate=str(source), sha256=expected, groups=groups,
            purpose='Current graded v61j paired controls and public geometry; no candidate endpoint selection.',
            reused_first_part=str(first_part) if first_part else None)
(out/'PLAN_v1.json').write_text(json.dumps(plan, indent=2)+'\n')
runner = R/'astra_oct18_bill_v1/run_guarded_product_census_v4.py'
with (out/'rows.jsonl').open('x') as merged:
    for index, group in enumerate(groups):
        part = out/f'part{index+1:02d}'
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
        if index == 0 and first_part is not None:
            prior = json.loads((first_part/'PLAN_v1.json').read_text())
            assert prior['sha256'] == expected and prior['ranks'] == '16,293,300'
            part.symlink_to(first_part, target_is_directory=True)
        else:
            subprocess.run([sys.executable, '-B', str(runner), str(source),
                            ','.join(map(str, group)), str(part)], check=True)
        guard = json.loads((part/'GUARD_v1.json').read_text())
        assert guard['whole_group_retired'] and guard['exit_code'] == 0
        rows = list(map(json.loads, (part/'rows.jsonl').read_text().splitlines()))
        assert [r['rank'] for r in rows] == group
        assert all(r['nonfinite'] == 0 and r['fast_rescue_stats']['calls'] == 0
                   and r['fast_rescue_control']['layer'] is None for r in rows)
        for row in rows:
            rank = row['rank']
            (out/f'pred_{rank}.bin').symlink_to(part/f'pred_{rank}.bin')
            merged.write(json.dumps(row)+'\n')
        merged.flush()
        print(json.dumps(dict(group=index+1, complete=min(3*(index+1),25), ranks=group)), flush=True)
(out/'COMPLETE_v1.json').write_text(json.dumps(dict(ranks=ranks, clean=25, sha256=expected))+'\n')
