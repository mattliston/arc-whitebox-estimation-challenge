"""Run seven fixed ranks in the same cold/warm groups as current v61j controls."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

D = Path(__file__).resolve().parent
candidate, output = map(lambda s: Path(s).resolve(), sys.argv[1:])
output.mkdir(exist_ok=False)
source_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
groups = ((35, 150, 200), (250, 259), (277, 287))
plan = dict(candidate=str(candidate), sha256=source_sha, groups=groups,
            control=str(D.parent/'astra_oct18_accuracy_v1/control_current_carrier_7_v1'))
(output/'PLAN_v1.json').write_text(json.dumps(plan, indent=2)+'\n')
rows = []
for index, ranks in enumerate(groups):
    part = output/f'part{index+1}'
    assert hashlib.sha256(candidate.read_bytes()).hexdigest() == source_sha
    subprocess.run([sys.executable, '-B', str(D/'run_guarded_candidate_v2.py'),
                    str(candidate), ','.join(map(str, ranks)), str(part)], check=True)
    guard = json.loads((part/'GUARD_v1.json').read_text())
    assert guard['whole_group_retired'] and guard['exit_code'] == 0
    got = list(map(json.loads, (part/'rows.jsonl').read_text().splitlines()))
    assert [r['rank'] for r in got] == list(ranks)
    assert all(r['nonfinite'] == 0 and r['fast_rescue_stats']['calls'] == 0
               and r['fast_rescue_control']['layer'] is None for r in got)
    for rank in ranks:
        (output/f'pred_{rank}.bin').symlink_to(part/f'pred_{rank}.bin')
    rows.extend(got)
    print(json.dumps(dict(group=index+1, clean_ranks=list(ranks))), flush=True)
assert len(rows) == 7
with (output/'rows.jsonl').open('x') as f:
    for row in rows:
        f.write(json.dumps(row)+'\n')
subprocess.run([sys.executable, '-B', str(D.parent/'opus_fix336687_v1/grade_store_v1.py'),
                str(output), plan['control'], '--layers'], check=True)
