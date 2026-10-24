"""[fable-lead 2026-10-19] grade the COMPLETED parts of a (possibly retired) 32-rank judge run against the matching 95c control parts,
writing <run>/candidate_all32/GRADE_STORE_v1.json over the ranks it has (science only; no gates). usage: grade_partial_fable_v1.py <run dir> <n parts>"""
import json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'astra_oct18_accuracy_v1'))
from run_shared32_continuation_v1 import merge, read, GRADE, SEVEN, GROUPS
R = HERE.parent
CONTROL_RUN = R / 'astra_oct18_lead_v1/compact10_repair_qualification_v1/science32'
ALL_GROUPS = [[35, 150, 200], [250, 259], [277, 287]] + GROUPS
run, n = Path(sys.argv[1]).resolve(), int(sys.argv[2])
parts, controls = [], []
for index, ranks in enumerate(ALL_GROUPS[:n], 1):
    p = run / f'candidate_part{index:02d}'
    guard = read(p / 'GUARD_v1.json')
    assert guard['completed'] and guard['error'] is None, (p, guard.get('error'))
    parts.append((p, ranks)); controls.append((CONTROL_RUN / f'candidate_part{index:02d}', ranks))
cp, bp = run / 'candidate_all32', run / 'control_all32'
merge(cp, parts, census=True); merge(bp, controls, census=True)
subprocess.run([sys.executable, '-B', str(GRADE), str(cp), str(bp), '--layers'], check=True)
g = read(cp / 'GRADE_STORE_v1.json')
print('graded %d ranks: %s' % (len(g), sorted(int(r['rank']) for r in g)))
