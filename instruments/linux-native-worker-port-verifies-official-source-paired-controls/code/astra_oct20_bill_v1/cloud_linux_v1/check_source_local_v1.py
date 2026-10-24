"""Light source/metadata checks only. Does not execute an estimator or Linux job."""
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import guard_linux_v1 as guard
from common_linux_v1 import no_governor, sha, utc, write_new

here = Path(__file__).resolve().parent
for file in here.glob('*.py'):
    compile(file.read_text(), str(file), 'exec')

# Metadata-only process ownership, including a descendant that changes its PGID.
table = {100: dict(pid=100, ppid=1, pgid=100, start_ticks=1, rss_bytes=100),
         101: dict(pid=101, ppid=100, pgid=100, start_ticks=2, rss_bytes=200),
         102: dict(pid=102, ppid=101, pgid=999, start_ticks=3, rss_bytes=300),
         103: dict(pid=103, ppid=1, pgid=103, start_ticks=4, rss_bytes=400)}
known = {100: 1}
assert set(guard.members(table, 100, known)) == {100, 101, 102}
reused = {100: dict(pid=100, ppid=1, pgid=100, start_ticks=10, rss_bytes=100),
          104: dict(pid=104, ppid=100, pgid=100, start_ticks=11, rss_bytes=100)}
assert guard.members(reused, 100, known) == {}, 'PID/PGID reuse must not be owned'

# Atomic reservation math with fake /proc/meminfo, not a Linux runtime claim.
pid = os.getpid()
guard.processes = lambda: {pid: dict(pid=pid, ppid=1, pgid=pid, start_ticks=77, rss_bytes=100)}
guard.memory = lambda: dict(total=120000000000, available=100000000000)
with tempfile.TemporaryDirectory(prefix='arc-source-check-') as tmp:
    admission = guard.Admission(Path(tmp))
    ok, _ = admission.update(dict(start_ticks=77, pgid=None), cap=7900000000, max_active=8, reserve=24000000000)
    assert ok
    admission.update(dict(start_ticks=77, pgid=pid))
    admission.update({}, remove=True)
    assert json.loads(admission.state.read_text()) == {}
    guard.memory = lambda: dict(total=120000000000, available=30000000000)
    ok, _ = admission.update(dict(start_ticks=77, pgid=None), cap=7900000000, max_active=8, reserve=24000000000)
    assert not ok, 'reserve+nextcap admission must fail'

tree = ast.parse((here / 'client_linux_v1.py').read_text())
evaluate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate')
context = next(n for n in evaluate.body if isinstance(n, ast.With))
calls = [ast.unparse(n.func) for n in ast.walk(context) if isinstance(n, ast.Call)]
assert all(x in calls for x in ['flops.BudgetContext', 'fnp.load', 'est.predict', 'prediction._fetch_data', 'ctx.summary_dict'])
assert any(isinstance(n, ast.Delete) and {x.id for x in n.targets} == {'blob', 'weights', 'prediction'} for n in context.body)
plan = json.loads((here / 'RUN_PLAN_v1.json').read_text())
assert [r for group in plan['groups'] for r in group] == plan['first7'] + plan['held25']
score = json.loads((here / 'MAC_SCORER_CHECK_v2.json').read_text())
assert score['all']['n'] == 32 and abs(score['all']['adjusted_pct'] - (-.14744406631038778)) < 1e-12
assert abs(score['held25']['adjusted_pct'] - (-.17037174930746346)) < 1e-12
report = dict(utc=utc(), status='PASS', checks=['compile_all', 'owned_descendant_metadata', 'PID_PGID_reuse_metadata',
              'atomic_admission_metadata', 'metered_scope_AST', 'original32group_order', 'savedMac32_scorer_reproduction'],
              no_Linux_runtime_executed=True, no_estimator_import=True, no_remote_operations=True,
              bank_adjusted_reference_check=score['all'],
              source_hashes={p.name: sha(p) for p in sorted(here.glob('*.py'))})
write_new(here / 'LOCAL_CHECK_v1.json', report)
print(json.dumps(dict(status='PASS', receipt=str(here / 'LOCAL_CHECK_v1.json'))))
