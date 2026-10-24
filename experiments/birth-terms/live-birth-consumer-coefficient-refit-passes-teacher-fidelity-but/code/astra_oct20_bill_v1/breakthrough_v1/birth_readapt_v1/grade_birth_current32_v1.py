"""Strict completed-native32 comparison with exact current submission337422."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/arcworker/arc-workspace')
RUNS = Path('/home/arcworker/arc-runs')
JOINT = HERE.parents[2] / 'fable_oct19_kc_v1/agent_astra_joint_v1'
PORT = JOINT / 'cloud_linux_capacity_v7'
L = JOINT / 'cloud_linux_markline_v1'
sys.path.insert(0, str(PORT))
from common_linux_v7 import load_plan, sha, utc, verified, write_new

GRADER = L / 'grade_linux_markline_v1.py'
assert sha(GRADER) == '602cfb761c7058816c7998cf19737e0c011f13f47934369ed9f30e8dbc4ea5f2'
spec = importlib.util.spec_from_file_location('birth_grade_common', GRADER)
grade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grade)
p = argparse.ArgumentParser()
p.add_argument('--run', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
assert a.run.resolve().parent == RUNS and a.run.name == 'birth_consumer_readapt_markline32_blas4_v1'
planpath = HERE / 'PLAN_BIRTH_CONSUMER_READAPT_v1.json'
assert sha(planpath) == '8f589fbc1d645ef43422e60681965630a87c6df07806cd8f7420262541233465'
plan = load_plan(ROOT, planpath)
candidate_pin = '37760350899443d78bd9a0a8e9ee8ad46e24bce74232c5d9ef4c333818702689'
baseline_pin = '777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
assert sha(ROOT / plan['candidates']['birth_consumer_readapt']['path']) == candidate_pin
bankpath = RUNS / 'msg8_loaderfix337422_control32_markline_blas4_v1'
guards = []
for folder in (a.run, bankpath):
    complete = json.loads((folder / 'COMPLETE.json').read_text())
    assert complete['successful']
    for path in sorted(folder.glob('part[0-9][0-9]_GUARD.json')):
        g = json.loads(path.read_text())
        assert g['successful'] and g['child_exit'] == 0 and g['whole_group_retired']
        assert not g['cleanup_errors'] and g['max_owned_swap_bytes'] == 0
        assert g['guard_sha256'] == '884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
        assert all(g['cgroup_events_final'].get(k, 0) == 0 for k in ('max', 'oom', 'oom_kill'))
        guards.append(dict(path=str(path), sha256=sha(path)))
candidate, cs, cm = grade.collect(a.run)
baseline, bs, bm = grade.collect(bankpath)
ranks = [rank for group in plan['groups'] for rank in group]
assert len(ranks) == 32 and cs == candidate_pin and bs == baseline_pin
assert cm == bm and cm[1] == 4 and set(ranks) == set(candidate) == set(baseline)
records = []
for rank in ranks:
    c, b = candidate[rank], baseline[rank]
    assert c['group'] == b['group']
    truths = []
    for half in ('A', 'B'):
        rec = plan['rows'][str(rank)]['truth_' + half]
        path = verified(ROOT, rec)
        with np.load(path, allow_pickle=False) as data:
            truth = data[rec['key']].astype(np.float64)
        assert truth.shape == (16, 1024) and np.isfinite(truth).all()
        truths.append(truth)
    A, B = truths
    u1 = np.mean((c['prediction'] - A) * (c['prediction'] - B), axis=1)
    u0 = np.mean((b['prediction'] - A) * (b['prediction'] - B), axis=1)
    assert u0[15] > 0 and u1[15] > 0
    ratio = c['row']['flops'] / b['row']['flops']
    assert ratio > 0
    records.append(dict(rank=rank, raw_pct=float(100 * (u1[15] / u0[15] - 1)),
        bill_pct=float(100 * (ratio - 1)), adjusted_pct=float(100 * (u1[15] / u0[15] * ratio - 1)),
        u1=float(u1[15]), u0=float(u0[15]), flops=c['row']['flops'], bank_flops=b['row']['flops'],
        requests=c['row']['requests'], bank_requests=b['row']['requests'],
        prediction_sha256=c['prediction_sha256'], bank_prediction_sha256=b['prediction_sha256'],
        max_abs_dy=float(np.max(np.abs(c['prediction'] - b['prediction']))),
        layer_raw_pct=[100 * (float(x) / float(y) - 1) if y else None for x, y in zip(u1, u0)]))
report = dict(utc=utc(), successful=True, reference='paired same-host actual337422', reference_submission=337422,
    candidate_sha256=cs, bank_sha256=bs, plan_sha256=sha(planpath), source_sha256=sha(__file__),
    setup_mode=cm, groups_exact=True, matched_cold_warm_order=True, guards=guards,
    all=grade.summary(records), first7=grade.summary([r for r in records if r['rank'] in plan['first7']]),
    held25=grade.summary([r for r in records if r['rank'] in plan['held25']]), rows=records,
    release_qualified=False, fresh_verified=False, no_submission=True)
assert report['first7']['n'] == 7 and report['held25']['n'] == 25
write_new(a.out, report)
print(json.dumps({k: report[k] for k in ('reference_submission', 'all', 'first7', 'held25')}))

