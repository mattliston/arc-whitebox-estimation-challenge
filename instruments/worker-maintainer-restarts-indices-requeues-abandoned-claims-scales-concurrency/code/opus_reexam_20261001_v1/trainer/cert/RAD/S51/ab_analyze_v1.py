"""TAG opus-trainer RAD S51 (10-12 14:5xZ; lead 14:5xZ A/B): rates and per-eval times of S51's phase-2 results in the two A/B windows of
cert/RAD/S51/ab_count_v1.sh (window A = 8 workers, window B = 12; index w3 runs TRAINER_RELIEF=0 in both). Pre-declared decisions (TEAM_LOG line
of this run's start): WORKER_CAP for S52+ = the faster window's count (evals / min from results mtimes); LAUNCHER_FLAVOR = r0 for S52+ iff w3's
mean forward seconds is <= 0.90 x the other workers' mean in BOTH windows and w3's peak RSS stays <= 6.5e9 bytes. Writes the files (old ones
archived), one TEAM_LOG line, and cert/RAD/S51/AB_v1.json.
usage (from O/trainer): python -B cert/RAD/S51/ab_analyze_v1.py <A0 epoch> <A1 epoch> <B0 epoch> <B1 epoch>"""
import glob
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

A0, A1, B0, B1 = map(float, sys.argv[1:5])
rows = []
for f in glob.glob('run_RAD_S51_gen/results/a_RS51_*.json'):
    m = os.path.getmtime(f)
    if A0 <= m < A1 or B0 <= m < B1:
        d = json.loads(Path(f).read_text())
        rows.append(dict(t=m, s=float(d['seconds']), w=d['worker'], rss=float(d.get('peak_rss') or 0), win='A' if A0 <= m < A1 else 'B'))
out = {}
for win, (t0, t1), n in (('A', (A0, A1), 8), ('B', (B0, B1), 12)):
    r = [x for x in rows if x['win'] == win]
    r3, ro = [x for x in r if x['w'] == 'w3'], [x for x in r if x['w'] != 'w3']
    out[win] = dict(workers=n, minutes=round((t1 - t0) / 60, 1), evals=len(r), rate_per_min=round(len(r) / ((t1 - t0) / 60), 3),
                    mean_forward_s=round(st.mean(x['s'] for x in r), 1) if r else None,
                    w3_evals=len(r3), w3_mean_s=round(st.mean(x['s'] for x in r3), 1) if r3 else None,
                    others_mean_s=round(st.mean(x['s'] for x in ro), 1) if ro else None,
                    w3_peak_rss_gb=round(max((x['rss'] for x in r3), default=0) / 1e9, 2), others_peak_rss_gb=round(max((x['rss'] for x in ro), default=0) / 1e9, 2))
cap = 8 if out['A']['rate_per_min'] > out['B']['rate_per_min'] else 12
r0 = all(out[w]['w3_mean_s'] is not None and out[w]['others_mean_s'] and out[w]['w3_mean_s'] <= 0.90 * out[w]['others_mean_s'] for w in 'AB') \
    and max(out[w]['w3_peak_rss_gb'] for w in 'AB') <= 6.5
stamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
for name, value in (('WORKER_CAP', str(cap)), ('LAUNCHER_FLAVOR', 'r0' if r0 else '')):
    p = Path('cert/RAD') / name
    if p.exists():
        p.rename(p.with_name('%s.archived_%s' % (name, stamp.replace(':', '').replace('-', ''))))
    if value:
        p.write_text(value + '\n')
Path('cert/RAD/S51/AB_v1.json').write_text(json.dumps(dict(utc=stamp, windows=out, worker_cap=cap, launcher_flavor='r0' if r0 else None), indent=1))
f = lambda w: '%d workers %.1f min: %d evals = %.2f/min, mean forward %ss (w3 r0 %ss on %d evals vs others %ss; peak RSS w3 %.2f GB vs others %.2f GB)' % (
    out[w]['workers'], out[w]['minutes'], out[w]['evals'], out[w]['rate_per_min'], out[w]['mean_forward_s'], out[w]['w3_mean_s'], out[w]['w3_evals'],
    out[w]['others_mean_s'], out[w]['w3_peak_rss_gb'], out[w]['others_peak_rss_gb'])
line = ('[trainer] %s S51 A/B DONE (lead 14:5xZ; same phase 2; trainer/cert/RAD/S51/AB_v1.json): A = %s | B = %s. By the pre-declared rule: WORKER_CAP '
        '= %d for every round on maintain_workers_v5 (S52+), LAUNCHER_FLAVOR = %s.' % (stamp, f('A'), f('B'), cap, 'r0 (TRAINER_RELIEF=0 for every new worker)' if r0 else 'none (relief stays on)'))
with open('../TEAM_LOG.md', 'a') as fh:
    fh.write(line + '\n')
print(line)
