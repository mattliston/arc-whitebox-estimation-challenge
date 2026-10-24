#!/usr/bin/env python3
"""TAG analyst A11: keep the board scan current. One call = one polite re-scan of NEW submission ids only:
  1. board state before = best graded score per team from every scan file (O/board_evals/scan_332296_333380.jsonl + analyst/scan_*.jsonl);
  2. run a5_scan_driver.py from (highest existing id + 1), first re-reading the ids that were still being graded at the last scan
     (status SUBMITTED), into a new file scan_<start>_plus_<UTC>.jsonl (scan files are never rewritten);
  3. rewrite the merged view scan_merged.jsonl (one record per id, the graded one where it exists) and the leaders table
     (a5_leaders.py -> a5_leaders.json, a5_leaders_<UTC>.md);
  4. compare the top six teams before and after. ALERT = a team in the top six whose board score changed by more than 2%, or a
     new team in the top six. CHANGE = any smaller score change of a top-six team, or a change of the top-six order.
     Both write one line to O/TEAM_LOG.md (the lead's rule since 2026-10-02 03:20Z: log ANY top-six change; before that only
     alerts were logged, previous version in _stale/). Every run writes a11_status_<UTC>.json and a line to a11_rescan.log.
  5. (since 2026-10-02 12:40Z) every alert / note carries the new entry's residual and wall (since 10-10 03:20Z also its final / layers-0-14 error ratio next to ours), and after every scan the derived tables are
     rebuilt on the new record: a18_third_place.md (rivals' pace and where each stands at the close) and a16_loss_risk.md (risk of each
     team's current best in a fresh run, expected ranking). Their hand-written readings are dated and are NOT rebuilt.
usage: a11_rescan.py [threshold_pct=2.0]"""
import sys, json, glob, subprocess
from pathlib import Path
from datetime import datetime, timezone
HERE = Path(__file__).resolve().parent; O = HERE.parent; THR = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
def load():
    rows = {}
    for f in [str(O / 'board_evals/scan_332296_333380.jsonl')] + sorted(glob.glob(str(HERE / 'scan_3*.jsonl'))):
        for line in open(f):
            r = json.loads(line)
            if r['id'] in rows and rows[r['id']].get('status') == 'GRADED' and r.get('status') != 'GRADED': continue
            rows[r['id']] = r
    return rows
def board(rows):
    best = {}
    for r in rows.values():
        if r.get('status') != 'GRADED' or r.get('score') is None or (r.get('agg') or {}).get('n_failed_mlps', 0): continue
        k = r.get('team') or r['who']
        if k not in best or r['score'] < best[k]['score']: best[k] = r
    return sorted(((k, r['score'], r['id'], r.get('score2'), (r.get('agg') or {}).get('mean_compute_utilization')) for k, r in best.items()), key=lambda x: x[1])
now = datetime.now(timezone.utc); stamp = now.strftime('%Y%m%dT%H%MZ')
rows = load(); before = board(rows)
last = max(i for i, r in rows.items() if r.get('http') == 200)
pending = sorted(i for i, r in rows.items() if r.get('http') == 200 and r.get('status') == 'SUBMITTED' and i > last - 60)
out = HERE / f'scan_{last + 1}_plus_{stamp}.jsonl'
subprocess.run([sys.executable, str(HERE / 'a5_scan_driver.py'), str(last + 1), str(out), '6', ','.join(map(str, pending))], check=True, stdout=open(HERE / f'a11_scan_{stamp}.log', 'w'), stderr=subprocess.STDOUT)
rows = load(); after = board(rows)
with open(HERE / 'scan_merged.jsonl', 'w') as f:
    for i in sorted(rows): f.write(json.dumps(rows[i]) + '\n')
md = subprocess.run([sys.executable, str(HERE / 'a5_leaders.py'), now.strftime('%Y-%m-%dT%H:%M:00Z')], check=True, capture_output=True, text=True).stdout
(HERE / f'a5_leaders_{stamp}.md').write_text(md); (HERE / 'a5_leaders.md').write_text(md)
b6, a6 = before[:6], after[:6]; bd = {k: (s, i) for k, s, i, *_ in before}; alerts = []; notes = []
def shape(i):      # (final-layer error, its ratio to the mean error of layers 0-14 = (16 x all-layer - final) / 15) of an entry
    r = rows.get(i) or {}; a = r.get('agg') or {}; fin = r.get('score2') or a.get('final_layer_mse'); al = a.get('all_layers_mse')
    return (fin, fin / ((16 * al - fin) / 15)) if fin and al and 16 * al > fin else None
OUR_ID = next((i for k, s_, i, *_ in after if k == 'mliston'), None)
def tele(i):
    r = rows.get(i) or {}; t = ''
    if r.get('resid') and r.get('wall'): t = f", residual {r['resid'][0]:.3f} / {r['resid'][1]:.3f} s, wall {r['wall'][0]:.0f} / {r['wall'][1]:.0f} s"
    sh, us = shape(i), (shape(OUR_ID) if OUR_ID else None)      # lead 10-10 03:08Z: log the new best's final / layers-0-14 ratio next to ours
    if sh: t += f", final / layers 0-14 {sh[1]:.2f}" + (f" (ours #{OUR_ID}: final-layer MSE {us[0]:.4e}, ratio {us[1]:.2f})" if us and i != OUR_ID else '')
    return t
for rank, (k, s, i, fin, share) in enumerate(a6, 1):
    if k not in bd or k not in [x[0] for x in b6] and abs(s / bd[k][0] - 1) * 100 > THR:
        alerts.append(f'{k} entered the top six at place {rank}: #{i} {s:.4e}' + ('' if k not in bd else f' (was {bd[k][0]:.4e})'))
    elif abs(s / bd[k][0] - 1) * 100 > THR:
        alerts.append(f'{k} (place {rank}) {bd[k][0]:.4e} -> {s:.4e} ({(s / bd[k][0] - 1) * 100:+.1f}%), entry #{i}, final-layer MSE {fin:.4e}, budget share {share:.4f}' + tele(i))
    elif s != bd[k][0]:
        notes.append(f'{k} (place {rank}) {bd[k][0]:.4e} -> {s:.4e} ({(s / bd[k][0] - 1) * 100:+.2f}%), entry #{i}, final-layer MSE {fin:.4e}, budget share {share:.4f}' + tele(i))
# a change below the threshold (or an order change) is a note: since 2026-10-02 03:20Z the lead wants ANY top-six change in TEAM_LOG,
# so notes are written there too (as BOARD CHANGE; above the threshold or a new entrant = BOARD ALERT)
if [k for k, *_ in b6] != [k for k, *_ in a6]: notes.append('top-six order: ' + ' > '.join(k for k, *_ in b6) + '  ->  ' + ' > '.join(k for k, *_ in a6))
new_ids = sorted(i for i, r in rows.items() if i > last and r.get('http') == 200)
status = dict(TAG='analyst A11 re-scan', utc=now.isoformat(), threshold_pct=THR, scanned_from=last + 1, new_ids=len(new_ids), last_existing=max([last] + new_ids), rechecked=pending,
              top6_before=b6, top6_after=a6, small_changes=[(k, bd[k][0], s) for k, s, *_ in a6 if k in bd and s != bd[k][0]], alerts=alerts, notes=notes, file=str(out))
json.dump(status, open(HERE / f'a11_status_{stamp}.json', 'w'), indent=1)
line = (f"{now:%Y-%m-%dT%H:%MZ} re-scan: {len(new_ids)} new ids (to {status['last_existing']}), top six " + ', '.join(f'{k} {s:.4e}' for k, s, *_ in a6)
        + (' | ALERT: ' + ' ; '.join(alerts) if alerts else ' | no top-six change above %.0f%%' % THR) + (' | ' + ' ; '.join(notes) if notes else ''))
with open(HERE / 'a11_rescan.log', 'a') as f: f.write(line + '\n')
if alerts or notes:
    with open(O / 'TEAM_LOG.md', 'a') as f: f.write(f'[analyst] {now:%Y-%m-%dT%H:%MZ} BOARD {"ALERT" if alerts else "CHANGE"} (automatic re-scan, ids to {status["last_existing"]}): ' + ' ; '.join(alerts + notes) + f'. Table: O/analyst/a5_leaders_{stamp}.md\n')
for script in ('a18_third_place.py', 'a16_loss_risk.py'):
    try: subprocess.run(['nice', '-n', '20', sys.executable, '-B', str(HERE / script)], check=True, stdout=subprocess.DEVNULL, stderr=open(HERE / 'a11_refresh.err', 'a'), timeout=600)
    except Exception as exc: open(HERE / 'a11_refresh.err', 'a').write(f'{now:%Y-%m-%dT%H:%MZ} {script}: {exc}\n')
print(line)
