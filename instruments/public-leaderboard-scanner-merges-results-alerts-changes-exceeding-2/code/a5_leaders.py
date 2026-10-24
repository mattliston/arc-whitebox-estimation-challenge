#!/usr/bin/env python3
"""TAG analyst A5: leaders table from the public submission scans (the lead's O/board_evals/scan_332296_333380.jsonl plus every
O/analyst/scan_*.jsonl; for an id present in several files the GRADED record wins, else the last one read).
Team = team name if the submission has one, else the participant name. Board entry of a team at time T = its graded submission
with the LOWEST adjusted score among those created at or before T. For the 12 best teams now: the board entry now, the board
entry 72 hours earlier, the change of each metric, the latest graded submission, and the count of graded submissions in the window.
usage: a5_leaders.py [now ISO, default = newest created time in the scans] -> a5_leaders.json + markdown on stdout"""
import sys, json, glob
from pathlib import Path
from datetime import datetime, timedelta, timezone
HERE = Path(__file__).resolve().parent; O = HERE.parent
files = [str(O / 'board_evals/scan_332296_333380.jsonl')] + sorted(glob.glob(str(HERE / 'scan_*.jsonl')))
rows = {}
for f in files:
    for line in open(f):
        r = json.loads(line)
        if r['id'] in rows and rows[r['id']].get('status') == 'GRADED' and r.get('status') != 'GRADED': continue
        rows[r['id']] = r
ts = lambda s: datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
graded = [r for r in rows.values() if r.get('status') == 'GRADED' and r.get('score') is not None and (r.get('agg') or {}).get('n_failed_mlps', 0) == 0]
for r in graded: r['t'] = ts(r['created']); r['key'] = r.get('team') or r['who']
now = ts(sys.argv[1]) if len(sys.argv) > 1 else max(r['t'] for r in graded)
then = now - timedelta(hours=72)
def entry(key, T):
    c = [r for r in graded if r['key'] == key and r['t'] <= T]
    return min(c, key=lambda r: r['score']) if c else None
def rec(r):
    if r is None: return None
    a = r['agg']
    return dict(id=r['id'], created=r['created'], who=r['who'], score=r['score'], final_mse=r.get('score2') or a['final_layer_mse'], all_mse=a['all_layers_mse'],
                share=a['mean_compute_utilization'], wall_mean=(r.get('wall') or [None, None])[0], wall_max=(r.get('wall') or [None, None])[1])
keys = sorted({r['key'] for r in graded}, key=lambda k: entry(k, now)['score'])[:12]
out = []
for i, k in enumerate(keys):
    n, o = rec(entry(k, now)), rec(entry(k, then))
    mine = sorted((r for r in graded if r['key'] == k), key=lambda r: r['t'])
    last = rec(mine[-1]); cnt = sum(1 for r in mine if r['t'] > then)
    ch = {m: (n[m] / o[m] - 1) * 100 for m in ('score', 'final_mse', 'all_mse', 'share', 'wall_mean', 'wall_max')} if o else None
    out.append(dict(place=i + 1, team=k, now=n, then=o, change_pct=ch, latest=last, graded_in_window=cnt, first_seen=mine[0]['created']))
json.dump(dict(now=now.isoformat(), then=then.isoformat(), files=files, ids=[min(rows), max(i for i, r in rows.items() if r.get('http') == 200)], teams=out),
          open(HERE / 'a5_leaders.json', 'w'), indent=1)
pc = lambda x: '' if x is None else f'{x:+.1f}%'
print(f'Board entries at {now:%Y-%m-%d %H:%M} UTC and their change since {then:%Y-%m-%d %H:%M} UTC (72 h). Scan ids {min(rows)}-{max(i for i, r in rows.items() if r.get("http") == 200)}.\n')
print('| # | team | board entry (id, UTC) | adjusted score | final-layer MSE | all-layer MSE | budget share | wall mean / max (s) | change in 72 h: score | final MSE | all-layer MSE | budget share | wall mean / max | graded entries in 72 h | latest graded entry (id: score) |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for t in out:
    n, c, l = t['now'], t['change_pct'], t['latest']
    chs = [pc(c[m]) for m in ('score', 'final_mse', 'all_mse', 'share')] + [f"{pc(c['wall_mean'])} / {pc(c['wall_max'])}"] if c else ['new'] * 5
    print(f"| {t['place']} | {t['team']} | #{n['id']}, {n['created'][5:16].replace('T', ' ')} | {n['score']:.4e} | {n['final_mse']:.4e} | {n['all_mse']:.3e} | {n['share']:.4f} | "
          f"{n['wall_mean']:.0f} / {n['wall_max']:.0f} | " + ' | '.join(chs) + f" | {t['graded_in_window']} | #{l['id']}: {l['score']:.4e} |")
