"""[opus-pubgov] hourly public-board watch: runs analyst/a11_rescan_astra_v2.py --scan (public submission pages only; no credentials,
no forum), reads its report.json, and compares the best entry of oliver_chen, J2W, marius_binner, Luna and us (mliston) with the
previous report this watch read; any change (new id or score) is appended to TEAM_LOG with a date -u stamp taken at write time.
State: board_watch_state_v1.json (last report per team). usage: board_watch_v1.py [--no-scan]"""
import json, subprocess, sys
from pathlib import Path
G = Path(__file__).resolve().parent; O = G.parents[2]; A = O/'analyst'; LOG = O/'TEAM_LOG.md'; ST = G/'board_watch_state_v1.json'
TEAMS = ('oliver_chen', 'J2W', 'marius_binner', 'Luna', 'mliston')
PY = '/Users/matt/arc-white-box-estimation/.venv/bin/python'
args = [PY, '-B', str(A/'a11_rescan_astra_v2.py')] + ([] if '--no-scan' in sys.argv else ['--scan'])
out = subprocess.run(args, cwd=A, capture_output=True, text=True, timeout=3000)
line = [l for l in out.stdout.splitlines() if l.startswith('{"report"')][-1]
rep = json.loads(Path(json.loads(line)['report']).read_text())
now = {r['team']: dict(place=r['place'], id=r['id'], score=r['score']) for r in rep['top12'] if r['team'] in TEAMS}
prev = json.loads(ST.read_text()) if ST.exists() else {}
changes = []
for t in TEAMS:
    a, b = prev.get(t), now.get(t)
    if b and (not a or a['id'] != b['id'] or abs(a['score'] - b['score']) > 1e-15):
        changes.append(f"{t} {'new' if not a else ('%.6e #%d' % (a['score'], a['id']))} -> {b['score']:.6e} #{b['id']} (place {b['place']})")
    elif a and a.get('place') != (b or {}).get('place'):
        changes.append(f"{t} place {a.get('place')} -> {(b or {}).get('place')}")
ts = subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()
board = ', '.join(f"{t} {now[t]['score']:.6e}" for t in sorted(now, key=lambda x: now[x]['place']))
with open(G/'board_watch_v1.log', 'a') as f:
    f.write(f"{ts} report {rep['completed']} last_http200 {rep['last_http200']} | {board} | changes: {'; '.join(changes) or 'none'}\n")
if changes and prev:
    with open(LOG, 'a') as f:
        f.write(f"\n[opus-pubgov] {ts} BOARD CHANGE (public scan, report {rep['completed']}): {'; '.join(changes)}. Board now: {board}.\n")
ST.write_text(json.dumps(now, indent=1) + '\n')
print(ts, 'changes:', changes or 'none'); print(board)
