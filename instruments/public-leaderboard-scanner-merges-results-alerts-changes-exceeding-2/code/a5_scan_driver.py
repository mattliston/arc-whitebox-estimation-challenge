#!/usr/bin/env python3
"""TAG analyst A5: extend the public submission scan past the lead's file (O/board_evals/scan_332296_333380.jsonl) with the
unmodified scan_submissions.py (copied into this directory): chunks of 8 ids, sequential, and it stops after K consecutive ids
that do not exist yet (HTTP 302). Ids given in RECHECK (not graded at the time of the lead's scan) are read first.
usage: a5_scan_driver.py START OUT.jsonl [K=6] [RECHECK ids comma list]"""
import sys, json, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
start, out = int(sys.argv[1]), sys.argv[2]; K = int(sys.argv[3]) if len(sys.argv) > 3 else 6
recheck = [int(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 and sys.argv[4] else []
def scan(a, b): subprocess.run([sys.executable, str(HERE / 'scan_submissions.py'), str(a), str(b), out], check=True)
for sid in recheck: scan(sid, sid)
s = start
while True:
    scan(s, s + 7); s += 8
    rows = {}
    for line in open(out):
        r = json.loads(line); rows[r['id']] = r
    tail = [rows[i].get('http') for i in sorted(rows) if i >= start][-K:]
    print(json.dumps(dict(scanned_to=s - 1, tail_http=tail)), flush=True)
    if len(tail) == K and all(h != 200 for h in tail): break
print(json.dumps(dict(done=True, last_existing=max((i for i, r in rows.items() if r.get('http') == 200), default=None))), flush=True)
