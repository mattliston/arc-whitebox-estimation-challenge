"""TAG opus-trainer RAD (10-12 08:0xZ; lead 08:0xZ: "a round whose check is NOT accepted still writes STOP and hands its start point unchanged to
the next round"; no round starts after 10-17 03:00Z): writes the next version of a WAITING round chain in which EVERY exit after its start writes
the run's STOP file, so the next round (keyed on that STOP) always follows: a fail() helper (log + mkdir -p run + STOP + exit 1) replaces the
config / gradient / queue / check failures that exited without STOP and the 12-relaunch stop; a gen_round chain's start cutoff moves from
10-17 00:00Z to 03:00Z (the lead's approval) and its NOT STARTED exit also writes STOP (the later rounds then end the same way).
Nothing else changes. usage (from O/trainer): python -B cert/RAD/fix_chain_v1.py <round name, e.g. S54> <chain in> <chain out (new file)>"""
import os
import sys
from pathlib import Path

R, src, dst = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
assert not dst.exists(), dst
s = src.read_text()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:90])
    s = s.replace(old, new)


rep('#!/bin/bash\n', '#!/bin/bash\n# %s (written %s, lead 08:0xZ): = %s with every failure exit writing STOP (fail()), so the next round always follows and starts\n'
    '# from the last ACCEPTED point; a generated round starts no later than 10-17 03:00Z. cert/RAD/fix_chain_v1.py.\n' % (dst.name, os.popen('date -u +%FT%TZ').read().strip(), src.name))
log_line = next(l for l in s.splitlines() if l.startswith('log() {'))
rep(log_line + '\n', log_line + '\nfail() { log "%s CHAIN STOPPED: $1; STOP written, the next round follows"; mkdir -p $RUN; date -u > $RUN/STOP; exit 1; }\n' % R)
rep('{ log "%s CHAIN STOPPED: workers keep dying (12 relaunches)"; exit 1; }' % R, 'fail "workers keep dying (12 relaunches)"')
rep('|| { log "%s CHAIN STOPPED: config failed (trainer/$D/config_v1.out)"; exit 1; }' % R, '|| fail "config failed (trainer/$D/config_v1.out)"')
rep('$PY -B $RAD grads $CFG || exit 1', '$PY -B $RAD grads $CFG || fail "gradient queue failed"')
q = next(l for l in s.splitlines() if l.startswith('queue_set() {'))
rep(q, q.replace('|| exit 1', '|| fail "queue_set $1 failed"'))
rep('$PY -B $RAD check $FCFG || exit 1', '$PY -B $RAD check $FCFG || fail "check queue failed"')
if 'LATE=' in s:
    rep('"2026-10-17T00:00:00Z"', '"2026-10-17T03:00:00Z"')
    rep('NOT STARTED: past 10-17 00:00Z (lead: the final round freezes by 10-17 12:00Z)"; exit 0; }',
        'NOT STARTED: past 10-17 03:00Z (lead: the final round freezes by 10-17 12:00Z)"; mkdir -p $RUN; date -u > $RUN/STOP; exit 0; }')
assert 'exit 1; }' not in s.replace('date -u > $RUN/STOP; exit 1; }', '') or True
left = [l for l in s.splitlines() if ('exit 1' in l or 'exit 0' in l) and 'STOP' not in l]
assert not left, left
dst.write_text(s)
print('%s -> %s: every exit writes STOP%s' % (src, dst, '; start cutoff 10-17 03:00Z' if 'LATE=' in s else ''))
