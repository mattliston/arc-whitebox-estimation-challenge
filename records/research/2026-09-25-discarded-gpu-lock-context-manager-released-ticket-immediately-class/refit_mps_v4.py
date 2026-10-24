#!/usr/bin/env python3
"""TAG fable-engineer (E-68 refit harness v4, 2026-09-25): refit_mps_v3.py with the MPS queue lock FIXED. v3 took the lock with
mps_lock.hold(...).__enter__() and discarded the object; CPython collected it at once and its finally deleted the ticket, so v3 runs
(and copies of the line) never held MPS (physicist's find). v4 holds the lock in a WITH-BLOCK around the whole run: this wrapper takes
it, then executes refit_mps_body_v4.py (= v3 minus its lock line, same arguments) inside the block; the ticket is released when the run
ends, including sys.exit and errors. CPU runs (--device cpu) take no lock. mps_lock v2 also keeps any entered hold alive until exit.
usage: same as refit_mps_v3.py, e.g.
  LOOP_DIALMOD=<dials> python3 refit_mps_v4.py --device mps --card s6bcard_loop3 --tag T --groups full+P --epochs 1 --resume P [--outdir .]
self-test (no MPS use, no training): refit_mps_v4.py --lock-selftest <seconds> [--tag T]: holds the lock through the same with-block
for <seconds>, checking its own ticket after gc.collect() at the start and the end, and that it is gone after the block."""
import os, sys, gc, time, runpy
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mps_lock

argv = sys.argv[1:]
def opt(name, default=None):
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default
tag = opt('--tag', 'untagged')
label = f'refit_mps_v4 {tag} (pid {os.getpid()})'

def own_ticket():
    return [t for t in mps_lock.tickets() if t['pid'] == os.getpid()]

if '--lock-selftest' in argv:
    secs = float(opt('--lock-selftest', '5'))
    with mps_lock.hold(label):
        gc.collect(); start_ok = len(own_ticket()) == 1
        time.sleep(secs)
        gc.collect(); end_ok = len(own_ticket()) == 1
    after_ok = len(own_ticket()) == 0
    ok = start_ok and end_ok and after_ok
    print(f'LOCK SELFTEST {"PASS" if ok else "FAIL"}: ticket held at start {start_ok}, at end {end_ok}, released after the block {after_ok}', flush=True)
    sys.exit(0 if ok else 1)

MPS = opt('--device') == 'mps'
body = str(HERE / 'refit_mps_body_v4.py')
sys.argv = [body] + argv
if MPS:
    with mps_lock.hold(label):
        runpy.run_path(body, run_name='__main__')
else:
    runpy.run_path(body, run_name='__main__')
