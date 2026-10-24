#!/usr/bin/env python3
"""TAG fable-engineer (2026-09-25): self-test of the MPS queue lock after the discarded-hold bug, in a PRIVATE queue directory (the real
/tmp/fable_mps_queue is never touched). Each case runs child processes and watches the queue from outside while they live.
  A  mps_lock v2, the v3 one-liner  mps_lock.hold(label).__enter__()  (object discarded, gc.collect()): ticket present all run, gone after exit
  B  mps_lock v1 (mps_lock_v1_gc_bug.py.bak), same one-liner: reproduces the bug (expect NO ticket while the child runs)
  C  refit_mps_v4.py --lock-selftest: its own in-process checks PASS, ticket seen from outside during the run, gone after
  D  FIFO with v2 with-blocks: a second holder waits until the first releases
usage: python3 mps_lock_selftest.py"""
import os, sys, time, json, subprocess, tempfile, glob
from pathlib import Path
HERE = Path(__file__).resolve().parent
Q = tempfile.mkdtemp(prefix='mps_lock_selftest_')
env = dict(os.environ, FABLE_MPS_QUEUE_DIR=Q, PYTHONPATH=str(HERE))
def tickets():
    out = []
    for f in glob.glob(os.path.join(Q, '*.json')):
        try: out.append(json.load(open(f)))
        except Exception: pass
    return out
def watch(proc, pid_of_interest, span, every=0.5):
    seen = []
    t0 = time.time()
    while time.time() - t0 < span and proc.poll() is None:
        seen.append(any(t['pid'] == pid_of_interest for t in tickets())); time.sleep(every)
    return seen
def child(code):
    return subprocess.Popen([sys.executable, '-c', code], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
results = {}
# A: v2 one-liner
pa = child("import mps_lock, gc, time\nmps_lock.hold('A', poll=1).__enter__()\ngc.collect()\ntime.sleep(4)")
time.sleep(0.8); seenA = watch(pa, pa.pid, 3.5); pa.wait(); time.sleep(0.3)
results['A v2 one-liner: held while running'] = bool(seenA) and all(seenA)
results['A v2 one-liner: released at exit'] = not any(t['pid'] == pa.pid for t in tickets())
# B: v1 one-liner (the bug)
v1 = str(HERE / 'mps_lock_v1_gc_bug.py.bak')
pb = child("import importlib.machinery, importlib.util, gc, time\n"
           f"ld = importlib.machinery.SourceFileLoader('mps_lock_v1', {v1!r}); sp = importlib.util.spec_from_loader('mps_lock_v1', ld)\n"
           "m = importlib.util.module_from_spec(sp); ld.exec_module(m); m.QDIR = __import__('os').environ['FABLE_MPS_QUEUE_DIR']\n"
           "m.hold('B', poll=1).__enter__()\ngc.collect()\ntime.sleep(4)")
time.sleep(0.8); seenB = watch(pb, pb.pid, 3.5); pb.wait()
results['B v1 one-liner reproduces the bug (no ticket while running)'] = bool(seenB) and not any(seenB)
# C: refit_mps_v4.py lock self-test
pc = subprocess.Popen([sys.executable, str(HERE / 'refit_mps_v4.py'), '--lock-selftest', '4', '--tag', 'selftest'], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
time.sleep(1.0); seenC = watch(pc, pc.pid, 3.0); outC = pc.communicate()[0]; time.sleep(0.3)
results['C refit_mps_v4 in-process checks'] = 'LOCK SELFTEST PASS' in outC
results['C refit_mps_v4 ticket seen from outside'] = bool(seenC) and all(seenC)
results['C refit_mps_v4 released after the run'] = not any(t['pid'] == pc.pid for t in tickets())
# D: FIFO
p1 = child("import mps_lock, time\nwith mps_lock.hold('D1', poll=1):\n    time.sleep(4)")
time.sleep(1.0)
p2 = child("import mps_lock, time\nt0 = time.time()\nwith mps_lock.hold('D2', poll=1):\n    print('D2 got it after %.1f s' % (time.time() - t0), flush=True)")
o1 = p1.communicate()[0]; o2 = p2.communicate()[0]
waited = [float(x.split('after ')[1].split(' s')[0]) for x in o2.splitlines() if 'D2 got it' in x]
results['D FIFO: second holder waited for the first'] = bool(waited) and waited[0] >= 2.0 and 'WAIT' in o2
for k, v in results.items(): print(('PASS ' if v else 'FAIL ') + k)
print('ALL PASS' if all(results.values()) else 'SOME FAIL', '| queue dir', Q, '| leftover tickets', len(tickets()))
