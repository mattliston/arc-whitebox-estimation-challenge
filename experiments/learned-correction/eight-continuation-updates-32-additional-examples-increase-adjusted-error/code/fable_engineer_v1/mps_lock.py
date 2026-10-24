#!/usr/bin/env python3
"""TAG fable-engineer (E-68/E-69): FIFO queue lock for the box's ONE MPS job at a time (Matt's rule; lead: one MPS job, FIFO).
Tickets live in /tmp/fable_mps_queue/<time_ns>_<pid>.json (label, pid, the process's start time). A job holds MPS when its ticket is the
oldest LIVE ticket (pid alive AND same start time, so a recycled pid never holds a dead ticket's place). Dead tickets are ignored,
never deleted by others; a job deletes only its own ticket (on release, or at exit). Check at any time: mps_lock.py status.
Shell use (acquire, run the command as a child, release when it exits; the ticket names the COMMAND's pid while it runs):
    python3 fable_engineer_v1/mps_lock.py run <label> -- <command ...>
Python use:
    import mps_lock
    with mps_lock.hold('my label'): ...          # blocks FIFO; polls every 20 s; prints one WAIT line and one HOLD line
Put it AFTER the machine gate (fable_porter_v1/machine_gate.sh): memory gate first, then the MPS queue.
v2 (2026-09-25, physicist's find): v1's hold() was a generator context manager, so the one-liner mps_lock.hold(label).__enter__() with
the object discarded let CPython collect the generator at once; its finally deleted the ticket right after HOLD printed and the job ran
WITHOUT the lock (refit_mps_v3.py and its copies; two MPS jobs ran at once). v2's hold is a class: an entered hold is kept in the
module-level _ACTIVE list until __exit__ or process exit, so the one-liner is safe too (the ticket lives until the process exits).
v1 is kept as mps_lock_v1_gc_bug.py.bak. Tests: mps_lock_selftest.py. FABLE_MPS_QUEUE_DIR overrides the queue directory (tests only)."""
import os, sys, json, time, glob, atexit, subprocess

QDIR = os.environ.get('FABLE_MPS_QUEUE_DIR', '/tmp/fable_mps_queue')
_ACTIVE = []   # every entered, not yet exited hold: a strong reference, so a discarded hold object can never release early


def _lstart(pid):
    r = subprocess.run(['ps', '-o', 'lstart=', '-p', str(pid)], capture_output=True, text=True)
    return r.stdout.strip()


def _live(t):
    try:
        os.kill(t['pid'], 0)
    except OSError:
        return False
    return _lstart(t['pid']) == t['lstart']


def tickets():
    out = []
    for f in sorted(glob.glob(os.path.join(QDIR, '*.json'))):
        try:
            t = json.load(open(f)); t['file'] = f; out.append(t)
        except (OSError, ValueError):
            pass
    return out


def status():
    live = [t for t in tickets() if _live(t)]
    for i, t in enumerate(live):
        print(('HOLDS ' if i == 0 else 'waits ') + f"pid {t['pid']} since {t['queued']}: {t['label']}")
    if not live:
        print('MPS queue empty')


def _write(pid, label):
    os.makedirs(QDIR, exist_ok=True)
    t = dict(pid=pid, lstart=_lstart(pid), label=label, queued=time.strftime('%F %T'))
    f = os.path.join(QDIR, f'{time.time_ns()}_{pid}.json')
    tmp = f + '.tmp'
    json.dump(t, open(tmp, 'w')); os.rename(tmp, f)
    return f


def _wait_first(f, label, poll):
    said = False
    while True:
        live = [t for t in tickets() if _live(t)]
        if live and live[0]['file'] == f:
            print(f"{time.strftime('%T')} MPS queue HOLD: {label}", flush=True)
            return
        if not said:
            ahead = [t['label'] for t in live if t['file'] < f]
            print(f"{time.strftime('%T')} MPS queue WAIT: {label} (ahead: {ahead})", flush=True); said = True
        time.sleep(poll)


def _release(f):
    try:
        os.remove(f)
    except OSError:
        pass


class hold:
    """FIFO hold of the box's one MPS job. `with mps_lock.hold(label): ...` releases when the block exits. The one-liner
    mps_lock.hold(label).__enter__() is also safe: the entered hold stays in _ACTIVE until __exit__ or process exit."""

    def __init__(self, label, poll=20):
        self.label, self.poll, self.f = label, poll, None

    def __enter__(self):
        self.f = _write(os.getpid(), self.label)
        _ACTIVE.append(self)
        atexit.register(_release, self.f)
        try:
            _wait_first(self.f, self.label, self.poll)
        except BaseException:
            self._drop()
            raise
        return self

    def __exit__(self, *exc):
        self._drop()
        return False

    def _drop(self):
        if self.f is not None:
            _release(self.f)
        if self in _ACTIVE:
            _ACTIVE.remove(self)


def run(label, cmd, poll=20):
    # queue under THIS pid, then hand the ticket to the child once it runs (the child's pid is what must stay alive)
    f = _write(os.getpid(), label)
    try:
        _wait_first(f, label, poll)
        p = subprocess.Popen(cmd)
        g = f.replace(f'_{os.getpid()}.json', f'_{p.pid}.json')
        t = json.load(open(f)); t.update(pid=p.pid, lstart=_lstart(p.pid)); json.dump(t, open(g + '.tmp', 'w')); os.rename(g + '.tmp', g)
        _release(f); f = g                     # same queue time: the child keeps the head position
        return p.wait()
    finally:
        _release(f)


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'status':
        status()
    elif len(sys.argv) >= 4 and sys.argv[1] == 'run' and '--' in sys.argv:
        k = sys.argv.index('--')
        sys.exit(run(' '.join(sys.argv[2:k]), sys.argv[k + 1:]))
    else:
        sys.exit(__doc__)
