"""TAG opus-trainer (2026-10-01, v2 19:30 UTC): keeps up to N workers alive on a run directory; never more than 3 trainer workers on
the whole box, counted over every supervisor (process scan for O/trainer/worker.py). It adopts workers of its run directory that
are already running (a supervisor can be replaced without touching them).
START RULE. Each worker is started on its own behind the machine gate O/gate.sh: the gate is asked for 7 GB (one worker's 6.4 GB
peak, rounded up) PLUS whatever the trainer workers already running may still claim (their 6.4 GB peak minus what is resident right
now), so three idle workers are gated exactly like `gate.sh 20`. A closed gate is retried every 15 s; at most 30 starts per index.
SHED RULE. The native guard of the other lanes kills their job when system swap grows by 256 MiB. Every second this supervisor
reads the swap in use; each time it has grown by 128 MiB above its lowest reading since the last shed it terminates its highest-index worker
and starts nothing for 10 minutes. Requests claimed by a worker that is gone are put back into the queue (any worker redoes them).
WORKER COUNT ON THE FLY. If <run>/WORKER_COUNT holds a number below N, only that many workers are kept: the surplus ones (highest
indices) are terminated, their requests requeued, and nothing is started above that number until the file says so.
Ends when <run>/STOP exists and every worker of this run has left. Workers run at nice 10 (the increment allows for the 5 that zsh
gives a background supervisor) with TRAINER_THREADS and the four BLAS thread variables = <threads>. Worker indices are <first>..<first>+N-1.
usage: nohup /Users/matt/anaconda3/bin/python -B supervisor.py <run dir> <workers 1..3> [threads=1] [first=0] >> <run>/workers/supervisor.log 2>&1 &"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import psutil

from otr_common import O, THREAD_ENV, run_dirs

HERE = Path(__file__).resolve().parent
WORKER = str(HERE / 'worker.py')
RUN = run_dirs(Path(sys.argv[1]).resolve())
COUNT = int(sys.argv[2])
THREADS = int(sys.argv[3]) if len(sys.argv) > 3 else 1
FIRST = int(sys.argv[4]) if len(sys.argv) > 4 else 0
assert 1 <= COUNT <= 3
PEAK_GB = float(os.environ.get('TRAINER_PEAK_GB', 6.4))   # one worker's peak resident memory (8.8 with TRAINER_RELIEF=0)
LIMIT, SHED_BYTES, COOL_SECONDS = 3, 128 * 2 ** 20, 600
children, starts = [], {k: 0 for k in range(FIRST, FIRST + COUNT)}


def say(text):
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), text, flush=True)


def swap_used():
    text = subprocess.check_output(['/usr/sbin/sysctl', '-n', 'vm.swapusage'], text=True, timeout=5)
    return float(re.search(r'used = ([\d.]+)M', text).group(1)) * 2 ** 20


def scan():
    """(index -> process of every live worker of this run, resident GB of every trainer worker on the box)."""
    mine, resident = {}, []
    for proc in psutil.process_iter(['cmdline', 'memory_info', 'status']):
        cmd = proc.info['cmdline'] or []
        if WORKER in cmd and proc.info['status'] != psutil.STATUS_ZOMBIE and proc.info['memory_info'] is not None:
            resident.append(proc.info['memory_info'].rss / 2 ** 30)
            at = cmd.index(WORKER)
            if cmd[at + 1] == str(RUN):
                mine[int(cmd[at + 2])] = proc
    return mine, resident


def requeue(mine):
    """Requests claimed by a worker of this supervisor that is no longer alive go back to the queue for any worker."""
    for k in range(FIRST, FIRST + COUNT):
        if k in mine:
            continue
        for claim in sorted((RUN / 'claimed').glob('*.w%d.json' % k)):
            name = claim.name[:-len('.w%d.json' % k)]
            if not any((RUN / 'results' / (name + suffix)).exists() for suffix in ('.pt', '.json', '.error')):
                os.rename(claim, RUN / 'queue' / (name + '.json'))
                say('requeued %s (claimed by w%d, which is gone)' % (name, k))


def wanted():
    try:
        return max(0, min(COUNT, int((RUN / 'WORKER_COUNT').read_text())))
    except (OSError, ValueError):
        return COUNT


def gate_open(label, resident):
    need = PEAK_GB + 0.6 + sum(max(0.0, PEAK_GB - rss) for rss in resident)
    try:
        done = subprocess.run(['bash', str(O / 'gate.sh'), '%.1f' % need, label], capture_output=True, timeout=8)
    except subprocess.TimeoutExpired as closed:
        say((closed.stdout or b'').decode().strip() or 'gate closed for %s (asked %.1f GB)' % (label, need))
        return False
    say(done.stdout.decode().strip())
    return done.returncode == 0


mine, resident = scan()
say('supervisor v2 start: run %s, indices %d..%d, %d thread(s) each; adopted %s; swap in use %.0f MiB' % (
    RUN, FIRST, FIRST + COUNT - 1, THREADS, {k: proc.pid for k, proc in sorted(mine.items())}, swap_used() / 2 ** 20))
reference, cool_until, tick = swap_used(), 0.0, 0
while True:
    used = swap_used()
    reference = min(reference, used)
    if used - reference >= SHED_BYTES:
        mine, resident = scan()
        say('swap in use grew %.0f MiB to %.0f MiB' % ((used - reference) / 2 ** 20, used / 2 ** 20))
        if mine:
            victim = max(mine)
            say('shedding worker w%d pid %d; no starts for %d s' % (victim, mine[victim].pid, COOL_SECONDS))
            mine[victim].terminate()
        reference, cool_until = used, time.monotonic() + COOL_SECONDS
    if tick % 15 == 0:
        children = [child for child in children if child.poll() is None]     # reaps the ones that left
        mine, resident = scan()
        requeue(mine)
        stop = (RUN / 'STOP').exists()
        keep = wanted()
        for k in sorted(k for k in mine if FIRST <= k < FIRST + COUNT)[keep:]:
            say('WORKER_COUNT file asks for %d: terminating worker w%d pid %d' % (keep, k, mine[k].pid))
            mine[k].terminate()
        for k in range(FIRST, FIRST + keep):
            if (k not in mine and not stop and starts[k] < 30 and time.monotonic() >= cool_until and len(resident) < LIMIT
                    and gate_open('trainer-w%d' % k, resident)):
                env = dict(os.environ, TRAINER_THREADS=str(THREADS), **{name: str(THREADS) for name in THREAD_ENV})
                with (RUN / 'workers' / ('w%d.log' % k)).open('a') as log:
                    children.append(subprocess.Popen(['nice', '-n', str(max(0, 10 - os.nice(0))), sys.executable, '-B', WORKER, str(RUN), str(k)],
                                                     stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                                                     cwd=str(HERE), start_new_session=True))
                starts[k] += 1
                say('started worker w%d pid %d (start %d of this supervisor)' % (k, children[-1].pid, starts[k]))
                break   # one start per round: the next gate reading sees this worker at work
        if stop and not mine:
            break
    tick += 1
    time.sleep(1)
say('supervisor end')
