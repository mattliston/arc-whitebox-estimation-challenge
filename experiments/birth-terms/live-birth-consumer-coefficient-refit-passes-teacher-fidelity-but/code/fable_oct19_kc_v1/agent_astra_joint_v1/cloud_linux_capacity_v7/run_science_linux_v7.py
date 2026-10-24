"""Launch original science32 groups, each under the shared Linux guard."""
import argparse
import collections
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from common_linux_v7 import thread_env, load_plan, sha, utc, write_new


def main():
    p = argparse.ArgumentParser()
    for key in ('root', 'plan', 'out', 'client-python', 'server-python', 'state-dir'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--jobs', type=int, required=True, choices=range(1, 37))
    p.add_argument('--max-active', type=int, required=True, choices=range(1, 37), help='shared global admission ceiling, same for every concurrent controller')
    p.add_argument('--blas-threads', type=int, required=True, choices=(1, 2, 4))
    p.add_argument('--groups', help='1-based original group IDs; default all12')
    p.add_argument('--cgroup-root', type=Path)
    p.add_argument('--setup', choices=['paced', 'real'], default='paced')
    p.add_argument('--warmup-each', action='store_true')
    a = p.parse_args(); assert sys.platform == 'linux'
    threads = thread_env(a.blas_threads)
    started = time.monotonic()
    assert a.jobs <= a.max_active
    plan = load_plan(a.root, a.plan); assert a.candidate in plan['candidates']
    groups = [int(x) for x in a.groups.split(',')] if a.groups else list(range(1, len(plan['groups']) + 1))
    assert len(set(groups)) == len(groups) and all(1 <= x <= len(plan['groups']) for x in groups)
    a.out.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    write_new(a.out / 'RUN_MANIFEST.json', dict(utc=utc(), plan_sha256=sha(a.plan), candidate=a.candidate,
              candidate_sha256=plan['candidates'][a.candidate]['effective_sha256'], groups=groups,
              jobs=a.jobs, max_active=a.max_active, blas_threads=a.blas_threads, thread_environment=threads,
              retirement_cap_bytes=6500000000, empirical_admission_reservation_bytes=4800000000,
              setup=a.setup, warmup_each=a.warmup_each, science_only=True, scheduler_sha256=sha(__file__)))
    pending = collections.deque(groups); active = {}; results = {}; failed = False
    def interrupted(*_):
        raise KeyboardInterrupt('science controller interrupted')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        while pending or active:
            while pending and len(active) < a.jobs and not failed:
                group = pending.popleft(); part = f'part{group:02d}'
                command = [sys.executable, '-B', str(here / 'guard_linux_v7.py'), '--receipt', str(a.out / (part + '_GUARD.json')),
                           '--state-dir', str(a.state_dir), '--max-active', str(a.max_active)]
                if a.cgroup_root:
                    command += ['--cgroup-root', str(a.cgroup_root)]
                command += ['--', sys.executable, '-B', str(here / 'run_group_linux_v7.py'),
                            '--root', str(a.root.resolve()), '--plan', str(a.plan.resolve()), '--candidate', a.candidate,
                            '--ranks', ','.join(map(str, plan['groups'][group - 1])), '--out', str(a.out / part),
                            '--client-python', str(a.client_python), '--server-python', str(a.server_python), '--setup', a.setup]
                if a.warmup_each:
                    command.append('--warmup-each')
                log = (a.out / (part + '.log')).open('x')
                process = subprocess.Popen(command, env=dict(os.environ, **threads, ARC_BLAS_THREADS=str(a.blas_threads)), stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                active[group] = (process, log)
            for group, (process, log) in list(active.items()):
                code = process.poll()
                if code is not None:
                    log.close(); del active[group]; results[str(group)] = code
                    failed |= code != 0
                    print(json.dumps(dict(group=group, exit=code)), flush=True)
            if failed:
                break
            time.sleep(.25)
    except BaseException:
        failed = True
        raise
    finally:
        # Terminating each guard invokes its owned-process-group cleanup.
        for process, _ in active.values():
            process.terminate()
        for group, (process, log) in active.items():
            try:
                process.wait(timeout=30)
            finally:
                log.close()
            results[str(group)] = process.returncode
        write_new(a.out / 'COMPLETE.json', dict(utc=utc(), successful=not failed and not pending,
                  results=results, pending=list(pending), plan_sha256=sha(a.plan),
                  blas_threads=a.blas_threads, wall_s=time.monotonic()-started, max_active=a.max_active))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
