"""TAG opus-release: the process-group guard of H's native lifecycle (codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py)
with the memory trigger on PHYSICAL FOOTPRINT instead of RSS, and RSS kept as a far backstop. WIRED IN since 2026-10-02 08:50Z
(lead's decision of 08:37Z: pass when server + client peak footprint is under 7.5 GB, group RSS only as a hard stop at 10 GB
that does not depend on the footprint reading): common.GUARD points here; every session of stages B, C, D and of the fast
path runs under it.
Why: on this machine RSS counts freed pages the system has not taken back yet, and it takes them back only under memory
pressure. Over 34 stored sessions the quiet ones show 6.0-6.8 GB of group RSS and the loaded ones 2.9-3.9 GB at the same
footprint of 2.1-2.7 GB; S38's 20-network worker in the quiet window of 07:39Z reached 6.81 GB of RSS at a footprint of 2.2 GB,
0.7 GB from the original guard's kill at 7.5 GB. Footprint (proc_pid_rusage: live dirty memory, compressed pages included) is
what a real memory blow-up raises, so the protection of the machine is the same or stricter; RSS is still recorded.
Triggers: group footprint >= --footprint-limit (default and cap 7.5 GB), group RSS >= --rss-backstop (default and cap 10 GB),
+256 MiB of system swap, the time limit.
usage: guard_footprint.py --record OUT.json --seconds N [--footprint-limit BYTES] [--rss-backstop BYTES] <command ...>"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import rusage


def swap():
    text = subprocess.check_output(['/usr/sbin/sysctl', '-n', 'vm.swapusage'], text=True, timeout=3)
    return float(re.search(r'used = ([\d.]+)M', text).group(1)) * 1024**2


def group_rows(group):
    rows = subprocess.check_output(['/bin/ps', '-axo', 'pid=,pgid=,rss='], text=True, timeout=3)
    rows = [dict(pid=int(a[0]), pgid=int(a[1]), rss_bytes=int(a[2]) * 1024)
            for line in rows.splitlines() if len(a := line.split()) == 3 and int(a[1]) == group]
    for row in rows:
        try:
            row['footprint_bytes'] = rusage.read(row['pid'])['footprint']
        except OSError:                    # the process ended between the listing and the read
            row['footprint_bytes'] = 0
    return rows


def stop_group(child):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not group_rows(child.pid):
            break
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass
        end = time.monotonic() + 3
        while time.monotonic() < end:
            child.poll()
            if not group_rows(child.pid):
                break
            time.sleep(.05)
    if child.poll() is None:
        child.wait(timeout=3)
    return group_rows(child.pid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--footprint-limit', type=int, default=7_500_000_000)
    parser.add_argument('--rss-backstop', type=int, default=10_000_000_000)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.command and not args.record.exists()
    assert 0 < args.footprint_limit <= 7_500_000_000 and 0 < args.rss_backstop <= 10_000_000_000 and 0 < args.seconds <= 14400
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', FLOPSCOPE_MAX_ARRAY_BYTES='4294967296')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
        env[name] = '1'
    started = time.monotonic()
    initial_swap = swap()
    child = subprocess.Popen(['/usr/bin/nice', '-n', '10', *args.command], env=env, start_new_session=True)
    rec = dict(pid=child.pid, command=args.command, initial_swap_bytes=initial_swap,
        footprint_trigger_bytes=args.footprint_limit, rss_backstop_bytes=args.rss_backstop, poll_sleep_s=.05, max_rss_bytes=0,
        max_footprint_bytes=0,
        max_swap_growth_bytes=0, completed=False, error=None, samples=0,
        maximum_sample_interval_s=0., minimum_sample_interval_s=None)
    last = None
    try:
        while child.poll() is None:
            rows = group_rows(child.pid)
            rss = sum(r['rss_bytes'] for r in rows)
            footprint = sum(r['footprint_bytes'] for r in rows)
            growth = max(0, swap() - initial_swap)
            now = time.monotonic()
            if last is not None:
                dt = now - last
                rec['maximum_sample_interval_s'] = max(dt, rec['maximum_sample_interval_s'])
                rec['minimum_sample_interval_s'] = min(dt, rec['minimum_sample_interval_s'] or dt)
            last = now
            rec['samples'] += 1
            rec['max_rss_bytes'] = max(rss, rec['max_rss_bytes'])
            rec['max_footprint_bytes'] = max(footprint, rec['max_footprint_bytes'])
            rec['max_swap_growth_bytes'] = max(growth, rec['max_swap_growth_bytes'])
            if footprint >= args.footprint_limit:
                rec['trigger_rows'] = rows
                raise RuntimeError('footprint trigger')
            if rss >= args.rss_backstop:
                rec['trigger_rows'] = rows
                raise RuntimeError('RSS backstop')
            if growth > 256 * 1024**2:
                raise RuntimeError('256 MiB additional system swap limit')
            if now - started > args.seconds:
                raise RuntimeError('Wall-time limit')
            time.sleep(.05)
        rec['exit_code'] = child.wait()
        rec['completed'] = rec['exit_code'] == 0
    except BaseException as error:
        rec['error'] = repr(error)
    finally:
        rec['remaining_rows'] = stop_group(child)
        rec['exit_code'] = child.returncode
        rec['elapsed_s'] = time.monotonic() - started
        rec['whole_group_retired'] = not rec['remaining_rows']
        rec['completed'] = rec['completed'] and rec['whole_group_retired']
        with args.record.open('x') as stream:
            json.dump(rec, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps(rec), flush=True)
    return 0 if rec['completed'] and rec['whole_group_retired'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
