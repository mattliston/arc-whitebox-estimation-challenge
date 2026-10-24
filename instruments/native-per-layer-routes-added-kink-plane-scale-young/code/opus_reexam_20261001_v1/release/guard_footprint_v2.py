"""TAG opus-release guard_footprint v2 (2026-10-12 08:30 UTC; lead 08:0xZ): guard_footprint.py (v1) with a monitor that survives a
slow or failed reading. v1 is unchanged and stays in use by whatever already runs under it.
Why: at about 08:06Z one /bin/ps call took more than 3 s (memory 73% free, swap flat) and each of the 8 S51 workers' v1 guards
retired its whole group on that single TimeoutExpired (trainer/cert/RAD/S51/worker_gen6t2_guard_20261012T075332Z.json). v1 also
slept only 0.05 s between readings, so every guard scanned the whole process table about 3 times a second (30-40 scans a second
over 10-14 guards), which loads the box and makes such a stall likelier.
Changes from v1, nothing else:
  (a) a failed reading (ps or sysctl: timeout, error, unparsable output) is tolerated, logged in the record (tolerated_failures,
      first 200 kept, failed_readings counts all) and retried at the next poll; only FAIL_COUNT (5) consecutive failures spanning
      at least FAIL_SPAN_S (30 s) retire the group ('monitor failure');
  (b) one reading every --poll seconds (default and minimum 1 s) instead of every 0.05 s; between readings the guard waits on its
      child, so it still notices the exit at once; the wall-time limit is checked even while readings fail;
  (c) stopping the group no longer needs ps: liveness is probed with killpg(group, 0); ps lists the rows left only at the end;
  (d) the limits are v1's: group footprint >= --footprint-limit (default and cap 7.5 GB), group RSS >= --rss-backstop (default and
      cap 10 GB), +256 MiB of system swap, the time limit.
usage: guard_footprint_v2.py --record OUT.json --seconds N [--poll S] [--footprint-limit BYTES] [--rss-backstop BYTES] <command ...>"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import rusage

FAIL_COUNT, FAIL_SPAN_S = 5, 30.
READ_ERRORS = (subprocess.SubprocessError, OSError, ValueError, AttributeError)


def swap():
    text = subprocess.check_output(['/usr/sbin/sysctl', '-n', 'vm.swapusage'], text=True, timeout=3)
    return float(re.search(r'used = ([\d.]+)M', text).group(1)) * 1024**2


def first_swap():
    for _ in range(9):
        try:
            return swap()
        except READ_ERRORS:
            time.sleep(1)
    return swap()


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


def group_alive(group):
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def wait(child, seconds):
    try:
        child.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        pass


def stop_group(child):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        child.poll()
        if not group_alive(child.pid):
            break
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass
        end = time.monotonic() + 3
        while time.monotonic() < end:
            child.poll()
            if not group_alive(child.pid):
                break
            time.sleep(.05)
    if child.poll() is None:
        child.wait(timeout=3)
    try:
        return group_rows(child.pid)
    except READ_ERRORS as error:
        return [dict(pgid=child.pid, listing_error=repr(error))] if group_alive(child.pid) else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--poll', type=float, default=1.)
    parser.add_argument('--footprint-limit', type=int, default=7_500_000_000)
    parser.add_argument('--rss-backstop', type=int, default=10_000_000_000)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.command and not args.record.exists()
    assert 0 < args.footprint_limit <= 7_500_000_000 and 0 < args.rss_backstop <= 10_000_000_000 and 0 < args.seconds <= 14400
    assert 1 <= args.poll <= 10
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', FLOPSCOPE_MAX_ARRAY_BYTES='4294967296')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
        env[name] = '1'
    started = time.monotonic()
    initial_swap = first_swap()
    child = subprocess.Popen(['/usr/bin/nice', '-n', '10', *args.command], env=env, start_new_session=True)
    rec = dict(guard_version=2, pid=child.pid, command=args.command, initial_swap_bytes=initial_swap,
        footprint_trigger_bytes=args.footprint_limit, rss_backstop_bytes=args.rss_backstop, poll_sleep_s=args.poll, max_rss_bytes=0,
        max_footprint_bytes=0,
        max_swap_growth_bytes=0, completed=False, error=None, samples=0,
        maximum_sample_interval_s=0., minimum_sample_interval_s=None,
        monitor_failure_rule=f'{FAIL_COUNT} consecutive failed readings spanning >= {FAIL_SPAN_S:g} s', failed_readings=0,
        tolerated_failures=[])
    last = None
    fails, first_fail = 0, None
    try:
        while child.poll() is None:
            if time.monotonic() - started > args.seconds:
                raise RuntimeError('Wall-time limit')
            try:
                rows = group_rows(child.pid)
                growth = max(0, swap() - initial_swap)
            except READ_ERRORS as error:
                now = time.monotonic()
                fails += 1
                first_fail = now if fails == 1 else first_fail
                rec['failed_readings'] += 1
                if len(rec['tolerated_failures']) < 200:
                    rec['tolerated_failures'].append(dict(t_s=round(now - started, 3), consecutive=fails, error=repr(error)))
                if fails >= FAIL_COUNT and now - first_fail >= FAIL_SPAN_S:
                    raise RuntimeError(f'monitor failure: {fails} consecutive failed readings over {now - first_fail:.1f} s')
                wait(child, args.poll)
                continue
            fails = 0
            rss = sum(r['rss_bytes'] for r in rows)
            footprint = sum(r['footprint_bytes'] for r in rows)
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
            wait(child, args.poll)
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
