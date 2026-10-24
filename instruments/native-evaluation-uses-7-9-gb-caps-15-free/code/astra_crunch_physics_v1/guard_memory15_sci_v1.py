"""[fable-lead 2026-10-19] SCIENCE-ONLY guard copy: hard system-free rule 50% -> 15% (Matt 09:1x AM PT: use up to ~80% of the 52 GB box); footprint cap 7.9 GB. Release files keep the 7.5 GB / 50% original. [fable-lead 2026-10-19] SCIENCE-ONLY copy of guard_memory50_v1: footprint cap 7.9 GB (value-only science arms of a lineage that peaks 7.3-7.8 GB locally; release files keep the 7.5 GB original). TAG opus-release guard_footprint v3 (2026-10-12 23:50 UTC; lead 23:4xZ: at about 23:39Z system swap grew 1717M -> 1986M while memory_pressure
showed 74% free, and every v2 guard on the box retired its group on the '256 MiB additional system swap' rule at once: S53's 4 RAD workers, both
scout-card workers, engineer2's xfer_B_7100266; macOS swap is sticky and grows without real pressure). v2 with ONE change, the swap rule:
  retire only if (swap growth > 256 MiB AND memory_pressure free < 35%) OR swap growth > 1 GiB.
memory_pressure (/usr/bin/memory_pressure, its 'System-wide memory free percentage') is read only while the growth is above 256 MiB, at most once
every 10 s (the last reading is reused in between); a failed or unparsable reading is a failed reading as in v2 (tolerated, logged, retried;
5 in a row over >= 30 s retire). The footprint trigger, the RSS backstop, the time limit, the 1 s poll and the record are v2's, plus
guard_version 3, pressure_samples, min_pressure_free_pct. v2 stays untouched for the jobs that run under it. v2 header follows.
TAG opus-release guard_footprint v2 (2026-10-12 08:30 UTC; lead 08:0xZ): guard_footprint.py (v1) with a monitor that survives a
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
usage: guard_footprint_v3.py --record OUT.json --seconds N [--poll S] [--footprint-limit BYTES] [--rss-backstop BYTES] <command ...>"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import sys
sys.path.insert(0, '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release')
import rusage

FAIL_COUNT, FAIL_SPAN_S = 5, 30.
READ_ERRORS = (subprocess.SubprocessError, OSError, ValueError, AttributeError)


def swap():
    text = subprocess.check_output(['/usr/sbin/sysctl', '-n', 'vm.swapusage'], text=True, timeout=3)
    return float(re.search(r'used = ([\d.]+)M', text).group(1)) * 1024**2


def pressure_free():
    text = subprocess.check_output(['/usr/bin/memory_pressure'], text=True, timeout=10)
    return int(re.search(r'System-wide memory free percentage: (\d+)%', text).group(1))


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
    assert 0 < args.footprint_limit <= 7_900_000_000 and 0 < args.rss_backstop <= 10_000_000_000 and 0 < args.seconds <= 14400
    assert 1 <= args.poll <= 10
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', FLOPSCOPE_MAX_ARRAY_BYTES='4294967296')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
        env[name] = '1'
    started = time.monotonic()
    initial_swap = first_swap()
    child = subprocess.Popen(['/usr/bin/nice', '-n', '10', *args.command], env=env, start_new_session=True)
    rec = dict(guard_version=3, pid=child.pid, command=args.command, initial_swap_bytes=initial_swap,
        footprint_trigger_bytes=args.footprint_limit, rss_backstop_bytes=args.rss_backstop, poll_sleep_s=args.poll, max_rss_bytes=0,
        swap_rule='retire if swap growth > 256 MiB and memory_pressure free < 35%, or swap growth > 1 GiB', pressure_samples=0,
        min_pressure_free_pct=None,
        max_footprint_bytes=0,
        max_swap_growth_bytes=0, completed=False, error=None, samples=0,
        maximum_sample_interval_s=0., minimum_sample_interval_s=None,
        monitor_failure_rule=f'{FAIL_COUNT} consecutive failed readings spanning >= {FAIL_SPAN_S:g} s', failed_readings=0,
        tolerated_failures=[])
    last = None
    fails, first_fail = 0, None
    pressure = None   # (monotonic time, free %) of the last memory_pressure reading
    try:
        while child.poll() is None:
            if time.monotonic() - started > args.seconds:
                raise RuntimeError('Wall-time limit')
            try:
                rows = group_rows(child.pid)
                growth = max(0, swap() - initial_swap)
                if pressure is None or time.monotonic() - pressure[0] >= 5:
                    pressure = (time.monotonic(), pressure_free())
                    rec['pressure_samples'] += 1
                    rec['min_pressure_free_pct'] = min(pressure[1], rec['min_pressure_free_pct'] if rec['min_pressure_free_pct'] is not None else 100)
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
            if pressure[1] < 15:
                raise RuntimeError('hard memory free below 15%')
            if footprint >= args.footprint_limit:
                rec['trigger_rows'] = rows
                raise RuntimeError('footprint trigger')
            if rss >= args.rss_backstop:
                rec['trigger_rows'] = rows
                raise RuntimeError('RSS backstop')
            if growth > 1024**3:
                raise RuntimeError('1 GiB additional system swap limit')
            if growth > 256 * 1024**2 and pressure[1] < 35:
                raise RuntimeError(f'256 MiB additional system swap with memory_pressure free {pressure[1]}% < 35%')
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
