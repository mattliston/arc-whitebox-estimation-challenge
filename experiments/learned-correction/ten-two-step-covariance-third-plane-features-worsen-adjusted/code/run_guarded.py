"""Serial process-group supervisor with bounded RAM, swap, and elapsed time."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


def swap():
    output = subprocess.check_output(['/usr/sbin/sysctl', '-n', 'vm.swapusage'], text=True, timeout=3)
    return float(re.search(r'used = ([\d.]+)M', output).group(1)) * 1024**2


def rss_group(group):
    rows = subprocess.check_output(['/bin/ps', '-axo', 'pgid=,rss='], text=True, timeout=3)
    return sum(int(r[1]) * 1024 for line in rows.splitlines()
               if len(r := line.split()) == 2 and int(r[0]) == group)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.command and not args.record.exists()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', FLOPSCOPE_MAX_ARRAY_BYTES='4294967296')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
        env[name] = '1'
    started = time.monotonic()
    initial_swap = swap()
    child = subprocess.Popen(['/usr/bin/nice', '-n', '10', *args.command], env=env, start_new_session=True)
    rec = dict(pid=child.pid, command=args.command, initial_swap_bytes=initial_swap,
               max_rss_bytes=0, max_swap_growth_bytes=0, completed=False, error=None)
    try:
        while child.poll() is None:
            rss = rss_group(child.pid)
            growth = max(0, swap() - initial_swap)
            rec['max_rss_bytes'] = max(rss, rec['max_rss_bytes'])
            rec['max_swap_growth_bytes'] = max(growth, rec['max_swap_growth_bytes'])
            if rss >= 8_000_000_000:
                raise RuntimeError('8 GB process-tree RSS limit')
            if growth > 256 * 1024**2:
                raise RuntimeError('256 MiB additional system swap limit')
            if time.monotonic() - started > args.seconds:
                raise RuntimeError('Wall-time limit')
            time.sleep(.5)
        rec['exit_code'] = child.wait()
        rec['completed'] = rec['exit_code'] == 0
    except BaseException as error:
        rec['error'] = repr(error)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=3)
        rec['elapsed_s'] = time.monotonic() - started
        rec['whole_group_retired'] = rss_group(child.pid) == 0
        if not rec['whole_group_retired']:
            os.killpg(child.pid, signal.SIGKILL)
            rec['completed'] = False
        with args.record.open('x') as stream:
            json.dump(rec, stream, indent=2)
        print(json.dumps(rec), flush=True)
    return 0 if rec['completed'] and rec['whole_group_retired'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
