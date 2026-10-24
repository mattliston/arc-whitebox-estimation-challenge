"""Bounded independent supervisor for three fixed rangefinder VJP cases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
CAP = 120.
RSS = 8_000_000_000


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def command(args):
    # subprocess.run both bounds and reaps each resource observation.
    return subprocess.run(args, text=True, capture_output=True, check=True,
                          timeout=.6).stdout


def swap_used():
    text = command(['/usr/sbin/sysctl', 'vm.swapusage'])
    match = re.search(r'used\s*=\s*([\d.]+)([MG])', text)
    require(match is not None, 'Actual swap observation')
    return round(float(match.group(1)) * (1024**2 if match.group(2) == 'M' else 1024**3))


def snapshot():
    rows = []
    for line in command(['/bin/ps', '-axo', 'pid=,ppid=,pgid=,rss=']).splitlines():
        pid, ppid, pgid, rss = map(int, line.split())
        rows.append((pid, ppid, pgid, rss * 1024))
    return rows


def owned(rows, worker_pid=None):
    ids = {os.getpid()}
    if worker_pid is not None:
        ids.add(worker_pid)
    changed = True
    while changed:
        before = len(ids)
        ids.update(pid for pid, ppid, pgid, rss in rows
                   if ppid in ids or (worker_pid is not None and pgid == worker_pid))
        changed = len(ids) != before
    return [(pid, rss) for pid, ppid, pgid, rss in rows if pid in ids]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pins', required=True)
    args = ap.parse_args()
    began = time.monotonic()
    directory = HERE/'rangefinder_vjp_watch_v1'
    output = HERE/'rangefinder_vjp_v1'
    require(not directory.exists() and not output.exists(), 'One fresh fixed CPU derivative gate, no overwrite')
    directory.mkdir()
    result = dict(passed=False, completed=False, maximum_combined_rss_bytes=0,
                  maximum_swap_growth_bytes=0, maximum_interval_s=0., observations=0)
    child = None
    last = began
    swap0 = None
    try:
        pins = json.loads(Path(args.pins).read_text())
        require(set(pins) == {'rangefinder_vjp.py', 'watch_rangefinder_vjp.py', 'differentiable_rangefinder.py',
                             'RANGEFINDER_VJP_PROTOCOL.md', 'rangefinder_vjp_inputs.json', 'rangefinder_vjp_unit.py'},
                'Exactly the reviewed six-source derivative roster')
        require(all(sha(HERE/name) == h for name, h in pins.items()), 'Reviewed forward source pins')
        require(Path(sys.executable).resolve() == Path('/Users/matt/anaconda3/bin/python').resolve(),
                'Fixed offline Torch interpreter')
        swap0 = swap_used()
        log = (directory/'worker.log').open('x')
        child = subprocess.Popen([sys.executable, '-B', '-u', str(HERE/'rangefinder_vjp.py'),
            '--pins', str(Path(args.pins).resolve()), '--output', str(output)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True)
        result['worker_pid'] = child.pid
        while True:
            rows = snapshot()
            current_swap = swap_used()
            now = time.monotonic()
            interval = now-last
            last = now
            combined = sum(rss for pid, rss in owned(rows, child.pid))
            growth = max(0, current_swap-swap0)
            result['observations'] += 1
            result['maximum_combined_rss_bytes'] = max(result['maximum_combined_rss_bytes'], combined)
            result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'], growth)
            result['maximum_interval_s'] = max(result['maximum_interval_s'], interval)
            require(now-began < CAP and interval <= 1., 'Whole120s and completed observations<=1s')
            require(combined < RSS and growth == 0, '8GB tree and zero added swap')
            if child.poll() is not None:
                break
            time.sleep(.15)
        code = child.wait(timeout=1)
        log.close()
        records = [json.loads((output/name).read_text())
                   for name in ('result.json', 'completion.json', 'finalization.json')]
        rh, ch, fh = (sha(output/name) for name in ('result.json', 'completion.json', 'finalization.json'))
        require(all(x['completed'] is True for x in records) and
                all(x['passed'] == records[0]['passed'] for x in records) and
                code == (0 if records[0]['passed'] else 1) and
                records[1]['result_sha256'] == records[2]['result_sha256'] == rh and
                records[2]['completion_sha256'] == ch, 'Actual child three-seal completion')
        require(records[0]['unit_cases'] == 3 and 15 <= records[0]['native_mm_calls'] <= 20 and
                records[0]['full_model_calls'] == records[0]['parameter_updates'] == 0 and
                records[0]['truth_accessed'] is False and records[0]['score_measured'] is False and
                records[0]['native_gradient_certified'] is False and records[0]['fit_authorized'] is False and
                records[0]['all_owners_retired'] and records[0]['complete_source_upper'] == 60_000_000_000,
                'Actual bounded three-case derivative qualification, not a fit/score')
        diagnostic=json.loads((output/'diagnostics.json').read_text())
        require(sha(output/'diagnostics.json') == records[0]['diagnostic_sha256'] and
                diagnostic['passed'] == records[0]['passed'] ==
                    records[0]['rangefinder_real_algebra_VJP_qualified'] ==
                    all(row['passed'] for row in diagnostic['rows']) and
                diagnostic['absolute_limit'] == 1e-8 and diagnostic['relative_limit'] == 1e-5 and
                diagnostic['no_finite_difference'] and diagnostic['all_five_production_paths_tested'],
                'Actual diagnostic bytes and unchanged prospective tangent/checkpoint gates')
        require(not any(pgid == child.pid for pid, ppid, pgid, rss in snapshot()),
                'Owned worker process group retired')
        require(all(sha(HERE/name) == h for name, h in pins.items()), 'Reviewed sources unchanged')
        result.update(passed=records[0]['passed'], completed=True, whole_owned_tree_retired=True,
                      source_bindings=pins, child_result_sha256=rh,
                      child_completion_sha256=ch, child_finalization_sha256=fh)
    except BaseException:
        result['error'] = traceback.format_exc()
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=2)
            except ProcessLookupError:
                child.wait(timeout=2)
        if child is not None:
            try:
                remaining = [pid for pid, ppid, pgid, rss in snapshot() if pgid == child.pid]
                if remaining:
                    os.killpg(child.pid, signal.SIGKILL)
                result['whole_owned_tree_retired'] = not remaining
            except BaseException:
                result['whole_owned_tree_retired'] = False
        try:
            rows = snapshot()
            final_swap = swap_used()
            now = time.monotonic()
            interval = now-last
            combined = sum(rss for pid, rss in owned(rows, None if child is None else child.pid))
            growth = 0 if swap0 is None else max(0, final_swap-swap0)
            result['observations'] += 1
            result['maximum_interval_s'] = max(result['maximum_interval_s'], interval)
            result['maximum_combined_rss_bytes'] = max(result['maximum_combined_rss_bytes'], combined)
            result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'], growth)
            require(now-began < CAP and interval <= 1. and combined < RSS and growth == 0,
                    'Final completed resource observation retains every gate')
            last = now
        except BaseException:
            result['passed'] = False
            result['tail_error'] = traceback.format_exc()
        result['elapsed_s'] = time.monotonic()-began
        result['passed'] = bool(result['passed'] and result.get('whole_owned_tree_retired') and
                                result['elapsed_s'] < CAP)
        write(directory/'result.json', result)
        rh = sha(directory/'result.json')
        require(time.monotonic() < min(began+CAP, last+1.), 'Bounded result-publication tail')
        write(directory/'completion.json', dict(passed=result['passed'], completed=result['completed'], result_sha256=rh))
        require(time.monotonic() < min(began+CAP, last+1.), 'Bounded completion-publication tail')
        write(directory/'finalization.json', dict(passed=result['passed'], completed=result['completed'],
            result_sha256=rh, completion_sha256=sha(directory/'completion.json'), elapsed_s=time.monotonic()-began,
            bounded_publication_tail=True))
        require(time.monotonic() < min(began+CAP, last+1.), 'Final seal remains inside observed deadline')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
