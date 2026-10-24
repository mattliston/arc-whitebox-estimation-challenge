"""One-shot, stdlib-only supervisor for the frozen CAP native-objective audit.

Inert without --run. A PASS certifies completion, provenance and the sampled
resource limits, never scientific advancement. Only the fresh child's process
group may be signalled. No numerical libraries or worker modules are imported.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import traceback

HERE = Path(__file__).resolve().parent
AP = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/.venv/bin/python')
WALL_LIMIT = 60.0
RSS_LIMIT = 2_000_000_000
SWAP_LIMIT = 512 * 1024 ** 2
PERIOD = .2
MAX_INTERVAL = 1.0
BLAS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, record):
    with Path(path).open('x') as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def swap_state():
    raw = subprocess.run(['/usr/sbin/sysctl', 'vm.swapusage'], check=True,
                         capture_output=True, text=True, timeout=.4).stdout.strip()
    match = re.search(r'used\s*=\s*([\d.]+)([KMGT])', raw)
    require(match is not None, 'Unrecognized vm.swapusage response')
    return {'raw': raw, 'used_bytes': float(match[1]) * 1024 ** ('KMGT'.index(match[2]) + 1)}


def process_table():
    raw = subprocess.run(['/bin/ps', '-axo', 'pid=,ppid=,pgid=,rss='], check=True,
                         capture_output=True, text=True, timeout=.4).stdout
    rows = [tuple(map(int, line.split())) for line in raw.splitlines() if line.strip()]
    require(all(len(row) == 4 and row[3] >= 0 for row in rows), 'Invalid process inventory')
    return rows


def descendants(seeds, rows):
    ids = set(seeds)
    while True:
        expanded = ids | {pid for pid, parent, group, rss in rows if parent in ids}
        if expanded == ids:
            return ids
        ids = expanded


def memory_snapshot(child_pid):
    rows = process_table()
    watcher_pid = os.getpid()
    require(any(pid == watcher_pid for pid, parent, group, rss in rows), 'Watcher RSS unavailable')
    owned = descendants({child_pid}, rows) | {pid for pid, parent, group, rss in rows if group == child_pid}
    owned = descendants(owned, rows)
    # Include the observation subprocess too: conservative watcher+child accounting.
    combined = descendants({watcher_pid}, rows) | owned
    present = [(pid, rss * 1024) for pid, parent, group, rss in rows if pid in combined]
    return {'combined_rss_bytes': sum(rss for pid, rss in present),
            'watcher_rss_bytes': sum(rss for pid, rss in present if pid == watcher_pid),
            'child_tree_rss_bytes': sum(rss for pid, rss in present if pid in owned),
            'pids': sorted(pid for pid, rss in present),
            'child_pids': sorted(pid for pid, rss in present if pid in owned)}


def stop_owned_group(child):
    """Drain the recorded group even if its leader has already exited."""
    require(child.pid > 1 and child.pid != os.getpgrp(), 'Unsafe process-group target')
    if child.poll() is None:
        try:
            require(os.getpgid(child.pid) == child.pid, 'Unexpected live child process group')
        except ProcessLookupError:
            pass  # A leader-exit race does not release surviving group members.
    for action, grace in ((signal.SIGTERM, 1.0), (signal.SIGKILL, 2.0)):
        try:
            os.killpg(child.pid, action)
        except ProcessLookupError:
            child.poll()
            return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            child.poll()
            try:
                if not any(group == child.pid for pid, parent, group, rss in process_table()):
                    return
            except Exception:
                # A failed observation must not prevent escalation to SIGKILL.
                if action == signal.SIGTERM:
                    break
            time.sleep(.05)
    child.poll()
    require(not any(group == child.pid for pid, parent, group, rss in process_table()),
            'Owned group survived bounded termination')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--worker-sha256')
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--source-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    for key in ('worker_sha256', 'protocol_sha256', 'source_sha256'):
        require(re.fullmatch(r'[0-9a-f]{64}', getattr(args, key) or '') is not None,
                'A reviewed SHA256 is required: ' + key)
    pins = {str(Path(__file__).resolve()): args.source_sha256,
            str(HERE/'score_v2.py'): args.worker_sha256,
            str(HERE/'PROTOCOL.md'): args.protocol_sha256}
    for path, digest in pins.items():
        require(sha(path) == digest, 'Reviewed input changed: ' + path)
    require(AP.is_file(), 'Fixed OFFPY interpreter missing')
    priority = os.getpriority(os.PRIO_PROCESS, 0)
    require(priority >= 10, 'Supervisor and inherited child require nice10')
    expected_env = {key: '1' for key in BLAS}
    expected_env.update(PYTHONDONTWRITEBYTECODE='1', PYTORCH_ENABLE_MPS_FALLBACK='0')
    for key, value in expected_env.items():
        require(os.environ.get(key) == value, 'Explicit inherited environment required: ' + key)
    out, child_out = HERE/'watch_v2', HERE/'analysis_v2'
    require(not child_out.exists(), 'One-shot worker output already exists')
    out.mkdir(exist_ok=False)
    command = [str(AP), '-B', '-u', str(HERE/'score_v2.py'), '--run',
               '--source-sha256', args.worker_sha256, '--protocol-sha256', args.protocol_sha256]
    result = dict(schema='frozen_cap_native_objective_watch_v2', passed=False,
                  completed=False, hashes_unchanged=False, sources_unchanged=False,
                  scientific_advancement=False,
                  command=command, source_bindings=pins, artifacts={}, nice=priority,
                  inherited_environment=expected_env, child_output=str(child_out),
                  wall_limit_s=WALL_LIMIT, combined_rss_limit_bytes=RSS_LIMIT,
                  swap_growth_limit_bytes=SWAP_LIMIT, requested_sample_period_s=PERIOD,
                  sample_interval_limit_s=MAX_INTERVAL, maximum_sample_interval_s=0.0,
                  maximum_sampled_rss_bytes=0, maximum_swap_growth_bytes=0.0, samples=0,
                  limitations=['RSS and swap are sampled, not continuous peak guarantees.',
                               'Swap growth is system-wide, not attributed to this child.',
                               'PASS does not imply advancement or estimator qualification.'],
                  started_utc=datetime.now(timezone.utc).isoformat())
    child = None
    started = last = time.monotonic()
    prior_handlers = {}

    def interrupted(signum, frame):
        raise RuntimeError('Supervisor interrupted by signal ' + str(signum))

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            prior_handlers[sig] = signal.signal(sig, interrupted)
        before = swap_state()
        result['before_swap'] = before
        write(out/'started.json', result)
        with (out/'stdout.log').open('xb') as stdout, (out/'stderr.log').open('xb') as stderr, \
                (out/'resources.jsonl').open('x') as resources:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout,
                                     stderr=stderr, env=os.environ.copy(), start_new_session=True)
            result['child_pid'] = result['child_process_group'] = child.pid
            while True:
                require(time.monotonic() - started <= WALL_LIMIT, 'Independent wall limit')
                code = child.poll()  # Reap before inventory; do not flag our zombie as a survivor.
                item = memory_snapshot(child.pid)
                sampled = time.monotonic()
                interval = sampled - last
                last = sampled
                swap = swap_state()
                now = time.monotonic()
                growth = swap['used_bytes'] - before['used_bytes']
                item.update(elapsed_s=now-started, rss_sample_elapsed_s=sampled-started, interval_s=interval,
                            swap=swap, swap_growth_bytes=growth)
                resources.write(json.dumps(item, allow_nan=False) + '\n')
                resources.flush()
                result['samples'] += 1
                result['maximum_sampled_rss_bytes'] = max(result['maximum_sampled_rss_bytes'], item['combined_rss_bytes'])
                result['maximum_sample_interval_s'] = max(result['maximum_sample_interval_s'], interval)
                result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'], growth)
                require(now-started <= WALL_LIMIT, 'Independent wall limit after observation')
                require(interval <= MAX_INTERVAL, 'Resource sampling interval exceeded one second')
                require(item['combined_rss_bytes'] <= RSS_LIMIT, 'Combined watcher/child RSS limit')
                require(growth <= SWAP_LIMIT, 'System swap-growth limit')
                require(item['child_pids'] or code is not None or child.poll() is not None,
                        'Live child tree not observable')
                if code is not None:
                    result['child_exit_code'] = code
                    require(not item['child_pids'], 'Child left owned descendants alive')
                    break
                time.sleep(min(PERIOD, max(0.0, WALL_LIMIT-(time.monotonic()-started))))
        require(result['child_exit_code'] == 0, 'Worker did not complete successfully')
        rp, cp = child_out/'result.json', child_out/'completion.json'
        record, seal = json.loads(rp.read_text()), json.loads(cp.read_text())
        result['child_result_sha256'], result['child_completion_sha256'] = sha(rp), sha(cp)
        require(record.get('completed') is True and record.get('hashes_unchanged') is True,
                'Worker completion/provenance flags missing')
        require(seal.get('completed') is True and seal.get('result_sha256') == sha(rp),
                'Worker completion seal invalid')
        result['child_diagnostic_completed'] = True
        # Do not require scientific or advancement fields, including a worker's
        # optional "passed" field: exit0 here means a completed diagnostic.
        for path, digest in pins.items():
            require(sha(path) == digest, 'Reviewed input changed during run: ' + path)
        result['after_swap'] = swap_state()
        result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],
            result['after_swap']['used_bytes'] - before['used_bytes'])
        require(result['maximum_swap_growth_bytes'] <= SWAP_LIMIT, 'Final swap-growth limit')
        require(time.monotonic()-started <= WALL_LIMIT, 'Completion exceeded wall limit')
        result.update(passed=True, completed=True, hashes_unchanged=True, sources_unchanged=True)
    except BaseException:
        result.update(passed=False, completed=False, error=traceback.format_exc())
        if child is not None:
            try:
                stop_owned_group(child)
                result['owned_group_cleanup_complete'] = True
                result['child_exit_code'] = child.poll()
            except BaseException:
                result['cleanup_error'] = traceback.format_exc()
    finally:
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)
        result.update(elapsed_s=time.monotonic()-started,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        try:
            for path in (out/'started.json', out/'stdout.log', out/'stderr.log', out/'resources.jsonl',
                         child_out/'result.json', child_out/'completion.json'):
                if path.is_file():
                    result['artifacts'][str(path)] = sha(path)
            for path, digest in pins.items():
                require(sha(path) == digest, 'Final reviewed input changed: ' + path)
            result.update(hashes_unchanged=True, sources_unchanged=True)
            require(time.monotonic()-started <= WALL_LIMIT or result['passed'] is False,
                    'Final sealing exceeded wall limit')
        except BaseException:
            result.update(passed=False, completed=False, hashes_unchanged=False, sources_unchanged=False,
                          final_sealing_error=traceback.format_exc())
        result.update(elapsed_s=time.monotonic()-started,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write(out/'result.json', result)
        write(out/'completion.json', {'passed': result['passed'], 'completed': result['completed'],
                                      'result_sha256': sha(out/'result.json')})
    print(json.dumps({key: result.get(key) for key in ('passed', 'completed', 'elapsed_s',
          'maximum_sampled_rss_bytes', 'maximum_sample_interval_s', 'maximum_swap_growth_bytes',
          'error', 'cleanup_error')}, sort_keys=True), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
