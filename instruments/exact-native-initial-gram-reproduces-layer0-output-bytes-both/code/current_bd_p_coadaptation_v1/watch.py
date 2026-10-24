"""Independent, stdlib-only one-child resource supervisor. No numerical imports.

Only the fresh process group created here is eligible for termination. The
stage and runner paths are fixed, with reviewed source hashes required. No
retry, continuation, overwrite, or deadline change after a stage starts.
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
AP = Path('/Users/matt/anaconda3/bin/python')
STAGES = ('cpu_forward', 'mps_forward', 'cpu_gradient', 'mps_gradient', 'judge', 'train')
RSS_LIMIT = 12_000_000_000
SWAP_GROWTH_LIMIT = 2 * 1024 ** 3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def swap_state():
    raw = subprocess.run(['/usr/sbin/sysctl', 'vm.swapusage'], capture_output=True,
                         text=True, check=True, timeout=3).stdout.strip()
    match = re.search(r'used\s*=\s*([\d.]+)([KMGT])', raw)
    if not match:
        raise RuntimeError('Unrecognized swap observation: ' + raw)
    return dict(raw=raw, used_bytes=float(match[1]) * 1024 ** ('KMGT'.index(match[2]) + 1))


def process_table():
    run = subprocess.run(['/bin/ps', '-axo', 'pid=,ppid=,pgid=,rss='],
                         capture_output=True, text=True, check=True, timeout=2)
    return [tuple(map(int, line.split())) for line in run.stdout.splitlines() if line.strip()]


def tree_memory(pid, rows):
    descendants = {pid}
    while True:
        expanded = descendants | {p for p, parent, group, rss in rows if parent in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    owned = [(p, parent, group, rss) for p, parent, group, rss in rows
             if p in descendants or group == pid]
    return sum(rss * 1024 for _, _, _, rss in owned), [p for p, _, _, _ in owned]


def stop_owned_group(child):
    # start_new_session=True guarantees this group belongs to this one job.
    # A dead leader does NOT imply its group has died. Observe that exact
    # recorded group, never substitute a new PID tree or the caller's group.
    if child.pid <= 1 or child.pid == os.getpgrp():
        raise RuntimeError('Refusing an unsafe recorded process group')
    if child.poll() is None:
        try:
            group = os.getpgid(child.pid)
        except ProcessLookupError:
            group = child.pid  # Leader vanished; still drain only its recorded group.
        if group != child.pid:
            raise RuntimeError('Refusing to terminate an unexpected process group')

    def survivors():
        return [p for p, parent, group, rss in process_table() if group == child.pid]

    for action, grace in ((signal.SIGTERM, 3), (signal.SIGKILL, 5)):
        if not survivors():
            break
        try:
            os.killpg(child.pid, action)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            child.poll()  # Reap the leader independently of group survival.
            if not survivors():
                break
            time.sleep(.1)
    child.poll()
    if survivors():
        raise RuntimeError('Owned process group survived forced termination')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=STAGES)
    parser.add_argument('--watch-source-sha256', required=True)
    parser.add_argument('--child-source-sha256', required=True)
    parser.add_argument('--model-sha256', required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--qualification-sha256')
    parser.add_argument('--mps-slot-clear', action='store_true')
    args = parser.parse_args()
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise RuntimeError('Watchdog and child require nice10')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        if os.environ.get(key) != '1':
            raise RuntimeError('Explicit one-thread environment required: ' + key)
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        raise RuntimeError('MPS fallback is not qualified')
    training = args.stage == 'train'
    if (training or args.stage.startswith('mps_')) and not args.mps_slot_clear:
        parser.error('Explicit sole-MPS-slot attestation required after process inventory check')
    if not training and not args.stage.startswith('mps_') and args.mps_slot_clear:
        parser.error('MPS slot attestation belongs only to an MPS stage or training')
    runner = HERE / ('train.py' if training else 'qualify.py')
    pins = {str(Path(__file__).resolve()): args.watch_source_sha256,
            str(runner): args.child_source_sha256, str(HERE/'model.py'): args.model_sha256,
            str(HERE/'PROTOCOL_V1.md'): args.protocol_sha256}
    if training:
        if not args.qualification_sha256:
            parser.error('Training requires the completed qualification hash')
        pins[str(HERE/'stage_judge_v1/result.json')] = args.qualification_sha256
    elif args.qualification_sha256:
        parser.error('No qualification result is supplied to a pre-fit stage')
    for path, digest in pins.items():
        if sha(path) != digest:
            raise RuntimeError('Reviewed watchdog input changed: ' + path)
    out = HERE / ('watch_' + args.stage + '_v1')
    out.mkdir(exist_ok=False)
    child_out = HERE / ('training_v1' if training else 'stage_' + args.stage + '_v1')
    if child_out.exists():
        raise RuntimeError('One-shot child output already exists')
    command = [str(AP), '-u', str(runner), '--run', '--source-sha256', args.child_source_sha256,
               '--model-sha256', args.model_sha256, '--protocol-sha256', args.protocol_sha256]
    if training:
        command += ['--qualification-sha256', args.qualification_sha256]
    else:
        command += ['--stage', args.stage]
        if args.stage.startswith('mps_'):
            command += ['--mps-slot-clear']
    limit = 7200 if training else 900
    before_swap = swap_state()
    result = dict(schema='current_d50_p_coadaptation_watch_v1', stage=args.stage,
                  completed=False, passed=False, command=command, source_bindings=pins,
                  wall_limit_s=limit, rss_limit_bytes=RSS_LIMIT, swap_growth_limit_bytes=SWAP_GROWTH_LIMIT,
                  requested_sample_period_s=.25, maximum_sampled_rss_bytes=0,
                  maximum_sample_interval_s=0., maximum_swap_growth_bytes=0., samples=0,
                  driver_memory='Cooperatively sampled by child; not an absolute-peak guarantee',
                  mps_slot_clear_attested=args.mps_slot_clear,
                  before_swap=before_swap, terminated=False,
                  started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'started.json', result)
    samples = (out/'resources.jsonl').open('x')
    child = None
    start = last = last_swap = time.monotonic()
    try:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True)
        result['child_pid'] = result['child_process_group'] = child.pid
        while child.poll() is None:
            now = time.monotonic()
            if now - start >= limit:
                result.update(terminated=True, reason='independent wall deadline')
                stop_owned_group(child)
                break
            rss, pids = tree_memory(child.pid, process_table())
            if not pids and child.poll() is None:
                raise RuntimeError('Cannot observe the still-running child tree')
            interval = now - last
            last = now
            result['maximum_sampled_rss_bytes'] = max(result['maximum_sampled_rss_bytes'], rss)
            result['maximum_sample_interval_s'] = max(result['maximum_sample_interval_s'], interval)
            result['samples'] += 1
            item = dict(elapsed_s=now-start, rss_bytes=rss, pids=pids, interval_s=interval)
            if now - last_swap >= 5:
                state = swap_state()
                growth = state['used_bytes'] - before_swap['used_bytes']
                item.update(swap=state, swap_growth_bytes=growth)
                result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'], growth)
                last_swap = now
            samples.write(json.dumps(item, allow_nan=False) + '\n')
            samples.flush()
            if rss >= RSS_LIMIT or result['maximum_swap_growth_bytes'] > SWAP_GROWTH_LIMIT:
                result.update(terminated=True, reason='independent tree RSS or swap-growth limit')
                stop_owned_group(child)
                break
            remaining = max(0., min(.25, limit - (time.monotonic() - start)))
            if remaining:
                time.sleep(remaining)
        result['child_exit_code'] = child.wait(timeout=5)
        result['child_elapsed_s'] = time.monotonic() - start
        if result['child_elapsed_s'] >= limit:
            result.update(terminated=True, reason='deadline exceeded at completion check')
        # A child may not leave numerical descendants alive on apparent success.
        rss, residual_pids = tree_memory(child.pid, process_table())
        if residual_pids:
            raise RuntimeError('Child left owned processes alive: ' + repr(residual_pids))
        rp, cp = child_out/'result.json', child_out/'completion.json'
        if rp.exists() and cp.exists():
            record, seal = json.loads(rp.read_text()), json.loads(cp.read_text())
            result['child_result_sha256'] = sha(rp)
            result['child_completed'] = bool(record['completed'] and seal['completed']
                                             and seal['result_sha256'] == sha(rp))
            result['passed'] = bool(not result['terminated'] and result['child_exit_code'] == 0
                                    and result['child_completed'] and record['passed'] and seal['passed'])
        for path, digest in pins.items():
            if sha(path) != digest:
                raise RuntimeError('Sources changed during stage: ' + path)
        result.update(completed=True, sources_unchanged=True)
    except BaseException:
        result.update(passed=False, error=traceback.format_exc())
        if child is not None:
            stop_owned_group(child)
    finally:
        samples.close()
        result.update(elapsed_s=time.monotonic()-start, finished_utc=datetime.now(timezone.utc).isoformat())
        try:
            result['after_swap'] = swap_state()
            if result['after_swap']['used_bytes'] - before_swap['used_bytes'] > SWAP_GROWTH_LIMIT:
                result['passed'] = False
                result['reason'] = 'swap-growth limit at final observation'
        except BaseException:
            result.update(passed=False, after_swap_error=traceback.format_exc())
        write(out/'result.json', result)
        write(out/'completion.json', dict(completed=result['completed'], passed=result['passed'],
                                          result_sha256=sha(out/'result.json')))
    print(json.dumps({k: result.get(k) for k in ('stage', 'passed', 'completed', 'elapsed_s',
          'maximum_sampled_rss_bytes', 'maximum_swap_growth_bytes', 'terminated', 'reason', 'error')},
          sort_keys=True), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
