#!/usr/bin/env python3
"""Finite launch after root grants released slots. Does not advance to another phase.

Usage: python3 -B launch_rank_redistribution_v1.py smoke|effects|screen|confirm --workers 1|2|3
Run smoke alone first. Scientific source/requests are pinned and never rewritten.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from table_early_vread_astra_v4 import HERE, O, HARNESS, identity, zero384, effects, prereg, sha, amendment, price

GUARD_PY = '/Users/matt/arc-white-box-estimation/.venv/bin/python'
MODEL_PY = '/Users/matt/anaconda3/bin/python'
THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')


def stamp():
    return subprocess.check_output(['date', '-u', '+%Y%m%dT%H%M%SZ'], text=True).strip()


def precheck(workers):
    output = subprocess.check_output(['/usr/bin/memory_pressure'], text=True, timeout=15)
    free = int(re.search(r'System-wide memory free percentage: (\d+)%', output).group(1))
    output = subprocess.check_output(['/bin/ps', '-axo', 'rss=,comm='], text=True, timeout=10)
    # Executable names only: no environment or other processes' arguments are read.
    heavy = sum(int(line.split(None, 1)[0]) >= 1_000_000 and 'python' in line.lower()
                for line in output.splitlines() if line.strip())
    if free < 50 or heavy + 1 + workers > 10:
        raise RuntimeError(f'No launch: free={free}% (need >=50), heavy_python={heavy}, new={workers}, cap=10')
    return dict(free_pct=free, heavy_python_ge1GB=heavy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('screen', 'confirm'))
    parser.add_argument('--workers', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--root-slot-grant', action='store_true')
    args = parser.parse_args()
    assert args.root_slot_grant, 'Root must explicitly grant released slots'
    assert os.getpriority(os.PRIO_PROCESS,0)==0, 'Controller must start at nice0; guard applies child nice10 once'
    amendment()
    prereg()
    price()  # Frozen MAX cost is mandatory before either prospective pool launches.
    assert args.phase not in ('smoke','zero384','effects') or args.workers == 1, 'Smoke uses exactly one released slot'
    if args.phase != 'smoke':
        assert identity()['passed'], 'Both exact zero-C gates must pass first'
    if args.phase in ('effects','screen','confirm'):
        assert zero384()['passed'], 'Full384 OFF must match oldrank physics'
    if args.phase in ('screen', 'confirm'):
        assert effects()['passed'], 'Fixed early-read effect must begin at L10 with full384 state'
    worker = HARNESS / 'worker_early_vread_astra_v1.py'
    run = HERE / ('run_' + args.phase)
    requests = json.loads((run / 'requests_manifest_v1.json').read_text())
    assert requests['prereg_sha256'] == sha(HERE / 'PREREG_v1.json')
    ids = set(requests['requests'])
    expected_counts = dict(smoke=(1,), zero384=(1,), effects=(1,), screen=(16,), confirm=(16,))
    assert len(ids) in expected_counts[args.phase]
    found = {}
    for subdir in ('queue', 'claimed'):
        for p in (run / subdir).glob('*.json'):
            rid = json.loads(p.read_text())['id']
            assert rid in ids and rid not in found and sha(p)==requests['requests'][rid], str(p)
            found[rid] = p
            if subdir == 'claimed' and not (run / 'results' / (rid+'.json')).exists():
                index = int(re.search(r'\.w(\d+)\.json$',p.name).group(1))
                assert index < args.workers, 'Unfinished claim requires its original worker index; do not silently strand it'
    assert set(found)==ids
    assert (run / 'STOP').exists(), 'Phase must be held before a launch'
    assert not list((run / 'results').glob('*.error')), 'Errors require review, not automatic retry'
    initial_gate = precheck(args.workers)
    lock = run / 'LAUNCH_LOCK'
    lock.mkdir()  # Exclusive; a stale lock needs root inspection, never auto-deleted.
    token = stamp() + '_' + str(os.getpid())
    receipt = dict(phase=args.phase, workers=args.workers, token=token,
                   initial_gate=initial_gate, runs=[], completed=False,
                   launcher_sha256=sha(Path(__file__)),
                   scheduling_amendment_sha256=sha(HERE/'ROOT_SCHEDULING_AMENDMENT_v1.json'),
                   priority=dict(controller=0,guard=0,native_child=10))
    stop_archive = run / ('STOP.held_' + token)
    assert not stop_archive.exists()
    (run / 'STOP').rename(stop_archive)
    held = run / 'HELD_BY_ROOT'
    if held.exists():
        held.rename(run / ('HELD_BY_ROOT.launched_' + token))
    guard_env = os.environ.copy()
    for key in THREAD_KEYS:
        guard_env[key] = '1'
    for key in ('TRAINER_TRUTH_PLAN', 'TRAINER_TRUTH_PLAN_SHA256'):
        guard_env.pop(key, None)
    child_env = dict(TRAINER_CARD='M', TRAINER_PL='1', TRAINER_F4C='1',
                     TRAINER_THREADS='2', TRAINER_RELIEF='1', TRAINER_STAR_ATTACH='0',
                     TRAINER_BIRTH_ATTACH='0', TRAINER_RSS_LIMIT='7500000000',
                     TRAINER_GENERATED_MIN_N='15990784',
                     TRAINER_GENERATED_ALLOW_FILE=str(run / 'allowed_seeds.txt'),
                     PYTHONPATH=str(O / 'trainer'),
                     OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                     VECLIB_MAXIMUM_THREADS='2', NUMEXPR_NUM_THREADS='2', BLIS_NUM_THREADS='2')
    # Freeze every expected guard BEFORE spawning the first process. The grader
    # requires this full set plus the final owner receipt and retired lock.
    owners = dict(phase=args.phase, workers=args.workers, token=token,
                  launcher_sha256=sha(Path(__file__)),
                  scheduling_amendment_sha256=sha(HERE/'ROOT_SCHEDULING_AMENDMENT_v1.json'),
                  priority=dict(controller=0,guard=0,native_child=10),
                  prereg_sha256=sha(HERE/'PREREG_v1.json'),
                  guard_records=[str(run/f'guard_{token}_w{i}.json') for i in range(args.workers)])
    with (run/('launch_owners_'+token+'.json')).open('x') as stream:
        json.dump(owners,stream,indent=2)
    running = []
    try:
        for index in range(args.workers):
            gate = precheck(args.workers - index)
            record = run / f'guard_{token}_w{index}.json'
            log_path = run / f'worker_{token}_w{index}.log'
            seconds = dict(smoke=1800, zero384=1800, effects=1800, screen=14400, confirm=14400)[args.phase]
            command = [GUARD_PY, '-B', str(O / 'release/guard_footprint_v3.py'),
                       '--record', str(record), '--seconds', str(seconds),
                       '--rss-backstop', '7500000000', '/usr/bin/env']
            command += [f'{key}={value}' for key, value in child_env.items()]
            command += [MODEL_PY, '-B', str(worker), str(run), str(index)]
            stream = log_path.open('x')
            process = subprocess.Popen(command, cwd=O / 'trainer', env=guard_env,
                                       stdout=stream, stderr=subprocess.STDOUT)
            running.append((process, stream))
            receipt['runs'].append(dict(index=index, guard_pid=process.pid,
                                        guard_record=str(record), log=str(log_path), gate=gate))
            print(json.dumps(receipt['runs'][-1]), flush=True)
            if index + 1 < args.workers:
                time.sleep(20)
        while any(process.poll() is None for process, stream in running):
            errors = sorted(p.name for p in (run / 'results').glob('*.error'))
            bad_exit = any(process.poll() not in (None,0) for process, stream in running)
            if errors or bad_exit:
                (run / 'STOP').touch(exist_ok=True)
                if 'boundary_alert' not in receipt:
                    receipt['boundary_alert'] = dict(errors=errors, nonzero_guard_exit=bad_exit)
                    print(json.dumps(dict(event='boundary_hold_for_review', **receipt['boundary_alert'])), flush=True)
            time.sleep(3)
        exit_codes = [process.returncode for process, stream in running]
        result_ids = {p.stem for p in (run / 'results').glob('*.json')}
        errors = sorted(p.name for p in (run / 'results').glob('*.error'))
        receipt.update(exit_codes=exit_codes, result_count=len(result_ids & ids), errors=errors)
        receipt['completed'] = all(code == 0 for code in exit_codes) and not errors and ids <= result_ids
        receipt['clean_boundary_hold'] = not receipt['completed'] and all(code == 0 for code in exit_codes) and not errors and (run/'STOP').exists()
        # Scientific identity is checked only AFTER the owner receipt is frozen.
    finally:
        # Only our own queue is stopped. Never signal another lane or automatically retry.
        (run / 'STOP').touch(exist_ok=True)
        for process, stream in running:
            if process.poll() is not None:
                stream.close()
        with (run / ('launch_receipt_' + token + '.json')).open('x') as stream:
            json.dump(receipt, stream, indent=2)
            stream.write('\n')
        if all(process.poll() is not None for process, stream in running):
            lock.rmdir()
    if receipt['completed'] and args.phase == 'smoke':
        verdict=identity()
        with (run/('identity_after_retirement_'+token+'.json')).open('x') as stream:
            json.dump(verdict,stream,indent=2)
        if not verdict['passed']:
            print(json.dumps(dict(event='identity_failed',verdict=verdict)),flush=True)
            return 2
    print(json.dumps(receipt), flush=True)
    return 0 if receipt['completed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
