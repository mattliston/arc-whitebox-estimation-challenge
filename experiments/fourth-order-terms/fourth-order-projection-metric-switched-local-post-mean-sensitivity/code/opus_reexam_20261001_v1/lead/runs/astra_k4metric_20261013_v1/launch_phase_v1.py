#!/usr/bin/env python3
"""Finite launch after root grants released slots. Does not advance to another phase.

Usage: python3 -B launch_phase_v1.py smoke|screen|confirm --workers 1|2
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

from table_v1 import HERE, O, smoke

GUARD_PY = '/Users/matt/arc-white-box-estimation/.venv/bin/python'
MODEL_PY = '/Users/matt/anaconda3/bin/python'
THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def stamp():
    return subprocess.check_output(['date', '-u', '+%Y%m%dT%H%M%SZ'], text=True).strip()


def precheck(workers):
    output = subprocess.check_output(['/usr/bin/memory_pressure'], text=True, timeout=15)
    free = int(re.search(r'System-wide memory free percentage: (\d+)%', output).group(1))
    output = subprocess.check_output(['/bin/ps', '-axo', 'rss=,comm='], text=True, timeout=10)
    # Executable names only: no environment or other processes' arguments are read.
    heavy = sum(int(line.split(None, 1)[0]) >= 1_000_000 and 'python' in line.lower()
                for line in output.splitlines() if line.strip())
    if free < 50 or heavy + workers > 10:
        raise RuntimeError(f'No launch: free={free}% (need >=50), heavy_python={heavy}, new={workers}, cap=10')
    return dict(free_pct=free, heavy_python_ge1GB=heavy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('smoke', 'screen', 'confirm'))
    parser.add_argument('--workers', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    assert args.phase != 'smoke' or args.workers == 1, 'Smoke uses exactly one released slot'
    if args.phase != 'smoke':
        assert smoke()['bitexact'], 'Smoke must be bitexact before any arm phase'
    manifest = json.loads((HERE / 'manifest_v1.json').read_text())
    for name, expected in manifest['staged_hashes'].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected, name
    finite = json.loads((HERE / 'finite_worker_audit_v1.json').read_text())
    worker = HERE / 'worker_k4metric_finite_v2.py'
    assert hashlib.sha256(worker.read_bytes()).hexdigest() == finite['finite_sha256']
    card = O / 'scout_card/otr_m_sc4.py'
    assert hashlib.sha256(card.read_bytes()).hexdigest() == manifest['shared_card_sha256']
    run = HERE / ('run_' + args.phase)
    ids = {json.loads(p.read_text())['id']
           for subdir in ('queue', 'claimed') for p in (run / subdir).glob('*.json')}
    expected_count = dict(smoke=1, screen=48, confirm=16)[args.phase]
    assert len(ids) == expected_count, (len(ids), expected_count)
    assert (run / 'STOP').exists(), 'Phase must be held before a launch'
    assert not list((run / 'results').glob('*.error')), 'Errors require review, not automatic retry'
    initial_gate = precheck(args.workers)
    lock = run / 'LAUNCH_LOCK'
    lock.mkdir()  # Exclusive; a stale lock needs root inspection, never auto-deleted.
    token = stamp() + '_' + str(os.getpid())
    receipt = dict(phase=args.phase, workers=args.workers, token=token,
                   initial_gate=initial_gate, runs=[], completed=False)
    stop_archive = run / ('STOP.held_' + token)
    assert not stop_archive.exists()
    (run / 'STOP').rename(stop_archive)
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
                     VECLIB_MAXIMUM_THREADS='2', NUMEXPR_NUM_THREADS='1')
    running = []
    try:
        for index in range(args.workers):
            gate = precheck(args.workers - index)
            record = run / f'guard_{token}_w{index}.json'
            log_path = run / f'worker_{token}_w{index}.log'
            seconds = dict(smoke=1800, screen=10800, confirm=5400)[args.phase]
            command = ['nice', '-n', '10', GUARD_PY, '-B', str(O / 'release/guard_footprint_v3.py'),
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
        exit_codes = [process.wait() for process, stream in running]
        result_ids = {p.stem for p in (run / 'results').glob('*.json')}
        errors = sorted(p.name for p in (run / 'results').glob('*.error'))
        receipt.update(exit_codes=exit_codes, result_count=len(result_ids & ids), errors=errors)
        receipt['completed'] = all(code == 0 for code in exit_codes) and not errors and ids <= result_ids
        receipt['smoke'] = smoke() if args.phase == 'smoke' else None
        if args.phase == 'smoke':
            receipt['completed'] = receipt['completed'] and receipt['smoke']['bitexact']
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
    print(json.dumps(receipt), flush=True)
    return 0 if receipt['completed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
