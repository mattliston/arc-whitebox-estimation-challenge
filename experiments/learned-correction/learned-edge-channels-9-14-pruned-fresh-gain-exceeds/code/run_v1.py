"""Prepared primary-slot screen; --run and a lead-owned reservation required."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

D = Path(__file__).resolve().parent
K = D.parent.parent / "fable_oct19_kc_v1"
PY = Path('/Users/matt/arc-white-box-estimation/.venv/bin/python')
RESERVE = K / 'queue_v14.txt.reserve_edge_hidden'
STOP = False


def utc():
    return subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def note(status, **fields):
    print(json.dumps(dict(utc=utc(), status=status, **fields)), flush=True)


def verify(m):
    for path, digest in m['source_pins'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    for t in m['truths']:
        assert hashlib.sha256(Path(t['path']).read_bytes()).hexdigest() == t['sha256']


def checked_result(path, seed, role, m):
    guard = json.loads((path/'GUARD_v1.json').read_text())
    assert guard['completed'] and guard['exit_code'] == 0 and guard['error'] is None
    assert guard['whole_group_retired'] and not guard.get('remaining_rows')
    assert guard['max_footprint_bytes'] <= 7900000000 and guard['max_rss_bytes'] <= 9500000000
    assert guard['min_pressure_free_pct'] >= 15
    log = (path/'run.log').read_text()
    assert 'EVAL_DONE' in log.splitlines() and 'Traceback' not in log
    metas = list((path/'predictions').glob('*.json'))
    assert len(metas) == 1
    r = json.loads(metas[0].read_text())
    assert r['seed'] == seed and r['setting_index'] == 0
    assert r['setting'] == m['settings'][0]
    assert r['candidate_sha256'] == m['variants'][role]['sha256']
    assert r['evaluator_sha256'] == m['evaluator_sha256']
    assert r['official_flopscope'].startswith('0.12.3')
    p = Path(r['prediction_path'])
    assert p.parent == path/'predictions'
    assert hashlib.sha256(p.read_bytes()).hexdigest() == r['prediction_sha256']
    return r


def admission():
    while not STOP:
        assert RESERVE.is_file(), 'Lead-owned primary reservation required.'
        try:
            pressure = subprocess.check_output(['/usr/bin/memory_pressure'], text=True, timeout=10)
            free = int(re.search(r'System-wide memory free percentage: (\d+)%', pressure).group(1))
            ps = subprocess.check_output(['/bin/ps', '-axo', 'pid=,command='], text=True, timeout=5)
            judges = sum(bool(re.search(r'run_current32_\S*_fable_v1\.py(?:\s|$)', row)) for row in ps.splitlines())
            capacity = int((K/'queue_v14.txt.capacity').read_text().strip())
            assert 1 <= capacity <= 5
            reservations = sum(p.is_file() for p in K.glob('queue_v14.txt.reserve_*'))
            reservations += int((K/'queue_v14.txt.single_lane').exists())
            fields = dict(free_pct=free, required_free_pct=45, judges=judges,
                          reservations=reservations, capacity=capacity)
            if free >= 45 and judges + reservations <= capacity:
                note('ADMITTED', **fields)
                return True
            note('ADMISSION_WAIT', **fields)
        except (subprocess.SubprocessError, OSError, ValueError, AttributeError) as error:
            note('ADMISSION_WAIT', reason=repr(error))
        time.sleep(10)
    return False


def stop_after_shard(*_):
    global STOP
    STOP = True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--variants', nargs='+', choices=['eh14', 'eh15'], default=['eh14', 'eh15'])
    args = p.parse_args()
    assert len(set(args.variants)) == len(args.variants)
    m = json.loads((D/'MANIFEST_v1.json').read_text())
    verify(m)
    note('PREPARED_SOURCE_CHECK_PASS', variants=args.variants, seeds=m['seeds'], will_launch=args.run)
    if not args.run:
        return 0
    lock = (D/'controller.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        note('CONTROLLER_ALREADY_OWNED')
        return 75
    assert RESERVE.is_file(), 'Lead must reserve one primary slot before --run.'
    signal.signal(signal.SIGTERM, stop_after_shard)
    signal.signal(signal.SIGINT, stop_after_shard)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
        env[key] = '1'
    try:
        # Complete EH14's fixed four-net cohort first, then EH15 fallback.
        for role in args.variants:
            for seed in m['seeds']:
                verify(m)
                path = D/'runs'/f'{role}_seed{seed}'
                if path.exists():
                    assert args.resume, 'Existing output requires --resume: '+str(path)
                    checked_result(path, seed, role, m)
                    note('COMPLETE_SHARD_REUSED', role=role, seed=seed)
                    continue
                if not admission() or STOP:
                    return 75
                path.mkdir(parents=True, exist_ok=False)
                command = [str(PY), '-B', m['guard'], '--record', str(path/'GUARD_v1.json'),
                           '--seconds', '600', '--footprint-limit', '7900000000',
                           '--rss-backstop', '9500000000', str(PY), '-B', m['evaluator'],
                           m['variants'][role]['source'], str(seed), '1', json.dumps(m['settings']),
                           str(path/'predictions')]
                (path/'PLAN_v1.json').write_text(json.dumps(dict(utc=utc(), seed=seed, role=role,
                    command=command, candidate_sha256=m['variants'][role]['sha256']), indent=2)+'\n')
                note('SHARD_START', role=role, seed=seed)
                with (path/'run.log').open('x') as log:
                    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    note('SHARD_RETIRED_OR_FAILED', role=role, seed=seed, returncode=result.returncode,
                         action='Stop, preserve partial output; lead inspects before recovery.')
                    return result.returncode
                checked_result(path, seed, role, m)
                note('SHARD_COMPLETE', role=role, seed=seed)
                if STOP:
                    return 75
            subprocess.run([str(PY), '-B', str(D/'analyze_v1.py'), '--if-complete'], check=True, env=env)
        note('SCREEN_COMPLETE', variants=args.variants)
        return 0
    finally:
        # Existing guard owns child retirement; subprocess.run has returned.
        if RESERVE.exists():
            RESERVE.unlink()
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
