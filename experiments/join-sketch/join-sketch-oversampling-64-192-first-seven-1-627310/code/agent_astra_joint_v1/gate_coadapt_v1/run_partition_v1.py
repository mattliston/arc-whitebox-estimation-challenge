"""Prepared controller. --run is required; default performs source checks only."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

D = Path(__file__).resolve().parent
K = D.parent.parent
R = K.parent
PY = Path('/Users/matt/arc-white-box-estimation/.venv/bin/python')
RESERVE = K / 'queue_v14.txt.reserve_gate_coadapt'
STOP = False


def utc():
    return subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def note(status, **fields):
    print(json.dumps(dict(utc=utc(), status=status, **fields)), flush=True)


def verify(manifest):
    for path, digest in manifest['source_pins'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    for truth in manifest['truths']:
        assert hashlib.sha256(Path(truth['path']).read_bytes()).hexdigest() == truth['sha256']


def pressure_and_processes():
    text = subprocess.check_output(['/usr/bin/memory_pressure'], text=True, timeout=10)
    free = int(re.search(r'System-wide memory free percentage: (\d+)%', text).group(1))
    processes = subprocess.check_output(['/bin/ps', '-axo', 'pid=,command='], text=True, timeout=5)
    return free, processes.splitlines()


def admission(mode):
    while not STOP:
        try:
            free, processes = pressure_and_processes()
            if mode == 'aux':
                others = [p for p in processes if re.search(r'/guard_memory(?:30|40)_aux_v\d+\.py(?:\s|$)', p)]
                okay = free >= 60 and not others
                fields = dict(free_pct=free, required_free_pct=60, other_auxiliary_guards=others)
            else:
                assert RESERVE.is_file(), 'Root-owned primary reservation is required.'
                capacity = int((K / 'queue_v14.txt.capacity').read_text().strip())
                assert 1 <= capacity <= 5
                lanes = sum(bool(re.search(r'run_current32_\S*_fable_v1\.py(?:\s|$)', p)) for p in processes)
                reserves = sum(p.is_file() for p in K.glob('queue_v14.txt.reserve_*'))
                reserves += int((K / 'queue_v14.txt.single_lane').exists())
                okay = free >= 45 and lanes + reserves <= capacity
                fields = dict(free_pct=free, required_free_pct=45, judges=lanes,
                              reservations=reserves, capacity=capacity)
            if okay:
                note('ADMITTED', mode=mode, **fields)
                return True
            note('ADMISSION_WAIT', mode=mode, **fields)
        except (subprocess.SubprocessError, OSError, ValueError, AttributeError) as error:
            note('ADMISSION_WAIT', mode=mode, reason=repr(error))
        time.sleep(10)
    return False


def checked_result(path, seed, role, manifest):
    guard = json.loads((path / 'GUARD_v1.json').read_text())
    assert guard['completed'] and guard['exit_code'] == 0 and guard['error'] is None
    assert guard['whole_group_retired'] and not guard.get('remaining_rows')
    assert guard['max_footprint_bytes'] <= 7900000000
    assert guard['max_rss_bytes'] <= 9500000000
    assert guard['min_pressure_free_pct'] >= (30 if seed >= 7100094 else 15)
    text = (path / 'run.log').read_text()
    assert text.rstrip().endswith('EVAL_DONE') and 'Traceback' not in text
    settings = manifest['bank_settings'] if role == 'bank' else manifest['settings']
    source_sha = manifest['bank_sha256'] if role == 'bank' else manifest['candidate_sha256']
    metadata = sorted((path / 'predictions').glob('*.json'))
    assert len(metadata) == len(settings), (path, len(metadata))
    for expected_index, meta in enumerate(metadata):
        record = json.loads(meta.read_text())
        assert record['seed'] == seed and record['setting_index'] == expected_index
        assert record['setting'] == settings[expected_index]
        assert record['candidate_sha256'] == source_sha
        assert record['evaluator_sha256'] == manifest['evaluator_sha256']
        prediction = Path(record['prediction_path'])
        assert prediction.parent == path / 'predictions'
        assert hashlib.sha256(prediction.read_bytes()).hexdigest() == record['prediction_sha256']
    return guard


def stop_after_shard(*_):
    # Let the existing guard own/retire its live process group. Never remove
    # the primary reservation while a native shard is still running.
    global STOP
    STOP = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['primary', 'aux'], required=True)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Reuse fully verified role shards only; never overwrite partial output.')
    args = parser.parse_args()
    manifest = json.loads((D / 'MANIFEST_v1.json').read_text())
    verify(manifest)
    seeds = manifest['partitions'][args.mode]
    guard_path = Path(manifest['guards'][args.mode])
    note('PREPARED_SOURCE_CHECK_PASS', mode=args.mode, seeds=seeds, will_launch=args.run)
    if not args.run:
        return 0
    partition_lock = (D / (args.mode + '_controller.lock')).open('a+')
    try:
        fcntl.flock(partition_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        note('PARTITION_ALREADY_OWNED', mode=args.mode)
        return 75
    auxiliary_lock = None
    if args.mode == 'aux':
        lock_path = R / 'astra_crunch_physics_v1/AUXILIARY32_SLOT_v1.lock'
        auxiliary_lock = lock_path.open('a+')
        try:
            fcntl.flock(auxiliary_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            note('AUXILIARY_SLOT_ALREADY_OWNED')
            return 75
        auxiliary_lock.seek(0)
        auxiliary_lock.truncate()
        auxiliary_lock.write(json.dumps(dict(pid=os.getpid(), utc=utc(), purpose='gate_coadapt_fresh')) + '\n')
        auxiliary_lock.flush()
    else:
        assert RESERVE.is_file(), 'Lead must reserve one primary slot before --run.'
    signal.signal(signal.SIGTERM, stop_after_shard)
    signal.signal(signal.SIGINT, stop_after_shard)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
        env[name] = '1'
    complete = False
    try:
        for seed in seeds:
            for role in ('bank', 'screen'):
                verify(manifest)
                path = D / 'runs' / ('%s_seed%d' % (role, seed))
                if path.exists():
                    assert args.resume, 'Existing output requires --resume: ' + str(path)
                    checked_result(path, seed, role, manifest)
                    note('COMPLETE_SHARD_REUSED', seed=seed, role=role)
                    continue
                if not admission(args.mode):
                    return 75
                if STOP:
                    return 75
                path.mkdir(parents=True, exist_ok=False)
                source = manifest['bank'] if role == 'bank' else manifest['candidate']
                settings = manifest['bank_settings'] if role == 'bank' else manifest['settings']
                command = [str(PY), '-B', str(guard_path), '--record', str(path/'GUARD_v1.json'),
                           '--seconds', '600', '--footprint-limit', '7900000000',
                           '--rss-backstop', '9500000000', str(PY), '-B', manifest['evaluator'],
                           source, str(seed), '1', json.dumps(settings), str(path/'predictions')]
                plan = dict(utc=utc(), seed=seed, role=role, mode=args.mode, command=command,
                            candidate_sha256=manifest['bank_sha256'] if role == 'bank' else manifest['candidate_sha256'])
                (path/'PLAN_v1.json').write_text(json.dumps(plan, indent=2) + '\n')
                note('SHARD_START', seed=seed, role=role, mode=args.mode)
                with (path/'run.log').open('x') as log:
                    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    note('SHARD_RETIRED_OR_FAILED', seed=seed, role=role, returncode=result.returncode,
                         action='Stop and preserve partial outputs. Lead inspects before recovery.')
                    return result.returncode
                checked_result(path, seed, role, manifest)
                note('SHARD_COMPLETE', seed=seed, role=role)
                if STOP:
                    return 75
            subprocess.run([str(PY), '-B', str(D/'analyze_v1.py'), '--if-complete'], check=True, env=env)
        complete = True
        note('PARTITION_COMPLETE', mode=args.mode, seeds=seeds)
        return 0
    finally:
        if args.mode == 'primary':
            # On a normal, stopped-after-shard or guarded-failure return there
            # is no live native child. The unchanged guard owns retirement.
            if RESERVE.exists():
                RESERVE.unlink()
        if auxiliary_lock is not None:
            auxiliary_lock.close()
        partition_lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
