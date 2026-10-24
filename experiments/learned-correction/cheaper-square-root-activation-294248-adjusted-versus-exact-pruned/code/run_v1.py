"""Predeclared four-seed Drop14-only screen in the existing auxiliary slot only."""
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
K = D.parent.parent / 'fable_oct19_kc_v1'
R = K.parent
PY = Path('/Users/matt/arc-white-box-estimation/.venv/bin/python')
STOP = False


def utc():
    return subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()


def note(status, **fields):
    print(json.dumps(dict(utc=utc(),status=status,**fields)),flush=True)


def verify(m):
    for path,digest in m['source_pins'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    for t in m['truths']:
        assert hashlib.sha256(Path(t['path']).read_bytes()).hexdigest() == t['sha256']


def checked_result(path, seed, m):
    guard = json.loads((path/'GUARD_v1.json').read_text())
    assert guard['completed'] and guard['exit_code'] == 0 and guard['error'] is None
    assert guard['whole_group_retired'] and not guard.get('remaining_rows')
    assert guard['max_footprint_bytes'] <= 7900000000 and guard['max_rss_bytes'] <= 9500000000
    assert guard['min_pressure_free_pct'] >= 30
    log = (path/'run.log').read_text()
    assert 'EVAL_DONE' in log.splitlines() and 'Traceback' not in log
    metas = list((path/'predictions').glob('*.json'))
    assert len(metas) == 1
    r = json.loads(metas[0].read_text())
    assert r['seed'] == seed and r['setting_index'] == 0 and r['setting'] == m['settings'][0]
    assert r['candidate_sha256'] == m['candidate_sha256']
    assert r['evaluator_sha256'] == m['evaluator_sha256']
    assert r['official_flopscope'].startswith('0.12.3')
    pred = Path(r['prediction_path'])
    assert pred.parent == path/'predictions'
    assert hashlib.sha256(pred.read_bytes()).hexdigest() == r['prediction_sha256']
    return r


def admission():
    while not STOP:
        try:
            text = subprocess.check_output(['/usr/bin/memory_pressure'],text=True,timeout=10)
            free = int(re.search(r'System-wide memory free percentage: (\d+)%',text).group(1))
            processes = subprocess.check_output(['/bin/ps','-axo','pid=,command='],text=True,timeout=5).splitlines()
            aux = [p for p in processes if re.search(r'/guard_memory(?:30|40)_aux_v\d+\.py(?:\s|$)',p)]
            capacity = int((K/'queue_v14.txt.capacity').read_text().strip())
            assert 1 <= capacity <= 5
            judges = sum(bool(re.search(r'run_current32_\S*_fable_v1\.py(?:\s|$)',p)) for p in processes)
            reserves = sum(p.is_file() for p in K.glob('queue_v14.txt.reserve_*'))
            reserves += int((K/'queue_v14.txt.single_lane').exists())
            fields = dict(free_pct=free,required_free_pct=60,other_auxiliary_guards=aux,
                          judges=judges,primary_reservations=reserves,primary_capacity=capacity)
            if free >= 60 and not aux and judges+reserves <= capacity:
                note('ADMITTED',**fields)
                return True
            note('ADMISSION_WAIT',**fields)
        except (subprocess.SubprocessError,OSError,ValueError,AttributeError) as error:
            note('ADMISSION_WAIT',reason=repr(error))
        time.sleep(10)
    return False


def stop_after_shard(*_):
    global STOP
    STOP = True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true')
    p.add_argument('--resume',action='store_true')
    args = p.parse_args()
    m = json.loads((D/'MANIFEST_v1.json').read_text())
    verify(m)
    note('PREPARED_SOURCE_CHECK_PASS',seeds=m['seeds'],will_launch=args.run)
    if not args.run:
        return 0
    # Lead launch is required. Also enforce the declared parent-cohort condition.
    report = json.loads((Path(m['parent_dir'])/'RESULT_FRESH4_v1.json').read_text())
    parent = next(v for v in report['variants'] if v['variant']=='eh14')
    assert parent['candidate_sha256'] == m['parent_sha256']
    assert [r['seed'] for r in parent['rows']] == m['seeds']
    assert parent['summaries']['adjusted_vs_parent_pct']['mean_pct'] < 0
    lock = (D/'controller.lock').open('a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        note('CONTROLLER_ALREADY_OWNED')
        return 75
    auxlock = (R/'astra_crunch_physics_v1/AUXILIARY32_SLOT_v1.lock').open('a+')
    try:
        fcntl.flock(auxlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        note('AUXILIARY_SLOT_ALREADY_OWNED')
        lock.close()
        return 75
    auxlock.seek(0); auxlock.truncate()
    auxlock.write(json.dumps(dict(pid=os.getpid(),utc=utc(),purpose='drop14_only_fresh92_95'))+'\n')
    auxlock.flush()
    signal.signal(signal.SIGTERM,stop_after_shard)
    signal.signal(signal.SIGINT,stop_after_shard)
    env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
        env[key] = '1'
    try:
        for seed in m['seeds']:
            verify(m)
            path = D/'runs'/f'drop14_seed{seed}'
            if path.exists():
                assert args.resume, 'Existing output requires --resume: '+str(path)
                checked_result(path,seed,m)
                note('COMPLETE_SHARD_REUSED',seed=seed)
                continue
            if not admission() or STOP:
                return 75
            path.mkdir(parents=True,exist_ok=False)
            command = [str(PY),'-B',m['guard'],'--record',str(path/'GUARD_v1.json'),
                       '--seconds','600','--footprint-limit','7900000000','--rss-backstop','9500000000',
                       str(PY),'-B',m['evaluator'],m['candidate'],str(seed),'1',json.dumps(m['settings']),
                       str(path/'predictions')]
            (path/'PLAN_v1.json').write_text(json.dumps(dict(utc=utc(),seed=seed,role='drop14',
                command=command,candidate_sha256=m['candidate_sha256']),indent=2)+'\n')
            note('SHARD_START',seed=seed)
            with (path/'run.log').open('x') as log:
                result = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                note('SHARD_RETIRED_OR_FAILED',seed=seed,returncode=result.returncode,
                     action='Stop and preserve partial output; lead inspects before recovery.')
                return result.returncode
            checked_result(path,seed,m)
            note('SHARD_COMPLETE',seed=seed)
            if STOP:
                return 75
        subprocess.run([str(PY),'-B',str(D/'analyze_v1.py')],check=True,env=env)
        note('SCREEN_COMPLETE',seeds=m['seeds'])
        return 0
    finally:
        auxlock.close()
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
