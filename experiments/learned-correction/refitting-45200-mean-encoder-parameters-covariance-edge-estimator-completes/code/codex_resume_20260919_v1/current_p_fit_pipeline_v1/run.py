"""One-shot metadata controller; existing sealed workers own all numerics."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
D = HERE.parent
FIT = D / 'current_full_lower_p_fit_v1'
FIT_GROUP = 19016
FIT_SEAL = '501902edc21f598fd7c95b6a97b34697e94c19713536fb2a8604bef57949a959'
ENDPOINT_SEAL = 'de2c1afdc3a6f48aad80f5ca25ef8e0f4bf7a86a9cd0d4cfe2fff76de27708e8'


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return dict(path=str(path), sha256=h.hexdigest(), bytes=path.stat().st_size)


def check(item):
    assert pin(item['path']) == {k: item[k] for k in ('path', 'sha256', 'bytes')}, item['path']


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def ready(path):
    if not path.exists():
        return None
    try:
        return read(path)
    except json.JSONDecodeError:
        return None


def processes():
    rows = []
    text = subprocess.check_output(['/bin/ps', '-axo', 'pid=,pgid=,rss=,command='], text=True, timeout=3)
    for line in text.splitlines():
        a = line.strip().split(None, 3)
        if len(a) == 4:
            rows.append(dict(pid=int(a[0]), pgid=int(a[1]), rss_bytes=int(a[2]) * 1024, command=a[3]))
    return rows


def verify_sources():
    for p in read(HERE / 'sources_sealed.json')['pins']:
        check(p)
    assert pin(FIT / 'sources_sealed.json')['sha256'] == FIT_SEAL
    assert pin(FIT / 'endpoint_sources_sealed.json')['sha256'] == ENDPOINT_SEAL
    for p in read(FIT / 'sources_sealed.json')['pins'] + read(FIT / 'endpoint_sources_sealed.json')['pins']:
        check(p)


def main():
    verify_sources()
    assert not {'numpy', 'torch', 'flopscope'} & set(sys.modules)
    assert not (HERE / 'run_claim.json').exists()
    for name in ('build.log', 'build_watch.json', 'build.json', 'endpoint_complete.json', 'candidate.py'):
        assert not (FIT / name).exists(), name
    rows = processes()
    expected = '/Users/matt/anaconda3/bin/python -B ' + str(FIT / 'train.py')
    workers = [r for r in rows if r['command'].startswith('/Users/matt/anaconda3/bin/python')]
    assert len(workers) == 1 and workers[0]['pid'] == workers[0]['pgid'] == FIT_GROUP
    assert workers[0]['command'] == expected
    start = time.monotonic()
    write(HERE / 'run_claim.json', dict(utc=datetime.now(timezone.utc).isoformat(),
        pid=os.getpid(), pgid=os.getpgrp(), existing_fit_worker=workers[0],
        fit_controller_session=66800, source_seal=pin(HERE / 'sources_sealed.json'),
        pending_endpoint=pin(FIT / 'run_endpoint.py'), only_one_endpoint_invocation=True))
    result = dict(completed=False, endpoint_started=False, endpoint_completed=False,
        endpoint_scientific_passed=None, error=None, no_fit_restart=True,
        no_downstream_panels=True, no_external_actions=True)
    print(dict(stage='waiting_for_existing_fit', group=FIT_GROUP), flush=True)
    try:
        while True:
            watch = ready(FIT / 'fit_watch.json')
            if watch is not None:
                break
            assert time.monotonic() - start < 14430, 'Existing-fit wait ceiling'
            time.sleep(1)
        result['fit_watch'] = pin(FIT / 'fit_watch.json')
        assert watch['pid'] == FIT_GROUP
        assert watch['completed'] and watch['exit_code'] == 0 and watch['whole_group_retired']
        assert not watch['remaining_rows'] and watch['max_swap_growth_bytes'] == 0
        receipt_deadline = time.monotonic() + 30
        while (completion := ready(FIT / 'fit_complete.json')) is None:
            assert time.monotonic() < receipt_deadline, 'Fit completion receipt absent after final guard'
            time.sleep(.25)
        assert completion['completed'] and not completion['validation_targets_accessed']
        check(completion['result'])
        check(completion['watch'])
        assert completion['watch'] == result['fit_watch']
        fit = read(completion['result']['path'])
        assert fit['completed'] and fit['passed'] and fit['model_fitted']
        assert fit['training_graphs'] == 160 and fit['updates'] == 40 and fit['epoch'] == 1
        assert not fit['validation_targets_accessed'] and not fit['checkpoint_selection']
        assert fit['fixed_parameters_unchanged'] and fit['inactive_parameters_unchanged']
        check(fit['terminal_payload'])
        verify_sources()
        rows = processes()
        assert not [r for r in rows if r['pid'] == FIT_GROUP or r['pgid'] == FIT_GROUP]
        assert not [r for r in rows if r['command'].startswith('/Users/matt/anaconda3/bin/python')
            or '/openblas_runtime_env_v1/bin/python -B ' in r['command']]
        command = [sys.executable, '-B', str(FIT / 'run_endpoint.py')]
        write(HERE / 'endpoint_start.json', dict(utc=datetime.now(timezone.utc).isoformat(),
            command=command, fit_completion=pin(FIT / 'fit_complete.json'),
            terminal_payload=fit['terminal_payload'], source_seal=pin(HERE / 'sources_sealed.json'),
            previous_numeric_group_retired=True, validation_targets_accessed=False))
        result['endpoint_started'] = True
        print(dict(stage='starting_sealed_endpoint'), flush=True)
        with (HERE / 'endpoint_controller.log').open('x') as log:
            rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        result['endpoint_controller_exit_code'] = rc
        assert rc == 0, 'Sealed endpoint controller failure'
        endpoint = read(FIT / 'endpoint_complete.json')
        assert endpoint['completed'] and endpoint['native_calls'] == 85
        assert endpoint['no_automatic_downstream_launch'] and not endpoint['official_changed']
        check(endpoint['score'])
        score = read(endpoint['score']['path'])
        assert score['completed'] and score['integrity_passed'] and score['native_score_measured']
        result.update(completed=True, endpoint_completed=True, endpoint_scientific_passed=score['passed'],
            fit_completion=pin(FIT / 'fit_complete.json'), endpoint_completion=pin(FIT / 'endpoint_complete.json'),
            score=endpoint['score'])
    except BaseException as error:
        result['error'] = repr(error)
    finally:
        result['wall_s'] = time.monotonic() - start
        result['numeric_modules_imported_in_controller'] = sorted({'numpy', 'torch', 'flopscope'} & set(sys.modules))
        assert not result['numeric_modules_imported_in_controller']
        write(HERE / 'result.json', result)
        print(json.dumps(result), flush=True)
    return 0 if result['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
