"""One serial literal build;60s/2GB, zero swap, exact completed observations."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
MON = NEW/'current_p1_native_objective_p2_monitor_v2/monitor.py'
CLASSIFIER = NEW/'current_d50_native_objective_p1_v1/watch.py'
FIXED = {
    MON: 'd4b3af53b716593670d8e0c7f0fc6f2133b57f1088d9966c06635d3a23f39873',
    CLASSIFIER: '4e8177d6ff943cdee37e48ccca750bf9d552b32cec344aa2b37e6304cbd55fb3',
}
COMMON = dict(
    source='d578b411601006c5a2ee890dd46005bd120a14cd0ebd4e4e5cd50e89f16cb71b',
    protocol='aac1bcada9e1c6ee45440dc207ee80dad0201d0195a86868b8cb9e97d0a060c3',
    model='f58c687782b1f5ff970dd52366366d83e33b81b85eb4f6d92d44d815973033db',
    model_protocol='1117752d081229cb4d6b64fff2fa86f65a6bb1a760ce9abd1458a6f26095833d',
    qualification_source='4544161a67ae7925c47d0ce1072b456116195d5abec71f6f55d0f900db397c3d',
    qualification_protocol='8cf7d8a66c0a0f20471bc6f612f2ea061da9d8b26c5f3be51c0c386d12c2253a',
    qualification_watcher='1800d669a5ddb12e9fd47f2409218024d924b22ea47ae2a4fc1841fcb77bf042',
    qualification_watch_spec='40d915077a897494b88beaaa3284b5a0d4836f90fbe8a532763c749b704f38a0',
    qualification_result='01b01857d11292a93ee7a9abf3928cdf82aba2dcdc03c0b468ab4daa5abc937a',
    qualification_watch='488fbabeacdab191bb08e705fbcaede2457954a6441cd0414b502f22c1478d30')
TRAIN = dict(
    training_source='f1ce872987d29676c7a415cd3cc346958bcd978c6aad17eec751cf1b5478463c',
    training_protocol='d9f6de519bc1bade3e4b3b76b2085967639b736ea84b645099821b16d88a8591',
    training_watcher='77f67a86fdefc3df81304771e5e5af7c38e37b7256bd8129ba66dc8e5052ef5c',
    training_result='c8040f6c0b9e6c0adc8cb54c9125714e727564039cee262c34538d2d6e8b1ede',
    training_watch='6dcc406f78019107686093c45a6896eb92681376038a196515df31c1df24f247')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true'); parser.add_argument('--role', choices=('zero', 'terminal'))
    parser.add_argument('--source-sha256'); args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    began = time.monotonic()
    require(args.role and os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'One fixed nice10 build')
    pins = {**FIXED, Path(__file__).resolve(): args.source_sha256,
            HERE/'build.py': COMMON['source'], HERE/'BUILD_PROTOCOL.md': COMMON['protocol']}
    require(all(re.fullmatch('[0-9a-f]{64}', h or '') and sha(p) == h for p, h in pins.items()), 'Reviewed source owners')
    spec = importlib.util.spec_from_file_location('covariance_build_resource_monitor', MON)
    mon = importlib.util.module_from_spec(spec); spec.loader.exec_module(mon)
    nodes = [n for n in ast.parse(CLASSIFIER.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'other_workers']
    require(len(nodes) == 1, 'Exact foreign-worker classifier')
    scope = dict(Path=Path, re=re, shlex=shlex, P2=P2, NEW=NEW)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(CLASSIFIER)+':original-classifier', 'exec'), scope)
    out = HERE/f'build_{args.role}_watch_v1'; out.mkdir(exist_ok=False)
    result = dict(schema='current_comp_joint_p_covariance_build_watch_v1', role=args.role,
        passed=False, completed=False, numerical_imports=False, full_model_calls=0,
        source_bindings={str(p): h for p, h in pins.items()}, maximum_combined_rss_bytes=0,
        maximum_interval_s=0, observation_samples=0)
    child = log = None; swap0 = None; previous = time.monotonic()
    def interrupted(*_):
        raise RuntimeError('Build supervisor interrupted')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    def observe(stream):
        nonlocal previous, swap0
        deadline = previous+1.; sample = mon.collect(deadline)
        owned = mon.scope(sample['rows'], os.getpid(), None if child is None else child.pid)
        require(not scope['other_workers'](sample['rows'], owned['allowed']), 'No other numerical worker')
        if swap0 is None:
            swap0 = sample['swap']['used_bytes']
        require(sample['swap']['used_bytes'] <= swap0 and owned['combined_rss_bytes'] < 2_000_000_000,
                'Zero added swap and2GB combined RSS')
        require(time.monotonic()-began < 60, 'Whole60s build including supervision')
        result['maximum_combined_rss_bytes'] = max(result['maximum_combined_rss_bytes'], owned['combined_rss_bytes'])
        stream.write(json.dumps(dict(elapsed_s=time.monotonic()-began, swap_bytes=sample['swap']['used_bytes'],
            combined_rss_bytes=owned['combined_rss_bytes'], child_pids=owned['child_pids']))+'\n'); stream.flush()
        previous, interval = mon.complete(previous, deadline)
        result['maximum_interval_s'] = max(result['maximum_interval_s'], interval); result['observation_samples'] += 1
    try:
        with (out/'samples.jsonl').open('x') as samples:
            observe(samples)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
            for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                env[key] = '1'
            reviewed = {**COMMON, **(TRAIN if args.role == 'terminal' else {})}
            argv = ['/Users/matt/anaconda3/bin/python', str(HERE/'build.py'), '--build', '--role', args.role]
            for key, value in reviewed.items():
                argv += ['--'+key.replace('_', '-')+'-sha256', value]
            log = (out/'worker.log').open('x')
            child = subprocess.Popen(argv, env=env, cwd=NEW, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(out/'started.json', dict(child_pid=child.pid, argv=argv, source_bindings=result['source_bindings']))
            while child.poll() is None:
                observe(samples); time.sleep(.15)
            observe(samples); require(child.wait(timeout=1) == 0, 'Completed builder exit')
            mon.cleanup([child], time.monotonic()+.2)
            folder = HERE/f'build_{args.role}_v1'
            r, e, f = (json.loads((folder/n).read_text()) for n in ('result.json', 'completion.json', 'finalization.json'))
            require(r['passed'] and r['completed'] and e['passed'] and e['completed'] and f['passed'] and f['completed']
                and e['result_sha256'] == f['result_sha256'] == sha(folder/'result.json')
                and f['completion_sha256'] == sha(folder/'completion.json') and f['elapsed_s'] < 60
                and r['peak_rss_bytes'] < 2_000_000_000 and r['role'] == args.role and r['parameter_updates'] == 0
                and not r['estimator_executed'] and not r['truth_arrays_accessed'], 'Complete bounded source-only build seals')
            require(all(sha(p) == h for p, h in pins.items()), 'Supervisor sources unchanged')
            observe(samples)
            result.update(passed=True, completed=True, child_result_sha256=sha(folder/'result.json'),
                child_completion_sha256=sha(folder/'completion.json'), child_finalization_sha256=sha(folder/'finalization.json'),
                sources_unchanged=True, whole_owned_tree_retired=True)
    except BaseException as exc:
        result.update(passed=False, completed=False, error=repr(exc))
    finally:
        try:
            if child is not None:
                mon.cleanup([child], time.monotonic()+.5)
            mon.retire_pending(time.monotonic()+.5); result['cleanup_passed'] = True
        except BaseException as exc:
            result.update(passed=False, completed=False, cleanup_error=repr(exc))
        if log is not None:
            log.close()
        result.update(elapsed_s=time.monotonic()-began, initial_swap_bytes=swap0)
        if result['elapsed_s'] >= 60:
            result.update(passed=False, completed=False, error='Whole60s including cleanup/log-close')
        write(out/'result.json', result); rh = sha(out/'result.json')
        ok = result['passed'] and time.monotonic()-began < 60
        write(out/'completion.json', dict(passed=ok, completed=ok, result_sha256=rh))
        eh = sha(out/'completion.json'); elapsed = time.monotonic()-began; ok = ok and elapsed < 60
        write(out/'finalization.json', dict(passed=ok, completed=ok, result_sha256=rh, completion_sha256=eh, elapsed_s=elapsed))
    print(json.dumps(result, sort_keys=True)); return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
