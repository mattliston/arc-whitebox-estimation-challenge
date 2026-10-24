"""Inert OFFPY-only n11 native birth-direction helper qualification.

Run only with --execute --out-dir NEW_DIR after independent source review.
No candidate, network, learned checkpoint or target is imported or read.
The reference/fixture diagnostic is outside the native helper's bill;
native ingress, coefficient initialization and helper work are billed
separately.  This does not qualify an integrated estimator.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW = HERE.parent.parent
OFFENV = NEW.parent / 'data_gate' / '.venv'
PARENT = NEW / 'unused_materializations_v1' / 'estimator.py'
CORE = HERE / 'core.py'
REFERENCE = HERE / 'reference_numpy.py'
PINS = {
    PARENT: '409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f',
    CORE: 'd6c858751291d2305e4b1aa74fd828d9fa0c4f5fdd3009e239825324344431ee',
    REFERENCE: '447f7234ae6810058f753bfc735ca132a877014e55d8b37c73a466cbbd8be493',
}
VERSIONS = {'numpy': '2.4.6', 'flopscope': '0.12.0',
            'whestbench': '0.16.0', 'threadpoolctl': '3.6.0'}
BLAS_ENV = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
N, WALL, RSS, FLOP_CAP = 11, 60, 1024**3, 2**26


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path):
    stat = path.stat()
    return dict(path=str(path.resolve()), sha256=sha(path),
                bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def rehash(bindings):
    for name, item in bindings.items():
        require(binding(Path(item['path'])) == item, 'Source changed: ' + name)


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def resource_guard(start):
    require(peak_rss() < RSS, 'Tiny helper qualification exceeded1GiB RSS')
    require(time.perf_counter() - start < WALL, 'Tiny helper qualification exceeded60s')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Import specification')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(output_dir):
    start = time.perf_counter()
    require(Path(sys.prefix).resolve() == OFFENV.resolve(), 'Use exact OFFPY environment')
    require('torch' not in sys.modules and 'anaconda3' not in sys.executable, 'No AP/Torch')
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'Requires nice10')
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'Disable bytecode writes')
    require(all(os.environ.get(k) == '1' for k in BLAS_ENV), 'Explicit BLAS1 environment')
    require({name: importlib.metadata.version(name) for name in VERSIONS} == VERSIONS,
            'Requires frozen native versions')
    require(not output_dir.exists(), 'Qualification --out-dir must be new')
    for path, expected in PINS.items():
        require(sha(path) == expected, 'Frozen source pin: ' + str(path))
    paths = [PARENT, CORE, REFERENCE, HERE / 'build.py', HERE / 'SOURCE_SPEC.md',
             HERE.parent / 'PROTOCOL.md', Path(__file__).resolve()]
    library = OFFENV / 'lib' / 'python3.11' / 'site-packages' / 'flopscope'
    paths += [library / name for name in
              ('_flops.py', '_pointwise.py', '_array_ops.py', '_ndarray.py',
               'data/default_weights.json')]
    sources = {str(path.resolve()): binding(path) for path in paths}
    output_dir.mkdir(parents=True, exist_ok=False)
    result = dict(
        schema='birth_direction_fnp_small_v1', passed=False, completed=False,
        records=[], source_bindings=sources, versions=VERSIONS,
        n=N, fixed_order=['nonzero', 'null'],
        parent_sha256=PINS[PARENT], core_sha256=PINS[CORE],
        reference_sha256=PINS[REFERENCE],
        executable=sys.executable, sys_prefix=sys.prefix, pid=os.getpid(),
        nice=os.getpriority(os.PRIO_PROCESS, 0),
        blas_env={key: os.environ.get(key) for key in BLAS_ENV},
        network_or_target_access=False, candidate_imported=False, no_fit=True,
        full_card_qualification=False,
        reference_scope='Fixed synthetic fixture; explicit stored signed-cube plane.',
        billing_scope='Separate native ingress, coefficient initialization and helper only; diagnostic NumPy work not included.')
    write(output_dir / 'started.json', dict(result, utc=datetime.now(timezone.utc).isoformat()))

    def alarm(_signum, _frame):
        raise TimeoutError('Fixed60s tiny helper watchdog')

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(WALL)
    try:
        import numpy as np
        import flopscope as flops
        import flopscope.numpy as fnp
        from threadpoolctl import threadpool_info, threadpool_limits
        flops.configure(symmetry_warnings=False)
        core = load(CORE, 'birth_direction_native_core_small')
        reference = load(REFERENCE, 'birth_direction_numpy_reference_small')
        core.fnp = fnp
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            require(all(p.get('num_threads') == 1 for p in pools if p.get('user_api') == 'blas'),
                    'Reported BLAS threads differ from1')
            result['threadpools'] = pools
            result['blas_pool_empty'] = not any(p.get('user_api') == 'blas' for p in pools)
            fixture = reference.fixture(np)
            require(fixture['bp'].shape == (N, N), 'Fixed fixture dimension')
            for case in result['fixed_order']:
                inputs = dict(fixture)
                if case == 'null':
                    inputs['coefficients'] = np.zeros(4, dtype=np.float32)
                expected = reference.update(np, **inputs)
                native = {}
                with flops.BudgetContext(flop_budget=FLOP_CAP, wall_time_limit_s=WALL, quiet=True) as ingress:
                    for name, value in inputs.items():
                        if name == 'coefficients':
                            continue
                        if name == 'aux':
                            native[name] = {key: fnp.array(array, dtype=fnp.float64)
                                            for key, array in value.items()}
                        else:
                            dtype = fnp.bool_ if name in ('alive', 'births') else fnp.float64
                            native[name] = fnp.array(value, dtype=dtype)
                require(all(isinstance(value, fnp.ndarray) for name, value in native.items()
                            if name != 'aux') and all(isinstance(value, fnp.ndarray)
                                                      for value in native['aux'].values()),
                        'Native ingress must be FNP arrays')

                row = tuple(float(value) for value in inputs['coefficients'])
                core._BD_COEFFICIENT_LITERAL = (row,) * 15
                core._BD_COEFFICIENTS = None
                with flops.BudgetContext(flop_budget=FLOP_CAP, wall_time_limit_s=WALL, quiet=True) as setup:
                    coefficients = core._bd_coefficients()
                with flops.BudgetContext(flop_budget=FLOP_CAP, wall_time_limit_s=WALL, quiet=True) as reuse:
                    reused = core._bd_coefficients()
                require(reused is coefficients and int(reuse.flops_used) == 0,
                        'Lazy coefficient reuse must be identical and free')
                require(isinstance(coefficients, fnp.ndarray)
                        and coefficients.shape == (15, 4) and str(coefficients.dtype) == 'float32',
                        'Stored coefficient shape/dtype')
                begin = time.perf_counter()
                with flops.BudgetContext(flop_budget=FLOP_CAP, wall_time_limit_s=WALL, quiet=True) as ctx:
                    native['coefficients'] = coefficients[0]
                    new_base, delta = core._bd_update(**native)
                elapsed = time.perf_counter() - begin
                require(isinstance(new_base, fnp.ndarray) and isinstance(delta, fnp.ndarray),
                        'Core must return concrete native arrays')
                actual_base, actual_delta = np.asarray(new_base), np.asarray(delta)
                require(actual_base.shape == actual_delta.shape == (N, N)
                        and actual_base.dtype == np.dtype('float32')
                        and actual_delta.dtype == np.dtype('float64'),
                        'Native output shape/dtype; no repair allowed')
                require(bool(np.isfinite(actual_base).all()) and bool(np.isfinite(actual_delta).all())
                        and bool(np.isfinite(expected['delta_plane']).all()),
                        'Finite native/reference outputs')
                base_bytes_equal = actual_base.tobytes() == expected['new_base32'].tobytes()
                delta_max_abs = float(np.max(np.abs(actual_delta - expected['delta_plane'])))
                zero64 = np.zeros((N, N), dtype=np.float64)
                null_base_equal = actual_base.tobytes() == expected['old_base32'].tobytes()
                null_delta_equal = actual_delta.tobytes() == zero64.tobytes()
                record = dict(
                    case=case, n=N, selected=int(np.count_nonzero(inputs['births'])),
                    new_base_byte_exact=base_bytes_equal, delta_max_abs=delta_max_abs,
                    null_base_byte_exact=null_base_equal if case == 'null' else None,
                    null_delta_byte_exact=null_delta_equal if case == 'null' else None,
                    native_ingress_flops=int(ingress.flops_used),
                    coefficient_setup_flops=int(setup.flops_used),
                    coefficient_reuse_flops=int(reuse.flops_used),
                    helper_flops=int(ctx.flops_used), helper_outer_wall_s=elapsed,
                    new_base_sha256=hashlib.sha256(actual_base.tobytes()).hexdigest(),
                    delta_sha256=hashlib.sha256(actual_delta.tobytes()).hexdigest())
                result['records'].append(record)
                print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
                require(base_bytes_equal, 'New stored FP32 base must match reference bytes')
                require(delta_max_abs <= 1e-12, 'Stored-pair plane delta exceeds1e-12')
                require(case != 'null' or (null_base_equal and null_delta_equal),
                        'Zero coefficients must have exact null base/plane bytes')
                require(math.isfinite(elapsed) and elapsed >= 0 and int(ctx.flops_used) > 0,
                        'Finite nonnegative timing and positive native helper bill')
                resource_guard(start)
                del native, expected, new_base, delta, actual_base, actual_delta, coefficients, reused
            resource_guard(start)
        rehash(sources)
        require('torch' not in sys.modules, 'No Torch imported during OFFPY qualification')
        resource_guard(start)
        result.update(passed=True, completed=True, sources_unchanged=True)
    except BaseException:
        result['error'] = traceback.format_exc()
        try:
            rehash(sources)
            result['sources_unchanged'] = True
        except BaseException:
            result['sources_unchanged'] = False
    finally:
        signal.alarm(0)
    result.update(elapsed_s=time.perf_counter() - start, peak_rss_bytes=peak_rss(),
                  ended_utc=datetime.now(timezone.utc).isoformat())
    write(output_dir / 'result.json', result)
    write(output_dir / 'completion.json', dict(
        passed=result['passed'], completed=result['completed'],
        result_sha256=sha(output_dir / 'result.json')))
    print(json.dumps(dict(passed=result['passed'], elapsed_s=result['elapsed_s'],
                          peak_rss_bytes=result['peak_rss_bytes']), allow_nan=False), flush=True)
    if not result['passed']:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args()
    if not args.execute:
        parser.print_help()
        return
    if args.out_dir is None:
        parser.error('--execute requires --out-dir')
    run(args.out_dir.resolve())


if __name__ == '__main__':
    main()
