"""Inert source/provenance/resource utilities; no numerical imports."""
from datetime import datetime, timezone
import gc
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
import weakref

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
OFFPY = P2 / 'data_gate/.venv'
SITE = OFFPY / 'lib/python3.11/site-packages'
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
BUDGET = 2**41
PINS = {
    NEW / 'unused_materializations_v1/estimator.py': '409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f',
    SITE / 'flopscope/_budget.py': 'ad9e78172bd0e8175533d4aae75b67df749d96ecc1ca868135eb3d781e542d01',
    SITE / 'flopscope/_array_ops.py': 'e45e927f18c54352503a4951d3d925a5f31f92ecea245e32360825d71a05da6c',
    SITE / 'flopscope/_ndarray.py': 'fa5bbecb48b123ab77f31ced3da877d2e333aa5b403a324c33da591b967d3e74',
    SITE / 'flopscope/stats/_norm.py': 'f23e49201c8e09e2785814b2b31663cda1c6bb217a57a211532e42fc9fe3b65d',
    SITE / 'flopscope/stats/_erf.py': '4cdd75fdb85981fc385417082985478566b18e19fa67ebb7d41225b5efaaa3bd',
    SITE / 'whestbench/domain.py': 'db736fec3f983258e363750384d943db1b662a027c998dc494f7646586e3ade5',
    SITE / 'whestbench/sdk.py': 'b0fcc52c6b531981e46da6955365aa786260fab53fd66dcf3675791ed8c3c105',
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_float': repr(value)}
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    return value


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(safe(value), f, indent=2, allow_nan=False)
        f.write('\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def rss():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == 'darwin' else value * 1024


def sources():
    result = {}
    for path, expected in PINS.items():
        require(sha(path) == expected, 'Source pin: ' + str(path))
        result[str(path)] = expected
    for name in ('PROTOCOL.md', 'ROOT_DECISION.md', 'NUMERICAL_ANNEX.md', 'estimator.py',
                 'common.py', 'check_small.py', 'run_native.py'):
        path = HERE / name
        result[str(path)] = sha(path)
    return result


def verify(bindings):
    for path, expected in bindings.items():
        require(sha(path) == expected, 'Changed binding: ' + path)


def runtime():
    require(Path(sys.prefix).resolve() == OFFPY.resolve(), 'OFFPY only')
    require('anaconda3' not in sys.executable, 'No AP interpreter')
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'nice >=10 required')
    env = {k: os.environ.get(k) for k in
           ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
    require(all(v == '1' for v in env.values()), 'All BLAS environment caps =1')
    expected = {'numpy': '2.4.6', 'flopscope': '0.12.0',
                'whestbench': '0.16.0', 'threadpoolctl': '3.6.0'}
    versions = {k: importlib.metadata.version(k) for k in expected}
    require(versions == expected, 'Pinned native package versions')
    return dict(versions=versions, env=env, executable=sys.executable,
                sys_prefix=sys.prefix, nice=os.getpriority(os.PRIO_PROCESS, 0))


def load_estimator():
    spec = importlib.util.spec_from_file_location('fixed_gaussian_terminal_skew', HERE / 'estimator.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def alarm(seconds):
    def expired(signum, frame):
        raise TimeoutError('Fixed diagnostic wall watchdog')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)


def metrics(ctx, start):
    return dict(flops=int(ctx.flops_used), wall_s=time.monotonic() - start,
                context_wall_s=float(ctx.wall_time_s),
                flopscope_backend_s=float(ctx.flopscope_backend_time_s),
                flopscope_overhead_s=float(ctx.flopscope_overhead_time_s),
                native_residual_s=float(ctx.residual_wall_time_s),
                operation_count=len(ctx.op_log), peak_rss_bytes=rss())


def guard(record, wall=120, memory=8_000_000_000):
    require(0 < record['flops'] <= BUDGET, 'Complete FLOP bound')
    require(record['wall_s'] < wall and record['peak_rss_bytes'] < memory, 'Resource gate')
    require(all(math.isfinite(record[k]) and record[k] >= 0 for k in
                ('wall_s', 'context_wall_s', 'flopscope_backend_s',
                 'flopscope_overhead_s', 'native_residual_s')), 'Finite nonnegative timings')


def reset_closed(flops, ctx, bill):
    """Caller saves a closed bill first, then deletes its own ctx reference."""
    fields = (ctx.flops_used, ctx.wall_time_s, ctx.flopscope_backend_time_s,
              ctx.flopscope_overhead_time_s, ctx.residual_wall_time_s, len(ctx.op_log))
    ref = weakref.ref(ctx)
    flops.budget_reset()
    require(fields == (ctx.flops_used, ctx.wall_time_s, ctx.flopscope_backend_time_s,
                       ctx.flopscope_overhead_time_s, ctx.residual_wall_time_s, len(ctx.op_log)),
            'Closed bill changed after public diagnostic reset')
    return ref, dict(saved_bill=str(bill), saved_bill_sha256=sha(bill),
                     after_context_exit=True, reset_api='flopscope.budget_reset',
                     closed_bill_unchanged=True, candidate_state_reset=False)


def finish(out, result):
    result['finished_utc'] = utc()
    result['peak_rss_bytes'] = rss()
    write(out / 'result.json', result)
    write(out / 'completion.json', dict(passed=result['passed'], completed=True,
                                        result_sha256=sha(out / 'result.json')))
    print(json.dumps(safe(dict(passed=result['passed'], error=result.get('error'),
                               result_sha256=sha(out / 'result.json')))), flush=True)
