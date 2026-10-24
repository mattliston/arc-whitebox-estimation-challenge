"""Inert, target-free gain-scalar interface diagnostic; never imports an estimator."""
import argparse
import ast
import base64
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
import struct
import sys
import time

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
P2 = NEW.parent
OPENBLAS = NEW / 'openblas_runtime_env_v1'
CLIENT = P2 / 'data_gate/.client-venv'
OSITE = OPENBLAS / 'lib/python3.11/site-packages'
CSITE = CLIENT / 'lib/python3.11/site-packages'
CANDIDATE = HERE / 'build_v1/estimator.py'
CANDIDATE_SHA = '4d98fef2c11c571efe1eca945018bb9fe9ece1454d16b4abde2eb52d71e755c3'
SCHEMA = 'current_comp_shared_old384_scalar_semantics_v1'
THREADS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
PINS = {
    CANDIDATE: CANDIDATE_SHA,
    OSITE / 'flopscope/_pointwise.py': '7d376515446f1b174bef375c48116a552098bd514f5552450876f56822beb0c7',
    OSITE / 'flopscope/_ndarray.py': 'fa5bbecb48b123ab77f31ced3da877d2e333aa5b403a324c33da591b967d3e74',
    OSITE / 'flopscope/_array_ops.py': 'e45e927f18c54352503a4951d3d925a5f31f92ecea245e32360825d71a05da6c',
    OSITE / 'flopscope/_dtype_billing.py': 'd6f4df8b2fcf162f41afde2dae23a66d894eba62f173e57d89051e5873569cb8',
    OSITE / 'flopscope/_budget.py': 'ad9e78172bd0e8175533d4aae75b67df749d96ecc1ca868135eb3d781e542d01',
    OSITE / 'flopscope_server/_request_handler.py': 'c6157a3512193573c425c93514c44cdb9438f91b7d52ced86e2ed2cff8348fde',
    OSITE / 'flopscope_server/_session.py': '76c22d0b368b03c98a33e97c5e33af42acaf509e7ec100b3dbe692e5775d9ee1',
    CSITE / 'flopscope/__init__.py': '86179621dc63a6e3e162da3c54817dfdf2f678b2b4d1b66aa536ac18bdf57be4',
    CSITE / 'flopscope/_remote_array.py': '8c74bfdcd08502d1b53daac4aa0b8c24809aaa1408c17e6178deb83a9876fd54',
    CSITE / 'flopscope/_budget.py': '8468c0cf31344ab28e470e3b40422e5c4eb0a361af24f51e918a6f6e1ebadc47',
    CSITE / 'flopscope/_connection.py': '6e1e822bdf1fec065d045dfb9da3e67b73e592b993d822aeacf250a218b78e82',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def numerical_modules():
    return sorted(n for n in sys.modules if n.split('.')[0] in
                  ('numpy', 'scipy', 'torch', 'flopscope', 'flopscope_server', 'threadpoolctl'))


def gain_source():
    """Parse just the existing source; evaluate only its 16-number gain literal."""
    raw = CANDIDATE.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == CANDIDATE_SHA, 'Unchanged candidate')
    source = raw.decode('utf-8')
    tree = ast.parse(source)
    assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == '_GNN_LITERAL' for t in n.targets)]
    require(len(assignments) == 1 and isinstance(assignments[0].value, ast.Dict), 'Unique P literal')
    mapping = assignments[0].value
    selected = [v for k, v in zip(mapping.keys, mapping.values)
                if isinstance(k, ast.Constant) and k.value == 'gain']
    require(len(selected) == 1 and isinstance(selected[0], (ast.List, ast.Tuple)), 'Unique gain vector')
    gains = ast.literal_eval(selected[0])
    require(len(gains) == 16 and all(type(x) in (float, int) and math.isfinite(x) for x in gains),
            'Sixteen finite original source values')
    original = '*(1+0.25*fnp.tanh(P["gain"][l]))'
    final_cast = 'return fnp.asarray(delta,dtype=fnp.float64)'
    parameter_dtype = '_GNN_PARAMS={k:fnp.asarray(v,dtype=fnp.float32) for k,v in _GNN_LITERAL.items()}'
    require(source.count(original) == source.count(final_cast) == source.count(parameter_dtype) == 1,
            'Exact original gain expression, final hook cast and FP32 P initialization')
    proof = dict(candidate_sha256=CANDIDATE_SHA, gain_literal_length=16,
                 gain_FP32_raw_sha256=hashlib.sha256(struct.pack('<16f', *gains)).hexdigest(),
                 original_expression=original, original_hook_return=final_cast,
                 candidate_imported=False, candidate_changed=False, only_gain_literal_evaluated=True)
    return list(gains), proof


def snapshot(value, mode):
    """Read bytes only: no numerical transformation, cast, or candidate callback."""
    dtype = str(value.dtype)
    shape = tuple(value.shape)
    require(dtype in ('float32', 'float64') and shape in ((), (1,), (16,), (1024,)), 'Tiny numeric output')
    if mode == 'remote' and hasattr(value, '_fetch_data'):
        raw, actual_shape, actual_dtype = value._fetch_data()
        require(tuple(actual_shape) == shape and str(actual_dtype) == dtype, 'Wire metadata matches bytes')
    elif mode == 'remote':
        require(shape == () and type(value).__name__ == 'RemoteScalar', 'Actual by-value scalar')
        raw = struct.pack('<f' if dtype == 'float32' else '<d', value.item())
    else:
        raw = memoryview(value).tobytes()
    count = math.prod(shape)
    code, width = ('f', 4) if dtype == 'float32' else ('d', 8)
    require(len(raw) == count * width, 'Exact snapshot byte count')
    values = struct.unpack('<' + str(count) + code, raw)
    require(all(math.isfinite(x) for x in values), 'All tiny diagnostic values finite')
    return dict(type_module=type(value).__module__, type_name=type(value).__name__, dtype=dtype,
                shape=list(shape), raw_sha256=hashlib.sha256(raw).hexdigest(),
                raw_base64=base64.b64encode(raw).decode('ascii'),
                scalar_value=values[0] if count == 1 else None,
                scalar_hex=values[0].hex() if count == 1 else None)


def arithmetic(flops, fnp, gains, mode):
    """Same original gain arithmetic; sliced control differs only in indexing."""
    rows = []
    with flops.BudgetContext(flop_budget=1_000_000, quiet=True,
                            namespace='shared-gain-scalar-' + mode) as ctx:
        start = int(ctx.flops_used)
        G = fnp.asarray(gains, dtype=fnp.float32)
        x = fnp.asarray([(i - 512) / 1024.0 for i in range(1024)], dtype=fnp.float32)
        inputs = dict(gain=snapshot(G, mode), vector=snapshot(x, mode))
        setup = int(ctx.flops_used) - start
        for layer in range(1, 16):
            for variant in ('indexed', 'slice_control'):
                operations = []

                def operation(name, action):
                    before = int(ctx.flops_used)
                    value = action()
                    after = int(ctx.flops_used)
                    state = snapshot(value, mode)
                    require(int(ctx.flops_used) == after, 'Byte inspection added no hidden arithmetic')
                    operations.append(dict(name=name, flops=after - before,
                                           cumulative_flops=after, value=state))
                    return value

                g = operation('gain_index', lambda: G[layer] if variant == 'indexed' else G[layer:layer + 1])
                t = operation('tanh', lambda: fnp.tanh(g))
                s = operation('quarter', lambda: 0.25 * t)
                f = operation('one_plus', lambda: 1 + s)
                y = operation('vector_multiply', lambda: x * f)
                z = operation('hook_float64', lambda: fnp.asarray(y, dtype=fnp.float64))
                rows.append(dict(layer=layer, variant=variant, operations=operations,
                                 complete_expression_flops=sum(op['flops'] for op in operations)))
                del g, t, s, f, y, z, operation
        del G, x
        gc.collect()
    summary = ctx.summary_dict()
    total = setup + sum(r['complete_expression_flops'] for r in rows)
    require(total == int(summary['flops_used']) and len(rows) == 30, 'Every actual FLOP is owned')
    ledger = [dict(op._asdict()) for op in ctx.op_log] if mode == 'local' else None
    if ledger is not None:
        require(sum(int(op['flop_cost']) for op in ledger) == total, 'Local full operation ledger')
    result = dict(inputs=inputs, input_preparation_flops=setup, rows=rows,
                  complete_flops=total, bill=summary, local_operation_ledger=ledger,
                  remote_per_operation_costs_are_actual_budget_deltas=(mode == 'remote'))
    del ctx
    gc.collect()
    return result


def compare(local, remote):
    """Stdlib-only diagnostic comparison, not a release or candidate-score gate."""
    require(local['passed'] and remote['passed'] and local['completed'] and remote['completed']
            and local['mode'] == 'local' and remote['mode'] == 'remote',
            'Two operationally complete actual modes')
    require(local['source_bindings'] == remote['source_bindings']
            and local['proof'] == remote['proof'], 'Same source and exact gain values')
    require(len(local['measurement']['rows']) == len(remote['measurement']['rows']) == 30,
            'Both full fixed diagnostic rosters')
    for key in ('gain', 'vector'):
        a, b = local['measurement']['inputs'][key], remote['measurement']['inputs'][key]
        require((a['dtype'], a['shape'], a['raw_base64']) == (b['dtype'], b['shape'], b['raw_base64']),
                'Exact matched paid inputs')
    details = []
    for a, b in zip(local['measurement']['rows'], remote['measurement']['rows'], strict=True):
        require((a['layer'], a['variant']) == (b['layer'], b['variant']), 'Fixed matched roster')
        expected = 1006 if a['variant'] == 'indexed' else 0
        detail = dict(layer=a['layer'], variant=a['variant'],
                      local_minus_remote_flops=a['complete_expression_flops'] - b['complete_expression_flops'],
                      predicted_local_minus_remote_flops=expected, operations=[])
        for x, y in zip(a['operations'], b['operations'], strict=True):
            require(x['name'] == y['name'], 'Matched scalar/vector operation')
            detail['operations'].append(dict(name=x['name'], local_dtype=x['value']['dtype'],
                remote_dtype=y['value']['dtype'], local_flops=x['flops'], remote_flops=y['flops'],
                raw_byte_exact=x['value']['raw_base64'] == y['value']['raw_base64']))
        details.append(detail)
    return dict(rows=details, measured_local_minus_remote_complete_flops=
                local['measurement']['complete_flops'] - remote['measurement']['complete_flops'],
                predicted_local_minus_remote_complete_flops=15090,
                all_predicted_expression_tariffs_observed=all(
                    r['local_minus_remote_flops'] == r['predicted_local_minus_remote_flops'] for r in details),
                slice_control_all_intermediate_bytes_exact=all(all(o['raw_byte_exact'] for o in r['operations'])
                    for r in details if r['variant'] == 'slice_control'),
                model_calls=0, truth_accessed=False, candidate_changed=False,
                full_estimator_fidelity_established=False)


def write(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    require(len(raw) <= 2 * 1024**2, 'Bounded diagnostic JSON')
    with Path(path).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--mode', choices=('local', 'remote'))
    parser.add_argument('--source-sha256')
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--server-url')
    parser.add_argument('--out', help='Optional explicit spelling of the fixed exclusive mode directory')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    require(args.mode and args.source_sha256 and args.protocol_sha256, 'Explicit mode and two reviewed source pins')
    require(not numerical_modules(), 'No numerical modules before source admission')
    began = time.monotonic()
    out = HERE / 'scalar_semantics_v1' / args.mode
    require(not args.out or Path(args.out).resolve() == out.resolve(), 'Only the fixed fresh mode output path')
    out.mkdir(parents=True, exist_ok=False)
    result = dict(schema=SCHEMA, mode=args.mode, passed=False, completed=False,
                  model_calls=0, truth_accessed=False, candidate_changed=False,
                  full_estimator_fidelity_established=False, source_bindings={}, input_bindings={}, artifacts={})

    def guard(*unused):
        require(time.monotonic() - began < 60, 'Internal whole-mode 60-second deadline')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 1_000_000_000, 'Worker RSS below1GB')

    def alarm(*unused):
        raise TimeoutError('Internal mode deadline')

    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        owners = {**PINS, Path(__file__).resolve(): args.source_sha256,
                  HERE / 'SCALAR_SEMANTICS_PROTOCOL.md': args.protocol_sha256}
        for path, digest in owners.items():
            guard()
            require(sha(path) == digest, 'Exact reviewed source: ' + str(path))
        result['source_bindings'] = {str(p): d for p, d in owners.items()}
        require(sys.platform == 'darwin' and sys.byteorder == 'little' and sys.flags.optimize == 0,
                'Original macOS little-endian nonoptimized Python interface')
        require(not os.environ.get('PYTHONPATH') and not os.environ.get('PYTHONOPTIMIZE'), 'No environment import overrides')
        require(all(os.environ.get(k) == '1' for k in THREADS), 'One-thread environment')
        os.nice(max(0, 10 - os.getpriority(os.PRIO_PROCESS, 0)))
        expected_prefix = OPENBLAS if args.mode == 'local' else CLIENT
        require(Path(sys.prefix).resolve() == expected_prefix.resolve(), 'Exact mode environment')
        versions = {'flopscope': '0.12.0', 'numpy': '2.4.6', 'flopscope-server': '0.12.0',
                    'threadpoolctl': '3.6.0'} if args.mode == 'local' else {'flopscope-client': '0.12.0'}
        require(all(importlib.metadata.version(k) == v for k, v in versions.items()), 'Exact package versions')
        if args.mode == 'remote':
            require(args.server_url and args.server_url.startswith('ipc:///private/tmp/')
                    and args.server_url.endswith('/flops.sock'), 'Explicit root-owned local IPC only')
            require(os.environ.get('FLOPSCOPE_SERVER_URL') == args.server_url, 'Same root-owned server endpoint')
            require(importlib.util.find_spec('numpy') is None, 'NumPy-free production client')
        else:
            require(not args.server_url and not os.environ.get('FLOPSCOPE_SERVER_URL'), 'Actual in-process local mode')
        gains, proof = gain_source()
        result.update(proof=proof, package_versions=versions)
        guard()
        require(not numerical_modules(), 'Metadata admission did not import numerical modules')
        import flopscope as flops
        import flopscope.numpy as fnp
        site = OSITE if args.mode == 'local' else CSITE
        require(Path(flops.__file__).resolve() == site / 'flopscope/__init__.py', 'Actual FNP origin')
        result['measurement'] = arithmetic(flops, fnp, gains, args.mode)
        require(result['measurement']['inputs']['gain']['raw_sha256'] == proof['gain_FP32_raw_sha256'],
                'Actual paid gain vector matches exact original FP32 bytes')
        if args.mode == 'remote':
            require('numpy' not in sys.modules, 'No NumPy at any client arithmetic point')
            from flopscope._connection import reset_connection
            reset_connection()
        del gains, fnp, flops
        gc.collect()
        for path, digest in owners.items():
            guard()
            require(sha(path) == digest, 'Fresh final source hash')
        result.update(passed=True, completed=True, own_complete_bill=True,
                      source_hashes_unchanged=True, thirty_expressions_completed=True,
                      numerical_imports=('local_numpy_and_flopscope' if args.mode == 'local' else 'client_flopscope_only'),
                      external_server_retirement_required=(args.mode == 'remote'))
    except BaseException as error:
        result.update(passed=False, completed=False, error_type=type(error).__name__, error=str(error)[:512])
    finally:
        if not result['passed']:
            signal.setitimer(signal.ITIMER_REAL, 0)
        result.update(elapsed_s=time.monotonic() - began,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if result['passed']:
            guard()
        digest = write(out / 'result.json', result)
        end = write(out / 'completion.json', dict(schema=SCHEMA, passed=result['passed'], completed=result['completed'],
                                                 result_sha256=digest))
        if result['passed']:
            guard()
        elapsed = time.monotonic() - began
        write(out / 'finalization.json', dict(schema=SCHEMA, passed=result['passed'], completed=result['completed'],
              result_sha256=digest, completion_sha256=end, elapsed_s=elapsed))
        if result['passed']:
            guard()
        signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps(dict(mode=args.mode, passed=result['passed'], result_sha256=digest)), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
