"""Inert, fixed n11 node-head qualification; no networks, truths or fitting.

Explicit stages: native (OFFPY FNP/NumPy), cpu (AP Torch CPU), mps (AP
Torch MPS, only after the sole MPS slot is free), judge (OFFPY NumPy).
Outputs are exclusive stage_native/stage_cpu/stage_mps/stage_judge dirs.
The judge requires all three sealed stage-result hashes.  Default/help
imports no numerical library.  A stage has a55s alarm/60s elapsed cap;
native/judge RSS<1GB, Torch RSS<2GB and MPS driver<2GB.  This is only the
prospective synthetic qualification in the frozen implementation annex.
"""

import argparse
import ast
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
import traceback

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
BD = NEW / 'birth_direction_fit_v1'
OFFENV = NEW.parent / 'data_gate/.venv'
APENV = Path('/Users/matt/anaconda3')
SITE = OFFENV / 'lib/python3.11/site-packages'
REFERENCE = BD / 'fnp/reference_numpy.py'
BLAS_ENV = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
VERSIONS = {'numpy': '2.4.6', 'flopscope': '0.12.0',
            'whestbench': '0.16.0', 'threadpoolctl': '3.6.0'}
N, WIDTH, CHANNELS, WALL, FLOP_CAP = 11, 64, 4, 60, 2**27
PINS = {
    HERE / 'model.py': '61833c47e71867f01376a0a3bf3a976b56b67024c485212015f38e7934aadc71',
    HERE / 'core.py': '8abad5831aed8a2fabf47f06b73c9867c482519bcd4dac666c01cc934ccd46a6',
    HERE / 'build.py': '42bef1c5c8aeaf0b60ae96961bb7c29d20535a555bbbfc554a61c5ec0aecb926',
    HERE / 'PROTOCOL.md': '52045c4d089092bc9728fa301a781b39934d8658ad3418ce9d459556554c8056',
    HERE / 'IMPLEMENTATION.md': 'a248f7e8a0edd45359af0efdc25b9d1491c551ae21ebf1ec4bffb9acfa3e1ceb',
    SITE / 'flopscope/_budget.py': 'ad9e78172bd0e8175533d4aae75b67df749d96ecc1ca868135eb3d781e542d01',
    SITE / 'flopscope/_array_ops.py': 'e45e927f18c54352503a4951d3d925a5f31f92ecea245e32360825d71a05da6c',
    SITE / 'flopscope/_ndarray.py': 'fa5bbecb48b123ab77f31ced3da877d2e333aa5b403a324c33da591b967d3e74',
    SITE / 'flopscope/_flops.py': 'bbdad83348b7ce7b63c9d8da66512c40c69e055c653d24687e62132bd34220c2',
    SITE / 'flopscope/_pointwise.py': '7d376515446f1b174bef375c48116a552098bd514f5552450876f56822beb0c7',
    SITE / 'flopscope/data/default_weights.json': 'f729bc47f07c080b8c7b9d7b045e5663f0af313669b4c2a962b5a9044a94cb7a',
}
BUILDS = {
    'zero': ('a560235a6cbee805b2d37f50b34583996d93e013fafcbb4c2b6243c408aa7f48',
             '1a99b49fcdc5cbfdaa03a40e676c0a910c5cf4e2d001965bf9e15e18678f1f05'),
    'nonzero': ('3eb96fcdf5553c211a21a1005f2043940b8fbcab033fca090f221e1dad5d8b41',
                'e3c242b9dfa53cb348dd90aeb9426f0dd5ecc4b45f062839297d22438124516f'),
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), sha256=sha(path), bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns)


def rehash(items):
    for item in items.values():
        require(binding(item['path']) == item, 'Changed sealed input: ' + item['path'])


def safe_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': repr(value)}
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    return value


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(safe_json(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Module specification')
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def resources(stage, start, torch=None):
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = int(value if sys.platform == 'darwin' else value * 1024)
    if stage == 'mps' and torch is None:
        torch = sys.modules.get('torch')
    driver = int(torch.mps.driver_allocated_memory()) if stage == 'mps' and torch else 0
    return dict(elapsed_s=time.perf_counter() - start, peak_rss_bytes=rss,
                mps_driver_bytes=driver)


def guard(stage, start, torch=None):
    data = resources(stage, start, torch)
    require(data['elapsed_s'] < WALL, 'Tiny stage exceeded60s')
    require(data['peak_rss_bytes'] < (2e9 if stage in ('cpu', 'mps') else 1e9),
            'Tiny stage RSS cap')
    require(data['mps_driver_bytes'] < 2e9, 'Tiny MPS driver cap')
    return data


def environment(stage, clear):
    expected = APENV if stage in ('cpu', 'mps') else OFFENV
    require(Path(sys.prefix).resolve() == expected.resolve(), 'Wrong interpreter environment')
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'Requires nice10')
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'Disable bytecode writes')
    require(all(os.environ.get(k) == '1' for k in BLAS_ENV), 'Explicit BLAS1 environment')
    require('torch' not in sys.modules and 'threadpoolctl' not in sys.modules,
            'Fresh stage process required')
    if stage in ('cpu', 'mps'):
        require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') == '0', 'No MPS fallback')
        versions = {key: importlib.metadata.version(key) for key in ('numpy', 'torch')}
        require(stage != 'mps' or clear, '--mps-slot-clear attests no other MPS job')
    else:
        versions = {key: importlib.metadata.version(key) for key in VERSIONS}
        require(versions == VERSIONS, 'Frozen OFFPY versions changed')
    return dict(executable=sys.executable, sys_prefix=sys.prefix, versions=versions,
                pid=os.getpid(), nice=os.getpriority(os.PRIO_PROCESS, 0),
                blas_env={key: os.environ.get(key) for key in BLAS_ENV},
                mps_slot_clear_attested=bool(clear) if stage == 'mps' else None)


def authenticate(result):
    sources = result['source_bindings']
    for path, expected in PINS.items():
        sources[str(path.resolve())] = binding(path)
        require(sha(path) == expected, 'Frozen checker dependency: ' + str(path))
    build = module(HERE / 'build.py', 'bdnh_tiny_inert_builder')
    for item in build.authenticate().values():
        sources[item['path']] = binding(item['path'])
    prototypes = {}
    for case, (build_sha, candidate_sha) in BUILDS.items():
        folder = HERE / f'build_{case}_v1'
        for name in ('build.json', 'completion.json', 'estimator.py', 'A.json'):
            sources[str((folder / name).resolve())] = binding(folder / name)
        require(sha(folder / 'build.json') == build_sha, 'Prototype build seal')
        require(sha(folder / 'estimator.py') == candidate_sha, 'Prototype source seal')
        record = json.loads((folder / 'build.json').read_text())
        seal = json.loads((folder / 'completion.json').read_text())
        require(record['passed'] and record['completed'] and not record['executed']
                and record['inverse_source_byte_exact'] and record['stored_birth_update_byte_exact'],
                'Prototype source proof')
        require(seal['passed'] and seal['completed'] and seal['build_sha256'] == build_sha,
                'Prototype completion')
        require(record['candidate_sha256'] == candidate_sha and record['fixture'] == case,
                'Prototype role identity')
        for item in record['source_bindings'].values():
            require(sha(item['path']) == item['sha256'], 'Prototype input changed')
            sources[item['path']] = binding(item['path'])
        for name, item in record['artifacts'].items():
            require(Path(item['path']).resolve() == (folder / name).resolve()
                    and sha(item['path']) == item['sha256'], 'Prototype artifact seal')
        data = json.loads((folder / 'A.json').read_text())
        literal = build.assignment_literal((folder / 'estimator.py').read_text(), '_BDNH_A_LITERAL')
        encoded = build.matrix_bytes(literal, 64, 4)
        require(encoded == build.matrix_bytes(data['A'], 64, 4)
                == build.matrix_bytes(build.prototype_matrix(case), 64, 4), 'Fixed A data identity')
        require(hashlib.sha256(encoded).hexdigest() == record['A_bytes_sha256'], 'A byte seal')
        prototypes[case] = dict(folder=folder, A=literal)
    result['prototype_seals'] = {case: dict(build_sha256=pair[0], candidate_sha256=pair[1])
                                 for case, pair in BUILDS.items()}
    return build, prototypes


def fixtures(np, build, prototypes):
    reference = module(REFERENCE, 'bdnh_tiny_frozen_numpy_reference')
    physical = reference.fixture(np)
    require(physical['bp'].shape == (N, N), 'Fixed physical n11 fixture')
    H = np.asarray([[math.sin((i+1)*(j+1)/19) + .1*math.cos((i+3)*(j+2)/11)
                     for j in range(WIDTH)] for i in range(N)], dtype=np.float32)
    H[:, 0] = np.float32(.25)
    cstar = np.asarray(build.frozen_coefficients()[8], dtype=np.float32)
    matrices = {case: np.asarray(item['A'], dtype=np.float32) for case, item in prototypes.items()}
    V = np.asarray([[math.sin((k+2)*(i+1)/7) for i in range(N)] for k in range(4)])
    D = np.asarray([[math.cos((j+1)*(k+2)/13)/8 for k in range(4)] for j in range(64)])
    nonsym = np.asarray([[.2*math.sin((k+1)*(j+2)) for j in range(N)]
                        for k in range(4)], dtype=np.float32)
    coefficients = {'zero': np.zeros((4, N), dtype=np.float32), 'scalar': cstar,
                    'broadcast': np.broadcast_to(cstar[:, None], (4, N)).copy(),
                    'nonsymmetric': nonsym}
    return reference, physical, H, cstar, matrices, V, D, coefficients


def head_reference(np, H, A, cstar, dtype):
    H, A, cstar = (np.asarray(x, dtype=dtype) for x in (H, A, cstar))
    centered = H - np.mean(H, axis=0, dtype=dtype)
    scale = np.sqrt(np.mean(centered*centered, axis=0, dtype=dtype) + dtype(1e-6))
    before_clip = centered/scale
    Z = np.clip(before_clip, dtype(-8), dtype(8))
    R = np.tanh(Z @ A)
    value = cstar[None, :] + (dtype(1)-np.abs(cstar))[None, :]*R
    return value.T, Z, R, before_clip


def ideal_fd(np, H, A, cstar, V, D):
    value, Z, R, before_clip = head_reference(np, H, A, cstar, np.float64)
    gradient = Z.T @ (V.T*(1-np.abs(cstar.astype(np.float64)))[None, :]*(1-R*R))
    analytic = float(np.sum(gradient*D))
    h = 1e-5
    plus = head_reference(np, H, A.astype(np.float64)+h*D, cstar, np.float64)[0]
    minus = head_reference(np, H, A.astype(np.float64)-h*D, cstar, np.float64)[0]
    finite = float((np.sum(plus*V)-np.sum(minus*V))/(2*h))
    relative = abs(finite-analytic)/max(abs(analytic), 1e-12)
    record = dict(step=h, analytic_directional=analytic, finite_difference=finite,
                  relative_error=relative, reference_value=float(np.sum(value*V)),
                  clip_boundary_count=int(np.count_nonzero(np.abs(before_clip) == 8)),
                  clipped_count=int(np.count_nonzero(np.abs(before_clip) > 8)))
    require(np.isfinite(gradient).all() and math.isfinite(relative), 'Finite ideal head derivative')
    return record, gradient


def coefficient_record(np, actual, expected, cstar, case):
    require(actual.shape == expected.shape == (4, N) and actual.dtype == np.float32,
            'Coefficient shape/dtype without casts')
    finite = bool(np.isfinite(actual).all())
    record = dict(case=case, finite=finite, dtype=str(actual.dtype), shape=list(actual.shape))
    if finite:
        record.update(max_abs_reference=float(np.max(np.abs(actual-expected))),
                      minimum=float(actual.min()), maximum=float(actual.max()),
                      zero_cstar_bytes=all(actual[:, j].tobytes() == cstar.tobytes()
                                           for j in range(N)) if case == 'zero' else None)
    return record


def cube_tensor(np, base, a, t, s, births):
    out = np.zeros((N, N, N), dtype=np.float64)
    for j in range(N):
        if not births[j]:
            continue
        p, q = base[:, j].astype(np.float64).copy(), base[:, j].astype(np.float64).copy()
        p[j], q[j] = float(t[j]), float(s[j])
        out += float(a[j])*(np.einsum('i,j,k->ijk', p, p, p)-np.einsum('i,j,k->ijk', q, q, q))
    return out


def physical_record(np, expected, actual_base, actual_delta, physical, case):
    require(actual_base.shape == actual_delta.shape == (N, N)
            and actual_base.dtype == np.float32 and actual_delta.dtype == np.float64,
            'Physical return shape/dtype without repairs')
    finite = bool(np.isfinite(actual_base).all() and np.isfinite(actual_delta).all())
    record = dict(case=case, finite=finite)
    if not finite:
        return record
    change = cube_tensor(np, actual_base, expected['a32'], expected['t32'],
                         expected['s32'], physical['births'])
    change -= cube_tensor(np, expected['old_base32'], expected['a32'], expected['t32'],
                          expected['s32'], physical['births'])
    repair = -3*actual_delta.T + 2*np.diag(np.diag(actual_delta))
    i, j, k = np.indices((N, N, N))
    star = (repair[i, j]*(j == k) + repair[j, i]*(i == k) + repair[k, i]*(i == j))/3
    restored = change+star
    repeated = (i == j) | (i == k) | (j == k)
    record.update(new_base_byte_exact=actual_base.tobytes() == expected['new_base32'].tobytes(),
        delta_max_abs=float(np.max(np.abs(actual_delta-expected['delta_plane']))),
        delta_diagonal_zero=bool(np.count_nonzero(np.diag(actual_delta)) == 0),
        repeated_restoration_max_abs=float(np.max(np.abs(restored[repeated]))),
        distinct_change_exact=bool(np.array_equal(restored[~repeated], change[~repeated])),
        factor_columns=2*int(np.count_nonzero(physical['births'])),
        zero_new_base_bytes=(actual_base.tobytes() == expected['old_base32'].tobytes()) if case == 'zero' else None,
        zero_delta_bytes=(actual_delta.tobytes() == np.zeros((N, N), np.float64).tobytes()) if case == 'zero' else None)
    return record


def check_physical(record):
    require(record['finite'] and record['new_base_byte_exact'], 'Physical FP32 storage mismatch')
    require(record['delta_max_abs'] <= 1e-12 and record['repeated_restoration_max_abs'] <= 1e-12,
            'Stored-cube/plane/restoration error exceeds1e-12')
    require(record['delta_diagonal_zero'] and record['distinct_change_exact'], 'Physical repair ownership')
    if record['case'] == 'zero':
        require(record['zero_new_base_bytes'] and record['zero_delta_bytes'], 'Zero total action is not byte-null')


def metered(flops, result, name, function):
    context, completed = None, False
    try:
        with flops.BudgetContext(flop_budget=FLOP_CAP, wall_time_limit_s=WALL, quiet=True) as context:
            value = function()
        completed = True
    finally:
        if context is not None:
            item = dict(name=name, completed=completed, flops=int(context.flops_used),
                        operation_count=len(context.op_log), context_wall_s=float(context.wall_time_s),
                        backend_s=float(context.flopscope_backend_time_s),
                        overhead_s=float(context.flopscope_overhead_time_s),
                        residual_s=float(context.residual_wall_time_s))
            result['bills'].append(item)
            flops.budget_reset()  # Public log reset, AFTER the context closes and its bill is copied.
            del context
            gc.collect()
    require(item['flops'] >= 0 and all(math.isfinite(item[key]) and item[key] >= 0
            for key in ('context_wall_s', 'backend_s', 'overhead_s', 'residual_s')),
            'Finite nonnegative closed-context bill/timings')
    return value


def native_stage(np, build, prototypes, result, outputs, start):
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_info, threadpool_limits
    flops.configure(symmetry_warnings=False)
    reference, physical, H, cstar, matrices, V, D, cases = fixtures(np, build, prototypes)
    outputs.update(H=H, cstar=cstar, A_zero=matrices['zero'], A_nonzero=matrices['nonzero'], V=V, D=D)
    src = (prototypes['nonzero']['folder'] / 'estimator.py').read_text()
    env = {'fnp': fnp, '_BDNH_PARAMETERS': None}
    for name in ('_bdnh_parameters', '_bdnh_coefficients', '_bd_update'):
        body = build.last_function(src, name)[2]
        other = build.last_function((prototypes['zero']['folder'] / 'estimator.py').read_text(), name)[2]
        require(body == other, 'Both sealed prototypes must use identical helper functions')
        exec(compile(body, str(HERE / 'check_tiny.py') + ':' + name, 'exec'), env)
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        require(all(pool['num_threads'] == 1 for pool in pools if pool['user_api'] == 'blas'), 'BLAS1')
        result['threadpools'] = pools
        result['blas_pool_empty'] = not any(pool['user_api'] == 'blas' for pool in pools)
        hf, cf = metered(flops, result, 'head_input_copy',
                         lambda: (fnp.array(H, dtype=fnp.float32), fnp.array(cstar, dtype=fnp.float32)))
        for case in ('zero', 'nonzero'):
            env['_BDNH_PARAMETERS'] = None
            env['_BDNH_A_LITERAL'] = prototypes[case]['A']
            af = metered(flops, result, case + '_A_initialization', env['_bdnh_parameters'])
            repeated = metered(flops, result, case + '_A_reuse', env['_bdnh_parameters'])
            require(repeated is af and result['bills'][-1]['flops'] == 0, 'A lazy reuse identity')
            value = metered(flops, result, case + '_head', lambda: env['_bdnh_coefficients'](hf, af, cf))
            require(isinstance(value, fnp.ndarray), 'Native coefficient result must be an FNP array')
            actual = np.asarray(value)
            expected = head_reference(np, H, matrices[case], cstar, np.float32)[0]
            record = coefficient_record(np, actual, expected, cstar, case)
            result['coefficient_checks'].append(record)
            outputs['coefficients_' + case] = actual.copy()
            outputs['reference_coefficients_' + case] = expected.copy()
            require(record['finite'] and record['max_abs_reference'] <= 2e-7, 'Native coefficient2e-7 gate')
            require(record['minimum'] >= -1 and record['maximum'] <= 1, 'Coefficient bounds')
            require(case != 'zero' or record['zero_cstar_bytes'], 'Zero head must reproduce cstar bytes')
            guard('native', start)
        def ingress():
            values = {}
            for key, value in physical.items():
                if key == 'coefficients':
                    continue
                if key == 'aux':
                    values[key] = {k: fnp.array(v, dtype=fnp.float64) for k, v in value.items()}
                else:
                    values[key] = fnp.array(value, dtype=fnp.bool_ if key in ('alive', 'births') else fnp.float64)
            return values
        native = metered(flops, result, 'physical_input_copy', ingress)
        require(all(isinstance(v, fnp.ndarray) for k, v in native.items() if k != 'aux')
                and all(isinstance(v, fnp.ndarray) for v in native['aux'].values()), 'Owned FNP physical ingress')
        for case, coefficient in cases.items():
            inputs = dict(physical, coefficients=coefficient)
            expected = reference.update(np, **inputs)
            native['coefficients'] = metered(flops, result, case + '_coefficient_copy',
                                            lambda: fnp.array(coefficient, dtype=fnp.float32))
            new, delta = metered(flops, result, case + '_birth_update', lambda: env['_bd_update'](**native))
            require(isinstance(new, fnp.ndarray) and isinstance(delta, fnp.ndarray), 'Concrete FNP physical outputs')
            actual_base, actual_delta = np.asarray(new), np.asarray(delta)
            record = physical_record(np, expected, actual_base, actual_delta, physical, case)
            result['physical_checks'].append(record)
            outputs[case + '_base'] = actual_base.copy()
            outputs[case + '_delta'] = actual_delta.copy()
            check_physical(record)
            guard('native', start)
        for suffix in ('base', 'delta'):
            require(outputs['scalar_' + suffix].tobytes() == outputs['broadcast_' + suffix].tobytes(),
                    'Scalar/birth-column broadcast byte identity')
        result['scalar_broadcast_byte_exact'] = True
        record, gradient = ideal_fd(np, H, matrices['nonzero'], cstar, V, D)
        result['ideal_fd'] = record
        outputs['ideal_gradient'] = gradient
        require(record['relative_error'] <= 1e-6, 'Fixed ideal FP64 FD gate')
    require('torch' not in sys.modules, 'OFFPY native stage imported Torch')


def torch_stage(np, build, prototypes, result, outputs, start, stage):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require('threadpoolctl' not in sys.modules, 'Never AP plus threadpoolctl')
    require(stage != 'mps' or torch.backends.mps.is_available(), 'MPS unavailable')
    model = module(HERE / 'model.py', 'bdnh_tiny_qualified_torch_model')
    reference, physical, H, cstar, matrices, V, D, cases = fixtures(np, build, prototypes)
    outputs.update(H=H, cstar=cstar, A_zero=matrices['zero'], A_nonzero=matrices['nonzero'], V=V, D=D)
    device = 'mps' if stage == 'mps' else 'cpu'
    for case in ('zero', 'nonzero'):
        ht = torch.tensor(H, device=device, dtype=torch.float32, requires_grad=(case == 'nonzero'))
        at = torch.tensor(matrices[case], device=device, dtype=torch.float32, requires_grad=(case == 'nonzero'))
        ct = torch.tensor(cstar, device=device, dtype=torch.float32)
        value = model.node_coefficients(ht, at, ct)
        if case == 'nonzero':
            loss = (value*torch.tensor(V, device=device, dtype=torch.float32)).sum()
            loss.backward()
            require(at.grad is not None and ht.grad is not None, 'Gradients must reach A AND H')
            grad = at.grad.detach().cpu().numpy().copy()
            hgrad = ht.grad.detach().cpu().numpy().copy()
            outputs['A_gradient'], outputs['H_gradient'] = grad, hgrad
            norm = float(np.linalg.norm(grad.astype(np.float64)))
            result['gradient'] = dict(norm=norm, H_norm=float(np.linalg.norm(hgrad.astype(np.float64))),
                                      finite=bool(np.isfinite(grad).all() and np.isfinite(hgrad).all()))
            require(result['gradient']['finite'] and norm > 1e-8, 'Finite non-negligible FP32 A gradient')
        if stage == 'mps':
            torch.mps.synchronize()
        actual = value.detach().cpu().numpy().copy()
        expected = head_reference(np, H, matrices[case], cstar, np.float32)[0]
        record = coefficient_record(np, actual, expected, cstar, case)
        result['coefficient_checks'].append(record)
        outputs['coefficients_' + case] = actual
        outputs['reference_coefficients_' + case] = expected.copy()
        require(record['finite'] and record['max_abs_reference'] <= 2e-6, 'Torch coefficient2e-6 gate')
        require(record['minimum'] >= -1 and record['maximum'] <= 1, 'Coefficient bounds')
        require(case != 'zero' or record['zero_cstar_bytes'], 'Zero head cstar bytes')
        guard(stage, start, torch)
    if stage == 'cpu':
        tensor = lambda x: torch.tensor(x, dtype=torch.float64)
        aux = {key: tensor(value) for key, value in physical['aux'].items()}
        for case, coefficient in cases.items():
            expected = reference.update(np, **dict(physical, coefficients=coefficient))
            new, delta = model.birth_action(tensor(physical['bp']), tensor(physical['Cpre']),
                tensor(physical['K21']), aux, torch.tensor(coefficient, dtype=torch.float32),
                tensor(physical['t']), tensor(physical['s']), tensor(physical['a']),
                tensor(physical['alive']), tensor(physical['births']))
            actual_base, actual_delta = new.detach().numpy(), delta.detach().numpy()
            record = physical_record(np, expected, actual_base, actual_delta, physical, case)
            result['physical_checks'].append(record)
            outputs[case + '_base'] = actual_base.copy()
            outputs[case + '_delta'] = actual_delta.copy()
            check_physical(record)
            guard(stage, start, torch)
        for suffix in ('base', 'delta'):
            require(outputs['scalar_' + suffix].tobytes() == outputs['broadcast_' + suffix].tobytes(),
                    'Torch scalar/broadcast bytes')
        result['scalar_broadcast_byte_exact'] = True
        record, gradient = ideal_fd(np, H, matrices['nonzero'], cstar, V, D)
        result['ideal_fd'] = record
        outputs['ideal_gradient'] = gradient
        require(record['relative_error'] <= 1e-6, 'Fixed ideal FP64 FD gate')
    require('threadpoolctl' not in sys.modules and 'flopscope' not in sys.modules,
            'AP stage mixed incompatible runtime libraries')
    result['torch_threads'] = torch.get_num_threads()
    result.update(guard(stage, start, torch))


def verify_stage(stage, digest, result):
    folder = HERE / ('stage_' + stage)
    require(sha(folder / 'result.json') == digest, 'Requested stage result hash')
    record = json.loads((folder / 'result.json').read_text())
    completion = json.loads((folder / 'completion.json').read_text())
    require(record['passed'] and record['completed'] and record['sources_unchanged']
            and completion['passed'] and completion['completed']
            and completion['result_sha256'] == digest, 'Completed stage seal')
    require(record['stage'] == stage and record['schema'] == 'birth_direction_node_head_tiny_v1', 'Stage identity')
    require(record['prototype_seals'] == result['prototype_seals'], 'Same prototype pair required')
    rehash(record['source_bindings'])
    require(set(record['artifacts']) == {'arrays.npz'}, 'Exact synthetic artifact set')
    item = record['artifacts']['arrays.npz']
    require(Path(item['path']).resolve() == (folder / 'arrays.npz').resolve(), 'Fixed artifact path')
    require(binding(folder / 'arrays.npz') == item, 'Synthetic array artifact hash')
    for path in (folder / 'result.json', folder / 'completion.json', folder / 'arrays.npz'):
        result['source_bindings'][str(path.resolve())] = binding(path)
    return record, folder / 'arrays.npz'


def judge_stage(np, args, result, start):
    verified = {stage: verify_stage(stage, getattr(args, stage + '_result_sha256'), result)
                for stage in ('native', 'cpu', 'mps')}
    result['stage_seals'] = {stage: getattr(args, stage + '_result_sha256') for stage in verified}
    result['all_stage_artifacts_verified_before_read'] = True
    write(HERE / 'stage_judge/input_seal.json', dict(source_bindings=result['source_bindings'],
        stage_seals=result['stage_seals'], synthetic_only=True, utc=datetime.now(timezone.utc).isoformat()))
    arrays = {}
    for stage, (record, path) in verified.items():
        with np.load(path, allow_pickle=False) as stored:
            arrays[stage] = {key: stored[key].copy() for key in stored.files}
        require(all(np.isfinite(value).all() for value in arrays[stage].values()), 'Finite sealed synthetic arrays')
    require(verified['cpu'][0]['environment']['versions'] == verified['mps'][0]['environment']['versions'],
            'Literal CPU/MPS checks must use the same library versions')
    for key in ('H', 'cstar', 'A_zero', 'A_nonzero', 'V', 'D'):
        a = arrays['native'][key]
        for stage in ('cpu', 'mps'):
            b = arrays[stage][key]
            require(a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(),
                    'All stages must use the identical fixed data: ' + key)
    checks = []
    for case in ('zero', 'nonzero'):
        native = arrays['native']['coefficients_' + case]
        for stage in ('cpu', 'mps'):
            value = arrays[stage]['coefficients_' + case]
            maximum = float(np.max(np.abs(value.astype(np.float64)-native.astype(np.float64))))
            row = dict(stage=stage, case=case, native_max_abs=maximum)
            checks.append(row)
            require(maximum <= 2e-6, 'CPU/MPS versus native coefficient gate')
    cpu = arrays['cpu']['A_gradient'].astype(np.float64)
    mps = arrays['mps']['A_gradient'].astype(np.float64)
    norm = float(np.linalg.norm(cpu))
    relative = float(np.linalg.norm(mps-cpu)/max(norm, 1e-300))
    result['A_gradient_comparison'] = dict(cpu_norm=norm, mps_norm=float(np.linalg.norm(mps)),
                                          relative_error=relative)
    require(norm > 1e-8 and float(np.linalg.norm(mps)) > 1e-8 and relative <= .05,
            'Fixed literal FP32 CPU/MPS gradient5% gate')
    for case in ('zero', 'scalar', 'broadcast', 'nonsymmetric'):
        require(arrays['native'][case + '_base'].tobytes() == arrays['cpu'][case + '_base'].tobytes(),
                'Native/CPU physical stored-base bytes')
        error = float(np.max(np.abs(arrays['native'][case + '_delta']-arrays['cpu'][case + '_delta'])))
        require(error <= 1e-12, 'Native/CPU physical delta1e-12')
        checks.append(dict(case=case, stored_base_bytes=True, delta_max_abs=error))
    result['cross_stage_checks'] = checks
    result['small_fidelity_passed'] = True
    result['wide_or_training_admitted_by_this_receipt'] = False
    guard('judge', start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('native', 'cpu', 'mps', 'judge'))
    parser.add_argument('--mps-slot-clear', action='store_true')
    for stage in ('native', 'cpu', 'mps'):
        parser.add_argument('--' + stage + '-result-sha256')
    args = parser.parse_args()
    if args.stage is None:
        parser.print_help()
        return
    if args.stage == 'judge' and not all(getattr(args, stage + '_result_sha256')
                                       for stage in ('native', 'cpu', 'mps')):
        parser.error('--stage judge requires the three completed stage-result hashes')
    folder = HERE / ('stage_' + args.stage)
    folder.mkdir(exist_ok=False)
    start = time.perf_counter()
    result = dict(schema='birth_direction_node_head_tiny_v1', stage=args.stage,
                  passed=False, completed=False, source_bindings={str(Path(__file__).resolve()): binding(__file__)},
                  artifacts={}, bills=[], coefficient_checks=[], physical_checks=[],
                  no_network_or_target_access=True, no_fit=True, full_card_qualification=False,
                  billing_scope='Native helper, parameter setup and ingress separated; not a full card bill.',
                  started_utc=datetime.now(timezone.utc).isoformat())
    outputs, np = {}, None
    write(folder / 'started.json', result)
    def timeout(_signum, _frame):
        raise TimeoutError('Fixed55s tiny-stage watchdog')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(55)
    try:
        result['environment'] = environment(args.stage, args.mps_slot_clear)
        build, prototypes = authenticate(result)
        import numpy as np
        if args.stage == 'native':
            native_stage(np, build, prototypes, result, outputs, start)
        elif args.stage in ('cpu', 'mps'):
            torch_stage(np, build, prototypes, result, outputs, start, args.stage)
        else:
            judge_stage(np, args, result, start)
        rehash(result['source_bindings'])
        result.update(guard(args.stage, start))
        result.update(passed=True, completed=True, sources_unchanged=True)
    except BaseException:
        result['error'] = traceback.format_exc()
        try:
            rehash(result['source_bindings'])
            result['sources_unchanged'] = True
        except BaseException:
            result['sources_unchanged'] = False
    finally:
        signal.alarm(0)
    if outputs and np is not None:
        try:
            # All entries are tiny base NumPy diagnostic arrays, NOT deferred FNP work.
            with (folder / 'arrays.npz').open('xb') as stream:
                np.savez(stream, **{key: np.asarray(value) for key, value in outputs.items()})
            result['artifacts']['arrays.npz'] = binding(folder / 'arrays.npz')
            result['array_descriptions'] = {key: dict(shape=list(value.shape), dtype=str(value.dtype),
                bytes_sha256=hashlib.sha256(value.tobytes()).hexdigest()) for key, value in outputs.items()}
        except BaseException:
            result.update(passed=False, completed=False, serialization_error=traceback.format_exc())
    result.update(resources(args.stage, start), ended_utc=datetime.now(timezone.utc).isoformat())
    # Preserve the final measured resource failure, including serialization time.
    try:
        guard(args.stage, start)
        rehash(result['source_bindings'])
    except BaseException:
        result.update(passed=False, completed=False, final_guard_error=traceback.format_exc())
    write(folder / 'result.json', result)
    write(folder / 'completion.json', dict(passed=result['passed'], completed=result['completed'],
        result_sha256=sha(folder / 'result.json')))
    print(json.dumps(dict(stage=args.stage, passed=result['passed'], elapsed_s=result['elapsed_s'],
                         peak_rss_bytes=result['peak_rss_bytes']), sort_keys=True), flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
