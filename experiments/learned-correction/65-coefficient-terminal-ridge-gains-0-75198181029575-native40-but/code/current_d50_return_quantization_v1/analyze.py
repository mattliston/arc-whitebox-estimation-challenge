"""Analyze sealed L1 return rounding; no new model, gradient or truth calls."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
CB = NEW / 'current_d50_covariance_boundary_v1'
CO = NEW / 'current_bd_p_coadaptation_v1'
RANKS = (35, 100)
CASES = ('null', 'plus025', 'minus025', 'plus050', 'minus050')
ARMS = ('N', 'CPU', 'L', 'R')
RECEIPTS = {
    'native_n_v1': '0cc32573881c8e63e49118460fa57d983653c069e14ff15db915db115769fd59',
    'cpu_v1': '660b35dd426d06ce46a1162cb50c002ee30db4bc629cbeff66b5109e0f7caa36',
    'native_controls_v1': '0ef450da95c54be5c19da33ca826c810d7d81e8a4098ccbd90c9c8f429e07009',
    'judgment_v1': 'db86ec983c629d098a1f0093575fb9c83de924b51600201623bd52c8bcb1e6ed',
}
PINS = {
    CO / 'qualify.py': '1a3df2687ed3bef6ffcfc6f690f3cfe8665ad04aae4da94012c608d4871a7b20',
    CO / 'model.py': '556f3b97613110e638f48127e40112ab203623fa9f6b076b0eab7f07d24658a0',
    CO / 'stage_cpu_forward_v1/result.json': '6f3dc66c43d25b96d6bf5c7620231e115d8527e906ae00e9741bb3e55c5bd607',
    CO / 'stage_cpu_forward_v1/completion.json': '716a7f3e310fea326e7d2ed5313440da88d8eb5d57ce5f6ddf7e99803e26a61c',
    CO / 'stage_cpu_forward_v1/probes_sealed_before_card_calls.json': 'b21b70ee6571ce908a2f37907dc705a23192c78fd5f1e8ab6c859cca94b4e4be',
}
BLAS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def exact_function(path, name, environment):
    nodes = [x for x in ast.parse(Path(path).read_text()).body
             if isinstance(x, ast.FunctionDef) and x.name == name]
    require(len(nodes) == 1, 'Unique original helper: ' + name)
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(path), 'exec'), environment)
    return environment[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--source-sha256')
    parser.add_argument('--protocol-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    require(all(x and len(x) == 64 for x in (args.source_sha256, args.protocol_sha256)),
            'Reviewed source and prospective protocol pins required')
    require(sha(__file__) == args.source_sha256
            and sha(HERE / 'PROTOCOL.md') == args.protocol_sha256, 'Reviewed files changed')
    sys.dont_write_bytecode = True
    require(Path(sys.prefix).resolve() == Path('/Users/matt/anaconda3'), 'Exact AP frame')
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'nice10 required')
    require(all(os.environ.get(k) == '1' for k in BLAS)
            and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
            and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') == '0', 'Fixed resource environment')
    output = HERE / 'analysis_v1'
    output.mkdir()
    started = time.monotonic()
    result = dict(schema='current_d50_return_quantization_v1', passed=False, completed=False,
                  started_utc=datetime.now(timezone.utc).isoformat(), sources_unchanged=False, hashes_unchanged=False,
                  truth_accessed=False, weight_arrays_decoded=False, network_calls=0,
                  gradient_calls=0, parameter_updates=0, score_measured=False,
                  training_admitted=False, original_gates_unchanged=True, MPS_used=False,
                  bindings={}, arrays_decoded=[], measurements=[], endpoint_reconstruction=[])
    bindings = result['bindings']

    def guard(*_):
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != 'darwin':
            peak *= 1024
        result['resources'] = dict(lifetime_peak_rss_bytes=int(peak), elapsed_s=time.monotonic()-started)
        require(peak < 2_000_000_000 and time.monotonic()-started < 120, 'Fixed resource cap')

    def pin(path, expected):
        path = Path(path).resolve()
        require(path.is_relative_to(NEW.parent), 'Unexpected input outside Phase2')
        stat = path.stat()
        value = sha(path)
        require(value == expected and path.stat().st_mtime_ns == stat.st_mtime_ns,
                'Changed artifact: ' + str(path))
        record = dict(sha256=value, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        require(str(path) not in bindings or bindings[str(path)] == record, 'Conflicting pin')
        bindings[str(path)] = record
        guard()

    prior = signal.signal(signal.SIGALRM, guard)
    signal.setitimer(signal.ITIMER_REAL, .5, .5)
    write(output / 'started.json', result)
    try:
        pin(Path(__file__), args.source_sha256)
        pin(HERE / 'PROTOCOL.md', args.protocol_sha256)
        for path, digest in PINS.items():
            pin(path, digest)
        records = {}
        for name, digest in RECEIPTS.items():
            rp, cp = CB/name/'result.json', CB/name/'completion.json'
            pin(rp, digest)
            record, completion = json.loads(rp.read_text()), json.loads(cp.read_text())
            pin(cp, sha(cp))
            require(record['completed'] and completion['completed']
                    and completion['result_sha256'] == digest
                    and completion['passed'] == record['passed']
                    and record['truth_accessed'] is False and record['no_fit']
                    and record['hashes_unchanged'], 'Prior complete target-blind seal')
            require(record['passed'] == (name != 'judgment_v1'), 'Preserve prior scientific failure')
            for group in ('source_bindings', 'input_bindings', 'artifacts'):
                for path, expected in record.get(group, {}).items():
                    pin(path, expected)
            records[name] = record
        probe = json.loads((CO/'stage_cpu_forward_v1/result.json').read_text())
        probe_end = json.loads((CO/'stage_cpu_forward_v1/completion.json').read_text())
        require(probe['passed'] and probe['completed'] and not probe['alltruth_accessed']
                and probe_end['result_sha256'] == PINS[CO/'stage_cpu_forward_v1/result.json'],
                'Original probe source completed')
        cpu, original = records['cpu_v1'], records['judgment_v1']
        require(cpu['cotangents'] == probe['cotangents'] == original['cotangents']
                and cpu['payloads'] == probe['payloads'] == original['payloads'], 'Same sealed probes')
        require(not any(x in sys.modules for x in ('numpy', 'torch', 'flopscope', 'threadpoolctl')),
                'Authenticate before array imports')
        import numpy as np
        import torch
        require(np.__version__ == '1.26.4' and torch.__version__ == '2.7.0', 'Original scalar frame')
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        original_scalar = exact_function(CO/'qualify.py', 'scalar', dict(math=math))
        original_served = exact_function(CO/'model.py', 'served', {})
        model = SimpleNamespace(served=original_served)

        def load(item):
            path = str(Path(item['path']).resolve())
            require(path in bindings and bindings[path]['sha256'] == item['sha256'], 'Presealed array required')
            value = np.load(path, allow_pickle=False)
            require(list(value.shape) == item['shape'] and str(value.dtype) == item['dtype']
                    and hashlib.sha256(value.tobytes()).hexdigest() == item['raw_sha256']
                    and np.isfinite(value).all(), 'Exact finite array bytes/metadata')
            result['arrays_decoded'].append(path)
            return value

        def scalar(row, cot, served):
            padded = torch.zeros((16, 1024), dtype=torch.float32 if served else torch.float64)
            padded[1] = torch.from_numpy(row)
            if served:
                answer = original_scalar(torch, model, padded, cot, torch.float64)
            else:
                y, q = padded, torch.as_tensor(cot, dtype=torch.float64)
                answer = (y[-1]*q[-1]).sum()/math.sqrt(1024) + .05*(y[:-1]*q[:-1]).sum()/math.sqrt(15*1024)
            return float(answer)

        def norm(value):
            return float(np.linalg.norm(value))

        for rank in RANKS:
            rk = str(rank)
            cot = load(cpu['cotangents'][rk])
            require(cot.shape == (16, 1024) and cot.dtype == np.float32, 'Original cotangent shape')
            entries = {'N': records['native_n_v1']['ranks'][rk]['outputs'],
                       'CPU': cpu['ranks'][rk]['outputs']}
            for arm in ('L', 'R'):
                entries[arm] = records['native_controls_v1']['ranks'][rk]['circuits'][arm]['outputs']
            endpoints = {}
            for arm in ARMS:
                require(set(entries[arm]) == set(CASES), 'Fixed five cases only')
                endpoints[arm] = {}
                for case in CASES:
                    item = entries[arm][case]
                    y, host, delta = (load(item[k]) for k in ('prediction', 'host1', 'delta1'))
                    require(y.shape == (2, 1024) and y.dtype == np.float32 and (y >= 0).all(), 'Served rows')
                    require(host.shape == delta.shape == (1024,) and host.dtype == np.float64
                            and delta.dtype == (np.float64 if arm == 'CPU' else np.float32), 'Owned host/delta dtypes')
                    z = np.maximum(host+delta.astype(np.float64), 0.)
                    require(z.astype(np.float32).tobytes() == y[1].tobytes(), 'Reconstructed served row is byte exact')
                    sy, sz = scalar(y[1], cot, True), scalar(z, cot, False)
                    require(sy == scalar(y[1].astype(np.float64), cot, False)
                            == original['ranks'][rk]['individual_original_scalars'][arm][case],
                            'Exact original scalar and linear reduction')
                    if case == 'null':
                        null_host, null_row0 = host.tobytes(), y[0].tobytes()
                    require(host.tobytes() == null_host and y[0].tobytes() == null_row0,
                            'P-independent host and first row')
                    endpoints[arm][case] = dict(y=y[1].astype(np.float64), z=z, sy=sy, sz=sz)
                    result['endpoint_reconstruction'].append(dict(rank=rank, arm=arm, case=case,
                        served_byte_exact=True, original_scalar_exact=True, served_scalar=sy,
                        pre_return_scalar=sz, rounding_scalar=sy-sz, clipping_count=int((z == 0).sum())))
            for suffix, h in (('025', .025), ('050', .05)):
                saved = [x for x in original['measurements'] if x['rank'] == rank and x['amplitude'] == h]
                require(len(saved) == 1, 'Original measurement roster')
                responses = {}
                for arm in ARMS:
                    p, m = endpoints[arm]['plus'+suffix], endpoints[arm]['minus'+suffix]
                    ry, rz = (p['y']-m['y'])/(2*h), (p['z']-m['z'])/(2*h)
                    sy, sz = (p['sy']-m['sy'])/(2*h), (p['sz']-m['sz'])/(2*h)
                    require(sy == saved[0]['original_individual_scalar_secants'][arm], 'Original secant exact')
                    responses[arm] = dict(y=ry, z=rz, q=ry-rz, sy=sy, sz=sz, sq=sy-sz)
                comparisons = {}
                for other in ('CPU', 'L', 'R'):
                    n, a = responses['N'], responses[other]
                    dy, dz, dq = n['y']-a['y'], n['z']-a['z'], n['q']-a['q']
                    sy, sz, sq = n['sy']-a['sy'], n['sz']-a['sz'], n['sq']-a['sq']
                    vr, sr = norm(dy-dz-dq), abs(sy-sz-sq)
                    require(vr <= 1e-12 and sr <= 1e-12, 'Additive decomposition identity')
                    nontrivial = norm(dy) > 1e-10 and abs(sy) > 1e-10
                    ratios = (norm(dz)/norm(dy), abs(sz)/abs(sy)) if nontrivial else (None, None)
                    comparisons[other] = dict(served_discrepancy_l2=norm(dy), pre_return_discrepancy_l2=norm(dz),
                        rounding_discrepancy_l2=norm(dq), served_scalar=sy, pre_return_scalar=sz,
                        rounding_scalar=sq, vector_identity_residual=vr, scalar_identity_residual=sr,
                        pre_return_norm_ratio=ratios[0], pre_return_absolute_scalar_ratio=ratios[1],
                        rounding_signed_scalar_fraction=sq/sy if nontrivial else None,
                        rounding_projection_fraction=float(np.dot(dy,dq)/np.dot(dy,dy)) if nontrivial else None,
                        nontrivial=nontrivial, local_dominance=nontrivial and ratios[0] <= .25 and ratios[1] <= .25)
                require(comparisons['CPU']['served_scalar'] == saved[0]['L']['scalar_discrepancy'],
                        'Original primary discrepancy exact')
                result['measurements'].append(dict(rank=rank, amplitude=h, comparisons=comparisons,
                    arm_scalar_responses={a: {k: v[k] for k in ('sy','sz','sq')} for a,v in responses.items()}))
                print(json.dumps(dict(rank=rank, amplitude=h, primary=comparisons['CPU'])), flush=True)
            guard()
        require(len(result['measurements']) == 4 and len(result['endpoint_reconstruction']) == 40,
                'All fixed comparisons completed')
        require(not any(x in sys.modules for x in ('flopscope','threadpoolctl')), 'No native or threadpool work')
        for path, item in list(bindings.items()):
            pin(path, item['sha256'])
        result.update(completed=True, passed=True, sources_unchanged=True, hashes_unchanged=True,
                      return_rounding_locally_dominant_all_four=all(x['comparisons']['CPU']['local_dominance']
                                                                  for x in result['measurements']),
                      whole_card_attribution_established=False, new_derivative_qualified=False)
        guard()
    except BaseException:
        result.update(completed=False, passed=False, error=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior)
        result.update(elapsed_s=time.monotonic()-started, finished_utc=datetime.now(timezone.utc).isoformat())
        write(output/'result.json', result)
        write(output/'completion.json', dict(passed=result['passed'], completed=result['completed'],
                                             result_sha256=sha(output/'result.json')))
        print(json.dumps({k:result.get(k) for k in ('passed','completed','elapsed_s','error',
             'return_rounding_locally_dominant_all_four')}), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
