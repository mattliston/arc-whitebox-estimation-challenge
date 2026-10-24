"""Target-free native joint-parameter secants; root guard required.

CLI: NATIVE -B native_probe.py RANK PROTOCOL_SHA
One untouched S33 module/setup per arm. Diagnostic timings cannot qualify a bank.
"""
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

HERE = Path(__file__).resolve().parent
assert len(sys.argv) == 3
rank = int(sys.argv[1])
assert rank in (35, 100)
protocol_path = HERE / 'protocol.json'
protocol_raw = protocol_path.read_bytes()
assert hashlib.sha256(protocol_raw).hexdigest() == sys.argv[2]
protocol = json.loads(protocol_raw)
assert protocol['root_admitted'] is True
assert protocol['fitting_admitted'] is False


def pin(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=path.stat().st_size)


for record in protocol['pins']:
    actual = pin(record['path'])
    assert actual['sha256'] == record['sha256']
    if 'bytes' in record:
        assert actual['bytes'] == record['bytes']
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    assert os.environ.get(name) == '1', name

import numpy as np
import flopscope as flops
from direction_common import EPSILONS, make_directions, perturb
from native_parameters import P2, SOURCE, SOURCE_SHA, load_s33_arrays, install_on_module

out = HERE / ('native_rank' + str(rank))
out.mkdir()
arrays, dials = load_s33_arrays(np)
directions, dial_directions, direction_metadata = make_directions(np, arrays, dials)
assert EPSILONS == (0.01, 0.005)
cases = (('zero', 0.0), ('plus_h', EPSILONS[0]), ('minus_h', -EPSILONS[0]),
         ('plus_half', EPSILONS[1]), ('minus_half', -EPSILONS[1]))
store = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
manifest = json.loads((store / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
weights_path = store / Path(manifest[rank]['path']).parent / 'weights.npy'
weights = np.load(weights_path)
weights_digest = hashlib.sha256(weights.tobytes()).hexdigest()
stub = types.ModuleType('whestbench')
stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub


class MLP:
    def __init__(self):
        self.width, self.depth = 1024, 16
        self.weights = [weights[layer] for layer in range(16)]
        self.seed, self.name = 0, f'phase2-{rank}'


libsystem = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
relief = libsystem.malloc_zone_pressure_relief
relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
relief.restype = ctypes.c_size_t
record = dict(rank=rank, protocol_sha256=sys.argv[2], source=pin(SOURCE),
    script=pin(__file__), helper=pin(HERE / 'native_parameters.py'),
    direction=pin(HERE / 'direction_common.py'), weights=pin(weights_path),
    direction_metadata=direction_metadata, runtime=sys.executable, numpy=np.__version__,
    targets_accessed=False, source_changed=False, timing_qualification_eligible=False)
with (out / 'manifest.json').open('x') as stream:
    json.dump(record, stream, indent=2, allow_nan=False)
    stream.write('\n')
predictions = {}
with (out / 'cases.jsonl').open('x') as log:
    for case, epsilon in cases:
        started = time.monotonic()
        perturbed, perturbed_dials = perturb(np, arrays, dials, directions, dial_directions, epsilon)
        name = 's33_joint_native_' + str(rank) + '_' + case
        mod = types.ModuleType(name)
        mod.__file__ = str(SOURCE)
        sys.modules[name] = mod
        exec(compile(SOURCE.read_text(), str(SOURCE), 'exec'), mod.__dict__)
        mutation = install_on_module(np, mod, perturbed, perturbed_dials)
        est = mod.Estimator()
        setup_start = time.monotonic()
        est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41,
            api_version='2', scratch_dir=None, submission_dir=None, seed=0))
        setup_s = time.monotonic() - setup_start
        setup_log = [list(item) for item in getattr(mod, '_PB_LOG', [])]
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            y = np.asarray(est.predict(MLP(), 2**41), dtype=np.float64)
        assert y.shape == (16, 1024) and np.isfinite(y).all()
        assert hashlib.sha256(weights.tobytes()).hexdigest() == weights_digest
        predictions[case] = y.copy()
        np.save(out / (case + '.npy'), y)
        row = dict(case=case, epsilon=epsilon, rank=rank,
            prediction=pin(out / (case + '.npy')), prediction_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
            flops=int(ctx.summary_dict()['flops_used']), setup_s=setup_s, setup_log=setup_log,
            wall_s=time.monotonic() - started, mutation=mutation,
            targets_accessed=False, timing_qualification_eligible=False)
        log.write(json.dumps(row, allow_nan=False) + '\n')
        log.flush()
        print(json.dumps({key: row[key] for key in ('case', 'epsilon', 'rank', 'flops', 'setup_s', 'wall_s')}), flush=True)
        sys.modules.pop(name)
        del est, mod, ctx, y, perturbed, perturbed_dials
        gc.collect()
        relief(None, 0)
np.savez(out / 'predictions.npz', **predictions)
assert pin(SOURCE)['sha256'] == SOURCE_SHA
for record in protocol['pins']:
    actual = pin(record['path'])
    assert actual['sha256'] == record['sha256']
    if 'bytes' in record:
        assert actual['bytes'] == record['bytes']
with (out / 'completed.json').open('x') as stream:
    json.dump(dict(completed=True, rank=rank, cases=5, predictions=pin(out / 'predictions.npz'),
        source_bytes_unchanged=True, inputs_unchanged=True, targets_accessed=False,
        timing_qualification_eligible=False, external_guard_required=True), stream, indent=2)
    stream.write('\n')
print('NATIVE_JOINT_QUALIFICATION_COMPLETE', flush=True)
