"""Inert, one-task CPU gradient/prediction worker; no optimizer or file writes.

initialize(expected_source_sha256=None) is called once in each spawned worker.
evaluate accepts id/rank/mode/payload and a supplied cotangent or target, never
a target path.  Every task owns fresh FP32 P tensors, weights and a card/graph.
Only authenticated modules and their immutable original-payload caches survive.
The coordinator owns admission, task ordering, aggregation, targets and seals.
"""
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import threading
import time
import traceback
import weakref

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
HY = NEW / 'current_d50_hybrid_cp_qualification_v1'
CO = NEW / 'current_bd_p_coadaptation_v1'
DATA = NEW / 'joint_prefix_final_data_v1.py'
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
RSS_LIMIT = 11_000_000_000
VERSIONS = dict(numpy='1.26.4', torch='2.7.0')
ENV_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
            'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
FIXED = {
    HY / 'model.py': 'c8cb5cfe961b2df2253ca660464793308f12769f97a039fcdf0325c151f1b025',
    HY / 'PROTOCOL.md': 'c6281a3fe713798d351684683ae01afe3c144618c5bf3e4db4bff46796d4c9ab',
    CO / 'model.py': '556f3b97613110e638f48127e40112ab203623fa9f6b076b0eab7f07d24658a0',
    CO / 'qualify.py': '1a3df2687ed3bef6ffcfc6f690f3cfe8665ad04aae4da94012c608d4871a7b20',
    CO / 'PROTOCOL_V1.md': 'daac62127665be18a568b90302095c8f83e17bfa85328f87122d39a555bed81c',
    DATA: '22fff8465a13e412c68c1e5c204d7bfa0ce3d7c1cb9b99e1e5378098a9fd8b6b',
    STORE / 'manifest.json': '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50',
}
_HYBRID = _RUNTIME = None
_FAILED = False


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    path = Path(path).resolve()
    require('truthpost' not in path.parts and not
            (path.suffix in ('.npy', '.npz') and 'truth' in path.name.lower()),
            'Worker may not open or hash a target array')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def unchanged(pins):
    for path, expected in pins.items():
        require(sha(path) == expected, 'Authenticated input changed: ' + str(path))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Module specification')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def authenticate():
    """Stdlib-only flat path-to-SHA source/frame/manifest authentication."""
    global _HYBRID
    pins = {str(path.resolve()): digest for path, digest in FIXED.items()}
    unchanged(pins)
    if _HYBRID is None:
        _HYBRID = load_module(HY / 'model.py', 'cpu_pilot_frozen_hybrid_owner')
    for path, digest in _HYBRID.authenticate().items():
        require(path not in pins or pins[path] == digest, 'Conflicting owner source pin')
        pins[path] = digest
    pins[str(Path(__file__).resolve())] = sha(__file__)
    unchanged(pins)
    return pins


def guard(signum=None, frame=None):
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    require(rss < RSS_LIMIT, 'Cooperative worker RSS limit exceeded: ' + str(rss))
    return rss


def initialize(expected_source_sha256=None):
    """One fresh spawned CPU numerical frame; never initializes/calls MPS."""
    global _RUNTIME
    if _RUNTIME is not None:
        require(_RUNTIME['pid'] == os.getpid(), 'Do not fork initialized numerical workers')
        if expected_source_sha256 is not None:
            require(sha(__file__) == expected_source_sha256, 'Reviewed worker source changed')
        require(not _FAILED, 'Failed workers are not reusable')
        return dict(_RUNTIME['metadata'])
    if expected_source_sha256 is not None:
        require(sha(__file__) == expected_source_sha256, 'Reviewed worker source changed')
    require(sys.flags.optimize == 0 and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1',
            'Assertions enabled/no bytecode writes required')
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'nice10 worker required')
    require(all(os.environ.get(key) == '1' for key in ENV_KEYS), 'Explicit BLAS1 environment')
    require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') == '0', 'No MPS fallback')
    require(not any(name in sys.modules for name in
                    ('numpy', 'torch', 'threadpoolctl', 'flopscope', 'whestbench')),
            'Spawn a fresh process before numerical initialization')
    pins = authenticate()
    helper = load_module(CO / 'qualify.py', 'cpu_pilot_frozen_scalar_helper')
    require('numpy' not in sys.modules and 'torch' not in sys.modules, 'Inert source imports required')
    import numpy as np
    import torch
    require(dict(numpy=np.__version__, torch=torch.__version__) == VERSIONS,
            'Qualified NumPy/Torch versions required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(torch.get_num_threads() == torch.get_num_interop_threads() == 1, 'One CPU thread')
    sys.path.insert(0, str(NEW))
    data = load_module(DATA, 'cpu_pilot_frozen_weight_loader')
    require('threadpoolctl' not in sys.modules and 'flopscope' not in sys.modules,
            'Forbidden numerical-frame import')
    unchanged(pins)
    metadata = dict(pid=os.getpid(), versions=dict(VERSIONS), source_bindings=dict(pins),
                    PKEYS=list(_HYBRID.owner().PKEYS), parameter_count=45200,
                    environment={key: os.environ.get(key) for key in ENV_KEYS},
                    moment_device='cpu', carrier_product_device='cpu', mps_used=False,
                    rss_limit_bytes=RSS_LIMIT, peak_rss_bytes=guard())
    _RUNTIME = dict(pid=os.getpid(), np=np, torch=torch, model=_HYBRID, owner=_HYBRID.owner(),
                    helper=helper, data=data, pins=pins, metadata=metadata)
    return dict(metadata)


def array_metadata(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                raw_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())


def payload_metadata(payload):
    """Canonical digest of sorted per-array shape/dtype/C-byte SHA records."""
    arrays = {key: array_metadata(payload[key]) for key in sorted(payload)}
    digest = hashlib.sha256(json.dumps(arrays, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode()).hexdigest()
    return digest, arrays


def weights_for_rank(rank, data):
    """Authenticate manifest, selected row and exact weights before np.load/card."""
    manifest = STORE / 'manifest.json'
    require(sha(manifest) == FIXED[manifest], 'Canonical manifest changed')
    rows = json.loads(manifest.read_text())['roles']['observer_fit']['row_manifests']
    matches = [item for item in rows if item['role_rank'] == rank]
    require(len(matches) == 1, 'Canonical weight row must be unique')
    item = matches[0]
    row_path = (STORE / item['path']).resolve()
    require(row_path.is_relative_to(STORE.resolve()) and sha(row_path) == item['sha256'],
            'Canonical row manifest binding')
    row = json.loads(row_path.read_text())
    path = row_path.parent / 'weights.npy'
    pins = {str(manifest): FIXED[manifest], str(row_path): item['sha256'],
            str(path): row['files']['weights.npy']['sha256']}
    unchanged(pins)
    require(Path(data.weight_path(rank)).resolve() == path, 'Original loader path differs')
    return path, pins


def _evaluate_one(task):
    """All task-local torch objects die on return; only detached copies escape."""
    runtime = _RUNTIME
    np, torch = runtime['np'], runtime['torch']
    model, owner, helper, data = (runtime[key] for key in ('model', 'owner', 'helper', 'data'))
    rank, mode, payload = task['rank'], task['mode'], task['payload']
    require(type(rank) is int and 35 <= rank < 235, 'Fixed Phase2 rank')
    require(mode in ('synthetic', 'train', 'predict'), 'Known task mode')
    if mode == 'synthetic':
        require(rank in (35, 100) and 'target' not in task, 'Fixed target-free qualification ranks')
    if mode == 'train':
        require(rank in data.TRAIN and 'cotangent' not in task, 'TRAIN-only supplied targets')
    if mode == 'predict':
        require('target' not in task and 'cotangent' not in task, 'Prediction mode has no labels/scalar')
    scale = float(task.get('loss_scale', .25 if mode == 'train' else 1.))
    require(scale == (.25 if mode == 'train' else 1.), 'Fixed per-net loss scaling')
    require(isinstance(payload, dict) and all(isinstance(key, str) for key in payload)
            and set(owner.PKEYS).issubset(payload), 'Complete supplied P mapping')
    for key, value in payload.items():
        require(type(value) is np.ndarray and value.dtype == np.float32
                and np.isfinite(value).all(), 'Finite, already-FP32 payload array: ' + key)
    digest, arrays = payload_metadata(payload)
    P = helper.torch_payload(torch, payload, owner.PKEYS, 'cpu', gradients=mode != 'predict')
    require(sum(P[key].numel() for key in owner.PKEYS) == 45200, 'Complete P leaf count')
    require(all(P[key].is_leaf and P[key].device.type == 'cpu'
                and P[key].requires_grad == (mode != 'predict') for key in owner.PKEYS),
            'Fresh complete CPU P leaves')
    weight_path, weight_pins = weights_for_rank(rank, data)
    weights = np.load(weight_path, mmap_mode='r', allow_pickle=False)
    require(weights.shape == (16, 1024, 1024) and weights.dtype == np.float32
            and np.isfinite(weights).all(), 'Exact finite weight shape/dtype')
    card = model.card(rank, P=P, weights=weights, enabled=False)
    guard()
    begin = time.monotonic()
    with torch.set_grad_enabled(mode != 'predict'):
        prediction, _ = card.run(training=mode != 'predict')
    forward_elapsed = time.monotonic() - begin
    owner.check_domain(card)
    forward = card.diagnostics()
    value = helper.prediction_array(np, owner, prediction)
    result = dict(payload_sha256=digest, payload_metadata=arrays, prediction=value,
                  prediction_metadata=array_metadata(value), gradients={}, gradient_metadata={},
                  forward_diagnostics=forward, forward_elapsed_s=forward_elapsed,
                  weight_bindings=weight_pins, loss_scale=scale)
    if mode != 'predict':
        if mode == 'synthetic':
            cotangent = task['cotangent']
            require(type(cotangent) is np.ndarray and cotangent.dtype == np.float32
                    and cotangent.shape == (16, 1024) and np.isfinite(cotangent).all(),
                    'Exact supplied FP32 synthetic cotangent')
            # Literal qualified CPU64 scalar, without an extra identity multiply.
            loss = helper.scalar(torch, owner, prediction, cotangent, torch.float64)
            result['synthetic_loss'] = float(loss.detach())
            result['cotangent_metadata'] = array_metadata(cotangent)
        else:
            target = task['target']
            require(type(target) is np.ndarray and target.dtype == np.float64
                    and target.shape == (16, 1024) and np.isfinite(target).all(),
                    'Supplied finite FP64 training target; no worker target I/O')
            error = owner.served(prediction).double() - torch.tensor(target, dtype=torch.float64, device='cpu')
            final, auxiliary = error[-1].square().mean(), error[:-1].square().mean()
            unscaled_loss = (final + .05 * auxiliary) / 2e-8
            loss = unscaled_loss * scale
            result.update(raw=float(final.detach()), auxiliary_raw=float(auxiliary.detach()),
                          unscaled_loss=float(unscaled_loss.detach()), target_metadata=array_metadata(target))
        result['loss'] = float(loss.detach())
        require(math.isfinite(result['loss']), 'Finite scalar required')
        loss.backward()
        owner.check_domain(card)
        after = owner.check_replay(card)
        require(after['selection_cells'] == forward['selection_cells']
                and after['variance_domain'] == forward['variance_domain'], 'Checkpoint replay changed')
        for key in owner.PKEYS:
            gradient = P[key].grad
            require(gradient is not None and gradient.dtype == torch.float32
                    and bool(torch.isfinite(gradient).all()), 'Missing/nonfinite FP32 gradient: ' + key)
            result['gradients'][key] = gradient.detach().numpy().copy()
            if key in owner.INACTIVE_KEYS:
                require(bool((gradient[0] == 0).all()), 'Inactive layer0 gradient')
        require(sum(value.size for value in result['gradients'].values()) == 45200, 'Complete returned VJP')
        require(all(not value.requires_grad and value.grad is None for key, value in P.items()
                    if key not in owner.PKEYS), 'Frozen fields acquired gradients')
        result['gradient_metadata'] = {key: array_metadata(value) for key, value in result['gradients'].items()}
        result['inactive_gradients_zero'] = True
    else:
        after = card.diagnostics()
    require(after['hybrid_enabled'] is False and after['offloaded_products'] == 0
            and after['transferred_bytes'] == 0 and after['moment_dtype'] == 'float64',
            'CPU-only qualified precision placement')
    require(all(P[key].detach().numpy().tobytes() == payload[key].tobytes() for key in P),
            'Worker may not update the supplied P snapshot')
    result.update(diagnostics=after, forward_backward_elapsed_s=time.monotonic() - begin,
                  parameter_updates=0, mps_used=False, target_files_accessed=False)
    unchanged(weight_pins)
    refs = [('card', weakref.ref(card)), ('prediction', weakref.ref(prediction)),
            ('weight_mapping', weakref.ref(weights))]
    refs.extend(('P/' + key, weakref.ref(value)) for key, value in P.items())
    return result, refs


def evaluate(task):
    """Return detached result/diagnostics, or a failed packet; never save files."""
    global _FAILED
    require(_RUNTIME is not None and _RUNTIME['pid'] == os.getpid(), 'initialize once after spawn')
    require(not _FAILED, 'A failed worker cannot accept another task')
    require(threading.current_thread() is threading.main_thread(), 'Cooperative timer requires main thread')
    require(isinstance(task, dict) and type(task.get('id')) in (str, int), 'Explicit scalar task id')
    start = time.monotonic()
    result = dict(id=task['id'], rank=task.get('rank'), mode=task.get('mode'), pid=os.getpid(),
                  passed=False, completed=False, source_bindings=dict(_RUNTIME['pins']),
                  versions=dict(VERSIONS), task_state_released=False)
    previous = signal.signal(signal.SIGALRM, guard)
    require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'Worker must not replace another timer')
    signal.setitimer(signal.ITIMER_REAL, .5, .5)
    refs = []
    try:
        guard()
        unchanged(_RUNTIME['pins'])
        body, refs = _evaluate_one(task)
        result.update(body)
        gc.collect()
        require(all(reference() is None for _, reference in refs),
                'Task owner retained: ' + repr([name for name, ref in refs if ref() is not None]))
        require('threadpoolctl' not in sys.modules and 'flopscope' not in sys.modules,
                'Forbidden numerical-frame import')
        unchanged(_RUNTIME['pins'])
        guard()
        result.update(task_state_released=True, passed=True, completed=True)
    except BaseException:
        _FAILED = True
        result.update(passed=False, completed=False, error=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        gc.collect()
        result.update(wall_s=time.monotonic() - start,
                      peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        if result['peak_rss_bytes'] >= RSS_LIMIT:
            _FAILED = True
            result.update(passed=False, completed=False, resource_failure='Worker RSS limit')
    return result
