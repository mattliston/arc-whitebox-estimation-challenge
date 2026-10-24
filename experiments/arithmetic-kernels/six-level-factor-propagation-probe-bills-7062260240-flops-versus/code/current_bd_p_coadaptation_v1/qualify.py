"""Fixed target-free current-d50 P qualification; inert unless --run.

Sequential CPU/MPS forwards, CPU/MPS full P VJPs, then a no-Torch judge.
No optimizer, targets, fidelity-threshold search, current-state replacement,
native candidate, or score. Every stage creates one exclusive directory.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
TRUTH = P2 / 'public_g22_operator_features_v1/truthpost'
STAGES = ('cpu_forward', 'mps_forward', 'cpu_gradient', 'mps_gradient', 'judge')
RANKS = (35, 100)
CASES = (('null', 0.), ('plus025', .025), ('minus025', -.025),
         ('plus050', .05), ('minus050', -.05))
SCHEMA = 'current_d50_p_coadaptation_qualification_v1'
STORE_SHA = '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50'
WEIGHT_SHA = {35: '14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2',
              100: 'c4923171c8ddef28396aabd70335dad29141361efc13865ccb660a4a57421ee5'}
VERSIONS = {'numpy': '1.26.4', 'torch': '2.7.0'}
ANNEX_SHA = '5dc8e433e13fb1b7cec20dbd61e784e8f7d9a50dd89d4e6f03225f73e378b92d'
MEAN_LIMIT, GRAD_LIMIT = 1e-5, .05


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    path = Path(path).resolve()
    require(not path.is_relative_to(TRUTH.resolve()), 'Qualification may not open/hash truths')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def bind(path, expected=None):
    path = Path(path).resolve()
    before = path.stat()
    value = sha(path)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            'Input changed during binding: ' + str(path))
    require(expected is None or value == expected, 'Input SHA mismatch: ' + str(path))
    return dict(path=str(path), sha256=value, bytes=after.st_size, mtime_ns=after.st_mtime_ns)


def recheck(bindings):
    for item in bindings.values():
        require(bind(item['path']) == item, 'Bound file changed: ' + item['path'])


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Module specification')
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def array_meta(array):
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                raw_sha256=hashlib.sha256(array.tobytes()).hexdigest())


def save_array(np, path, value, result):
    require(np.isfinite(value).all(), 'Nonfinite saved array')
    with path.open('xb') as stream:
        np.save(stream, value, allow_pickle=False)
    item = bind(path)
    result['artifacts'][str(path)] = item['sha256']
    result['artifact_bindings'][str(path)] = item
    return dict(item, **array_meta(value))


def save_bundle(np, path, values, result):
    require(all(np.isfinite(x).all() for x in values.values()), 'Nonfinite saved bundle')
    with path.open('xb') as stream:
        np.savez(stream, **values)
    item = bind(path)
    result['artifacts'][str(path)] = item['sha256']
    result['artifact_bindings'][str(path)] = item
    return dict(item, arrays={key: array_meta(value) for key, value in values.items()})


def load_array(np, item):
    bind(item['path'], item['sha256'])
    value = np.load(item['path'], allow_pickle=False)
    require(array_meta(value) == {key: item[key] for key in ('shape', 'dtype', 'raw_sha256')}
            and np.isfinite(value).all(), 'Array bytes/metadata mismatch')
    return value


def load_bundle(np, item):
    bind(item['path'], item['sha256'])
    with np.load(item['path'], allow_pickle=False) as source:
        require(set(source.files) == set(item['arrays']), 'Bundle keys mismatch')
        values = {key: source[key].copy() for key in source.files}
    require(all(array_meta(value) == item['arrays'][key] and np.isfinite(value).all()
                for key, value in values.items()), 'Bundle bytes/metadata mismatch')
    return values


def swap_state():
    raw = subprocess.run(['/usr/sbin/sysctl', 'vm.swapusage'], capture_output=True,
                         text=True, check=True, timeout=3).stdout.strip()
    match = re.search(r'used\s*=\s*([\d.]+)([KMGT])', raw)
    require(match is not None, 'Unrecognized swap metadata')
    return dict(raw=raw, used_bytes=float(match[1]) * 1024 ** ('KMGT'.index(match[2]) + 1))


def weight_bindings():
    records = {str(STORE / 'manifest.json'): bind(STORE / 'manifest.json', STORE_SHA)}
    rows = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
    paths = {}
    for rank in RANKS:
        matches = [x for x in rows if x['role_rank'] == rank]
        require(len(matches) == 1, 'Unique canonical observer-fit row')
        rp = (STORE / matches[0]['path']).resolve()
        require(rp.is_relative_to(STORE.resolve()), 'Canonical row escapes store')
        records[str(rp)] = bind(rp, matches[0]['sha256'])
        row = json.loads(rp.read_text())
        require(row['role_rank'] == rank and row['role'] == 'observer_fit', 'Weight frame mismatch')
        wp = rp.parent / 'weights.npy'
        require(row['files']['weights.npy']['sha256'] == WEIGHT_SHA[rank], 'Weight manifest pin')
        records[str(wp)] = bind(wp, WEIGHT_SHA[rank])
        paths[rank] = wp
    return paths, records


def sealed_stage(stage, sources, dependencies):
    folder = HERE / ('stage_' + stage + '_v1')
    rp, cp = folder / 'result.json', folder / 'completion.json'
    rb, cb = bind(rp), bind(cp)
    result, completion = json.loads(rp.read_text()), json.loads(cp.read_text())
    require(result['schema'] == SCHEMA and result['stage'] == stage
            and result['passed'] and result['completed'] and result['sources_unchanged']
            and completion['passed'] and completion['completed']
            and completion['stage'] == stage and completion['result_sha256'] == rb['sha256'],
            'Unqualified preceding stage: ' + stage)
    require(result['source_bindings'] == sources and result['alltruth_accessed'] is False
            and result['training_targets_accessed'] is False
            and result['validation_targets_accessed'] is False
            and result['no_fit'] and result['parameter_updates'] == 0, 'Preceding stage lineage/scope')
    require(result['versions'] == VERSIONS, 'Preceding numerical library versions')
    require(result['environment']['versions'] == result['versions'], 'Preceding imported library frame')
    require(result['resources']['peak_rss_bytes'] <= 12_000_000_000
            and result['maximum_sampled_mps_driver_bytes'] <= 16_000_000_000
            and result['elapsed_s'] < 900
            and result['swap_growth_bytes'] <= 2 * 1024 ** 3, 'Preceding stage resource qualification')
    for group in ('source_metadata', 'weight_bindings', 'artifact_bindings'):
        recheck(result[group])
        for path, item in result[group].items():
            if group == 'artifact_bindings':
                require(Path(path).is_relative_to(folder), 'Artifact escapes its fixed stage')
                require(result['artifacts'][path] == item['sha256'], 'Artifact hash interface')
            dependencies[path] = item
    dependencies[str(rp)], dependencies[str(cp)] = rb, cb
    return result, rb['sha256']


def export_probes(np, model, folder, result):
    original = model.payload_numpy()
    rng = np.random.Generator(np.random.PCG64DXSM(9309097))
    directions, scales = {}, {}
    for key in model.PKEYS:
        z = rng.standard_normal(original[key].shape)
        if key in model.INACTIVE_KEYS:
            z[0] = 0.
        rms = float(np.sqrt(np.mean(z * z)))
        require(rms > 0 and math.isfinite(rms), 'Nonzero deterministic direction group')
        scale = max(float(np.sqrt(np.mean(original[key].astype(np.float64) ** 2))), .001)
        directions[key], scales[key] = z * (scale / rms), scale
    result['direction'] = save_bundle(np, folder / 'direction64.npz', directions, result)
    result['direction_scales'] = scales
    result['payloads'] = {}
    for name, amplitude in CASES:
        values = {key: value.astype(np.float32).copy() for key, value in original.items()}
        if amplitude:
            for key in model.PKEYS:
                values[key] = (original[key].astype(np.float64) + amplitude * directions[key]).astype(np.float32)
        for key in model.INACTIVE_KEYS:
            require(values[key][0].tobytes() == original[key][0].tobytes(), 'Inactive row moved')
        result['payloads'][name] = save_bundle(np, folder / ('payload_' + name + '.npz'), values, result)
    result['cotangents'] = {}
    for rank in RANKS:
        rng = np.random.Generator(np.random.PCG64DXSM(9309098 + rank))
        value = rng.standard_normal((16, 1024)).astype(np.float32)
        result['cotangents'][str(rank)] = save_array(np, folder / f'cotangent_{rank:04d}.npy', value, result)
    seal = folder / 'probes_sealed_before_card_calls.json'
    write(seal, dict(payloads=result['payloads'], direction=result['direction'],
                     direction_scales=scales, cotangents=result['cotangents'],
                     frozen_cstar_sha256=result['frozen_cstar_sha256'], card_calls=0,
                     alltruth_accessed=False, parameter_updates=0))
    result['probe_seal_sha256'] = sha(seal)
    result['artifacts'][str(seal)] = sha(seal)
    result['artifact_bindings'][str(seal)] = bind(seal)


def torch_payload(torch, values, keys, device, gradients=False):
    return {key: torch.tensor(value, dtype=torch.float32, device=device,
                              requires_grad=bool(gradients and key in keys))
            for key, value in values.items()}


def scalar(torch, model, prediction, cotangent, dtype):
    # The served operands are identical FP32 data; reductions use moment dtype.
    y = model.served(prediction).to(dtype)
    q = torch.as_tensor(cotangent, device=y.device, dtype=dtype)
    return (y[-1] * q[-1]).sum() / math.sqrt(1024) + .05 * (y[:-1] * q[:-1]).sum() / math.sqrt(15 * 1024)


def prediction_array(np, model, prediction):
    value = model.served(prediction).detach().cpu().numpy().copy()
    require(value.shape == (16, 1024) and value.dtype == np.float32
            and np.isfinite(value).all() and (value >= 0).all(), 'All16 served prediction contract')
    return value


def selector_difference(left, right):
    require(set(left) == set(right), 'Selector cell roster mismatch')
    changed_membership = [key for key in left if left[key]['membership'] != right[key]['membership']]
    changed_order = [key for key in left if left[key]['ordered_ids'] != right[key]['ordered_ids']]
    return dict(membership_changed_cells=changed_membership, ordering_changed_cells=changed_order)


def release(torch, device):
    gc.collect()
    if device == 'mps':
        torch.mps.synchronize()
        torch.mps.empty_cache()


def forwards(np, torch, model, stage, paths, folder, result, previous, record, guard):
    device = stage.split('_')[0]
    probes = result if stage == 'cpu_forward' else previous['cpu_forward']
    result['probe_owner_stage'] = 'cpu_forward'
    if stage != 'cpu_forward':
        for key in ('payloads', 'cotangents', 'direction', 'direction_scales', 'probe_seal_sha256'):
            result[key] = probes[key]
    result['outputs'], result['entries'], result['secants'], result['parent_outputs'] = {}, [], [], {}
    for rank in RANKS:
        weights = np.load(paths[rank], mmap_mode='r', allow_pickle=False)
        require(weights.shape == (16, 1024, 1024) and weights.dtype == np.float32, 'Weights shape/dtype')
        cot = load_array(np, probes['cotangents'][str(rank)])
        parent = model.parent_card(rank, device=device, weights=weights)
        with torch.no_grad():
            pred, _ = parent.run()
            parent_value = prediction_array(np, model, pred)
        result['parent_outputs'][str(rank)] = save_array(np, folder / f'parent_{rank:04d}.npy', parent_value, result)
        parent_memberships = [sorted(map(int, pair[0])) for pair in parent.selections]
        require(len(parent_memberships) == 47, 'Unmodified parent selector roster')
        del parent, pred
        release(torch, device)
        result['outputs'][str(rank)] = {}
        losses = {}
        for name, amplitude in CASES:
            values = load_bundle(np, probes['payloads'][name])
            P = torch_payload(torch, values, model.PKEYS, device)
            card = model.PCard(rank, P=P, device=device, weights=weights, audit_selections=True)
            with torch.no_grad():
                pred, _ = card.run()
                loss = float(scalar(torch, model, pred, cot, card.dtype).detach().cpu())
                value = prediction_array(np, model, pred)
            require(math.isfinite(loss), 'Finite synthetic loss')
            diagnostics = model.check_replay(card)
            item = save_array(np, folder / f'row_{rank:04d}_{name}.npy', value, result)
            entry = dict(rank=rank, case=name, amplitude=amplitude, prediction=item,
                         synthetic_loss=loss, diagnostics=diagnostics, alltruth_accessed=False)
            if name == 'null':
                entry['parent_byte_exact'] = value.tobytes() == parent_value.tobytes()
                cells = diagnostics['selection_cells']
                own_memberships = [cells[f'{l}:{s}']['membership'] for l in range(16)
                                   for s in range(2 if l == 15 else 3)]
                entry['parent_selector_membership_equal'] = own_memberships == parent_memberships
            if stage == 'mps_forward':
                cpu = previous['cpu_forward']['outputs'][str(rank)][name]
                cpu_value = load_array(np, cpu['prediction'])
                entry['cpu_maximum_absolute_difference'] = float(np.max(np.abs(value.astype(np.float64) - cpu_value)))
                entry['cpu_selector_comparison'] = selector_difference(
                    cpu['diagnostics']['selection_cells'], diagnostics['selection_cells'])
            result['entries'].append(entry)
            result['outputs'][str(rank)][name] = entry
            losses[name] = loss
            record(kind='forward', rank=rank, case=name, synthetic_loss=loss,
                   parent_byte_exact=entry.get('parent_byte_exact'),
                   cpu_maximum_absolute_difference=entry.get('cpu_maximum_absolute_difference'))
            # Preserve this call's output and attempted metrics before fail-stop.
            require(name != 'null' or (entry['parent_byte_exact'] and entry['parent_selector_membership_equal']),
                    'Null differs from unmodified frozen BD parent')
            require(stage != 'mps_forward' or entry['cpu_maximum_absolute_difference'] <= MEAN_LIMIT,
                    'CPU/MPS all16 forward gate')
            guard()
            del pred, value, card, P, values
            release(torch, device)
        for suffix, amplitude in (('025', .025), ('050', .05)):
            secant = (losses['plus' + suffix] - losses['minus' + suffix]) / (2 * amplitude)
            item = dict(rank=rank, amplitude=amplitude, symmetric_secant=secant)
            if stage == 'mps_forward':
                cpu = next(x for x in previous['cpu_forward']['secants']
                           if x['rank'] == rank and x['amplitude'] == amplitude)['symmetric_secant']
                item['cpu_secant'] = cpu
                item['relative_backend_error'] = abs(cpu - secant) / max(abs(cpu), abs(secant), 1e-10)
            result['secants'].append(item)
            record(kind='secant', **item)
            require(math.isfinite(secant), 'Finite secant')
            require(stage != 'mps_forward' or item['relative_backend_error'] <= GRAD_LIMIT,
                    'Actual-FP32 finite-direction backend gate')
        del weights, cot, parent_value
        release(torch, device)


def compare_gradients(np, cpu, mps, model):
    a, b = load_bundle(np, cpu['gradient']), load_bundle(np, mps['gradient'])
    require(set(a) == set(b) == set(model.PKEYS), 'Full P gradient roster')
    norms_a = {k: float(np.linalg.norm(a[k].astype(np.float64).ravel())) for k in model.PKEYS}
    norms_b = {k: float(np.linalg.norm(b[k].astype(np.float64).ravel())) for k in model.PKEYS}
    errors = {k: float(np.linalg.norm(a[k].astype(np.float64).ravel() - b[k].astype(np.float64).ravel()))
              for k in model.PKEYS}
    na = math.sqrt(sum(x * x for x in norms_a.values()))
    nb = math.sqrt(sum(x * x for x in norms_b.values()))
    err = math.sqrt(sum(x * x for x in errors.values()))
    require(na > 0 and all(math.isfinite(x) for x in (na, nb, err)), 'Nonzero finite full CPU VJP')
    groups = {}
    for key in model.PKEYS:
        large = norms_a[key] >= 1e-6 * na
        relative = errors[key] / max(norms_a[key], norms_b[key], 1e-300)
        passed = relative <= GRAD_LIMIT if large else errors[key] <= 5e-8 * na
        groups[key] = dict(cpu_norm=norms_a[key], mps_norm=norms_b[key], error_norm=errors[key],
                           substantial_cpu_group=large, relative_error=relative, passed=passed)
    return dict(cpu_norm=na, mps_norm=nb, error_norm=err, relative_error=err / max(na, nb),
                groups=groups, passed=err / max(na, nb) <= GRAD_LIMIT
                and all(x['passed'] for x in groups.values()))


def gradients(np, torch, model, stage, paths, folder, result, previous, record, guard):
    device = stage.split('_')[0]
    probes = previous['cpu_forward']
    result['gradients'], result['gradient_comparisons'] = {}, {}
    result['probe_owner_stage'] = 'cpu_forward'
    for rank in RANKS:
        weights = np.load(paths[rank], mmap_mode='r', allow_pickle=False)
        require(weights.shape == (16, 1024, 1024) and weights.dtype == np.float32, 'Weights shape/dtype')
        values = load_bundle(np, probes['payloads']['null'])
        P = torch_payload(torch, values, model.PKEYS, device, gradients=True)
        require(all(P[k].is_leaf and P[k].requires_grad for k in model.PKEYS), 'Full P leaves')
        card = model.PCard(rank, P=P, device=device, weights=weights, audit_selections=True)
        cot = load_array(np, probes['cotangents'][str(rank)])
        prediction, _ = card.run(training=True)
        model.check_domain(card)
        before = card.diagnostics()
        value = prediction_array(np, model, prediction)
        reference = load_array(np, previous[device + '_forward']['outputs'][str(rank)]['null']['prediction'])
        item = dict(rank=rank, prediction=save_array(np, folder / f'vjp_primal_{rank:04d}.npy', value, result),
                    null_forward_byte_exact=value.tobytes() == reference.tobytes(),
                    forward_diagnostics=before, alltruth_accessed=False)
        result['gradients'][str(rank)] = item
        record(kind='gradient_primal', rank=rank, null_forward_byte_exact=item['null_forward_byte_exact'])
        require(item['null_forward_byte_exact'], 'Checkpointed null primal changed')
        loss = scalar(torch, model, prediction, cot, card.dtype)
        item['synthetic_loss'] = float(loss.detach().cpu())
        require(math.isfinite(item['synthetic_loss']), 'Finite synthetic VJP scalar')
        loss.backward()
        diagnostics = model.check_replay(card)
        require(diagnostics['selection_cells'] == before['selection_cells']
                and diagnostics['variance_domain'] == before['variance_domain'], 'Backward replay consistency')
        grads, inactive = {}, {}
        for key in model.PKEYS:
            require(P[key].grad is not None and bool(torch.isfinite(P[key].grad).all().detach().cpu()),
                    'Missing/nonfinite P VJP: ' + key)
            grads[key] = P[key].grad.detach().cpu().numpy().copy()
            if key in model.INACTIVE_KEYS:
                inactive[key] = bool((grads[key][0] == 0).all())
                require(inactive[key], 'Inactive layer0 gradient must be zero')
        require(sum(x.size for x in grads.values()) == model.PARAMETER_COUNT, 'Complete 45200-coordinate VJP')
        require(all(not value.requires_grad and value.grad is None for key, value in P.items()
                    if key not in model.PKEYS), 'Frozen normalizers must not acquire gradients')
        item.update(gradient=save_bundle(np, folder / f'gradient_{rank:04d}.npz', grads, result),
                    diagnostics=diagnostics, inactive_gradients_zero=inactive)
        if stage == 'mps_gradient':
            comparison = compare_gradients(np, previous['cpu_gradient']['gradients'][str(rank)], item, model)
            result['gradient_comparisons'][str(rank)] = comparison
            record(kind='full_vjp_comparison', rank=rank, **comparison)
            require(comparison['passed'], 'Full/group current-P VJP gate')
        else:
            record(kind='cpu_full_vjp_saved', rank=rank, coordinate_count=model.PARAMETER_COUNT)
        guard()
        del loss, prediction, value, reference, card, P, values, grads, weights, cot
        release(torch, device)


def judge(np, model, previous, result):
    cpu, mps = previous['cpu_forward'], previous['mps_forward']
    require(cpu['payloads'] == mps['payloads'] and cpu['cotangents'] == mps['cotangents']
            and cpu['probe_seal_sha256'] == mps['probe_seal_sha256'], 'Shared actual-FP32 probes')
    result['forward_comparisons'], result['secant_comparisons'], result['gradient_comparisons'] = [], [], {}
    for rank in RANKS:
        for case, _ in CASES:
            a, b = cpu['outputs'][str(rank)][case], mps['outputs'][str(rank)][case]
            av, bv = load_array(np, a['prediction']), load_array(np, b['prediction'])
            difference = float(np.max(np.abs(av.astype(np.float64) - bv)))
            result['forward_comparisons'].append(dict(rank=rank, case=case, maximum_absolute_difference=difference,
                selectors=selector_difference(a['diagnostics']['selection_cells'], b['diagnostics']['selection_cells'])))
            require(difference <= MEAN_LIMIT, 'Judge all16 backend gate')
            if case == 'null':
                for rec in (cpu, mps):
                    own = load_array(np, rec['outputs'][str(rank)][case]['prediction'])
                    parent = load_array(np, rec['parent_outputs'][str(rank)])
                    require(own.tobytes() == parent.tobytes(), 'Judge parent-null bytes')
        for amplitude in (.025, .05):
            a = next(x for x in cpu['secants'] if x['rank'] == rank and x['amplitude'] == amplitude)['symmetric_secant']
            b = next(x for x in mps['secants'] if x['rank'] == rank and x['amplitude'] == amplitude)['symmetric_secant']
            err = abs(a - b) / max(abs(a), abs(b), 1e-10)
            result['secant_comparisons'].append(dict(rank=rank, amplitude=amplitude, cpu=a, mps=b, relative_error=err))
            require(err <= GRAD_LIMIT, 'Judge actual-FP32 secant gate')
        comparison = compare_gradients(np, previous['cpu_gradient']['gradients'][str(rank)],
                                       previous['mps_gradient']['gradients'][str(rank)], model)
        result['gradient_comparisons'][str(rank)] = comparison
        require(comparison['passed'], 'Judge complete/group P VJP gate')
    result.update(numerical_qualification_passed=True, full_P_VJP=True, frozen_cstar=True,
                  native_candidate_qualified=False, score_measured=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=STAGES)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--source-sha256')
    parser.add_argument('--model-sha256')
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--mps-slot-clear', action='store_true')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    if not args.stage or any(not x or len(x) != 64 for x in
                            (args.source_sha256, args.model_sha256, args.protocol_sha256)):
        parser.error('--run requires stage and all three reviewed source hashes')
    stage = args.stage
    folder = HERE / ('stage_' + stage + '_v1')
    folder.mkdir(exist_ok=False)
    start = time.monotonic()
    result = dict(schema=SCHEMA, stage=stage, completed=False, passed=False,
        alltruth_accessed=False, training_targets_accessed=False, validation_targets_accessed=False,
        no_fit=True, parameter_updates=0, no_native_candidate=True, source_bindings={}, source_metadata={},
        weight_bindings={}, artifacts={}, artifact_bindings={}, prerequisite_seals={}, sources_unchanged=False,
        maximum_sampled_mps_driver_bytes=0, requested_monitor_period_s=.5,
        resources={}, versions={}, started_utc=datetime.now(timezone.utc).isoformat())
    write(folder / 'started.json', result)
    progress = (folder / 'progress.jsonl').open('x')
    torch, monitoring, dependencies = None, False, {}

    def record(**item):
        row = dict(elapsed_s=time.monotonic() - start, **item)
        line = json.dumps(row, allow_nan=False, sort_keys=True)
        progress.write(line + '\n'); progress.flush(); print(line, flush=True)

    def guard(_s=None, _f=None):
        nonlocal monitoring
        if monitoring:
            return
        monitoring = True
        try:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss = int(rss if sys.platform == 'darwin' else rss * 1024)
            driver = int(torch.mps.driver_allocated_memory()) if torch is not None and stage.startswith('mps') else 0
            result['resources'] = dict(peak_rss_bytes=rss, mps_driver_bytes=driver, elapsed_s=time.monotonic() - start)
            result['maximum_sampled_mps_driver_bytes'] = max(result['maximum_sampled_mps_driver_bytes'], driver)
            require(rss <= 12_000_000_000 and driver <= 16_000_000_000, 'Host/driver resource cap')
            require(time.monotonic() - start < 900, '900s fixed stage deadline')
        except BaseException:
            signal.setitimer(signal.ITIMER_REAL, 0)
            raise
        finally:
            monitoring = False

    old_handler = signal.signal(signal.SIGALRM, guard)
    signal.setitimer(signal.ITIMER_REAL, .5, .5)
    try:
        require(Path(sys.prefix).resolve() == Path('/Users/matt/anaconda3'), 'Qualified AP interpreter required')
        require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'nice10 required')
        require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'No bytecode writes')
        require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')), 'Explicit BLAS1 environment')
        require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') == '0', 'No MPS fallback')
        require(not any(k in sys.modules for k in ('numpy', 'torch', 'threadpoolctl', 'flopscope')), 'Fresh numerical process')
        require(not stage.startswith('mps') or args.mps_slot_clear, 'Sole MPS slot attestation required')
        pins = {Path(__file__).resolve(): args.source_sha256, HERE / 'model.py': args.model_sha256,
                HERE / 'PROTOCOL_V1.md': args.protocol_sha256}
        for path, digest in pins.items():
            item = bind(path, digest)
            result['source_metadata'][str(path)] = item
        result['source_metadata'][str(HERE / 'implementation.md')] = bind(HERE / 'implementation.md', ANNEX_SHA)
        model = module(HERE / 'model.py', 'current_d50_p_qualification_model')
        for path, digest in model.authenticate().items():
            item = bind(path, digest)
            if path in result['source_metadata']:
                require(result['source_metadata'][path] == item, 'Conflicting model source pin')
            result['source_metadata'][path] = item
        result['source_bindings'] = {path: item['sha256'] for path, item in result['source_metadata'].items()}
        require('numpy' not in sys.modules and 'torch' not in sys.modules, 'Source admission must remain inert')
        result['frozen_cstar_sha256'] = hashlib.sha256(model.coefficient_bytes(model.frozen_coefficients())).hexdigest()
        prior_count = STAGES.index(stage)
        previous = {}
        for prior in STAGES[:prior_count]:
            previous[prior], digest = sealed_stage(prior, result['source_bindings'], dependencies)
            result['prerequisite_seals'][prior] = digest
            require(previous[prior]['frozen_cstar_sha256'] == result['frozen_cstar_sha256'], 'Same frozen cstar')
        paths, result['weight_bindings'] = weight_bindings()
        result['before_swap'] = swap_state()
        result['versions'] = {key: importlib.metadata.version(key) for key in VERSIONS}
        require(result['versions'] == VERSIONS, 'Qualified NumPy/Torch versions changed')
        result['environment'] = dict(executable=sys.executable, sys_prefix=sys.prefix, pid=os.getpid(),
            nice=os.getpriority(os.PRIO_PROCESS, 0), mps_slot_clear_attested=args.mps_slot_clear,
            versions=dict(result['versions']),
            scalar_reduction='CPU float64 / MPS float32; shared served FP32 operands')
        import numpy as np
        require(np.__version__ == result['versions']['numpy'], 'Imported NumPy version')
        if stage != 'judge':
            import torch as t
            torch = t
            require(torch.__version__ == result['versions']['torch'], 'Imported Torch version')
            torch.set_num_threads(1); torch.set_num_interop_threads(1)
            require(not stage.startswith('mps') or torch.backends.mps.is_available(), 'MPS unavailable')
        if stage == 'cpu_forward':
            export_probes(np, model, folder, result)
        if stage.endswith('forward'):
            forwards(np, torch, model, stage, paths, folder, result, previous, record, guard)
        elif stage.endswith('gradient'):
            gradients(np, torch, model, stage, paths, folder, result, previous, record, guard)
        else:
            judge(np, model, previous, result)
            require('torch' not in sys.modules, 'Judge may not import Torch')
        require('threadpoolctl' not in sys.modules and 'flopscope' not in sys.modules, 'Forbidden AP numerical runtime')
        for imported in tuple(sys.modules.values()):
            path = getattr(imported, '__file__', None)
            if path and Path(path).suffix == '.py' and Path(path).is_file():
                path = Path(path).resolve()
                if path.is_relative_to(NEW):
                    require(str(path) in result['source_bindings'], 'Unbound local imported source: ' + str(path))
        recheck(result['source_metadata']); recheck(result['weight_bindings'])
        recheck(result['artifact_bindings']); recheck(dependencies)
        result['after_swap'] = swap_state()
        result['swap_growth_bytes'] = result['after_swap']['used_bytes'] - result['before_swap']['used_bytes']
        require(result['swap_growth_bytes'] <= 2 * 1024 ** 3, 'Swap growth exceeds2GiB')
        guard()
        result.update(passed=True, completed=True, sources_unchanged=True)
    except BaseException as exc:
        result.update(passed=False, completed=False, sources_unchanged=False)
        result['error'], result['traceback'] = repr(exc), traceback.format_exc()
        record(kind='failure', error=repr(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        result['elapsed_s'] = time.monotonic() - start
        progress.close()
        write(folder / 'result.json', result)
        write(folder / 'completion.json', dict(schema=SCHEMA, stage=stage, completed=result['completed'],
              passed=result['passed'], result_sha256=sha(folder / 'result.json'),
              finished_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(stage=stage, passed=result['passed'], completed=result['completed'],
                          result_sha256=sha(folder / 'result.json')), sort_keys=True), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
