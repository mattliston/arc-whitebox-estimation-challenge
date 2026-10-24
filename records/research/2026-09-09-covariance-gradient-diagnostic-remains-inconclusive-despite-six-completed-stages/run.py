"""One current-d50 covariance gradient diagnostic; no optimizer or submission.

Default is inert help. Explicit sequential stages create exclusive folders.
The first three stages open no targets; both backend forward qualifications
must complete before either two-TRAIN-target gradient stage is admitted.
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
import resource
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
BD = NEW / 'birth_direction_fit_v1'
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
TRUTH = P2 / 'public_g22_operator_features_v1/truthpost'
INDEX = NEW / 'joint_prefix_final_fit_v1/manifest.json'
RANKS, MODES = (35, 100), ('full', 'read_cut')
STAGES = ('tiny', 'cpu_forward', 'mps_forward', 'cpu_gradient', 'mps_gradient', 'judge')
PINS = {
    HERE / 'model.py': 'faf98e7f35820cf80074386153aea7f9d9c5b10c0c567d30ab97b0da559b281a',
    STORE / 'manifest.json': '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50',
    INDEX: 'd61757d28236e3d35d87415962d1a57d18ff38ae189651df52df5e3412b9a971',
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    path = Path(path).resolve()
    if path.is_relative_to(TRUTH.resolve()):
        require(path in {TRUTH / f'truthpost_rank{r:04d}.npz' for r in RANKS}, 'Forbidden truth path')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    path = Path(path).resolve()
    before = path.stat()
    digest = sha(path)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Input changed while binding')
    return dict(path=str(path), sha256=digest, bytes=after.st_size, mtime_ns=after.st_mtime_ns)


def bind_expected(path, expected, destination):
    item = binding(path)
    require(item['sha256'] == expected, 'Input digest mismatch: ' + str(path))
    destination[item['path']] = item
    return item


def recheck(items):
    for item in items.values():
        require(binding(item['path']) == item, 'Sealed input changed: ' + item['path'])


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Module specification')
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def completed(stage, digest, sources):
    require(digest and len(digest) == 64, 'Missing completed stage pin: ' + stage)
    folder = HERE / ('stage_' + stage + '_v1')
    bind_expected(folder / 'result.json', digest, sources)
    rec = json.loads((folder / 'result.json').read_text())
    seal = json.loads((folder / 'completion.json').read_text())
    require(rec['stage'] == stage and rec['passed'] and rec['completed']
            and seal['result_sha256'] == digest and seal['completed'], 'Incomplete prerequisite ' + stage)
    for group in ('source_bindings', 'weight_bindings', 'artifacts'):
        for item in rec[group].values():
            actual = bind_expected(item['path'], item['sha256'], sources)
            require(actual == item, 'Prerequisite metadata changed')
    # Target bindings are intentionally NOT followed by prerequisite readers.
    sources[str((folder / 'completion.json').resolve())] = binding(folder / 'completion.json')
    return rec


def data_bindings(sources):
    manifest = json.loads((STORE / 'manifest.json').read_text())
    entries = manifest['roles']['observer_fit']['row_manifests']
    rows, weights = {}, {}
    for rank in RANKS:
        selected = [x for x in entries if x['role_rank'] == rank]
        require(len(selected) == 1, 'Unique canonical row')
        entry = selected[0]
        path = STORE / entry['path']
        bind_expected(path, entry['sha256'], weights)
        row = json.loads(path.read_text())
        wp = path.parent / 'weights.npy'
        bind_expected(wp, row['files']['weights.npy']['sha256'], weights)
        rows[rank] = str(wp.resolve())
    return rows, weights


def target_bindings():
    index = json.loads(INDEX.read_text())
    require(set(RANKS).issubset(index['train_ranks']), 'Diagnostic ranks must be TRAIN')
    expected = {x['rank']: x['target_sha256'] for x in index['target_sources']}
    result = {}
    for rank in RANKS:
        path = TRUTH / f'truthpost_rank{rank:04d}.npz'
        bind_expected(path, expected[rank], result)
    return result


def save_array(np, path, array, artifacts):
    require(np.isfinite(array).all(), 'Nonfinite artifact')
    with path.open('xb') as stream:
        np.save(stream, array, allow_pickle=False)
    artifacts[str(path.resolve())] = binding(path)
    return dict(path=str(path.resolve()), sha256=sha(path), shape=list(array.shape),
                dtype=str(array.dtype), raw_sha256=hashlib.sha256(array.tobytes()).hexdigest())


def load_array(np, item):
    require(sha(item['path']) == item['sha256'], 'Array seal changed')
    value = np.load(item['path'], allow_pickle=False)
    require(list(value.shape) == item['shape'] and str(value.dtype) == item['dtype']
            and hashlib.sha256(value.tobytes()).hexdigest() == item['raw_sha256'], 'Array metadata/bytes')
    require(np.isfinite(value).all(), 'Nonfinite array')
    return value


def relative(np, difference, reference):
    return float(np.linalg.norm(difference.astype(np.float64).ravel()) /
                 max(float(np.linalg.norm(reference.astype(np.float64).ravel())), 1e-30))


def chart(card, before=None):
    d = card.diagnostics()
    require(d['frozen_P'] and d['frozen_cstar'] and d['frozen_V0'], 'Frozen parameter ownership')
    require(set(d['variance_domain']) == set(range(16)), 'Incomplete variance domain')
    require(set(d['action_calls']) == set(range(10, 15)), 'Wrong covariance-write layers')
    require(all(v['finite'] and v['minimum'] >= 1e-10 and not v['below_floor_count']
                for v in d['variance_domain'].values()), 'Invalid prevariance')
    expected = {f'{l}:{s}' for l in range(16) for s in range(3 if l < 15 else 2)}
    require(set(d['selection_cells']) == expected, 'Incomplete selector audit')
    if before is not None:
        require(d['selection_cells'] == before, 'Backward replay changed selections')
    return d


def served(torch, value):
    return torch.where(value < 0, torch.zeros_like(value), value).float()


def tiny(np, torch, model, out, result, record):
    rng = np.random.Generator(np.random.PCG64DXSM(9309195))
    n = 7
    h = rng.standard_normal((n, 64)); h[:, 0] = .25
    b = rng.standard_normal((n, n))
    c = b @ b.T / n + np.eye(n) + .001 * rng.standard_normal((n, n))
    sigma = np.exp(.2 * rng.standard_normal(n))
    alive = np.asarray([1, 1, 0, 1, 0, 1, 1], dtype=np.float64)
    a = .07 * rng.standard_normal((65, 8))
    v0 = result['_v0'].astype(np.float64)
    cot = rng.standard_normal((n, n))
    direction = rng.standard_normal((65, 8)); direction /= np.linalg.norm(direction)
    errors, references = {}, {}
    for name, aa in (('zero', np.zeros_like(a)), ('nonzero', a)):
        args = [torch.tensor(x, dtype=torch.float64) for x in (c, h, sigma, alive, aa, v0)]
        args[4].requires_grad_(True)
        value = model.covariance_action(*args)
        reference = model.covariance_action_numpy64(c, h, sigma, alive, aa, v0)
        val = value.detach().numpy().copy()
        error = relative(np, val - reference, reference)
        require(error <= 1e-11, 'Tiny explicit congruence mismatch')
        loss = (value * torch.tensor(cot, dtype=torch.float64)).sum()
        gradient = torch.autograd.grad(loss, args[4])[0].detach().numpy().copy()
        ad = float(np.sum(gradient * direction))
        require(abs(ad) > 1e-10, 'Tiny fixed directional derivative is negligible')
        fds = []
        for eps in (1e-5, 5e-6):
            plus = model.covariance_action_numpy64(c, h, sigma, alive, aa + eps * direction, v0)
            minus = model.covariance_action_numpy64(c, h, sigma, alive, aa - eps * direction, v0)
            fd = float(np.sum((plus - minus) * cot) / (2 * eps))
            rel = abs(fd - ad) / abs(ad)
            require(rel <= 1e-6, 'Tiny continuous64 derivative mismatch')
            fds.append(dict(step=eps, finite_difference=fd, relative_error=rel))
        references[name] = (val, gradient)
        errors[name] = dict(explicit_relative_error=error, directional_ad=ad, finite_differences=fds)
        for device in ('cpu', 'mps'):
            q = [torch.tensor(x, dtype=torch.float32, device=device) for x in (c, h, sigma, alive, aa, v0)]
            q[4].requires_grad_(True)
            got = model.covariance_action(*q)
            scalar = (got * torch.tensor(cot, dtype=torch.float32, device=device)).sum()
            grad = torch.autograd.grad(scalar, q[4])[0].detach().cpu().numpy().copy()
            output = got.detach().cpu().numpy().copy()
            eout = relative(np, output - val, val)
            eg = relative(np, grad - gradient, gradient)
            require(eout <= 1e-4 and eg <= 1e-4, 'Tiny32 output/gradient qualification')
            errors[name][device] = dict(output_relative_error=eout, gradient_relative_error=eg)
            del q, got, scalar, grad, output
        del args, value, loss
        record(stage='tiny_case', case=name, checks=errors[name])
    result['tiny_checks'] = errors
    result['tiny_qualified'] = True


def forwards(np, torch, model, device, rows, out, result, previous, record):
    cstar = torch.tensor(model.frozen_coefficients(), device=device, dtype=torch.float32)
    v0 = torch.tensor(result['_v0'], device=device, dtype=torch.float32)
    cls = model._load_card_class()
    base = sys.modules['covariance_gap_v1_frozen_bd']
    result['outputs'], result['diagnostics'] = {}, {}
    with torch.no_grad():
        for rank in RANKS:
            wn = np.load(rows[rank], mmap_mode='r', allow_pickle=False)
            parent = base.BirthDirectionCard(rank, coefficients=cstar, device=device, weights=wn,
                                             audit_selections=True)
            pred, _ = parent.run(training=False)
            wanted = served(torch, pred).cpu().numpy().copy()
            del parent, pred
            outputs, diagnostics = {}, {}
            for mode in MODES:
                card = cls(rank, v0, mode=mode, device=device, weights=wn, audit_selections=True)
                pred, _ = card.run(training=False)
                actual = served(torch, pred).cpu().numpy().copy()
                require(actual.shape == (16, 1024) and np.isfinite(actual).all(), 'Wide finite output')
                require(actual.tobytes() == wanted.tobytes(), 'Zero action changed same-backend parent bytes')
                diagnostics[mode] = chart(card)
                outputs[mode] = save_array(np, out / f'{rank:04d}_{mode}.npy', actual, result['artifacts'])
                del card, pred, actual
                gc.collect()
                if device == 'mps':
                    torch.mps.empty_cache()
            require(diagnostics['full']['selection_cells'] == diagnostics['read_cut']['selection_cells'],
                    'Full/cut zero forward selection mismatch')
            if previous is not None:
                for mode in MODES:
                    cpu = load_array(np, previous['outputs'][str(rank)][mode])
                    maximum = float(np.max(np.abs(wanted.astype(np.float64) - cpu.astype(np.float64))))
                    require(maximum <= 1e-5, 'CPU/MPS zero output qualification')
                    diagnostics[mode]['cpu_max_output_difference'] = maximum
            result['outputs'][str(rank)], result['diagnostics'][str(rank)] = outputs, diagnostics
            record(stage='zero_forward', rank=rank, byte_exact_parent=True,
                   full_cut_selections_equal=True,
                   zero_sign_changes={mode: sum(x['zero_sign_changes'] for x in diagnostics[mode]['action_calls'].values())
                                      for mode in MODES})
            del wn, wanted
            gc.collect()
    result['forward_qualified'] = True
    write(out / 'target_free_forward_seal.json', dict(outputs=result['outputs'],
          training_targets_accessed=False, validation_targets_accessed=False,
          same_backend_parent_byte_exact=True, utc=datetime.now(timezone.utc).isoformat()))
    result['artifacts'][str((out / 'target_free_forward_seal.json').resolve())] = binding(out / 'target_free_forward_seal.json')


def gradients(np, torch, model, device, rows, own_forward, out, result, record):
    targets = target_bindings()
    result['target_bindings'] = targets
    write(out / 'two_train_targets_sealed.json', dict(bindings=targets, ranks=list(RANKS),
          all_forward_prerequisites_passed=True, target_arrays_decoded=False))
    result['artifacts'][str((out / 'two_train_targets_sealed.json').resolve())] = binding(out / 'two_train_targets_sealed.json')
    v0 = torch.tensor(result['_v0'], device=device, dtype=torch.float32)
    result['gradients'], result['diagnostics'], result['raw_errors'] = {}, {}, {}
    for rank in RANKS:
        wp = rows[rank]
        wn = np.load(wp, mmap_mode='r', allow_pickle=False)
        tp = TRUTH / f'truthpost_rank{rank:04d}.npz'
        require(binding(tp) == targets[str(tp.resolve())], 'Training target changed before decode')
        result['training_targets_accessed'] = True
        with np.load(tp, allow_pickle=False) as data:
            truth = data['mc_post'].copy()
        require(truth.shape == (16, 1024) and np.isfinite(truth).all(), 'Finite TRAIN means')
        target = torch.tensor(truth[-1], device=device, dtype=torch.float64 if device == 'cpu' else torch.float32)
        del truth
        records, charts, raw = {}, {}, {}
        for mode in MODES:
            A = torch.zeros((5, 65, 8), device=device, dtype=torch.float32, requires_grad=True)
            card = model.CovarianceGapCard(rank, v0, A=A, mode=mode, device=device, weights=wn,
                                           audit_selections=True)
            pred, _ = card.run(training=True)
            primal = served(torch, pred)
            expected = load_array(np, own_forward['outputs'][str(rank)][mode])
            require(primal.detach().cpu().numpy().tobytes() == expected.tobytes(), 'Gradient primal differs from sealed zero output')
            before = chart(card)['selection_cells']
            mse = ((primal[-1].to(target.dtype) - target) ** 2).mean()
            loss = mse / 2e-8
            require(bool(torch.isfinite(loss).detach().cpu()), 'Finite diagnostic loss')
            loss.backward()
            require(A.grad is not None and bool(torch.isfinite(A.grad).all().detach().cpu()), 'Finite A gradient')
            g = A.grad.detach().cpu().numpy().copy()
            records[mode] = save_array(np, out / f'{rank:04d}_{mode}_gradient.npy', g, result['artifacts'])
            records[mode]['norm'] = float(np.linalg.norm(g.astype(np.float64)))
            charts[mode] = chart(card, before)
            raw[mode] = float(mse.detach().cpu())
            require(not bool(A.detach().any().cpu()), 'A must never be updated')
            require(all(not x.requires_grad and x.grad is None for x in card.P0.values()), 'P must remain frozen')
            record(stage='gradient', rank=rank, mode=mode, raw_final_mse=raw[mode],
                   gradient_norm=records[mode]['norm'], parameter_updates=0)
            del A, card, pred, primal, expected, before, mse, loss, g
            gc.collect()
            if device == 'mps':
                torch.mps.empty_cache()
        require(raw['full'] == raw['read_cut'], 'Identical primals must have identical loss')
        result['gradients'][str(rank)], result['diagnostics'][str(rank)], result['raw_errors'][str(rank)] = records, charts, raw
        del target, wn
        gc.collect()
    recheck(targets)


def judge(np, cpu, mps, result, record):
    result['comparisons'] = {}
    numerical, information = True, True
    for rank in RANKS:
        row = {}
        vectors = {}
        for mode in MODES:
            c = load_array(np, cpu['gradients'][str(rank)][mode]).astype(np.float64)
            m = load_array(np, mps['gradients'][str(rank)][mode]).astype(np.float64)
            cn, mn = float(np.linalg.norm(c)), float(np.linalg.norm(m))
            e = relative(np, m - c, c)
            okay = cn > 1e-10 and mn > 1e-10 and e <= .05
            numerical = numerical and okay
            row[mode] = dict(cpu_norm=cn, mps_norm=mn, cross_backend_relative_error=e, qualified=okay)
            vectors[mode] = c
        f, q = vectors['full'].ravel(), vectors['read_cut'].ravel()
        nf, nq = float(np.linalg.norm(f)), float(np.linalg.norm(q))
        if nf > 1e-10 and nq > 1e-10:
            gap = float(np.linalg.norm(f-q)/nf)
            cosine = float(np.dot(f, q)/(nf*nq))
            derivative = float(-np.dot(f, q)/nq)
            gate = gap >= .30 and cosine <= .90
            row['information'] = dict(relative_gap=gap, cosine=cosine,
                full_derivative_along_unit_negative_cut=derivative, gate_passed=gate)
            information = information and gate
        else:
            row['information'] = dict(inconclusive=True, reason='negligible gradient norm', gate_passed=False)
            information = False
        result['comparisons'][str(rank)] = row
        record(stage='comparison', rank=rank, comparisons=row)
    result['gradient_numerical_qualified'] = numerical
    result['information_gate_passed'] = numerical and information
    result['scientific_conclusion'] = ('different_qualified_descent_directions' if numerical and information
        else 'missing_gradient_argument_not_supported' if numerical else 'numerically_inconclusive')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=STAGES)
    p.add_argument('--source-sha256')
    p.add_argument('--protocol-sha256')
    p.add_argument('--implementation-sha256')
    p.add_argument('--mps-slot-clear', action='store_true')
    for stage in STAGES[:-1]:
        p.add_argument('--' + stage.replace('_', '-') + '-sha256')
    args = p.parse_args()
    if args.stage is None:
        p.print_help(); return 0
    require(all(isinstance(x, str) and len(x) == 64 for x in
                (args.source_sha256, args.protocol_sha256, args.implementation_sha256)), 'Reviewed source/protocol pins required')
    stage = args.stage
    out = HERE / ('stage_' + stage + '_v1')
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    result = dict(schema='current_covariance_gradient_gap_v1', stage=stage, passed=False, completed=False,
        source_bindings={}, weight_bindings={}, target_bindings={}, artifacts={},
        training_targets_accessed=False, validation_targets_accessed=False, parameter_updates=0,
        no_fit=True, no_native_candidate=True, complete_score_measured=False,
        automatic_advancement=False, started_utc=datetime.now(timezone.utc).isoformat(),
        maximum_sampled_mps_driver_bytes=0)
    write(out / 'started.json', result)
    progress = (out / 'progress.jsonl').open('x')
    torch = None
    def record(**item):
        row = dict(utc=datetime.now(timezone.utc).isoformat(), elapsed_s=time.monotonic()-start, **item)
        line = json.dumps(row, sort_keys=True, allow_nan=False)
        progress.write(line + '\n'); progress.flush(); print(line, flush=True)
    def guard(_s=None, _f=None):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss = int(rss if sys.platform == 'darwin' else rss * 1024)
        use_mps = stage == 'tiny' or stage.startswith('mps')
        driver = int(torch.mps.driver_allocated_memory()) if torch is not None and use_mps else 0
        result['maximum_sampled_mps_driver_bytes'] = max(result['maximum_sampled_mps_driver_bytes'], driver)
        result['resources'] = dict(peak_rss_bytes=rss, mps_driver_bytes=driver, elapsed_s=time.monotonic()-start)
        require(rss < 12000000000 and driver < 16000000000, '12GB host/16GB driver resource cap')
        require(time.monotonic()-start < (120 if stage in ('tiny', 'judge') else 900), 'Stage wall cap')
    old_handler = signal.signal(signal.SIGALRM, guard)
    signal.setitimer(signal.ITIMER_REAL, .25, .25)
    try:
        require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'Requires nice10')
        require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'No bytecode writes')
        require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')), 'BLAS1 required')
        require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') == '0', 'MPS fallback prohibited')
        require('torch' not in sys.modules and 'threadpoolctl' not in sys.modules, 'Fresh process; no AP threadpoolctl')
        expected = P2 / 'data_gate/.venv' if stage == 'judge' else Path('/Users/matt/anaconda3')
        require(Path(sys.prefix).resolve() == expected.resolve(), 'Wrong interpreter')
        if stage == 'tiny' or stage.startswith('mps'):
            require(args.mps_slot_clear, 'Must attest sole MPS slot')
        pins = dict(PINS)
        pins.update({Path(__file__).resolve():args.source_sha256, HERE/'PROTOCOL.md':args.protocol_sha256,
                     HERE/'IMPLEMENTATION.md':args.implementation_sha256})
        for path, digest in pins.items():
            bind_expected(path, digest, result['source_bindings'])
        model = load_module(HERE / 'model.py', 'covariance_gap_diagnostic_model')
        for item in model.authenticate().values():
            bind_expected(item['path'], item['sha256'], result['source_bindings'])
        require('torch' not in sys.modules and 'numpy' not in sys.modules, 'Admission must remain inert')
        prerequisites = {}
        required = []
        if stage != 'tiny': required.append('tiny')
        if stage in ('mps_forward','cpu_gradient','mps_gradient','judge'): required.append('cpu_forward')
        if stage in ('cpu_gradient','mps_gradient','judge'): required.append('mps_forward')
        if stage in ('mps_gradient','judge'): required.append('cpu_gradient')
        if stage == 'judge': required.append('mps_gradient')
        for prior in required:
            prerequisites[prior] = completed(prior, getattr(args, prior+'_sha256'), result['source_bindings'])
        result['prerequisite_seals'] = {s:getattr(args,s+'_sha256') for s in required}
        import numpy as np
        rng = np.random.Generator(np.random.PCG64DXSM(9309095))
        result['_v0'] = (rng.standard_normal((65,8))/math.sqrt(65)).astype(np.float32)
        result['V0'] = save_array(np, out/'V0.npy', result['_v0'], result['artifacts'])
        for prior in prerequisites.values():
            require(load_array(np, prior['V0']).tobytes() == result['_v0'].tobytes(), 'Fixed V0 differs across stages')
        if stage != 'judge':
            import torch as t
            torch = t
            torch.set_num_threads(1); torch.set_num_interop_threads(1)
            require(torch.backends.mps.is_available(), 'MPS is unavailable')
            result['versions'] = {k:importlib.metadata.version(k) for k in ('numpy','torch')}
            require(result['versions'] == {'numpy':'1.26.4','torch':'2.7.0'}, 'Qualified AP versions changed')
        result['environment'] = dict(executable=sys.executable, pid=os.getpid(), nice=os.getpriority(os.PRIO_PROCESS,0))
        if stage == 'tiny':
            tiny(np, torch, model, out, result, record)
        elif stage == 'judge':
            judge(np, prerequisites['cpu_gradient'], prerequisites['mps_gradient'], result, record)
        else:
            rows, weights = data_bindings(result['source_bindings'])
            result['weight_bindings'] = weights
            write(out/'weights_sealed.json', dict(rows=rows, bindings=weights, targets_decoded=False))
            result['artifacts'][str((out/'weights_sealed.json').resolve())] = binding(out/'weights_sealed.json')
            device = stage.split('_')[0]
            if stage.endswith('forward'):
                forwards(np, torch, model, device, rows, out, result, prerequisites.get('cpu_forward'), record)
            else:
                gradients(np, torch, model, device, rows, prerequisites[device+'_forward'], out, result, record)
        # Every imported local implementation was pinned before numerical use.
        for item in tuple(sys.modules.values()):
            path = getattr(item,'__file__',None)
            if path:
                path = Path(path).resolve()
                if path.is_file() and path.suffix == '.py' and path.is_relative_to(NEW):
                    require(str(path) in result['source_bindings'], 'Unbound local import: '+str(path))
        require('threadpoolctl' not in sys.modules and 'flopscope' not in sys.modules, 'Unexpected numerical runtime')
        recheck(result['source_bindings']); recheck(result['weight_bindings']); recheck(result['target_bindings'])
        guard()
        result['passed'] = result['completed'] = True
        result['sources_unchanged'] = True
    except BaseException as exc:
        result['error'] = repr(exc)
        result['traceback'] = traceback.format_exc()
        record(stage='failure', error=repr(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,old_handler)
        result.pop('_v0',None)
        result['elapsed_s'] = time.monotonic()-start
        progress.close()
        write(out/'result.json', result)
        write(out/'completion.json', dict(passed=result['passed'], completed=result['completed'],
              result_sha256=sha(out/'result.json'), stage=stage, finished_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(stage=stage, passed=result['passed'], completed=result['completed'],
          result_sha256=sha(out/'result.json'), elapsed_s=result['elapsed_s']),sort_keys=True),flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
