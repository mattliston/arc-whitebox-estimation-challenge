"""Inert, target-free full-suffix pair-influence span diagnostic. CPU only."""
import argparse
import ast
import copy
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
CO = NEW/'current_bd_p_coadaptation_v1/model.py'
CO_SHA = '556f3b97613110e638f48127e40112ab203623fa9f6b076b0eab7f07d24658a0'
CORE = NEW/'whole_pair_kernel_core_v1.py'
CORE_SHA = '86c51be6d15f8b1c8bce463a4fb649c4974a339b5bebb20b7592669075620654'
P1 = NEW/'current_d50_native_objective_p1_v1'
BUILD_SHA = '877990bc84cb85431ff98f83e1e7012517c8b7914a1e0b39b29bf7d0b776b0c1'
BUILD_END_SHA = 'cdcb0a934224f242cad404dec3bae7eba60a712d8bd430ace0c7ab7111faea5c'
CANDIDATE_SHA = '0deb237350e753220b575796c0278f86884b8461bfeec4ab53eae9c15522c435'
PAYLOAD_SHA = '6b7b5ab8e357e26598cf64febd9f8b6993c3a7508efe6565d74ff3f02d816b65'
DESIGN_SHA = '461713beeb82d47299d2f0c02e78bed2598d6f675eaee38e1a74062d09d3c839'
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
STORE_SHA = '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50'
SCHEMA = 'current_endpoint_pair_quadrature_information_v1'
RANKS = (35, 36, 38, 100)
BLAS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
AP = Path('/Users/matt/anaconda3/bin/python')
LIMIT, RSS_LIMIT = 900., 12_000_000_000


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    path = Path(path).resolve()
    require('truthpost' not in path.parts and not
            (path.suffix in ('.npy', '.npz') and 'truth' in path.name.lower()),
            'Target files are forbidden')
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.publishing')
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    os.link(temporary, path)
    temporary.unlink()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Module loader')
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def raw_sha(array):
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


class Stage:
    def __init__(self, args):
        self.start, self.args = time.monotonic(), args
        self.folder = HERE/'information_v1'
        self.folder.mkdir(exist_ok=False)
        self.record = dict(schema=SCHEMA, stage='information', passed=False,
            completed=False, hashes_unchanged=False, sources_unchanged=False,
            information_promising=False, source_bindings={}, input_bindings={},
            weight_bindings={}, artifacts={}, rank_summaries=[], call_records=[],
            reference_calls=0, suffix_vjp_calls=0, retired_call_count=0,
            parameter_updates=0, compression_calls=0, target_accessed=False,
            alltruth_accessed=False, validation_targets_accessed=False,
            native_estimator_imported=False, native_gradient_certified=False,
            native_score_measured=False, complete_inference_cost_measured=False,
            deployment_metric_supplied_by_teacher=False, mps_used=False,
            started_utc=datetime.now(timezone.utc).isoformat())

    def guard(self):
        require(time.monotonic()-self.start < LIMIT, '900s whole worker cap')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < RSS_LIMIT,
                '12GB worker RSS cap')
        require('threadpoolctl' not in sys.modules, 'No AP threadpoolctl import')

    def pin(self, path, expected, group='input_bindings'):
        path = Path(path).resolve()
        require(path.is_relative_to(P2), 'Fixed project source/input scope')
        require(re.fullmatch('[0-9a-f]{64}', expected or '') is not None,
                'Explicit SHA binding')
        require(sha(path) == expected, 'Immutable input: '+str(path))
        previous = self.record[group].get(str(path))
        require(previous is None or previous == expected, 'Conflicting input pin')
        self.record[group][str(path)] = expected
        self.guard()
        return path

    def save(self, np, name, value):
        value = np.ascontiguousarray(value)
        require(value.dtype.kind in 'fiub' and np.isfinite(value).all(), 'Finite numeric artifact')
        path = self.folder/name
        with path.open('xb') as stream:
            np.save(stream, value, allow_pickle=False)
        digest = sha(path)
        self.record['artifacts'][str(path)] = digest
        return dict(path=str(path), sha256=digest, dtype=str(value.dtype),
                    shape=list(value.shape), raw_sha256=raw_sha(value))

    def json(self, name, value):
        path = self.folder/name
        write(path, value)
        digest = sha(path)
        self.record['artifacts'][str(path)] = digest
        return dict(path=str(path), sha256=digest)

    def retired(self, rank, kind, details):
        gc.collect()
        self.guard()
        index = self.record['retired_call_count']
        record = dict(schema=SCHEMA, event='retired', call=index, rank=rank,
            kind=kind, passed=True, completed=True, graph_retired=True,
            constant_cut_retained=True, target_accessed=False,
            elapsed_s=time.monotonic()-self.start,
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            **details)
        info = self.json(f'call_{index:02d}_complete.json', record)
        self.record['call_records'].append(info)
        self.record['retired_call_count'] += 1
        self.record['reference_calls' if kind == 'reference' else 'suffix_vjp_calls'] += 1
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)

    def recheck(self):
        for group in ('source_bindings', 'input_bindings', 'weight_bindings', 'artifacts'):
            for path, expected in self.record[group].items():
                require(sha(path) == expected, 'Final immutable inventory: '+path)
                self.guard()

    def close(self):
        self.record.update(elapsed_s=time.monotonic()-self.start,
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        if self.record['elapsed_s'] >= LIMIT or self.record['peak_rss_bytes'] >= RSS_LIMIT:
            self.record.update(passed=False, completed=False, information_promising=False)
        result_path, completion_path = self.folder/'result.json', self.folder/'completion.json'
        write(result_path, self.record)
        result_digest = sha(result_path)
        passed = self.record['passed'] and time.monotonic()-self.start < LIMIT
        write(completion_path, dict(passed=passed, completed=self.record['completed'] and passed,
                                    result_sha256=result_digest))
        completion_digest = sha(completion_path)
        elapsed = time.monotonic()-self.start
        final_passed = passed and elapsed < LIMIT
        write(self.folder/'finalization.json', dict(passed=final_passed,
            completed=self.record['completed'] and final_passed, elapsed_s=elapsed,
            result_sha256=result_digest, completion_sha256=completion_digest))
        return final_passed


def admit(stage):
    args = stage.args
    for path, expected in ((Path(__file__), args.source_sha256),
            (HERE/'PROTOCOL.md', args.protocol_sha256), (HERE/'watch.py', args.watcher_sha256),
            (HERE/'DESIGN.md', DESIGN_SHA), (CO, CO_SHA), (CORE, CORE_SHA)):
        stage.pin(path, expected, 'source_bindings')
    co = module(CO, 'endpoint_pair_frozen_co')
    for path, expected in co.authenticate().items():
        stage.pin(path, expected, 'source_bindings')
    stage.pin(P1/'build_v1/result.json', BUILD_SHA)
    stage.pin(P1/'build_v1/completion.json', BUILD_END_SHA)
    build, end = read(P1/'build_v1/result.json'), read(P1/'build_v1/completion.json')
    require(build['schema'] == 'current_d50_native_objective_p1_v1'
            and build['stage'] == 'build' and end['result_sha256'] == BUILD_SHA,
            'The sealed P1 build')
    require(all(build.get(k) is True for k in ('passed', 'completed', 'hashes_unchanged',
            'sources_unchanged', 'inverse_source_byte_exact', 'non_PKEYS_byte_exact',
            'unused_layer0_byte_exact', 'cstar_byte_exact', 'terminal_literal_roundtrip_byte_exact'))
            and end['passed'] and end['completed'], 'Qualified full-P1 payload/source lineage')
    require(build['candidate_sha256'] == CANDIDATE_SHA and build['updates'] == 40
            and build['terminal_epoch'] == 1, 'No alternate P1 checkpoint')
    for group in ('source_bindings', 'input_bindings', 'artifacts'):
        for path, expected in build[group].items():
            stage.pin(path, expected, 'source_bindings' if group == 'source_bindings' else 'input_bindings')
    payload = build['terminal_payload']
    require(Path(payload['path']).resolve() == P1/'training_v1/terminal_payload.npz'
            and payload['sha256'] == PAYLOAD_SHA, 'Sole sealed P1 parameter payload')
    stage.pin(payload['path'], PAYLOAD_SHA)
    stage.pin(STORE/'manifest.json', STORE_SHA)
    entries = read(STORE/'manifest.json')['roles']['observer_fit']['row_manifests']
    weights = {}
    for rank in RANKS:
        rows = [row for row in entries if row['role_rank'] == rank]
        require(len(rows) == 1, 'Unique original TRAIN weight row')
        path = (STORE/rows[0]['path']).resolve()
        require(path.is_relative_to(STORE), 'Weight row scope')
        stage.pin(path, rows[0]['sha256'], 'weight_bindings')
        row = read(path)
        require(row['role'] == 'observer_fit' and row['role_rank'] == rank, 'Original weight identity')
        wp = path.parent/'weights.npy'
        stage.pin(wp, row['files']['weights.npy']['sha256'], 'weight_bindings')
        weights[rank] = wp
    require(not any(name in sys.modules for name in ('numpy', 'torch', 'flopscope', 'threadpoolctl')),
            'Complete stdlib-only admission before numerical imports')
    stage.json('admission.json', dict(source_bindings=stage.record['source_bindings'],
        input_bindings=stage.record['input_bindings'], weight_bindings=stage.record['weight_bindings'],
        target_accessed=False, ranks=list(RANKS), reference_calls=4, suffix_vjp_calls=8))
    return co, payload, weights


def extracted_core(np, torch):
    source = CORE.read_text()
    nodes = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    feature_source = ast.get_source_segment(source, nodes['_gpk_features'])
    plane_original = ast.get_source_segment(source, nodes['_gpk_plane'])
    changes = (('.astype(fnp.float64)', '.to(torch.float64)'),
               ('fnp.sum(', 'torch.sum('), ('axis=1', 'dim=1'),
               ('    fnp.fill_diagonal(value, diagonal)',
                '    value = torch.diagonal_scatter(value, diagonal)'))
    plane_source = plane_original
    for before, after in changes:
        require(before in plane_source and after not in plane_source, 'Known frozen plane syntax')
        plane_source = plane_source.replace(before, after)
    inverse = plane_source
    for before, after in reversed(changes):
        inverse = inverse.replace(after, before)
    require(inverse == plane_original, 'Exact plane source inverse')
    env = dict(fnp=np, torch=torch)
    exec(compile(feature_source, str(CORE)+':exact_numpy_features', 'exec'), env)
    exec(compile(plane_source, str(CORE)+':functional_torch_plane', 'exec'), env)
    return env['_gpk_features'], env['_gpk_plane'], dict(
        features_source_sha256=hashlib.sha256(feature_source.encode()).hexdigest(),
        original_plane_source_sha256=hashlib.sha256(plane_original.encode()).hexdigest(),
        torch_plane_source_sha256=hashlib.sha256(plane_source.encode()).hexdigest(),
        plane_source_inverse_byte_exact=True, original_module_imported=False,
        additive_repair='star - L(after-before); L(R)=3R.T with diagonal diag(R)')


def tensor_hash(value):
    return raw_sha(value.detach().cpu().contiguous().numpy())


def state_hashes(state):
    return [dict(dtype=str(value.dtype), shape=list(value.shape), raw_sha256=tensor_hash(value))
            for value in state]


def capture_class(parent, stage):
    class Observed(parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.capture_reference, self.cut = True, None
            self.birth_rows, self.forward_step_visits = [], 0

        def step(self, layer, state, control, P):
            stage.guard()
            self.forward_step_visits += 1
            result = super().step(layer, state, control, P)
            if self.capture_reference and layer <= 7:
                ids = list(self.selection_cells[f'{layer}:2']['ordered_ids'])
                require(result[2].shape[1]-state[2].shape[1] == 2*len(ids),
                        'Observed pair count owns the actual concatenation')
                self.birth_rows.append(dict(layer=layer, ordered_birth_indices=ids,
                    before_columns=state[2].shape[1], after_columns=result[2].shape[1]))
                if layer == 7:
                    self.cut = tuple(value.detach() for value in result)
            stage.guard()
            return result
    return Observed


def pair_map(np, torch, card):
    require([row['layer'] for row in card.birth_rows] == list(range(8)), 'Exactly cohorts0..7')
    plus, minus, mapping, cursor = [], [], [], 0
    for row in card.birth_rows:
        count = len(row['ordered_birth_indices'])
        require(row['before_columns'] == cursor and row['after_columns'] == cursor+2*count,
                'No unowned factor columns')
        if row['layer'] < 7:
            for j, neuron in enumerate(row['ordered_birth_indices']):
                plus.append(cursor+j); minus.append(cursor+count+j)
                mapping.append(dict(cohort=row['layer'], birth_neuron=neuron,
                                    positive_column=cursor+j, negative_column=cursor+count+j))
        cursor += 2*count
    require(card.cut[2].shape[1] == cursor and len(plus) > 64, 'Complete physical pair roster')
    prefix = card.birth_rows[7]['before_columns']
    require(sorted(plus+minus) == list(range(prefix)), 'Exactly old cohorts0..6')
    ip, im = torch.tensor(plus, dtype=torch.long), torch.tensor(minus, dtype=torch.long)
    require(torch.equal(card.cut[3][ip], -card.cut[3][im]), 'Actual opposite stored pair weights')
    return ip, im, prefix, mapping


def cut_with_live_repair(torch, plane, cut, ip, im, prefix, coefficient, old_plane, active):
    m, C, p, lam, star, b, alive, radial, hs = cut
    scales = torch.ones(prefix, dtype=torch.float64)
    scales = scales.index_put((ip,), coefficient).index_put((im,), coefficient)
    old_weights = (lam[:prefix].double()*scales).float()
    live_weights = torch.cat((old_weights, lam[prefix:]))
    live_plane = plane(p[active, :prefix], old_weights, lambda a, b: a@b)
    delta = live_plane-old_plane
    correction = torch.diagonal_scatter(3*delta.T, torch.diagonal(delta))
    full = torch.zeros_like(star, dtype=torch.float64).index_put(
        (active[:, None], active[None, :]), correction)
    live_star = (star.double()-full).float()
    return (m, C, p, live_weights, live_star, b, alive, radial, hs)


def cotangent(np, rank, number):
    bits = []
    for block in range(4):
        digest = hashlib.sha256(f'EPQ1-cot:{rank}:{number}:{block}'.encode()).digest()
        bits.extend(1/32 if (byte >> bit) & 1 else -1/32 for byte in digest for bit in range(8))
    return np.asarray(bits, dtype=np.float64)


def suffix_vjp(stage, np, torch, checkpoint, co, card, plane, ip, im, prefix,
               old_plane, active, reference_raw, reference_served, reference_diag, rank, number):
    cut = card.cut
    before_hashes = state_hashes(cut)
    card.capture_reference, card._training = False, True
    card.selection_cells = copy.deepcopy({k: v for k, v in reference_diag['selection_cells'].items()
                                          if int(k.split(':')[0]) < 8})
    card.selection_visits = {key: 1 for key in card.selection_cells}
    card.variance_domain = {int(k): copy.deepcopy(v) for k, v in reference_diag['variance_domain'].items()
                           if int(k) < 8}
    card.variance_visits = {key: 1 for key in card.variance_domain}
    card._layer, card._selection_slot = None, 0
    coefficient = torch.ones(len(ip), dtype=torch.float64, requires_grad=True)
    state = cut_with_live_repair(torch, plane, cut, ip, im, prefix, coefficient, old_plane, active)
    require(state_hashes(state) == before_hashes, 'All nine null cut-state tensors byte exact')
    zeros = torch.zeros(card.n, dtype=card.dtype)
    means = []
    visits_before = card.forward_step_visits
    for layer in range(8, 16):
        def block(*values, layer=layer):
            return card.step(layer, values, zeros, card.P0)
        state = checkpoint(block, *state, use_reentrant=False)
        means.append(state[0])
    output = torch.stack(means)
    served = co.served(output)
    require(tensor_hash(output) == raw_sha(reference_raw[8:])
            and tensor_hash(served) == raw_sha(reference_served[8:]),
            'Unmodified same-CPU P1 raw and served suffix bytes')
    xi = cotangent(np, rank, number)
    scalar = (served[-1].double()*torch.from_numpy(xi)).sum()
    gradient, = torch.autograd.grad(scalar, coefficient)
    require(gradient.shape == coefficient.shape and torch.isfinite(gradient).all(), 'Finite full repaired VJP')
    diagnostic = co.check_replay(card)
    require(diagnostic['selection_cells'] == reference_diag['selection_cells']
            and diagnostic['variance_domain'] == reference_diag['variance_domain'],
            'All16 domains/all47 selectors unchanged through null backward')
    require(state_hashes(cut) == before_hashes, 'Immutable retained cut after full backward')
    answer = gradient.detach().numpy().copy()
    details = dict(cotangent=number, null_cut_byte_exact=True,
        raw_suffix_byte_exact=True, served_suffix_byte_exact=True,
        full_state_paths_live=True, live_weight_and_star_repair=True,
        forward_and_checkpoint_step_visits=card.forward_step_visits-visits_before,
        gradient_raw_sha256=raw_sha(answer), cotangent_raw_sha256=raw_sha(xi),
        all16_domains_qualified=True, all47_selectors_qualified=True,
        scalar=float(scalar.detach()), diagnostics=diagnostic)
    return answer, xi, details


def walsh(np, size, columns, scale):
    return np.asarray([[scale if (i & j).bit_count() % 2 == 0 else -scale
                        for j in range(columns)] for i in range(size)], dtype=np.float32)


def features(np, feature_owner, card, ip, im):
    cut = card.cut
    hidden = cut[8].numpy()@walsh(np, 64, 32, 1/8)
    future = walsh(np, 1024, 32, 1/32)
    for layer in range(15, 7, -1):
        future = card.w[layer].numpy()@future
        require(future.dtype == np.float32, 'Stored FP32 backward weight probes')
    values = np.concatenate((hidden, future), axis=1).astype(np.float64)
    values *= cut[6].numpy()[:, None]
    norms = np.sqrt(np.sum(values*values, axis=0))
    probes = (values/np.maximum(norms, 1e-12)[None, :]).astype(np.float32)
    v = probes.astype(np.float64)
    p = cut[2].numpy()
    h, d, u, cross, t = feature_owner(p[:, ip.numpy()], p[:, im.numpy()])
    v2 = v*v
    a, b = v.T@h, v.T@d
    c, e = v2.T@u, v2.T@cross
    f = (v2*v).T@t
    psi = 3*a*a*b+b*b*b/4-3*(c*b+e*a)+2*f
    phi = np.ascontiguousarray(psi.T*cut[3][ip].numpy().astype(np.float64)[:, None])
    require(probes.shape == (1024, 64) and np.isfinite(phi).all(), 'Finite fixed feature pack')
    return probes, phi


def crossfit(np, rank, mapping, phi, gradients):
    folds = np.asarray([hashlib.sha256(
        f"EPQ1-fold:{rank}:{item['cohort']}:{item['birth_neuron']}".encode()).digest()[0] & 1
        for item in mapping], dtype=np.int64)
    for cohort in range(7):
        require(set(folds[[item['cohort'] == cohort for item in mapping]].tolist()) == {0, 1},
                'Every old cohort occurs in both fixed folds')
    scales = np.sqrt(np.mean(phi*phi, axis=0))
    x = phi/np.where(scales > 0, scales, 1)[None, :]
    predictions = np.zeros_like(gradients)
    diagnostics = []
    for number, gradient in enumerate(gradients):
        parts = []
        for fitting in (0, 1):
            take, test = folds == fitting, folds != fitting
            require(int(take.sum()) >= 64 and int(test.sum()) >= 64, 'Crossfit sample support')
            xt, gt = x[take], gradient[take]
            gram = xt.T@xt/len(xt)+1e-6*np.eye(64)
            rhs = xt.T@gt/len(xt)
            coefficients = np.linalg.solve(gram, rhs)
            predictions[number, test] = x[test]@coefficients
            require(np.isfinite(coefficients).all(), 'Finite unique fixed-ridge solve')
            error = predictions[number, test]-gradient[test]
            energy, residual = float(gradient[test]@gradient[test]), float(error@error)
            parts.append(dict(fitting_fold=fitting, fitting_count=int(take.sum()),
                test_count=int(test.sum()), gradient_energy=energy, residual_energy=residual,
                residual_ratio=None if energy == 0 else residual/energy,
                gram_condition=float(np.linalg.cond(gram)), coefficient_norm=float(np.linalg.norm(coefficients))))
        error = predictions[number]-gradient
        energy, residual = float(gradient@gradient), float(error@error)
        diagnostics.append(dict(cotangent=number, gradient_energy=energy, residual_energy=residual,
            residual_ratio=None if energy == 0 else residual/energy, folds=parts))
    energy = sum(x['gradient_energy'] for x in diagnostics)
    residual = sum(x['residual_energy'] for x in diagnostics)
    ratio = None if energy == 0 else residual/energy
    promising = (ratio is not None and ratio <= .10 and
                 all(x['residual_ratio'] is not None and x['residual_ratio'] <= .20 for x in diagnostics))
    return folds, scales, predictions, dict(cotangents=diagnostics,
        gradient_energy=energy, residual_energy=residual, aggregate_residual_ratio=ratio,
        information_promising=bool(promising),
        zero_energy_inconclusive=any(item['gradient_energy'] == 0 for item in diagnostics),
        fitted_coefficients_are_teacher_only=True, parameter_updates=0)


def run_rank(stage, np, torch, checkpoint, co, Card, feature_owner, plane, payload, wp, rank):
    stage.guard()
    weights = np.load(wp, allow_pickle=False, mmap_mode='r')
    require(weights.shape == (16, 1024, 1024) and weights.dtype == np.float32
            and np.isfinite(weights).all(), 'Original finite FP32 MLP weights')
    P = {key: torch.from_numpy(value.copy()) for key, value in payload.items()}
    card = Card(rank, P=P, device='cpu', weights=weights, audit_selections=True)
    require(all(not value.requires_grad and value.device.type == 'cpu' for value in card.P0.values()),
            'All P fixed; CPU only')
    with torch.no_grad():
        reference, _ = card.run(training=False, capture=False)
        served = co.served(reference)
    reference_diag = copy.deepcopy(co.check_replay(card))
    require(card.forward_step_visits == 16 and card.cut is not None, 'One original full16 reference')
    reference_raw, reference_served = reference.numpy().copy(), served.numpy().copy()
    ip, im, prefix, mapping = pair_map(np, torch, card)
    cut_hash = state_hashes(card.cut)
    active = card.cut[6].nonzero().flatten()
    with torch.no_grad():
        before = plane(card.cut[2][active, :prefix], card.cut[3][:prefix], lambda a, b: a@b)
    reference_info = stage.save(np, f'rank_{rank:04d}_reference_raw.npy', reference_raw)
    served_info = stage.save(np, f'rank_{rank:04d}_reference_served.npy', reference_served)
    pair_info = stage.json(f'rank_{rank:04d}_pair_map.json', dict(mapping=mapping,
        births=card.birth_rows, old_prefix_columns=prefix, old_pair_count=len(ip),
        cut_state_hashes=cut_hash, physical_columns=card.cut[2].shape[1]))
    del reference, served
    stage.retired(rank, 'reference', dict(reference=reference_info, served=served_info,
        pair_map=pair_info, full_model_calls=1, all16_domains_qualified=True,
        all47_selectors_qualified=True, diagnostics=reference_diag))
    gradients, cotangents = [], []
    for number in (0, 1):
        gradient, xi, details = suffix_vjp(stage, np, torch, checkpoint, co, card, plane,
            ip, im, prefix, before, active, reference_raw, reference_served, reference_diag, rank, number)
        gradients.append(gradient); cotangents.append(xi)
        details['gradient'] = stage.save(np, f'rank_{rank:04d}_cot_{number}_gradient.npy', gradient)
        details['cotangent_array'] = stage.save(np, f'rank_{rank:04d}_cot_{number}.npy', xi)
        stage.retired(rank, 'suffix_vjp', details)
    stage.guard()
    probes, phi = features(np, feature_owner, card, ip, im)
    require(state_hashes(card.cut) == cut_hash, 'Feature construction leaves retained state unchanged')
    require(all(raw_sha(card.P0[key].numpy()) == raw_sha(value) for key, value in payload.items()),
            'No P updates or in-place payload mutation')
    M, n = len(ip), card.n
    del card, P, weights, before, active, ip, im
    gc.collect(); stage.guard()
    gradients = np.stack(gradients)
    folds, scales, predictions, report = crossfit(np, rank, mapping, phi, gradients)
    report.update(rank=rank, old_pairs=M, constant_cut_retired=True,
        feature_shape=list(phi.shape), reference=reference_info, served=served_info,
        pair_map=pair_info, feature_pack='32_hidden_Walsh_plus32_backward_weight_Walsh',
        feature_GEMM_subtotal_F=20*64*n*M,
        probe_GEMM_subtotal_F=8*2*n*n*32+2*n*64*32,
        complete_constructor_cost_measured=False, no_inference_cost_claim=True,
        probes=stage.save(np, f'rank_{rank:04d}_probes.npy', probes),
        features=stage.save(np, f'rank_{rank:04d}_features.npy', phi),
        folds_array=stage.save(np, f'rank_{rank:04d}_folds.npy', folds),
        feature_scales=stage.save(np, f'rank_{rank:04d}_feature_scales.npy', scales),
        crossfit_predictions=stage.save(np, f'rank_{rank:04d}_crossfit_predictions.npy', predictions))
    stage.record['rank_summaries'].append(report)
    stage.json(f'rank_{rank:04d}_summary.json', report)
    del probes, phi, gradients, folds, scales, predictions
    gc.collect(); stage.guard()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    for name in ('source', 'protocol', 'watcher'):
        parser.add_argument('--'+name+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    require(all(re.fullmatch('[0-9a-f]{64}', getattr(args, key+'_sha256') or '')
                for key in ('source', 'protocol', 'watcher')), 'Explicit reviewed source hashes')
    require(Path(sys.executable).resolve() == AP.resolve() and sys.flags.optimize == 0
            and os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'Fixed AP nice10 interpreter')
    require(all(os.environ.get(key) == '1' for key in BLAS)
            and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'BLAS1 and bytecode-off launch')
    stage = Stage(args)
    def interrupted(*_):
        raise TimeoutError('Fixed CPU teacher interruption/deadline')
    signal.signal(signal.SIGALRM, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    signal.setitimer(signal.ITIMER_REAL, LIMIT)
    try:
        co, payload_metadata, weights = admit(stage)
        import numpy as np
        import torch
        from torch.utils.checkpoint import checkpoint
        require(np.__version__ == '1.26.4' and torch.__version__ == '2.7.0', 'Fixed CPU library frame')
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        require(torch.get_num_threads() == torch.get_num_interop_threads() == 1, 'Torch threads1')
        stage.record['environment'] = dict(versions=dict(numpy=np.__version__, torch=torch.__version__),
            thread_variables={key: os.environ.get(key) for key in BLAS},
            torch_threads=1, torch_interop_threads=1, device='cpu', moment_dtype='float64',
            cp_dtype='float32', teacher='declared_smooth_lift_not_native_derivative')
        with np.load(payload_metadata['path'], allow_pickle=False) as source:
            payload = {key: source[key].copy() for key in source.files}
        original = co.payload_numpy()
        require(set(payload) == set(original) == set(payload_metadata['arrays']), 'Complete fixed P mapping')
        for key, value in payload.items():
            require(value.dtype == np.float32 and value.shape == original[key].shape
                    and np.isfinite(value).all(), 'Finite same-shaped P field: '+key)
            require(dict(shape=list(value.shape), dtype=str(value.dtype), raw_sha256=raw_sha(value))
                    == payload_metadata['arrays'][key], 'Sealed P raw bytes: '+key)
            require(key in co.PKEYS or value.tobytes() == original[key].tobytes(), 'Frozen non-P field')
        for key in co.INACTIVE_KEYS:
            require(payload[key][0].tobytes() == original[key][0].tobytes(), 'Frozen inactive P row')
        feature_owner, plane, source_proofs = extracted_core(np, torch)
        stage.record['core_source_proofs'] = source_proofs
        Card = capture_class(co._card_class(), stage)
        for rank in RANKS:
            run_rank(stage, np, torch, checkpoint, co, Card, feature_owner, plane, payload, weights[rank], rank)
        require(stage.record['reference_calls'] == 4 and stage.record['suffix_vjp_calls'] == 8
                and stage.record['retired_call_count'] == 12
                and [x['rank'] for x in stage.record['rank_summaries']] == list(RANKS), 'Exact fixed workload')
        require(all(x['constant_cut_retired'] for x in stage.record['rank_summaries']), 'All private cuts retired')
        del payload, original
        gc.collect(); stage.recheck(); stage.guard()
        stage.record.update(passed=True, completed=True, hashes_unchanged=True, sources_unchanged=True,
            all_graphs_retired=True, all_private_cut_states_retired=True,
            information_promising=all(x['information_promising'] for x in stage.record['rank_summaries']))
    except BaseException as exc:
        stage.record.update(passed=False, completed=False, hashes_unchanged=False,
            sources_unchanged=False, information_promising=False, error=f'{type(exc).__name__}: {exc}')
    finally:
        gc.collect()
        signal.setitimer(signal.ITIMER_REAL, 0)
        passed = stage.close()
    print(json.dumps(dict(passed=passed, completed=stage.record['completed'],
        information_promising=stage.record['information_promising'],
        reference_calls=stage.record['reference_calls'], suffix_vjp_calls=stage.record['suffix_vjp_calls'],
        retired_call_count=stage.record['retired_call_count'], error=stage.record.get('error')), sort_keys=True), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
