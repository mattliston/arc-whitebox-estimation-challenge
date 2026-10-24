"""Target-free tiny function checks and exactly five current-COMP MPS calls."""
import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import sys
import time
import traceback
import weakref

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
JOINT = NEW/'current_comp_joint_p_node_v1'
CO = NEW/'current_bd_p_coadaptation_v1'
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
REF, OUT = JOINT/'qualification_v1', HERE/'qualification_v1'
ROSTER = ((35, 'null_forward'), (100, 'null_forward'), (35, 'receiver_backward'),
          (100, 'receiver_backward'), (35, 'receiver_backward_repeat'))
PINS = {
    JOINT/'qualification.py': '6d161c22229016f524eeea28313af996de309e43e7585c5f101fe7ce82218e04',
    JOINT/'QUALIFICATION_PROTOCOL.md': 'd5bd03fed976479e74580c7f2d0f7f2727c23b629225a94799af9f9cc4f0460e',
    REF/'result.json': '25d1beb88adf81264705c15b287b676f1bb15ba8f24b0515bad58eeff7d04469',
    REF/'completion.json': '542378dc2b7cc51ad17a078a1e67d0cb0b415df713c61fe965ce8fdd17c1d721',
    REF/'finalization.json': '857cb560c1c93170726ee8f1b943ab754607e3dcc3686570d1ad1090e802e084',
    CO/'qualify.py': '1a3df2687ed3bef6ffcfc6f690f3cfe8665ad04aae4da94012c608d4871a7b20',
    NEW/'current_comp_receiver_birth_v1/watch_qualification.py': '4b5de5318afc57644ab81b5bdafebe566b939b6c7098622f1f848d9dbd58a1bb',
    NEW/'current_p1_native_objective_p2_monitor_v2/monitor.py': 'd4b3af53b716593670d8e0c7f0fc6f2133b57f1088d9966c06635d3a23f39873',
    NEW/'current_d50_native_objective_p1_v1/watch.py': '4e8177d6ff943cdee37e48ccca750bf9d552b32cec344aa2b37e6304cbd55fb3',
    STORE/'manifest.json': '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50',
    STORE/'rows/observer_fit/0035/weights.npy': '14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2',
    STORE/'rows/observer_fit/0100/weights.npy': 'c4923171c8ddef28396aabd70335dad29141361efc13865ccb660a4a57421ee5',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    path = Path(path).resolve()
    require(path.is_relative_to(P2) and 'truthpost' not in path.parts,
        'Only target-free Phase2 source/input/artifact access')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def tiny(np, torch, model, fixture, save):
    """No card: original versus modified birth function and direct signed cubes."""
    rng = np.random.Generator(np.random.PCG64DXSM(9309105))
    n = 8
    z = rng.standard_normal((n, n))
    C = z@z.T/n + np.eye(n)
    sigma = np.sqrt(np.diag(C)); alpha = np.linspace(-1.7, 1.9, n)
    pdf = np.exp(-alpha*alpha/2)/math.sqrt(2*math.pi)
    gate = np.asarray([.5*math.erfc(-x/math.sqrt(2)) for x in alpha])
    K = .02*rng.standard_normal((n, n))
    B = gate[:, None]*C; np.fill_diagonal(B, 0)
    H = (.2*rng.standard_normal((n, 64))).astype(np.float32)
    coefficients = (np.asarray([.2, -.3, .1, .15])[:, None]
        + .03*np.sin(np.arange(4*n).reshape(4, n))).astype(np.float32)
    alive = np.asarray([1, 1, 0, 1, 1, 1, 0, 1.])
    births = np.asarray([1, 0, 1, 1, 0, 1, 0, 1.])
    t = np.asarray([2.5, -.3, 1.4, .01, .2, -1.2, .8, .4])
    s = t-np.linspace(.15, 1.1, n); a = np.linspace(-.2, .7, n)
    cot_base, cot_plane = (rng.standard_normal((n, n)) for _ in range(2))
    base = model.runtime(); function = model.numeric_functions()['birth']
    records = {}; cpu_new = cpu_plane = None
    for device in ('cpu', 'mps'):
        dtype = torch.float64 if device == 'cpu' else torch.float32
        T = lambda value: torch.tensor(value, dtype=dtype, device=device)
        bt, ct, kt = T(B), T(C), T(K)
        aux = dict(sigma=T(sigma), alpha=T(alpha), pdf=T(pdf), gate=T(gate))
        arguments = (bt, ct, kt, aux, torch.tensor(coefficients, dtype=torch.float32, device=device),
                     T(t), T(s), T(a), T(alive), T(births))
        h = torch.tensor(H, dtype=torch.float32, device=device, requires_grad=True)
        zero = torch.zeros((64, 4), dtype=torch.float32, device=device, requires_grad=True)
        original_new, original_plane = base.birth_action(*arguments)
        null_new, null_plane = function(*arguments, model.row_modulation(h, zero))
        array = lambda value: value.detach().cpu().contiguous().numpy().copy()
        require(array(null_new).tobytes() == array(original_new).tobytes()
            and array(null_plane).tobytes() == array(original_plane).tobytes(),
            'Tiny zero-R branch is byte-identical to original birth action')
        null_loss = ((null_new.to(dtype)*T(cot_base)).sum()+(null_plane*T(cot_plane)).sum())/(n*n)
        null_loss.backward()
        null_gradient = array(zero.grad)
        require(np.isfinite(null_gradient).all() and np.linalg.norm(null_gradient) > 0,
            'The zero receiver branch stays differentiable, not bypassed')
        r = torch.tensor(fixture, dtype=torch.float32, device=device, requires_grad=True)
        h.grad = None
        new, plane = function(*arguments, model.row_modulation(h, r))
        loss = ((new.to(dtype)*T(cot_base)).sum()+(plane*T(cot_plane)).sum())/(n*n)
        loss.backward(); gradient, h_gradient = array(r.grad), array(h.grad)
        require(np.isfinite(gradient).all() and np.linalg.norm(gradient) > 0
            and np.isfinite(h_gradient).all() and np.linalg.norm(h_gradient) > 0,
            'Finite connected R and endogenous-H derivatives')
        records[device] = dict(null_bytes_exact=True, zero_gradient_connected=True,
            R_gradient=save('tiny_'+device+'_R_gradient', gradient),
            H_gradient=save('tiny_'+device+'_H_gradient', h_gradient),
            zero_R_gradient=save('tiny_'+device+'_zero_R_gradient', null_gradient),
            new=save('tiny_'+device+'_new', array(new)), plane=save('tiny_'+device+'_plane', array(plane)))
        if device == 'cpu':
            cpu_new, cpu_plane = array(new), array(plane)
            cpu_gradient = gradient.astype(np.float64)
        else:
            relative = float(np.linalg.norm(gradient.astype(np.float64)-cpu_gradient)
                / max(np.linalg.norm(cpu_gradient), 1e-12))
            require(relative <= .05, 'Fixed tiny CPU64/MPS32 R-gradient relative agreement<=5%')
        del bt, ct, kt, aux, arguments, h, zero, original_new, original_plane, null_new, null_plane
        del null_loss, r, new, plane, loss
        gc.collect(); torch.mps.synchronize(); torch.mps.empty_cache()
    old = (B*alive[:, None]).astype(np.float32)
    tt = (t*alive).astype(np.float32).astype(np.float64)
    ss = (s*alive).astype(np.float32).astype(np.float64)
    aa = a.astype(np.float32).astype(np.float64)*births
    def tensor(stored):
        p, q = stored.astype(np.float64).copy(), stored.astype(np.float64).copy()
        np.fill_diagonal(p, tt); np.fill_diagonal(q, ss)
        return sum(aa[j]*(np.einsum('i,j,k->ijk', p[:, j], p[:, j], p[:, j])
            - np.einsum('i,j,k->ijk', q[:, j], q[:, j], q[:, j])) for j in range(n))
    changed = tensor(cpu_new)-tensor(old)
    literal = np.einsum('iij->ij', changed)
    error = float(np.max(np.abs(literal-cpu_plane)))
    require(error <= 1e-12, 'Actual stored signed-cube repeated plane agrees with sole old-to-new repair')
    star_A = -3*literal.T+2*np.diag(np.diag(literal))
    star = np.empty((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                star[i, j, k] = (star_A[i, j]*(j == k)+star_A[j, i]*(i == k)+star_A[k, i]*(i == j))/3
    mask = np.fromfunction(lambda i, j, k: (i == j) | (i == k) | (j == k), (n, n, n))
    restored = changed+star
    require(np.max(np.abs(restored[mask])) <= 1e-12
        and np.array_equal(restored[~mask], changed[~mask]), 'Only repeated planes restored, distinct change retained')
    return dict(passed=True, n=n, seed=9309105, no_card_calls=True, new_function_evaluations=4,
        original_function_evaluations=2, null_byte_exact_both_devices=True,
        CPU64_MPS32_R_gradient_relative=relative, R_gradient_limit=.05,
        signed_cube_plane_max_error=error, repeated_plane_restored=True,
        null_R_gradient_connected=True, endogenous_H_gradient_connected=True, records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', action='store_true')
    for key in ('source', 'model', 'protocol', 'watcher'):
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    began = time.monotonic()
    def expired(*_):
        raise TimeoutError('180s whole receiver-direction qualification')
    signal.signal(signal.SIGALRM, expired); signal.alarm(180)
    OUT.mkdir(exist_ok=False)
    result = dict(schema='current_comp_receiver_directions_qualification_v1', passed=False, completed=False,
        source_bindings={}, input_bindings={}, artifacts={}, calls=[], full_model_calls=0, retired_call_count=0,
        full_model_call_roster=[list(item) for item in ROSTER], parameter_updates=0,
        training_targets_accessed=False, validation_targets_accessed=False, truth_accessed=False,
        native_gradient_qualified=False, no_native_gradient_claim=True, native_score_measured=False,
        execution_integrity_passed=False, maximum_sampled_mps_driver_bytes=0)
    write(OUT/'started.json', result)
    try:
        def pin(path, digest):
            path = Path(path).resolve()
            require(re.fullmatch('[0-9a-f]{64}', digest or '') and sha(path) == digest, 'Frozen source/input '+str(path))
            group = 'source_bindings' if path.suffix in ('.py', '.md') else 'input_bindings'
            require(result[group].get(str(path), digest) == digest, 'Consistent source/input owner')
            result[group][str(path)] = digest
        for path, digest in {**PINS, Path(__file__).resolve(): args.source_sha256,
            HERE/'model.py': args.model_sha256, HERE/'QUALIFICATION_PROTOCOL.md': args.protocol_sha256,
            HERE/'watch_qualification.py': args.watcher_sha256}.items():
            pin(path, digest)
        pin(HERE/'WATCH_SPEC.md', sha(HERE/'WATCH_SPEC.md'))
        model = module(HERE/'model.py', 'qualified_receiver_directions_model')
        for path, digest in model.authenticate().items():
            pin(path, digest)
        qr, qc, qf = (read(REF/name) for name in ('result.json', 'completion.json', 'finalization.json'))
        require(all(qr.get(k) is True for k in ('passed', 'completed', 'execution_integrity_passed',
            'sources_unchanged', 'hashes_unchanged', 'all_graphs_retired'))
            and qc['passed'] and qc['completed'] and qf['passed'] and qf['completed']
            and qr['full_model_calls'] == qr['retired_call_count'] == 5
            and not qr['training_targets_accessed'] and not qr['validation_targets_accessed']
            and qc['result_sha256'] == qf['result_sha256'] == PINS[REF/'result.json']
            and qf['completion_sha256'] == PINS[REF/'completion.json'] and qf['elapsed_s'] < 180,
            'Completed exact-JOINT target-free five-call reference and all three seals')
        for group in ('source_bindings', 'input_bindings', 'artifacts'):
            for path, digest in qr[group].items():
                pin(path, digest)
        require([(x['rank'], x['kind']) for x in qr['calls'][:2]] == [(35, 'reference_forward'), (100, 'reference_forward')],
            'Exactly the two sealed current-COMP MPS reference outputs')
        require(sys.flags.optimize == 0 and sys.dont_write_bytecode
            and Path(sys.prefix).resolve() == Path('/Users/matt/anaconda3').resolve()
            and os.getpriority(os.PRIO_PROCESS, 0) >= 10
            and all(os.environ.get(k) == '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
            and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
            and os.environ.get('PYTORCH_MPS_LOW_WATERMARK_RATIO') == '0.1', 'Fixed AP nice10/BLAS1/bytecode/MPS environment')
        require(not any(k in sys.modules for k in ('numpy', 'torch', 'flopscope', 'threadpoolctl')),
            'Complete inert source admission before numerical imports')
        import numpy as np
        import torch
        require(np.__version__ == '1.26.4' and torch.__version__.split('+')[0] == '2.7.0'
            and torch.backends.mps.is_available() and 'threadpoolctl' not in sys.modules, 'Qualified AP runtime')
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        recommended = int(torch.mps.recommended_max_memory()); require(recommended > 0, 'Positive MPS capacity')
        fraction = min(1., 16_000_000_000/recommended); torch.mps.set_per_process_memory_fraction(fraction)
        result['mps_allocator'] = dict(recommended_max_memory_bytes=recommended, configured_fraction=fraction)
        model.runtime()
        functions = [n for n in ast.parse((CO/'qualify.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'scalar']
        require(len(functions) == 1, 'Exact original synthetic scalar owner')
        scalar_env = {'math': math}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(CO/'qualify.py')+':scalar', 'exec'), scalar_env)
        scalar = scalar_env['scalar']
        def array(value):
            return value.detach().cpu().contiguous().numpy().copy()
        def rawsha(value):
            return hashlib.sha256(value.tobytes()).hexdigest()
        def save(name, value):
            path = OUT/(name+'.npy')
            with path.open('xb') as stream:
                np.save(stream, value, allow_pickle=False)
            digest = sha(path); result['artifacts'][str(path)] = digest
            return dict(path=str(path), sha256=digest, raw_sha256=rawsha(value), shape=list(value.shape), dtype=str(value.dtype))
        def doc(name, value):
            path = OUT/(name+'.json'); write(path, value); digest = sha(path)
            result['artifacts'][str(path)] = digest
            return dict(path=str(path), sha256=digest)
        def load(info):
            require(sha(info['path']) == info['sha256'], 'Previously sealed reference artifact')
            value = np.load(info['path'], allow_pickle=False)
            require(list(value.shape) == info['shape'] and str(value.dtype) == info['dtype']
                and rawsha(value) == info['raw_sha256'] and np.isfinite(value).all(), 'Finite exact reference metadata/raw bytes')
            return value
        def guard():
            torch.mps.synchronize(); driver = int(torch.mps.driver_allocated_memory())
            result['maximum_sampled_mps_driver_bytes'] = max(result['maximum_sampled_mps_driver_bytes'], driver)
            rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            require(driver < 16_000_000_000 and rss < 12_000_000_000 and time.monotonic()-began < 180,
                'Complete180s/12GB host/16GB driver bound')
            return dict(peak_rss_bytes=rss, mps_driver_bytes=driver, elapsed_s=time.monotonic()-began)
        def domain(card):
            d = model._CO.check_replay(card)
            d = card.diagnostics()
            require(set(d['node_head_calls']) == set(d['receiver_head_calls']) == {str(l) for l in range(1, 15)},
                'Every actual original NODE and new receiver birth call')
            return d
        def frozen_hash(card):
            return {key: rawsha(array(value)) for key, value in
                dict(card.P0, NODE_A=card.A, receiver_R=card.R, cstar=card.coefficients, weights=card.w).items()}
        def retire(index, refs):
            gc.collect(); torch.mps.synchronize(); torch.mps.empty_cache(); gc.collect()
            require(all(ref() is None for ref in refs), 'All card/output/gradient graph owners retired')
            result['last_resource_sample'] = guard()
            result['full_model_calls'] += 1; result['retired_call_count'] += 1
            print(json.dumps(dict(event='retired', call=index)), flush=True)
        p, q = np.arange(1, 65, dtype=np.float64)[:, None], np.arange(1, 5, dtype=np.float64)[None, :]
        fixture = (.005*np.sin(.37*p+.19*q)).astype(np.float32)
        result['R_fixture'] = save('fixed_nonzero_R', fixture)
        result['tiny'] = tiny(np, torch, model, fixture, save)
        gc.collect(); torch.mps.synchronize(); torch.mps.empty_cache(); guard()
        references = {item['rank']: (load(item['raw']), load(item['served']), read(item['diagnostics']['path']))
            for item in qr['calls'][:2]}
        cots = {rank: load(qr['cotangents'][str(rank)]) for rank in (35, 100)}
        require(all(value.shape == (16, 1024) and value.dtype == np.float32 for value in cots.values()),
            'Original full-card synthetic cotangents')
        for rank, kind in ROSTER[:2]:
            index = result['full_model_calls']; weights = np.load(STORE/f'rows/observer_fit/{rank:04d}/weights.npy', allow_pickle=False)
            P, A, R = model.frozen('mps', trainable_R=False)
            card = model.card(rank, P=P, A=A, R=R, device='mps', weights=weights); before = frozen_hash(card)
            require(card.R is R and card.A is A and all(card.P0[k] is value for k, value in P.items()), 'Exact supplied leaf identities')
            with torch.no_grad():
                raw, trace = card.run(training=False)
            served = model._CO.served(raw); rv, sv = array(raw), array(served); d = domain(card)
            require(rv.tobytes() == references[rank][0].tobytes() and sv.tobytes() == references[rank][1].tobytes()
                and all(d[k] == references[rank][2][k] for k in ('variance_domain', 'selection_cells')),
                'All16 raw/served null bytes and original47 selector/domain reference records')
            require(frozen_hash(card) == before, 'Null P/A/R/cstar/weights remain frozen')
            rec = dict(call=index, rank=rank, kind=kind, passed=True, completed=True, current_COMP_byte_exact=True,
                raw=save(f'call_{index:02d}_raw', rv), served=save(f'call_{index:02d}_served', sv),
                diagnostics=doc(f'call_{index:02d}_diagnostics', d), frozen_hashes=before)
            result['calls'].append(rec); doc(f'call_{index:02d}_complete', rec)
            refs = [weakref.ref(card), weakref.ref(raw), weakref.ref(served)]
            del card, raw, served, trace, P, A, R, weights, rv, sv, d
            retire(index, refs)
        P, A, initial_zero = model.frozen('mps', trainable_R=False); del initial_zero
        R = torch.tensor(fixture, dtype=torch.float32, device='mps', requires_grad=True)
        require(R.is_leaf and R.requires_grad and not A.requires_grad and all(not v.requires_grad for v in P.values()),
            'Sole actual256-coordinate receiver leaf, all parent parameters frozen')
        initial = array(R); optimizer = torch.optim.Adam([R], lr=0., betas=(.9, .999), eps=1e-8, weight_decay=0, foreach=False)
        first = None
        for update, (rank, kind) in enumerate(ROSTER[2:], 1):
            index = result['full_model_calls']; optimizer.zero_grad(set_to_none=True)
            weights = np.load(STORE/f'rows/observer_fit/{rank:04d}/weights.npy', allow_pickle=False)
            card = model.card(rank, P=P, A=A, R=R, device='mps', weights=weights); before = frozen_hash(card)
            require(card.R is R and card.A is A and all(card.P0[k] is value for k, value in P.items()), 'Uncopied persistent R/P/A identities')
            raw, trace = card.run(training=True); served = model._CO.served(raw); rv, sv = array(raw), array(served)
            require(np.isfinite(rv).all() and np.isfinite(sv).all() and (sv >= 0).all()
                and sv[0].tobytes() == references[rank][1][0].tobytes(), 'Finite full output and untouched first Gaussian row')
            forward_d = card.diagnostics()
            loss = scalar(torch, model._CO, raw, cots[rank], card.dtype)
            require(bool(torch.isfinite(loss).detach().cpu()), 'Finite original synthetic scalar')
            loss.backward(); d = domain(card)
            require(all(d[k] == forward_d[k] for k in ('variance_domain', 'selection_cells')),
                'Every actually observed checkpoint replay retains the same trajectory')
            require(R.grad is not None and not R.grad.requires_grad and R.grad.grad_fn is None,
                'Owned detached full receiver gradient')
            gradient = array(R.grad)
            require(gradient.shape == (64, 4) and gradient.dtype == np.float32 and np.isfinite(gradient).all()
                and np.linalg.norm(gradient) > 0, 'Finite nonzero256-coordinate R gradient')
            require(all(not v.requires_grad and v.grad is None for v in P.values()) and A.grad is None
                and not card.coefficients.requires_grad and card.coefficients.grad is None,
                'No parent P/A/cstar gradients')
            require(frozen_hash(card) == before and array(R).tobytes() == initial.tobytes(), 'Parameters and weights unchanged through backward')
            if first is None:
                first = dict(raw=rv.copy(), served=sv.copy(), gradient=gradient.copy(),
                    variance_domain=d['variance_domain'], selection_cells=d['selection_cells'])
            elif rank == 35:
                require(rv.tobytes() == first['raw'].tobytes() and sv.tobytes() == first['served'].tobytes()
                    and gradient.tobytes() == first['gradient'].tobytes()
                    and all(d[k] == first[k] for k in ('variance_domain', 'selection_cells')), 'Exact repeated nonzero-R35 output/domain/gradient')
            rec = dict(call=index, rank=rank, kind=kind, raw=save(f'call_{index:02d}_raw', rv),
                served=save(f'call_{index:02d}_served', sv), gradient=save(f'call_{index:02d}_R_gradient', gradient),
                diagnostics=doc(f'call_{index:02d}_diagnostics', d), gradient_coordinates=256,
                synthetic_scalar=float(loss.detach().cpu()), frozen_hashes=before)
            optimizer.step()
            require(len(optimizer.state) == 1 and R in optimizer.state
                and int(optimizer.state[R]['step'].item()) == update
                and all(bool(torch.isfinite(value).all().cpu()) for value in optimizer.state[R].values()),
                'One finite persistent Adam state, steps1/2/3')
            require(array(R).tobytes() == initial.tobytes() and frozen_hash(card) == before, 'Zero-LR Adam changes no parameter bytes')
            optimizer.zero_grad(set_to_none=True)
            rec.update(passed=True, completed=True, zero_lr_Adam_R_byte_exact=True, Adam_step=update)
            result['calls'].append(rec); doc(f'call_{index:02d}_complete', rec)
            refs = [weakref.ref(card), weakref.ref(raw), weakref.ref(served), weakref.ref(loss)]
            del card, raw, served, trace, loss, weights, rv, sv, d, forward_d, gradient
            retire(index, refs)
        optimizer.state.clear(); optimizer.zero_grad(set_to_none=True)
        parameter_refs = [weakref.ref(R), weakref.ref(A)] + [weakref.ref(value) for value in P.values()]
        del optimizer, P, A, R; gc.collect(); guard()
        require(all(ref() is None for ref in parameter_refs), 'All fixed and trainable parameter owners retired at completion')
        require(result['full_model_calls'] == result['retired_call_count'] == 5
            and [(x['rank'], x['kind']) for x in result['calls']] == list(ROSTER), 'Exact complete fixed five-call roster')
        result['source_proof'] = model.source_proof()
        for group in ('source_bindings', 'input_bindings', 'artifacts'):
            for path, digest in result[group].items():
                require(sha(path) == digest, 'Final unchanged '+group); guard()
        result.update(passed=True, completed=True, execution_integrity_passed=True, sources_unchanged=True,
            hashes_unchanged=True, all_graphs_retired=True, all_parameter_owners_retired=True,
            null_forwards=2, receiver_backwards=3, zero_lr_Adam_calls=3,
            repeat_gradient_byte_exact=True, all16_domains_qualified=True, all47_selectors_qualified=True)
    except BaseException:
        result.update(passed=False, completed=False, error=traceback.format_exc())
    result['elapsed_s'] = time.monotonic()-began
    write(OUT/'result.json', result); digest = sha(OUT/'result.json')
    passed = result['passed'] and time.monotonic()-began < 180
    write(OUT/'completion.json', dict(passed=passed, completed=passed, result_sha256=digest))
    end = sha(OUT/'completion.json'); elapsed = time.monotonic()-began; passed = passed and elapsed < 180
    write(OUT/'finalization.json', dict(passed=passed, completed=passed, result_sha256=digest,
        completion_sha256=end, elapsed_s=elapsed))
    signal.alarm(0)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
