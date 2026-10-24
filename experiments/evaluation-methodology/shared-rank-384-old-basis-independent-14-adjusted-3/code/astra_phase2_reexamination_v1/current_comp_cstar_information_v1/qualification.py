"""Inert fresh34-call cstar qualification; no fit, targets or information stage."""
import argparse
import ast
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
import time
import traceback
import weakref

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
CO, RB = NEW/'current_bd_p_coadaptation_v1', NEW/'current_comp_receiver_birth_v1'
REF = NEW/'current_p1_node_composition_v1/panel_v2/terminal'
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
OUT = HERE/'qualification_v1'
MODEL_SHA = 'a34f7de73914b76aee295ad8dba488bbd6257e93cfa33aa9d58124c2610ed0f1'
PROTOCOL_SHA = 'c3660d88e1a9e90d3ff040b5f9cf191a323b0cf00589d1167fe4ff60bf851682'
SOURCE_PINS = {
    HERE/'model.py': MODEL_SHA, HERE/'PROTOCOL.md': PROTOCOL_SHA,
    CO/'qualify.py': '1a3df2687ed3bef6ffcfc6f690f3cfe8665ad04aae4da94012c608d4871a7b20',
    RB/'qualification_v3.py': 'ce5dfc095fe7fb16f3538ae200a046ca19e8b311324c2035e21914722fe043cf',
}
INPUT_PINS = {
    REF/'result.json': 'de766126360ce24263f548c27a2cb51f60e52107386c4aeb7cf15f9ea20db625',
    REF/'completion.json': '0973d38be5b37c2ff99d5b6243119a22feb5fe591c851ed4aac0aae7346a9073',
    REF/'call_00_rank35.npy': '1df707ad91b1b75b4d50eb6d63d036e0633d1cb9e9a39c3853b0d78ccfe894ea',
    REF/'call_01_rank100.npy': 'b059807fff1e4d1a41766dc70b0b05a0f31b23f819fe33fb144370b0bc1b6721',
    STORE/'manifest.json': '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50',
    STORE/'rows/observer_fit/0035/row_manifest.json': '5067b577620cb6d5dc97c61b62f10e8fb3f98880b99fe9320f1ef8a8ca567809',
    STORE/'rows/observer_fit/0100/row_manifest.json': 'c1f755a84e36825df94871b49ce83884030b34ae8e307930f7ef7d7a6f54a9a9',
    STORE/'rows/observer_fit/0035/weights.npy': '14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2',
    STORE/'rows/observer_fit/0100/weights.npy': 'c4923171c8ddef28396aabd70335dad29141361efc13865ccb660a4a57421ee5',
    CO/'stage_cpu_forward_v1/cotangent_0035.npy': '12be6cfa113058b1a09360b0d5930216d5434789304d9c79952bf346630547f8',
    CO/'stage_cpu_forward_v1/cotangent_0100.npy': '01f4650de1ad6fa4628c8a305f9d444e99df65e4c122b3b282cb2b18fe2464bf',
}
OFFSETS = (('plus025', .025), ('minus025', -.025), ('plus050', .05), ('minus050', -.05))
LOCAL_ROSTER = ((35, 'reference'), (100, 'reference'), (35, 'null'), (100, 'null')) + tuple(
    (r, k) for r in (35, 100) for k, _ in OFFSETS) + (
    (35, 'backward'), (100, 'backward'), (35, 'backward_repeat'), (35, 'jvp'), (100, 'jvp'))
ROSTER = tuple((dev, rank, kind) for dev in ('cpu', 'mps') for rank, kind in LOCAL_ROSTER)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    path = Path(path).resolve()
    require('truthpost' not in path.parts and 'mc_post' not in path.name, 'No truth hashing/reading')
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
    require(spec is not None and spec.loader is not None, 'Pinned source module')
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def source_helpers():
    """Stdlib-only extraction; never execute the old worker or any model."""
    require(all(sha(p) == h for p, h in SOURCE_PINS.items()), 'Frozen numerical source owners')
    old = module(RB/'qualification_v3.py', 'cstar_actual_vector_evidence_owner')
    source, prior = old.derived_source()
    owner = old.parent()
    pins = {**SOURCE_PINS, **old.PINS, **owner.SOURCE_PINS, **owner.INPUT_PINS}
    start = '                            dual_served = m._CO.served(raw)\n'
    end = '                            del dual_served, yp, yt, vector, j64, q64, raw_primal\n'
    require(source.count(start) == source.count(end) == 1, 'Unique qualified actual-vector readout')
    block = source[source.index(start):source.index(end)+len(end)]
    lines = block.splitlines(keepends=True)
    require(all(line.startswith(' '*28) for line in lines), 'Exact readout nesting')
    body = ''.join('    '+line[28:] for line in lines)
    require(''.join(' '*28+line[4:] for line in body.splitlines(keepends=True)) == block,
            'Readout extraction byte inverse, apart from function indentation')
    readout = ('def actual_vector_readout(raw, card, cot, device, rank, scalar_value, array, save, m, np, torch):\n'
               + body + '    return value, tangent, served_jvp\n')
    original = (CO/'qualify.py').read_text()
    functions = [node for node in ast.parse(original).body if isinstance(node, ast.FunctionDef) and node.name == 'scalar']
    require(len(functions) == 1, 'One original CO synthetic scalar')
    scalar = ''.join(original.splitlines(keepends=True)[functions[0].lineno-1:functions[0].end_lineno])
    proof = dict(original_scalar_owner_sha256=SOURCE_PINS[CO/'qualify.py'],
        original_scalar_source_sha256=hashlib.sha256(scalar.encode()).hexdigest(),
        actual_vector_owner_sha256=SOURCE_PINS[RB/'qualification_v3.py'], parent_vector_proof=prior,
        actual_vector_block_sha256=hashlib.sha256(block.encode()).hexdigest(),
        exact_scalar_extraction=True, exact_vector_readout_extraction=True,
        extraction_indentation_inverse_byte_exact=True, original_scalar_not_forward_differentiated=True,
        actual_served_primal_and_tangent_saved=True, old_workers_not_executed=True)
    return scalar, readout, proof, pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    for key in ('source', 'model', 'protocol', 'watcher', 'watch-spec'):
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    began = time.monotonic()
    def expired(*_):
        raise TimeoutError('600s entire fresh CPU-then-MPS qualification')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(600)
    OUT.mkdir(exist_ok=False)
    result = dict(schema='current_comp_cstar_qualification_v1', stage='qualification',
        passed=False, completed=False, execution_integrity_passed=False, numerical_qualification_passed=False,
        source_bindings={}, input_bindings={}, artifacts={}, calls=[], secants={}, gradient_comparisons={},
        full_model_calls=0, retired_call_count=0, tiny_model_calls=0, optimizer_updates=0,
        parameter_updates=0, parameter_proposals=0, training_targets_accessed=False,
        validation_targets_accessed=False, truth_accessed=False, alltruth_accessed=False,
        native_gradient_qualified=False, native_score_measured=False, no_native_gradient_claim=True,
        maximum_sampled_mps_driver_bytes=0, full_model_call_roster=[list(x) for x in ROSTER])
    try:
        def pin(path, digest):
            path = Path(path).resolve()
            require(isinstance(digest, str) and len(digest) == 64 and sha(path) == digest,
                    'Exact reviewed source/input pin: '+str(path))
            group = 'source_bindings' if path.suffix in ('.py', '.md') else 'input_bindings'
            require(str(path) not in result[group] or result[group][str(path)] == digest, 'Consistent file owner')
            result[group][str(path)] = digest
        require(args.model_sha256 == MODEL_SHA, 'Frozen cstar model unchanged')
        for path, digest in {**SOURCE_PINS, **INPUT_PINS,
                Path(__file__).resolve(): args.source_sha256,
                HERE/'QUALIFICATION_PROTOCOL.md': args.protocol_sha256,
                HERE/'watch_qualification.py': args.watcher_sha256,
                HERE/'WATCH_SPEC.md': args.watch_spec_sha256}.items():
            pin(path, digest)
        scalar_source, vector_source, helper_proof, helper_pins = source_helpers()
        for path, digest in helper_pins.items():
            pin(path, digest)
        result['helper_source_proof'] = helper_proof
        m = module(HERE/'model.py', 'cstar_fresh_qualification_model')
        for path, digest in m.authenticate().items():
            pin(path, digest)
        ref, end = read(REF/'result.json'), read(REF/'completion.json')
        require(ref['passed'] and ref['completed'] and ref['hashes_unchanged'] and ref['sources_unchanged']
                and end['passed'] and end['completed'] and end['result_sha256'] == INPUT_PINS[REF/'result.json']
                and ref['candidate_sha256'] == m.COMP_SHA and not ref['truth_accessed']
                and not ref['truth_files_accessed'], 'Sealed exact COMP TRAIN35/100 reference')
        rows = read(STORE/'manifest.json')['roles']['observer_fit']['row_manifests']
        for index, rank in enumerate((35, 100)):
            rowpath = STORE/f'rows/observer_fit/{rank:04d}/row_manifest.json'
            wp = rowpath.parent/'weights.npy'
            entries = [x for x in rows if x['role_rank'] == rank]
            row, fixture = read(rowpath), ref['fixture_bindings'][f'weights_{rank}']
            require(len(entries) == 1 and (STORE/entries[0]['path']).resolve() == rowpath.resolve()
                    and entries[0]['sha256'] == INPUT_PINS[rowpath]
                    and row['role'] == 'observer_fit' and row['role_rank'] == rank
                    and row['files']['weights.npy']['sha256'] == INPUT_PINS[wp]
                    and Path(fixture['path']).resolve() == wp.resolve() and fixture['sha256'] == INPUT_PINS[wp]
                    and ref['trials'][index]['rank'] == rank, 'Identical canonical TRAIN/native input frame')
        for path, digest in ref['source_bindings'].items():
            if Path(path).suffix in ('.py', '.md'):
                pin(path, digest)
        require(Path(sys.executable).resolve() == Path('/Users/matt/anaconda3/bin/python').resolve()
                and sys.dont_write_bytecode, 'Exact AP interpreter and disabled bytecode writes')
        require(sys.flags.optimize == 0 and os.getpriority(os.PRIO_PROCESS, 0) >= 10
                and all(os.environ.get(k) == '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
                and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
                and os.environ.get('PYTORCH_MPS_LOW_WATERMARK_RATIO') == '0.1', 'Fixed nice10/BLAS/MPS environment')
        require(not any(k in sys.modules for k in ('numpy', 'torch', 'flopscope', 'threadpoolctl')),
                'All source admission remains nonnumerical')
        import numpy as np
        import torch
        require(np.__version__ == '1.26.4' and torch.__version__.split('+')[0] == '2.7.0'
                and torch.backends.mps.is_available() and 'threadpoolctl' not in sys.modules, 'Qualified AP runtime')
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        result['versions'] = dict(numpy=np.__version__, torch=torch.__version__)
        m.runtime()
        env = dict(math=math, require=require)
        exec(compile(scalar_source, str(CO/'qualify.py')+':unchanged_scalar', 'exec'), env)
        exec(compile(vector_source, str(RB/'qualification_v3.py')+':unchanged_vector_readout', 'exec'), env)
        scalar, actual_vector_readout = env['scalar'], env['actual_vector_readout']
        mps_active = False

        def array(tensor):
            return tensor.detach().cpu().contiguous().numpy().copy()

        def rawsha(value):
            return hashlib.sha256(value.tobytes()).hexdigest()

        def save(name, value):
            require(np.isfinite(value).all(), 'Finite archived array')
            path = OUT/(name+'.npy')
            with path.open('xb') as stream:
                np.save(stream, value, allow_pickle=False)
            digest = sha(path)
            result['artifacts'][str(path)] = digest
            return dict(path=str(path), sha256=digest, raw_sha256=rawsha(value), shape=list(value.shape), dtype=str(value.dtype))

        def doc(name, value):
            path = OUT/(name+'.json')
            write(path, value)
            result['artifacts'][str(path)] = sha(path)
            return dict(path=str(path), sha256=result['artifacts'][str(path)])

        def guard():
            if mps_active:
                torch.mps.synchronize()
            driver = int(torch.mps.driver_allocated_memory()) if mps_active else 0
            peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            result['maximum_sampled_mps_driver_bytes'] = max(result['maximum_sampled_mps_driver_bytes'], driver)
            require(driver < 16_000_000_000 and peak < 12_000_000_000
                    and time.monotonic()-began < 600, '600s/12GB host/16GB driver guard')
            return dict(mps_driver_bytes=driver, mps_initialized=mps_active, host_peak_rss_bytes=peak)

        cstar0 = np.asarray(m._JOINT._LITERALS['_BD_COEFFICIENT_LITERAL'], dtype=np.float32)
        require(cstar0.shape == (15, 4) and np.isfinite(cstar0).all() and (np.abs(cstar0) < 1).all()
                and cstar0[0].tobytes() == np.zeros(4, np.float32).tobytes(), 'Exact exported60/active56 null')
        direction64 = np.random.Generator(np.random.PCG64DXSM(9309097)).standard_normal((15, 4))
        direction64[0] = 0.
        direction64 /= np.linalg.norm(direction64[1:])
        direction_np = direction64.astype(np.float32)
        payloads = {'null': cstar0.copy()}
        for name, amplitude in OFFSETS:
            payloads[name] = (cstar0.astype(np.float64)+amplitude*direction_np.astype(np.float64)).astype(np.float32)
            require(payloads[name][0].tobytes() == cstar0[0].tobytes()
                    and np.isfinite(payloads[name]).all() and (np.abs(payloads[name]) < 1).all(), 'Valid fixed exported offset')
        cots = {}
        for rank in (35, 100):
            path = CO/f'stage_cpu_forward_v1/cotangent_{rank:04d}.npy'
            value = np.load(path, allow_pickle=False)
            generated = np.random.Generator(np.random.PCG64DXSM(9309098+rank)).standard_normal((16, 1024)).astype(np.float32)
            require(value.shape == (16, 1024) and value.dtype == np.float32
                    and value.tobytes() == generated.tobytes(), 'Exact original synthetic cotangent bytes and seed')
            cots[rank] = value
        result['fixtures'] = dict(direction=save('direction32', direction_np), direction64=save('direction64', direction64),
            payloads={k: save('cstar_'+k, v) for k, v in payloads.items()},
            cotangents={str(r): save(f'cotangent_{r:04d}', v) for r, v in cots.items()})
        result['fixture_seal'] = doc('fixtures_sealed_before_calls', dict(fixtures=result['fixtures'],
            full_model_calls=0, parameter_updates=0, direction_seed=9309097, exact_exported_cstar=True,
            frozen_model_sha256=MODEL_SHA, truth_accessed=False))

        def frozen_hash(card, cstar):
            tensors = dict(card.P0, NODE_A=card.A, cstar=card.coefficients,
                supplied_cstar=cstar, weights=card.w, physical_weights=card.ws, fixed_features=card.fixed)
            return {k: rawsha(array(v)) for k, v in tensors.items()}

        def run_card(device, rank, kind, weights, cot):
            backward, jvp = kind in ('backward', 'backward_repeat'), kind == 'jvp'
            P, A, cstar = m.frozen(device, trainable_cstar=backward)
            fixture = payloads[kind] if kind in payloads else payloads['null']
            if kind in dict(OFFSETS):
                cstar = torch.tensor(fixture, dtype=torch.float32, device=device)
            require(array(cstar).tobytes() == fixture.tobytes(), 'Actual exported cstar supplied without retanh')
            direction = torch.tensor(direction_np, dtype=torch.float32, device=device)
            gradient = tangent = served_jvp = None
            with torch.autograd.forward_ad.dual_level():
                parameter = torch.autograd.forward_ad.make_dual(cstar, direction) if jvp else cstar
                card = (m.reference_card(rank, P=P, A=A, device=device, weights=weights) if kind == 'reference'
                        else m.card(rank, P=P, A=A, cstar=parameter, device=device, weights=weights))
                require(card.dtype == (torch.float64 if device == 'cpu' else torch.float32)
                        and card.n == 1024 and card.depth == 16, 'Unchanged full physical backend')
                require(kind == 'reference' or card.coefficients is parameter, 'Actual leaf/dual reaches original NODE')
                before = frozen_hash(card, cstar)
                require(before['weights'] == rawsha(weights) and before['cstar'] == rawsha(fixture), 'Card input bytes')
                def scalar_value(raw):
                    return scalar(torch, m._CO, raw, cot, card.dtype)
                if backward:
                    require(cstar.is_leaf and cstar.requires_grad, 'One actual live cstar leaf')
                    raw, trace = card.run(training=True)
                    forward_d = card.diagnostics()
                    value = scalar_value(raw)
                    require(bool(torch.isfinite(value).detach().cpu()), 'Finite original synthetic scalar')
                    value.backward()
                    require(cstar.grad is not None and cstar.grad.grad_fn is None
                            and not cstar.grad.requires_grad, 'Owned detached reverse gradient')
                    gradient = array(cstar.grad)
                    require(gradient.dtype == np.float32 and gradient.shape == (15, 4)
                            and np.isfinite(gradient).all() and (gradient[0] == 0).all()
                            and np.linalg.norm(gradient.astype(np.float64)) > 1e-12, 'Finite nontrivial full60/active56 gradient')
                else:
                    with torch.no_grad():
                        raw, trace = card.run(training=False)
                        if jvp:
                            value, tangent, served_jvp = actual_vector_readout(raw, card, cot, device, rank,
                                scalar_value, array, save, m, np, torch)
                        else:
                            value = scalar_value(raw)
                require(trace == [], 'No retained capture or extra model calls')
                primal = torch.autograd.forward_ad.unpack_dual(raw).primal
                served = m._CO.served(primal)
                raw_np, served_np = array(primal), array(served)
                require(raw_np.shape == served_np.shape == (16, 1024) and served_np.dtype == np.float32
                        and np.isfinite(raw_np).all() and np.isfinite(served_np).all() and (served_np >= 0).all(), 'Finite full outputs')
                diagnostics = m._CO.check_replay(card)
                require(set(diagnostics['node_head_calls']) == {str(l) for l in range(1, 15)}
                        and all(v > 0 for v in diagnostics['node_head_calls'].values()), 'All14 original NODE keys with actual visits')
                if backward:
                    require(all(forward_d[k] == diagnostics[k] for k in ('variance_domain', 'selection_cells')),
                            'Original forward and every checkpoint replay agree')
                    del forward_d
                require(before == frozen_hash(card, cstar) and all(v.grad is None and not v.requires_grad for v in P.values())
                        and A.grad is None and not A.requires_grad, 'Only cstar differentiable; every input/parent byte unchanged')
                answer = dict(raw=raw_np, served=served_np, gradient=gradient, tangent=tangent, served_jvp=served_jvp,
                    scalar=float(value.detach().cpu()), scalar_raw_sha256=rawsha(array(value)),
                    diagnostics=diagnostics, frozen_hashes=before, live_resources=guard())
                refs = [weakref.ref(v) for v in (card, raw, primal, served, value, cstar, parameter, direction,
                    A, card.w, card.ws, card.fixed, *P.values())]
                if backward:
                    refs.append(weakref.ref(cstar.grad))
                del scalar_value, card, raw, primal, served, value, cstar, parameter, direction, P, A, trace
            gc.collect()
            if device == 'mps':
                torch.mps.synchronize()
                torch.mps.empty_cache()
            gc.collect()
            require(all(ref() is None for ref in refs), 'All card/output/parameter owners retired')
            answer['resources'] = guard()
            return answer

        def exact_state(a, b, gradient=False):
            keys = ('raw', 'served', 'gradient') if gradient else ('raw', 'served')
            require(all(a[k].tobytes() == b[k].tobytes() for k in keys)
                    and a['scalar_raw_sha256'] == b['scalar_raw_sha256']
                    and all(a['diagnostics'][k] == b['diagnostics'][k] for k in ('variance_domain', 'selection_cells')),
                    'Same-backend original output/scalar/domain/selector byte identity')

        previous = {}
        for index, (device, rank, kind) in enumerate(ROSTER):
            if device == 'mps' and not mps_active:
                require(result['retired_call_count'] == 17, 'All CPU calls retired before MPS begins')
                recommended = int(torch.mps.recommended_max_memory())
                require(recommended > 0, 'Positive MPS recommended capacity')
                fraction = min(1., 16_000_000_000/recommended)
                torch.mps.set_per_process_memory_fraction(fraction)
                result['mps_allocator'] = dict(configured=True, api='torch.mps.set_per_process_memory_fraction',
                    recommended_max_memory_bytes=recommended, configured_fraction=fraction, low_watermark_ratio='0.1')
                mps_active = True
            guard()
            wp = STORE/f'rows/observer_fit/{rank:04d}/weights.npy'
            require(sha(wp) == INPUT_PINS[wp], 'Same immutable weights before each call')
            weights = np.load(wp, allow_pickle=False)
            require(weights.shape == (16, 1024, 1024) and weights.dtype == np.float32 and np.isfinite(weights).all(), 'Canonical full weights')
            attempt = run_card(device, rank, kind, weights, cots[rank])
            del weights
            rec = {k: v for k, v in attempt.items() if k not in ('raw', 'served', 'gradient', 'diagnostics')}
            rec.update(call=index, device=device, rank=rank, kind=kind, graph_retired=True,
                raw=save(f'call_{index:02d}_raw', attempt['raw']), served=save(f'call_{index:02d}_served', attempt['served']),
                diagnostics=doc(f'call_{index:02d}_diagnostics', attempt['diagnostics']))
            if attempt['gradient'] is not None:
                rec['gradient'] = save(f'call_{index:02d}_gradient', attempt['gradient'])
            doc(f'call_{index:02d}_attempt', rec)
            if kind == 'null':
                exact_state(previous[(device, rank, 'reference')], attempt)
                if device == 'mps':
                    native = np.load(REF/('call_00_rank35.npy' if rank == 35 else 'call_01_rank100.npy'), allow_pickle=False)
                    require(native.shape == (16, 1024) and native.dtype == np.float32 and np.isfinite(native).all(), 'Sealed COMP native output contract')
                    rec['native_zero_max_abs'] = float(np.max(np.abs(native.astype(np.float64)-attempt['served'].astype(np.float64))))
                    require(rec['native_zero_max_abs'] <= 1e-5, 'Current COMP native null fidelity')
                    del native
            if device == 'mps' and (kind == 'null' or kind in dict(OFFSETS)):
                cpu = previous[('cpu', rank, kind)]
                rec['cpu_mps_output_max_abs'] = {k: float(np.max(np.abs(cpu[k].astype(np.float64)-attempt[k].astype(np.float64))))
                                                for k in ('raw', 'served')}
                require(max(rec['cpu_mps_output_max_abs'].values()) <= 1e-5, 'Strict all16 CPU/MPS output gate')
            if kind in ('minus025', 'minus050'):
                amplitude = .025 if kind == 'minus025' else .05
                plus = previous[(device, rank, kind.replace('minus', 'plus'))]['scalar']
                secant = (plus-attempt['scalar'])/(2*amplitude)
                require(math.isfinite(secant), 'Finite actual-exported scalar secant')
                key = f'{rank}:{amplitude}'
                item = dict(device=device, rank=rank, amplitude=amplitude, symmetric_secant=secant)
                if device == 'mps':
                    cpu_secant = result['secants']['cpu:'+key]['symmetric_secant']
                    item.update(cpu_secant=cpu_secant, relative_backend_error=abs(cpu_secant-secant)/max(abs(cpu_secant), abs(secant), 1e-10))
                result['secants'][device+':'+key] = item
                doc(f'call_{index:02d}_secant', item)
                require(device == 'cpu' or item['relative_backend_error'] <= .05, 'Original strict scalar step gate; no rescue')
            if kind in ('backward', 'backward_repeat'):
                exact_state(previous[(device, rank, 'null')], attempt)
                if kind == 'backward_repeat':
                    old = previous[(device, rank, 'backward')]
                    exact_state(old, attempt, gradient=True)
                    require(old['diagnostics'] == attempt['diagnostics'] and old['frozen_hashes'] == attempt['frozen_hashes'], 'Full repeated reverse diagnostics and frozen hashes')
                elif device == 'mps':
                    a = previous[('cpu', rank, 'backward')]['gradient'].astype(np.float64)
                    b = attempt['gradient'].astype(np.float64)
                    na, nb, error = float(np.linalg.norm(a)), float(np.linalg.norm(b)), float(np.linalg.norm(a-b))
                    groups = {}
                    for layer in range(1, 15):
                        la, lb, le = float(np.linalg.norm(a[layer])), float(np.linalg.norm(b[layer])), float(np.linalg.norm(a[layer]-b[layer]))
                        substantial = la >= 1e-6*na
                        relative = le/max(la, lb, 1e-300)
                        groups[str(layer)] = dict(cpu_norm=la, mps_norm=lb, error_norm=le,
                            substantial_cpu_group=substantial, relative_error=relative,
                            passed=bool(relative <= .05 if substantial else le <= 5e-8*na))
                    comparison = dict(cpu_norm=na, mps_norm=nb, error_norm=error, relative_error=error/max(na, nb),
                        groups=groups, passed=bool(error/max(na, nb) <= .05 and all(x['passed'] for x in groups.values())))
                    result['gradient_comparisons'][str(rank)] = comparison
                    doc(f'call_{index:02d}_gradient_comparison', comparison)
                    require(comparison['passed'], 'Strict full/substantial-group56-coordinate reverse gate')
            if kind == 'jvp':
                back = previous[(device, rank, 'backward')]
                exact_state(back, attempt)
                reverse = float(np.sum(back['gradient'].astype(np.float64)*direction_np.astype(np.float64)))
                tangent = attempt['tangent']
                require(math.isfinite(reverse) and math.isfinite(tangent) and max(abs(reverse), abs(tangent)) > 1e-10,
                        'Finite nontrivial actual directional response')
                identity = dict(reverse_dot=reverse, forward_ad=tangent, absolute_error=abs(reverse-tangent),
                    tolerance=1e-10+1e-4*max(abs(reverse), abs(tangent)), actual_served_jvp=attempt['served_jvp'])
                doc(f'call_{index:02d}_directional_identity', identity)
                require(identity['absolute_error'] <= identity['tolerance'], 'Original strict actual-vector FAD/RAD identity')
                rec['ad_identity'] = identity
            previous[(device, rank, kind)] = attempt
            rec.update(passed=True, completed=True)
            doc(f'call_{index:02d}_complete', rec)
            result['calls'].append(rec)
            result['full_model_calls'] += 1
            result['retired_call_count'] += 1
            gc.collect()
            resources = guard()
            print(json.dumps(dict(event='retired', call=index, device=device, rank=rank, kind=kind,
                                  **resources)), flush=True)
        require(len(result['calls']) == result['full_model_calls'] == result['retired_call_count'] == 34,
                'Exactly34 fully retired calls; no extras')
        result['model_source_proof'] = m.source_proof()
        for group in ('source_bindings', 'input_bindings', 'artifacts'):
            require(all(sha(path) == digest for path, digest in result[group].items()), 'Every bound byte unchanged: '+group)
        del previous, attempt, cots, payloads
        gc.collect()
        guard()
        result.update(passed=True, completed=True, execution_integrity_passed=True, numerical_qualification_passed=True,
            sources_unchanged=True, hashes_unchanged=True, all_graphs_retired=True,
            all16_domains_qualified=True, all47_selectors_qualified=True, same_backend_null_byte_exact=True,
            repeat_gradient_byte_exact=True, strict_secant_gates_passed=True, strict_gradient_gates_passed=True,
            served_vector_ad_identity_passed=True, frozen_P_A_cstar_weights_passed=True,
            reference_forwards=4, null_forwards=4, offset_forwards=16, backwards=6, served_vector_jvps=4,
            allocated_cstar_coordinates=60, active_cstar_coordinates=56, cpu_calls=17, mps_calls=17)
    except BaseException:
        result.update(passed=False, completed=False, execution_integrity_passed=False,
                      numerical_qualification_passed=False, error=traceback.format_exc())
    result['elapsed_s'] = time.monotonic()-began
    write(OUT/'result.json', result)
    result_digest = sha(OUT/'result.json')
    write(OUT/'completion.json', dict(passed=result['passed'], completed=result['completed'], result_sha256=result_digest))
    completion_digest = sha(OUT/'completion.json')
    elapsed = time.monotonic()-began
    ok = bool(result['passed'] and elapsed < 600)
    write(OUT/'finalization.json', dict(passed=ok, completed=ok, result_sha256=result_digest,
        completion_sha256=completion_digest, elapsed_s=elapsed))
    sha(OUT/'finalization.json')
    within = time.monotonic()-began < 600
    signal.alarm(0)
    return 0 if ok and within else 1


if __name__ == '__main__':
    raise SystemExit(main())
