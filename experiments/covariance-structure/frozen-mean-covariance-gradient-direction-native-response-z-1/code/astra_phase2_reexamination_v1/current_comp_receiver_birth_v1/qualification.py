"""Target-free qualification only; inert without --run and reviewed hashes."""
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
COMP = NEW/'current_p1_node_composition_v1'
CO = NEW/'current_bd_p_coadaptation_v1'
REF = COMP/'panel_v2/terminal'
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
OUT = HERE/'qualification_v1'
SOURCE_PINS = {
    CO/'qualify.py': '1a3df2687ed3bef6ffcfc6f690f3cfe8665ad04aae4da94012c608d4871a7b20',
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
ROSTER = ((35, 'zero'), (100, 'zero'), (35, 'backward'), (35, 'jvp'),
          (100, 'backward'), (100, 'jvp'), (35, 'backward_repeat'))


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def read(path):
    return json.loads(Path(path).read_text())


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    for key in ('source', 'model', 'protocol', 'watcher'):
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    began = time.monotonic()
    def alarm(*_):
        raise TimeoutError('Prospective180s qualification deadline')
    signal.signal(signal.SIGALRM, alarm); signal.alarm(180)
    require(not OUT.exists(), 'Exclusive qualification directory; no resume')
    OUT.mkdir()
    result = dict(schema='current_comp_receiver_birth_qualification_v1', passed=False,
                  completed=False, source_bindings={}, input_bindings={}, artifacts={},
                  calls=[], training_targets_accessed=False, validation_targets_accessed=False,
                  truth_accessed=False, parameter_updates=0, native_gradient_qualified=False,
                  no_native_gradient_claim=True, full_model_call_roster=[list(x) for x in ROSTER],
                  maximum_sampled_mps_driver_bytes=0)
    try:
        source_pins = dict(SOURCE_PINS)
        source_pins.update({Path(__file__).resolve(): args.source_sha256,
                           HERE/'model.py': args.model_sha256, HERE/'PROTOCOL.md': args.protocol_sha256,
                           HERE/'watch_qualification.py': args.watcher_sha256})
        for path, digest in source_pins.items():
            require(isinstance(digest, str) and len(digest) == 64 and sha(path) == digest,
                    'Reviewed source pin: '+str(path))
        result['source_bindings'] = {str(p): h for p, h in source_pins.items()}
        m = load_module(HERE/'model.py', 'receiver_qualified_model')
        for path, digest in m.authenticate().items():
            result['source_bindings' if Path(path).suffix in ('.py', '.md') else 'input_bindings'][path] = digest
        for path, digest in INPUT_PINS.items():
            require(sha(path) == digest, 'Fixed target-blind input pin: '+str(path))
            result['input_bindings'][str(path)] = digest
        ref, end = read(REF/'result.json'), read(REF/'completion.json')
        require(ref['passed'] and ref['completed'] and ref['hashes_unchanged'] and ref['sources_unchanged']
                and end['passed'] and end['completed'] and end['result_sha256'] == INPUT_PINS[REF/'result.json']
                and ref['candidate_sha256'] == m.COMP_SHA and not ref['truth_accessed']
                and not ref['truth_files_accessed'], 'Complete current-COMP target-blind reference')
        for index, rank in enumerate((35, 100)):
            trial = ref['trials'][index]
            wp = STORE/f'rows/observer_fit/{rank:04d}/weights.npy'
            fixtures = [x for x in ref['fixture_bindings'].values() if x['path'] == str(wp)]
            require(trial['rank'] == rank and len(fixtures) == 1 and fixtures[0]['sha256'] == INPUT_PINS[wp],
                    'Native and Torch references own the identical canonical TRAIN weights')
        for path, digest in ref['source_bindings'].items():
            if Path(path).suffix in ('.py', '.md'):
                require(sha(path) == digest, 'Archived reference source owner')
                result['source_bindings'][path] = digest
        require(sys.flags.optimize == 0 and os.getpriority(os.PRIO_PROCESS, 0) >= 10,
                'Assertions enabled and inherited nice10')
        require(all(os.environ.get(k) == '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')), 'Five BLAS thread limits')
        require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
                and os.environ.get('PYTORCH_MPS_LOW_WATERMARK_RATIO') == '0.1', 'No MPS fallback; low watermark')
        import numpy as np
        import torch
        require(np.__version__ == '1.26.4' and torch.__version__.split('+')[0] == '2.7.0', 'Qualified AP versions')
        require('threadpoolctl' not in sys.modules and torch.backends.mps.is_available(), 'One available MPS; no threadpoolctl')
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        recommended = int(torch.mps.recommended_max_memory())
        fraction = min(1., 16_000_000_000 / recommended)
        torch.mps.set_per_process_memory_fraction(fraction)
        result['mps_allocator'] = dict(configured=True, api='torch.mps.set_per_process_memory_fraction',
            recommended_max_memory_bytes=recommended, configured_fraction=fraction, low_watermark_ratio='0.1')
        result['versions'] = dict(numpy=np.__version__, torch=torch.__version__)
        m.runtime()
        nodes = [n for n in ast.parse((CO/'qualify.py').read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name == 'scalar']
        require(len(nodes) == 1, 'Unchanged original scalar extraction')
        env = {'math': math}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(CO/'qualify.py')+':scalar', 'exec'), env)
        original_scalar = env['scalar']

        def guard():
            torch.mps.synchronize()
            driver = int(torch.mps.driver_allocated_memory())
            result['maximum_sampled_mps_driver_bytes'] = max(result['maximum_sampled_mps_driver_bytes'], driver)
            require(driver < 16_000_000_000 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 12_000_000_000
                    and time.monotonic()-began < 180, 'Internal time/host/driver bound')
            return dict(mps_driver_bytes=driver, host_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

        def array(tensor):
            return tensor.detach().cpu().contiguous().numpy().copy()

        def save(name, value):
            path = OUT/(name+'.npy')
            with path.open('xb') as stream:
                np.save(stream, value, allow_pickle=False)
            digest = sha(path); result['artifacts'][str(path)] = digest
            return dict(path=str(path), sha256=digest, shape=list(value.shape), dtype=str(value.dtype),
                        raw_sha256=hashlib.sha256(value.tobytes()).hexdigest())

        def frozen_hash(card):
            items = dict(card.P0, NODE_A=card.A, cstar=card.coefficients, receiver_Theta=card.Theta,
                         frozen_weights=card.w)
            return {key: hashlib.sha256(array(value).tobytes()).hexdigest() for key, value in items.items()}

        def diagnostics(card):
            d = card.diagnostics()
            depth = card.depth
            expected = {f'{l}:{s}' for l in range(depth) for s in range(2 if l == depth-1 else 3)}
            require(set(d['variance_domain']) == {str(l) for l in range(depth)}
                    and set(d['selection_cells']) == expected, 'Every actual domain and selector')
            require(all(v['finite'] and v['below_floor_count'] == 0 and v['minimum'] >= 1e-10
                        for v in d['variance_domain'].values()), 'Unfloored variance domain')
            if card._training:
                require(max(d['variance_visits'].values()) >= 2 and max(d['selection_visits'].values()) >= 2,
                        'Actual non-reentrant checkpoint replay')
            return d

        p, q = np.arange(1, 65, dtype=np.float64)[:, None], np.arange(1, 65, dtype=np.float64)[None, :]
        theta_np = (.05*np.sin(.37*p+.19*q)).astype(np.float32)
        direction64 = np.cos(.23*p-.41*q)
        direction_np = (64*direction64/np.linalg.norm(direction64)).astype(np.float32)
        result['fixtures'] = dict(theta=save('fixed_nonzero_theta', theta_np), direction=save('fixed_direction', direction_np))

        def run_card(rank, mode, device, weights, cot, *, enabled=True, zero=False):
            P, A = m.frozen(device)
            theta = torch.tensor(np.zeros((64, 64), np.float32) if zero else theta_np,
                                 device=device, dtype=torch.float32, requires_grad=mode == 'backward')
            direction = torch.tensor(direction_np, device=device, dtype=torch.float32)
            tangent = gradient = None
            with torch.autograd.forward_ad.dual_level():
                parameter = torch.autograd.forward_ad.make_dual(theta, direction) if mode == 'jvp' else theta
                card = m.card(rank, parameter, P=P, A=A, device=device, weights=weights, enabled=enabled)
                before = frozen_hash(card)
                def scalar_value(raw):
                    if card.depth == 16:
                        return original_scalar(torch, m._CO, raw, cot, card.dtype)
                    y = m._CO.served(raw).to(card.dtype)
                    q = torch.as_tensor(cot, dtype=card.dtype, device=y.device)
                    return (y[-1]*q[-1]).sum()/math.sqrt(card.n) + .05*(y[:-1]*q[:-1]).sum()/math.sqrt((card.depth-1)*card.n)
                if mode == 'backward':
                    raw, _ = card.run(training=True)
                    value = scalar_value(raw)
                    value.backward()
                    require(theta.grad is not None and bool(torch.isfinite(theta.grad).all().cpu()), 'Connected finite Theta gradient')
                    gradient = array(theta.grad)
                else:
                    with torch.no_grad():
                        raw, _ = card.run(training=False)
                        value = scalar_value(raw)
                    if mode == 'jvp':
                        _, tv = torch.autograd.forward_ad.unpack_dual(value)
                        require(tv is not None, 'MPS forward-AD scalar tangent is connected')
                        tangent = float(tv.detach().cpu())
                primal = torch.autograd.forward_ad.unpack_dual(raw).primal
                served = m._CO.served(primal)
                raw_np, served_np = array(primal), array(served)
                require(np.isfinite(raw_np).all() and np.isfinite(served_np).all() and (served_np >= 0).all(), 'Finite complete output')
                d = diagnostics(card)
                require(before == frozen_hash(card), 'P/NODEA/cstar/Theta bytes unchanged')
                require(all(x.grad is None for x in P.values()) and A.grad is None, 'Only Theta is differentiable')
                answer = dict(raw=raw_np, served=served_np, gradient=gradient, tangent=tangent,
                              scalar=float(torch.autograd.forward_ad.unpack_dual(value).primal.detach().cpu()),
                              diagnostics=d, frozen_hashes=before)
                refs = [weakref.ref(card), weakref.ref(raw), weakref.ref(theta)]
                del scalar_value, card, raw, primal, served, value, theta, parameter, direction, P, A
                if mode == 'jvp':
                    del tv, _
            gc.collect(); torch.mps.synchronize(); torch.mps.empty_cache(); gc.collect()
            require(all(ref() is None for ref in refs), 'All card/output/parameter graph owners retired')
            answer['resources'] = guard()
            return answer

        def identities(back, forward):
            require(back['raw'].tobytes() == forward['raw'].tobytes()
                    and back['served'].tobytes() == forward['served'].tobytes(), 'Reverse/JVP primal exact bytes')
            for key in ('selection_cells', 'variance_domain'):
                require(back['diagnostics'][key] == forward['diagnostics'][key], 'Reverse/JVP state cells exact')
            grad = back['gradient'].astype(np.float64)
            require(np.linalg.norm(grad) > 1e-12, 'Nontrivial full Theta gradient')
            reverse = float(np.sum(grad*direction_np.astype(np.float64)))
            tangent = forward['tangent']
            require(math.isfinite(tangent) and max(abs(reverse), abs(tangent)) > 1e-10, 'Nontrivial directional scalar')
            error = abs(reverse-tangent)
            require(error <= 1e-10+1e-4*max(abs(reverse), abs(tangent)), 'Forward-AD/reverse-dot identity')
            return dict(reverse_dot=reverse, forward_ad=tangent, absolute_error=error)

        # Independent n=8 cubic contraction, including an exactly zero column.
        B0 = torch.tensor(np.sin(np.arange(64).reshape(8, 8)+.3)*.2, dtype=torch.float32)
        B0 = m._BASE.diagonal(B0, torch.zeros(8)); B0[:, 7] = 0
        H = torch.tensor(np.sin(np.arange(512).reshape(8, 64)*.13), dtype=torch.float32)
        th = torch.tensor(theta_np)
        B1, F, D = m.geometry(B0, H, th, torch.float64)
        t = torch.linspace(.2, .9, 8, dtype=torch.float64)
        s = -torch.linspace(.1, .8, 8, dtype=torch.float64)
        a = torch.linspace(.1, .4, 8, dtype=torch.float64)
        alive, births = torch.ones(8, dtype=torch.float64), torch.ones(8, dtype=torch.float64)
        alive[-1] = 0; births[-1] = 0
        def explicit_plane(B):
            bp = m._BASE.diagonal(B.double(), (t*alive).float().double())
            bm = m._BASE.diagonal(B.double(), (s*alive).float().double())
            lam = a.float().double()*births
            total = (bp*bp)@(bp*lam).T-(bm*bm)@(bm*lam).T
            return m._BASE.diagonal(total, torch.zeros(8, dtype=torch.float64))
        direct = explicit_plane(B1)-explicit_plane(B0)
        repair = m.extra_plane(B0, D, t, s, a, alive, births, torch.float64)
        repair_error = float((direct-repair).abs().max())
        orthogonal = float((B0.double()*F).sum(0).abs().max())
        result['tiny_algebra'] = dict(repair_max_abs=repair_error, perpendicular_abs=orthogonal,
                                      stored_perpendicular_abs=float((B0.double()*D).sum(0).abs().max()))
        write(OUT/'tiny_plane_attempt.json', result['tiny_algebra'])
        result['artifacts'][str(OUT/'tiny_plane_attempt.json')] = sha(OUT/'tiny_plane_attempt.json')
        require(repair_error <= 1e-12+1e-11*float(direct.abs().max()) and orthogonal <= 1e-12,
                'Independent stored CP repair and real projection')
        null, _, _ = m.geometry(B0, H, torch.zeros((64, 64)), torch.float64)
        require(array(null).tobytes() == array(B0).tobytes(), 'Exact zero geometry, including signed zero')
        require(bool((B1[B0 == 0] == B0[B0 == 0]).all()), 'Exact zero support')
        del B0, B1, F, D, H, th, t, s, a, alive, births, direct, repair, null
        tiny_w = (.7*np.eye(8)[None, :, :]+.04*np.sin(np.arange(1, 4)[:, None, None]
                  *np.arange(1, 9)[None, :, None]*np.arange(1, 9)[None, None, :])).astype(np.float32)
        tiny_cot = np.sin(np.arange(24, dtype=np.float64).reshape(3, 8)*.31).astype(np.float32)
        tiny = {}
        for device in ('cpu', 'mps'):
            bypass = run_card(35, 'forward', device, tiny_w, tiny_cot, enabled=False, zero=True)
            zero = run_card(35, 'forward', device, tiny_w, tiny_cot, zero=True)
            require(bypass['raw'].tobytes() == zero['raw'].tobytes()
                    and bypass['served'].tobytes() == zero['served'].tobytes(), 'Tiny whole-card null byte identity')
            back = run_card(35, 'backward', device, tiny_w, tiny_cot)
            jvp = run_card(35, 'jvp', device, tiny_w, tiny_cot)
            tiny[device] = back
            result['tiny_algebra'][device+'_ad'] = identities(back, jvp)
        tiny_error = float(np.max(np.abs(tiny['cpu']['served'].astype(np.float64)-tiny['mps']['served'])))
        gcpu, gmps = tiny['cpu']['gradient'].astype(np.float64), tiny['mps']['gradient'].astype(np.float64)
        relative = float(np.linalg.norm(gcpu-gmps)/max(np.linalg.norm(gcpu), 1e-30))
        result['tiny_algebra'].update(cpu_mps_max_abs=tiny_error, cpu_mps_gradient_relative_l2=relative)
        write(OUT/'tiny_attempt.json', result['tiny_algebra']); result['artifacts'][str(OUT/'tiny_attempt.json')] = sha(OUT/'tiny_attempt.json')
        require(tiny_error <= 1e-5 and relative <= .05, 'Tiny cross-backend primal/gradient gates')
        del tiny, bypass, zero, back, jvp, gcpu, gmps
        gc.collect(); guard()
        print(json.dumps(dict(event='retired', call=0)), flush=True)

        previous = {}
        for call, (rank, kind) in enumerate(ROSTER, 1):
            wp = STORE/f'rows/observer_fit/{rank:04d}/weights.npy'
            with wp.open('rb') as stream:
                weights = np.load(stream, allow_pickle=False)
            cot = np.load(CO/f'stage_cpu_forward_v1/cotangent_{rank:04d}.npy', allow_pickle=False)
            require(weights.shape == (16, 1024, 1024) and weights.dtype == np.float32
                    and cot.shape == (16, 1024) and cot.dtype == np.float32, 'Exact full-card input contracts')
            mode = 'forward' if kind == 'zero' else ('jvp' if kind == 'jvp' else 'backward')
            attempt = run_card(rank, mode, 'mps', weights, cot, zero=kind == 'zero')
            metadata = {k: v for k, v in attempt.items() if k not in ('raw', 'served', 'gradient')}
            metadata.update(call=call, rank=rank, kind=kind, graph_retired=True,
                            raw=save(f'call_{call:02d}_raw', attempt['raw']),
                            served=save(f'call_{call:02d}_served', attempt['served']))
            if attempt['gradient'] is not None:
                metadata['gradient'] = save(f'call_{call:02d}_gradient', attempt['gradient'])
            write(OUT/f'call_{call:02d}_attempt.json', metadata)
            result['artifacts'][str(OUT/f'call_{call:02d}_attempt.json')] = sha(OUT/f'call_{call:02d}_attempt.json')
            if kind == 'zero':
                native = np.load(REF/('call_00_rank35.npy' if rank == 35 else 'call_01_rank100.npy'), allow_pickle=False)
                error = float(np.max(np.abs(native.astype(np.float64)-attempt['served'].astype(np.float64))))
                metadata['native_zero_max_abs'] = error
                require(error <= 1e-5, 'Current-COMP native zero primal fidelity')
            elif kind == 'jvp':
                metadata['ad_identity'] = identities(previous[rank], attempt)
            elif kind == 'backward_repeat':
                old = previous[rank]
                require(all(old[key].tobytes() == attempt[key].tobytes() for key in ('raw', 'served', 'gradient'))
                        and old['diagnostics'] == attempt['diagnostics']
                        and old['frozen_hashes'] == attempt['frozen_hashes'], 'Exact35 reverse repeat')
            else:
                previous[rank] = attempt
            metadata.update(passed=True, completed=True)
            result['calls'].append(metadata)
            del weights, cot, attempt
            gc.collect(); guard()
            print(json.dumps(dict(event='retired', call=call)), flush=True)
        require(len(result['calls']) == 7, 'Exact seven full-card calls')
        for group in ('source_bindings', 'input_bindings', 'artifacts'):
            require(all(sha(path) == digest for path, digest in result[group].items()), 'Every bound byte unchanged: '+group)
        guard()
        result.update(passed=True, completed=True, execution_integrity_passed=True,
                      hashes_unchanged=True, sources_unchanged=True, all_graphs_retired=True,
                      retired_call_count=8, full_model_calls=7, tiny_model_calls=8)
    except BaseException:
        result.update(passed=False, completed=False, error=traceback.format_exc())
    result['elapsed_s'] = time.monotonic()-began
    write(OUT/'result.json', result)
    result_digest = sha(OUT/'result.json')
    write(OUT/'completion.json', dict(passed=result['passed'], completed=result['completed'], result_sha256=result_digest))
    completion_digest = sha(OUT/'completion.json')
    within = time.monotonic()-began < 180
    write(OUT/'finalization.json', dict(passed=bool(result['passed'] and within), completed=bool(result['completed'] and within),
          result_sha256=result_digest, completion_sha256=completion_digest, elapsed_s=time.monotonic()-began))
    signal.alarm(0)
    return 0 if result['passed'] and within else 1


if __name__ == '__main__':
    raise SystemExit(main())
