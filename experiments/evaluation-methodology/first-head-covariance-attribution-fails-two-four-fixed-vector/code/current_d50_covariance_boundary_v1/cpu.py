"""Inert common-input covariance circuits and two original NG CPU steps.

No run(), backward(), target, fit, native call, or arithmetic replacement.
The exact NG step return frames are observed only to retain existing locals.
The shared stdlib admission/receipt helpers are also used by judge.py.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
CO = NEW / 'current_bd_p_coadaptation_v1'
NG = NEW / 'current_d50_native_g0_forward_v1'
SCHEMA = 'current_d50_covariance_boundary_v1'
PROTOCOL_SHA = '26b60da71938e72025c2d2a2ca9634e65aa8bd4f0e3e8827333d67b356848179'
HELPER_SHA = '1a3df2687ed3bef6ffcfc6f690f3cfe8665ad04aae4da94012c608d4871a7b20'
MODEL_SHA = '00f7a37825577ef083b8a5eac1f230183bcbae7f9b1f2a68739e4f37d52b7049'
PROBE_SHA = '6f3dc66c43d25b96d6bf5c7620231e115d8527e906ae00e9741bb3e55c5bd607'
NG_RESULT_SHA = '7794394632f3b96866d1e539753700699c53e37734aa5d2455b9f62b6c7ef1b3'
NG_COMPLETION_SHA = '1be6c2e43a9103d91b6b34d4fa0eebabad3e29b79a67a1ce12faef96eb993f75'
RANKS = (35, 100)
CASES = (('null', 0.), ('plus025', .025), ('minus025', -.025),
         ('plus050', .05), ('minus050', -.05))
BLAS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


class Stage:
    """Stdlib-only admission; numerical imports occur explicitly in each main."""
    def __init__(self, source, source_sha, protocol_sha, stage, directory):
        sys.dont_write_bytecode = True
        helper = CO / 'qualify.py'
        if hashlib.sha256(helper.read_bytes()).hexdigest() != HELPER_SHA:
            raise RuntimeError('Frozen scalar/IO helper changed')
        self.h = module(helper, 'covariance_boundary_original_helpers')
        h = self.h
        h.bind(source, source_sha)
        h.require(protocol_sha == PROTOCOL_SHA, 'Frozen prospective protocol')
        h.bind(HERE / 'PROTOCOL.md', PROTOCOL_SHA)
        self.folder = HERE / directory
        h.require(not self.folder.exists(), 'Exclusive fixed stage')
        self.folder.mkdir()
        self.started = time.monotonic()
        self.bindings = {}
        self.result = dict(schema=SCHEMA, stage=stage, passed=False, completed=False,
            hashes_unchanged=False, sources_unchanged=False, truth_accessed=False,
            no_fit=True, parameter_updates=0, gradient_calls=0, full_card_calls=0,
            MPS_used=False, old_BR_and_NG_remain_failed=True,
            source_bindings={}, input_bindings={}, artifacts={}, ranks={},
            sequence=[], started_utc=datetime.now(timezone.utc).isoformat())
        self.pin(source, source_sha, source=True)
        self.pin(HERE / 'PROTOCOL.md', PROTOCOL_SHA, source=True)
        self.pin(helper, HELPER_SHA, source=True)
        h.write(self.folder / 'started.json', self.result)
        self.old_handler = signal.signal(signal.SIGALRM, self.guard)
        signal.setitimer(signal.ITIMER_REAL, .5, .5)

    def pin(self, path, digest=None, source=False):
        item = self.h.bind(path, digest)
        path = item['path']
        if path in self.bindings:
            self.h.require(self.bindings[path] == item, 'Changed/conflicting binding: ' + path)
        self.bindings[path] = item
        self.result['input_bindings'][path] = item['sha256']
        if source or Path(path).suffix == '.py':
            self.result['source_bindings'][path] = item['sha256']
        return item

    def inventory(self, record):
        for group in ('source_bindings', 'input_bindings', 'artifacts'):
            for path, digest in record.get(group, {}).items():
                self.pin(path, digest, source=(group == 'source_bindings'))

    def receipt(self, directory, digest, stage, source, source_sha):
        self.pin(source, source_sha, source=True)
        rp, cp = HERE / directory / 'result.json', HERE / directory / 'completion.json'
        self.pin(rp, digest); self.pin(cp)
        r, c = json.loads(rp.read_text()), json.loads(cp.read_text())
        self.h.require(r['schema'] == SCHEMA and r['stage'] == stage
            and r['passed'] and r['completed'] and r['hashes_unchanged']
            and r['truth_accessed'] is False and c['passed'] and c['completed']
            and c['result_sha256'] == digest, 'Complete preceding stage required')
        self.h.require(r['source_bindings'][str(Path(source).resolve())] == source_sha
            and r['source_bindings'][str(HERE / 'PROTOCOL.md')] == PROTOCOL_SHA,
            'Preceding source/protocol ownership')
        self.inventory(r)
        return r

    def environment(self):
        h = self.h
        h.require(Path(sys.prefix).resolve() == Path('/Users/matt/anaconda3'), 'AP CPU only')
        h.require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'nice10 required')
        h.require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'No bytecode writes')
        h.require(all(os.environ.get(k) == '1' for k in BLAS), 'BLAS1 required')
        h.require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') == '0', 'No fallback')
        h.require(not any(x in sys.modules for x in ('numpy', 'torch', 'flopscope', 'threadpoolctl')),
                  'Authenticate before numerical imports')
        self.result['before_swap'] = h.swap_state()

    def guard(self, *_):
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != 'darwin':
            rss *= 1024
        self.result['resources'] = dict(peak_rss_bytes=rss, elapsed_s=time.monotonic() - self.started)
        if not (rss < 4_000_000_000 and time.monotonic() - self.started < 600):
            signal.setitimer(signal.ITIMER_REAL, 0)
            raise RuntimeError('Fixed CPU RSS/deadline guard')

    def save(self, np, name, value):
        path = self.folder / name
        with path.open('xb') as stream:
            np.save(stream, value, allow_pickle=False)
        item = self.h.bind(path)
        self.bindings[str(path)] = item
        self.result['artifacts'][str(path)] = item['sha256']
        return dict(item, **self.h.array_meta(value))

    def attempt(self, name, record):
        path = self.folder / name
        self.h.write(path, record)
        self.bindings[str(path)] = self.h.bind(path)
        self.result['artifacts'][str(path)] = self.bindings[str(path)]['sha256']

    def load(self, np, item):
        self.h.require(str(Path(item['path']).resolve()) in self.bindings,
                       'Array must already belong to the pre-import authenticated inventory')
        self.pin(item['path'], item['sha256'])
        value = np.load(item['path'], allow_pickle=False)
        meta = self.h.array_meta(value)
        self.h.require(all(key not in item or item[key] == val for key, val in meta.items()),
                       'Array metadata/raw-byte binding')
        self.h.require(np.isfinite(value).all(), 'Finite input array required')
        return value

    def finish(self, passed):
        if passed:
            self.h.recheck(self.bindings)
            for mod in tuple(sys.modules.values()):
                path = getattr(mod, '__file__', None)
                if path and Path(path).suffix == '.py' and Path(path).is_file():
                    path = Path(path).resolve()
                    if path.is_relative_to(NEW):
                        self.h.require(str(path) in self.bindings, 'Unbound local source: ' + str(path))
            self.result['after_swap'] = self.h.swap_state()
            growth = self.result['after_swap']['used_bytes'] - self.result['before_swap']['used_bytes']
            self.result['swap_growth_bytes'] = growth
            self.h.require(growth <= 2 * 1024**3, 'Fixed swap-growth guard')
            self.guard()
            self.result.update(hashes_unchanged=True, sources_unchanged=True)
        self.result.update(passed=bool(passed), completed=True)

    def close(self):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.old_handler)
        self.result.update(elapsed_s=time.monotonic() - self.started,
                           finished_utc=datetime.now(timezone.utc).isoformat())
        self.h.write(self.folder / 'result.json', self.result)
        self.h.write(self.folder / 'completion.json', dict(passed=self.result['passed'],
            completed=self.result['completed'], result_sha256=self.h.sha(self.folder / 'result.json')))
        print(json.dumps({k: self.result.get(k) for k in ('stage', 'passed', 'completed', 'elapsed_s', 'error')}), flush=True)


def authenticate_old(a):
    """No model/target/array execution; preserve failed NG and sealed probes."""
    h = a.h
    a.pin(NG / 'model.py', MODEL_SHA, source=True)
    model = module(NG / 'model.py', 'covariance_boundary_original_ng')
    for path, digest in model.authenticate().items():
        a.pin(path, digest)
    probe_path = CO / 'stage_cpu_forward_v1/result.json'
    a.pin(probe_path, PROBE_SHA); a.pin(probe_path.parent / 'completion.json')
    probes = json.loads(probe_path.read_text())
    seal = json.loads((probe_path.parent / 'completion.json').read_text())
    h.require(probes['passed'] and probes['completed'] and probes['sources_unchanged']
        and probes['no_fit'] and not probes['alltruth_accessed'] and probes['parameter_updates'] == 0
        and seal['passed'] and seal['completed'] and seal['result_sha256'] == PROBE_SHA,
        'Original complete CPU probe owner')
    for group in ('source_metadata', 'weight_bindings', 'artifact_bindings'):
        for item in probes[group].values():
            a.pin(item['path'], item['sha256'])
    h.require(probes['versions'] == h.VERSIONS == {'numpy': '1.26.4', 'torch': '2.7.0'},
              'Original CPU numerical frame')
    old_path = NG / 'forward_v1/result.json'
    a.pin(old_path, NG_RESULT_SHA); a.pin(old_path.parent / 'completion.json', NG_COMPLETION_SHA)
    old = json.loads(old_path.read_text())
    oldseal = json.loads((old_path.parent / 'completion.json').read_text())
    h.require(not old['passed'] and not old['completed'] and not oldseal['passed']
        and not oldseal['completed'] and oldseal['result_sha256'] == NG_RESULT_SHA
        and old['card_calls'] == 6 and set(old['outputs']) == {'35'}
        and set(old['outputs']['35']) == {c for c, _ in CASES}
        and old['alltruth_accessed'] is False, 'Preserve failed, rank35-only NG sequence')
    a.inventory(old)
    a.result.update(payloads=probes['payloads'], cotangents=probes['cotangents'],
                    probe_seal_sha256=probes['probe_seal_sha256'])
    return model, probes, old


def valid_prediction(h, np, value):
    h.require(value.shape == (2, 1024) and value.dtype == np.float32
              and np.isfinite(value).all() and (value >= 0).all(), 'Served two-row contract')


def compare(np, x, y):
    d = x.astype(np.float64) - y.astype(np.float64)
    return dict(maximum_absolute_difference=float(np.max(np.abs(d))), l2=float(np.linalg.norm(d)),
                byte_exact=x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true')
    for name in ('source', 'native-source', 'native-result', 'model', 'protocol'):
        p.add_argument('--' + name + '-sha256')
    args = p.parse_args()
    if not args.run:
        p.print_help(); return 0
    if any(not v or len(v) != 64 for k, v in vars(args).items() if k != 'run'):
        p.error('All reviewed SHA256 arguments are required')
    a = Stage(Path(__file__), args.source_sha256, args.protocol_sha256, 'cpu', 'cpu_v1')
    h, r = a.h, a.result
    try:
        a.environment()
        h.require(args.model_sha256 == MODEL_SHA, 'Unchanged native-G0 CPU model')
        native = a.receipt('native_n_v1', args.native_result_sha256, 'n',
                           HERE / 'native.py', args.native_source_sha256)
        h.require(set(native['ranks']) == {'35', '100'} and native['prefix_calls'] == 11
            and native['native_baseline_reproduces_all_old_two_rows'] and native['repeat_byte_exact'],
            'Fixed native inputs, all old native prefixes, and repeat')
        model, probes, old = authenticate_old(a)
        own = model.owner()
        h.require(not any(k in sys.modules for k in ('numpy', 'torch', 'flopscope', 'threadpoolctl')),
                  'All authentication before numerical imports')
        import numpy as np
        import torch
        h.require(dict(numpy=np.__version__, torch=torch.__version__) == h.VERSIONS, 'AP versions')
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        r['environment'] = dict(versions=h.VERSIONS, device='cpu', blas={k: os.environ[k] for k in BLAS})
        r.update(native_n_result_sha256=args.native_result_sha256, model_sha256=MODEL_SHA,
                 card_constructions=0, recurrent_steps=0, passive_step_return_observation=True,
                 full16_weight_feature_setup=True, CPU_work_is_paid_diagnostic_not_deployment_bill=True)
        for rank in RANKS:
            nr = native['ranks'][str(rank)]
            Cn, W = (a.load(np, nr['arrays'][k]) for k in ('C0', 'W1'))
            ix, alive = (a.load(np, nr['arrays'][k]) for k in ('indices', 'alive'))
            h.require(Cn.shape == (1024, 1024) and Cn.dtype == np.float64
                and W.shape == Cn.shape and W.dtype == np.float32
                and ix.shape == (1024,) and np.issubdtype(ix.dtype, np.integer)
                and np.array_equal(ix, np.arange(1024)) and alive.shape == (1024,)
                and alive.dtype == np.bool_ and alive.all(), 'Canonical all-active shared input')
            h.require(np.diag(Cn).min() >= 1e-10, 'Native C0 positive domain')
            row = dict(arrays={}, outputs={}, circuits={})
            r['ranks'][str(rank)] = row
            w = torch.tensor(W, dtype=torch.float32)
            c = torch.tensor(Cn, dtype=torch.float64).float()
            with torch.no_grad():
                L = ((w.T @ c) @ w).to(torch.float64).numpy().copy()
                R = (w.T @ (c @ w)).to(torch.float64).numpy().copy()
            for label, value in (('L', L), ('R', R)):
                item = a.save(np, f'row_{rank:04d}_{label}.npy', value)
                row['circuits'][label] = dict(Cpre=item, Bpre=item)
            a.attempt(f'row_{rank:04d}_circuits_attempt.json', row['circuits'])
            h.require(np.isfinite(L).all() and np.isfinite(R).all()
                      and np.diag(L).min() >= 1e-10 and np.diag(R).min() >= 1e-10, 'Dense circuit domain')
            native_pre = a.load(np, nr['arrays']['Cpre'])
            native_metric = a.load(np, nr['arrays']['Bpre'])
            h.require(native_pre.dtype == native_metric.dtype == np.float64
                and native_pre.shape == native_metric.shape == Cn.shape
                and native_pre.tobytes() == native_metric.tobytes(), 'Native all-active Cpre/Bpre identity')
            row['circuit_differences'] = dict(N_minus_L=compare(np, native_pre, L),
                N_minus_R=compare(np, native_pre, R), R_minus_L=compare(np, R, L))
            reference_C0 = None
            for case, amplitude in CASES:
                values = h.load_bundle(np, probes['payloads'][case])
                P = h.torch_payload(torch, values, own.PKEYS, 'cpu')
                card = model.card(rank, P=P, enabled=True)
                r['card_constructions'] += 1
                r['sequence'].append(dict(rank=rank, case=case, layers=[0, 1]))
                captured = {}
                expected_filename = str(NG / 'model.py') + ':step'

                def observe(frame, event, _arg):
                    if event == 'return' and frame.f_code.co_name == 'step' and frame.f_code.co_filename == expected_filename:
                        loc = frame.f_locals
                        captured[int(loc['l'])] = {k: loc[k] for k in ('host', 'delta', 'Cpre', 'Bpre')}

                h.require(sys.getprofile() is None, 'No existing profiler to replace')
                begin = time.monotonic()
                try:
                    sys.setprofile(observe)
                    with torch.no_grad():
                        state = card.initial()
                        control = torch.zeros(card.n, dtype=card.dtype, device='cpu')
                        state0 = card.step(0, state, control, card.P0)
                        state1 = card.step(1, state0, control, card.P0)
                        prediction = own.served(torch.stack((state0[0], state1[0]))).numpy().copy()
                finally:
                    sys.setprofile(None)
                r['recurrent_steps'] += 2
                h.require(set(captured) == {0, 1}, 'Both original compiled step returns observed')
                host = captured[1]['host'].detach().numpy().copy()
                delta = captured[1]['delta'].detach().numpy().copy()
                C0 = state0[1].detach().numpy().copy()
                entry = dict(rank=rank, case=case, amplitude=amplitude,
                    prediction=a.save(np, f'row_{rank:04d}_{case}.npy', prediction),
                    host1=a.save(np, f'row_{rank:04d}_{case}_host1.npy', host),
                    delta1=a.save(np, f'row_{rank:04d}_{case}_delta1.npy', delta),
                    diagnostics=card.diagnostics(), elapsed_s=time.monotonic() - begin,
                    payload_sha256=probes['payloads'][case]['sha256'])
                row['outputs'][case] = entry
                a.attempt(f'row_{rank:04d}_{case}_attempt.json', entry)
                valid_prediction(h, np, prediction)
                h.require(host.shape == delta.shape == (1024,) and host.dtype == delta.dtype == np.float64
                    and np.isfinite(host).all() and np.isfinite(delta).all(), 'Actual host/delta values')
                diagnostics = entry['diagnostics']
                h.require(set(diagnostics['variance_domain']) == {'0', '1'}
                    and all(x['finite'] and x['minimum'] >= 1e-10 and x['below_floor_count'] == 0
                            for x in diagnostics['variance_domain'].values())
                    and set(diagnostics['selection_cells']) == {f'{l}:{s}' for l in (0, 1) for s in range(3)}
                    and set(diagnostics['variance_visits'].values()) == {1}
                    and set(diagnostics['selection_visits'].values()) == {1}, 'Exactly two forward domains and six selectors')
                h.require(diagnostics['source_inverse_byte_exact'] and diagnostics['only_Cpre_Bpre_at_layer0_replaced']
                    and diagnostics['native_G0_enabled'] and not diagnostics['native_G0_requires_grad']
                    and not diagnostics['training'], 'Original NG arithmetic and forward-only convention')
                h.require(card.w[1].numpy().tobytes() == W.tobytes()
                    and bool((state0[6] == 1).all()) and bool((state0[8] == 0).all())
                    and all(diagnostics['selection_cells'][f'0:{s}']['ordered_ids'] == list(range(1024))
                            for s in range(3)), 'Identical W1 and P-independent all-active L0')
                h.require(all(not P[k].requires_grad and P[k].numpy().tobytes() == values[k].tobytes()
                              for k in P), 'P unchanged and no gradient leaves')
                if reference_C0 is None:
                    reference_C0 = C0.copy()
                    row['arrays']['C0'] = a.save(np, f'row_{rank:04d}_CPU_C0.npy', C0)
                    with torch.no_grad():
                        own_left = ((card.w[1].T @ state0[1].float()) @ card.w[1]).to(torch.float64).numpy().copy()
                    item = a.save(np, f'row_{rank:04d}_L_CPU_C0.npy', own_left)
                    row['circuits']['L_cpu_C0'] = dict(Cpre=item, Bpre=item)
                    a.attempt(f'row_{rank:04d}_C0_attempt.json', dict(C0=row['arrays']['C0'], L_cpu_C0=item))
                    h.require(np.isfinite(C0).all() and np.diag(C0).min() >= 1e-10
                        and own_left.tobytes() == captured[1]['Cpre'].numpy().tobytes()
                        and own_left.tobytes() == captured[1]['Bpre'].numpy().tobytes(), 'Original own-input left expression reproduced')
                    row['C0_native_vs_cpu'] = dict(**compare(np, Cn, C0),
                        diagonal=compare(np, np.diag(Cn), np.diag(C0)),
                        native_asymmetry_l2=float(np.linalg.norm(Cn - Cn.T)),
                        cpu_asymmetry_l2=float(np.linalg.norm(C0 - C0.T)),
                        float32_byte_exact=Cn.astype(np.float32).tobytes() == C0.astype(np.float32).tobytes())
                    row['input_law_contribution'] = compare(np, L, own_left)
                    if row['C0_native_vs_cpu']['float32_byte_exact']:
                        h.require(L.tobytes() == own_left.tobytes(), 'Identical FP32 inputs must give identical left product')
                else:
                    h.require(reference_C0.tobytes() == C0.tobytes(), 'CPU C0 invariant across all five P cases')
                entry['C0_P_invariant'] = True
                if rank == 35:
                    ref = h.load_array(np, old['outputs']['35'][case]['prediction'])
                    entry['saved_NG_two_rows_byte_exact'] = prediction.tobytes() == ref[:2].tobytes()
                    oldcells = old['outputs']['35'][case]['diagnostics']['selection_cells']
                    entry['saved_NG_ordered_selectors_exact'] = all(
                        v == oldcells[k] for k, v in diagnostics['selection_cells'].items())
                    h.require(entry['saved_NG_two_rows_byte_exact'] and entry['saved_NG_ordered_selectors_exact'],
                              'Rank35 existing NG prefix bytes/selectors')
                    del ref
                a.guard()
                del card, P, values, state, state0, state1, captured, prediction, host, delta, C0
                gc.collect()
            row['C0_all_five_P_byte_exact'] = True
            del Cn, W, c, w, L, R, native_pre, native_metric, reference_C0
            gc.collect()
        h.require(r['card_constructions'] == 10 and r['recurrent_steps'] == 20
                  and not any(k in sys.modules for k in ('flopscope', 'threadpoolctl')), 'Complete fixed CPU scope')
        r['all_fixed_arrays_sealed'] = True
        a.finish(True)
    except BaseException:
        r.update(passed=False, completed=False, hashes_unchanged=False, sources_unchanged=False,
                 error=traceback.format_exc())
    finally:
        a.close()
    return 0 if r['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
