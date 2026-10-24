"""Explicit-run fixed current-point block-jet certificate, never a score."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
OLD = NEW/'current_response_quadratic_v1'
ENV = NEW.parent/'data_gate/.venv'
CAPTURE_SHA = '575727d6f465cbdfeeb9f58d201e9a6e2983aef94dcd2d2389108c58fe569b42'
PINS = {
    NEW/'unused_materializations_v1/estimator.py':
        '409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f',
    OLD/'feature_second_jet_reference.py':
        'dd4484fc3baaefb783ff54497b43ff801214e0583cd16d28a9d9b4c212cb218f',
    OLD/'gnn_second_jet_reference.py':
        '2ce74c478040d456078091751520d3a845b2fc8ca47acd6269d8fd8d255d8b21',
    OLD/'wide/result.json':
        '76823abbdf7d8f4a8b65c0c483a32c5a82cbbeb97a30a1469215699999aa8a47',
    OLD/'capture/result.json': CAPTURE_SHA,
    NEW/'check_response_rows_adjusted_v1.py':
        '8e99890bdc7742164a53303a23691e9f1cc54909c9cccc9cf64fc362eb459304',
}
VERSIONS = dict(numpy='2.4.6', flopscope='0.12.0', whestbench='0.16.0', threadpoolctl='3.6.0')
BLAS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def run():
    assert Path(sys.prefix).resolve() == ENV.resolve() and 'anaconda3' not in sys.executable
    assert os.getpriority(os.PRIO_PROCESS, 0) >= 10
    assert all(os.environ.get(k) == '1' for k in BLAS)
    assert os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
    assert {k: importlib.metadata.version(k) for k in VERSIONS} == VERSIONS
    assert all(sha(p) == digest for p, digest in PINS.items())
    h = module(NEW/'check_response_rows_adjusted_v1.py', 'rounding_receipt_helper')
    admission = h.load_json(HERE/'ADMISSION.json')
    assert admission['admitted'] and admission['checker_sha256'] == sha(Path(__file__))
    h.check_bindings(admission['source_bindings'])
    required = {str((HERE/name).resolve()) for name in (
        'PROTOCOL.md', 'native_reference.py', 'feature_native_jet.py',
        'gnn_native_jet.py', 'qualify_native.py')}
    assert required <= {v['path'] for v in admission['source_bindings'].values()}
    capture = h.load_json(OLD/'capture/result.json')
    completion = h.load_json(OLD/'capture/completion.json')
    assert capture['passed'] and completion['passed'] and completion['result_sha256'] == CAPTURE_SHA
    assert [t['rank'] for t in capture['trials']] == [35, 100]
    assert all(t['all16_byte_exact'] and t['flops'] == t['reference_flops'] for t in capture['trials'])
    for field in ('sources', 'fixture_bindings', 'artifacts'):
        h.check_bindings(capture[field])
    sources = {str(p): h.binding(p) for p in (*PINS, HERE/'ADMISSION.json')}
    sources.update(admission['source_bindings'])
    directory = HERE/'qualification'
    directory.mkdir(exist_ok=False)
    h.write_json(directory/'started.json', dict(
        utc=h.utc(), pid=os.getpid(), executable=sys.executable, prefix=sys.prefix,
        nice=os.getpriority(os.PRIO_PROCESS, 0), versions=VERSIONS, sources=sources,
        capture_sha256=CAPTURE_SHA, blas_env={k: os.environ.get(k) for k in BLAS},
        watchdog_s=180, rss_limit_bytes=2_000_000_000, truth_files_accessed=False,
        score_evidence=False, served_fp32_derivative_claim=False,
        convention='Native-point uniform composite-block lift with frozen base residuals.'))
    result = dict(passed=False, records=[], artifacts={}, source_bindings=sources,
                  capture_sha256=CAPTURE_SHA, truth_files_accessed=False,
                  score_evidence=False, served_fp32_derivative_claim=False)
    timer = time.perf_counter()

    def watchdog(signum, frame):
        raise TimeoutError('Fixed180s native block-jet certificate watchdog')

    signal.signal(signal.SIGALRM, watchdog)
    signal.alarm(180)
    try:
        import numpy as np
        from threadpoolctl import threadpool_info, threadpool_limits
        import native_reference
        import feature_native_jet as feat
        import gnn_native_jet as gnn
        scalar = module(OLD/'feature_second_jet_reference.py', 'rounding_scalar_reference')
        env = native_reference.original_functions(np)
        with np.load(OLD/'capture/current_P.npz', allow_pickle=False) as archive:
            payload = dict(archive)
        assert all(a.dtype == np.float32 and np.isfinite(a).all() for a in payload.values())

        def resources():
            assert h.peak_rss_bytes() < 2_000_000_000
            assert time.perf_counter()-timer < 180

        def exact(actual, expected):
            actual, expected = np.asarray(actual), np.asarray(expected)
            compatible = actual.dtype == expected.dtype and actual.shape == expected.shape
            finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
            maximum = (float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64))))
                       if actual.shape == expected.shape and finite else None)
            return dict(passed=bool(compatible and finite and actual.tobytes() == expected.tobytes()),
                        actual_dtype=str(actual.dtype), expected_dtype=str(expected.dtype),
                        actual_shape=list(actual.shape), expected_shape=list(expected.shape),
                        max_absolute=maximum, all_finite=finite)

        def mixed(actual, expected, tolerance, absolute):
            finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
            if not finite:
                return dict(error_norm=None, reference_norm=None, relative_error=None,
                            tolerance=tolerance, absolute=absolute, all_finite=False, passed=False)
            norm = float(np.linalg.norm(expected))
            error = float(np.linalg.norm(actual-expected))
            return dict(error_norm=error, reference_norm=norm,
                        relative_error=error/max(norm, 1e-300),
                        tolerance=tolerance, absolute=absolute,
                        all_finite=True, passed=bool(error <= absolute+tolerance*norm))

        with threadpool_limits(limits=1):
            pools = threadpool_info()
            assert all(p['num_threads'] == 1 for p in pools if p.get('user_api') == 'blas')
            result['threadpools'] = pools
            result['blas_attestation'] = 'Environment1; detected pools checked if present. Empty inventory is not observed pool evidence.'
            for rank in (35, 100):
                weights = np.load(capture['fixture_bindings'][f'weights_{rank}']['path'],
                                  mmap_mode='r', allow_pickle=False)
                saved = {k: [] for k in ('gaussian_gate', 'host_response', 'learned_response',
                         'delta_derivative', 'delta_second_derivative', 'host_derivative',
                         'host_second_derivative')}
                for layer in range(8, 15):
                    with np.load(OLD/f'capture/rank{rank}_layer{layer:02d}.npz', allow_pickle=False) as archive:
                        frame = dict(archive)
                    mu, pre, k21, k4, k22 = (frame[k] for k in
                        ('mu', 'pre_cov', 'pre_k21', 'pre_k4', 'pre_k22'))
                    assert all(a.dtype == np.float64 and np.isfinite(a).all()
                               for a in (mu, pre, k21, k4, k22))
                    w = np.asarray(weights[layer])
                    sigma, state = frame['aux_sigma'], frame['prior_state']
                    native = env['_nonlinear_pair_batch'](mu, pre, k21, k4, k22, 'covariance')
                    x = env['features'](layer, native[0], native[1], k21, native[4], w)
                    delta, next_state = env['gnn_step'](layer, x, sigma, w, state, payload)
                    primal = dict(mean=exact(native[0], frame['host_mean']),
                                  covariance=exact(native[1], frame['host_cov']),
                                  k21=exact(native[2], frame['host_k21']),
                                  radial=exact(native[3], frame['pair_radial_before_omitted']),
                                  features=exact(x, frame['features']),
                                  head_delta=exact(delta, frame['head_delta']),
                                  head_state=exact(next_state, frame['next_state']))
                    primal.update({'aux_'+k: exact(v, frame['aux_'+k])
                                   for k, v in native[4].items() if k != 'pre_cov'})
                    rec = dict(rank=rank, layer=layer, independent_native=primal,
                               finite_difference=[], passed=False)
                    result['records'].append(rec)
                    assert all(v['passed'] for v in primal.values()), 'Independent native replay failed'
                    prepared = feat.prepare(dict(frame, layer=layer, weights=w), env, scalar.raw_jets)
                    jet = feat.features(prepared)
                    head = gnn.prepare(layer, sigma, w, state, payload)
                    response = gnn.step(head, jet.value, jet.first, jet.second)
                    base_lift = feat.lift(prepared, np.zeros_like(mu))
                    head_lift = response.lift(base_lift)
                    lifted = dict(features=exact(jet.value, frame['features']),
                                  covariance=exact(jet.covariance.value, frame['host_cov']),
                                  mean=exact(jet.aux['mean'].value, frame['host_mean']),
                                  head_delta=exact(response.delta, frame['head_delta']),
                                  head_state=exact(response.state, frame['next_state']),
                                  feature_zero=exact(base_lift, frame['features'].astype(np.float64)),
                                  delta_zero=exact(head_lift.delta, frame['head_delta'].astype(np.float64)),
                                  state_zero=exact(head_lift.state, frame['next_state'].astype(np.float64)))
                    lifted.update({'aux_'+k: exact(v.value if hasattr(v, 'value') else v,
                                                   frame['aux_'+k])
                                   for k, v in jet.aux.items() if k not in ('pre_cov', 'mean')})
                    assert len(jet.native) == 5
                    lifted.update({'native_'+k: exact(actual, frame[expected])
                        for k, actual, expected in zip(('mean', 'covariance', 'k21', 'radial'),
                            jet.native[:4], ('host_mean', 'host_cov', 'host_k21',
                                             'pair_radial_before_omitted'))})
                    rec['lifted_base'] = lifted
                    rec['feature_diagnostics'] = jet.diagnostics
                    rec['gnn_diagnostics'] = response.diagnostics
                    boundaries = int(jet.diagnostics['moving_boundary_count'])
                    boundaries += sum(response.diagnostics['clip_endpoint_counts'].values())
                    rec['moving_boundary_count'] = boundaries
                    rec['cost'] = dict(feature_subtotal=jet.cost, gnn_subtotal=response.cost,
                                       head_preparation=response.preparation,
                                       zero_lift_gnn=head_lift.cost,
                                       qualification='Offline separately priced subtotals, not complete new-estimator or FNP bill.')
                    assert all(v['passed'] for v in lifted.values()), 'Native-point base compatibility failed'
                    assert boundaries == 0, 'Exact moving nonsmooth boundary'
                    assert jet.diagnostics['all_finite']
                    assert response.diagnostics['all_finite']
                    assert head_lift.diagnostics['all_finite']
                    assert response.diagnostics['nonpositive_layer_scales'] == 0
                    assert np.isfinite(jet.first).all() and np.isfinite(jet.second).all()
                    host = jet.aux['mean'].first
                    learned = host+response.ddelta
                    rec['response_magnitudes'] = dict(
                        delta_first_rms=float(np.sqrt(np.mean(response.ddelta**2))),
                        delta_second_sigma_rms=float(np.sqrt(np.mean((response.d2delta*sigma)**2))),
                        host_clipped=int(np.count_nonzero((host < 0) | (host > 1))),
                        learned_clipped=int(np.count_nonzero((learned < 0) | (learned > 1))))
                    if layer in (8, 14):
                        ids = np.array([0, 257, 514, 771])
                        for factor in (.001, .0005):
                            first, second, gfirst, gsecond = [], [], [], []
                            lift_evaluations = []
                            for j in ids:
                                epsilon = factor*sigma[j]
                                shift = np.zeros_like(mu)
                                shift[j] = epsilon
                                xp, xm = feat.lift(prepared, shift), feat.lift(prepared, -shift)
                                assert np.isfinite(xp).all() and np.isfinite(xm).all()
                                gp, gm = response.lift(xp), response.lift(xm)
                                lift_evaluations.append(dict(unit=int(j),
                                    plus_diagnostics=gp.diagnostics, minus_diagnostics=gm.diagnostics,
                                    plus_cost={k: gp.cost[k] for k in ('total_flops', 'by_phase')},
                                    minus_cost={k: gm.cost[k] for k in ('total_flops', 'by_phase')}))
                                assert gp.diagnostics['all_finite'] and gm.diagnostics['all_finite']
                                dp, dm = gp.delta, gm.delta
                                first.append((xp[j]-xm[j])/(2*epsilon))
                                second.append((xp[j]-2*base_lift[j]+xm[j])/(epsilon*epsilon))
                                gfirst.append((dp[j]-dm[j])/(2*epsilon))
                                gsecond.append((dp[j]-2*head_lift.delta[j]+dm[j])/(epsilon*epsilon))
                                resources()
                            first, second = np.array(first), np.array(second)
                            ff = [dict(feature=k, **mixed(jet.first[ids, k]*sigma[ids],
                                  first[:, k]*sigma[ids], 2e-6, 1e-8)) for k in range(30)]
                            fs = [dict(feature=k, **mixed(jet.second[ids, k]*sigma[ids]**2,
                                  second[:, k]*sigma[ids]**2, 5e-4, 1e-6)) for k in range(30)]
                            gf = mixed(response.ddelta[ids], np.array(gfirst), 1e-4, 1e-9)
                            gs = mixed(response.d2delta[ids]*sigma[ids], np.array(gsecond)*sigma[ids], 5e-4, 1e-8)
                            rec['finite_difference'].append(dict(epsilon_factor=factor,
                                units=ids.tolist(), feature_first=ff, feature_second=fs,
                                gnn_first=gf, gnn_second=gs,
                                lift_evaluations=lift_evaluations,
                                passed=all(t['passed'] for t in ff+fs) and gf['passed'] and gs['passed']))
                    rec['passed'] = all(t['passed'] for t in rec['finite_difference'])
                    print(json.dumps(dict(rank=rank, layer=layer, passed=rec['passed'],
                        native_bytes=all(t['passed'] for t in primal.values()),
                        lifted_base=all(t['passed'] for t in lifted.values()),
                        moving_boundaries=boundaries, response=rec['response_magnitudes'],
                        fd_passes=[t['passed'] for t in rec['finite_difference']])), flush=True)
                    if not rec['passed']:
                        raise RuntimeError(f'Fixed certificate failed rank{rank}/layer{layer}; no later contexts or targets')
                    saved['gaussian_gate'].append(jet.aux['gate'].value.copy())
                    saved['host_response'].append(np.clip(host, 0, 1))
                    saved['learned_response'].append(np.clip(learned, 0, 1))
                    saved['delta_derivative'].append(response.ddelta.copy())
                    saved['delta_second_derivative'].append(response.d2delta.copy())
                    saved['host_derivative'].append(host.copy())
                    saved['host_second_derivative'].append(jet.aux['mean'].second.copy())
                    resources()
                    del frame, native, x, prepared, jet, head, response, base_lift, head_lift
                name = f'rank{rank}_responses.npz'
                with (directory/name).open('xb') as handle:
                    np.savez(handle, **{k: np.stack(v) for k, v in saved.items()}, layers=np.arange(8, 15))
                result['artifacts'][name] = h.binding(directory/name)
                del weights, saved
        for field in ('sources', 'fixture_bindings', 'artifacts'):
            h.check_bindings(capture[field])
        h.check_bindings(sources)
        h.check_bindings(result['artifacts'])
        resources()
        result['passed'] = len(result['records']) == 14 and all(r['passed'] for r in result['records'])
    except BaseException:
        result['error'] = traceback.format_exc()
    finally:
        signal.alarm(0)
    result.update(completed_utc=h.utc(), wall_s=time.perf_counter()-timer, rss_bytes=h.peak_rss_bytes())
    seal = h.write_json(directory/'result.json', result)
    h.write_json(directory/'completion.json', dict(passed=result['passed'], result_sha256=seal))
    print(json.dumps(dict(passed=result['passed'], result_sha256=seal, wall_s=result['wall_s'],
                         rss_bytes=result['rss_bytes'], error=result.get('error'))), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        parser.error('Explicit --run required; inert by default')
    raise SystemExit(run())
