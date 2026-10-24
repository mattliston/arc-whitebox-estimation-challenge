"""Inert prospective successor: independently enclosed feature21 derivative oracle.

Requires sealed S diagnosis PASS plus new independent review/admission.  The
original R/Q sources and R's literal FAIL remain unchanged.  No gate truth,
accuracy action, fitting, estimator change or automatic scientific admission.
"""
import argparse
from decimal import Decimal
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
R = NEW/'current_response_rounding_v1'
S = NEW/'current_response_scalar_certificate_v1'
ENV = NEW.parent/'data_gate/.venv'
CAPTURE_SHA = '575727d6f465cbdfeeb9f58d201e9a6e2983aef94dcd2d2389108c58fe569b42'
R_FAILURE_SHA = '3cf6986864a238c4c71a68287e385006d339b3c276703ac2d995281edab12a7d'
S_DIAGNOSIS_SHA = '14b3dbabcd594587f9898acb1f8e3287932e00e9e25bf17347ff5c14a2d92227'
VERSIONS = dict(numpy='2.4.6', flopscope='0.12.0', whestbench='0.16.0', threadpoolctl='3.6.0')
BLAS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
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
    R/'qualification/result.json': R_FAILURE_SHA,
    R/'qualify_native.py':
        '4ee0af45db3738798d7182ecf2cc1fa2a60f7cb6d64641bab6c903ed77dccf67',
    R/'PROTOCOL.md':
        '1c7a69ad6e9f1f9e6186be5a48ef45a495daca76a668f24dfc68fd83a546b2cc',
    R/'native_reference.py':
        'bea92276493fb491e90f9b24e54b588bb31175a57bc84eb6e3a345134ce29ac4',
    R/'feature_native_jet.py':
        'c41d904a64ada75fe047944dbe276112db1bec9b1d7b7b4d39ad19444630c274',
    R/'gnn_native_jet.py':
        '30b421b8c413a31c6941c3b1aa10f10930b5fbd390dbaf89da658a38a2da8a2c',
    S/'scalar_interval.py':
        '80a2cb4ec800648a4c7792fd019f9154f29d43b9dbc8d559b15d0329edc520b0',
    S/'diagnose_scalar.py':
        '799ae77214f8fd4cfdfe1733a0e2becc2e69bd2da589cfbecfc223e6826469f9',
    S/'PROTOCOL.md':
        '2f19f4765c8b9102161384e0ab3f8b73a75bc22df38441e70a1d27e9b1b8af8b',
    S/'diagnosis/result.json': S_DIAGNOSIS_SHA,
    ENV/'lib/python3.11/site-packages/flopscope/stats/_norm.py':
        'f23e49201c8e09e2785814b2b31663cda1c6bb217a57a211532e42fc9fe3b65d',
    ENV/'lib/python3.11/site-packages/flopscope/stats/_erf.py':
        '4cdd75fdb85981fc385417082985478566b18e19fa67ebb7d41225b5efaaa3bd',
}


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
    h = module(NEW/'check_response_rows_adjusted_v1.py', 'certified_native_receipts')
    admission = h.load_json(HERE/'ADMISSION.json')
    checker_sha, protocol_sha = sha(Path(__file__)), sha(HERE/'PROTOCOL.md')
    assert admission['admitted'] and admission['checker_sha256'] == checker_sha
    review = admission['independent_review']
    assert review['passed'] and isinstance(review['reviewer'], str) and review['reviewer'].strip()
    assert review['checker_sha256'] == checker_sha and review['protocol_sha256'] == protocol_sha
    h.check_bindings(admission['source_bindings'])
    required = {str((HERE/name).resolve()) for name in ('PROTOCOL.md', 'check.py')}
    assert required <= {v['path'] for v in admission['source_bindings'].values()}
    diagnosis_path = S/'diagnosis/result.json'
    assert sha(diagnosis_path) == admission['scalar_diagnosis_sha256'] == S_DIAGNOSIS_SHA
    diagnosis = h.load_json(diagnosis_path)
    diagnosis_completion = h.load_json(S/'diagnosis/completion.json')
    assert diagnosis['passed'] and diagnosis_completion['passed']
    assert diagnosis_completion['result_sha256'] == admission['scalar_diagnosis_sha256']
    assert diagnosis['original_failure_sha256'] == R_FAILURE_SHA
    assert diagnosis['original_qualification_remains_failed']
    assert not diagnosis['truth_files_accessed'] and not diagnosis['final_score_evidence']
    assert diagnosis['ad_accuracy_passed']
    assert [r['unit'] for r in diagnosis['units']] == [0, 257, 514, 771]
    assert [p['epsilon_factor'] for p in diagnosis['probes']] == [.001, .0005]
    assert all(p['reproduction_passed'] and p['floating_first_evaluation_passed']
               for p in diagnosis['probes'])
    assert diagnosis['probes'][0]['coarse_failure_explained']
    h.check_bindings(diagnosis['source_bindings'])
    capture = h.load_json(OLD/'capture/result.json')
    completion = h.load_json(OLD/'capture/completion.json')
    assert capture['passed'] and completion['passed'] and completion['result_sha256'] == CAPTURE_SHA
    assert [t['rank'] for t in capture['trials']] == [35, 100]
    assert all(t['all16_byte_exact'] and t['flops'] == t['reference_flops'] for t in capture['trials'])
    for field in ('sources', 'fixture_bindings', 'artifacts'):
        h.check_bindings(capture[field])
    sources = {str(p): h.binding(p) for p in (*PINS, HERE/'ADMISSION.json',
                                             diagnosis_path, S/'diagnosis/completion.json')}
    sources.update(admission['source_bindings'])
    directory = HERE/'qualification'
    directory.mkdir(exist_ok=False)
    h.write_json(directory/'started.json', dict(
        utc=h.utc(), pid=os.getpid(), executable=sys.executable, prefix=sys.prefix,
        nice=os.getpriority(os.PRIO_PROCESS, 0), versions=VERSIONS, sources=sources,
        capture_sha256=CAPTURE_SHA, scalar_diagnosis_sha256=admission['scalar_diagnosis_sha256'],
        original_failure_sha256=R_FAILURE_SHA, blas_env={k: os.environ.get(k) for k in BLAS},
        watchdog_s=180, rss_limit_bytes=2_000_000_000, truth_files_accessed=False,
        score_evidence=False, served_fp32_derivative_claim=False,
        convention='Original native-point block lift; uniform independent feature21 derivative oracle.',
        independent_review=review))
    result = dict(passed=False, records=[], artifacts={}, source_bindings=sources,
                  capture_sha256=CAPTURE_SHA, truth_files_accessed=False,
                  score_evidence=False, served_fp32_derivative_claim=False,
                  original_qualification_remains_failed=True,
                  original_failure_sha256=R_FAILURE_SHA,
                  scalar_diagnosis_sha256=admission['scalar_diagnosis_sha256'],
                  independent_review=review,
                  oracle_replacement='Feature21 first AND second at every original derivative context; '
                                     'all original FD metrics preserved as diagnostics.')
    timer = time.perf_counter()

    def watchdog(signum, frame):
        raise TimeoutError('Fixed180s independently certified native-block watchdog')

    signal.signal(signal.SIGALRM, watchdog)
    signal.alarm(180)
    try:
        import numpy as np
        from threadpoolctl import threadpool_info, threadpool_limits
        native_reference = module(R/'native_reference.py', 'certified_original_native')
        feat = module(R/'feature_native_jet.py', 'certified_feature_block')
        gnn = module(R/'gnn_native_jet.py', 'certified_gnn_block')
        interval = module(S/'scalar_interval.py', 'certified_scalar_interval')
        scalar = module(OLD/'feature_second_jet_reference.py', 'certified_scalar_jet')
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
                        relative_error=error/max(norm, 1e-300), tolerance=tolerance,
                        absolute=absolute, all_finite=True,
                        passed=bool(error <= absolute+tolerance*norm))

        def bounds(value):
            assert value.lo.is_finite() and value.hi.is_finite() and value.lo <= value.hi
            return dict(lower=str(value.lo), upper=str(value.hi))

        def independent_scalar(jet, frame, ids):
            sigma = frame['aux_sigma']
            scale1, scale2 = sigma[ids], sigma[ids]**2
            v = sigma*sigma
            denominator = v*v
            actual1 = jet.first[ids, 21]*scale1
            actual2 = jet.second[ids, 21]*scale2
            error1, error2, unit_rows = [], [], []
            for index, j in enumerate(ids):
                parameters = dict(mu=float(frame['mu'][j]), rawv=float(frame['pre_cov'][j, j]),
                    sigma=float(sigma[j]), k3=float(frame['pre_k21'][j, j]),
                    k4=float(frame['pre_k4'][j]), feature21_denominator=float(denominator[j]))
                reference = interval.evaluate(**parameters, order=2)
                ref1 = reference['feature21_mu_derivatives'][1]*interval.point(float(scale1[index]))
                ref2 = reference['feature21_mu_derivatives'][2]*interval.point(float(scale2[index]))
                e1 = interval.point(float(actual1[index]))-ref1
                e2 = interval.point(float(actual2[index]))-ref2
                error1.append(e1)
                error2.append(e2)
                unit_rows.append(dict(unit=int(j), parameters=parameters,
                    actual_scaled_first=float(actual1[index]), actual_scaled_second=float(actual2[index]),
                    reference_scaled_first=bounds(ref1), reference_scaled_second=bounds(ref2),
                    signed_first_error=bounds(e1), signed_second_error=bounds(e2),
                    primitives=reference['primitives']))
                resources()
            norm1, norm2 = interval.outward_l2(error1), interval.outward_l2(error2)
            passed = norm1.hi <= Decimal('1e-9') and norm2.hi <= Decimal('1e-9')
            return dict(passed=bool(passed), feature=21, units=ids.tolist(),
                oracle='Independent outward Decimal polynomial-CDF derivatives, fixed precision160',
                first_error_norm=bounds(norm1), second_error_norm=bounds(norm2),
                fixed_upper_limit='1e-9', original_finite_differences_retained=True,
                point_evaluations=4, derivative_order=2, unit_results=unit_rows,
                deployment_cost_claim=False)

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
                    lifted.update({'aux_'+k: exact(v.value if hasattr(v, 'value') else v, frame['aux_'+k])
                                   for k, v in jet.aux.items() if k not in ('pre_cov', 'mean')})
                    assert len(jet.native) == 5
                    lifted.update({'native_'+k: exact(actual, frame[expected])
                        for k, actual, expected in zip(('mean', 'covariance', 'k21', 'radial'),
                            jet.native[:4], ('host_mean', 'host_cov', 'host_k21', 'pair_radial_before_omitted'))})
                    rec['lifted_base'] = lifted
                    rec['feature_diagnostics'] = jet.diagnostics
                    rec['gnn_diagnostics'] = response.diagnostics
                    boundaries = int(jet.diagnostics['moving_boundary_count'])
                    boundaries += sum(response.diagnostics['clip_endpoint_counts'].values())
                    rec['moving_boundary_count'] = boundaries
                    rec['cost'] = dict(feature_subtotal=jet.cost, gnn_subtotal=response.cost,
                                       head_preparation=response.preparation, zero_lift_gnn=head_lift.cost,
                                       qualification='Offline separately priced subtotals, not complete new-estimator or FNP bill.')
                    assert all(v['passed'] for v in lifted.values()), 'Native-point base compatibility failed'
                    assert boundaries == 0, 'Exact moving nonsmooth boundary'
                    assert jet.diagnostics['all_finite'] and response.diagnostics['all_finite']
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
                        rec['independent_feature21'] = independent_scalar(jet, frame, ids)
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
                            unchanged_pass = all(t['passed'] for t in ff+fs if t['feature'] != 21)
                            unchanged_pass = unchanged_pass and gf['passed'] and gs['passed']
                            rec['finite_difference'].append(dict(epsilon_factor=factor,
                                units=ids.tolist(), feature_first=ff, feature_second=fs,
                                gnn_first=gf, gnn_second=gs, lift_evaluations=lift_evaluations,
                                original_fd_passed=all(t['passed'] for t in ff+fs) and gf['passed'] and gs['passed'],
                                feature21_fd_diagnostic_only=True,
                                passed_other_features_and_gnn=unchanged_pass, passed=unchanged_pass))
                    rec['passed'] = (all(t['passed'] for t in rec['finite_difference'])
                                     and rec.get('independent_feature21', {}).get('passed', True))
                    print(json.dumps(dict(rank=rank, layer=layer, passed=rec['passed'],
                        native_bytes=all(t['passed'] for t in primal.values()),
                        lifted_base=all(t['passed'] for t in lifted.values()), moving_boundaries=boundaries,
                        independent_feature21=rec.get('independent_feature21', {}).get('passed'),
                        original_fd_passes=[t['original_fd_passed'] for t in rec['finite_difference']],
                        retained_fd_passes=[t['passed'] for t in rec['finite_difference']])), flush=True)
                    if not rec['passed']:
                        raise RuntimeError(f'Fixed successor failed rank{rank}/layer{layer}; no later contexts or targets')
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
        h.check_bindings(diagnosis['source_bindings'])
        h.check_bindings(result['artifacts'])
        resources()
        assert [(r['rank'], r['layer']) for r in result['records']] == [
            (rank, layer) for rank in (35, 100) for layer in range(8, 15)]
        assert [(r['rank'], r['layer']) for r in result['records'] if 'independent_feature21' in r] == [
            (35, 8), (35, 14), (100, 8), (100, 14)]
        result['passed'] = len(result['records']) == 14 and all(r['passed'] for r in result['records'])
    except BaseException:
        result['error'] = traceback.format_exc()
    finally:
        signal.alarm(0)
    result.update(completed_utc=h.utc(), wall_s=time.perf_counter()-timer, rss_bytes=h.peak_rss_bytes(),
        qualification='Prospective native-block numerical prerequisite only. Original R FAIL preserved; '
                      'no gate truth, covariance action, final raw or adjusted-score evidence.')
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
        parser.error('Explicit --run, S PASS and new independent admission required; inert by default')
    raise SystemExit(run())
