"""One-shot interval diagnosis of the frozen scalar FD failure; never a score."""
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
from decimal import Decimal

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
OLD = NEW/'current_response_quadratic_v1'
NATIVE = NEW/'current_response_rounding_v1'
ENV = NEW.parent/'data_gate/.venv'
VERSIONS = dict(numpy='2.4.6', flopscope='0.12.0', whestbench='0.16.0', threadpoolctl='3.6.0')
BLAS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
FAILURE_SHA = '3cf6986864a238c4c71a68287e385006d339b3c276703ac2d995281edab12a7d'
PINS = {
    NATIVE/'qualification/result.json': FAILURE_SHA,
    NATIVE/'feature_native_jet.py':
        'c41d904a64ada75fe047944dbe276112db1bec9b1d7b7b4d39ad19444630c274',
    OLD/'capture/result.json':
        '575727d6f465cbdfeeb9f58d201e9a6e2983aef94dcd2d2389108c58fe569b42',
    OLD/'capture/rank100_layer08.npz':
        '293d7a7ce5f5e04a4f01af68a5a8a71c7175dc24c38cba28ea82c93b1da50d79',
    OLD/'feature_second_jet_reference.py':
        'dd4484fc3baaefb783ff54497b43ff801214e0583cd16d28a9d9b4c212cb218f',
    NEW/'check_response_rows_adjusted_v1.py':
        '8e99890bdc7742164a53303a23691e9f1cc54909c9cccc9cf64fc362eb459304',
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
    h = module(NEW/'check_response_rows_adjusted_v1.py', 'scalar_diagnosis_receipts')
    admission = h.load_json(HERE/'ADMISSION.json')
    assert admission['admitted'] and admission['checker_sha256'] == sha(Path(__file__))
    h.check_bindings(admission['source_bindings'])
    required = {str((HERE/name).resolve()) for name in
                ('PROTOCOL.md', 'scalar_interval.py', 'diagnose_scalar.py')}
    assert required <= {v['path'] for v in admission['source_bindings'].values()}
    failed = h.load_json(NATIVE/'qualification/result.json')
    assert not failed['passed'] and len(failed['records']) == 8
    record = failed['records'][-1]
    assert record['rank'] == 100 and record['layer'] == 8 and not record['passed']
    assert record['moving_boundary_count'] == 0
    assert all(v['passed'] for v in record['independent_native'].values())
    assert all(v['passed'] for v in record['lifted_base'].values())
    probes = record['finite_difference']
    assert [p['epsilon_factor'] for p in probes] == [.001, .0005]
    assert [p['units'] for p in probes] == [[0, 257, 514, 771]]*2
    for i, probe in enumerate(probes):
        assert [v['feature'] for v in probe['feature_first'] if not v['passed']] == ([21] if i == 0 else [])
        assert all(v['passed'] for v in probe['feature_second'])
        assert probe['gnn_first']['passed'] and probe['gnn_second']['passed']
    h.check_bindings(failed['source_bindings'])
    sources = {str(p): h.binding(p) for p in (*PINS, HERE/'ADMISSION.json')}
    sources.update(admission['source_bindings'])
    directory = HERE/'diagnosis'
    directory.mkdir(exist_ok=False)
    h.write_json(directory/'started.json', dict(utc=h.utc(), pid=os.getpid(), executable=sys.executable,
        nice=os.getpriority(os.PRIO_PROCESS, 0), versions=VERSIONS, sources=sources,
        blas_env={k: os.environ.get(k) for k in BLAS}, watchdog_s=60,
        rss_limit_bytes=1_000_000_000, truth_files_accessed=False, final_score_evidence=False,
        original_qualification_remains_failed=True, fixture=dict(rank=100, layer=8,
            feature=21, units=[0, 257, 514, 771], epsilon_factors=[.001, .0005])))
    result = dict(passed=False, source_bindings=sources, probes=[], units=[],
                  truth_files_accessed=False, final_score_evidence=False,
                  original_qualification_remains_failed=True, original_failure_sha256=FAILURE_SHA)
    timer = time.perf_counter()

    def timeout(signum, frame):
        raise TimeoutError('Fixed60s scalar interval diagnosis watchdog')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(60)
    try:
        import numpy as np
        from threadpoolctl import threadpool_info, threadpool_limits
        import scalar_interval as interval
        scalar = module(OLD/'feature_second_jet_reference.py', 'diagnosis_frozen_scalar_jet')

        def resources():
            assert h.peak_rss_bytes() < 1_000_000_000 and time.perf_counter()-timer < 60

        def bounds(value):
            assert value.lo.is_finite() and value.hi.is_finite() and value.lo <= value.hi
            return dict(lower=str(value.lo), upper=str(value.hi))

        def norm(values):
            return interval.outward_l2(values)

        def anchor(base, increment):
            return np.where(increment == 0, base, base+increment)

        def reproduce(actual, expected):
            error = abs(actual-expected)
            return dict(actual=actual, expected=expected, error=error,
                        passed=error <= 1e-18+1e-12*abs(expected))

        with threadpool_limits(limits=1):
            pools = threadpool_info()
            assert all(p['num_threads'] == 1 for p in pools if p.get('user_api') == 'blas')
            result['threadpools'] = pools
            result['blas_attestation'] = 'Environment1; detected pools checked if present. Empty list is not observed pool evidence.'
            with np.load(OLD/'capture/rank100_layer08.npz', allow_pickle=False) as archive:
                frame = dict(archive)
            mu, cov, k21, k4 = (frame[k] for k in ('mu', 'pre_cov', 'pre_k21', 'pre_k4'))
            sigma = frame['aux_sigma']
            v = sigma*sigma
            denominator = v*v
            raw = scalar.raw_jets(mu, cov, k21, k4)
            assert np.array_equal(raw['sigma'], sigma)
            base_scalar = frame['aux_post_k4']
            base_uncast = base_scalar/denominator
            base_feature = frame['features'][:, 21].astype(np.float64)
            ids = np.array([0, 257, 514, 771])
            ad1 = (raw['post_k4'].first/denominator)[ids]*sigma[ids]
            ad2 = (raw['post_k4'].second/denominator)[ids]*sigma[ids]**2
            assert np.isfinite(ad1).all() and np.isfinite(ad2).all()

            def actual_feature(new_mu):
                current = scalar.raw_jets(new_mu, cov, k21, k4)
                shifted = anchor(base_scalar, current['post_k4'].value-raw['post_k4'].value)
                return anchor(base_feature, shifted/denominator-base_uncast)

            assert actual_feature(mu).tobytes() == base_feature.tobytes()
            parameters = [dict(mu=float(mu[j]), rawv=float(cov[j, j]), sigma=float(sigma[j]),
                              k3=float(k21[j, j]), k4=float(k4[j]),
                              feature21_denominator=float(denominator[j])) for j in ids]
            reference = [interval.evaluate(**p, order=3) for p in parameters]
            d1 = [r['feature21_mu_derivatives'][1]*interval.point(float(sigma[j]))
                  for j, r in zip(ids, reference)]
            d2 = [r['feature21_mu_derivatives'][2]*interval.point(float(sigma[j]**2))
                  for j, r in zip(ids, reference)]
            ad1_error = norm([interval.point(float(a))-r for a, r in zip(ad1, d1)])
            ad2_error = norm([interval.point(float(a))-r for a, r in zip(ad2, d2)])
            tight = interval.point(Decimal('1e-9')).hi
            result['ad_first_error'] = bounds(ad1_error)
            result['ad_second_error'] = bounds(ad2_error)
            result['ad_accuracy_passed'] = ad1_error.hi <= tight and ad2_error.hi <= tight
            for j, p, r, a, b in zip(ids, parameters, reference, ad1, ad2):
                result['units'].append(dict(unit=int(j), parameters=p, actual_ad_first=float(a),
                    actual_ad_second=float(b),
                    true_feature21_alpha_derivatives=[bounds(t) for t in r['feature21_alpha_derivatives']],
                    primitive_diagnostics=r['primitives']))
            resources()
            for index, factor in enumerate((.001, .0005)):
                f1, f2, ideal1, ideal2, component_rows = [], [], [], [], []
                for j, p, ref in zip(ids, parameters, reference):
                    epsilon = factor*sigma[j]
                    shift = np.zeros_like(mu)
                    shift[j] = epsilon
                    plus, minus = mu+shift, mu-shift
                    xp, xm = actual_feature(plus), actual_feature(minus)
                    f1.append(((xp[j]-xm[j])/(2*epsilon))*sigma[j])
                    f2.append(((xp[j]-2*base_feature[j]+xm[j])/(epsilon*epsilon))*sigma[j]**2)
                    ip = interval.evaluate(**dict(p, mu=float(plus[j])), order=0)
                    im = interval.evaluate(**dict(p, mu=float(minus[j])), order=0)
                    vp, vm, v0 = (r['feature21_alpha_derivatives'][0] for r in (ip, im, ref))
                    one = (vp-vm)/interval.point(float(2*epsilon))*interval.point(float(sigma[j]))
                    two = (vp-2*v0+vm)/interval.point(float(epsilon*epsilon))*interval.point(float(sigma[j]**2))
                    ideal1.append(one)
                    ideal2.append(two)
                    component_rows.append(dict(unit=int(j), epsilon=float(epsilon),
                        rounded_mu_plus=float(plus[j]), rounded_mu_minus=float(minus[j]),
                        original_second_denominator=float(epsilon*epsilon),
                        ideal_first_fd=bounds(one), ideal_second_fd=bounds(two),
                        ideal_first_bias=bounds(one-d1[len(ideal1)-1]),
                        original_float_first_fd=float(f1[-1]), original_float_second_fd=float(f2[-1])))
                    resources()
                f1, f2 = np.array(f1), np.array(f2)
                assert np.isfinite(f1).all() and np.isfinite(f2).all()
                old = probes[index]['feature_first'][21]
                reproduction = dict(
                    first_error_norm=reproduce(float(np.linalg.norm(ad1-f1)), old['error_norm']),
                    first_reference_norm=reproduce(float(np.linalg.norm(f1)), old['reference_norm']))
                allowance = old['absolute']+old['tolerance']*old['reference_norm']
                bias = norm([a-b for a, b in zip(ideal1, d1)])
                evaluation = norm([interval.point(float(a))-b for a, b in zip(f1, ideal1)])
                second_evaluation = norm([interval.point(float(a))-b for a, b in zip(f2, ideal2)])
                # Interval subtraction keeps a downward-rounded lower bound.
                explained = bias-evaluation-ad1_error
                rec = dict(epsilon_factor=factor, reproduction=reproduction,
                    reproduction_passed=all(v['passed'] for v in reproduction.values()),
                    original_allowance=allowance, ideal_first_fd_bias=bounds(bias),
                    floating_first_fd_evaluation_error=bounds(evaluation),
                    floating_second_fd_evaluation_error=bounds(second_evaluation),
                    ideal_bias_minus_both_evaluation_errors=bounds(explained),
                    floating_first_evaluation_passed=evaluation.hi <= tight,
                    coarse_failure_explained=(explained.lo > interval.point(allowance).hi) if index == 0 else None,
                    components=component_rows)
                result['probes'].append(rec)
                print(json.dumps(dict(epsilon_factor=factor, reproduction=rec['reproduction_passed'],
                    ideal_first_fd_bias=rec['ideal_first_fd_bias'],
                    floating_first_fd_evaluation_error=rec['floating_first_fd_evaluation_error'],
                    coarse_failure_explained=rec['coarse_failure_explained'])), flush=True)
                if not rec['reproduction_passed']:
                    raise RuntimeError('Original failed comparison was not reproduced; no next probe')
            h.check_bindings(sources)
            h.check_bindings(failed['source_bindings'])
            resources()
            result['passed'] = bool(result['ad_accuracy_passed']
                and all(p['reproduction_passed'] and p['floating_first_evaluation_passed'] for p in result['probes'])
                and result['probes'][0]['coarse_failure_explained'])
    except BaseException:
        result['error'] = traceback.format_exc()
    finally:
        signal.alarm(0)
    result.update(completed_utc=h.utc(), wall_s=time.perf_counter()-timer, rss_bytes=h.peak_rss_bytes(),
        qualification='Scalar numerical-reference diagnosis only. Does not rerun, override or pass the frozen whole certificate; no gate truth, covariance action, final raw or adjusted evidence.')
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
        parser.error('Explicit --run and sealed diagnosis admission required')
    raise SystemExit(run())
