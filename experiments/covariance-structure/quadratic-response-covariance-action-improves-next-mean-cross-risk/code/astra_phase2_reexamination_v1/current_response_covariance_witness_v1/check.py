"""Inert fixed post12 covariance-only witness; no fit or final-score claim."""
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
P2 = NEW.parent
Q = NEW/'current_response_quadratic_v1'
T = NEW/'current_response_certified_native_v1'
ALG = NEW/'current_response_covariance_algebra_v1'
ENV = P2/'data_gate/.venv'
CAPTURE_SHA = '575727d6f465cbdfeeb9f58d201e9a6e2983aef94dcd2d2389108c58fe569b42'
QUALIFICATION_SHA = '52938606d3f116f5f41021fd19f5f3b03f8b9d7c2713741d0d19cbada4221494'
GATE_SHA = 'b549ad49f54d134d4110dcc2c476e2f892b2e3018fb6001958bc50750a7f1f07'
ALGEBRA_SHA = '01802bd2032803b88028f8bbe85351a0f4fd66c1fb6fda1e9b2eac14aacd3792'
TRUTHS = (
    P2/'public_g19_attribution_v1/mps_cov2_rank0035.npz',
    NEW/'covariance_replica_truth_v1/rank_0035_8M_v1.npz')
TRUTH_SHAS = (
    '7e6b97166885de0cdd870a835ae536830fb6da41ca345d8a66e2eb8d0d6380e6',
    'e6a7787d974c6b56f39a4444bd54c793ff3efa6e62acb55054774ddb2a2e3358')
VERSIONS = dict(numpy='2.4.6', flopscope='0.12.0', whestbench='0.16.0', threadpoolctl='3.6.0')
BLAS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
PINS = {
    Q/'capture/result.json': CAPTURE_SHA,
    Q/'capture/rank35_layer12.npz': '58e9d41f4c6449d56ab172107c32e7632300a7902392bead6bc5e9be066607d2',
    Q/'capture/rank35_layer13.npz': 'a2b3f087c66fb24a1d5bb1ac14db9efd63bd41a81b85a3471dd97f34ef8354b3',
    Q/'quadratic_covariance_reference.py': 'baa80b74e065e0e974b734a43a2d605de497c369cc32e03de2df1d5bca214527',
    Q/'feature_second_jet_reference.py': 'dd4484fc3baaefb783ff54497b43ff801214e0583cd16d28a9d9b4c212cb218f',
    T/'qualification/result.json': QUALIFICATION_SHA,
    T/'qualification/rank35_responses.npz': 'b9691a8cf464392bcb2409f87f96ee08e0e9c0959fc7574c27b52f3e24d59d47',
    T/'gate_information/result.json': GATE_SHA,
    ALG/'certificate/result.json': ALGEBRA_SHA,
    NEW/'check_response_rows_adjusted_v1.py': '8e99890bdc7742164a53303a23691e9f1cc54909c9cccc9cf64fc362eb459304',
    NEW/'current_pair_oracle_attribution_v2/rank_0035.json': 'a4c2500db4fed72972d917cbfb41cb3d1dd53c789591a9a5cd80788b26b3c0cd',
    NEW/'covariance_innovation_replica_r35_v1.json': '5642a89dcd025bc50417c5743268695acef583b971071a0bcb7f4b3bacbc3e5e',
    NEW/'covariance_replica_truth_v1/rank_0035_8M_v1.json': '133e3720488415b2a86790466fa4f8475352b6a2de7526830c596fbe794b303a',
    P2/'public_g19_attribution_v1/mc_truth_mps_indep.py': '5b887ec3ef484f1e715273a375dd4358fa57165f9a955b9061c6c5a228087cb5',
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
    assert all(sha(p) == expected for p, expected in PINS.items())
    h = module(NEW/'check_response_rows_adjusted_v1.py', 'fixed_covariance_witness_receipts')
    admission = h.load_json(HERE/'ADMISSION.json')
    assert admission['admitted'] and admission['checker_sha256'] == sha(Path(__file__))
    h.check_bindings(admission['source_bindings'])
    required = {str((HERE/n).resolve()) for n in ('PROTOCOL.md', 'check.py')}
    assert required <= {v['path'] for v in admission['source_bindings'].values()}
    prerequisites = {}
    for name, path in (('capture', Q/'capture/result.json'),
                       ('qualification', T/'qualification/result.json'),
                       ('gate', T/'gate_information/result.json'),
                       ('algebra', ALG/'certificate/result.json')):
        value, completion = h.load_json(path), h.load_json(path.parent/'completion.json')
        assert value['passed'] and completion['passed'] and completion['result_sha256'] == PINS[path]
        prerequisites[name] = value
    capture = prerequisites['capture']
    assert capture['candidate_sha256'] == '409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f'
    assert [r['rank'] for r in capture['trials']] == [35, 100]
    assert all(r['all16_byte_exact'] and r['flops'] == r['reference_flops'] for r in capture['trials'])
    for name in ('sources', 'fixture_bindings', 'artifacts'):
        h.check_bindings(capture[name])
    for name in ('qualification', 'gate', 'algebra'):
        h.check_bindings(prerequisites[name]['source_bindings'])
    assert len(prerequisites['qualification']['records']) == 14
    assert all(r['passed'] for r in prerequisites['qualification']['records'])
    assert capture['fixture_bindings']['weights_35']['sha256'] == '14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2'
    row = h.load_json(capture['fixture_bindings']['row_35']['path'])
    assert int(row['mlp_seed']) == 2758000291903877946
    metadata = h.load_json(NEW/'covariance_replica_truth_v1/rank_0035_8M_v1.json')
    assert metadata['rank'] == 35 and metadata['samples'] == 8000000
    assert metadata['seed_xor'] == '0x4153545241434f56' and metadata['output_sha256'] == TRUTH_SHAS[1]
    archive = h.load_json(NEW/'current_pair_oracle_attribution_v2/rank_0035.json')
    assert archive['provenance']['mps_cov2_rank0035.npz']['sha256'] == TRUTH_SHAS[0]
    sources = {str(p): h.binding(p) for p in (*PINS, HERE/'ADMISSION.json')}
    sources.update(admission['source_bindings'])
    directory = HERE/'witness'
    directory.mkdir(exist_ok=False)
    h.write_json(directory/'started.json', dict(utc=h.utc(), pid=os.getpid(), executable=sys.executable,
        nice=os.getpriority(os.PRIO_PROCESS, 0), versions=VERSIONS, source_bindings=sources,
        blas_env={k: os.environ.get(k) for k in BLAS}, watchdog_s=60, rss_limit_bytes=1_000_000_000,
        fixed_fixture=dict(rank=35, post_layer=12, next_layer=13, amplitude=1), no_fit=True,
        final_score_evidence=False, covariance_action_only=True))
    result = dict(passed=False, source_bindings=sources, records=[], artifacts={}, target_bindings={},
                  no_fit=True, final_score_evidence=False, covariance_action_only=True,
                  historical_original_execution_seal_available=False,
                  target_provenance_qualification='Archived original stream lacks a found contemporaneous source/seed/weight execution seal. Pinned bytes/rank and affine consistency are checked, not relabeled as that missing seal. Independent streams share MPS float32 arithmetic bias.')
    timer = time.perf_counter()

    def timeout(signum, frame):
        raise TimeoutError('Fixed60s covariance-consumer witness watchdog')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(60)
    try:
        import numpy as np
        from threadpoolctl import threadpool_info, threadpool_limits
        module(Q/'feature_second_jet_reference.py', 'feature_second_jet_reference')
        q = module(Q/'quadratic_covariance_reference.py', 'fixed_current_covariance_delta')

        def resources():
            assert h.peak_rss_bytes() < 1_000_000_000 and time.perf_counter()-timer < 60

        def record(name, passed, **values):
            result['records'].append(dict(name=name, passed=bool(passed), **values))
            if not passed:
                raise RuntimeError('Fixed covariance witness guard failed: '+name)
            resources()

        def finite(*arrays):
            return all(np.isfinite(a).all() for a in arrays)

        def fixed_mask(matrix, alive):
            diagonal = np.diag(matrix).copy()
            answer = matrix*(alive[:, None]*alive[None, :])
            np.fill_diagonal(answer, diagonal)
            return answer

        def cross_risk(a, b, delta):
            denominator = float(np.mean(a*b))
            numerator = float(np.mean((a-delta)*(b-delta)))
            if not (np.isfinite(denominator) and np.isfinite(numerator)):
                raise FloatingPointError('Nonfinite covariance-consumer risk')
            ratio = numerator/denominator if denominator > 0 else None
            if ratio is not None and not np.isfinite(ratio):
                raise FloatingPointError('Nonfinite covariance-consumer risk ratio')
            return dict(reference=denominator, corrected=numerator,
                        ratio=ratio)

        with threadpool_limits(limits=1):
            pools = threadpool_info()
            assert all(p['num_threads'] == 1 for p in pools if p.get('user_api') == 'blas')
            result['threadpools'] = pools
            result['blas_attestation'] = 'Environment1; detected pools checked. Empty inventory is not observed pool evidence.'
            weights = np.load(capture['fixture_bindings']['weights_35']['path'], mmap_mode='r', allow_pickle=False)
            record('weights_shape_dtype', weights.shape == (16, 1024, 1024) and weights.dtype == np.float32)
            with np.load(Q/'capture/rank35_layer12.npz', allow_pickle=False) as data:
                frame = dict(data)
            with np.load(Q/'capture/rank35_layer13.npz', allow_pickle=False) as data:
                next_frame = dict(data)
            with np.load(T/'qualification/rank35_responses.npz', allow_pickle=False) as data:
                record('fixed_response_layers', np.array_equal(data['layers'], np.arange(8, 15)))
                a = data['delta_derivative'][4].copy()
                b = .5*data['delta_second_derivative'][4]
            record('coefficient_shape_finite', a.shape == b.shape == (1024,) and finite(a, b))
            reference = q.prepare_reference(frame['mu'], frame['pre_cov'], frame['pre_k21'],
                                            frame['pre_k4'], frame['pre_k22'])
            record('raw_variance_domain', reference.below_sigma_floor_count == 0)
            delta_raw = q.covariance_delta(reference, a, b)['covariance']
            alive = frame['alive']
            record('captured_mask', alive.dtype == np.bool_ and alive.shape == (1024,))
            host, dm = frame['host_mean'], frame['head_delta'].astype(np.float64)
            current = frame['host_cov']-np.outer(host, dm)-np.outer(dm, host)-np.outer(dm, dm)
            current = fixed_mask(current, alive)
            delta = fixed_mask(delta_raw, alive)
            w = weights[13].astype(np.float64)
            omega = next_frame['aux_pdf']/(2*next_frame['aux_sigma'])
            record('finite_prediction', finite(current, delta, w, omega) and np.all(omega >= 0))
            record('all_diagonals_retained', np.array_equal(np.diag(delta), np.diag(delta_raw)))
            frame_error = float(np.linalg.norm(w.T@(host+dm)-next_frame['mu']))
            frame_norm = float(np.linalg.norm(next_frame['mu']))
            record('current_next_mean_frame', frame_error <= 1e-10+1e-10*frame_norm,
                   error_norm=frame_error, reference_norm=frame_norm)

            def consumer(matrix):
                return omega*np.sum(w*(matrix@w), axis=0)

            signal_delta = consumer(delta)
            record('finite_frozen_consumer', signal_delta.shape == (1024,) and finite(signal_delta))
            prediction_path = directory/'prediction.npz'
            with prediction_path.open('xb') as handle:
                np.savez(handle, a=a, b=b, current_covariance=current, covariance_delta=delta,
                         omega=omega, consumer_delta=signal_delta, current_mean=host+dm,
                         alive=alive, rank=np.array(35), post_layer=np.array(12), amplitude=np.array(1))
            result['artifacts']['prediction'] = h.binding(prediction_path)
            result['prediction_frozen_before_targets'] = dict(utc=h.utc(), **result['artifacts']['prediction'])
            result['action_magnitude'] = dict(a_rms=float(np.sqrt(np.mean(a*a))),
                two_b_sigma_rms=float(np.sqrt(np.mean((2*b*reference.sigma)**2))),
                covariance_delta_rms=float(np.sqrt(np.mean(delta*delta))),
                consumer_delta_rms=float(np.sqrt(np.mean(signal_delta*signal_delta))),
                alive_count=int(alive.sum()))
            del frame, next_frame, reference, delta_raw
            errors, covariance_errors, streams = [], [], []
            for index, (path, expected) in enumerate(zip(TRUTHS, TRUTH_SHAS)):
                assert h.digest(path) == expected
                result['target_bindings'][str(index)] = h.binding(path)
                with np.load(path, allow_pickle=False) as data:
                    record(f'truth{index}/rank_N', data['rank'].shape == () and int(data['rank']) == 35
                           and data['P'].shape == () and int(data['P']) == 8000000)
                    full_cov = data['mc_post_cov']
                    pre_mean, post_mean = data['mc_pre_m'], data['mc_post']
                    record(f'truth{index}/shape_dtype', full_cov.shape == (16, 1024, 1024)
                           and full_cov.dtype == np.float32 and pre_mean.shape == post_mean.shape == (16, 1024))
                    record(f'truth{index}/finite', finite(full_cov, pre_mean, post_mean))
                    target = full_cov[12].astype(np.float64)*(8000000/7999999)
                    for layer in range(1, 16):
                        predicted = weights[layer].astype(np.float64).T@post_mean[layer-1]
                        error = float(np.linalg.norm(predicted-pre_mean[layer]))
                        norm = float(np.linalg.norm(pre_mean[layer]))
                        record(f'truth{index}/affine_mean_layer{layer}', error <= 1e-8+1e-5*norm,
                               error_norm=error, reference_norm=norm, absolute=1e-8, relative=1e-5)
                    del full_cov, pre_mean, post_mean
                error_cov = target-current
                error_mean = consumer(error_cov)
                record(f'truth{index}/finite_consumer', finite(error_cov, error_mean))
                covariance_errors.append(error_cov)
                errors.append(error_mean)
                streams.append(dict(stream=('A', 'B')[index], path=str(path), sha256=expected,
                    samples=8000000, finite_N_covariance_factor=8000000/7999999,
                    historical_weight_execution_seal_available=False,
                    replica_source_seed_metadata_available=(index == 1),
                    consumer=cross_risk(error_mean, error_mean, signal_delta),
                    covariance=cross_risk(error_cov, error_cov, delta)))
                resources()
            primary = cross_risk(*errors, signal_delta)
            secondary = cross_risk(*covariance_errors, delta)
            result.update(primary_consumer=primary, secondary_covariance=secondary, streams=streams,
                consumer_inner_products=dict(signal_energy=float(np.mean(signal_delta**2)),
                    signal_error_A=float(np.mean(signal_delta*errors[0])),
                    signal_error_B=float(np.mean(signal_delta*errors[1])),
                    half_stream_disagreement_energy=float(np.mean((errors[0]-errors[1])**2)/2),
                    qualifications='Two reused development references; independent sampling correction in expectation only. No covariance-batch jackknife or confidence interval; common FP32 bias remains.'))
            positive = primary['reference'] > 0 and all(s['consumer']['reference'] > 0 for s in streams)
            eligible = positive and primary['ratio'] <= .75 and all(s['consumer']['ratio'] <= .85 for s in streams)
            result['decision'] = dict(passed=bool(eligible), denominators_positive=positive,
                fixed_cross_limit=.75, fixed_each_stream_limit=.85,
                no_strength_layer_curvature_rescue=True)
            h.check_bindings(result['artifacts'])
            h.check_bindings(result['target_bindings'])
            h.check_bindings(sources)
            for name in ('sources', 'fixture_bindings', 'artifacts'):
                h.check_bindings(capture[name])
            for name in ('qualification', 'gate', 'algebra'):
                h.check_bindings(prerequisites[name]['source_bindings'])
            resources()
            result['passed'] = bool(eligible and all(r['passed'] for r in result['records']))
    except BaseException:
        result['error'] = traceback.format_exc()
    finally:
        signal.alarm(0)
    result.update(completed_utc=h.utc(), wall_s=time.perf_counter()-timer, rss_bytes=h.peak_rss_bytes(),
        qualification='Fixed current-P post12 covariance-only FP64 Gaussian-consumer witness. Reused development truths with explicitly limited historical provenance and common MPS-FP32 bias. Not native propagation, full feedback, final raw MSE, complete compute or adjusted-score evidence.')
    seal = h.write_json(directory/'result.json', result)
    h.write_json(directory/'completion.json', dict(passed=result['passed'], result_sha256=seal))
    print(json.dumps(dict(passed=result['passed'], result_sha256=seal, primary=result.get('primary_consumer'),
        streams=[s['consumer'] for s in result.get('streams', [])], wall_s=result['wall_s'],
        rss_bytes=result['rss_bytes'], error=result.get('error'))), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        parser.error('Explicit --run and independent admission required; inert by default')
    raise SystemExit(run())
