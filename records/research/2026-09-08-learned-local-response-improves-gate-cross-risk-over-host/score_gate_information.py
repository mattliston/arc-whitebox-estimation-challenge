"""Conditional one-shot current response information test; no fit or carrier.

This file is inert without --run and a sealed GATE_ADMISSION.json binding a
passing numerical qualification. The old gate truths are opened only then.
This does not measure final mean MSE, complete compute or adjusted score.
Additive repointing of the frozen native-block gate scorer to the independently
certified scalar-reference successor; scientific thresholds are unchanged.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

from check import module, sha, VERSIONS, BLAS, ENV, HERE, NEW, OLD, CAPTURE_SHA

NAMES = ('gaussian_gate', 'host_response', 'learned_response')
META_PINS = {
    (35, 'A'): 'ab84e121a49a2829a316aa7be87efd115fd5d89d187590b29c6b868893ab991e',
    (35, 'B'): '0aa23a408088963650d0b0e307748e5bb3e38556fc5db8bba56b038b3abc7c2f',
    (100, 'A'): '7de10c8fdb2276574d5cef9e348a74205d8c02b7a7b03678d1d306f6708d75ab',
    (100, 'B'): 'c37732cfaa6e8b3d5ad1bf24114ad0f9a0b59e5b52456e365acefe2a60be107a',
}


def run():
    assert Path(sys.prefix).resolve() == ENV.resolve() and 'anaconda3' not in sys.executable
    assert os.getpriority(os.PRIO_PROCESS, 0) >= 10
    assert all(os.environ.get(k) == '1' for k in BLAS)
    assert os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
    assert {k: importlib.metadata.version(k) for k in VERSIONS} == VERSIONS
    h = module(NEW/'check_response_rows_adjusted_v1.py', 'rounding_gate_receipts')
    admission = h.load_json(HERE/'GATE_ADMISSION.json')
    assert admission['admitted'] and admission['checker_sha256'] == sha(Path(__file__))
    h.check_bindings(admission['source_bindings'])
    qpath = HERE/'qualification/result.json'
    assert sha(qpath) == admission['qualification_sha256']
    qualification = h.load_json(qpath)
    completion = h.load_json(HERE/'qualification/completion.json')
    assert qualification['passed'] and completion['passed']
    assert completion['result_sha256'] == admission['qualification_sha256']
    assert len(qualification['records']) == 14 and all(r['passed'] for r in qualification['records'])
    assert qualification['capture_sha256'] == CAPTURE_SHA
    h.check_bindings(qualification['source_bindings'])
    h.check_bindings(qualification['artifacts'])
    assert sha(OLD/'capture/result.json') == CAPTURE_SHA
    capture = h.load_json(OLD/'capture/result.json')
    metadata = {}
    for (rank, stream), digest in META_PINS.items():
        path = NEW/f'mean_response_gate_truth_r{rank}_{stream}_v1.json'
        assert sha(path) == digest
        row = h.load_json(path)
        assert row['rank'] == rank and row['stream'] == stream
        assert row['samples'] == 2**20 and row['batch'] == 65536
        assert row['layers'] == list(range(8, 15))
        assert row['weights_sha256'] == capture['fixture_bindings'][f'weights_{rank}']['sha256']
        metadata[(rank, stream)] = row
    assert all(metadata[(rank, 'A')]['seed_xor'] != metadata[(rank, 'B')]['seed_xor'] for rank in (35, 100))
    sources = dict(admission['source_bindings'])
    sources.update({str(p): h.binding(p) for p in (Path(__file__), HERE/'GATE_ADMISSION.json', qpath)})
    directory = HERE/'gate_information'
    directory.mkdir(exist_ok=False)
    h.write_json(directory/'started.json', dict(utc=h.utc(), pid=os.getpid(), executable=sys.executable,
        nice=os.getpriority(os.PRIO_PROCESS, 0), versions=VERSIONS,
        qualification_sha256=admission['qualification_sha256'], source_bindings=sources,
        blas_env={k: os.environ.get(k) for k in BLAS}, watchdog_s=60,
        rss_limit_bytes=1_000_000_000, final_score_evidence=False, no_fit=True,
        fixed_gate='Learned cross MSE <=.80 Gaussian and <=.90 host on BOTH nets, all seven late layers equally.'))
    result = dict(passed=False, records=[], source_bindings=sources, target_bindings={},
                  final_score_evidence=False, no_fit=True,
                  qualification_sha256=admission['qualification_sha256'])
    timer = time.perf_counter()

    def timeout(signum, frame):
        raise TimeoutError('Fixed60s saved gate-information test watchdog')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(60)
    try:
        import numpy as np
        from threadpoolctl import threadpool_info, threadpool_limits

        def resources():
            assert h.peak_rss_bytes() < 1_000_000_000 and time.perf_counter()-timer < 60

        def summarize(predictions, a, b):
            mse = np.mean((predictions-a)*(predictions-b), axis=(-2, -1))
            return np.r_[mse, mse[2]-.8*mse[0], mse[2]-.9*mse[1]]

        with threadpool_limits(limits=1):
            pools = threadpool_info()
            assert all(p['num_threads'] == 1 for p in pools if p.get('user_api') == 'blas')
            result['threadpools'] = pools
            result['blas_attestation'] = 'Environment1; detected pools checked if present. Empty list is not observed pool evidence.'
            for rank in (35, 100):
                with np.load(HERE/f'qualification/rank{rank}_responses.npz', allow_pickle=False) as probe:
                    assert np.array_equal(probe['layers'], np.arange(8, 15))
                    pred = np.stack([probe[name] for name in NAMES])
                assert pred.shape == (3, 7, 1024) and np.isfinite(pred).all()
                assert (pred >= 0).all() and (pred <= 1).all()
                batches, probabilities, archives = [], [], []
                for stream in ('A', 'B'):
                    path = NEW/f'mean_response_gate_truth_r{rank}_{stream}_v1.npz'
                    row = metadata[(rank, stream)]
                    assert h.digest(path) == row['output_sha256']
                    result['target_bindings'][f'{rank}_{stream}'] = h.binding(path)
                    with np.load(path, allow_pickle=False) as truth:
                        assert int(truth['samples']) == 2**20 and int(truth['batch_size']) == 65536
                        assert np.array_equal(truth['layers'], np.arange(8, 15))
                        counts = truth['batch_counts']
                        assert counts.shape == (16, 7, 1024) and counts.dtype.kind in 'iu'
                        assert (counts >= 0).all() and (counts <= 65536).all()
                        bp = counts/65536
                        prob = bp.mean(0)
                        assert np.array_equal(prob, truth['probability'])
                    batches.append(bp)
                    probabilities.append(prob)
                    archives.append(dict(stream=stream, sha256=row['output_sha256'],
                        samples=2**20, cpu64_sign_disagreements=row['cpu64_sign_disagreements'],
                        numerical_qualification=row['qualification']))
                summary = summarize(pred, *probabilities)
                variance = np.zeros(5)
                for stream in (0, 1):
                    n = len(batches[stream])
                    deleted = []
                    for j in range(n):
                        pair = list(probabilities)
                        pair[stream] = (n*probabilities[stream]-batches[stream][j])/(n-1)
                        deleted.append(summarize(pred, *pair))
                    deleted = np.stack(deleted)
                    variance += (n-1)/n*np.sum((deleted-deleted.mean(0))**2, axis=0)
                se = np.sqrt(variance)
                assert np.isfinite(summary).all() and np.isfinite(se).all()
                a, b = probabilities
                per_layer = ((pred-a)*(pred-b)).mean(-1)
                stream_mse = [((pred-q)**2).mean(axis=(-2, -1)) for q in probabilities]
                positive = bool(summary[0] > 0 and summary[1] > 0)
                passed = bool(positive and summary[3] <= 0 and summary[4] <= 0)
                rec = dict(rank=rank, passed=passed, denominators_positive=positive,
                    cross_mse={name: float(summary[i]) for i, name in enumerate(NAMES)},
                    cross_mse_jackknife_se={name: float(se[i]) for i, name in enumerate(NAMES)},
                    learned_over_gaussian=float(summary[2]/summary[0]) if summary[0] > 0 else None,
                    learned_over_host=float(summary[2]/summary[1]) if summary[1] > 0 else None,
                    paired_gates=dict(learned_minus_point8_gaussian=float(summary[3]),
                        learned_minus_point8_gaussian_jackknife_se=float(se[3]),
                        learned_minus_point9_host=float(summary[4]),
                        learned_minus_point9_host_jackknife_se=float(se[4])),
                    per_layer=[dict(layer=l, **{name: float(per_layer[i, k])
                        for i, name in enumerate(NAMES)}) for k, l in enumerate(range(8, 15))],
                    independent_stream_noisy_mse=[dict(stream=stream,
                        **{name: float(m[i]) for i, name in enumerate(NAMES)},
                        learned_over_gaussian=float(m[2]/m[0]), learned_over_host=float(m[2]/m[1]))
                        for stream, m in zip(('A', 'B'), stream_mse)], archives=archives)
                result['records'].append(rec)
                print(json.dumps(rec), flush=True)
                resources()
        h.check_bindings(qualification['source_bindings'])
        h.check_bindings(qualification['artifacts'])
        h.check_bindings(sources)
        h.check_bindings(result['target_bindings'])
        resources()
        result['passed'] = len(result['records']) == 2 and all(r['passed'] for r in result['records'])
    except BaseException:
        result['error'] = traceback.format_exc()
    finally:
        signal.alarm(0)
    result.update(completed_utc=h.utc(), wall_s=time.perf_counter()-timer, rss_bytes=h.peak_rss_bytes(),
        qualification='Gate information only, not covariance quality, final raw MSE or adjusted score. Cross-stream products remove sampling-MSE in expectation; common MPS float32 bias remains. Jackknife deletes whole independent input batches.')
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
        parser.error('Explicit --run and post-qualification admission required')
    raise SystemExit(run())
