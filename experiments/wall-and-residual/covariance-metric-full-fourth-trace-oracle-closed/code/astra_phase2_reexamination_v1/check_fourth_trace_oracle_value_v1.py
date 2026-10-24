"""Fixed independent-replica value of a full oracle fourth-trace core."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
from current_pair_oracle_math_v1 import exact_change, independent_fourth_map
from fourth_trace_distinct_necessity_v2 import inverse_core, core_trace, relative
from thin_fourth_wide_helpers_v1 import radial_query

HERE = Path(__file__).resolve().parent
OUT = HERE/'fourth_trace_oracle_value_v1.json'
assert not OUT.exists() and not OUT.with_suffix('.npz').exists()
input_path = HERE/'fourth_trace_oracle_inputs_r35_v1.npz'
input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
assert input_hash == json.loads(input_path.with_suffix('.json').read_text())['output_sha256']
f = dict(np.load(input_path))
truth = {}
provenance = {}
for stream in ('A', 'B'):
    path = HERE/f'fourth_trace_oracle_r35_{stream}_v1.npz'
    record = json.loads(path.with_suffix('.json').read_text())
    assert record['passed'] and record['frozen_inputs_sha256'] == input_hash
    assert record['output_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    truth[stream] = dict(np.load(path))
    provenance[stream] = record['output_sha256']
assert int(truth['A']['seed_xor']) != int(truth['B']['seed_xor'])
assert json.loads((HERE/'current_pair_oracle_certificate_v1.json').read_text())['passed']

start = time.time()
saved = {}
records = []
with threadpool_limits(limits=3):
    C = (f['post12_covariance']+f['post12_covariance'].T)/2
    radial_R = float(f['post12_radial'])*radial_query(C, np.eye(1024))
    w13 = f['W13'].astype(np.float64)
    w14 = f['W14'].astype(np.float64)
    metric = w13.T@C@w13
    s = f['pre13_sigma']
    alpha = f['pre13_alpha']
    pdf = f['pre13_pdf']
    gate = f['pre13_gate']
    R = f['pre13_Cpre']/np.outer(s, s)
    common = (R, s, alpha, pdf, gate, f['pre13_gaussian_mean']/s, f['pre13_m']/s)
    fn = independent_fourth_map()
    baseline_q31 = f['pre13_k31']/(s[:, None]**3*s[None, :])
    np.fill_diagonal(baseline_q31, 0)
    baseline_m, baseline_C, *_ = fn(f['pre13_mu'], f['pre13_Cpre'], f['pre13_K21'], f['pre13_k4'], f['pre13_k22'], 'covariance', q31_override=baseline_q31)
    assert np.max(np.abs(baseline_m-f['pre13_m'])) <= 1e-12
    assert np.max(np.abs(baseline_C-f['pre13_C'])) <= 1e-12

    def change(q22, q31, k4):
        dq22 = q22/np.outer(s*s, s*s)
        dq31 = q31/(s[:, None]**3*s[None, :])
        np.fill_diagonal(dq22, 0)
        np.fill_diagonal(dq31, 0)
        dq4 = k4/s**4
        dm, dc = exact_change(common, dQ22=dq22, dQ31=dq31, dq4=dq4)
        mcheck, ccheck, *_ = fn(f['pre13_mu'], f['pre13_Cpre'], f['pre13_K21'], f['pre13_k4']+k4,
                              f['pre13_k22']+q22, 'covariance', q31_override=baseline_q31+dq31)
        discrepancy = max(float(np.max(np.abs(mcheck-baseline_m-dm))),
                          float(np.max(np.abs((ccheck-baseline_C+ccheck.T-baseline_C.T)/2-dc))))
        assert discrepancy <= 1e-12, discrepancy
        value = np.sum(w14*(dc@w14), axis=0)*f['response14']
        return dm, dc, value, discrepancy

    for stream in ('A', 'B'):
        t = truth[stream]
        residual = (t['R12']+t['R12'].T)/2-radial_R
        A = inverse_core(C, residual)
        reconstruction = relative(core_trace(A, C), residual)
        assert reconstruction <= 1e-8, reconstruction
        pushed = w13.T@A@w13
        av, bv = np.diag(pushed), np.diag(metric)
        q22 = (np.outer(av, bv)+np.outer(bv, av)+4*pushed*metric)/6
        q31 = (av[:, None]*metric+bv[:, None]*pushed)/2
        k4 = av*bv
        pred = change(q22, q31, k4)
        target = change(t['Q22']-f['pre13_k22'], t['Q31']-f['pre13_k31'], t['k4']-f['pre13_k4'])
        for name, values in (('prediction', pred), ('target', target)):
            saved[f'{name}_{stream}_mean'] = values[0]
            saved[f'{name}_{stream}_covariance'] = values[1]
            saved[f'{name}_{stream}_next_mean'] = values[2]
        saved[f'core_{stream}'] = A
        records.append(dict(stream=stream, core_reconstruction_relative=reconstruction,
                            exact_response_direct_map_max=max(pred[3], target[3]),
                            core_norm=float(np.linalg.norm(A)), fourth_trace_residual_norm=float(np.linalg.norm(residual))))

    consumers = {}
    for consumer in ('covariance', 'next_mean'):
        ta, tb = (saved[f'target_{stream}_{consumer}'] for stream in ('A', 'B'))
        pa, pb = (saved[f'prediction_{stream}_{consumer}'] for stream in ('A', 'B'))
        ea, eb = ta-pa, tb-pb
        signal = float(np.sum(ta*tb))
        error_signal = float(np.sum(ea*eb))
        noise = float(np.sum((ta-tb)**2)/2)
        cross_a_b = float(np.sum((pa-tb)**2)/np.sum(tb*tb))
        cross_b_a = float(np.sum((pb-ta)**2)/np.sum(ta*ta))
        ratio = error_signal/signal if signal > 0 else None
        passed = signal > 0 and ratio <= .50 and max(cross_a_b, cross_b_a) <= .65
        consumers[consumer] = dict(passed=passed, entries=int(ta.size),
            baseline_replica_cross_energy=signal, residual_replica_cross_energy=error_signal,
            signal_residual_ratio=ratio, target_replica_noise_energy=noise,
            prediction_replica_noise_energy=float(np.sum((pa-pb)**2)/2),
            target_signal_to_noise=signal/noise if noise > 0 else None,
            ordinary_cross_predictionA_targetB=cross_a_b, ordinary_cross_predictionB_targetA=cross_b_a,
            qualification='Independent-replica point estimates, no confidence interval. Target is the fourth-only exact pair response, not total covariance/mean error.')
    result = dict(passed=all(c['passed'] for c in consumers.values()),
                  positive_signal_qualified=all(c['baseline_replica_cross_energy'] > 0 for c in consumers.values()),
                  records=records, consumers=consumers, rank=35, source_post_layer=12, consumer_layer=13,
                  next_mean_layer=14, samples_per_stream=8000000, frozen_input_sha256=input_hash,
                  truth_source_sha256=provenance, wall_s=time.time()-start,
                  peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  utc_finished=datetime.now(timezone.utc).isoformat(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  qualification='One fixed full-rank covariance-metric oracle trace core. Actual bank prefix/inputs frozen; K3 fixed. New fourth-only response statistics from two independent reused8M streams. No trained estimator, complete-carrier accuracy, adjusted score, panel, or submission.')
    np.savez(OUT.with_suffix('.npz'), **saved)
    OUT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
