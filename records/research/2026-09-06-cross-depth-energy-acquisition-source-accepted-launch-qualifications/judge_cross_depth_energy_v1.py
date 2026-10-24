"""Independent NumPy-only fixed energy-bin oracle information judgment.

No new sampling, fits, estimator calls, clipping of moments/cross energies,
or changed primary consumer. Uses only the two frozen acquisition files.
Requires explicit review flag; does not import Torch or load a target store.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np

from cross_depth_energy_statistics_v1 import (
    HERE, WIDTH, SAMPLES, BATCH, BLOCKS, BLOCK_SIZE, RANK, SEED, XORS,
    RNG_SEEDS, THRESHOLDS, ROW, BANK_INPUT, PRIOR_HASHES,
    sha256, file_record, utc_now, verify_frozen_inputs,
    moments, scalar_empirical_algebra, legacy_finish, consumer, gaussian_relu,
)


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2)))


def divide(a,b):
    return float(a/b) if np.isfinite(a) and np.isfinite(b) and b > 0 else None


def energies(a,b):
    return dict(cross=float(np.mean(a*b)),mse_A=float(np.mean(a*a)),
                mse_B=float(np.mean(b*b)),replica_noise=float(.5*np.mean((a-b)**2)))


def comparison(x,bank):
    return dict(cross_ratio=divide(x['cross'],bank['cross']),
                ordinary_ratio_A=divide(x['mse_A'],bank['mse_A']),
                ordinary_ratio_B=divide(x['mse_B'],bank['mse_B']),
                replica_noise_over_bank_cross=divide(x['replica_noise'],bank['cross']))


def load_stream(label):
    path = HERE/f'cross_depth_energy_r35_{label}_v1.npz'
    receipt_path = path.with_suffix('.json')
    record = json.loads(receipt_path.read_text())
    if record['schema'] != 'cross-depth-angular-energy-acquisition-v1' or record['stream'] != label:
        raise ValueError(f'{label}: wrong acquisition identity')
    if sha256(path) != record['output_sha256']:
        raise ValueError(f'{label}: acquisition output digest mismatch')
    for name,item in record['source_and_input_provenance'].items():
        if sha256(name) != item['sha256']:
            raise ValueError(f'{label}: frozen acquisition source/input changed: {name}')
    with np.load(path,allow_pickle=False) as z:
        data = {key:z[key].copy() for key in z.files}
    checks = dict(saved_qualification=bool(record['passed']),
                  count=int(data['samples']) == SAMPLES,
                  batch=int(data['batch']) == BATCH,
                  block_size=int(data['block_size']) == BLOCK_SIZE,
                  rank=int(data['rank']) == RANK,seed=int(data['seed']) == SEED,
                  xor=int(data['seed_xor']) == XORS[label],
                  rng_seed=int(data['rng_seed']) == RNG_SEEDS[label],
                  thresholds=bool(np.array_equal(data['thresholds'],THRESHOLDS)),
                  indices=bool(np.array_equal(data['indices'],np.arange(0,WIDTH,32))),
                  block_shape=data['block_counts'].shape == (BLOCKS,4),
                  sums_shape=data['block_sums'].shape == (BLOCKS,4,4,WIDTH),
                  global_shape=data['global_sums'].shape == (4,WIDTH),
                  exact_block_counts=bool(np.array_equal(np.sum(data['block_counts'],axis=1),
                                                        np.full(BLOCKS,BLOCK_SIZE))),
                  finite=bool(np.all(np.isfinite(data['block_sums']))
                              and np.all(np.isfinite(data['global_sums']))),
                  bin_support=bool(np.all(np.sum(data['block_counts'],axis=0) >= 8192)))
    prior_path = HERE/f'residual_one_high_truth_r35_l12_{label}_v1.npz'
    if sha256(prior_path) != PRIOR_HASHES[label]:
        raise ValueError(f'{label}: legacy target digest mismatch')
    with np.load(prior_path,allow_pickle=False) as z:
        prior = {key:z[key].copy() for key in
                 ('post_mean_residual','next_pre_mean_residual','next_K21','next_shift_actual')}
    replay = legacy_finish(data['legacy_post_sum'],data['legacy_next_sum'],
                           data['legacy_next_second'],data['legacy_next_square'],
                           data['legacy_next_third'],data['indices'],SAMPLES)
    replay_parity = {key:bool(np.array_equal(value,prior[key])
                             and np.array_equal(value,data[key])) for key,value in replay.items()}
    checks['legacy_replay_byte_parity'] = all(replay_parity.values())
    checks['shift'] = bool(np.array_equal(data['shift'],prior['next_shift_actual'].astype(np.float64)))
    total = moments(SAMPLES,data['global_sums'],data['shift'])
    indices = data['indices']
    old_diagonal = prior['next_K21'][np.arange(len(indices)),indices]
    third_error = total['third'][indices]-old_diagonal
    relative_third = float(np.linalg.norm(third_error)/np.linalg.norm(old_diagonal))
    checks['cpu_legacy_third'] = relative_third <= 1e-4
    algebra = scalar_empirical_algebra(data['block_counts'],data['block_sums'],
                                      data['global_sums'],data['shift'])
    checks['empirical_algebra'] = max(algebra.values()) <= 1e-11
    cond = moments(np.sum(data['block_counts'],axis=0),np.sum(data['block_sums'],axis=0),data['shift'])
    checks['nonnegative_conditional_variance'] = bool(np.all(cond['variance'] >= 0)
                                                     and np.all(np.isfinite(cond['variance'])))
    diagnostic = dict(qualifications=checks,legacy_byte_parity=replay_parity,
                      empirical_algebra=algebra,cpu_legacy_third_relative=relative_third,
                      cpu_legacy_third_max_abs=float(np.max(np.abs(third_error))),
                      acquisition=file_record(path),acquisition_receipt=file_record(receipt_path))
    return data,diagnostic


def full_consumer(data):
    result = consumer(data['block_counts'],data['block_sums'],data['shift'])
    # Primary true mean and G0 use the independently accumulated global sums.
    # The partition equivalence was checked above before any interpretation.
    result['total'] = moments(SAMPLES,data['global_sums'],data['shift'])
    result['true'] = result['total']['true_postmean']
    result['G0'] = gaussian_relu(result['total']['mean'],result['total']['variance'])
    return result


def jackknife(data,full,bank):
    leave_G = []
    leave_true = []
    for omitted in range(BLOCKS):
        keep = np.arange(BLOCKS) != omitted
        remaining = consumer(data['block_counts'][keep],data['block_sums'][keep],data['shift'])
        leave_G.append(remaining['G'])
        leave_true.append(remaining['true'])
    leave_G = np.stack(leave_G)
    leave_true = np.stack(leave_true)
    return dict(G_leave=leave_G,true_leave=leave_true,
                g_leave=leave_G-leave_true,b_leave=bank[None,:]-leave_true,
                bias=7*(np.mean(leave_G,axis=0)-full['G']))


def jackknife_scalar_report(a,b):
    a = np.asarray(a,dtype=np.float64)
    b = np.asarray(b,dtype=np.float64)
    variance = float((7/8)*(np.sum((a-np.mean(a))**2)+np.sum((b-np.mean(b))**2)))
    return dict(leave_A=a.tolist(),leave_B=b.tolist(),
                jackknife_variance=variance,jackknife_standard_error=float(np.sqrt(variance)),
                overall_min=float(min(np.min(a),np.min(b))),
                overall_max=float(max(np.max(a),np.max(b))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute-reviewed-judge',action='store_true')
    args = parser.parse_args()
    if not args.execute_reviewed_judge:
        parser.error('Judgment requires root review and --execute-reviewed-judge.')
    if 'torch' in sys.modules:
        raise RuntimeError('Judge must use the NumPy-only interpreter')
    nice = os.getpriority(os.PRIO_PROCESS,0)
    if nice < 10:
        raise RuntimeError(f'launch judge at nice >=10; observed {nice}')
    out = HERE/'cross_depth_energy_judgment_v1.npz'
    receipt_path = out.with_suffix('.json')
    failure_path = out.with_suffix('.failure.json')
    for path in (out,receipt_path,failure_path):
        if path.exists():
            raise FileExistsError(f'preserve existing artifact; refuse overwrite: {path}')
    started = time.time()
    try:
        provenance = verify_frozen_inputs()
        for name in ('cross_depth_energy_statistics_v1.py','acquire_cross_depth_energy_v1.py',Path(__file__).name):
            provenance[str(HERE/name)] = file_record(HERE/name)
        data = {}
        stream_checks = {}
        for label in ('A','B'):
            data[label],stream_checks[label] = load_stream(label)
        all_stream_checks = all(all(check['qualifications'].values()) for check in stream_checks.values())
        base_record = dict(schema='cross-depth-angular-energy-judgment-v1',rank=RANK,pre_layer=13,
                           samples_per_stream=SAMPLES,streams=['A','B'],thresholds=THRESHOLDS.tolist(),
                           stream_checks=stream_checks,source_and_input_provenance=provenance,
                           primary_gates=dict(cross=.50,ordinary_A=.65,ordinary_B=.65,
                                              bank_noise_over_bank_cross=.10,
                                              mixture_noise_over_bank_cross=.10),
                           numpy_version=np.__version__,nice=nice,torch_imported=False)
        if not all_stream_checks:
            result = dict(base_record,interpretation='INCONCLUSIVE_FIDELITY_OR_NUMERICAL',
                          accuracy_consumer_evaluated=False,utc_finished=utc_now(),
                          wall_seconds=time.time()-started,
                          qualification='Replay/support/numerical failure; no accuracy interpretation or extra acquisition.')
            with receipt_path.open('x',encoding='utf-8') as handle:
                json.dump(result,handle,indent=2,allow_nan=False)
                handle.write('\n')
            print(json.dumps(result,indent=2,allow_nan=False),flush=True)
            return

        raw_bank = np.load(ROW/'baseline.npy',mmap_mode='r',allow_pickle=False)[13].copy()
        clipped = np.maximum(raw_bank,0)
        bank = clipped.astype(np.float32).astype(np.float64)
        served = dict(clipped_coordinates=int(np.count_nonzero(raw_bank < 0)),
                      minimum_unclipped=float(np.min(raw_bank)),
                      maximum_clip_change=float(np.max(np.abs(clipped-raw_bank))),
                      maximum_dtype_change=float(np.max(np.abs(bank-clipped))),
                      rms_dtype_change=rms(bank-clipped),
                      qualification='Original NumPy bank reference with served clamp/f32 transform; '
                                    'not a claim of native-D4 byte parity.')
        with np.load(BANK_INPUT,allow_pickle=False) as z:
            v = np.maximum(np.diag(z['pre13_Cpre']),1e-10)
            a = -z['pre13_alpha']*z['pre13_pdf']/(6*v)
            k_bank = np.diag(z['pre13_K21']).copy()
        full = {label:full_consumer(data[label]) for label in ('A','B')}
        g = {label:full[label]['G']-full[label]['true'] for label in full}
        b = {label:bank-full[label]['true'] for label in full}
        g0 = {label:full[label]['G0']-full[label]['true'] for label in full}
        primary = energies(g['A'],g['B'])
        bank_energy = energies(b['A'],b['B'])
        single = energies(g0['A'],g0['B'])
        primary_comparison = comparison(primary,bank_energy)
        single_comparison = comparison(single,bank_energy)

        pA,pB = full['A']['p'],full['B']['p']
        pbar = .5*(pA+pB)
        membership_limit = 6*np.sqrt(2*pbar*(1-pbar)/SAMPLES)
        membership_difference = np.abs(pA-pB)
        variance_A = full['A']['conditional']['variance']
        variance_B = full['B']['conditional']['variance']
        variance_relative = np.linalg.norm(variance_A-variance_B,axis=1)/np.linalg.norm(.5*(variance_A+variance_B),axis=1)
        stability_checks = dict(membership=bool(np.all(membership_difference <= membership_limit)),
                                conditional_variance=bool(np.all(variance_relative <= .02)))
        stability = dict(qualifications=stability_checks,
                         counts_A=np.sum(data['A']['block_counts'],axis=0).tolist(),
                         counts_B=np.sum(data['B']['block_counts'],axis=0).tolist(),
                         weights_A=pA.tolist(),weights_B=pB.tolist(),
                         absolute_membership_difference=membership_difference.tolist(),
                         membership_six_sigma_limit=membership_limit.tolist(),
                         conditional_variance_relative_L2=variance_relative.tolist(),
                         norm_membership_mismatch_A=data['A']['norm_membership_mismatch'].tolist(),
                         norm_membership_mismatch_B=data['B']['norm_membership_mismatch'].tolist())

        third_e = {label:a*(full[label]['total']['third']-k_bank) for label in full}
        third_o = {label:a*np.einsum('b,bi->i',full[label]['p'],full[label]['conditional']['third'])
                   for label in full}
        third_bank = energies(third_e['A'],third_e['B'])
        third_within = energies(third_o['A'],third_o['B'])
        kA,kB = full['A']['conditional']['third'],full['B']['conditional']['third']
        membership_term = a*np.einsum('b,bi->i',pA-pB,.5*(kA+kB))
        conditional_term = a*np.einsum('b,bi->i',.5*(pA+pB),kA-kB)
        decomposition_difference = third_o['A']-third_o['B']-membership_term-conditional_term
        decomposition_relative = float(np.max(np.abs(decomposition_difference))
                                       / max(1.,float(np.max(np.abs(third_o['A']-third_o['B'])))))
        if decomposition_relative > 1e-11:
            raise ValueError('exact replica-difference identity failed')
        third_diagnostic = dict(bank_third_error=third_bank,omitted_within_third=third_within,
                                comparison=comparison(third_within,third_bank),
                                membership_term_energy=float(np.mean(membership_term**2)),
                                conditional_estimation_term_energy=float(np.mean(conditional_term**2)),
                                twice_cross_term=float(2*np.mean(membership_term*conditional_term)),
                                decomposition_relative_max=decomposition_relative,
                                qualification='Fixed same-layer physical third response; diagnostic only. '
                                'No third gate or veto of a direct-mean pass.')

        jack = {label:jackknife(data[label],full[label],bank) for label in full}
        jgA = np.mean(jack['A']['g_leave']*g['B'][None,:],axis=1)
        jgB = np.mean(g['A'][None,:]*jack['B']['g_leave'],axis=1)
        jbA = np.mean(jack['A']['b_leave']*b['B'][None,:],axis=1)
        jbB = np.mean(b['A'][None,:]*jack['B']['b_leave'],axis=1)
        jack_scalar = dict(mixture_cross=jackknife_scalar_report(jgA,jgB),
                           bank_cross=jackknife_scalar_report(jbA,jbB))
        if np.all(jbA > 0) and np.all(jbB > 0):
            jack_scalar['cross_ratio'] = jackknife_scalar_report(jgA/jbA,jgB/jbB)
        else:
            jack_scalar['cross_ratio'] = dict(defined=False,qualification='Nonpositive leave-block bank cross denominator.',
                                              leave_A=[divide(x,y) for x,y in zip(jgA,jbA)],
                                              leave_B=[divide(x,y) for x,y in zip(jgB,jbB)])
        ordinary_leave = {label:np.mean(jack[label]['g_leave']**2,axis=1)
                               / np.mean(jack[label]['b_leave']**2,axis=1) for label in full}
        bias_rms = {label:rms(jack[label]['bias']) for label in full}
        bank_cross = bank_energy['cross']
        bias_scaled = {label:divide(bias_rms[label],np.sqrt(bank_cross)) if bank_cross > 0 else None
                       for label in full}
        cross_envelope = divide(rms(g['A'])*bias_rms['B']+rms(g['B'])*bias_rms['A']
                                +bias_rms['A']*bias_rms['B'],bank_cross)
        ordinary_envelope = {label:divide(2*rms(g[label])*bias_rms[label]+bias_rms[label]**2,
                                         bank_energy[f'mse_{label}']) for label in full}
        material_warning = any(value is None or value > .10 for value in bias_scaled.values())
        boundary_warning = (cross_envelope is None or primary_comparison['cross_ratio'] is None
                            or cross_envelope >= abs(primary_comparison['cross_ratio']-.50))
        for label in full:
            ratio = primary_comparison[f'ordinary_ratio_{label}']
            envelope = ordinary_envelope[label]
            boundary_warning = boundary_warning or envelope is None or ratio is None or envelope >= abs(ratio-.65)
        jack_diagnostic = dict(bias_rms=bias_rms,bias_rms_over_bank_signal=bias_scaled,
                               cross_ratio_sensitivity_envelope=cross_envelope,
                               ordinary_ratio_sensitivity_envelope=ordinary_envelope,
                               material_bias_warning=bool(material_warning),
                               boundary_sensitive_bias_warning=bool(boundary_warning),
                               scalar_stability=jack_scalar,
                               ordinary_leave_ratios={label:ordinary_leave[label].tolist() for label in full},
                               qualification='Fixed8x1M blocks from unchanged RNG streams. Approximate '
                               'jackknife plug-in bias/stability only; primary G is NOT corrected.')

        noise_bank = divide(bank_energy['replica_noise'],bank_cross)
        noise_mixture = primary_comparison['replica_noise_over_bank_cross']
        snr_checks = dict(positive_bank_cross=bank_cross > 0,
                          nonnegative_mixture_cross=primary['cross'] >= 0,
                          bank_noise=noise_bank is not None and noise_bank <= .10,
                          mixture_noise=noise_mixture is not None and noise_mixture <= .10)
        gate_checks = dict(cross=primary_comparison['cross_ratio'] is not None
                                and primary_comparison['cross_ratio'] <= .50,
                           ordinary_A=primary_comparison['ordinary_ratio_A'] is not None
                                and primary_comparison['ordinary_ratio_A'] <= .65,
                           ordinary_B=primary_comparison['ordinary_ratio_B'] is not None
                                and primary_comparison['ordinary_ratio_B'] <= .65)
        if not all(stability_checks.values()) or not all(snr_checks.values()):
            primary_verdict = 'INCONCLUSIVE_FIXED_ACQUISITION'
        else:
            primary_verdict = 'PASS_FIXED_INFORMATION_GATE' if all(gate_checks.values()) else 'FAIL_FIXED_INFORMATION_GATE'
        if primary_verdict.startswith('INCONCLUSIVE'):
            interpretation = primary_verdict
        elif material_warning or boundary_warning:
            interpretation = 'INCONCLUSIVE_ESTIMATED_PLUGIN_BIAS'
        else:
            interpretation = primary_verdict

        arrays = dict(bank_served=bank,bank_raw=raw_bank,third_response=a,bank_kappa3=k_bank,
                      membership_term=membership_term,conditional_estimation_term=conditional_term)
        for label in full:
            for key in ('G','G0','true','p'):
                arrays[f'{label}_{key}'] = full[label][key]
            for key,value in full[label]['conditional'].items():
                arrays[f'{label}_conditional_{key}'] = value
            for key,value in full[label]['total'].items():
                arrays[f'{label}_total_{key}'] = value
            arrays[f'{label}_g'] = g[label]
            arrays[f'{label}_b'] = b[label]
            arrays[f'{label}_g0'] = g0[label]
            arrays[f'{label}_weighted_third_bank_error'] = third_e[label]
            arrays[f'{label}_weighted_third_omission'] = third_o[label]
            for key,value in jack[label].items():
                arrays[f'{label}_jackknife_{key}'] = value
        for path,item in provenance.items():
            if sha256(path) != item['sha256']:
                raise ValueError(f'source/input changed during judgment: {path}')
        with out.open('xb') as handle:
            np.savez(handle,**arrays)
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != 'darwin':
            peak *= 1024
        result = dict(base_record,accuracy_consumer_evaluated=True,
                      served_bank_transform=served,bank_mean_error=bank_energy,
                      mixture_mean_error=primary,primary_comparison=primary_comparison,
                      primary_gate_checks=gate_checks,snr_qualification=snr_checks,
                      bank_replica_noise_over_bank_cross=noise_bank,
                      primary_v2_verdict=primary_verdict,interpretation=interpretation,
                      replica_membership_variance_audit=stability,third_diagnostic=third_diagnostic,
                      plugin_bias_and_block_stability=jack_diagnostic,
                      single_gaussian_oracle=dict(mean_error=single,comparison=single_comparison,
                          single_over_fourbin_cross_ratio=divide(single['cross'],primary['cross']),
                          conditioning_specific_gain_not_established=single['cross'] <= primary['cross'],
                          qualification='Same unconditional moments only; context, not a gate or additional arm.'),
                      output_sha256=sha256(out),utc_finished=utc_now(),wall_seconds=time.time()-started,
                      peak_rss_bytes=peak,
                      qualification='One-cell oracle Gaussianization information, NOT a final raw or adjusted '
                      'score. Cross products do not remove common finite-N plug-in or f32 forward bias. '
                      'No affordable conditional covariance source, fit, carrier, or submission is authorized.')
        with receipt_path.open('x',encoding='utf-8') as handle:
            json.dump(result,handle,indent=2,allow_nan=False)
            handle.write('\n')
        print(json.dumps(dict(interpretation=interpretation,primary_v2_verdict=primary_verdict,
                              primary_comparison=primary_comparison,primary_gate_checks=gate_checks,
                              snr_qualification=snr_checks,stability_qualification=stability_checks,
                              material_bias_warning=bool(material_warning),
                              boundary_sensitive_bias_warning=bool(boundary_warning),
                              third_ratio=third_diagnostic['comparison']['cross_ratio'],
                              single_gaussian_comparison=single_comparison,
                              output=str(out),receipt=str(receipt_path)),indent=2,allow_nan=False),flush=True)
    except BaseException as exc:
        failure = dict(utc=utc_now(),exception_type=type(exc).__name__,message=str(exc),
                       traceback=traceback.format_exc(),wall_seconds=time.time()-started)
        with failure_path.open('x',encoding='utf-8') as handle:
            json.dump(failure,handle,indent=2)
            handle.write('\n')
        raise


if __name__ == '__main__':
    main()
