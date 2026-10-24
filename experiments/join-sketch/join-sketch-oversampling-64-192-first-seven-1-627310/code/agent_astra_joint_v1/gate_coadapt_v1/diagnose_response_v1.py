"""Read completed fresh4 only. Describe response curvature; never fit a new arm."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

import numpy as np

D = Path(__file__).resolve().parent


def summary(values):
    return dict(mean=statistics.mean(values), se=statistics.stdev(values)/math.sqrt(len(values)),
                minimum=min(values), maximum=max(values), negative=sum(v < 0 for v in values))


def main():
    result = json.loads((D/'RESULT_FRESH4_v1.json').read_text())
    manifest = json.loads((D/'MANIFEST_v2.json').read_text())
    assert result['status'] == 'PAIRED_FRESH4_COMPLETE_NOT_CURRENT32_QUALIFICATION'
    assert result['candidate_sha256'] == manifest['candidate_sha256']
    assert result['reference_sha256'] == manifest['bank_sha256']
    assert result['seeds'] == manifest['seeds']
    rows = []
    for seed in manifest['seeds']:
        metadata = [json.loads(p.read_text()) for p in sorted(
            (D/'runs'/f'screen_seed{seed}'/'predictions').glob('*.json'))]
        assert len(metadata) == 6
        predictions = []
        for index, meta in enumerate(metadata):
            assert meta['setting_index'] == index and meta['setting'] == manifest['settings'][index]
            path = Path(meta['prediction_path'])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['prediction_sha256']
            predictions.append(np.load(path, allow_pickle=False)[15])
        truth_path = next(x['path'] for x in manifest['truths'] if x['seed'] == seed)
        with np.load(truth_path, allow_pickle=False) as truth:
            a, b = truth['mc_post_A'][15], truth['mc_post_B'][15]

        def u(prediction):
            return float(np.mean((prediction-a)*(prediction-b)))

        p0, ps_minus, ps_plus, pk_minus, pk_plus, ppair = predictions
        U = [u(p) for p in predictions]
        default = U[0]
        assert all(x > 0 for x in U)
        raw = [100*(x/default-1) for x in U]
        # Coordinates x,y equal absolute coefficient changes /0.1. These
        # six coefficients interpolate the six native outputs' raw scores;
        # they do not validate unsampled points or a quadratic whole model.
        slope_s = (raw[2]-raw[1])/2
        slope_k = (raw[4]-raw[3])/2
        curve_s = (raw[2]+raw[1])/2
        curve_k = (raw[4]+raw[3])/2
        cross = raw[5]-raw[1]-raw[3]
        linear_pair = ps_minus + pk_minus - p0
        mixed = ppair - linear_pair
        direction_norm = float(np.linalg.norm(ps_minus-p0)+np.linalg.norm(pk_minus-p0))
        hessian = np.array([[2*curve_s,cross],[cross,2*curve_k]])
        rows.append(dict(seed=seed,
                         raw_score_surface_pct_per_unit_step=dict(gsk_slope=slope_s,gk4_slope=slope_k,
                             gsk_quadratic=curve_s,gk4_quadratic=curve_k,cross=cross,
                             hessian_eigenvalues=np.linalg.eigvalsh(hessian).tolist()),
                         gsk_midpoint_prediction_norm=float(np.linalg.norm((ps_plus+ps_minus)/2-p0)),
                         gk4_midpoint_prediction_norm=float(np.linalg.norm((pk_plus+pk_minus)/2-p0)),
                         mixed_prediction_norm=float(np.linalg.norm(mixed)),
                         mixed_to_single_direction_norm=float(np.linalg.norm(mixed))/direction_norm if direction_norm else None,
                         actual_pair_raw_pct=raw[5],
                         linear_prediction_pair_proxy_raw_pct=100*(u(linear_pair)/default-1),
                         nonlinear_response_raw_effect_pct=100*(U[5]-u(linear_pair))/default,
                         actual_bills=[meta['flops'] for meta in metadata],
                         same_bill_all_six=len({meta['flops'] for meta in metadata})==1))
    contrasts = []
    for setting in result['settings'][1:]:
        values = [r['adjusted_vs_same_screen_default_pct'] for r in setting['rows']]
        leave_one_out = [statistics.mean(values[:i]+values[i+1:]) for i in range(4)]
        contrasts.append(dict(index=setting['index'],label=setting['label'],
                              actual_incremental_adjusted_pct=summary(values),
                              leave_one_seed_out_means_pct=leave_one_out,
                              maximum_leave_one_out_mean_pct=max(leave_one_out)))
    report = dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),
                  status='DESCRIPTIVE_FRESH4_RESPONSE_ONLY_NO_NEW_FIT_OR_CANDIDATE',
                  reference_submission_id=337393, candidate_sha256=manifest['candidate_sha256'],
                  coordinate_definition='x=(GSK-.9)/.1; y=(GK4-1.04)/.1',
                  slope_summary={key:summary([r['raw_score_surface_pct_per_unit_step'][key] for r in rows])
                                 for key in ['gsk_slope','gk4_slope','gsk_quadratic','gk4_quadratic','cross']},
                  actual_endpoint_robustness=contrasts, rows=rows,
                  limitations=['Four independent fresh networks and six prespecified settings; screening uncertainty and multiplicity remain.',
                               'No held25 or store truth used. No off-grid candidate chosen or optimum fitted.',
                               'Linear prediction proxy is explicitly unqualified; the native pair row measures its discrepancy.',
                               'The quadratic score surface is an interpolation diagnostic, not a full-model law or additivity claim.',
                               'Actual paired-bank score means remain RESULT_FRESH4_v1.json; these curvature numbers are raw incremental diagnostics.'])
    (D/'RESPONSE_DIAGNOSTIC_v1.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','limitations')},indent=2))


if __name__ == '__main__':
    main()
