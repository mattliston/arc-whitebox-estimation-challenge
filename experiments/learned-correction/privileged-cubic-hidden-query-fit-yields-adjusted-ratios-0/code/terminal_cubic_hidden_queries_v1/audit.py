"""Reconstruct the sealed diagnostic; do not select or fit a new candidate."""
import hashlib
import math
import resource

from common import *


def main():
    import numpy as np
    from threadpoolctl import threadpool_info
    from capture import observed

    assert all(x['num_threads'] == 1 for x in threadpool_info())
    for seal in ('sources_sealed.json', 'captures_sealed.json',
                 'features_sealed.json', 'audit_sources_sealed.json',
                 'source_review_pins.json'):
        verify(seal)
    review = read(HERE / 'source_review_pins.json')['review']
    assert sha(review['path']) == review['sha256']
    _, proof = observed()
    captures = read(HERE / 'capture/result.json')
    features = read(HERE / 'features_result.json')
    result = read(HERE / 'result.json')
    original = read(D / 'full_lower_multilayer_mean_observer_v1/parent_triple/result.json')['records']
    controls = read(HERE / 'controls.json')
    assert captures['source_proof'] == proof == read(HERE / 'unit.json')['source_proof']
    assert read(HERE / 'unit.json')['passed'] and features['passed']
    assert captures['truth_accessed'] is False and captures['extra_parent_operations'] == 0
    assert [r['rank'] for r in captures['records']] == [35, 100, 35]
    assert captures['records'][0]['array_digests'] == captures['records'][2]['array_digests']
    assert features['rows'][0]['array_digests'] == features['rows'][2]['array_digests']
    assert features['rows'][0]['helper_flops'] == features['rows'][2]['helper_flops']
    for rec, base in zip(captures['records'], original):
        assert rec['flops'] == base['flops'] and rec['rank'] == base['rank']
        assert rec['all16_and_bill_exact'] and rec['cache_retired']
    with np.load(result['oracle_coefficients_not_deployable']['path'], allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}

    def mse(a, b):
        return math.fsum(float(x) ** 2 for x in (a.astype(np.float64) - b)) / a.size

    metric_errors = []

    def scalar_equal(actual, expected):
        relative = abs(actual - expected) / max(abs(expected), 1e-30)
        assert relative < 5e-12, (actual, expected, relative)
        metric_errors.append(relative)

    rows = []
    for row in result['rows']:
        rank = row['rank']
        caprow = next(x for x in captures['records'] if x['rank'] == rank)
        frow = next(x for x in features['rows'] if x['rank'] == rank)
        with np.load(caprow['capture']['path'], allow_pickle=False) as z:
            for k, digest in caprow['array_digests'].items():
                arr = z[k]
                assert list(arr.shape) == digest['shape'] and str(arr.dtype) == digest['dtype']
                assert hashlib.sha256(arr.tobytes()).hexdigest() == digest['sha256']
            pred_all = z['prediction']
            expected_affine = z['sigma'][:, None] * np.column_stack([np.ones(1024), z['H'].astype(np.float64)])
        assert np.array_equal(pred_all, np.load(controls[str(rank)]['output'], allow_pickle=False))
        with np.load(frow['file']['path'], allow_pickle=False) as z:
            for k, digest in frow['array_digests'].items():
                assert hashlib.sha256(z[k].tobytes()).hexdigest() == digest
            B, Z = z['affine'], z['feature']
            assert np.array_equal(z['prediction'], pred_all)
        assert np.array_equal(B, expected_affine)
        assert frow['helper_flops'] <= 20_000_000_000
        assert frow['reference_relative'] <= 2e-10 and frow['formal_diagonal_relative'] <= 1e-3
        assert frow['input_preserved']
        pred = pred_all[-1].astype(np.float64)
        with np.load(truth(rank), allow_pickle=False) as z:
            target = z['mc_post'][-1].astype(np.float64)
        error = target - pred
        parent_raw = mse(pred, target)
        parent_flops = controls[str(rank)]['flops']
        parent_adjusted = parent_raw * max(.1, parent_flops / 2**41)
        assert parent_flops == row['parent']['flops']
        scalar_equal(parent_raw, row['parent']['raw'])
        scalar_equal(parent_adjusted, row['parent']['adjusted'])
        arms = {}
        for name, design, extra in (
            ('affine', B, 1_000_000),
            ('extended', np.column_stack([B, Z]), frow['helper_flops'] + 1_000_000),
        ):
            # Full numerical rank is checked independently before a QR solve.
            # The primary solver used truncated-SVD projection and coefficients.
            scale = np.maximum(np.sqrt(np.mean(design * design, axis=0)), 1e-30)
            normalized = design / scale
            spectrum = np.linalg.svd(normalized, compute_uv=False)
            kept = int(np.count_nonzero(spectrum > 1e-10 * spectrum[0]))
            assert kept == design.shape[1] == row['arms'][name]['checks']['rank']
            Q, R = np.linalg.qr(normalized, mode='reduced')
            qr_coef = np.linalg.solve(R, Q.T @ error) / scale
            saved_coef = payload[f'r{rank}_{name}_coefficients']
            coefficient_relative = float(np.linalg.norm(qr_coef - saved_coef) / max(np.linalg.norm(saved_coef), 1e-30))
            assert coefficient_relative < 1e-8, coefficient_relative
            saved_ideal = payload[f'r{rank}_{name}_ideal']
            saved_served = payload[f'r{rank}_{name}_served']
            assert np.array_equal(pred + design @ saved_coef, saved_ideal)
            qr_ideal = pred + design @ qr_coef
            qr_correction_relative = float(np.linalg.norm(design @ qr_coef - design @ saved_coef) / max(np.linalg.norm(design @ saved_coef), 1e-30))
            assert qr_correction_relative < 1e-8
            assert np.array_equal(np.maximum(qr_ideal, 0).astype(np.float32), saved_served)
            assert np.array_equal(np.maximum(saved_ideal, 0).astype(np.float32), saved_served)
            normal_relative = float(np.linalg.norm(Q.T @ (target - saved_ideal)) / max(np.linalg.norm(error), 1e-30))
            assert normal_relative < 1e-8
            charged = parent_flops + extra
            ideal_raw, served_raw = mse(saved_ideal, target), mse(saved_served, target)
            metrics = dict(ideal_raw=ideal_raw, served_raw=served_raw,
                           ideal_adjusted=ideal_raw * max(.1, charged / 2**41),
                           served_adjusted=served_raw * max(.1, charged / 2**41))
            assert charged == row['arms'][name]['charged_flops']
            for k, value in metrics.items():
                scalar_equal(value, row['arms'][name][k])
            arms[name] = dict(**metrics, rank=kept, coefficient_relative=coefficient_relative,
                              qr_correction_relative=qr_correction_relative,
                              normal_relative=normal_relative, rounded_payload_exact=True)
        ext, aff = arms['extended'], arms['affine']
        ratios = dict(ideal_adjusted_parent=ext['ideal_adjusted'] / parent_adjusted,
                      served_adjusted_parent=ext['served_adjusted'] / parent_adjusted,
                      ideal_raw_affine=ext['ideal_raw'] / aff['ideal_raw'],
                      served_raw_affine=ext['served_raw'] / aff['served_raw'])
        for k, value in ratios.items():
            scalar_equal(value, row['ratios'][k])
        passed = all(v <= .8 for v in ratios.values())
        assert passed == row['passed']
        rows.append(dict(rank=rank, arms=arms, ratios=ratios, information_gate_passed=passed,
                         all16_equal_independent_cold_control=True))
    assert all(r['information_gate_passed'] for r in rows) == result['passed']
    audit = dict(passed=True, rows=rows, independent_solver='full-rank reduced QR and triangular solve',
                 scalar_mse='math.fsum', max_metric_relative=max(metric_errors),
                 source_inverse_verified=True, all_artifact_digests_verified=True,
                 new_candidate=False, new_feature_or_hyperparameter_selection=False,
                 peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(HERE / 'audit_result.json', audit)
    print(json.dumps(dict(passed=True, max_metric_relative=audit['max_metric_relative'],
                          max_coefficient_relative=max(a['coefficient_relative'] for r in rows for a in r['arms'].values()),
                          all_rounded_payloads_exact=True)), flush=True)


if __name__ == '__main__':
    main()
