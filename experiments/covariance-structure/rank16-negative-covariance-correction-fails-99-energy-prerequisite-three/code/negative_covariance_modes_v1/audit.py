"""Independent completed-spectrum audit; no reference answers are decoded."""
import resource

from common import *


def main():
    import numpy as np
    from threadpoolctl import threadpool_info
    assert all(x['num_threads'] == 1 for x in threadpool_info())
    verify();verify('geometry_sealed.json');verify('audit_sources_sealed.json')
    spectrum = read(HERE/'spectrum_result.json')
    completed = read(HERE/'completed.json')
    assert not spectrum['passed'] and completed['stage'] == 'rank16_spectrum'
    assert completed['targets_opened'] is False and not (HERE/'score_result.json').exists()
    rows = []
    for caprow, row in zip(captures(), spectrum['rows']):
        assert caprow['rank'] == row['rank']
        W = np.load(caprow['weights']['path'], allow_pickle=False)
        layers = []
        with np.load(caprow['capture']['path'], allow_pickle=False) as cap, np.load(row['file']['path'], allow_pickle=False) as modes:
            for original in row['layers']:
                l = original['layer'];C = cap[f'C_post_{l:02d}'].astype(np.float64);S = (C+C.T)*.5
                saved = modes[f'values_{l}'];Q = modes[f'Q_{l}'];d = modes[f'd_{l}']
                # Independent eigenvalue-only solver, using the opposite triangle.
                values = np.linalg.eigvalsh(S, UPLO='U')
                eigen_relative = float(np.linalg.norm(values-saved)/max(np.linalg.norm(saved), 1e-30))
                assert eigen_relative <= 1e-10
                neg = np.minimum(values, 0.);full_energy = float(neg@neg)
                kept_energy = float(neg[:16]@neg[:16]);fraction = kept_energy/full_energy if full_energy else 1.
                fraction_delta = abs(fraction-original['retained_fraction'])
                assert fraction_delta <= 1e-8
                assert (fraction >= .99) == original['spectral_gate_passed']
                eigen_residual = float(np.linalg.norm(S@Q+Q*d)/max(np.linalg.norm(S), 1e-30))
                assert eigen_residual <= 1e-10
                assert np.linalg.norm(Q.T@Q-np.eye(len(d))) <= 1e-10
                assert np.all(d >= 0) and len(d) <= 16
                # Post-result geometry only: any full negative-part Gaussian
                # response is bounded by ||S_-||_2 times each weight norm.
                # This does not include the learned head or recurrent feedback.
                weight = W[l+1].astype(np.float64)
                factor = cap[f'pdf_{l+1:02d}'].astype(np.float64)/(2*cap[f'sigma_{l+1:02d}'].astype(np.float64))
                assert np.all(factor >= 0)
                bound = max(-float(values[0]), 0.) * factor * np.sum(weight*weight, axis=0)
                selected = factor * np.sum(d[:,None]*(Q.T@weight)**2, axis=0)
                assert np.all(selected >= 0) and np.all(selected <= bound+1e-12)
                dense = (Q*d)@Q.T
                direct = factor*np.diag(weight.T@dense@weight)
                response_relative = float(np.linalg.norm(selected-direct)/max(np.linalg.norm(selected), 1e-30))
                assert response_relative <= 1e-10
                layers.append(dict(layer=l, eigenvalue_relative=eigen_relative,
                              energy_fraction_delta=fraction_delta, eigenvector_relative=eigen_residual,
                              selected_response_relative=response_relative,
                              full_negative_gaussian_response_rms_bound=float(np.sqrt(np.mean(bound*bound))),
                              full_negative_gaussian_response_max_bound=float(bound.max()),
                              selected_gaussian_response_rms=float(np.sqrt(np.mean(selected*selected))),
                              full_negative_relative_norm=full_energy**.5/max(float(np.linalg.norm(C)), 1e-30),
                              original_negative_count=original['negative_count'],
                              spectral_gate_passed=original['spectral_gate_passed']))
        rows.append(dict(rank=row['rank'], layers=layers))
    result = dict(passed=True, rows=rows, target_arrays_opened=False, candidate_created=False,
                  primary_result_unchanged=True,
                  bound_scope='Post-result source-free bound on the immediate frozen Gaussian variance response only; excludes learned head and recurrence. No new acceptance gate.',
                  eigenvalue_count_scope='Counts refer to original eigh signs; tiny eigenvalues near zero can be sign-sensitive. Energy and declared gates independently verified.',
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(HERE/'audit_result.json', result)
    print(json.dumps(dict(passed=True, max_eigenvalue_relative=max(x['eigenvalue_relative'] for r in rows for x in r['layers']),
                         max_response_bound=max(x['full_negative_gaussian_response_rms_bound'] for r in rows for x in r['layers']),
                         no_targets=True)), flush=True)


if __name__ == '__main__':
    main()
