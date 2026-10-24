"""Paired two-half U score and separately labeled Monte Carlo uncertainty.

This is an offline reporting helper. It neither fits nor selects a candidate.
The first-order delta estimate uses the saved final-layer batch means, so it
retains covariance between output channels and between the paired estimators.
It is distinct from the across-network standard error of a reported gain.
"""
import numpy as np


def score_and_gradients(prediction, reference, truth_a, truth_b, bill_ratio):
    p, p0, a, b = [np.asarray(x, dtype=np.float64).reshape(-1)
                    for x in (prediction, reference, truth_a, truth_b)]
    assert p.shape == p0.shape == a.shape == b.shape
    assert all(np.isfinite(x).all() for x in (p, p0, a, b))
    assert np.isfinite(bill_ratio) and bill_ratio > 0
    u1 = float(np.mean((p - a) * (p - b)))
    u0 = float(np.mean((p0 - a) * (p0 - b)))
    assert u0 > 0 and u1 > 0, 'Nonpositive two-half U: ratio is not reportable.'
    ratio = float(bill_ratio) * u1 / u0
    scale = float(bill_ratio) / (p.size * u0)
    grad_a = scale * (-(p - b) + (u1 / u0) * (p0 - b))
    grad_b = scale * (-(p - a) + (u1 / u0) * (p0 - a))
    return u1, u0, ratio, grad_a, grad_b


def paired_report(prediction, reference, truth_a, truth_b, batch_means,
                  candidate_bill, reference_bill):
    batch_means = np.asarray(batch_means, dtype=np.float64)
    assert batch_means.ndim == 2 and batch_means.shape[0] % 2 == 0
    h = batch_means.shape[0] // 2
    assert h > 1 and np.isfinite(batch_means).all()
    a, b = [np.asarray(x, dtype=np.float64).reshape(-1)
            for x in (truth_a, truth_b)]
    assert batch_means.shape[1] == a.size == b.size
    np.testing.assert_allclose(batch_means[:h].mean(axis=0), a, rtol=0, atol=2e-13)
    np.testing.assert_allclose(batch_means[h:].mean(axis=0), b, rtol=0, atol=2e-13)
    u1, u0, ratio, ga, gb = score_and_gradients(
        prediction, reference, a, b, candidate_bill / reference_bill)
    contributions = [float(np.var(batch_means[:h] @ ga, ddof=1) / h),
                     float(np.var(batch_means[h:] @ gb, ddof=1) / h)]
    return dict(candidate_u=u1, reference_u=u0,
                raw_pct=100 * (u1 / u0 - 1),
                bill_pct=100 * (candidate_bill / reference_bill - 1),
                adjusted_pct=100 * (ratio - 1),
                paired_mc_se_pct=100 * float(np.sqrt(sum(contributions))),
                mc_se_method='First-order delta estimate from independent half batch means; channel covariance retained.',
                mc_variance_by_half=contributions,
                half_batches=h,
                across_network_se_is_separate=True,
                no_fit_or_selection=True)


def self_check():
    p = np.array([.82, 1.13, -.04, .21])
    p0 = np.array([.81, 1.15, -.01, .24])
    a = np.array([.8, 1.1, 0., .2])
    b = np.array([.801, 1.099, .001, .199])
    bill_ratio = .997
    _, _, ratio, ga, gb = score_and_gradients(p, p0, a, b, bill_ratio)
    max_error = 0.
    for half, gradient in ((0, ga), (1, gb)):
        for index in range(len(a)):
            plus = [a.copy(), b.copy()]
            minus = [a.copy(), b.copy()]
            eps = 1e-7
            plus[half][index] += eps
            minus[half][index] -= eps
            rp = score_and_gradients(p, p0, *plus, bill_ratio)[2]
            rm = score_and_gradients(p, p0, *minus, bill_ratio)[2]
            finite = (rp - rm) / (2 * eps)
            max_error = max(max_error, abs(finite - gradient[index]))
            np.testing.assert_allclose(finite, gradient[index], rtol=2e-7, atol=2e-8)
    _, _, identity, ia, ib = score_and_gradients(p0, p0, a, b, 1.)
    assert identity == 1. and np.array_equal(ia, np.zeros_like(ia)) and np.array_equal(ib, np.zeros_like(ib))
    return dict(successful=True, max_gradient_absolute_error=max_error,
                identical_estimator_has_exact_zero_paired_mc_uncertainty=True,
                diagnostic_ratio=ratio)


if __name__ == '__main__':
    import json
    print(json.dumps(self_check()))
