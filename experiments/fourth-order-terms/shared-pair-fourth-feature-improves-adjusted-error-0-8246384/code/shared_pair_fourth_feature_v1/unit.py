"""Small independent tensor, gap and feature identities for actual FNP helpers."""
import hashlib
import json
from pathlib import Path
import resource
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from feature_fnp import epsilon_from_aux, pair_gamma, paired_fourth_diagonal, terminal_feature

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    rng = np.random.Generator(np.random.PCG64(2026092015))
    n = 7
    alpha = np.linspace(-2, 2, n)
    sigma = np.linspace(.3, 2, n)
    pdf = np.exp(-alpha*alpha/2)/np.sqrt(2*np.pi)
    selected = np.array([0, 2, 4, 6])
    t = np.array([.7, .3, .5, .125], dtype=np.float32)
    s = np.array([-.1, .3, -.2, -.125], dtype=np.float32)
    p = rng.normal(size=(n, 8)).astype(np.float32)
    p[:, 4:] = p[:, :4]+np.float32(1e-4)*rng.normal(size=(n, 4)).astype(np.float32)
    expected_epsilon = -alpha*pdf/(sigma*sigma)
    gap = t.astype(np.float64)-s.astype(np.float64)
    expected_gamma = expected_epsilon[selected]/np.where(gap != 0, gap, 1.)
    expected_gamma[gap == 0] = 0
    coeff = np.concatenate([expected_gamma, -expected_gamma])
    dense = np.einsum('ir,jr,kr,lr,r->ijkl', p.astype(np.float64), p.astype(np.float64),
                      p.astype(np.float64), p.astype(np.float64), coeff)
    expected = np.einsum('iiii->i', dense)
    errors = []
    with flops.BudgetContext(flop_budget=2**30, quiet=True) as ctx:
        aux = {k:fnp.asarray(a) for k,a in dict(alpha=alpha, sigma=sigma, pdf=pdf).items()}
        epsilon = epsilon_from_aux(aux)
        assert np.max(np.abs(np.asarray(epsilon)-expected_epsilon)) < 1e-14
        gamma = pair_gamma(epsilon, fnp.asarray(selected), fnp.asarray(t), fnp.asarray(s))
        assert np.array_equal(np.asarray(gamma), expected_gamma)
        pp = fnp.asarray(p)
        actual = paired_fourth_diagonal(pp, [(0, gamma)])
        repeat = paired_fourth_diagonal(pp, [(0, gamma)])
        assert np.asarray(actual).tobytes() == np.asarray(repeat).tobytes()
        error = float(np.max(np.abs(np.asarray(actual)-expected))/max(1., np.max(np.abs(expected))))
        errors.append(error)
        assert error <= 1e-10
        raw = np.sum(p.astype(np.float64)**4*coeff[None,:],axis=1)
        direct_error = float(np.max(np.abs(np.asarray(actual)-raw))/max(1.,np.max(np.abs(raw))))
        assert direct_error <= 1e-10
        shape, feature = terminal_feature(aux, actual)
        expected_shape = sigma*(alpha*alpha-1)*pdf/24
        assert np.max(np.abs(np.asarray(shape)-expected_shape)) < 1e-14
        expected_feature = expected_shape*expected/(sigma**4)
        feature_error = float(np.max(np.abs(np.asarray(feature)-expected_feature))/max(1.,np.max(np.abs(expected_feature))))
        assert feature_error <= 1e-10
        zero = paired_fourth_diagonal(pp, [(0, gamma*0)])
        assert np.count_nonzero(np.asarray(zero)) == 0
        assert np.asarray(pp).tobytes() == p.tobytes()
        assert np.asarray(aux['sigma']).tobytes() == sigma.tobytes()
        q = rng.normal(size=(n, 14)).astype(np.float32)
        coeffs = [rng.normal(size=3), rng.normal(size=4)]
        cohorts = [(0,fnp.asarray(coeffs[0])),(6,fnp.asarray(coeffs[1]))]
        fullcoef = np.concatenate([coeffs[0],-coeffs[0],coeffs[1],-coeffs[1]])
        got = paired_fourth_diagonal(fnp.asarray(q), cohorts)
        reference = np.sum(q.astype(np.float64)**4*fullcoef[None,:],axis=1)
        multi_error = float(np.max(np.abs(np.asarray(got)-reference))/max(1.,np.max(np.abs(reference))))
        assert multi_error <= 1e-10
        errors.append(multi_error)
    result = dict(passed=True, dense_tensor_scaled_error=errors[0], signed_power_scaled_error=direct_error,
                  multi_cohort_scaled_error=multi_error, feature_scaled_error=feature_error,
                  zero_gap_exact=True, zero_coefficient_exact=True, repeat_byte_exact=True,
                  input_preservation=True, full_estimator_executed=False, truth_accessed=False,
                  flops=ctx.summary_dict()['flops_used'], source_sha256=sha(__file__),
                  helper_sha256=sha(HERE/'feature_fnp.py'),
                  max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out = HERE/'unit_v1'
    out.mkdir(exist_ok=False)
    with(out/'result.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
