"""Independent dense-tensor certificate, including signed cross-pair terms."""
import hashlib
import json
import resource
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from algebra import full_gram, leading_basis, contract, plane, project

HERE = Path(__file__).resolve().parent


def dense(p, lam):
    answer = np.zeros((p.shape[0],) * 3)
    for x, coefficient in zip(p.T, lam):
        answer += coefficient * np.einsum('i,j,k->ijk', x, x, x)
    return answer


def error(actual, expected):
    return float(np.linalg.norm(actual - expected) /
                 max(1., float(np.linalg.norm(expected))))


def main():
    rng = np.random.Generator(np.random.PCG64(2026092009))
    cases = []
    for name, n, c, epsilon in [('ordinary', 7, 9, 1.),
                                 ('near_cancelling', 8, 6, 1e-6)]:
        base = rng.normal(size=(n, c))
        delta = epsilon * rng.normal(size=(n, c))
        p = np.concatenate([base + delta, base - delta], axis=1)
        coefficient = rng.normal(size=c)
        lam = np.concatenate([coefficient, -coefficient])
        t = dense(p, lam)
        unfolded = t.reshape(n, n * n)
        expected_gram = unfolded @ unfolded.T
        actual_gram = full_gram(p, lam)
        q, eigenvalues, metrics = leading_basis(actual_gram, 3)
        h = q @ q.T
        projected_tensor = np.einsum('ai,bj,ck,ijk->abc', h, h, h, t)
        pp = project(p, q)
        a, b = rng.normal(size=(2, n, 5))
        expected_query = np.einsum('ijk,js,ks->is', projected_tensor, a, b)
        expected_plane = np.einsum('iij->ij', projected_tensor)
        retained = float(np.linalg.norm(q.T @ unfolded) ** 2)
        checks = dict(gram_scaled_error=error(actual_gram, expected_gram),
                      projection_scaled_error=error(dense(pp, lam), projected_tensor),
                      query_scaled_error=error(contract(pp, lam, a, b), expected_query),
                      plane_scaled_error=error(plane(pp, lam), expected_plane),
                      one_mode_energy_scaled_error=abs(retained - eigenvalues[-3:].sum()) /
                          max(1., abs(retained)))
        assert all(value <= 1e-11 for value in checks.values()), (name, checks)
        cases.append(dict(name=name, checks=checks, metrics=metrics))

    x, y = np.eye(3)[:2]
    p = np.stack([x, x, y, y], axis=1)
    lam = np.array([1., -1., -1., 1.])
    t = dense(p, lam)
    full = full_gram(p, lam)
    pairs = [dense(p[:, [k, k + 2]], lam[[k, k + 2]]) for k in range(2)]
    pair_sum = sum(z.reshape(3, 9) @ z.reshape(3, 9).T for z in pairs)
    assert np.array_equal(t, np.zeros((3, 3, 3)))
    assert np.array_equal(full, np.zeros((3, 3)))
    assert np.array_equal(pair_sum, np.diag([2., 2., 0.]))
    result = dict(passed=True, cases=cases, exact_cross_pair_cancellation=True,
                  sum_complete_pair_grams=pair_sum.tolist(),
                  full_cancelled_gram=full.tolist(), truth_accessed=False,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  algebra_sha256=hashlib.sha256((HERE / 'algebra.py').read_bytes()).hexdigest(),
                  max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with (HERE / 'unit.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
