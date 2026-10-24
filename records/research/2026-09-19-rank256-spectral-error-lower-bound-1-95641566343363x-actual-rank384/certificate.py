"""Research-only spectral bound; constructs no compression basis."""
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
K = HERE.parent / 'full_tensor_sketch_information_v1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(p, value):
    with p.open('x') as stream:
        json.dump(value, stream, indent=2)


def cross_gram(p, a, s, b):
    h = p.T @ s
    # Independent scalar contraction: sum_ab a_a b_b <p_a,s_b>^3.
    # Block the temporary cube and weight product to avoid another full h.
    scalar_parts = []
    for start in range(0, len(a), 128):
        block = h[start:start + 128]
        scalar_parts.append(float(np.einsum('ij,i,j->', block ** 3,
                                             a[start:start + 128], b)))
    scalar = float(np.sum(np.asarray(scalar_parts), dtype=np.float64))
    h *= h
    gram = ((p * a[None, :]) @ h) @ (s * b[None, :]).T
    return gram, scalar


def dense(p, a):
    t = np.zeros((p.shape[0],) * 3)
    for v, coefficient in zip(p.T, a):
        t += coefficient * np.einsum('i,j,k->ijk', v, v, v)
    return t


def scaled_error(a, b):
    return float(np.linalg.norm(a - b) / max(1., float(np.linalg.norm(b))))


def unit():
    rng = np.random.Generator(np.random.PCG64(2026092011))
    cases = []
    for name, n, m, perturbation in [('ordinary', 7, 11, 1.),
                                     ('small_residual', 9, 12, .001),
                                     ('exact_null', 8, 13, 0.)]:
        p = rng.normal(size=(n, m))
        s = p + perturbation * rng.normal(size=(n, m))
        a = rng.normal(size=m)
        t, t0 = dense(p, a), dense(s, a)
        unfolded, old = t.reshape(n, n * n), t0.reshape(n, n * n)
        g, norm = cross_gram(p, a, p, a)
        g0, norm0 = cross_gram(s, a, s, a)
        cross, inner = cross_gram(p, a, s, a)
        residual = g + g0 - cross - cross.T
        residual = (residual + residual.T) * .5
        expected = (unfolded - old) @ (unfolded - old).T
        u, sv, vh = np.linalg.svd(unfolded, full_matrices=False)
        best_matrix = (u[:, :3] * sv[:3]) @ vh[:3]
        tail = float(np.linalg.eigvalsh((g + g.T) * .5)[:-3].sum())
        attained = float(np.sum((unfolded - best_matrix) ** 2))
        checks = dict(cross_gram_error=scaled_error(cross, unfolded @ old.T),
                      residual_gram_error=scaled_error(residual, expected),
                      scalar_inner_error=abs(inner - float(np.sum(unfolded * old))) /
                          max(1., abs(inner)),
                      scalar_residual_error=abs(norm + norm0 - 2 * inner - np.trace(expected)) /
                          max(1., float(np.trace(expected))),
                      spectral_tail_error=abs(tail - attained) / max(1., attained))
        assert all(value <= 1e-11 for value in checks.values()), (name, checks)
        if perturbation == 0:
            assert np.array_equal(residual, np.zeros((n, n)))
            assert norm + norm0 - 2 * inner == 0
        cases.append(dict(name=name, checks=checks, rank=3,
                          theoretical_lower_bound=tail, svd_attained_error=attained))
    result = dict(passed=True, cases=cases, truth_accessed=False,
                  source_sha256=sha(Path(__file__)),
                  max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    dump(HERE / 'unit.json', result)
    print(json.dumps(result), flush=True)


def run():
    started = time.monotonic()
    seal = json.loads((HERE / 'source_sealed.json').read_text())
    for name, digest in seal['files_sha256'].items():
        assert sha(HERE / name) == digest, name
    for name, digest in seal['external_sha256'].items():
        assert sha(Path(name)) == digest, name
    assert json.loads((HERE / 'unit.json').read_text())['passed']
    for path in [HERE / 'unit_watch.json', K / 'capture_watch.json']:
        watch = json.loads(path.read_text())
        assert watch['completed'] and watch['whole_group_retired']
        assert watch['max_swap_growth_bytes'] == 0
    capture = json.loads((K / 'capture/result.json').read_text())
    assert capture['all16_parent_bytes_exact'] and capture['parent_bill_exact']
    assert capture['cache_retired'] and not capture['truth_accessed']
    with np.load(K / 'capture/join.npz', allow_pickle=False) as packet:
        q, f, j, lam, w, parent = [packet[key].astype(np.float64)
                                  for key in ['Q', 'F', 'J', 'lambda', 'W', 'parent_pushed']]
    p = w.T @ np.concatenate([q @ f, j], axis=1)
    del q, f, j, w
    assert p.shape == parent.shape == (1024, 8128)
    assert all(np.isfinite(z).all() for z in [p, parent, lam])
    g, norm = cross_gram(p, lam, p, lam)
    print(json.dumps(dict(stage='reference_gram', seconds=time.monotonic() - started)), flush=True)
    g0, norm0 = cross_gram(parent, lam, parent, lam)
    cross, inner = cross_gram(p, lam, parent, lam)
    residual = g + g0 - cross - cross.T
    del g0, cross
    asymmetry = float(np.linalg.norm(g - g.T) / np.linalg.norm(g))
    residual_asymmetry = float(np.linalg.norm(residual - residual.T) / np.linalg.norm(residual))
    assert asymmetry <= 1e-10 and residual_asymmetry <= 1e-10
    g = (g + g.T) * .5
    residual = (residual + residual.T) * .5
    assert np.isfinite(g).all() and np.isfinite(residual).all()
    values, vectors = np.linalg.eigh(g)
    residual_values = np.linalg.eigvalsh(residual)
    eigen_error = float(np.linalg.norm(g @ vectors - vectors * values) / np.linalg.norm(g))
    del vectors
    assert eigen_error <= 1e-10
    assert values[-1] > 0 and values[0] >= -1e-10 * values[-1]
    assert residual_values[-1] > 0
    assert residual_values[0] >= -1e-8 * residual_values[-1]
    parent_error = float(np.trace(residual))
    scalar_parent_error = norm + norm0 - 2 * inner
    scalar_discrepancy = abs(parent_error - scalar_parent_error) / parent_error
    assert parent_error > 0 and scalar_discrepancy <= 1e-8
    rng = np.random.Generator(np.random.PCG64(2026092012))
    checks = []
    for v in rng.normal(size=(1024, 8)).T:
        action = (p * (lam * (p.T @ v))[None, :]) @ p.T
        action -= (parent * (lam * (parent.T @ v))[None, :]) @ parent.T
        exact = float(np.sum(action * action))
        quadratic = float(v @ residual @ v)
        relative = abs(exact - quadratic) / exact
        assert np.isfinite(relative) and relative <= 1e-8, relative
        checks.append(dict(direct_residual_squared_norm=exact,
                           gram_quadratic=quadratic, relative_discrepancy=relative))
    del action
    tail = float(values[:-256].sum())
    ratio = tail / parent_error
    assert np.isfinite(ratio)
    supported = ratio > .75 * 1.0001
    with (HERE / 'grams_and_spectrum.npz').open('xb') as stream:
        np.savez_compressed(stream, reference_gram=g, residual_gram=residual,
                            reference_eigenvalues=values, residual_eigenvalues=residual_values)
    old = json.loads((K / 'information/result.json').read_text())
    result = dict(numerical_checks_passed=True,
                  expected_query_threshold_infeasible_at_rank256=bool(supported),
                  spectral_tail_rank256=tail, exact_parent_frobenius_error_squared=parent_error,
                  independent_scalar_parent_error_squared=scalar_parent_error,
                  lower_bound_to_parent_error_ratio=ratio,
                  threshold=.75, decision_relative_margin=1e-4,
                  reference_tensor_squared_norm=float(np.trace(g)),
                  normalized_spectral_tail=tail / float(np.trace(g)),
                  normalized_parent_error=parent_error / float(np.trace(g)),
                  historical_512_probe_parent_mean=old['parent_probe_energy'] / 512,
                  gram_relative_asymmetry=asymmetry,
                  residual_gram_relative_asymmetry=residual_asymmetry,
                  minimum_reference_eigenvalue=float(values[0]),
                  maximum_reference_eigenvalue=float(values[-1]),
                  minimum_residual_eigenvalue=float(residual_values[0]),
                  maximum_residual_eigenvalue=float(residual_values[-1]),
                  eigendecomposition_relative_residual=eigen_error,
                  scalar_parent_energy_relative_discrepancy=scalar_discrepancy,
                  quadratic_form_checks=checks,
                  theoretical_statement='mode-one rank256 Frobenius lower bound; equivalently expected independent Gaussian bilinear-query error',
                  formal_interval_arithmetic_certificate=False,
                  bound_for_selected_finite_probe_set=False,
                  endpoint_accuracy_bound=False, production_cost_measured=False,
                  new_basis_or_candidate_constructed=False, truth_accessed=False,
                  new_estimator_calls=0, network_rank=35, entry_layer=8,
                  packet_sha256=capture['packet_sha256'], source_sha256=sha(Path(__file__)),
                  seconds=time.monotonic() - started,
                  max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    dump(HERE / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        if sys.argv[1:] == ['unit']:
            unit()
        elif sys.argv[1:] == ['run']:
            run()
        else:
            raise SystemExit('usage: certificate.py unit|run')
