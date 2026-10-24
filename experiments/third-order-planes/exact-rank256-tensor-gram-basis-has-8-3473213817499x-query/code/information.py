"""Fixed exact-Gram screen on the archived, target-free Lazy join state."""
import hashlib
import json
import resource
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
from algebra import full_gram, leading_basis, contract, plane, project

HERE = Path(__file__).resolve().parent
D = HERE.parent
K = D / 'full_tensor_sketch_information_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)


def main():
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
    out = HERE / 'information'
    out.mkdir(exist_ok=False)
    with np.load(K / 'capture/join.npz', allow_pickle=False) as packet:
        q0, f0, j, lam, w, parent = [packet[key].astype(np.float64)
                                   for key in ['Q', 'F', 'J', 'lambda', 'W', 'parent_pushed']]
    p = np.concatenate([q0 @ f0, j], axis=1)
    del q0, f0, j
    assert p.shape == (896, 8128) and parent.shape == (1024, 8128)
    assert w.shape == (896, 1024) and lam.shape == (8128,)
    assert all(np.isfinite(v).all() for v in [p, lam, w, parent])

    gram_started = time.monotonic()
    g = full_gram(p, lam)
    assert np.isfinite(g).all()
    q, eigenvalues, metrics = leading_basis(g, 256)
    basis_seconds = time.monotonic() - gram_started
    assert np.isfinite(q).all() and np.isfinite(eigenvalues).all()
    assert metrics['gram_relative_asymmetry'] <= 1e-10, metrics
    assert metrics['maximum_eigenvalue'] > 0
    assert metrics['minimum_eigenvalue'] >= -1e-10 * metrics['maximum_eigenvalue'], metrics
    assert metrics['eigendecomposition_relative_residual'] <= 1e-10, metrics
    assert metrics['basis_relative_orthogonality'] <= 1e-10, metrics
    diagnostic_rng = np.random.Generator(np.random.PCG64(2026092010))
    vectors = diagnostic_rng.normal(size=(p.shape[0], 8))
    checks = []
    for v in vectors.T:
        two_leg = (p * (lam * (p.T @ v))[None, :]) @ p.T
        reference_quadratic = float(np.sum(two_leg * two_leg))
        gram_quadratic = float(v @ g @ v)
        relative = abs(gram_quadratic - reference_quadratic) / max(
            abs(reference_quadratic), np.finfo(np.float64).tiny)
        checks.append(dict(gram_quadratic=gram_quadratic,
                           contraction_quadratic=reference_quadratic,
                           relative_discrepancy=relative))
        assert np.isfinite(relative) and relative <= 1e-8, checks
    del two_leg, vectors, g
    basis_file = out / 'fixed_basis.npz'
    with basis_file.open('xb') as stream:
        np.savez_compressed(stream, Q=q, eigenvalues=eigenvalues)
    dump(out / 'basis_sealed_before_verification.json', dict(
        basis_sha256=sha(basis_file), packet_sha256=capture['packet_sha256'],
        rank=256, method='full signed mode-one Gram leading eigenspace',
        metrics=metrics, quadratic_form_checks=checks,
        verification_probes_opened=False, truth_accessed=False,
        basis_seconds=basis_seconds))
    print(json.dumps(dict(stage='basis_sealed', seconds=basis_seconds, metrics=metrics,
                          maximum_quadratic_discrepancy=max(c['relative_discrepancy'] for c in checks))),
          flush=True)

    reference = w.T @ p
    trial = w.T @ project(p, q)
    rng = np.random.Generator(np.random.PCG64(2026092004))
    a, b = rng.normal(size=(2, 1024, 512))
    target_query = contract(reference, lam, a, b)
    parent_query = contract(parent, lam, a, b)
    trial_query = contract(trial, lam, a, b)
    parent_per_probe = np.sum((parent_query - target_query) ** 2, axis=0)
    trial_per_probe = np.sum((trial_query - target_query) ** 2, axis=0)
    parent_probe_energy = float(parent_per_probe.sum())
    trial_probe_energy = float(trial_per_probe.sum())
    target_probe_energy = float(np.sum(target_query ** 2))
    target_plane, parent_plane, trial_plane = [plane(z, lam) for z in [reference, parent, trial]]
    parent_plane_error = (parent_plane - target_plane) ** 2
    trial_plane_error = (trial_plane - target_plane) ** 2
    parent_plane_energy = float(parent_plane_error.sum())
    trial_plane_energy = float(trial_plane_error.sum())
    target_plane_energy = float(np.sum(target_plane ** 2))
    old = json.loads((K / 'information/result.json').read_text())
    assert abs(parent_probe_energy / old['parent_probe_energy'] - 1) <= 1e-10
    assert abs(parent_plane_energy / old['parent_plane_squared_error'] - 1) <= 1e-10
    assert all(np.isfinite(z).all() for z in [parent_per_probe, trial_per_probe,
                                              parent_plane_error, trial_plane_error])
    probe_ratio = trial_probe_energy / parent_probe_energy
    plane_ratio = trial_plane_energy / parent_plane_energy
    paired = trial_per_probe - parent_per_probe
    with (out / 'residual_statistics.npz').open('xb') as stream:
        np.savez_compressed(stream, parent_per_probe=parent_per_probe,
                            candidate_per_probe=trial_per_probe,
                            parent_plane_squared_error=parent_plane_error,
                            candidate_plane_squared_error=trial_plane_error)
    n, m = p.shape
    conventional_subtotal = m * m * (2 * n - 1) + m * m + n * m + \
        n * m * (2 * m - 1) + n * n * (2 * m - 1)
    result = dict(
        information_gate_passed=bool(probe_ratio <= .75 and plane_ratio <= .75),
        numerical_checks_passed=True, probe_residual_energy_ratio=probe_ratio,
        plane_residual_energy_ratio=plane_ratio,
        parent_probe_energy=parent_probe_energy, candidate_probe_energy=trial_probe_energy,
        parent_relative_probe_residual_squared=parent_probe_energy / target_probe_energy,
        candidate_relative_probe_residual_squared=trial_probe_energy / target_probe_energy,
        parent_plane_squared_error=parent_plane_energy, candidate_plane_squared_error=trial_plane_energy,
        parent_relative_plane_residual_squared=parent_plane_energy / target_plane_energy,
        candidate_relative_plane_residual_squared=trial_plane_energy / target_plane_energy,
        marginal_diagonal_residual_energy_ratio=float(np.trace(trial_plane_error) /
                                                      np.trace(parent_plane_error)),
        probe_wins=int(np.count_nonzero(trial_per_probe < parent_per_probe)),
        paired_probe_energy_difference_mean=float(paired.mean()),
        paired_probe_energy_difference_se=float(paired.std(ddof=1) / np.sqrt(512)),
        verification_probe_count=512, verification_seed=2026092004,
        verification_probes_previously_used=True, verification_probes_used_for_basis=False,
        rank=256, parent_rank=384, layer=8, network_rank=35,
        metrics=metrics, maximum_quadratic_form_discrepancy=max(c['relative_discrepancy'] for c in checks),
        conventional_dense_gram_arithmetic_subtotal=conventional_subtotal,
        conventional_subtotal_is_metered_flopscope_bill=False,
        eigendecomposition_and_other_operations_priced=False,
        truth_accessed=False, new_estimator_calls=0,
        actual_accuracy_score_measured=False, production_cost_measured=False,
        precision='optimistic FP64 trial versus actual rounded parent factors',
        basis_seconds=basis_seconds, seconds=time.monotonic() - started,
        max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        threads=threadpool_info(), packet_sha256=capture['packet_sha256'],
        basis_sha256=sha(basis_file), source_sha256=sha(Path(__file__)),
        algebra_sha256=sha(HERE / 'algebra.py'))
    dump(out / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
