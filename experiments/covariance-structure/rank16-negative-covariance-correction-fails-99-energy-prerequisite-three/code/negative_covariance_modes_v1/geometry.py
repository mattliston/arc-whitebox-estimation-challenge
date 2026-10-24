"""Offline reference geometry, not a billed inference implementation."""


def negative_modes(np, covariance, limit=16):
    symmetric = (covariance + covariance.T) * .5
    values, vectors = np.linalg.eigh(symmetric)
    negative = np.flatnonzero(values < 0)
    selected = negative[:limit]
    Q = vectors[:, selected]
    d = -values[selected]
    negative_energy = float(np.dot(values[negative], values[negative]))
    kept_energy = float(np.dot(d, d))
    fraction = kept_energy / negative_energy if negative_energy else 1.
    residual = float(np.linalg.norm(symmetric @ vectors - vectors * values) / max(np.linalg.norm(symmetric), 1e-30))
    orthogonality = float(np.linalg.norm(vectors.T @ vectors - np.eye(len(values))) / len(values)**.5)
    assert max(residual, orthogonality) <= 1e-10
    info = dict(min_eigenvalue=float(values[0]), max_eigenvalue=float(values[-1]),
                negative_count=len(negative), kept_count=len(selected),
                negative_energy=negative_energy, kept_energy=kept_energy,
                retained_fraction=fraction, correction_trace=float(d.sum()),
                correction_relative=float(kept_energy**.5 / max(np.linalg.norm(covariance), 1e-30)),
                eigen_residual=residual, orthogonality=orthogonality,
                spectral_gate_passed=fraction >= .99)
    return values, Q, d, info


def response(np, Q, d, weights, factor):
    return factor * np.sum(d[:, None] * (Q.T @ weights)**2, axis=0)
