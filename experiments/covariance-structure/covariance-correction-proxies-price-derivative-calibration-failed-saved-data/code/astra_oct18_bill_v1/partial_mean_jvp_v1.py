"""Partial current-head derivative for the Price covariance hypothesis.

Research helper only. Supply numpy or public flopscope.numpy as xp. Messages,
previous state, energy features, HC bias and nonlocal covariance features are
held fixed. dfeatures_dalpha specifies exactly which local feature derivatives
are retained. No estimator, model data, SDK internals or file IO is imported.
"""


def _silu_prime(xp, z):
    sigmoid = 1.0 / (1.0 + xp.exp(-z))
    return sigmoid * (1.0 + z * (1.0 - sigmoid))


def partial_mean_jvp(xp, layer, x, dfeatures_dalpha, messages, z0, z1, P,
                     mean=None, delta=None, base_mean_dmu=None):
    """Return partial d(delta_i)/d(mu_pre_i) with fixed sigma/cumulants.

    messages has the exact _e2_messages layout (3,n,cm). z0 and z1 are the
    two encoder pre-SiLU arrays from the ORIGINAL forward call. x is the raw
    features array before normalization/clipping. P is the current head's
    public parameter dictionary. Optional mean/delta/base_mean_dmu implement
    the current FHM lower clamp consistently.
    """
    n, nx = map(int, x.shape)
    cm = int(P['msg_w'].shape[0]) // 3
    assert tuple(dfeatures_dalpha.shape) == (n, nx)
    assert tuple(messages.shape) == (3, n, cm)
    assert int(P['enc0_w'].shape[1]) == nx + 6 * cm
    assert tuple(z0.shape) == tuple(z1.shape) == (n, int(P['enc0_w'].shape[0]))
    assert 1 <= int(layer) <= 14  # This proposal changes propagated covariance.

    normalized = (x - P['centers'][layer]) / P['scales'][layer]
    Hx = xp.clip(normalized, -8.0, 8.0)
    dx = xp.where(xp.abs(normalized) < 8.0,
                  dfeatures_dalpha / P['scales'][layer], 0.0)
    # Hm[:48] = Hg * M, with Hg the normalized/clipped feature5, NOT raw gate.
    # The original final clip on Hm must also gate its derivative.
    M = xp.reshape(xp.transpose(messages, (1, 0, 2)), (n, 3 * cm))
    message_raw = Hx[:, 5:6] * M
    dm = xp.where(xp.abs(message_raw) < 8.0, dx[:, 5:6] * M, 0.0)
    # The energy/log1p message half is fixed and has zero tangent: omit its
    # columns from this paid product instead of multiplying explicit zeros.
    dH = xp.concatenate((dx, dm), axis=1)
    dz0 = dH @ P['enc0_w'][:, :nx + 3 * cm].T
    dh0 = dz0 * _silu_prime(xp, z0)
    dz1 = dh0 @ P['enc2_w'].T
    # Collapse the fixed final64-dimensional state/head pair beforehand.
    # In production this128-vector belongs in per-layer setup constants.
    head128 = P['enc4_w'].T @ P['head'][layer]
    dalpha = (dz1 * _silu_prime(xp, z1)) @ head128
    gain = 1.0 + 0.25 * xp.tanh(P['gain'][layer:layer + 1])
    # delta=sigma*rms*gain*head, d alpha/d mu=1/sigma, so sigma cancels.
    derivative = dalpha * P['rms'][layer] * gain
    if mean is not None or delta is not None or base_mean_dmu is not None:
        assert mean is not None and delta is not None and base_mean_dmu is not None
        assert tuple(mean.shape) == tuple(delta.shape) == tuple(base_mean_dmu.shape) == (n,)
        derivative = xp.where(delta > -mean, derivative, -base_mean_dmu)
    return derivative


def first_price_covariance(xp, pre_cov, gaussian_gate, derivative):
    """Leading weak-correlation correction; not an arbitrary-law identity."""
    row = derivative[:, None] * gaussian_gate[None, :]
    correction = pre_cov * (row + row.T)
    xp.fill_diagonal(correction, 0.0)
    return correction
