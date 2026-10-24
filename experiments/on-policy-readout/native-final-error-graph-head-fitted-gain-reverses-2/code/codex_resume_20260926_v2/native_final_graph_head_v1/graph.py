"""Target-free final-error observer features; no values return to the host."""
import numpy as np


def graph_fields(m, C, aux):
    C = np.array(C, dtype=np.float64)
    variance = np.maximum(np.diag(C), 1e-12)
    sd = np.sqrt(variance)
    R = C / (sd[:, None] * sd[None, :])
    # The host can be slightly asymmetric. Retain directed neighbor rows.
    np.fill_diagonal(R, 0)
    a, g, p = (np.asarray(aux[k], dtype=np.float64) for k in ('alpha', 'gate', 'pdf'))
    phi = np.column_stack((a, g, p))
    n = len(a)
    idx = np.argsort(-np.abs(R), axis=1, kind='stable')[:, :8]
    rn = np.take_along_axis(R, idx, axis=1)
    neigh = phi[idx]
    signed = np.mean(rn[:, :, None] * neigh, axis=1)
    absolute = np.mean(np.abs(rn)[:, :, None] * neigh, axis=1)
    energy = np.column_stack((np.sum(rn * rn, axis=1), np.max(np.abs(rn), axis=1)))
    soft = []
    for gamma in (8., 32.):
        z = gamma * R * R
        # Exclude self from the soft neighborhood as well as the top eight.
        np.fill_diagonal(z, -np.inf)
        z -= z.max(axis=1, keepdims=True)
        w = np.exp(z)
        w /= w.sum(axis=1, keepdims=True)
        soft.append(w @ phi)
    lv = np.log(variance)
    seed = np.column_stack((np.ones(n), a, g, p, a*p, (a*a-1)*p,
                            np.asarray(m, dtype=np.float64)/sd, lv-lv.mean()))
    seed /= np.maximum(np.linalg.norm(seed, axis=0), 1e-12)
    Q = np.linalg.qr(seed, mode='reduced')[0]
    S = (C + C.T) * .5 / variance.mean()
    for _ in range(8):
        Q = np.linalg.qr(S @ Q, mode='reduced')[0]
    # Projector features avoid arbitrary eigenvector signs and rotations.
    projected = Q @ (Q.T @ phi)
    leverage = np.sum(Q*Q, axis=1)
    energy_fraction = np.sum(Q*(S @ Q), axis=1) / (variance/variance.mean())
    out = np.concatenate((signed, absolute, energy, *soft, projected,
                          leverage[:, None], energy_fraction[:, None]), axis=1)
    assert out.shape == (n, 19) and np.isfinite(out).all()
    return out


def transport(history, gate, weight):
    return (np.asarray(weight, dtype=np.float64).T @ history) * np.asarray(gate)[:, None]
