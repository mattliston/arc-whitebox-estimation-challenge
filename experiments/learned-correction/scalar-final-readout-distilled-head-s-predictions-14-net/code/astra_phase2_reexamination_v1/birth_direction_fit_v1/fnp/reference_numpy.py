"""Independent, inert NumPy reference and one small deterministic fixture.

NumPy is passed explicitly by the caller: this module imports no candidate,
FNP, learned payload, model, target, or numerical library.  The explicit
signed-cube reader is for tiny qualification only, never deployment.
"""

import math


def directions(np, bp, Cpre, K21, aux, d):
    """The four literal FP64 directions, including column projection."""
    B = np.asarray(bp, dtype=np.float64)
    C = np.asarray(Cpre, dtype=np.float64)
    T = np.asarray(K21, dtype=np.float64)
    g = np.asarray(aux['gate'], dtype=np.float64)
    alpha = np.asarray(aux['alpha'], dtype=np.float64)
    sigma = np.asarray(aux['sigma'], dtype=np.float64)
    pdf = np.asarray(aux['pdf'], dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    t3 = np.asarray(aux['pre_k3'], dtype=np.float64) / (sigma * sigma * sigma)
    eta = alpha / sigma
    raw = [
        (0.5 * d[:, None]) * T,
        (-0.5 * g[:, None]) * T.T * eta[None, :],
        (-0.5 * d[:, None]) * (C * C) * eta[None, :],
        C * (t3 * (alpha * alpha - 1) * pdf / 6)[:, None],
    ]
    denominator = np.maximum(np.sum(B * B, axis=0), 1e-20)
    result = []
    for F in raw:
        np.fill_diagonal(F, 0.0)
        inner = np.sum(B * F, axis=0)
        result.append(F - B * (inner / denominator)[None, :])
    return tuple(result)


def signed_cube_plane(np, base32, a32, t32, s32, births):
    """Direct sum of actual stored signed cubes, not the delta identity.

    All factors are interpreted as exact values of their FP32 storage.
    A coordinate-born pair at column j has base[j,j]=0, with its two
    diagonal entries replaced by the stored t_j and s_j respectively.
    """
    n = int(base32.shape[0])
    result = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        if not bool(births[j]):
            continue
        p = np.asarray(base32[:, j], dtype=np.float64).copy()
        q = p.copy()
        p[j] = float(t32[j])
        q[j] = float(s32[j])
        result += float(a32[j]) * (np.outer(p * p, p) - np.outer(q * q, q))
    return result


def update(np, bp, Cpre, K21, aux, d, alive, births, a, t, s, coefficients):
    """Independent bases plus explicit stored-cube plane difference.

    Return a dictionary to expose basis, storage and plane ownership
    separately to the caller's qualification.  No assertion or tolerance is
    selected here, and no native participant is imported.
    """
    basis = directions(np, bp, Cpre, K21, aux, d)
    c = np.asarray(coefficients, dtype=np.float32).astype(np.float64)
    change = np.zeros_like(bp, dtype=np.float64)
    for j in range(4):
        change = change + c[j] * basis[j]
    np.fill_diagonal(change, 0.0)
    old32 = (bp * alive[:, None]).astype(np.float32)
    new32 = ((bp + change) * alive[:, None]).astype(np.float32)
    a32 = np.asarray(a, dtype=np.float32)
    t32 = (t * alive).astype(np.float32)
    s32 = (s * alive).astype(np.float32)
    old_plane = signed_cube_plane(np, old32, a32, t32, s32, births)
    new_plane = signed_cube_plane(np, new32, a32, t32, s32, births)
    delta = new_plane - old_plane
    return dict(basis=basis, change=change, old_base32=old32,
                new_base32=new32, a32=a32, t32=t32, s32=s32,
                old_plane=old_plane, new_plane=new_plane, delta_plane=delta)


def fixture(np):
    """Fixed n=11 non-symmetric plane, inactive rows and nontrivial signs.

    This only constructs a fixture when explicitly called.  It includes a
    zero common-base column to exercise the fixed 1e-20 projection floor.
    It does not purport to be a network state or a score witness.
    """
    n = 11
    index = np.arange(n * n, dtype=np.float64).reshape(n, n)
    W = np.sin((index + 1) / 13) / 5
    Cpre = W.T @ W + np.diag(np.linspace(0.7, 1.3, n))
    sigma = np.sqrt(np.diag(Cpre))
    alpha = np.linspace(-1.7, 1.4, n)
    gate = np.asarray([0.5 * (1 + math.erf(float(x) / math.sqrt(2)))
                       for x in alpha], dtype=np.float64)
    pdf = np.exp(-alpha * alpha / 2) / math.sqrt(2 * math.pi)
    K21 = np.cos((index + 3) / 7) / 30
    np.fill_diagonal(K21, np.linspace(-0.04, 0.05, n) * sigma ** 3)
    bp = gate[:, None] * Cpre
    np.fill_diagonal(bp, 0.0)
    bp[:, 2] = 0.0
    alive = np.asarray([True, False, True, True, True, False,
                        True, True, True, True, True], dtype=bool)
    births = np.asarray([True, False, True, False, True, False,
                         True, True, False, True, True], dtype=bool)
    aux = dict(alpha=alpha, sigma=sigma, gate=gate, pdf=pdf,
               pre_k3=np.diag(K21).copy())
    return dict(bp=bp, Cpre=Cpre, K21=K21, aux=aux, d=pdf / sigma,
                alive=alive, births=births,
                a=np.linspace(0.025, 0.075, n),
                t=np.linspace(-0.35, 0.9, n),
                s=np.linspace(-0.8, 0.1, n),
                coefficients=np.asarray([0.125, -0.2, 0.15, -0.1],
                                        dtype=np.float32))
