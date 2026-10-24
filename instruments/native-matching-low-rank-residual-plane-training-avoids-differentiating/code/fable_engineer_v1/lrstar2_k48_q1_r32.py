


# BEGIN FABLE-ENGINEER LOW-RANK STAR PLANE v2 (E-62 --lrstar2 k=48 q=1 r=32, 2026-09-24; an APPROXIMATION, gated on VALID40, not bit-exact):
# the star plane A (the K21-plane residual leaving each layer, 1024x1024 float32) feeds only the next layer's star term and, leaving
# L14, the L15 star diagonal. The porter's no-refit torch test found rank truncation of that plane is not a cliff (rank 32: +0.37%
# raw pooled over 8 VALID nets, CI incl. 0). Here A ~ Q B with a randomized range finder: Y = A Omega (Omega a fixed n x k Gaussian
# drawn once per worker from a seeded generator), Q = qr(Y), B = Q^T A. Then push = w^T A = (w^T Q) B and s2 = (w^T o w^T) push^T
# = ((w^T o w^T) B^T)(w^T Q)^T are thin products; s1 = (push o w^T) w stays a full product and now shares one batched depth-6 pair
# with the covariance's C w (same right operand w), so 2 of the 4 batched 1024^3 products per layer disappear.
# v2: the star plane's spectrum is flat (98% of its energy needs rank 131-720), so a plain sketch keeps the right energy but the
# wrong directions (k=48 without power steps: final MSE +4.5..+10.9% on 35/100/37/39 against torch's exact rank-32 truncation
# +0.5..+2.6% on 37/39). v2 adds 1 subspace (power) steps with re-orthonormalization and, when _LR_R is set, the exact rank-r
# truncation of the sketched plane (small SVD of B), i.e. a randomized SVD.
_LR_K = 48
_LR_Q = 1
_LR_R = 32   # None = keep the rank-k projection
_LR_SEED = 20260924
_LR_OMEGA = [None]


def _lr_factor(A32):
    n = int(A32.shape[0])
    om = _LR_OMEGA[0]
    if om is None or int(om.shape[0]) != n:
        om = fnp.random.default_rng(_LR_SEED).standard_normal((n, _LR_K), dtype=fnp.float32)
        _LR_OMEGA[0] = om
    Q = fnp.linalg.qr(A32 @ om)[0]
    for _ in range(_LR_Q):
        Q = fnp.linalg.qr(A32 @ fnp.linalg.qr(A32.T @ Q)[0])[0]
    B = Q.T @ A32
    if _LR_R is None:
        return Q, B
    U, S, Vt = fnp.linalg.svd(B, full_matrices=False)
    return Q @ U[:, :_LR_R], S[:_LR_R, None] * Vt[:_LR_R]


def _flms_cov_star(w, C, A):
    # = (_flms_covariance(w, C), _flms_star(w, A)) with the star plane projected to rank _LR_K
    if w.ndim != 2 or C.shape != (w.shape[0], w.shape[0]) or A.shape != (w.shape[0], w.shape[0]):
        raise ValueError('FLMS covariance/star shapes')
    if str(w.dtype) != 'float32':
        raise TypeError('FLMS weights must be FP32')
    wt = w.T
    C32 = C.astype(fnp.float32)
    A32 = A if str(A.dtype) == 'float32' else A.astype(fnp.float32)
    Q, B = _lr_factor(A32)
    P1 = wt @ Q
    pushed = P1 @ B
    pw = pushed * wt
    wt2 = wt * wt
    first, s1 = _mm_batch([(C32, w), (pw, w)])
    s2 = (wt2 @ B.T) @ P1.T
    cov = _sym_upper(wt, first)
    Cpre = cov.astype(fnp.float64)
    sub = (2 * s1 + s2) / 3
    diagonal = fnp.sum(pw * wt, axis=1, dtype=fnp.float64)
    result = sub.astype(fnp.float64)
    fnp.fill_diagonal(result, diagonal)
    return Cpre, result


def _flms_star(w, A):
    # the unbatched star term (not reached while the covariance/star pair is batched), same projection
    if w.ndim != 2 or A.shape != (w.shape[0], w.shape[0]):
        raise ValueError('FLMS star shapes')
    if str(w.dtype) != 'float32':
        raise TypeError('FLMS weights must be FP32')
    wt = w.T
    Q, B = _lr_factor(A if str(A.dtype) == 'float32' else A.astype(fnp.float32))
    P1 = wt @ Q
    pw = (P1 @ B) * wt
    sub = (2 * _mm(pw, w) + ((wt * wt) @ B.T) @ P1.T) / 3
    diagonal = fnp.sum(pw * wt, axis=1, dtype=fnp.float64)
    result = sub.astype(fnp.float64)
    fnp.fill_diagonal(result, diagonal)
    return result


def _flms_star_diagonal(w, A):
    # the L15 star diagonal (w = the kept rows' weight columns), same projection of the plane leaving L14
    if w.ndim != 2 or A.shape != (w.shape[0], w.shape[0]):
        raise ValueError('FLMS star diagonal shapes')
    if str(w.dtype) != 'float32':
        raise TypeError('FLMS weights must be FP32')
    wt = w.T
    Q, B = _lr_factor(A.astype(fnp.float32))
    pushed = (wt @ Q) @ B
    return fnp.sum((pushed * wt) * wt, axis=1, dtype=fnp.float64)
# END FABLE-ENGINEER LOW-RANK STAR PLANE v2
