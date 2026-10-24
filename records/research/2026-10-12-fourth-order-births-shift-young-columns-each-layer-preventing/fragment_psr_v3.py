# BEGIN OPUS-ENGINEER a62 PAIR-SPLIT YOUNG READ v3 (file written 2026-10-12 03:40Z; CP2) = v1 (split read; modes None / 'split' / 'basis_full', certified)
# plus mode 'nys': the C half through a rank-_PSR_RANK Nystrom basis rebuilt EVERY layer from the current exact c's at the live rows
# (the old tier's join recipe: rows = the _PSR_RANK largest diagonal entries of the lam+^2-weighted c Gram on the live rows;
# Y = C (lam^2 o C[rows])', F = Y[rows]^-1 C[rows] in float64, so C ~ Y F, exact on those rows): C half = ((A + B) F') Y'.
# A pushed basis cannot hold the c's (F4C shifts the young columns by a quadratic form for ages <= _F4C_AGE), so none is kept.
# _PSR_RANK = None: full rank on the live rows (rows = every live row: C Lam^2 C' (C Lam^2 C')^-1 C = C), the CP2 certificate.
# v1 header: (file written 2026-10-12 03:25Z; lead: the physicist's torch A384 passed; CP0 / CP1 of the native build).
# The young part of the bank read (_cp_read_young_sq's young_plane = YL (YR o lam)', YL the response rows of the squared columns with the
# pair law's fold, YR the live rows of the pushed columns) split EXACTLY by pairs: every young cohort is [K plus columns | K minus columns]
# (the pair law's own roster blocks, _NP_STATE['sizes'], from the old tier's column count on), so with A = L+ o lam+, B = L- o lam- and
# C = (P+ + P-)/2, H = (P+ - P-)/2 (P the live rows of the pair's two columns):  YL (YR o lam)' = (A + B) C' + (A - B) H'
# (general weights: no lam- = -lam+ assumption). Modes (_PSR_MODE): None = the parent read (bit for bit); 'split' = the split read (CP0:
# the parent to rounding); 'basis_full' = the C half through a full-rank orthonormal basis of the live rows, (A + B) C' = ((A + B) F') U'
# with F = U' C (CP1: the parent to rounding, exercises the (A) path). Layers _PSR_LAYERS only; anything unexpected falls back to the
# parent read. The old tier's part is computed exactly as the parent computes it.
_PSR_MODE = None
_PSR_LAYERS = (8, 9, 10, 11, 12, 13, 14)
_PSR_LAYER = [None]
_PSR_RANK = 128
_PSR_STATS = dict(split=0, fallback=0, basis=0, nys=0)
_PSR_UFULL = {}
_PSR_PARENT_READ = _RC_PARENT_READ          # the original read (fragment_readcol_v1 calls _RC_PARENT_READ by name)
_PSR_PARENT_LEFT = _np_left


def _np_left(l, *args, **kwargs):
    _PSR_LAYER[0] = l
    return _PSR_PARENT_LEFT(l, *args, **kwargs)


def _psr_young_blocks(old, young_width):
    sizes = _NP_STATE.get('sizes')
    if sizes is None:
        return None
    at, i = 0, 0
    while at < old and i < len(sizes):
        at += int(sizes[i]); i += 1
    if at != old:
        return None
    blocks = [int(x) for x in sizes[i:]]
    if sum(blocks) != young_width or len(blocks) % 2 or any(blocks[j] != blocks[j + 1] for j in range(0, len(blocks), 2)):
        return None
    return blocks


def _psr_plus_spans(blocks):
    # (offset, K) of every cohort's plus block inside the young columns
    out, at = [], 0
    for j in range(0, len(blocks), 2):
        out.append((at, blocks[j])); at += 2 * blocks[j]
    return out


def _psr_ufull(L):
    U = _PSR_UFULL.get(L)
    if U is None:
        import numpy as _psr_np
        g = _psr_np.random.default_rng(20261012).standard_normal((L, L)).astype(_psr_np.float32)
        U = fnp.linalg.qr(fnp.asarray(g))[0].astype(fnp.float32)
        _PSR_UFULL[L] = U
    return U


def _psr_read(left, young_right, lam, live_indices, state):
    l = _PSR_LAYER[0]
    old = int(state['columns'])
    if _PSR_MODE is None or l not in _PSR_LAYERS or not old:
        return _PSR_PARENT_READ(left, young_right, lam, live_indices, state)
    blocks = _psr_young_blocks(old, int(young_right.shape[1]))
    if blocks is None:
        _PSR_STATS['fallback'] += 1
        return _PSR_PARENT_READ(left, young_right, lam, live_indices, state)
    hold = _E2Hold()
    old_left = hold(left[:, :old])                     # the old tier's part, as the parent computes it
    inner = hold(_cp_slr_inner_sq(old_left, lam[:old], state))
    old_plane = hold(_shared_old384_mm(inner, hold(state['pre_basis'][live_indices, :].T)))
    yl = hold(left[:, old:])
    lamy = hold(lam[old:])
    Ap, Am, Cp, Hp = [], [], [], []
    at = 0
    for j in range(0, len(blocks), 2):
        K = blocks[j]
        a = hold(hold(yl[:, at:at + K]) * hold(lamy[at:at + K]))
        b = hold(hold(yl[:, at + K:at + 2 * K]) * hold(lamy[at + K:at + 2 * K]))
        Ap.append(hold(a + b)); Am.append(hold(a - b))
        pplus, pminus = hold(young_right[:, at:at + K]), hold(young_right[:, at + K:at + 2 * K])
        Cp.append(hold(hold(pplus + pminus) * 0.5)); Hp.append(hold(hold(pplus - pminus) * 0.5))
        at += 2 * K
    D = hold(fnp.concatenate(Ap, axis=1)) if len(Ap) > 1 else Ap[0]
    S = hold(fnp.concatenate(Am, axis=1)) if len(Am) > 1 else Am[0]
    C = hold(fnp.concatenate(Cp, axis=1)) if len(Cp) > 1 else Cp[0]
    H = hold(fnp.concatenate(Hp, axis=1)) if len(Hp) > 1 else Hp[0]
    if _PSR_MODE == 'basis_full':
        U = _psr_ufull(int(C.shape[0]))
        F = hold(_mm(hold(U.T), C))                          # (L x K) coefficients of the c's in the basis
        cpart = hold(_mm(hold(_mm(D, hold(F.T))), hold(U.T)))
        _PSR_STATS['basis'] += 1
    elif _PSR_MODE == 'nys' and (_PSR_RANK is None or int(_PSR_RANK) < min(int(C.shape[0]), int(C.shape[1]))):
        r = min(int(C.shape[0]), int(C.shape[1])) if _PSR_RANK is None else int(_PSR_RANK)   # None: full rank (the certificate)
        w2 = hold(fnp.concatenate([hold(hold(lamy[at2:at2 + K2]) * hold(lamy[at2:at2 + K2])) for at2, K2 in _psr_plus_spans(blocks)]))   # lam+^2 per pair
        diag = hold(fnp.sum(hold(hold(C * C) * w2), axis=1))  # the weighted Gram's diagonal on the live rows
        rows = hold(fnp.sort(hold(fnp.argsort(hold(-diag.astype(fnp.float64)), kind='stable'))[:r]))
        Cr = hold(C[rows, :])                                # (r, K)
        Y = hold(_mm(C, hold(hold(Cr * w2).T)))              # (L, r) = C Lam^2 C[rows]'
        F = hold(fnp.linalg.solve(hold(Y[rows, :]).astype(fnp.float64), Cr.astype(fnp.float64)).astype(fnp.float32))   # (r, K)
        cpart = hold(_mm(hold(_mm(D, hold(F.T))), hold(Y.T)))
        _PSR_STATS['nys'] += 1
    else:
        cpart = hold(_mm(D, hold(C.T)))
    young_plane = hold(cpart + hold(_mm(S, hold(H.T))))
    _PSR_STATS['split'] += 1
    return young_plane + old_plane


_RC_PARENT_READ = _psr_read
# END OPUS-ENGINEER a62 PAIR-SPLIT YOUNG READ v3
