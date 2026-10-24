def _rap4_factor(Fu, tail, Fv, U, rows, lam_pair):
    """Preserve four full-bank primal and read raw common-read actions in a fixed248+8 frame."""
    h = _E2Hold()
    energy = h(fnp.sum(h(Fu * Fu), axis=1))
    order = h(fnp.argsort(energy,kind='stable'))
    drop,keep = fnp.sort(order[:8]),fnp.sort(order[8:])
    joint = fnp.concatenate([drop,fnp.arange(256,320)])
    T = h(fnp.concatenate([Fu[drop],tail]))
    # Independent thin products: never touch private UBC/UBX transform caches.
    Ur = h(U[rows])
    full = h(fnp.concatenate([Fu,tail]))
    Pu = h(Ur[:,:320] @ full)
    Pv = h(Ur @ Fv)
    S = h(h(h(Pu * Pv) * lam_pair[None,:]) * 4.0)
    H = h(S @ T.T)
    # Reduced QR has a deterministic orthonormal completion even at rank<8.
    # No inverse, rank threshold, ridge, or outcome-dependent fallback.
    Q,R = fnp.linalg.qr(h(h(fnp.concatenate([Ur[:,joint].T,H.T],axis=1)).astype(fnp.float64)),mode='reduced')
    A = h(Q.astype(fnp.float32))
    B = h(A.T @ T)
    return fnp.concatenate([Fu[keep],B]),keep,joint,A


def _rap4_probe_rows(U,ctx):
    h = _E2Hold()
    mapping = ctx['rows']
    global_rows = fnp.arange(int(U.shape[0])) if mapping is None else mapping
    if int(global_rows.shape[0]) != int(U.shape[0]):
        raise RuntimeError('RAP4 full basis row map')
    # The birth-layer response roster and sigma/alpha refer to the same frame.
    alpha = h(_E2_LAYER['a'][global_rows])
    sigma = h(_E2_LAYER['s'][global_rows])
    response = fnp.zeros(int(_E2_LAYER['a'].shape[0]),dtype=fnp.bool_)
    h(fnp.put(response,_E2_LAYER['ecut_rows'],True))
    eligible = h(response[global_rows])
    if int(fnp.count_nonzero(eligible)) < 4:
        raise RuntimeError('RAP4 requires four actual response rows')
    pdf = h(fnp.exp(h(h(alpha * alpha) * -0.5)))
    sensitivity = h(h(pdf * 0.3989422804014327) / sigma)
    ranked = h(fnp.argsort(h(fnp.where(eligible,-sensitivity,fnp.inf)),kind='stable'))
    return h(ranked[:4]),global_rows
