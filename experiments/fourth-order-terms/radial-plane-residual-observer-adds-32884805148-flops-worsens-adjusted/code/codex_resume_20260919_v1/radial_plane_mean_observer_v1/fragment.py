"""A paid fourth-plane residual observer; no writes to the parent state."""
_R22_ENABLED = True


def _r22_source(covariance, fourth_plane):
    if covariance.shape != fourth_plane.shape or covariance.ndim != 2:
        raise ValueError('Fourth-plane observer source shape')
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError('Fourth-plane observer source must be square')
    c = (covariance + covariance.T) / 2
    q = (fourth_plane + fourth_plane.T) / 2
    v = fnp.diag(c)
    cc = c*c
    radial = 3*fnp.sum(q)/(fnp.sum(v)**2 + 2*fnp.sum(cc))
    return (q - (radial/3)*(fnp.outer(v, v) + 2*cc)).astype(fnp.float32)


def _r22_ray(plane, weight):
    if plane.shape != (weight.shape[0], weight.shape[0]):
        raise ValueError('Fourth-plane observer ray shape')
    if str(plane.dtype) != 'float32' or str(weight.dtype) != 'float32':
        raise TypeError('Fourth-plane observer ray operands must be FP32')
    square = weight*weight
    product = fnp.matmul(plane, square)
    e = square.astype(fnp.float64)
    diagonal = fnp.diag(plane).astype(fnp.float64)
    return (3*fnp.sum(e*product.astype(fnp.float64), axis=0)
            - 2*fnp.sum(diagonal[:, None]*(e*e), axis=0))


def _r22_advance(state, weight32, weight64, aux):
    if state['plane'] is None:
        return
    ray = _r22_ray(state['plane'], weight32)
    a, p, s = aux['alpha'], aux['pdf'], aux['sigma']
    source = ((a*a-1)*p/(24*s*s*s))*ray
    if state['mean'] is not None:
        source = source + aux['gate']*(weight64.T @ state['mean'])
    state['mean'] = source


def _r22_read(mean, state):
    if state['mean'] is None:
        return mean.copy()
    return (mean + state['mean']).copy()



def _nonlinear_first_gaussian(mu, C, K21, k4, k22_override=None, metric='identity'):
    gauss_order=6;edge_order=2;local=True;k31=True
    n = len(mu)
    v = fnp.diag(C).copy()
    s, a, pdf, g, mg, r2g, r3g, r4g = gaussian_raw(mu, v)
    d = pdf / s
    e = -a * pdf / v
    f = (a * a - 1) * pdf / (v * s)
    k3 = fnp.diag(K21).copy()
    m = mg + k3 * e / 6 + k4 * f / 24
    r2 = r2g + k3 * d / 3 + k4 * e / 12
    r3 = r3g + k3 * g + k4 * d / 4
    r4 = r4g + 4 * k3 * mg + k4 * g
    var = r2 - m * m
    post_k3 = r3 - 3 * m * r2 + 2 * m ** 3
    post_k4 = r4 - 4 * m * r3 + 6 * m * m * r2 - 3 * m ** 4 - 3 * var * var
    rho = C / fnp.outer(s, s)
    fnp.fill_diagonal(rho, 0)
    T21 = K21 / (v[:, None] * s[None, :])
    fnp.fill_diagonal(T21, 0)
    K22 = k4 / 3 if k22_override is None else k22_override
    T22 = K22 / fnp.outer(v, v)
    t3 = k3 / s ** 3
    t4 = fnp.broadcast_to(k4, (n,)) / v ** 2
    T31 = t4[:, None] * rho if k31 else None
    maxorder = max(gauss_order, edge_order + 4)
    A, B = derivatives(a, pdf, g, mg / s, maxorder)

    aa=fnp.stack(A,axis=0);bb=fnp.stack(B,axis=0)
    F=fnp.stack([aa,bb,bb],axis=0);G=fnp.stack([aa,aa,bb],axis=0)
    ans=fnp.zeros((3,n,n),dtype=fnp.float64)
    rr=fnp.ones_like(C)
    for r in range(7):
        if r:
            ans=ans+rr[None,:,:]*(F[:,r,:,None]*G[:,r,None,:])
        if False:
            ec=(T21[None,:,:]*(F[:,r+2,:,None]*G[:,r+1,None,:])+T21.T[None,:,:]*(F[:,r+1,:,None]*G[:,r+2,None,:]))/2
            ec=ec+T22[None,:,:]*(F[:,r+2,:,None]*G[:,r+2,None,:])/4
            ec=ec+(T31[None,:,:]*(F[:,r+3,:,None]*G[:,r+1,None,:])+T31.T[None,:,:]*(F[:,r+1,:,None]*G[:,r+3,None,:]))/6
            if r:
                ec=ec+((t3[None,:]*F[:,r+3,:])[:,:,None]*G[:,r,None,:]+F[:,r,:,None]*(t3[None,:]*G[:,r+3,:])[:,None,:])/6
                ec=ec+((t4[None,:]*F[:,r+4,:])[:,:,None]*G[:,r,None,:]+F[:,r,:,None]*(t4[None,:]*G[:,r+4,:])[:,None,:])/24
            ans=ans+rr[None,:,:]*ec
        rr=rr*rho/(r+1)
    cov=ans[0]*fnp.outer(s,s)
    pk21=ans[1]*fnp.outer(v,s)
    pk22=ans[2]*fnp.outer(v,v)
    post21 = pk21 - 2 * m[:, None] * cov
    post22 = pk22 - 2 * m[:, None] * pk21.T - 2 * m[None, :] * pk21 + 4 * fnp.outer(m, m) * cov - 2 * cov * cov
    fnp.fill_diagonal(post22, post_k4)
    radial = 3 * fnp.sum(post22) / (n * (n + 2))
    fnp.fill_diagonal(cov, var)
    fnp.fill_diagonal(post21, post_k3)
    if metric == 'covariance':
        radial = 3 * fnp.sum(post22) / (fnp.trace(cov) ** 2 + 2 * fnp.sum(cov * cov))
    aux = dict(sigma=s, alpha=a, pdf=pdf, gate=g, gaussian_mean=mg, pre_v=v, pre_cov=C, pre_k3=k3, pre_k4=fnp.broadcast_to(k4, (n,)).copy(), post_v=var, post_k3=post_k3, post_k4=post_k4)
    aux['r22_post22'] = post22
    return (m, cov, post21, radial, aux)

def _nonlinear_pair_batch(mu, C, K21, k4, k22_override=None, metric='identity'):
    gauss_order=6;edge_order=2;local=True;k31=True
    n = len(mu)
    v = fnp.diag(C).copy()
    s, a, pdf, g, mg, r2g, r3g, r4g = gaussian_raw(mu, v)
    d = pdf / s
    e = -a * pdf / v
    f = (a * a - 1) * pdf / (v * s)
    k3 = fnp.diag(K21).copy()
    m = mg + k3 * e / 6 + k4 * f / 24
    r2 = r2g + k3 * d / 3 + k4 * e / 12
    r3 = r3g + k3 * g + k4 * d / 4
    r4 = r4g + 4 * k3 * mg + k4 * g
    var = r2 - m * m
    post_k3 = r3 - 3 * m * r2 + 2 * m ** 3
    post_k4 = r4 - 4 * m * r3 + 6 * m * m * r2 - 3 * m ** 4 - 3 * var * var
    rho = C / fnp.outer(s, s)
    fnp.fill_diagonal(rho, 0)
    T21 = K21 / (v[:, None] * s[None, :])
    fnp.fill_diagonal(T21, 0)
    K22 = k4 / 3 if k22_override is None else k22_override
    T22 = K22 / fnp.outer(v, v)
    t3 = k3 / s ** 3
    t4 = fnp.broadcast_to(k4, (n,)) / v ** 2
    T31 = t4[:, None] * rho if k31 else None
    maxorder = max(gauss_order, edge_order + 4)
    A, B = derivatives(a, pdf, g, mg / s, maxorder)

    aa=fnp.stack(A,axis=0);bb=fnp.stack(B,axis=0)
    F=fnp.stack([aa,bb,bb],axis=0);G=fnp.stack([aa,aa,bb],axis=0)
    ans=fnp.zeros((3,n,n),dtype=fnp.float32)
    F=F.astype(fnp.float32)
    G=G.astype(fnp.float32)
    rho=rho.astype(fnp.float32)
    T21=T21.astype(fnp.float32)
    T22=T22.astype(fnp.float32)
    T31=T31.astype(fnp.float32)
    t3=t3.astype(fnp.float32)
    t4=t4.astype(fnp.float32)
    ans=_pair_series_fused(F,G,rho,T21,T22,T31,t3,t4,n)   # W-G: the r-loop (~300 remote requests) as broadcast products + axis sums; same float32 products, rounding-level
    assert str(ans.dtype)=='float32'
    ans=ans.astype(fnp.float64)
    cov=ans[0]*fnp.outer(s,s)
    pk21=ans[1]*fnp.outer(v,s)
    pk22=ans[2]*fnp.outer(v,v)
    post21 = pk21 - 2 * m[:, None] * cov
    post22 = pk22 - 2 * m[:, None] * pk21.T - 2 * m[None, :] * pk21 + 4 * fnp.outer(m, m) * cov - 2 * cov * cov
    fnp.fill_diagonal(post22, post_k4)
    radial = 3 * fnp.sum(post22) / (n * (n + 2))
    fnp.fill_diagonal(cov, var)
    fnp.fill_diagonal(post21, post_k3)
    if metric == 'covariance':
        radial = 3 * fnp.sum(post22) / (fnp.trace(cov) ** 2 + 2 * fnp.sum(cov * cov))
    aux = dict(sigma=s, alpha=a, pdf=pdf, gate=g, gaussian_mean=mg, pre_v=v, pre_cov=C, pre_k3=k3, pre_k4=fnp.broadcast_to(k4, (n,)).copy(), post_v=var, post_k3=post_k3, post_k4=post_k4)
    aux['r22_post22'] = post22
    return (m, cov, post21, radial, aux)

nonlinear = _nonlinear_pair_batch

def _predict_binary(mlp, hook=None, cov32=True, metric='covariance', collect_states=False, gate_floor=0.01, final_hook=None, coefficient_hook=None):
    global _W5_PORT_SCOPE
    if _W5_PORT_SCOPE is not None:
        raise RuntimeError('W5 no concurrent or nested network prediction')
    _w5_context = _w5_port_new(mlp)
    _W5_PORT_SCOPE = _w5_context
    try:
        global _SCRATCH
        depth, n = (int(mlp.depth), int(mlp.width))
        W = mlp.weights
        _bd_c = _bd_coefficients()
        if _SCRATCH is None:
            arena = fnp.empty((_PACKED_ELEMENTS,), dtype=fnp.float32)
            _SCRATCH = [arena, _cp3_bind_arena_views(arena, _PACKED_PUSH), _cp3_bind_arena_views(arena, _PACKED_READ), None, None, None, None]
            _bind_push6(_SCRATCH[0])   # W-H: only the batched six-level push layout (the serial-top binding is unused)
        previous = fnp.arange(n)
        previous_alive = fnp.ones(n, dtype=fnp.bool_)
        m = fnp.zeros(n, dtype=fnp.float64)
        C = fnp.eye(n)
        radial = 0.0
        b = fnp.zeros(n, dtype=fnp.float64)
        p = fnp.zeros((n, 0), fnp.float32)
        lam = fnp.zeros(0, fnp.float32)
        star = fnp.zeros((n, n), fnp.float32)
        means, states = ([], [])
        newborn = None
        shared_old = _shared_old384_state()
        r22_state = {'mean': None, 'plane': None}
        for l in range(depth):
            w32 = fnp.asarray(W[l], dtype=fnp.float32)
            _w5_port_begin(_w5_context,mlp,l,w32,previous)
            w = fnp.asarray(W[l], dtype=fnp.float64)
            mu = w.T @ m
            if l == depth - 1 and final_hook is not None:
                v, active_v = (_flms_covariance_diagonal(w32, C) if _FULL_LOWER_MOMENTS
                                else _cov_compact_diagonal(w32, C, previous, previous_alive))
                bpre = w.T @ b
                pp = _shared_old384_push(l, w32, p, previous, newborn, lam, shared_old)
                k3 = fnp.sum(_cp_diagonal_cube(pp.astype(fnp.float64)) * lam.astype(fnp.float64), axis=1)
                k3 = k3 + (_flms_star_diagonal(w32, star) if _FULL_LOWER_MOMENTS
                           else _star_compact_diagonal(w32, star, previous, previous_alive))
                k3 = k3 + 3 * active_v * bpre
                m, aux = nonlinear_scalar(mu, v, k3, radial * v * v)
                if _R22_ENABLED:
                    _r22_advance(r22_state, w32, w, aux)
                m = m + final_hook(l, m, aux, W[l])
                means.append(_r22_read(m, r22_state))
                break
            if l == 0:
                Cpre = _resident_gram512(w32.T, w32).astype(fnp.float64)
                Bpre = Cpre
            elif cov32:
                if _FULL_LOWER_MOMENTS:
                    Cpre = _flms_covariance(w32, C)
                    Bpre = Cpre
                else:
                    Cpre, Bpre = _cov_compact(w32, C, previous, previous_alive)
            else:
                Cpre = w.T @ C @ w
                Bpre = Cpre if _FULL_LOWER_MOMENTS else w.T @ (C * previous_alive[:, None] * previous_alive[None, :]) @ w
            bpre = w.T @ b
            vv = fnp.diag(Cpre)
            response_gate = flops.stats.norm.cdf(mu / fnp.sqrt(fnp.maximum(vv, 1e-10)))
            live_indices, alive = select(response_gate, gate_floor)
            live = alive[:, None] * alive[None, :]
            response_score = fnp.where(alive, fnp.minimum(response_gate, 1 - response_gate), -1.)
            response_indices, response_mask = select(response_score, gate_floor)
            omitted = alive & ~response_mask
            pp = _shared_old384_push(l, w32, p, previous, newborn, lam, shared_old)
            cp_pre = fnp.zeros((n, n))
            if len(lam):
                left = pp[response_indices]
                right = pp[live_indices]
                response_plane = _shared_old384_read(left, right, lam, live_indices, shared_old).astype(fnp.float64)
                fnp.put(cp_pre, response_indices[:, None] * n + live_indices[None, :], response_plane)
                diagonal = fnp.sum(_cp_diagonal_cube(pp.astype(fnp.float64)) * lam.astype(fnp.float64), axis=1)
                fnp.fill_diagonal(cp_pre, diagonal)
            star_pre = (fnp.zeros((n, n), dtype=fnp.float64) if l == 0 else
                        (_flms_star(w32, star) if _FULL_LOWER_MOMENTS else
                         _star_compact(w32, star, previous, previous_alive, live_indices)))
            _w5_port_end_layer(_w5_context)
            K21 = cp_pre + star_pre + fnp.diag(Bpre)[:, None] * bpre[None, :] + 2 * Bpre * bpre[:, None]
            k4 = radial * vv * vv
            k22 = radial * (fnp.outer(vv, vv) + 2 * Cpre * Cpre) / 3
            pair_map = _nonlinear_first_gaussian if l == 0 else (_nonlinear_terminal_live if l == depth - 1 else nonlinear)
            m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric)
            if _R22_ENABLED:
                _r22_advance(r22_state, w32, w, aux)
                if l < depth - 1:
                    r22_state['plane'] = _r22_source(C, aux['r22_post22'])
            host = m.copy()
            if len(lam) and l < depth - 1:
                source_sum = omitted_radial_source(pp, lam, diagonal, omitted, alive, aux, host)
                denominator = fnp.trace(C) ** 2 + 2 * fnp.sum(C * C) if metric == 'covariance' else n * (n + 2)
                radial_new = radial_new + source_sum / denominator
            if _EDGE_ENABLED and 8 <= l <= 14:
                C = C + _edge_source(l,m,K21,aux)
            if hook is not None:
                dm = fnp.asarray(hook(l, m, C, K21, aux, W[l]), fnp.float64)
                C = C - fnp.outer(m, dm) - fnp.outer(dm, m) - fnp.outer(dm, dm)
                m = m + dm
            means.append(_r22_read(m, r22_state))
            if l == depth - 1:
                break
            g = aux['gate']
            d = aux['pdf'] / aux['sigma']
            birth = d * (1 + radial / 3)
            b = ((g * bpre + radial * d * vv / 6) if _FULL_LOWER_MOMENTS
                 else (g * bpre + radial * d * vv / 6) * alive)
            selected, births = select(fnp.minimum(g, 1 - g), gate_floor)
            if not _FULL_LOWER_MOMENTS:
                cv = fnp.diag(C).copy()
                C = C * live
                fnp.fill_diagonal(C, cv)
                td = fnp.diag(target).copy()
                target = target * live
                fnp.fill_diagonal(target, td)
            inherited = g[:, None] ** 2 * g[None, :] * cp_pre
            bp = g[:, None] * Cpre
            fnp.fill_diagonal(bp, 0)
            Bpost = C.copy()
            if not _FULL_LOWER_MOMENTS:
                fnp.fill_diagonal(Bpost, fnp.diag(C) * alive)
            trace_post = fnp.diag(Bpost)[:, None] * b[None, :] + 2 * Bpost * b[:, None]
            # Innovation owns only the diagonal of the old residual temporary.
            beta = 2 * (aux['gaussian_mean'] - host * g)
            innovation = (fnp.diag(target) - fnp.diag(inherited) - fnp.diag(trace_post)).copy()
            safe_birth = fnp.maximum(birth, 1e-12)
            u = fnp.clip(beta / safe_birth, -8 * vv, 8 * vv)
            disc = 4 * innovation / safe_birth - 3 * u * u
            delta = fnp.minimum(fnp.sqrt(fnp.maximum(disc, 0.01 * vv * vv)), 16 * vv)
            t = (u + delta) / 2
            s = (u - delta) / 2
            a = birth / delta
            # Newest numerical tails are owned by the shared-newborn tuple.
            born = bp * bp * (birth * births)[None, :] + (beta * births)[:, None] * bp.T
            fnp.fill_diagonal(born, birth * births * (3 * u * u + delta * delta) / 4)
            _bd_current_c = (_bd_c[l] if l == 0 or coefficient_hook is None
                             else coefficient_hook(l))
            _bd_base32, _bd_plane_delta = _bd_update(
                bp, Cpre, K21, aux, d, alive, births, a, t, s, _bd_current_c)
            del _bd_current_c
            born = born + _bd_plane_delta
            del _bd_plane_delta
            cp_post = (inherited + born) * live
            newborn = (int(pp.shape[1]), selected, _bd_base32[:, selected], (t[selected] * alive[selected]).astype(fnp.float32), (s[selected] * alive[selected]).astype(fnp.float32))
            del _bd_base32
            p = g.astype(fnp.float32)[:, None] * pp
            p = alive.astype(fnp.float32)[:, None] * p
            _shared_old384_commit(l, g, alive, selected, shared_old)
            lam = fnp.concatenate([lam, a[selected].astype(fnp.float32), -a[selected].astype(fnp.float32)])
            rest = target - cp_post - trace_post
            star = (3 * rest.T - 2 * fnp.diag(fnp.diag(rest))).astype(fnp.float32)
            if collect_states:
                states.append(dict(layer=l, active=int(alive.sum()), births=len(selected), columns=len(lam)))
            previous, previous_alive = (live_indices, alive)
            radial = radial_new
        return fnp.asarray(fnp.maximum(fnp.stack(means), 0.0), dtype=fnp.float32)
    except BaseException:
        _w5_context['failed'] = True
        raise
    finally:
        try:
            _w5_port_finish(_w5_context)
        finally:
            _W5_PORT_SCOPE = None
