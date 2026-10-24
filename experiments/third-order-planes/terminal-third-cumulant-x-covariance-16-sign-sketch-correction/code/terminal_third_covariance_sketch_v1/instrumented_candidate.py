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
        _ttcs_packet = None
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
        for l in range(depth):
            w32 = fnp.asarray(W[l], dtype=fnp.float32)
            _w5_port_begin(_w5_context,mlp,l,w32,previous)
            w = fnp.asarray(W[l], dtype=fnp.float64)
            mu = w.T @ m
            if l == depth - 1 and final_hook is not None:
                v, active_v = _cov_compact_diagonal(w32, C, previous, previous_alive)
                bpre = w.T @ b
                pp = _shared_old384_push(l, w32, p, previous, newborn, lam, shared_old)
                k3 = fnp.sum(_cp_diagonal_cube(pp.astype(fnp.float64)) * lam.astype(fnp.float64), axis=1)
                k3 = k3 + _star_compact_diagonal(w32, star, previous, previous_alive)
                k3 = k3 + 3 * active_v * bpre
                m, aux = nonlinear_scalar(mu, v, k3, radial * v * v)
                m = m + final_hook(l, m, aux, W[l])
                if _ttcs_packet is not None:
                    _ttcs_delta = _TTCS_FINISH(_ttcs_packet,w32,pp,aux,m)
                    m = m + _ttcs_delta
                    del _ttcs_packet,_ttcs_delta
                means.append(m.copy())
                break
            if l == 0:
                Cpre = _resident_gram512(w32.T, w32).astype(fnp.float64)
                Bpre = Cpre
            elif cov32:
                Cpre, Bpre = _cov_compact(w32, C, previous, previous_alive)
            else:
                Cpre = w.T @ C @ w
                Bpre = w.T @ (C * previous_alive[:, None] * previous_alive[None, :]) @ w
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
            star_pre = fnp.zeros((n, n), dtype=fnp.float64) if l == 0 else _star_compact(w32, star, previous, previous_alive, live_indices)
            _w5_port_end_layer(_w5_context)
            K21 = cp_pre + star_pre + fnp.diag(Bpre)[:, None] * bpre[None, :] + 2 * Bpre * bpre[:, None]
            k4 = radial * vv * vv
            k22 = radial * (fnp.outer(vv, vv) + 2 * Cpre * Cpre) / 3
            pair_map = _nonlinear_first_gaussian if l == 0 else (_nonlinear_terminal_live if l == depth - 1 else nonlinear)
            m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric)
            if l == 14 and depth == 16 and n == 1024:
                _ttcs_packet = _TTCS_PREPARE(pp,lam,w32,star,previous_alive,Bpre,bpre,Cpre,aux["gate"],aux["pdf"],aux["sigma"],alive,C,K21)
            host = m.copy()
            if len(lam) and l < depth - 1:
                source_sum = omitted_radial_source(pp, lam, diagonal, omitted, alive, aux, host)
                denominator = fnp.trace(C) ** 2 + 2 * fnp.sum(C * C) if metric == 'covariance' else n * (n + 2)
                radial_new = radial_new + source_sum / denominator
            if hook is not None:
                dm = fnp.asarray(hook(l, m, C, K21, aux, W[l]), fnp.float64)
                C = C - fnp.outer(m, dm) - fnp.outer(dm, m) - fnp.outer(dm, dm)
                m = m + dm
            means.append(m.copy())
            if l == depth - 1:
                break
            g = aux['gate']
            d = aux['pdf'] / aux['sigma']
            birth = d * (1 + radial / 3)
            b = (g * bpre + radial * d * vv / 6) * alive
            selected, births = select(fnp.minimum(g, 1 - g), gate_floor)
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
