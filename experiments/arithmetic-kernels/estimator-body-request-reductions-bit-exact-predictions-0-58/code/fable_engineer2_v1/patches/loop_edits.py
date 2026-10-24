# E2 live-loop edits: (old, new) exact replacements applied ONLY inside the live (last) _predict_binary definition.
# Every new line computes the same float64/float32 values in the same order; shared intermediates are published in _E2_LAYER
# (validated by object identity at each consumer) so the pair map, _bd_update, omitted_radial_source and the old-tier commit
# reuse them instead of recomputing.
EDITS = [
("""            w = fnp.asarray(W[l], dtype=fnp.float64)
            mu = w.T @ m
""", """            w = fnp.asarray(W[l], dtype=fnp.float64)
            wT = w.T
            mu = wT @ m
"""),
("""                bpre = w.T @ b
                pp = _shared_old384_push(l, w32, p, previous, newborn, lam, shared_old)
""", """                bpre = wT @ b
                pp = _shared_old384_push(l, w32, p, previous, newborn, lam, shared_old)
"""),
("""                means.append(m.copy())
                break
""", """                means.append(m)
                break
"""),
("""            bpre = w.T @ b
            vv = fnp.diag(Cpre)
            response_gate = flops.stats.norm.cdf(mu / fnp.sqrt(fnp.maximum(vv, 1e-10)))
            live_indices, alive = select(response_gate, gate_floor)
            live = alive[:, None] * alive[None, :]
            response_score = fnp.where(alive, fnp.minimum(response_gate, 1 - response_gate), -1.)
""", """            bpre = wT @ b
            vv = fnp.diag(Cpre)
            _e2_vc = fnp.maximum(vv, 1e-10)
            _e2_s = fnp.sqrt(_e2_vc)
            _e2_a = mu / _e2_s
            response_gate = flops.stats.norm.cdf(_e2_a)
            _E2_LAYER.clear()
            _E2_LAYER.update(mu=mu, Cpre=Cpre, vc=_e2_vc, s=_e2_s, a=_e2_a, g=response_gate)
            live_indices, alive = select(response_gate, gate_floor)
            live = fnp.outer(alive, alive)
            _e2_gate_min = fnp.minimum(response_gate, 1 - response_gate)
            response_score = fnp.where(alive, _e2_gate_min, -1.)
"""),
("""                fnp.put(cp_pre, response_indices[:, None] * n + live_indices[None, :], response_plane)
                diagonal = fnp.sum(_cp_diagonal_cube(pp.astype(fnp.float64)) * lam.astype(fnp.float64), axis=1)
""", """                fnp.put(cp_pre, (response_indices * n)[:, None] + live_indices, response_plane)
                _e2_pp64 = pp.astype(fnp.float64)
                _e2_pp2 = _e2_pp64 * _e2_pp64
                _e2_lam64 = lam.astype(fnp.float64)
                _E2_LAYER.update(pp=pp, lam=lam, pp64=_e2_pp64, pp2=_e2_pp2, lam64=_e2_lam64)
                diagonal = fnp.sum(_e2_pp2 * _e2_pp64 * _e2_lam64, axis=1)
                del _e2_pp64, _e2_pp2
"""),
("""            K21 = cp_pre + star_pre + fnp.diag(Bpre)[:, None] * bpre[None, :] + 2 * Bpre * bpre[:, None]
            k4 = radial * vv * vv
            k22 = radial * (fnp.outer(vv, vv) + 2 * Cpre * Cpre) / 3
""", """            K21 = cp_pre + star_pre + fnp.outer(vv if Bpre is Cpre else fnp.diag(Bpre), bpre) + Bpre * (2 * bpre)[:, None]
            k4 = radial * vv * vv
            _e2_VV = fnp.outer(vv, vv)
            _e2_CC = Cpre * Cpre
            _E2_LAYER.update(VV=_e2_VV, CC=_e2_CC)
            k22 = radial * (_e2_VV + 2 * _e2_CC) / 3
            del _e2_VV
"""),
("""            host = m.copy()
            if len(lam) and l < depth - 1:
                source_sum = omitted_radial_source(pp, lam, diagonal, omitted, alive, aux, host)
                denominator = fnp.trace(C) ** 2 + 2 * fnp.sum(C * C) if metric == 'covariance' else n * (n + 2)
""", """            host = m
            _e2_beta = 2 * (aux['gaussian_mean'] - host * aux['gate'])
            _E2_LAYER.update(beta_host=host, beta_aux=aux, beta0=_e2_beta)
            if len(lam) and l < depth - 1:
                source_sum = omitted_radial_source(pp, lam, diagonal, omitted, alive, aux, host)
                _e2_pair = _E2_LAYER.get('pair')
                denominator = (_e2_pair['den'] if metric == 'covariance' and _e2_pair is not None and _e2_pair['cov'] is C
                               else (fnp.trace(C) ** 2 + 2 * fnp.sum(C * C) if metric == 'covariance' else n * (n + 2)))
"""),
("""                dm = fnp.asarray(hook(l, m, C, K21, aux, W[l]), fnp.float64)
""", """                dm = hook(l, m, C, K21, aux, W[l])
                if str(dm.dtype) != 'float64':
                    dm = fnp.asarray(dm, fnp.float64)
"""),
("""            means.append(m.copy())
            if l == depth - 1:
                break
            g = aux['gate']
            d = aux['pdf'] / aux['sigma']
""", """            means.append(m)
            if l == depth - 1:
                break
            g = aux['gate']
            _e2_pair = _E2_LAYER.get('pair')
            d = _e2_pair['d'] if _e2_pair is not None and _e2_pair['aux'] is aux else aux['pdf'] / aux['sigma']
"""),
("""            selected, births = select(fnp.minimum(g, 1 - g), gate_floor)
""", """            selected, births = select(_e2_gate_min, gate_floor)   # E2: g is bitwise response_gate (same operations on the same inputs)
"""),
("""            inherited = g[:, None] ** 2 * g[None, :] * cp_pre
            bp = g[:, None] * Cpre
            fnp.fill_diagonal(bp, 0)
            Bpost = C.copy()
""", """            _e2_gc = g[:, None]
            inherited = _e2_gc ** 2 * g * cp_pre
            bp = _e2_gc * Cpre
            fnp.fill_diagonal(bp, 0)
            Bpost = C if _FULL_LOWER_MOMENTS else C.copy()
"""),
("""            trace_post = fnp.diag(Bpost)[:, None] * b[None, :] + 2 * Bpost * b[:, None]
            # Innovation owns only the diagonal of the old residual temporary.
            beta = 2 * (aux['gaussian_mean'] - host * g)
            innovation = (fnp.diag(target) - fnp.diag(inherited) - fnp.diag(trace_post)).copy()
""", """            trace_post = fnp.outer(fnp.diag(Bpost), b) + Bpost * (2 * b)[:, None]
            # Innovation owns only the diagonal of the old residual temporary.
            beta = _e2_beta
            innovation = fnp.diag(target) - fnp.diag(inherited) - fnp.diag(trace_post)
"""),
("""            born = bp * bp * (birth * births)[None, :] + (beta * births)[:, None] * bp.T
            fnp.fill_diagonal(born, birth * births * (3 * u * u + delta * delta) / 4)
""", """            _e2_bb = birth * births
            born = bp * bp * _e2_bb + (beta * births)[:, None] * bp.T
            fnp.fill_diagonal(born, _e2_bb * (3 * u * u + delta * delta) / 4)
"""),
("""            newborn = (int(pp.shape[1]), selected, _bd_base32[:, selected], (t[selected] * alive[selected]).astype(fnp.float32), (s[selected] * alive[selected]).astype(fnp.float32))
            del _bd_base32
            p = g.astype(fnp.float32)[:, None] * pp
            p = alive.astype(fnp.float32)[:, None] * p
            _shared_old384_commit(l, g, alive, selected, shared_old)
            lam = fnp.concatenate([lam, a[selected].astype(fnp.float32), -a[selected].astype(fnp.float32)])
""", """            _e2_alive_sel = alive[selected]
            newborn = (int(pp.shape[1]), selected, _bd_base32[:, selected], (t[selected] * _e2_alive_sel).astype(fnp.float32), (s[selected] * _e2_alive_sel).astype(fnp.float32))
            del _bd_base32
            _e2_ga = (g.astype(fnp.float32) * alive)[:, None]   # alive_i*(g_i*x) == (g_i*alive_i)*x exactly (alive is 0/1, g >= 0)
            p = _e2_ga * pp
            _E2_LAYER.update(ga_gate=g, ga_alive=alive, ga=_e2_ga)
            _shared_old384_commit(l, g, alive, selected, shared_old)
            _e2_asel = a[selected].astype(fnp.float32)
            lam = fnp.concatenate([lam, _e2_asel, -_e2_asel])
"""),
]
