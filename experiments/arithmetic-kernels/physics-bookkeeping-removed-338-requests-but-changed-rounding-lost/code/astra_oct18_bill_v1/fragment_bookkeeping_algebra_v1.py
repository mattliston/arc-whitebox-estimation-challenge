# ASTRA bookkeeping v1: algebra
def _astra_fhm_core(mlp, hook=None, cov32=True, metric='covariance', collect_states=False, gate_floor=0.01, final_hook=None, coefficient_hook=None, covariance_hook=None):
    if (not _ASTRA_DENSE_ON or not cov32 or not _FULL_LOWER_MOMENTS
            or _T31_LAYERS or _t31_commit is not None or _BD_J1_PHYS is None):
        return _ASTRA_DENSE_CORE_PARENT(mlp, hook=hook, cov32=cov32, metric=metric,
            collect_states=collect_states, gate_floor=gate_floor, final_hook=final_hook,
            coefficient_hook=coefficient_hook, covariance_hook=covariance_hook)
    for _astra_name, _astra_function in _ASTRA_DENSE_CONTEXT:
        if globals().get(_astra_name) is not _astra_function:
            raise RuntimeError('Dense bookkeeping source context changed: ' + _astra_name)
    _ASTRA_DENSE_STATS.update(cc_vv_layers=0, l0_calls=0)
    _R320_RESP.clear()   # PT-60 R320 router: per-MLP runtime features
    global _W5_PORT_SCOPE
    _e2_H = _E2Hold(); _e2_append_inline = _e2_H.append; _e2_value_inline = None   # E2: temporaries of this call, released together (one hidden free trip)
    if _W5_PORT_SCOPE is not None:
        raise RuntimeError('W5 no concurrent or nested network prediction')
    _w5_context = _w5_port_new(mlp)
    _W5_PORT_SCOPE = _w5_context
    try:
        global _SCRATCH
        depth, n = (int(mlp.depth), int(mlp.width))
        assert (depth, n) == (16, 1024) and metric == 'covariance' and _FULL_LOWER_MOMENTS and hook is not None
        W = mlp.weights
        _bd_c = _bd_coefficients()
        if _SCRATCH is None:
            arena = fnp.empty((_PACKED_ELEMENTS,), dtype=fnp.float32)
            _SCRATCH = [arena, (_e2_append_inline((_e2_value_inline := _cp3_bind_arena_views(arena, _PACKED_PUSH))) or _e2_value_inline), (_e2_append_inline((_e2_value_inline := _cp3_bind_arena_views(arena, _PACKED_READ))) or _e2_value_inline), None, None, None, None]
            _bind_push6((_e2_append_inline((_e2_value_inline := _SCRATCH[0])) or _e2_value_inline))
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
        _t31_carry = {}
        _PHYS_HANDLES.clear()
        for l in range(depth):
            _e2_H = _E2Hold(); _e2_append_inline = _e2_H.append; _e2_value_inline = None   # E2: this iteration's temporaries, released together at the next iteration
            w32 = fnp.asarray((_e2_append_inline((_e2_value_inline := W[l])) or _e2_value_inline), dtype=fnp.float32)
            _w5_port_begin(_w5_context,mlp,l,w32,previous)
            w = fnp.asarray((_e2_append_inline((_e2_value_inline := W[l])) or _e2_value_inline), dtype=fnp.float64)
            wT = w.T
            mu = wT @ m
            if l == depth - 1 and final_hook is not None:
                v, active_v = (_dc_cov_diagonal(w32, C, response_gate, gate_floor) if _FULL_LOWER_MOMENTS   # opus-engineer dcv: the same cut
                                else _cov_compact_diagonal(w32, C, previous, previous_alive))
                bpre = wT @ b
                # fable-physicist LG-4: L15 k3 row cut (CP + star parts only on rows with |mu/sigma| <= _L15_ROWCUT_Z).
                k3 = _l15_rowcut_k3(l, mu, v, w32, p, previous, newborn, lam, shared_old, star)
                k3 = k3 + (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (3 * active_v))) or _e2_value_inline) * bpre))) or _e2_value_inline)
                m, aux = nonlinear_scalar(mu, v, k3, (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (radial * v))) or _e2_value_inline) * v))) or _e2_value_inline) + (_e2_append_inline((_e2_value_inline := _k4l_final(mu, v))) or _e2_value_inline))   # opus-engineer K4L: the final layer's per-unit fourth cumulant
                m = m + (_e2_append_inline((_e2_value_inline := final_hook(l, m, aux, _e2_append_inline((_e2_value_inline := W[l])) or _e2_value_inline))) or _e2_value_inline)
                means.append(m)
                break
            _bb_star_pre = None   # E-35: set by the batched covariance/star call
            _plane_cov32 = _plane_star32 = None
            if l == 0:
                Cpre = (_e2_append_inline((_e2_value_inline := _resident_gram512(_e2_append_inline((_e2_value_inline := w32.T)) or _e2_value_inline, w32))) or _e2_value_inline).astype(fnp.float64)
                Bpre = Cpre
            elif cov32:
                if _FULL_LOWER_MOMENTS:
                    if 1 <= l <= 14:
                        Cpre, _bb_star_pre, _plane_cov32, _plane_star32 = _dc_cov_star(w32, C, star, response_gate, gate_floor, retain32=True)   # opus-engineer dcv: the units of the layer below under the gate floor cut
                    else:
                        Cpre, _bb_star_pre = _flms_cov_star(w32, C, star)
                    Bpre = Cpre
                else:
                    Cpre, Bpre = _cov_compact(w32, C, previous, previous_alive)
            else:
                Cpre = (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := w.T)) or _e2_value_inline) @ C))) or _e2_value_inline) @ w
                Bpre = Cpre if _FULL_LOWER_MOMENTS else (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := w.T)) or _e2_value_inline) @ (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (C * (_e2_append_inline((_e2_value_inline := previous_alive[:, None])) or _e2_value_inline)))) or _e2_value_inline) * (_e2_append_inline((_e2_value_inline := previous_alive[None, :])) or _e2_value_inline)))) or _e2_value_inline)))) or _e2_value_inline) @ w
            bpre = wT @ b
            vv = fnp.diag(Cpre)
            _e2_vc = fnp.maximum(vv, 1e-10)
            _e2_s = fnp.sqrt(_e2_vc)
            _e2_a = mu / _e2_s
            response_gate = flops.stats.norm.cdf(_e2_a)
            _E2_LAYER.clear()
            _E2_LAYER.update(mu=mu, Cpre=Cpre, vc=_e2_vc, s=_e2_s, a=_e2_a, g=response_gate, rqs2_vv=vv)   # RQS2 E12
            if _plane_cov32 is not None:
                _E2_LAYER['ps_Cpre32'] = _plane_cov32
            _E2_LAYER['psp_blocks'] = (128 if l > 0 and cov32 and _FULL_LOWER_MOMENTS and n == 1024 and _AB_ON and _WINOGRAD and _psdb_fits() and all(_ipt_fits(6, pm, pk, pn) for pm, pk, pn in _PSD_SHAPES) else 0)
            live_indices, alive = select(response_gate, gate_floor)
            live = fnp.outer(alive, alive)
            _e2_gate_min = fnp.minimum(response_gate, (_e2_append_inline((_e2_value_inline := (1 - response_gate))) or _e2_value_inline))
            response_score = fnp.where(alive, _e2_gate_min, (_e2_append_inline((_e2_value_inline := (-1.0))) or _e2_value_inline))
            response_indices, response_mask = select(response_score, gate_floor)
            _R320_RESP[l] = int(response_indices.shape[0])   # PT-60: resp_L, before this layer's push/join
            _E2_LAYER.update(ecut_rows=response_indices, ecut_cols=live_indices)   # opus-engineer Ecut: the rosters the covariance edge keeps
            omitted = fnp.greater(alive, response_mask)   # E2: == alive & ~response_mask for booleans
            _E2_LAYER['ott_live_mask'] = alive
            pp = _deadcut_push(l, w32, p, previous, newborn, lam, shared_old, _e2_a)
            del p   # E2-M: dead until the end-of-layer reassignment (it lived through the live peak)
            cp_pre = fnp.zeros((n, n), dtype=fnp.float32)
            if len(lam):
                _e2_pp2 = pp * pp   # F32 (E-29): third-order sums in float32 (pp, lam float32); the reduced vector widens to float64. E-48 --colpack E3: formed before the read, which takes its squared response rows from it
                left2 = _np_left(l, _e2_pp2, pp, response_indices, response_mask, alive, lam, mu, _e2_vc, vv, Cpre)   # opus-engineer NP: the response rows of the squared columns, with the pair law of the norm coupling folded in
                right = _dcr_right(pp, int(shared_old['columns']), live_indices)   # E2-R: only the young columns of right are ever read (opus-engineer dcr: a slice of the pushed rows when the rosters agree)
                response_plane = (_e2_append_inline((_e2_value_inline := _cp_read_young_sq(left2, right, lam, live_indices, shared_old))) or _e2_value_inline)
                if (response_plane.dtype) != 'float32':
                    raise TypeError('Exact CP plane retention requires float32 response')
                del left2   # E-48 --colpack E4: right stays until the end-of-layer gating, which forms the compact p from it
                fnp.put(cp_pre, (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (_e2_append_inline((_e2_value_inline := (response_indices * n))) or _e2_value_inline)[:, None])) or _e2_value_inline) + live_indices))) or _e2_value_inline), response_plane)
                _np_forms(l, cp_pre, _plane_cov32)   # opus-engineer NP: the forms the fold must not contain, back on the read's block
                _E2_LAYER.update(pp=pp, lam=lam, pp64=pp, pp2=_e2_pp2, lam64=lam)
                _e2_t3 = _e2_pp2 * pp
                _k4l_loc = _k4l_local(l, pp, _e2_pp2, _e2_t3, shared_old)   # opus-engineer K4L: the young pairs' hub terms of the per-unit fourth cumulant
                _b6_old(l, pp, _e2_pp2, lam, shared_old, _e2_vc, response_mask, int(response_indices.shape[0]))   # opus-engineer b6: the old cohorts' energy and local quartic
                _e2_t3h = fnp.multiply(_e2_t3, lam, out=_e2_t3)   # E2-M: in place (same products and order): one (n,K) transient fewer at the live peak
                diagonal = (_e2_append_inline((_e2_value_inline := fnp.sum(_e2_t3, axis=1))) or _e2_value_inline).astype(fnp.float64)
                diagonal = _ott_diagonal(diagonal, lam)
                del _e2_pp2, _e2_t3, _e2_t3h
                fnp.fill_diagonal(cp_pre, diagonal)
                _plane_cpdiag64 = diagonal
            else:
                _plane_cpdiag64 = fnp.zeros(n, dtype=fnp.float64)
            star_pre = (fnp.zeros((n, n), dtype=fnp.float64) if l == 0 else
                        ((_bb_star_pre if _bb_star_pre is not None else _flms_star(w32, star)) if _FULL_LOWER_MOMENTS else
                         _star_compact(w32, star, previous, previous_alive, live_indices)))
            _w5_port_end_layer(_w5_context)
            _ps_bv = vv if Bpre is Cpre else fnp.diag(Bpre)
            _ps_kdiag = (_plane_cpdiag64 + fnp.diag(star_pre)) + _ps_bv * (3 * bpre)   # RQS2 E1: _ps_bv is diag(Bpre)
            _ps_cp32 = _ps32(cp_pre); _ps_bpre32 = _ps32(bpre)
            _ps_K2132 = (_ps_cp32 + (_plane_star32 if _plane_star32 is not None else _ps32(star_pre)) + fnp.outer(_ps32(_ps_bv), _ps_bpre32)) + fnp.einsum('ij,i->ij', _ps_pre32(Bpre), (2 * _ps_bpre32), optimize=False)
            K21 = _ps_K2132.astype(fnp.float64)
            fnp.fill_diagonal(K21, _ps_kdiag)
            fnp.fill_diagonal(_ps_K2132, _ps_kdiag)   # RQS2 E7: the float32 target casts as _ps32 would
            _E2_LAYER.update(ps_K21=K21, ps_K2132=_ps_K2132, rqs2_kdiag=_ps_kdiag)   # RQS2 E13
            _b6_qb(l, _ps_cp32, _plane_cpdiag64, mu, response_gate, _e2_s, _e2_vc, response_mask, int(response_indices.shape[0]))   # opus-engineer b6: the read's plane against the scale mode's column
            k4 = (_e2_append_inline((_e2_value_inline := (radial * vv))) or _e2_value_inline) * vv
            _s19_k4u = k4
            if _S19_K4BAND is not None and l >= 1:   # PT-33 S19: per-unit k4 x band L1-7/8-11/12-15 (covhunter; torch pair_in, k4_15)
                k4 = k4 * (_S19_K4BAND[0] if l <= 7 else (_S19_K4BAND[1] if l <= 11 else _S19_K4BAND[2]))
            if l >= 1 and len(lam):   # opus-engineer K4L: the non-isotropic part of the per-unit fourth cumulant, at the level the card was trained with
                _hc_k4l = (_e2_append_inline((_e2_value_inline := _k4l_dk(_k4l_loc, _e2_a, vv))) or _e2_value_inline)
                k4 = (_e2_append_inline((_e2_value_inline := k4)) or _e2_value_inline) + _hc_k4l
                _hc_put(l, 'k4_local_rows', _hc_k4l / (_e2_vc * _e2_vc), response_mask, int(response_indices.shape[0]))   # opus-engineer hc: K4L's per-unit value over v^2, on the response rows
            # Cpre and vv are exact widenings of the owned float32 covariance.
            # v1 reassociates only exact-real diagonal identities; native endpoint gate required.
            _astra_q64 = vv * vv
            _ps_k22diag = radial * _astra_q64
            _astra_v32 = _ps32(vv)
            _ps_vv32 = fnp.outer(_astra_v32, _astra_v32)
            _astra_c32 = _ps_pre32(Cpre)
            _astra_cc32 = _idc_get('cc32', _astra_c32, 1, lambda: _astra_c32 * _astra_c32)
            _E2_LAYER.update(VV_for=Cpre, ps_vv32=_ps_vv32)
            k22 = _ps32(radial) * (_ps_vv32 + 2 * _astra_cc32) / 3
            fnp.fill_diagonal(k22, _ps_k22diag)   # RQS2 E8
            _ASTRA_DENSE_STATS['cc_vv_layers'] += 1
            _b6_m2(l, _e2_a, _e2_vc, _e2_s, _ps_kdiag, response_mask, int(response_indices.shape[0]))   # opus-engineer b6: the second-order mean term's column
            pair_map = _nonlinear_first_gaussian if l == 0 else _nonlinear_terminal_live if l == (_e2_append_inline((_e2_value_inline := (depth - 1))) or _e2_value_inline) else nonlinear
            _PHYS_LAYER[0] = l
            if l in _T31_LAYERS:
                m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric, extra31=(_e2_append_inline((_e2_value_inline := _t31_extra(l, _e2_append_inline((_e2_value_inline := locals())) or _e2_value_inline))) or _e2_value_inline))
            else:
                m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric)
            if l in _GN_COV_LAYERS:
                C = C + (_e2_append_inline((_e2_value_inline := _gn_dcov(l, mu, Cpre, K21, _s19_k4u if _S19_GN_TORCH else k4, aux))) or _e2_value_inline)
            host = m
            _e2_beta = 2 * (_e2_append_inline((_e2_value_inline := (aux['gaussian_mean'] - (_e2_append_inline((_e2_value_inline := (host * aux['gate']))) or _e2_value_inline)))) or _e2_value_inline)
            _E2_LAYER.update(beta_host=host, beta_aux=aux, beta0=_e2_beta)
            if len(lam) and l < depth - 1:
                source_sum = omitted_radial_source(pp, lam, diagonal, omitted, alive, aux, host)
                _e2_pair = _E2_LAYER.get('pair')
                denominator = _e2_pair['den'] if metric == 'covariance' and _e2_pair is not None and ((_e2_append_inline((_e2_value_inline := _e2_pair['cov'])) or _e2_value_inline) is C) else (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := fnp.trace(C))) or _e2_value_inline) ** 2))) or _e2_value_inline) + (_e2_append_inline((_e2_value_inline := (2 * (_e2_append_inline((_e2_value_inline := fnp.sum(_e2_append_inline((_e2_value_inline := (C * C))) or _e2_value_inline))) or _e2_value_inline)))) or _e2_value_inline) if metric == 'covariance' else n * (_e2_append_inline((_e2_value_inline := (n + 2))) or _e2_value_inline)
                radial_new = radial_new + (_e2_append_inline((_e2_value_inline := (source_sum / denominator))) or _e2_value_inline)
            if _EDGE_ENABLED and 8 <= l <= 14:
                if covariance_hook is None:
                    C = C + (_e2_append_inline((_e2_value_inline := _edge_source(l, m, K21, aux))) or _e2_value_inline)
                else:
                    C = C + (_e2_append_inline((_e2_value_inline := covariance_hook(l, m, K21, aux, _e2_append_inline((_e2_value_inline := W[l])) or _e2_value_inline))) or _e2_value_inline)
            if hook is not None:
                dm = hook(l, m, C, K21, aux, (_e2_append_inline((_e2_value_inline := W[l])) or _e2_value_inline))
                if (dm.dtype) != 'float64':
                    dm = fnp.asarray(dm, fnp.float64)
                if 1 <= l <= 14:
                    dm = _astra_fhm_effective(l, m, dm)
                # Real-arithmetic identity for the same rank-two mean correction.
                # Own all new vectors/matrices. Original C and pair post_v are read-only.
                _rqs2_pair = _E2_LAYER.get('pair')
                _ps_Cdiag0 = (_rqs2_pair['aux']['post_v'] if (_rqs2_pair is not None and _rqs2_pair['cov'] is C) else fnp.diag(C))
                _gn_rc = _GN_RC.get(l, 0.0) - _GN_KC.get(l, 0.0)
                _bk_z64 = m + 0.5 * dm
                _bk_d64 = (1.0 + _gn_rc) * dm if _gn_rc else dm
                _ps_Cdiag = _ps_Cdiag0 - (2 * _bk_z64) * _bk_d64
                _bk_z32 = _ps32(_bk_z64)
                _bk_d32 = _ps32(_bk_d64)
                _e2_mdm = fnp.outer(_bk_z32, _bk_d32)
                _ps_C0 = _ps32(C)
                C = (_ps_C0 - _e2_mdm) - _e2_mdm.T
                _ps_post_cov32 = C
                C = C.astype(fnp.float64)
                fnp.fill_diagonal(C, _ps_Cdiag)
                if _FULL_LOWER_MOMENTS:
                    fnp.fill_diagonal(_ps_post_cov32, _ps_Cdiag)
                    _E2_LAYER["post_cov32"] = (C, _ps_post_cov32)
                m = m + dm
            means.append(m)
            if l == depth - 1:
                break
            g = aux['gate']
            _e2_pair = _E2_LAYER.get('pair')
            d = _e2_pair['d'] if _e2_pair is not None and (_e2_append_inline((_e2_value_inline := _e2_pair['aux'])) or _e2_value_inline) is aux else aux['pdf'] / aux['sigma']
            gq = _gq(g, d, aux, bpre)   # opus-engineer gsk: the gate that carries the third-order content, with the mean gate's first-order corrections
            birth = d * (_e2_append_inline((_e2_value_inline := (1 + (_e2_append_inline((_e2_value_inline := (radial / 3))) or _e2_value_inline)))) or _e2_value_inline)
            birth = _bsk(birth, aux)   # opus-engineer bsk: the kink density to first order in the unit's skewness
            b = (_e2_append_inline((_e2_value_inline := (gq * bpre))) or _e2_value_inline) + (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (radial * d))) or _e2_value_inline) * vv))) or _e2_value_inline) / _S19_BSRC_DIV))) or _e2_value_inline) if _FULL_LOWER_MOMENTS else (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (gq * bpre))) or _e2_value_inline) + (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (radial * d))) or _e2_value_inline) * vv))) or _e2_value_inline) / _S19_BSRC_DIV))) or _e2_value_inline)))) or _e2_value_inline) * alive
            if _BFIX_GAMMA:
                b = b + (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (_BFIX_GAMMA * (_e2_append_inline((_e2_value_inline := (radial_new - radial))) or _e2_value_inline)))) or _e2_value_inline) / 6))) or _e2_value_inline) * m))) or _e2_value_inline)
            if _S19_BSCALE is not None:   # PT-33 S19: carried b x bscale after the b-fix, before trace_post (covhunter; torch b_post)
                b = b * _S19_BSCALE
            selected, births = select(_e2_gate_min, gate_floor)   # E2: g is bitwise response_gate (same operations on the same inputs)
            if not _FULL_LOWER_MOMENTS:
                cv = (_e2_append_inline((_e2_value_inline := fnp.diag(C))) or _e2_value_inline).copy()
                C = C * live
                fnp.fill_diagonal(C, cv)
                td = (_e2_append_inline((_e2_value_inline := fnp.diag(target))) or _e2_value_inline).copy()
                target = target * live
                fnp.fill_diagonal(target, td)
            _ps_g2 = gq ** 2
            _ps_inherited_diag = fnp.einsum('i,i,i->i', _ps_g2, gq, _plane_cpdiag64, optimize=False)
            inherited = fnp.einsum('i,j,ij->ij', _ps32(_ps_g2), _ps32(gq), _ps_cp32, optimize=False)
            bp = fnp.einsum('i,ij->ij', g, Cpre, optimize=False)
            fnp.fill_diagonal(bp, 0)
            Bpost = C if _FULL_LOWER_MOMENTS else C.copy()
            if not _FULL_LOWER_MOMENTS:
                fnp.fill_diagonal(Bpost, (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := fnp.diag(C))) or _e2_value_inline) * alive))) or _e2_value_inline))
            _rqs2_pc = _E2_LAYER.get("post_cov32"); _ps_Bdiag = _ps_Cdiag if (_rqs2_pc is not None and _rqs2_pc[0] is Bpost) else fnp.diag(Bpost)   # RQS2 E3
            _ps_trace_diag = _ps_Bdiag * (3 * b)
            _f4 = _f4c_step(l, right if len(lam) else None, gq, d, live_indices, _ps_Bdiag, lam, shared_old)   # opus-engineer F4c: the shift of the young pairs' columns (rule + exact terms) and its star anchor
            _f4_b = b if _f4 is None else b + _f4[1]   # the star's trace term takes b + beta; the carried b and the births keep b
            trace_post = fnp.outer(_ps32(_ps_Bdiag), _ps32(_f4_b)) + fnp.einsum('ij,i->ij', _ps32(Bpost), _ps32(2 * _f4_b), optimize=False)
            # Innovation owns only the diagonal of the old residual temporary.
            beta = _e2_beta
            _rqs2_tdiag = fnp.diag(target); innovation = (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := _rqs2_tdiag)) or _e2_value_inline) - _ps_inherited_diag))) or _e2_value_inline) - _ps_trace_diag   # RQS2 E4 (reused at E5)
            safe_birth = fnp.maximum(birth, 1e-12)
            u = fnp.clip((_e2_append_inline((_e2_value_inline := (beta / safe_birth))) or _e2_value_inline), (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (-8))) or _e2_value_inline) * vv))) or _e2_value_inline), (_e2_append_inline((_e2_value_inline := (8 * vv))) or _e2_value_inline))
            _bk_3u2 = (3 * u) * u
            disc = (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (4 * innovation))) or _e2_value_inline) / safe_birth))) or _e2_value_inline) - _bk_3u2
            delta = fnp.minimum((_e2_append_inline((_e2_value_inline := fnp.sqrt(_e2_append_inline((_e2_value_inline := fnp.maximum(disc, _e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (0.01 * vv))) or _e2_value_inline) * vv))) or _e2_value_inline))) or _e2_value_inline))) or _e2_value_inline), (_e2_append_inline((_e2_value_inline := (16 * vv))) or _e2_value_inline))
            t = (_e2_append_inline((_e2_value_inline := (u + delta))) or _e2_value_inline) / 2
            s = (_e2_append_inline((_e2_value_inline := (u - delta))) or _e2_value_inline) / 2
            a = birth / delta
            # Newest numerical tails are owned by the shared-newborn tuple.
            _e2_bb = birth * births
            _ps_bp32 = _ps32(bp)
            born = fnp.einsum('ij,ij,j->ij', _ps_bp32, _ps_bp32, _ps32(_e2_bb), optimize=False) + fnp.einsum('i,ji->ij', _ps32(beta * births), _ps_bp32, optimize=False)
            _ps_born_diag = (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (_e2_bb * (_e2_append_inline((_e2_value_inline := (_bk_3u2 + (_e2_append_inline((_e2_value_inline := (delta * delta))) or _e2_value_inline)))) or _e2_value_inline)))) or _e2_value_inline) / 4))) or _e2_value_inline)
            _bd_current_c = (_bd_c[l] if l == 0 or coefficient_hook is None
                             else coefficient_hook(l))
            _bd_base32, _bd_plane_delta = _bd_update(
                _ps_bp32, Cpre, K21, aux, d, alive, births, a, t, s, _bd_current_c)   # opus-engineer bd32: the float32 copy made three lines up
            del _bd_current_c
            born = fnp.add(born, _bd_plane_delta, out=born)   # opus-engineer bd32: both float32 (was widened to float64 and stored in float32)
            _ps_born_diag = _ps_born_diag + fnp.diag(_bd_plane_delta)
            fnp.fill_diagonal(born, _ps_born_diag)
            del _bd_plane_delta
            _ps_cpdiag = (_ps_inherited_diag + _ps_born_diag) * alive   # RQS2 E6: diag(outer(alive, alive)) is alive
            cp_post = (inherited + _ps32(born)) * live
            if (_E2_LAYER.get('ats_a') is a and _E2_LAYER.get('ats_t') is t and _E2_LAYER.get('ats_s') is s
                    and _E2_LAYER.get('ats_alive') is alive):
                _e2_asel, _e2_tsel, _e2_ssel = fnp.unstack((_e2_append_inline((_e2_value_inline := _E2_LAYER['ats32'][:, selected])) or _e2_value_inline))
            else:
                _e2_alive_sel = alive[selected]
                _e2_asel = (_e2_append_inline((_e2_value_inline := a[selected])) or _e2_value_inline).astype(fnp.float32)
                _e2_tsel = (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := t[selected])) or _e2_value_inline) * _e2_alive_sel))) or _e2_value_inline).astype(fnp.float32)
                _e2_ssel = (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := s[selected])) or _e2_value_inline) * _e2_alive_sel))) or _e2_value_inline).astype(fnp.float32)
            newborn = (int((_e2_append_inline((_e2_value_inline := pp.shape[1])) or _e2_value_inline)), selected, (_e2_append_inline((_e2_value_inline := _bd_base32[:, selected])) or _e2_value_inline), _e2_tsel, _e2_ssel)
            _k4l_birth(l, aux, host, selected, t, s)   # opus-engineer K4L: the newborn pairs' four hub coefficients
            _f4c_birth(l, aux, d, host, selected, t, s, newborn[2], _e2_asel, radial)   # opus-engineer F4c: the newborn pairs' six shift coefficients per age
            del _bd_base32
            _e2_ga = (_e2_append_inline((_e2_value_inline := _ps32(gq))) or _e2_value_inline) * alive
            p = fnp.einsum('i,ij->ij', _e2_ga[live_indices], right if len(lam) else pp[live_indices, int(shared_old['columns']):], optimize=False)   # E-48 --colpack E4: the gated young columns on the rows the next push reads
            if _f4 is not None:   # opus-engineer F4c: shift the young gated columns in place
                _f4_py = p[:, int(p.shape[1]) - int(_f4[0].shape[1]):]
                (_e2_append_inline((_e2_value_inline := fnp.add(_f4_py, _f4[0], out=_f4_py))) or _e2_value_inline)
            if len(lam):
                del right
            _E2_LAYER.update(ga_gate=gq, ga_alive=alive, ga=_e2_ga)
            _shared_old384_commit(l, gq, alive, selected, shared_old)
            _np_update(l, lam, pp, mu, vv, m, d, aux, t, s, delta, safe_birth, _e2_vc, response_mask, selected, radial, _ps_K2132, _ps_kdiag)   # opus-engineer NP: the second weight per bank column and the cross weight per pair
            if _S19_LAM is None:
                lam = fnp.concatenate([lam, _e2_asel, -_e2_asel])
            else:   # PT-33 S19: carried CP weights x s_carry, newborn x s_new (covhunter; torch state_out)
                _s19_new = _e2_asel * _S19_LAM[1]
                lam = fnp.concatenate([lam * _S19_LAM[0], _s19_new, -_s19_new])
            if _t31_commit is not None:
                _t31_commit(l, (_e2_append_inline((_e2_value_inline := locals())) or _e2_value_inline))
            _ps_rest_diag = (_rqs2_tdiag - _ps_cpdiag) - _ps_trace_diag   # RQS2 E5
            if _f4 is not None:   # opus-engineer F4c: the exact diagonal of the plane the shift adds
                _ps_rest_diag = _ps_rest_diag - _f4[2]
            _ps_star_diag = _ps_rest_diag.astype(fnp.float32)
            rest = (_ps_target32(target) - cp_post) - trace_post
            star = 3 * rest.T
            fnp.fill_diagonal(star, _ps_star_diag)
            if collect_states:
                states.append(dict(layer=l, active=int((_e2_append_inline((_e2_value_inline := alive.sum())) or _e2_value_inline)), births=len(selected), columns=len(lam)))
            previous, previous_alive = (live_indices, alive)
            radial = radial_new
        return fnp.asarray((_e2_append_inline((_e2_value_inline := fnp.maximum(_e2_append_inline((_e2_value_inline := fnp.stack(means))) or _e2_value_inline, 0.0))) or _e2_value_inline), dtype=fnp.float32)
    except BaseException:
        _w5_context['failed'] = True
        raise
    finally:
        try:
            _w5_port_finish(_w5_context)
        finally:
            _E2_LAYER.pop("post_cov32", None)
            _W5_PORT_SCOPE = None

_ASTRA_FHM_ACTIVE_CORE = _tr_clearing(_astra_fhm_core)
