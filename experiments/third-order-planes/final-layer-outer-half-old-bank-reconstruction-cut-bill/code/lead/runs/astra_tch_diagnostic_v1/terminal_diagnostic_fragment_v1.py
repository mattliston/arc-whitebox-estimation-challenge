# Diagnostic only: one TCH trajectory, shadow exact terminal bank, four heads.
# All shadow arithmetic runs inside the same native BudgetContext.
_TD_DATA = {}
_TD_VALUES = {}
_TD_RECON = _slr_reconstruct
_TD_HCF = _hcf_rows15
_TD_STAR = _flms_star_diagonal
_TD_L15 = _l15_rowcut_k3
_TD_SCALAR = _K415_PARENT_SCALAR
_TD_BINARY = _predict_binary


def _td_shared(value):
    _TD_DATA['shared_k3'] = value
    return value


def _slr_reconstruct(U, state):
    answer = _TD_RECON(U, state)
    ctx = _TCH_CTX[0]
    if ctx is not None:
        if _UVP_CUR[0] != 15 or _OTT_CONTEXT[0] is not None:
            raise RuntimeError('terminal diagnostic scope')
        _TD_DATA['rows'] = ctx['rows']
        _TD_DATA['columns'] = int(state['columns'])
        _TD_DATA['exact_old'] = _TCH_PARENT_RECON(U, state)
    return answer


def _hcf_rows15(pp_rows, lam):
    answer = _TD_HCF(pp_rows, lam)
    if _TCH_CTX[0] is not None:
        exact = fnp.concatenate([_TD_DATA.pop('exact_old'),
                                 pp_rows[:, _TD_DATA['columns']:]], axis=1)
        exact_hcf = _TCH_PARENT_HCF(exact, lam)
        _TD_DATA['exact_cp_k3'] = fnp.sum(_cp_diagonal_cube(exact) * lam,
                                         axis=1).astype(fnp.float64)
        _TD_VALUES['hcf_diag_exact'] = exact_hcf['f15_diag']
        _TD_VALUES['hcf_diag_approx'] = answer['f15_diag']
    return answer


def _flms_star_diagonal(w, star):
    answer = _TD_STAR(w, star)
    if _TCH_CTX[0] is not None:
        _TD_DATA['star_k3'] = answer
    return answer


def _l15_rowcut_k3(layer, mu, v, w32, p, previous, newborn, lam, state, star):
    answer = _TD_L15(layer, mu, v, w32, p, previous, newborn, lam, state, star)
    if layer == 15:
        exact = fnp.zeros(int(mu.shape[0]), dtype=fnp.float64)
        fnp.put(exact, _TD_DATA['rows'], _TD_DATA.pop('exact_cp_k3') + _TD_DATA.pop('star_k3'))
        _TD_DATA['exact_host_k3'] = exact
        _TD_VALUES['host_k3_approx'] = answer
        _TD_VALUES['host_k3_exact'] = exact
    return answer


def _td_scalar(mu, v, k3, k4):
    _TD_DATA.update(mu=mu, v=v, effective_k4=k4)
    _TD_VALUES['total_k3_approx'] = k3
    _TD_VALUES['total_k3_exact'] = _TD_DATA['exact_host_k3'] + _TD_DATA['shared_k3']
    return _TD_SCALAR(mu, v, k3, k4)


_K415_PARENT_SCALAR = _td_scalar


def _td_exact_hcf_raw(approximate):
    rows = _TD_DATA['rows']
    mu, v = _TD_DATA['mu'], _TD_DATA['v']
    kept = int(rows.shape[0])
    vr, mr = v[rows], mu[rows]
    X = fnp.stack([vr, fnp.ones(kept, dtype=fnp.float64), mr * mr,
                   _TD_VALUES['hcf_diag_exact']])
    weights = fnp.linalg.solve(X @ X.T, _HCF_CONST['unit4'])
    full = fnp.zeros(int(v.shape[0]), dtype=fnp.float64)
    fnp.put(full, rows, (weights @ X) / vr)
    result = dict(approximate)
    original = approximate['f15_diag']
    result['f15_diag'] = (full, original[1], original[2])
    _TD_VALUES['hcf_residual_exact'] = full
    _TD_VALUES['hcf_residual_approx'] = original[0]
    return result


def _predict_binary(mlp, *args, **kwargs):
    terminal = kwargs['final_hook']

    def diagnostic_terminal(layer, mean_approx, aux_approx, w):
        if layer != 15 or tuple(_HCF_COLUMNS) != ('f15_diag',):
            raise RuntimeError('terminal diagnostic consumer')
        approximate = dict(_HCF_STATE['raw'])
        exact = _td_exact_hcf_raw(approximate)
        ordinary_hc = dict(_HC_STATE)
        ordinary_hc['raw'] = dict(_HC_STATE['raw'])
        mean_exact, aux_exact = _TD_SCALAR(_TD_DATA['mu'], _TD_DATA['v'],
                                          _TD_VALUES['total_k3_exact'],
                                          _TD_DATA['effective_k4'])
        routes = [('exact_skew_exact_hcf', mean_exact, aux_exact, exact),
                  ('approx_skew_exact_hcf', mean_approx, aux_approx, exact),
                  ('exact_skew_approx_hcf', mean_exact, aux_exact, approximate),
                  ('approx_skew_approx_hcf', mean_approx, aux_approx, approximate)]
        live_delta = None
        for name, mean, aux, raw in routes:
            _HC_STATE.clear()
            _HC_STATE.update(ordinary_hc)
            _HC_STATE['raw'] = dict(ordinary_hc['raw'])
            _HCF_STATE['raw'] = dict(raw)
            delta = terminal(layer, mean, aux, w)
            _TD_VALUES[name] = mean + delta
            live_delta = delta
        return live_delta

    kwargs['final_hook'] = diagnostic_terminal
    return _TD_BINARY(mlp, *args, **kwargs)
