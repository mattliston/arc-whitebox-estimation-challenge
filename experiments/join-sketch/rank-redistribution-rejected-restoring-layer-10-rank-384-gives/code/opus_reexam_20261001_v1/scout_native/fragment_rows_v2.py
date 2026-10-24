

# BEGIN SCOUT-NATIVE ROWS v2 (2026-10-12; = v1's code, header only: v1 called it rounding-level): omitted_radial_source (the active
# E2 version, body copied) forms row_energy = p2' a only over the rows where a is non-zero (the omitted rows: alive and not response,
# about 100-250 of 1024), gathered by XSG's scatter in the twin (_xsg_rows, site 8624) or a plain row gather in the safe: the census put
# the full matvec at 0.150% of the bill. The zero rows only added exact zeros; the shorter float32 matvec may group its sums differently,
# which reaches only the radial source's scalar contraction. Predictions measured byte-identical: O/scout_native/run_rows_certs_v1.out.
_ROWS_STATS = dict(calls=0, rows=0)


def omitted_radial_source(pp, lam, diagonal, omitted, alive, aux, host):
    """Exact r=0 marginal-pair source sum; higher rho responses are not claimed.
    E2: reuses the live loop's float64 pp, pp*pp, lam and 2*(gaussian_mean - host*gate) when they belong to these objects."""
    _e2_H = _E2Hold()   # E2: temporaries of this call, released together (one hidden free trip)
    L = _E2_LAYER
    if L.get('pp') is pp and L.get('lam') is lam and 'pp64' in L:
        p, p2, lam64 = L.pop('pp64'), L.pop('pp2'), L['lam64']   # F32 (E-29): float32 plane, its square and lam
    else:
        p = pp
        p2 = p * p
        lam64 = lam
    a = _e2_H(aux['gate'] - _e2_H(_e2_H(host * aux['pdf']) / aux['sigma'])) * omitted
    if L.get('beta_host') is host and L.get('beta_aux') is aux:
        beta = L['beta0'] * alive
    else:
        beta = _e2_H(2 * _e2_H(aux['gaussian_mean'] - _e2_H(host * aux['gate']))) * alive
    _rw_idx = _e2_H(fnp.nonzero(omitted)[0])   # scout-native ROWS: a is zero outside the omitted rows
    if int(_rw_idx.shape[0]) == 0:
        row_energy = fnp.zeros(int(p2.shape[1]), dtype=fnp.float64)
    else:
        _rw_f = globals().get('_xsg_rows')
        _rw_p2 = _e2_H(_rw_f(8624, p2, _rw_idx) if _rw_f is not None else p2[_rw_idx, :])
        row_energy = _e2_H(_e2_H(_rw_p2.T) @ _e2_H(_e2_H(a[_rw_idx]).astype(fnp.float32))).astype(fnp.float64)
    _ROWS_STATS['calls'] += 1; _ROWS_STATS['rows'] += int(_rw_idx.shape[0])
    column_sum = _e2_H(_e2_H(p.T) @ _e2_H(beta.astype(fnp.float32))).astype(fnp.float64)
    if _OTT_DATA:
        tail = _OTT_DATA['full_tail']
        a_tail = a[tail]
        beta_tail = beta[tail]
        energies, sums = [], []
        for Ut, F, norm2, local_tail in _OTT_DATA['blocks']:
            row_variance = fnp.sum(Ut * Ut, axis=1) / int(Ut.shape[1])
            energy_scale = fnp.sum(a_tail * row_variance)
            energies.append(norm2 * energy_scale)
            sums.append((Ut.T @ beta_tail.astype(fnp.float32)) @ F)   # opus-engineer a08: Ut and F are float32
        delta_energy = fnp.concatenate(energies)
        delta_sum = fnp.concatenate(sums)
        columns = int(delta_energy.shape[0])
        # Only the old columns changed. The caller supplied the repaired
        # diagonal, so its subtraction already includes the tail estimate.
        _ott_h0 = fnp.add(row_energy[:columns], delta_energy, out=row_energy[:columns])
        _ott_h1 = fnp.add(column_sum[:columns], delta_sum, out=column_sum[:columns])
    L['np_column_sum'] = column_sum   # opus-engineer NP: pp' beta (tail rows of the old columns included), reused by the carried weights
    contraction = fnp.sum(_e2_H(_g5_chain(_e2_H(lam64.astype(fnp.float64)), row_energy, column_sum)))
    contraction = contraction - _e2_H(fnp.sum(_e2_H(_g5_chain(diagonal, a, beta))))
    return 6 * contraction
# END SCOUT-NATIVE ROWS v2
