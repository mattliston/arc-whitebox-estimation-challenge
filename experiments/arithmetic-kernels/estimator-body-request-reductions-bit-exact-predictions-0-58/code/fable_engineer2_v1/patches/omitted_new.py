def omitted_radial_source(pp, lam, diagonal, omitted, alive, aux, host):
    """Exact r=0 marginal-pair source sum; higher rho responses are not claimed.
    E2: reuses the live loop's float64 pp, pp*pp, lam and 2*(gaussian_mean - host*gate) when they belong to these objects."""
    L = _E2_LAYER
    if L.get('pp') is pp and L.get('lam') is lam and 'pp64' in L:
        p, p2, lam64 = L.pop('pp64'), L.pop('pp2'), L['lam64']
    else:
        p = pp.astype(fnp.float64)
        p2 = p * p
        lam64 = lam.astype(fnp.float64)
    a = (aux["gate"] - host * aux["pdf"] / aux["sigma"]) * omitted
    if L.get('beta_host') is host and L.get('beta_aux') is aux:
        beta = L['beta0'] * alive
    else:
        beta = 2 * (aux["gaussian_mean"] - host * aux["gate"]) * alive
    row_energy = p2.T @ a
    column_sum = p.T @ beta
    contraction = fnp.sum(lam64 * row_energy * column_sum)
    contraction = contraction - fnp.sum(diagonal * a * beta)
    return 6 * contraction


