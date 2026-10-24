# Reuse the pair map's covariance-channel plane bonds in the immediate GN call.
_GNPR_PARENT_STATE = _e1gn_state

def _e1gn_state(st, n):
    g = _GNPR_PARENT_STATE(st, n)
    if 'pair_bonds' not in g:
        g['pair_bonds'] = _E2_PAIR_ARENA[n][1]['T'][0:4:3, 0, :, :]
    return g

def _e1gn_contract(g, rho, hold):
    hold(fnp.multiply(g['D'], g['pair_bonds'], out=g['D']))
    hold(fnp.divide(g['D'], g['divisors'], out=g['D']))
    hold(fnp.add(g['D0'], g['D1'], out=g['B0']))
    return g['B0']

