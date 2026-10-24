# Reuse the pair map's already-computed covariance-channel plane bonds.
# They remain in arena [45*n*n, 60*n*n) until this immediate GN consumer.
_GNPR_PARENT_CONTRACT = _e1gn_contract

def _e1gn_contract(g, rho, hold):
    n = int(rho.shape[0])
    T = _E2_PAIR_ARENA[n][1]['T']
    bonds = T[0:4:3, 0, :, :]
    hold(fnp.multiply(g['D'], bonds, out=g['D']))
    hold(fnp.divide(g['D'], g['divisors'], out=g['D']))
    hold(fnp.add(g['D0'], g['D1'], out=g['B0']))
    return g['B0']

