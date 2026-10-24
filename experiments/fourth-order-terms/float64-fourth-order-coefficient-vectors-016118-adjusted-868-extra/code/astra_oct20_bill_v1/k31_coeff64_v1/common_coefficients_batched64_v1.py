"""Same direct common coefficients, with small matrix calls to limit requests."""
_O21_K31_COEFF64_CONST = None
def _o21_k31_common64(lam_plus, quartic, square_plus, square_minus, cross, n, hold):
    global _O21_K31_COEFF64_CONST
    if _O21_K31_COEFF64_CONST is None:
        base = [[-1/12,0,0,0],[.5,1/6,0,0],[-.75,-.5,-.25,0],[1/3,1/3,1/3,1/3],
                [1/6,0,0,0],[-.5,0,0,0],[.75,.5,.5,0],[-1/3,-1/3,-1/3,0]]
        means = [[1,0,0],[1,0,-1],[1,1,-2]]
        lower = [[0,-6,0,0,0,-1,0,0,0,0,0,0],[-3,0,0,0,-2,0,0,0,-1,0,0,0],
                 [0,0,0,-1,0,0,0,-2,0,0,0,-3]]
        _O21_K31_COEFF64_CONST = tuple(fnp.asarray(x,dtype=fnp.float64)for x in(base,means,lower))
    basis, meanmap, lowermap = _O21_K31_COEFF64_CONST
    inv = hold(1.0 / hold(lam_plus.astype(fnp.float64)))
    g = hold(hold(lam_plus > 0.0).astype(fnp.float64))
    both = hold(basis @ hold(quartic.astype(fnp.float64)))
    coeff = hold(hold(hold(both[:4] + hold(both[4:] * g)) * inv))
    raw = hold(hold(fnp.stack((square_plus,square_minus,cross))).astype(fnp.float64))
    means = hold(hold(meanmap @ raw) * (1.0 / n))
    products = hold(hold(coeff[:,None,:] * means[None,:,:]).reshape((12,int(lam_plus.shape[0]))))
    lower = hold(lowermap @ products)
    return hold(fnp.unstack(hold(hold(fnp.concatenate((coeff,lower),axis=0)).astype(fnp.float32)),axis=0))
