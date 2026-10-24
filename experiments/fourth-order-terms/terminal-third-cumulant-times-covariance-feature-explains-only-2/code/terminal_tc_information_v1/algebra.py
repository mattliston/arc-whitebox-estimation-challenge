"""Research FP64 selected TC sector. No deployment or full-law claim."""
import numpy as np

class Products:
    def __init__(self):
        self.flops = 0
        self.calls = 0
    def __call__(self, a, b):
        assert a.ndim == b.ndim == 2 and a.shape[1] == b.shape[0]
        self.flops += 2*a.shape[0]*b.shape[1]*(2*a.shape[1]-1)
        self.calls += 1
        return a @ b

def owners(packet, mm):
    P = packet['P'].astype(np.float64)
    lam = packet['lambda'].astype(np.float64)
    V = packet['W14'].astype(np.float64).T.copy()
    A = packet['star'].astype(np.float64)
    ix = packet['previous'].astype(np.int64)
    alive = packet['previous_alive'].astype(bool)
    assert np.array_equal(np.sort(ix), np.flatnonzero(alive))
    Ae = np.zeros_like(A)
    Ae[np.ix_(ix, ix)] = A[np.ix_(ix, ix)]
    cold = np.flatnonzero(~alive)
    Ae[cold, cold] = np.diag(A)[cold]
    U = mm(V, Ae)
    B0 = packet['Bpre'].astype(np.float64)
    C0 = packet['Cpre'].astype(np.float64)
    B, C = (B0+B0.T)/2, (C0+C0.T)/2
    return P, lam, V, U, packet['bpre'].astype(np.float64), B, C

def query_and_plane(P, lam, V, U, b, B, Z, mm):
    M = mm(P.T, Z)
    Q = mm(P, lam[:, None]*(M*M))
    del M
    K = mm((P*P)*lam[None, :], P.T)
    a, c = mm(V.T, Z), mm(U.T, Z)
    Q += (mm(U, a*a)+2*mm(V, c*a))/3
    del a, c
    K += (2*mm(U*V, V.T)+mm(V*V, U.T))/3
    BZ = mm(B, Z)
    Q += b[:, None]*np.sum(Z*BZ, axis=0)[None, :]
    Q += 2*BZ*(b@Z)[None, :]
    K += np.diag(B)[:, None]*b[None, :]+2*B*b[:, None]
    return Q, K

def source(packet):
    mm = Products()
    P, lam, V, U, b, B, C = owners(packet, mm)
    W = packet['W15'].astype(np.float64)
    g = packet['gate14'].astype(np.float64)
    d = packet['pdf14'].astype(np.float64)/packet['sigma14']
    Z = g[:, None]*W
    CZ = mm(C, Z)
    Q, K = query_and_plane(P, lam, V, U, b, B, Z, mm)
    q4 = 12*np.sum((d[:, None]*W)*CZ*Q, axis=0)
    g2 = g*g
    trace = 4*(np.dot(d*np.diag(C)*g, K.T@g2)
               +2*np.dot(d*g, (C*K)@g2))
    post = packet['Cpost_preP'].astype(np.float64)
    denominator = float(np.trace(post)**2+2*np.sum(post*post))
    assert np.isfinite(denominator) and denominator > 0
    v = packet['v15'].astype(np.float64)
    sigma = packet['sigma15'].astype(np.float64)
    alpha = packet['alpha15'].astype(np.float64)
    pdf = packet['pdf15'].astype(np.float64)
    assert (v > 0).all() and (sigma > 0).all()
    rho = 3*trace/denominator
    delta4 = q4-rho*v*v
    feature = delta4*(alpha*alpha-1)*pdf/(24*v*sigma)
    arrays = dict(q4=q4, delta4=delta4, feature=feature)
    assert all(np.isfinite(a).all() for a in arrays.values())
    report = dict(trace=float(trace), radial_coefficient=float(rho),
                  radial_denominator=denominator, matrix_products=mm.calls,
                  fp64_classical_gemm_subtotal=mm.flops,
                  subtotal_is_not_complete_fnp_bill=True,
                  covariance_asymmetry=float(np.linalg.norm(packet['Cpre']-packet['Cpre'].T)),
                  trace_metric_asymmetry=float(np.linalg.norm(packet['Bpre']-packet['Bpre'].T)),
                  feature_rms=float(np.sqrt(np.mean(feature*feature))))
    return arrays, report
