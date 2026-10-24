"""Small cross-block covariance-metric fourth core from R@U and trace(R).

Exact mode is for the small certificate ONLY. Polynomial mode uses thin
matrix products; neither mode is a complete estimator or cost certificate.
"""
import numpy as np


def project(C,U,RU,trace_R,mode='polynomial',degree=32):
    n,q = U.shape
    assert q < n and np.linalg.norm(U.T@U-np.eye(q)) < 1e-10
    h,rotation = np.linalg.eigh(U.T@C@U)
    U,RU = U@rotation,RU@rotation
    t = np.trace(C)
    CU = C@U
    F = CU-U*h[None,:]
    Rhigh = U.T@RU
    Rhigh = (Rhigh+Rhigh.T)/2
    Rlow = RU-U@Rhigh
    Dnorm = np.sqrt(max(0.,np.sum(C*C)-2*np.sum(CU*CU)+np.sum(h*h)))
    def Q(X):
        return X-U@(U.T@X)
    def D(X):
        return Q(C@Q(X))
    rhs = np.concatenate([np.column_stack((F,6*Rlow[:,j])) for j in range(q)],axis=1)
    shifts = np.repeat(t+2*h,q+1)
    info = dict(mode=mode,degree=degree,residual_covariance_frobenius=float(Dnorm),
                minimum_shift=float(shifts.min()),inverse_interval_certified=bool(shifts.min()>2*Dnorm))
    if mode == 'exact':
        dense_D = D(np.eye(n))
        inverse = np.concatenate([np.linalg.solve((t+2*h[j])*np.eye(n)+2*dense_D,
                                                  rhs[:,j*(q+1):(j+1)*(q+1)]) for j in range(q)],axis=1)
    else:
        assert mode == 'polynomial' and info['inverse_interval_certified'],info
        if Dnorm < 1e-20:
            inverse = rhs/shifts[None,:]
            info['max_chebyshev_ratio'] = 0.
        else:
            delta = np.sqrt(shifts*shifts-4*Dnorm*Dnorm)
            ratio = -2*Dnorm/(shifts+delta)
            previous = rhs
            current = D(rhs)/Dnorm
            inverse = rhs/delta[None,:]+current*(2*ratio/delta)[None,:]
            power = ratio.copy()
            for k in range(2,degree+1):
                following = 2*D(current)/Dnorm-previous
                power *= ratio
                inverse += following*(2*power/delta)[None,:]
                previous,current = current,following
            info['max_chebyshev_ratio'] = float(np.max(np.abs(ratio)))
            info['inverse_absolute_tail_bound'] = float(np.max(2*np.abs(ratio)**(degree+1)/(delta*(1-np.abs(ratio)))))
    Z = [inverse[:,j*(q+1):j*(q+1)+q] for j in range(q)]
    Y = np.column_stack([inverse[:,j*(q+1)+q] for j in range(q)])
    H = np.diag(h)
    R0high = (t*H+2*(H@H+F.T@F))/3
    radial_norm = t*t+2*np.sum(C*C)
    upper = np.triu_indices(q)
    def low(S,alpha):
        return np.column_stack([Y[:,j]-np.trace(S)*Z[j][:,j]-2*Z[j]@S[:,j]-2*alpha*F[:,j] for j in range(q)])
    def response(S,alpha):
        L = low(S,alpha)
        high = (t*S+np.trace(S)*H+2*(H@S+S@H+F.T@L+L.T@F))/6+alpha*R0high
        trace = (t*np.trace(S)+2*np.sum(H*S)+4*np.sum(F*L)+alpha*radial_norm)/3
        return np.concatenate((high[upper],[trace/n]))
    zero = np.zeros((q,q))
    offset = response(zero,0.)
    columns = []
    for i,j in zip(*upper):
        S = np.zeros((q,q)); S[i,j] = S[j,i] = 1.
        columns.append(response(S,0.)-offset)
    columns.append(response(zero,1.)-offset)
    system = np.stack(columns,axis=1)
    target = np.concatenate((Rhigh[upper],[trace_R/n]))
    solution = np.linalg.solve(system,target-offset)
    S = np.zeros((q,q))
    S[upper] = solution[:-1]
    S = S+S.T-np.diag(np.diag(S))
    alpha = float(solution[-1])
    L = low(S,alpha)
    V = L+.5*U@S
    info['small_system_condition'] = float(np.linalg.cond(system))
    info['low_orthogonality_error'] = float(np.linalg.norm(U.T@L))
    return dict(U=U,V=V,alpha=alpha,info=info)
