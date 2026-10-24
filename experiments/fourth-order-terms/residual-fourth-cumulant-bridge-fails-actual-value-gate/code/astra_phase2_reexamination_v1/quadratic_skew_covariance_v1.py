"""Offline quadratic skew proxy and analytic pair-response columns.

The pairwise conditional extensions are explicitly approximate, not a
consistent fourth-order carrier. No labels, fits, or FlopScope emulation.
"""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from radial_mehler import derivatives


def proxy(R,S,indices):
    n=len(R);idx=np.asarray(indices)
    r=np.array(R[np.ix_(idx,idx)],np.float64)
    r=(r+r.T)/2
    t=np.array(S[np.ix_(idx,idx)],np.float64)
    factor=cho_factor(r,lower=True,check_finite=False)
    solved=cho_solve(factor,t.T,check_finite=False)
    solve_error=float(np.linalg.norm(r@solved-t.T)/max(np.linalg.norm(t),1e-30))
    if solve_error>1e-8:raise ValueError(f'Unqualified solve residual {solve_error}')
    D=t@solved;D=(D+D.T)/2
    rho=r.copy();np.fill_diagonal(rho,0)
    denominator=1-rho*rho
    if denominator.min()<=1e-8:raise ValueError('Near-singular pair conditioning')
    H=(t*t+t.T*t.T-2*rho*t*t.T)/denominator
    skew=np.diag(t)
    J=(skew[:,None]*(t-rho*t.T)+t*(t.T-rho*t))/denominator
    np.fill_diagonal(H,np.diag(D));np.fill_diagonal(J,np.diag(D))
    q22=np.zeros((n,n));q31=np.zeros((n,n));q4=np.zeros(n)
    q22[np.ix_(idx,idx)]=(4/9)*(D+2*H)
    q31[np.ix_(idx,idx)]=(4/3)*J
    q4[idx]=(4/3)*np.diag(D)
    return q22,q31,q4,dict(active=len(idx),solve_relative_error=solve_error,
                          min_pair_denominator=float(denominator.min()),
                          gram_diagonal_min=float(np.diag(D).min()))


def response(R,sigma,alpha,pdf,gate,gaussian_mean_over_sigma,host_mean_over_sigma,
             dS=None,dQ22=None,dQ31=None,dq4=None):
    """Derivative of the parent's first-order Gauss6/Edge2 covariance map."""
    n=len(R)
    A,_=derivatives(alpha,pdf,gate,gaussian_mean_over_sigma,6)
    rho=np.array(R,np.float64);np.fill_diagonal(rho,0)
    S=np.zeros_like(rho) if dS is None else np.array(dS,np.float64)
    t3=np.diag(S).copy();np.fill_diagonal(S,0)
    Q22=np.zeros_like(rho) if dQ22 is None else dQ22
    Q31=np.zeros_like(rho) if dQ31 is None else np.array(dQ31,np.float64)
    np.fill_diagonal(Q31,0)
    q4=np.zeros(n) if dq4 is None else dq4
    out=np.zeros_like(rho);rr=np.ones_like(rho)
    for r in range(3):
        ec=(S*np.outer(A[r+2],A[r+1])+S.T*np.outer(A[r+1],A[r+2]))/2
        ec+=Q22*np.outer(A[r+2],A[r+2])/4
        ec+=(Q31*np.outer(A[r+3],A[r+1])+Q31.T*np.outer(A[r+1],A[r+3]))/6
        if r:
            ec+=(np.outer(t3*A[r+3],A[r])+np.outer(A[r],t3*A[r+3]))/6
            ec+=(np.outer(q4*A[r+4],A[r])+np.outer(A[r],q4*A[r+4]))/24
        out+=rr*ec
        rr=rr*rho/(r+1)
    out*=np.outer(sigma,sigma)
    # Exact derivative of r2-m^2, using the CURRENT uncorrected host mean.
    diag=sigma*sigma*(t3*(pdf+host_mean_over_sigma*alpha*pdf)/3
                    +q4*(-alpha*pdf-host_mean_over_sigma*(alpha*alpha-1)*pdf)/12)
    np.fill_diagonal(out,diag)
    return (out+out.T)/2


def columns(C,K21,k4,mu,host,indices):
    from scipy.special import ndtr
    sigma=np.sqrt(np.diag(C));alpha=mu/sigma
    pdf=np.exp(-alpha*alpha/2)/np.sqrt(2*np.pi);gate=ndtr(alpha)
    mg=pdf+alpha*gate
    R=C/np.outer(sigma,sigma);R=(R+R.T)/2
    S=K21/(sigma[:,None]**2*sigma[None,:])
    radial=float(np.mean(k4/sigma**4))
    q22,q31,q4,info=proxy(R,S,indices)
    common=(R,sigma,alpha,pdf,gate,mg,host/sigma)
    third=response(*common,dS=S)
    fourth=response(*common,dQ22=radial*(1+2*R*R)/3,dQ31=radial*R,
                    dq4=np.full(len(R),radial))
    quadratic=response(*common,dQ22=q22,dQ31=q31,dq4=q4)
    return np.stack((third,fourth,quadratic)),info
