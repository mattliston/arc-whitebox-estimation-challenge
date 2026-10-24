"""Correct only the ancillary covariance response for the effective pair map."""
from reference import *

def covariance_response(delta,aux,cov):
    alpha=np.asarray(aux['alpha'],dtype=np.float64)
    pdf=np.asarray(aux['pdf'],dtype=np.float64)
    gate=np.asarray(aux['gate'],dtype=np.float64)
    sigma=np.asarray(aux['sigma'],dtype=np.float64)
    v=np.asarray(aux['pre_v'],dtype=np.float64)
    assert np.all(v>0)
    rho=np.asarray(aux['pre_cov'],dtype=np.float64)/np.outer(sigma,sigma)
    np.fill_diagonal(rho,0)
    A=[None,gate,pdf,-alpha*pdf,(alpha*alpha-1)*pdf]
    change=np.zeros_like(delta,dtype=np.float64)
    rr=np.ones_like(delta,dtype=np.float64)
    for r in range(3):
        change+=rr*(np.outer(A[r+2]*sigma/v,A[r+1])*delta
                    +np.outer(A[r+1],A[r+2]*sigma/v)*delta.T)/2
        rr=rr*rho/(r+1)
    assert not np.any(np.diag(delta)) and not np.any(np.diag(change))
    covariance=np.array(cov,dtype=np.float64,copy=True);np.fill_diagonal(covariance,0)
    cn=norm(change);base=norm(covariance)
    return dict(local_covariance_error_norm=cn,original_covariance_offdiag_norm=base,
        local_covariance_relative=ratio(cn,base),local_covariance_maxabs=float(np.max(np.abs(change),initial=0)),
        local_covariance_is_formal_fp64_series_response=True),change
