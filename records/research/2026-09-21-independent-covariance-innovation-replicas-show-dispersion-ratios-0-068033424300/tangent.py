def tangent_cov(pre,e,E,W):
    import numpy as np
    u=W.T@e;V=W.T@E@W;z=np.diag(V)
    s=pre['sigma'];g=pre['gate'];p=pre['pdf'];a=pre['alpha'];mg=pre['mg']
    dg=p*u/s-a*p*z/(2*s*s)
    T=(g[:,None]*g[None,:])*V+pre['C_pre']*(dg[:,None]*g[None,:]+g[:,None]*dg[None,:])
    np.fill_diagonal(T,(g-mg*p/s)*z+2*mg*(1-g)*u)
    return (T+T.T)*.5
