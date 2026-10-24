"""Complete the current radial map's quadratic third-cumulant density term.

This is the familiar L3**2/2 Edgeworth term, not a new connected sixth
cumulant. The historical D11 already owned it. The NEW independent map
currently contains only linear L3/L4 terms, so this checks a concrete
omission on a different, frozen host rather than reopening D11's sentinel.
Correlation order stays two, with no new state or tuned coefficients.
"""
import inspect
import numpy as np
import radial_mehler as base


def quadratic_pair_term(F,G,r,K,t3):
    ui=t3[:,None];uj=t3[None,:]
    term=(K*K/8+ui*K.T/12)*np.outer(F[r+4],G[r+2])
    term=term+(K.T*K.T/8+uj*K/12)*np.outer(F[r+2],G[r+4])
    term=term+ui*K*np.outer(F[r+5],G[r+1])/12
    term=term+uj*K.T*np.outer(F[r+1],G[r+5])/12
    term=term+K*K.T*np.outer(F[r+3],G[r+3])/4
    if r:
        term=term+(np.outer(t3*t3*F[r+6],G[r])+np.outer(F[r],t3*t3*G[r+6]))/72
        term=term+np.outer(t3*F[r+3],t3*G[r+3])/36
    return term


source=inspect.getsource(base.make_nonlinear)
edits={
    "        var=r2-m*m":
        "        q=(k3/s**3)**2/72\n"
        "        h2=a*a-1;h3=a*a*a-3*a;h4=a**4-6*a*a+3\n"
        "        m=m+s*q*h4*pdf\n"
        "        r2=r2-2*s*s*q*h3*pdf\n"
        "        r3=r3+6*s**3*q*h2*pdf\n"
        "        r4=r4-24*s**4*q*a*pdf\n"
        "        var=r2-m*m",
    "maxorder=max(gauss_order,edge_order+4)":"maxorder=max(gauss_order,edge_order+6)",
    "                    ans+=rr*ec":
        "                    ec+=quadratic_pair_term(F,G,r,T21,t3)\n"
        "                    ans+=rr*ec",
}
for old,new in edits.items():
    assert source.count(old)==1,old
    source=source.replace(old,new)
env=dict(vars(base),quadratic_pair_term=quadratic_pair_term)
exec(source,env)
make_nonlinear=env["make_nonlinear"]
