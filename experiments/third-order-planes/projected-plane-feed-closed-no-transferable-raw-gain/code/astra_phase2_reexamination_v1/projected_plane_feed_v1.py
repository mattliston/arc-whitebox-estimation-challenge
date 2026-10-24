"""Frobenius projection of the one-bond plane feed into existing CP births."""
import numpy as np
import radial_compact_feed_fullpair_v1 as base

def projected_birth(C,K21,aux,alive):
    cov=C.copy();plane=K21.copy()
    np.fill_diagonal(cov,0);np.fill_diagonal(plane,0)
    weights=(aux["gate"]*alive)**2
    vv=np.sum(cov*cov*weights[None,:],axis=1)
    uv=np.sum(cov*plane*weights[None,:],axis=1)
    uu=np.sum(plane*plane*weights[None,:],axis=1)
    coef=uv/np.maximum(vv,1e-30)
    third_derivative=-aux["alpha"]*aux["pdf"]/aux["pre_v"]
    extra=third_derivative*coef
    energy=.5*(uu*vv+uv*uv)*third_derivative**2*alive
    captured=uv*uv*third_derivative**2*alive
    return extra,dict(projected_fraction=float(np.sum(captured)/max(np.sum(energy),1e-30)),
        coefficient_rms=float(np.sqrt(np.mean(coef*coef))))

source=base.PREDICT_SOURCE
old="        birth = d*(1+radial/3)"
assert source.count(old)==1
source=source.replace(old,old+'\n        extra_birth,feed_profile=projected_birth(Cpre,K21,aux,alive)\n'
    '        birth=birth+extra_birth\n'
    '        assert np.all(birth>=-1e-14),"Negative projected birth coefficient"\n')
old="states.append(dict(layer=l, active=int(alive.sum()), births=len(selected), columns=p.shape[1]))"
assert source.count(old)==1
source=source.replace(old,"states.append(dict(layer=l, active=int(alive.sum()), births=len(selected), columns=p.shape[1],feed=feed_profile,birth_relative_rms=float(np.sqrt(np.mean((extra_birth/np.maximum(d,1e-30))**2)))))")
env=dict(base.env,projected_birth=projected_birth)
exec(source,env)
predict=env["predict"]
