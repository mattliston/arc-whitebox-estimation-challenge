"""Fourth-cumulant metric includes a controlled mean outer product."""
import inspect
import numpy as np
import k3_radial_numpy as base

def make_predict(beta):
    def nonlinear(mu,C,K21,k4,k22_override=None,metric="identity"):
        m,cov,k21,alpha,aux=base.nonlinear(mu,C,K21,k4,k22_override,metric)
        M=cov+beta*np.outer(m,m)
        alpha*= (np.trace(cov)**2+2*np.sum(cov*cov))/(np.trace(M)**2+2*np.sum(M*M))
        return m,cov,k21,alpha,aux
    src=inspect.getsource(base.predict)
    old='''            vv = np.diag(Cpre)
            k4 = radial*vv*vv
            k22 = radial*(np.outer(vv,vv)+2*Cpre*Cpre)/3'''
    new='''            Mpre = Cpre+beta*np.outer(mu,mu)
            vv = np.diag(Mpre)
            k4 = radial*vv*vv
            k22 = radial*(np.outer(vv,vv)+2*Mpre*Mpre)/3'''
    assert old in src
    env=dict(vars(base));env.update(nonlinear=nonlinear,beta=beta)
    exec(src.replace(old,new),env)
    return env["predict"]
