"""Real-arithmetic local tangent, with explicit native-primal anchors."""
import math
import numpy as np

class Dual:
    def __init__(self,v,d):self.v=np.asarray(v,dtype=np.float64);self.d=np.asarray(d,dtype=np.float64)
    @property
    def k(self):return self.d.shape[-1]
    def coerce(self,x):return x if isinstance(x,Dual) else constant(x,self.k)
    def __add__(self,x):x=self.coerce(x);return Dual(self.v+x.v,self.d+x.d)
    __radd__=__add__
    def __neg__(self):return Dual(-self.v,-self.d)
    def __sub__(self,x):return self+-self.coerce(x)
    def __rsub__(self,x):return self.coerce(x)+-self
    def __mul__(self,x):x=self.coerce(x);return Dual(self.v*x.v,self.d*x.v[...,None]+x.d*self.v[...,None])
    __rmul__=__mul__
    def __truediv__(self,x):return self*self.coerce(x)**-1
    def __rtruediv__(self,x):return self.coerce(x)*self**-1
    def __pow__(self,p):return Dual(self.v**p,p*self.v[...,None]**(p-1)*self.d)
    def __getitem__(self,key):return Dual(self.v[key],self.d[key])
    def mean(self,axis=0):return Dual(self.v.mean(axis=axis),self.d.mean(axis=axis))
    def exp(self):v=np.exp(self.v);return Dual(v,v[...,None]*self.d)
    def log(self):return Dual(np.log(self.v),self.d/self.v[...,None])
    def clip(self,lo,hi):return Dual(np.clip(self.v,lo,hi),self.d*((self.v>lo)&(self.v<hi))[...,None])
    def cdf(self):
        v=np.fromiter((.5*math.erfc(-float(x)/math.sqrt(2.)) for x in self.v.flat),dtype=np.float64,count=self.v.size).reshape(self.v.shape)
        return Dual(v,self.d*(np.exp(-self.v*self.v/2)/math.sqrt(2*math.pi))[...,None])
    def matmul(self,w):
        assert self.v.ndim==2
        n,f=self.v.shape
        d=(self.d.transpose(0,2,1).reshape(n*self.k,f)@w).reshape(n,self.k,w.shape[1]).transpose(0,2,1)
        return Dual(self.v@w,d)
    def anchor(self,v):return Dual(v,self.d)

def constant(v,k):
    v=np.asarray(v,dtype=np.float64);return Dual(v,np.zeros((*v.shape,k),dtype=np.float64))
def stack(xs,axis=1):return Dual(np.stack([x.v for x in xs],axis=axis),np.stack([x.d for x in xs],axis=axis))
def cat(xs,axis=1):return Dual(np.concatenate([x.v for x in xs],axis=axis),np.concatenate([x.d for x in xs],axis=axis))
def silu(x):return x/(1+(-x).exp())

def covariance_value(C,W,indices,alive,X,q):
    z=X@q;s=np.exp(.005*np.tanh(z));changed=(C*s[:,None])*s[None,:]
    wa=W[indices];ca=changed[np.ix_(indices,indices)]
    active=np.sum(wa*(ca@wa),axis=0)
    cold=~alive;v=active+np.sum(W[cold]**2*np.diag(changed)[cold,None],axis=0)
    return v,active

def covariance_jacobian(C,W,indices,alive,X):
    wa=W[indices].astype(np.float64);ca=C[np.ix_(indices,indices)].astype(np.float64)
    Xa=.005*X[indices]
    active=(wa*(ca@wa)+wa*(ca.T@wa)).T@Xa
    cold=~alive
    dv=active+2*(W[cold].astype(np.float64)**2).T@(np.diag(C)[cold,None]*(.005*X[cold]))
    return dv,active

def terminal(a,dv,da,*,native_anchors=False,v_override=None,active_override=None):
    """Complete final scalar/feature/GNN tangent for fixed native messages."""
    k=dv.shape[1];n=len(a['v'])
    v=Dual(a['v'] if v_override is None else v_override,dv)
    active=Dual(a['active_v'] if active_override is None else active_override,da)
    mu=constant(a['mu'],k);k3=constant(a['k3base'],k)+3*active*a['bpre']
    k4=float(a['radial'])*v*v;s=v**.5;alpha=mu/s
    pdf=(-alpha*alpha/2).exp()/math.sqrt(2*math.pi);g=alpha.cdf()
    mg=s*pdf+mu*g
    r2g=(mu*mu+v)*g+mu*s*pdf
    r3g=mu*(mu*mu+3*v)*g+(mu*mu+2*v)*s*pdf
    r4g=(mu**4+6*mu*mu*v+3*v*v)*g+mu*(mu*mu+5*v)*s*pdf
    d=pdf/s;e=-alpha*pdf/v;f=(alpha*alpha-1)*pdf/(v*s)
    m=mg+k3*e/6+k4*f/24;r2=r2g+k3*d/3+k4*e/12
    r3=r3g+k3*g+k4*d/4;r4=r4g+4*k3*mg+k4*g
    postv=r2-m*m;post3=r3-3*m*r2+2*m**3
    post4=r4-4*m*r3+6*m*m*r2-3*m**4-3*postv*postv
    w=a['W'].astype(np.float64);w2=w*w;w3=w2*w;w4=w2*w2
    ones=constant(np.ones(n),k);zero=constant(np.zeros(n),k)
    values=[ones,ones*(15/16),ones,alpha,s.log()-s.log().mean(),g,pdf,mg/s,m/s,(m-mg)/s,
            k3/(v*s),postv/v,k4/(v*v),zero,zero,zero,
            constant(w.sum(axis=0)*.7071067811865475,k),
            constant((w2-.001953125).sum(axis=0)*11.313708498984761,k),
            constant(w3.sum(axis=0)*93.47798314754834,k),
            constant((w4-1.1444091796875e-5).sum(axis=0)*836.0924988699916,k),
            post3/(v*s),post4/(v*v),zero,zero,zero,zero,zero,zero,zero,ones*(m/s).mean()]
    P={key[2:]:val.astype(np.float64) for key,val in a.items() if key.startswith('P_')}
    x=((stack(values)-P['centers'][15])/P['scales'][15]).clip(-8,8)
    mask=np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,1])
    x=x*mask
    if native_anchors:
        x=x.anchor(a['normalized_x']);x.d*=((x.v>-8)&(x.v<8))[...,None]
    gate=x[:,5:6]
    msgs=cat([gate*a['m1'],gate*a['m2'],gate*a['m3']]+[constant(np.log1p(np.maximum(a[name],0)),k) for name in ['e1','e2','e3']]).clip(-8,8)
    h=cat([x,msgs]).matmul(P['enc0_w'].T)+P['enc0_b']
    if native_anchors:h=h.anchor(a['pre1'])
    h=silu(h).matmul(P['enc2_w'].T)+P['enc2_b']
    if native_anchors:h=h.anchor(a['pre2'])
    h=silu(h).matmul(P['enc4_w'].T)+P['enc4_b']
    if native_anchors:h=h.anchor(a['H15'])
    read=h.matmul(P['head'][15,:,None])[:,0]+P['head_b'][15]
    out=m+(s*P['rms'][15])*read*(1+.25*np.tanh(P['gain'][15]))
    if native_anchors:
        out.d*=(a['prediction'][-1]>0)[:,None]
    else:out=out.clip(0,np.inf)
    return out
