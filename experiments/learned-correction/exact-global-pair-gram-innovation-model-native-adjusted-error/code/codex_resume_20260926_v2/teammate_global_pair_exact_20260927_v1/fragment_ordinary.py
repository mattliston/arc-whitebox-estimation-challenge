"""Native finite-innovation pair supplier with actual covariance feedback."""
_CF_ENABLED = True
_CF_PARAMS = None
_CF_CURRENT = {}
_CF_STATS = dict(calls=0)

def _cf_init():
    global _CF_PARAMS
    if _CF_PARAMS is None:
        _CF_PARAMS = {key:fnp.asarray(value,dtype=fnp.float32) for key,value in _CF_PAYLOAD.items()}

def _cf_project(A,m):
    h=m/fnp.sqrt(fnp.sum(m*m))
    d=1-2*h*h
    if not bool(fnp.all(d>0)):
        raise ValueError('Covariance supplier projection conditioning')
    A=(A+A.T)*0.5
    fnp.fill_diagonal(A,0)
    b=A@h;v=h/d
    u=2*(b/d-v*((h@(b/d))/(1+h@v)))
    D=A-(fnp.outer(u,h)+fnp.outer(h,u))*0.5
    fnp.fill_diagonal(D,0)
    return D

def _cf_delta(F,C,K,s,m):
    _cf_init();p=_CF_PARAMS
    F=fnp.clip(F.astype(fnp.float32)/p['feature_scale'],-8,8)
    ss=fnp.outer(s,s)
    Rphysical=(C/ss).astype(fnp.float32)
    G=(Rphysical@Rphysical.T)/p['global_scale']
    R=Rphysical/p['rho_scale']
    S=(K/fnp.outer(s*s,s)).astype(fnp.float32)/p['third_scale']
    L=F@p['left.weight'].T
    B=F@p['right.weight'].T
    n=int(F.shape[0]);raw=fnp.empty((n,n),dtype=fnp.float32)
    for lo in range(0,n,64):
        hi=min(lo+64,n)
        z=L[lo:hi,None,:]+B[None,:,:]+p['bias']
        z=z+R[lo:hi,:,None]*p['edge.weight'][:,0]
        z=z+S[lo:hi,:,None]*p['edge.weight'][:,1]
        z=z+S[:,lo:hi].T[:,:,None]*p['edge.weight'][:,2]
        z=z+G[lo:hi,:,None]*p['edge.weight'][:,3]
        y=fnp.tanh(z)@p['output.weight'].T+p['output.bias']
        fnp.copyto(raw[lo:hi],y[:,:,0])
    A=(raw+raw.T).astype(fnp.float64)*(0.001*ss)
    return _cf_project(A,m)

_CF_OLD_FEATURES=features
_CF_OLD_STEP=gnn_step
_CF_OLD_SETUP=Estimator.setup

def features(l,m,C,K21,aux,w):
    x=_CF_OLD_FEATURES(l,m,C,K21,aux,w)
    if _CF_ENABLED and 8<=l<=14:
        _CF_CURRENT.clear()
        _CF_CURRENT.update(layer=l,x=x,K=K21,s=aux['sigma'])
    return x

def gnn_step(l,x,sigma,w,state,P):
    answer=_CF_OLD_STEP(l,x,sigma,w,state,P)
    if _CF_ENABLED and 8<=l<=14:
        if _CF_CURRENT.get('layer')!=l:
            raise RuntimeError('Covariance supplier feature ownership')
        _CF_CURRENT['H']=answer[1]
    return answer

def _cf_feedback(l,m,C):
    if not _CF_ENABLED or not 8<=l<=14:
        return C
    a=_CF_CURRENT
    if a.get('layer')!=l or 'H' not in a:
        raise RuntimeError('Covariance supplier current state missing')
    F=fnp.concatenate((a['x'],a['H']),axis=1)
    D=_cf_delta(F,C,a['K'],a['s'],m)
    _CF_CURRENT.clear()
    _CF_STATS['calls']+=1
    return C+D

def _cf_setup(self,*args,**kwargs):
    result=_CF_OLD_SETUP(self,*args,**kwargs)
    _cf_init()
    return result

Estimator.setup=_cf_setup
