"""One post14 physical covariance supplier; every inference operation is metered."""
_TPC_ENABLED = True
_TPC_PARAMS = None
_TPC_CURRENT = {}
_TPC_STATS = dict(calls=0)

def _tpc_init():
    global _TPC_PARAMS
    if _TPC_PARAMS is None:
        _TPC_PARAMS = _unpack_static_fp32(_TPC_LITERAL)

def _tpc_project(A,m):
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

def _tpc_delta(F,C,K,s,m):
    _tpc_init();p=_TPC_PARAMS
    F=fnp.clip(F.astype(fnp.float32)/p['feature_scale'],-8,8)
    ss=fnp.outer(s,s)
    R=(C/ss).astype(fnp.float32)/p['rho_scale']
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
        y=fnp.tanh(z)@p['output.weight'].T+p['output.bias']
        fnp.copyto(raw[lo:hi],y[:,:,0])
    A=(raw+raw.T).astype(fnp.float64)*(0.001*ss)
    return _tpc_project(A,m)

_TPC_OLD_FEATURES=features
_TPC_OLD_STEP=gnn_step
_TPC_OLD_SETUP=Estimator.setup

def features(l,m,C,K21,aux,w):
    x=_TPC_OLD_FEATURES(l,m,C,K21,aux,w)
    if _TPC_ENABLED and l==14:
        _TPC_CURRENT.clear()
        _TPC_CURRENT.update(layer=l,x=x,K=K21,s=aux['sigma'])
    return x

def gnn_step(l,x,sigma,w,state,P):
    answer=_TPC_OLD_STEP(l,x,sigma,w,state,P)
    if _TPC_ENABLED and l==14:
        if _TPC_CURRENT.get('layer')!=l:
            raise RuntimeError('Covariance supplier feature ownership')
        _TPC_CURRENT['H']=answer[1]
    return answer

def _tpc_feedback(l,m,C):
    if not _TPC_ENABLED or not l==14:
        return C
    a=_TPC_CURRENT
    if a.get('layer')!=l or 'H' not in a:
        raise RuntimeError('Covariance supplier current state missing')
    F=fnp.concatenate((a['x'],a['H']),axis=1)
    D=_tpc_delta(F,C,a['K'],a['s'],m)
    _TPC_CURRENT.clear()
    _TPC_STATS['calls']+=1
    return C+D

def _tpc_setup(self,*args,**kwargs):
    result=_TPC_OLD_SETUP(self,*args,**kwargs)
    _tpc_init()
    return result

Estimator.setup=_tpc_setup
