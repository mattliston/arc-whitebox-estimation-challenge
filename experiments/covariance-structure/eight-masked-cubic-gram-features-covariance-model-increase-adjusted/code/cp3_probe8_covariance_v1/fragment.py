"""Paid FP32 masked CP8 feature and finite-innovation covariance supplier."""
_CP8_ENABLED = True
_CP8_PARAMS = None
_CP8_PROBES = None
_CP8_CURRENT = {}
_CP8_STATS = dict(calls=0, probe_calls=0)

def _cp8_init():
    global _CP8_PARAMS, _CP8_PROBES
    if _CP8_PARAMS is None:
        _CP8_PARAMS = {k:fnp.asarray(v,dtype=fnp.float32) for k,v in _CP8_PAYLOAD.items()}
        _CP8_PROBES = {k:fnp.asarray(v,dtype=fnp.float32) for k,v in _CP8_PROBE_PAYLOAD.items()}

def _cp8_probe(pp,lam,s,Omega,Psi):
    # Reciprocal is FP64, then cast: casting sigma before reciprocal is different.
    assert pp.dtype == fnp.float32 and s.dtype == fnp.float64
    assert Omega.dtype == fnp.float32 and Psi.dtype == fnp.float32
    assert pp.shape[0] == len(s) and pp.shape[1] == len(lam)
    assert Omega.shape == Psi.shape == (len(s),8)
    invsigma = (1/s).astype(fnp.float32)
    A = pp.T @ (Omega*invsigma[:,None])
    B = pp.T @ (Psi*invsigma[:,None])
    Z = (pp @ (lam.astype(fnp.float32)[:,None]*A*B))*invsigma[:,None]
    _CP8_STATS['probe_calls'] += 1
    return Z

def _cp8_project(A,m):
    h=m/fnp.sqrt(fnp.sum(m*m))
    d=1-2*h*h
    if not bool(fnp.all(d>0)):
        raise ValueError('CP8 covariance supplier projection conditioning')
    A=(A+A.T)*0.5
    fnp.fill_diagonal(A,0)
    b=A@h;v=h/d
    u=2*(b/d-v*((h@(b/d))/(1+h@v)))
    D=A-(fnp.outer(u,h)+fnp.outer(h,u))*0.5
    fnp.fill_diagonal(D,0)
    return D

def _cp8_delta(F,C,K,s,m,Z):
    _cp8_init();p=_CP8_PARAMS
    assert Z.dtype == fnp.float32 and Z.shape == (F.shape[0],8)
    F=fnp.clip(F.astype(fnp.float32)/p['feature_scale'],-8,8)
    ss=fnp.outer(s,s)
    R=(C/ss).astype(fnp.float32)/p['rho_scale']
    S=(K/fnp.outer(s*s,s)).astype(fnp.float32)/p['third_scale']
    G=((Z@Z.T)/fnp.float32(8))/p['global_scale']
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
    return _cp8_project(A,m)

_CP8_OLD_FEATURES=features
_CP8_OLD_STEP=gnn_step

def features(l,m,C,K21,aux,w):
    x=_CP8_OLD_FEATURES(l,m,C,K21,aux,w)
    if _CP8_ENABLED and 8<=l<=14:
        _CP8_CURRENT.clear()
        _CP8_CURRENT.update(layer=l,x=x,K=K21,s=aux['sigma'],
            pp=_E2_LAYER['pp'],lam=_E2_LAYER['lam'])
    return x

def gnn_step(l,x,sigma,w,state,P):
    answer=_CP8_OLD_STEP(l,x,sigma,w,state,P)
    if _CP8_ENABLED and 8<=l<=14:
        if _CP8_CURRENT.get('layer')!=l:
            raise RuntimeError('CP8 covariance supplier feature ownership')
        _CP8_CURRENT['H']=answer[1]
    return answer

def _cp8_feedback(l,m,C):
    if not _CP8_ENABLED or not 8<=l<=14:
        return C
    a=_CP8_CURRENT
    if a.get('layer')!=l or 'H' not in a:
        raise RuntimeError('CP8 covariance supplier current state missing')
    if a['pp'] is not _E2_LAYER.get('pp') or a['lam'] is not _E2_LAYER.get('lam'):
        raise RuntimeError('CP8 resident factor ownership changed before feedback')
    _cp8_init()
    Z=_cp8_probe(a['pp'],a['lam'],a['s'],_CP8_PROBES['Omega'],_CP8_PROBES['Psi'])
    F=fnp.concatenate((a['x'],a['H']),axis=1)
    D=_cp8_delta(F,C,a['K'],a['s'],m,Z)
    _CP8_CURRENT.clear()
    _CP8_STATS['calls']+=1
    return C+D
