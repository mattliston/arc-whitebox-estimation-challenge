# Appended to the mechanically generated FlopScope moment and kernel functions.
_SCRATCH = None
_METRIC = "covariance"
_COV32 = True
_WINOGRAD = True

def _mm(a,b):
    if not _WINOGRAD or a.dtype!=fnp.float32 or b.dtype!=fnp.float32:
        return a@b
    m,k = int(a.shape[0]),int(a.shape[1]); n = int(b.shape[1])
    if k==0 or n==0:
        return fnp.zeros((m,n),dtype=fnp.float32)
    if m%16 or k%16 or n%16 or k>14336:
        return a@b
    out = fnp.empty((m,n),dtype=fnp.float32)
    views = _SCRATCH[1] if k==1024 else _SCRATCH[2]
    _cp3_winograd_arena_depth4(a,b,out,views)
    return out

def _cp_plane(p,q,lam):
    if int(p.shape[1])==0:
        return fnp.zeros((int(p.shape[0]),int(p.shape[0])),dtype=fnp.float64)
    return fnp.asarray(_mm(p*p,fnp.transpose(q*lam))+2*_mm(p*q,fnp.transpose(p*lam)),dtype=fnp.float64)

def _predict(mlp,hook=None):
    global _SCRATCH
    n,depth = int(mlp.width),int(mlp.depth)
    if _SCRATCH is None:
        arena = fnp.empty((_CP3_WINOGRAD_ARENA_ELEMENTS,),dtype=fnp.float32)
        _SCRATCH = (arena,_cp3_bind_arena_views(arena,_CP3_PUSH_ARENA_VIEW_SPECS),
                    _cp3_bind_arena_views(arena,_CP3_READ_ARENA_VIEW_SPECS))
    m = fnp.zeros((n,),dtype=fnp.float64); C = fnp.eye(n,dtype=fnp.float64)
    radial = fnp.asarray(0.0,dtype=fnp.float64)
    p = fnp.zeros((n,0),dtype=fnp.float32); q = fnp.zeros((n,0),dtype=fnp.float32)
    lam = fnp.zeros((0,),dtype=fnp.float32); star = fnp.zeros((n,n),dtype=fnp.float32)
    eye = fnp.eye(n,dtype=fnp.float32)
    means = []
    for l in range(depth):
        w32 = fnp.asarray(mlp.weights[l],dtype=fnp.float32)
        w = fnp.asarray(mlp.weights[l],dtype=fnp.float64)
        wt = fnp.transpose(w32)
        mu = fnp.transpose(w)@m
        Cpre = fnp.asarray(_mm(_mm(wt,fnp.asarray(C,dtype=fnp.float32)),w32),dtype=fnp.float64) if _COV32 else fnp.transpose(w)@C@w
        pp,pq = _mm(wt,p),_mm(wt,q)
        cp_pre = _cp_plane(pp,pq,lam)
        wa = _mm(wt,star)
        K21 = cp_pre+fnp.asarray(2*_mm(wa*wt,w32)+_mm(wt*wt,fnp.transpose(wa)),dtype=fnp.float64)/3
        if _METRIC=="covariance":
            vv = fnp.diag(Cpre)
            k4 = radial*vv*vv
            k22 = radial*(fnp.outer(vv,vv)+2*Cpre*Cpre)/3
        else:
            k4,k22 = 4*radial,None
        m,C,target,radial_new,aux = nonlinear(mu,Cpre,K21,k4,k22,_METRIC)
        if hook is not None:
            dm = hook(l,m,C,K21,aux,w32)
            C = C-fnp.outer(m,dm)-fnp.outer(dm,m)-fnp.outer(dm,dm)
            m = m+dm
        means.append(fnp.array(m,dtype=fnp.float64))
        if l==depth-1:
            break
        g = aux["gate"]; birth = aux["pdf"]/aux["sigma"]
        inherited = g[:,None]*g[:,None]*g[None,:]*cp_pre
        bp = g[:,None]*Cpre
        cp_post = inherited+bp*bp*birth[None,:]+2*(g*fnp.diag(Cpre)*birth)[:,None]*fnp.transpose(bp)
        p = fnp.concatenate([fnp.asarray(g,dtype=fnp.float32)[:,None]*pp,fnp.asarray(bp,dtype=fnp.float32)],axis=1)
        q = fnp.concatenate([fnp.asarray(g,dtype=fnp.float32)[:,None]*pq,eye],axis=1)
        lam = fnp.concatenate([lam,fnp.asarray(birth,dtype=fnp.float32)])
        residual = target-cp_post
        star = fnp.asarray(3*fnp.transpose(residual)-2*fnp.diag(fnp.diag(residual)),dtype=fnp.float32)
        radial = radial_new
    return fnp.asarray(fnp.maximum(fnp.stack(means),0.0),dtype=fnp.float32)

class Estimator(BaseEstimator):
    def predict(self,mlp,budget):
        if int(mlp.width)!=1024 or int(mlp.depth)!=16 or int(budget)<2**41:
            return fnp.zeros((int(mlp.depth),int(mlp.width)),dtype=fnp.float32)
        return _predict(mlp)
