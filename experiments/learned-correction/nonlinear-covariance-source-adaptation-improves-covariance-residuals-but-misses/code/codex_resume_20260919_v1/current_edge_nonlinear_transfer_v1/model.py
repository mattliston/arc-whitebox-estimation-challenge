"""Fixed edge-source proxy and its real-arithmetic reverse, not native autodiff."""
import numpy as np
KEYS=('left','right','edge','bias','output')
def forward(d,p,cache=False):
    h=(d['F']@p['left'])[:,None,:]+(d['F']@p['right'])[None,:,:]+p['bias']
    h=h+d['R'][:,:,None]*p['edge'][0]+d['S'][:,:,None]*p['edge'][1]+d['S'].T[:,:,None]*p['edge'][2]
    t=np.tanh(h)
    raw=(t@p['output'])*d['pf'][:,None]*d['gf'][None,:]
    value=.5*(raw+raw.T)*(d['sf'][:,None]*d['sf'][None,:])*d['factor']
    return (value,t) if cache else value

def objective(d,p,p0,nc,nv,hidden=True,gradient=True):
    value,t=forward(d,p,True)
    error=value-d['base']-d['U'];n=len(error)
    v=np.sum(d['W']*(error@d['W']),axis=0)*d['response']
    cov=float(np.mean(error.astype(np.float64)**2));resp=float(np.mean(v.astype(np.float64)**2))
    penalty=.001*sum(float(np.sum((p[k].astype(np.float64)-p0[k])**2)) for k in KEYS)/528
    loss=cov/nc+resp/nv+penalty
    if not gradient:return loss,dict(cov=cov,response=resp,penalty=penalty)
    G=(2/(n*n*nc))*error+(d['W']*(2/(n*nv)*v*d['response'])[None,:])@d['W'].T
    dq=.5*(G+G.T)*(d['sf'][:,None]*d['sf'][None,:])*d['factor']
    dq=dq*d['pf'][:,None]*d['gf'][None,:]
    out={'output':np.einsum('ijc,ij->c',t,dq,optimize=True)}
    if hidden:
        dh=dq[:,:,None]*p['output'][None,None,:]*(1-t*t)
        out.update(left=d['F'].T@np.sum(dh,axis=1),right=d['F'].T@np.sum(dh,axis=0),
            bias=np.sum(dh,axis=(0,1)),edge=np.stack([np.einsum('ijc,ij->c',dh,x,optimize=True) for x in [d['R'],d['S'],d['S'].T]]))
    for k in out:out[k]=out[k].astype(np.float64)+(.002/528)*(p[k].astype(np.float64)-p0[k])
    return loss,out,dict(cov=cov,response=resp,penalty=penalty)

def reference(d,p):
    """Independent explicit channel sum with FP64 (or complex) scalar parameters."""
    dtype=np.result_type(*[v.dtype for v in p.values()],np.float64)
    n=len(d['F']);q=np.zeros((n,n),dtype=dtype)
    F=np.asarray(d['F'],dtype=dtype);R=np.asarray(d['R'],dtype=dtype);S=np.asarray(d['S'],dtype=dtype)
    for c in range(16):
        a=sum(F[:,j]*p['left'][j,c] for j in range(14))
        b=sum(F[:,j]*p['right'][j,c] for j in range(14))
        h=a[:,None]+b[None,:]+p['bias'][c]+R*p['edge'][0,c]+S*p['edge'][1,c]+S.T*p['edge'][2,c]
        q+=np.tanh(h)*p['output'][c]
    raw=q*d['pf'][:,None]*d['gf'][None,:]
    return .5*(raw+raw.T)*d['sf'][:,None]*d['sf'][None,:]*d['factor']
