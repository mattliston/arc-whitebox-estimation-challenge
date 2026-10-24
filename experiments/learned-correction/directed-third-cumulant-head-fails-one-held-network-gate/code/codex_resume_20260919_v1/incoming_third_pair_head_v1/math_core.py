import numpy as np
KEYS=('left','right','edge','bias','output')

def initialize(seed=712031,dtype=np.float32):
    rng=np.random.default_rng(seed)
    return dict(left=rng.normal(0,.15/np.sqrt(140),(140,32)).astype(dtype),
                right=rng.normal(0,.15/np.sqrt(140),(140,32)).astype(dtype),
                edge=rng.normal(0,.15,(3,32)).astype(dtype),bias=np.zeros(32,dtype),output=np.zeros(32,dtype))

def forward(F,R,S,t,target_scale):
    u=F@t['left'];v=F@t['right'];out=np.empty_like(R)
    for i in range(0,len(F),64):
        z=u[i:i+64,None,:]+v[None,:,:]+t['bias']
        z=z+R[i:i+64,:,None]*t['edge'][0]+S[i:i+64,:,None]*t['edge'][1]+S.T[i:i+64,:,None]*t['edge'][2]
        out[i:i+64]=(np.tanh(z)@t['output'])*target_scale
    np.fill_diagonal(out,0.)
    return out

def backward(F,R,S,t,target_scale,grad):
    grad=grad.copy();np.fill_diagonal(grad,0.)
    u=F@t['left'];v=F@t['right'];du=np.zeros_like(u,dtype=np.float64);dv=np.zeros_like(v,dtype=np.float64)
    result={k:np.zeros_like(t[k],dtype=np.float64) for k in KEYS}
    for i in range(0,len(F),64):
        z=u[i:i+64,None,:]+v[None,:,:]+t['bias']
        z=z+R[i:i+64,:,None]*t['edge'][0]+S[i:i+64,:,None]*t['edge'][1]+S.T[i:i+64,:,None]*t['edge'][2]
        h=np.tanh(z);g=grad[i:i+64]*target_scale
        result['output']+=np.einsum('ij,ijh->h',g,h,dtype=np.float64)
        dh=g[:,:,None]*t['output']*(1-h.astype(np.float64)**2)
        du[i:i+64]=np.sum(dh,axis=1);dv+=np.sum(dh,axis=0);result['bias']+=np.sum(dh,axis=(0,1))
        for k,edge in enumerate((R[i:i+64],S[i:i+64],S.T[i:i+64])):
            result['edge'][k]+=np.einsum('ij,ijh->h',edge,dh,dtype=np.float64)
    result['left']=F.astype(np.float64).T@du;result['right']=F.astype(np.float64).T@dv
    return result

def consumer(c,D):
    q=c['L']*D;C=q+q.T;np.fill_diagonal(C,0.)
    N=np.sum(c['Wnext']*(C@c['Wnext']),axis=0)*c['gnext']
    return C,N

def losses(c,pred,energy=None,gradient=False):
    e=pred.astype(np.float64)-c['target'];np.fill_diagonal(e,0.)
    C,N=consumer(c,e);n=len(e);off=n*(n-1)
    terms=np.array([np.sum(c['weight']*e*e)/off,np.sum(C*C)/off,np.mean(N*N)])
    if energy is None:return terms
    factors=np.array([.25,.25,.5])/energy;value=float(terms@factors)
    if not gradient:return value,terms
    G=(2*factors[1]/off)*C
    G+=(c['Wnext']*(2*factors[2]/n*N*c['gnext'])[None,:])@c['Wnext'].T
    d=(2*factors[0]/off)*c['weight']*e+c['L']*(G+G.T)
    np.fill_diagonal(d,0.)
    return value,terms,d
