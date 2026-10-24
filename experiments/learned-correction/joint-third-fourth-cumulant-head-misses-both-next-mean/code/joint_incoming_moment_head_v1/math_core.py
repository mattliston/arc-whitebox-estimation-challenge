import math
import numpy as np
KEYS=('left','right','edge','bias','output');LOSS_WEIGHTS=np.array([.1,.1,.1,.25,.45])
def initialize(dtype=np.float32):
    rng=np.random.default_rng(923071)
    return dict(left=rng.normal(0,.15/np.sqrt(140),(140,32)).astype(dtype),right=rng.normal(0,.15/np.sqrt(140),(140,32)).astype(dtype),
                edge=rng.normal(0,.15,(3,32)).astype(dtype),bias=np.zeros(32,dtype),output=np.zeros((32,3),dtype))
def coefficients(C,s,a,p,g):
    R=C/np.outer(s,s);np.fill_diagonal(R,0);A=[g,p,-a*p,(a*a-1)*p,-(a**3-3*a)*p]
    B=np.zeros((5,len(s),len(s)),np.float64)
    for r in range(3):
        factor=np.outer(s,s)*(R**r/math.factorial(r))
        B[0]+=factor*np.outer(A[r+1],A[r])*.5;B[1]+=factor*np.outer(A[r],A[r+1])*.5
        B[2]+=factor*np.outer(A[r+2],A[r])/6;B[3]+=factor*np.outer(A[r],A[r+2])/6
        B[4]+=factor*np.outer(A[r+1],A[r+1])*.25
    for b in B:np.fill_diagonal(b,0)
    weight=np.stack([B[0]**2+B[1].T**2,B[2]**2+B[3].T**2,.5*(B[4]**2+B[4].T**2)])
    weight/=np.sum(weight,axis=(1,2))[:,None,None]/(len(s)*(len(s)-1))
    return B,weight
def forward(F,R,S,t,scale):
    u=F@t['left'];v=F@t['right'];out=np.empty((len(F),len(F),3),F.dtype)
    for i in range(0,len(F),64):
        z=u[i:i+64,None,:]+v[None,:,:]+t['bias']
        z=z+R[i:i+64,:,None]*t['edge'][0]+S[i:i+64,:,None]*t['edge'][1]+S.T[i:i+64,:,None]*t['edge'][2]
        out[i:i+64]=(np.tanh(z)@t['output'])*scale
    out=out.transpose(2,0,1).copy();out[2]=.5*(out[2]+out[2].T)
    for x in out:np.fill_diagonal(x,0)
    return out
def backward(F,R,S,t,scale,grad):
    grad=grad.copy();grad[2]=.5*(grad[2]+grad[2].T)
    for x in grad:np.fill_diagonal(x,0)
    grad=grad.transpose(1,2,0);u=F@t['left'];v=F@t['right'];du=np.zeros_like(u,dtype=np.float64);dv=np.zeros_like(v,dtype=np.float64)
    result={k:np.zeros_like(t[k],dtype=np.float64) for k in KEYS}
    for i in range(0,len(F),64):
        z=u[i:i+64,None,:]+v[None,:,:]+t['bias']
        z=z+R[i:i+64,:,None]*t['edge'][0]+S[i:i+64,:,None]*t['edge'][1]+S.T[i:i+64,:,None]*t['edge'][2]
        h=np.tanh(z);d=grad[i:i+64]*scale
        result['output']+=np.einsum('ijh,ijc->hc',h,d,dtype=np.float64)
        dh=(d@t['output'].astype(np.float64).T)*(1-h.astype(np.float64)**2)
        du[i:i+64]=np.sum(dh,axis=1);dv+=np.sum(dh,axis=0);result['bias']+=np.sum(dh,axis=(0,1))
        for k,e in enumerate((R[i:i+64],S[i:i+64],S.T[i:i+64])):result['edge'][k]+=np.einsum('ij,ijh->h',e,dh,dtype=np.float64)
    result['left']=F.astype(np.float64).T@du;result['right']=F.astype(np.float64).T@dv
    return result
def consumer(c,E):
    B=c['B'];C=B[0]*E[0]+B[1]*E[0].T+B[2]*E[1]+B[3]*E[1].T+B[4]*E[2]
    np.fill_diagonal(C,0);N=np.sum(c['Wnext']*(C@c['Wnext']),axis=0)*c['gnext'];return C,N
def losses(c,pred,energy=None,gradient=False):
    e=pred.astype(np.float64)-c['target']
    for x in e:np.fill_diagonal(x,0)
    C,N=consumer(c,e);n=len(C);off=n*(n-1)
    terms=np.concatenate((np.sum(c['weight']*e*e,axis=(1,2))/off,[np.sum(C*C)/off,np.mean(N*N)]))
    if energy is None:return terms
    factors=LOSS_WEIGHTS/energy;value=float(terms@factors)
    if not gradient:return value,terms
    G=2*factors[3]/off*C+(c['Wnext']*(2*factors[4]/n*N*c['gnext'])[None,:])@c['Wnext'].T;B=c['B']
    grad=2*factors[:3,None,None]/off*c['weight']*e
    grad[0]+=B[0]*G+(B[1]*G).T;grad[1]+=B[2]*G+(B[3]*G).T;grad[2]+=B[4]*G
    for x in grad:np.fill_diagonal(x,0)
    return value,terms,grad
