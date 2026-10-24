"""Independent observer replay, with a dense projection multiplier solve."""
import numpy as np


def project_dense(A,m):
    m=m/np.linalg.norm(m);A=(A+A.T)/2;np.fill_diagonal(A,0)
    system=np.diag(1-2*m*m)+np.outer(m,m)
    u=np.linalg.solve(system,2*(A@m))
    D=A-(np.outer(u,m)+np.outer(m,u))/2;np.fill_diagonal(D,0)
    return D


def source(P,a,norm,U):
    s=a['sigma'];F=np.clip(a['F']/norm['feature_scale'],-8,8)
    R=(a['C']/np.outer(s,s)).astype(F.dtype)/norm['rho_scale']
    S=(a['K']/np.outer(s*s,s)).astype(F.dtype)/norm['third_scale']
    memory=(U/(.002*np.outer(s,s))).astype(F.dtype)
    L=F@P['left.weight'].T;B=F@P['right.weight'].T;parts=[]
    for lo in range(0,len(F),64):
        hi=min(lo+64,len(F));h=L[lo:hi,None,:]+B[None,:,:]+P['bias']
        h=h+R[lo:hi,:,None]*P['edge.weight'][:,0]
        h=h+S[lo:hi,:,None]*P['edge.weight'][:,1]
        h=h+S[:,lo:hi].T[:,:,None]*P['edge.weight'][:,2]
        h=h+memory[lo:hi,:,None]*P['edge.weight'][:,3]
        parts.append((np.tanh(h)@P['output.weight'].T+P['output.bias']).squeeze(-1))
    raw=np.concatenate(parts)
    return (raw+raw.T).astype(float)*(.001*np.outer(s,s))


def replay(P,cells,terminal,norm,mode):
    assert mode in ('history','reset');D=np.zeros_like(cells[0]['C']);e=np.zeros_like(cells[0]['m']);states=[]
    for index,a in enumerate(cells):
        if index:
            W=a['W'].astype(float);U=W.T@D@W
            e=a['gate']*(W.T@e)+a['pdf']/(2*a['sigma'])*np.diag(U)
        else:U=np.zeros_like(D)
        if mode=='history':carry=np.outer(a['gate'],a['gate'])*U;memory=U
        else:carry=np.zeros_like(U);memory=np.zeros_like(U)
        D=project_dense(carry+source(P,a,norm,memory),a['m']);states.append(D)
    W=terminal['W'].astype(float);U=W.T@D@W
    e=terminal['gate']*(W.T@e)+terminal['pdf']/(2*terminal['sigma'])*np.diag(U)
    return states,e,np.maximum(terminal['parent']+e,0)
