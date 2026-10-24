"""Target-free helper math for the first full-width fourth-core diagnostic."""
import inspect
import numpy as np
from radial_mehler import make_nonlinear


def detailed_pair():
    original=make_nonlinear(6,2,True,True)
    src=inspect.getsource(original)
    # inspect source is a nested definition, strip one indentation level.
    src='\n'.join(line[4:] if line.startswith('    ') else line for line in src.splitlines())
    needle='        return m,cov,post21,radial,aux'
    # After removing the factory indent the body's return has four spaces.
    needle=needle[4:]
    assert src.count(needle)==1
    src=src.replace(needle,'''    D=[r3g/(s*s*s)]+[3*B[r-1] for r in range(1,maxorder+1)]
    p31=pair(D,A)*np.outer(v*s,s)
    post31=p31-3*m[:,None]*pk21+3*(m*m-var)[:,None]*cov
    np.fill_diagonal(post31,post_k4)
    aux['post22']=post22
    aux['post31']=post31
    aux['wick']=[A[r]*s**(1-r) for r in range(7)]
    aux['h']=[B[r]*s**(2-r)-2*m*aux['wick'][r] for r in range(7)]
    return m,cov,post21,radial,aux''')
    env=dict(original.__globals__)
    env.update(gauss_order=6,edge_order=2,local=True,k31=True)
    exec(src,env)
    return env['nonlinear']


PAIR=detailed_pair()


class MixedArray(np.ndarray):
    """Offline new-kernel precision probe, not a flopscope implementation."""
    def __matmul__(self,other):
        a,b=np.asarray(self),np.asarray(other)
        if a.ndim==2 and b.ndim==2 and max(a.shape[0],b.shape[1])>=256 and a.shape[1]>=4:
            result=a.astype(np.float32)@b.astype(np.float32)
            return result.astype(np.float64).view(MixedArray)
        return np.asarray(a@b).view(MixedArray)

    def __rmatmul__(self,other):
        return mixed(other).__matmul__(self)


def mixed(x):
    return np.asarray(x,np.float64).view(MixedArray)


def modes(C,q=4):
    n=len(C);ix=np.arange(n)
    # Four fixed orthonormal Walsh vectors; no target-dependent seeds.
    U=np.column_stack([np.ones(n),1-2*(ix&1),1-2*((ix>>1)&1),
                       (1-2*(ix&1))*(1-2*((ix>>1)&1))])[:,:q]/np.sqrt(n)
    U=np.linalg.qr(U)[0]
    for _ in range(3):U=np.linalg.qr(C@U)[0]
    CU=C@U
    eigen,R=np.linalg.eigh(U.T@CU)
    assert eigen.min()>0
    V=(CU@R)/np.sqrt(eigen)[None,:]
    return U,V


def radial_query(C,U):
    return (np.trace(C)*(C@U)+2*C@(C@U))/3


def core_query(A,C,U):
    return (np.trace(C)*(A@U)+np.trace(A)*(C@U)+2*(A@(C@U)+C@(A@U)))/6
