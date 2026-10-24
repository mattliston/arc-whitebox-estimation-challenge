import numpy as np
def central(mu,m2,m3,m4,M11,M21,M31,M22):
    M11=M11.astype(np.float64);M21=M21.astype(np.float64)
    C=M11-np.outer(mu,mu);v=m2-mu*mu;np.fill_diagonal(C,v)
    K21=M21-2*mu[:,None]*M11-m2[:,None]*mu[None,:]+2*mu[:,None]**2*mu[None,:]
    K31=M31.astype(np.float64)-m3[:,None]*mu[None,:]-3*mu[:,None]*M21+3*mu[:,None]*mu[None,:]*m2[:,None]
    K31+=3*mu[:,None]**2*M11-3*mu[:,None]**3*mu[None,:]-3*v[:,None]*C
    K22=M22.astype(np.float64)-2*mu[None,:]*M21-2*mu[:,None]*M21.T
    K22+=mu[None,:]**2*m2[:,None]+mu[:,None]**2*m2[None,:]+4*np.outer(mu,mu)*M11-3*np.outer(mu*mu,mu*mu)
    K22-=np.outer(v,v)+2*C*C
    k4=m4-4*mu*m3+6*mu*mu*m2-3*mu**4-3*v*v
    np.fill_diagonal(K31,k4);np.fill_diagonal(K22,k4)
    return K21,K31,K22,C,v

def storage_bounds(mu,m2,M11,M21,M31,M22):
    def half(x):return .5*np.abs(np.spacing(x.astype(np.float32))).astype(np.float64)
    b11,b21,b31,b22=map(half,(M11,M21,M31,M22));v=m2-mu*mu;C=M11.astype(np.float64)-np.outer(mu,mu)
    B31=b31+3*np.abs(mu[:,None])*b21+(3*mu[:,None]**2+3*np.abs(v[:,None]))*b11
    B22=b22+2*np.abs(mu[None,:])*b21+2*np.abs(mu[:,None])*b21.T
    B22+=(4*np.abs(np.outer(mu,mu))+4*np.abs(C))*b11+2*b11*b11
    np.fill_diagonal(B31,0);np.fill_diagonal(B22,0)
    return B31,B22

def gaussian_fourth_variances(C,count):
    v=np.diag(C);return (6*v[:,None]**3*v[None,:]+18*v[:,None]**2*C*C)/count,(
        4*np.outer(v*v,v*v)+16*np.outer(v,v)*C*C+4*C**4)/count
