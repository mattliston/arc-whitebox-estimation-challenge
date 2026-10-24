"""Exact pair kernel and fixed weighted endpoint plane ridge operator."""
import numpy as np
from radial_mehler import derivatives
from fourth_plane_pair_span_v1 import PlaneOperator


def third_covariance_kernel(R,sigma,alpha,pdf,gate,gaussian_mean_over_sigma):
    A,_=derivatives(alpha,pdf,gate,gaussian_mean_over_sigma,6)
    rho=R.copy();np.fill_diagonal(rho,0)
    rr=np.ones_like(rho);kernel=np.zeros_like(rho)
    for r in range(3):
        kernel += .5*(rr+rr.T)/2*np.outer(A[r+2],A[r+1])
        rr=rr*rho/(r+1)
    kernel *= np.outer(sigma,sigma)
    np.fill_diagonal(kernel,0)
    return kernel


def input_basis(state,R):
    raw=np.column_stack((np.ones(len(state)),state.astype(np.float64)))
    U,s,V=np.linalg.svd(raw,full_matrices=False)
    rank=int(np.count_nonzero(s>s[0]*1e-12))
    H=U[:,:rank]*np.sqrt(len(raw))
    reconstruction=float(np.linalg.norm(H@(H.T@raw)/len(H)-raw)/np.linalg.norm(raw))
    assert reconstruction<=1e-10
    R=(R+R.T)/2
    powers=np.stack((np.ones_like(R),R,R*R))
    for r in powers:np.fill_diagonal(r,0)
    rms=np.sqrt(np.mean(powers*powers,axis=(1,2)))
    powers/=rms[:,None,None]
    return H,powers,dict(original_features=65,numerical_rank=rank,singular_values=s.tolist(),
                         reconstruction_relative=reconstruction,correlation_power_rms=rms.tolist())


class WeightedPlaneOperator(PlaneOperator):
    def __init__(self,H,powers,weights2,ridge=.001):
        self.H,self.powers,self.weights2=H,powers,weights2
        self.symmetric,self.ridge=False,ridge
        self.shape=(len(powers),H.shape[1],H.shape[1])
        self.scale=float(len(H)**2)
        square=H*H
        self.preconditioner=np.stack([square.T@(r*r*weights2)@square/self.scale+ridge for r in powers])

    def normal(self,theta):
        return self.adjoint(self.weights2*self.forward(theta))/self.scale+self.ridge*theta

    def fit(self,target,tolerance=1e-7,max_iterations=300):
        return super().fit(self.weights2*target,tolerance,max_iterations)
