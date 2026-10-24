import math
import numpy as np
import torch

class Supplier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left=torch.nn.Linear(140,32,bias=False)
        self.right=torch.nn.Linear(140,32,bias=False)
        self.edge=torch.nn.Linear(3,32,bias=False)
        self.bias=torch.nn.Parameter(torch.zeros(32))
        self.output=torch.nn.Linear(32,1)
        self.diagonal=torch.nn.Sequential(torch.nn.Linear(140,32),torch.nn.Tanh(),torch.nn.Linear(32,1))
        for t in (self.left.weight,self.right.weight,self.diagonal[0].weight):torch.nn.init.normal_(t,0.,.15/math.sqrt(140))
        torch.nn.init.normal_(self.edge.weight,0.,.15)
        for t in (self.output.weight,self.output.bias,self.diagonal[0].bias,self.diagonal[2].weight,self.diagonal[2].bias):torch.nn.init.zeros_(t)

    def pair(self,a,b,e):
        # e=[Rij,Rji,Sij,Sji]; directed rho is retained in both orientations.
        h=torch.tanh(self.left(a)+self.right(b)+self.edge(e[:,[0,2,3]])+self.bias)
        k=torch.tanh(self.left(b)+self.right(a)+self.edge(e[:,[1,3,2]])+self.bias)
        return .5*(self.output(h)+self.output(k)).squeeze(-1)

    def diag(self,a):return self.diagonal(a).squeeze(-1)

    def matrix(self,F,R,S,off_scale,diag_scale):
        n=len(F);out=torch.empty((n,n),dtype=F.dtype)
        L=self.left(F);B=self.right(F)
        for lo in range(0,n,64):
            hi=min(lo+64,n)
            h=L[lo:hi,None,:]+B[None,:,:]+self.bias
            h=h+R[lo:hi,:,None]*self.edge.weight[None,None,:,0]
            h=h+S[lo:hi,:,None]*self.edge.weight[None,None,:,1]
            h=h+S[:,lo:hi].T[:,:,None]*self.edge.weight[None,None,:,2]
            out[lo:hi]=self.output(torch.tanh(h)).squeeze(-1)
        D=(out+out.T)*(.5*off_scale)
        D[torch.arange(n),torch.arange(n)]=self.diag(F)*diag_scale
        return D

def numpy_pair(a,b,e,p):
    left=a@p['left.weight'].T;right=b@p['right.weight'].T
    h=np.tanh(left+right+e[:,[0,2,3]]@p['edge.weight'].T+p['bias'])
    k=np.tanh(b@p['left.weight'].T+a@p['right.weight'].T+e[:,[1,3,2]]@p['edge.weight'].T+p['bias'])
    return (.5*((h+k)@p['output.weight'].T)+p['output.bias'])[:,0]

def numpy_diag(a,p):
    h=np.tanh(a@p['diagonal.0.weight'].T+p['diagonal.0.bias'])
    return (h@p['diagonal.2.weight'].T+p['diagonal.2.bias'])[:,0]
