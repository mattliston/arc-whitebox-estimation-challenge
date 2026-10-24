"""Frobenius projection to symmetric, zero-diagonal, mean-orthogonal matrices."""
import torch

def project(A,m):
    m=m/torch.linalg.vector_norm(m)
    d=1-2*m*m
    assert bool(torch.all(d>0))
    A=(A+A.T)/2
    A=A-torch.diag_embed(torch.diagonal(A))
    b=A@m;v=m/d
    u=2*(b/d-v*torch.dot(m,b/d)/(1+torch.dot(m,v)))
    D=A-(torch.outer(u,m)+torch.outer(m,u))/2
    D=D-torch.diag_embed(torch.diagonal(D))
    return D
