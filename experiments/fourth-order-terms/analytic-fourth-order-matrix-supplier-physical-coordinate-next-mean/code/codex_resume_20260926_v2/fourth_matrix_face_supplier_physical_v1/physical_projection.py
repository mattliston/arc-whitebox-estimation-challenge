"""Canonical repeated-face least squares for an arbitrary signed metric."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve


def faces(C,D):
 v=np.diag(C);a=np.diag(D)
 return v*a,(v[:,None]*D+a[:,None]*C)/2,(np.outer(v,a)+np.outer(a,v)+4*C*D)/6


def project(C,k4,q31,q22):
 n=len(C);v=np.diag(C);i,j=np.triu_indices(n,1)
 assert np.max(np.abs(C-C.T))<1e-12 and np.max(np.abs(q22-q22.T))<1e-12
 r=C[i,j];vi=v[i];vj=v[j]
 A,B,H=2*q31[i,j],2*q31[j,i],6*q22[i,j]
 h=vi*vi+vj*vj+(8/3)*r*r
 assert np.all(h>0)
 ua=r*(vi+(2/3)*vj);ub=r*(vj+(2/3)*vi)
 rx=vi*A+vj*B+(2/3)*r*H
 da=r*r+vj*vj/6-ua*ua/h
 db=r*r+vi*vi/6-ub*ub/h
 e=vi*vj/6-ua*ub/h
 ra=r*A+vj*H/6-ua*rx/h
 rb=r*B+vi*H/6-ub*rx/h
 M=np.zeros((n,n));M[i,j]=e;M[j,i]=e
 np.fill_diagonal(M,v*v+np.bincount(i,weights=da,minlength=n)+np.bincount(j,weights=db,minlength=n))
 rhs=v*k4+np.bincount(i,weights=ra,minlength=n)+np.bincount(j,weights=rb,minlength=n)
 factor=cho_factor(M,lower=True,check_finite=True)
 a=cho_solve(factor,rhs,check_finite=True)
 x=(rx-ua*a[i]-ub*a[j])/h
 D=np.diag(a);D[i,j]=x;D[j,i]=x
 error=float(np.linalg.norm(M@a-rhs)/max(np.linalg.norm(rhs),1e-300))
 assert error<1e-10
 return D,dict(solve_relative_residual=error,cholesky_diagonal_min=float(np.diag(factor[0]).min()),
               eliminated_offdiagonal_denominator_min=float(h.min()),diagonal_unknowns=n,offdiagonal_unknowns=len(i))
