"""Thin full augmented trace queries, given repeated-plane corrections.

Only costly covariance-edge uses in the triangle/third-input queries use
the supplied low-rank edge model. The non-third repeated correction must
be supplied for the FULL covariance; its triangle adjustment is explicit.
This is not a complete production card or a fourth-core source transport.
"""
import numpy as np


def fused_query(C, K21, K22, A, B, h, wick, modes, third_queries,
                repeated_non_third, X):
    g,d = wick[1],wick[2]
    c0 = (modes*modes).sum(1)
    edge = modes@modes.T
    np.fill_diagonal(edge,0)
    gram_diag = (h[2][:,None]*C*C).sum(0)
    def gram(Z):
        return C.T@(h[2][:,None]*(C@Z))
    out = g[:,None]*gram(g[:,None]*X)-(g*g*gram_diag)[:,None]*X
    dx = d[:,None]*X
    triangle = -(c0*gram_diag)[:,None]*dx
    for v in modes.T:
        triangle += v[:,None]*gram(v[:,None]*dx)
    out += d[:,None]*triangle

    heavy22 = K22/4+C*C/2
    heavy = (h[3][:,None]*(K21*g[None,:]/2+heavy22*d[None,:])
             +h[2][:,None]*K21.T*d[None,:]/2)
    out += heavy.T@(C@(g[:,None]*X))+g[:,None]*(C.T@(heavy@X))
    out -= (2*g*(heavy*C).sum(0))[:,None]*X

    legs = [(1,1,C,1),(2,1,K21/2,2),(1,2,K21.T/2,2),(2,2,heavy22,2)]
    centered = np.zeros_like(C)
    for al,bl,sl,cl in legs:
        projected = sl@h[bl]
        for ar,br,sr,cr in legs:
            if cl+cr <= 3:
                centered += (projected*wick[al+ar])[:,None]*sr*wick[br][None,:]
    centered += centered.T.copy()
    np.fill_diagonal(centered,0)
    out += centered@X

    out += third_queries.fourth_third_part(h[1],h[2],g,d,X)

    ah,bh = float(h[2]@np.diag(A)),float(h[2]@np.diag(B))
    def q0(Z):
        return (ah*(B@Z)+bh*(A@Z)
                +2*(A.T@(h[2][:,None]*(B@Z))+B.T@(h[2][:,None]*(A@Z))))/12
    q0diag = (ah*np.diag(B)+bh*np.diag(A)+4*(h[2][:,None]*A*B).sum(0))/12
    out += g[:,None]*q0(g[:,None]*X)-(g*g*q0diag)[:,None]*X
    av,bv = A.T@h[1],B.T@h[1]
    qi = (np.diag(A)[:,None]*bv[None,:]+np.diag(B)[:,None]*av[None,:]
          +2*av[:,None]*B+2*bv[:,None]*A)/12
    qi *= d[:,None]*g[None,:]
    qi += qi.T.copy()
    np.fill_diagonal(qi,0)
    out += qi@X

    # The non-third correction supplied by the caller uses full covariance.
    # Only the output edge of its Gaussian triangle has been approximated.
    spectator = h[2]*np.diag(C)
    rep_triangle_delta = (spectator[:,None]*C+C*spectator[None,:])*(edge-C)*np.outer(d,d)
    np.fill_diagonal(rep_triangle_delta,0)
    out -= (repeated_non_third+rep_triangle_delta)@X
    return out


def complete_trace_query(fused, postcov, post31, post22, X):
    off = postcov.copy()
    np.fill_diagonal(off,0)
    out = fused-2*off.T@(off@X)+(post31+post31.T)@X
    diagonal_adjust = post22.sum(1)+2*(off*off).sum(0)-2*np.diag(post31)
    return out+diagonal_adjust[:,None]*X
