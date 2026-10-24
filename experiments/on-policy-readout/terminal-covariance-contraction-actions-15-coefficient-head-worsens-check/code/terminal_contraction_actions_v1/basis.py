"""Fixed local and nonlocal normalized covariance actions; no labels."""
import numpy as np

NAMES=('R_E3','S_E3','ST_E3','R_E4','S_E4',
       'RR_E3','RR_E4','RS_E3','RS_E4','RST_E3','RST_E4','SST_E3','SST_E4','RS_T_E3','RST_T_E3')

def matrices(C,K,s):
    R=((C+C.T)/2)/np.outer(s,s);S=K/np.outer(s*s,s)
    np.fill_diagonal(R,0);np.fill_diagonal(S,0)
    n=len(s)
    return R,S,(R@R/n,R@S/n,R@S.T/n,S@S.T/n)

def actions(C,K,s,g,pdf):
    R,S,extra=matrices(C,K,s);E3=np.outer(pdf,g);E4=np.outer(pdf,pdf);ss=np.outer(s,s)
    pairs=[(R,E3),(S,E3),(S.T,E3),(R,E4),(S,E4)]
    pairs += [(M,E) for M in extra for E in (E3,E4)]
    pairs += [(extra[1].T,E3),(extra[2].T,E3)]
    for name,(M,E) in zip(NAMES,pairs):
        D=M*E;D=(D+D.T)*(.5*ss);np.fill_diagonal(D,0)
        yield name,D
