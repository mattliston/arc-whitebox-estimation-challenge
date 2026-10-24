import sys
import numpy as np
from common import *
sys.path[:0]=[str(PROJECTION),str(ASTRA)]
from projection import project
from current_pair_oracle_math_v1 import exact_change,independent_fourth_map

def network(ordinal):
    cr=read(OLD/f'capture_mini{ordinal:04d}.json');ir=read(OLD/f'inputs_mini{ordinal:04d}.json')
    tr=read(JOINT/f'targets_mini{ordinal:04d}.json');dr=read(OLD/f'download_mini{ordinal:04d}.json')
    assert all(x['passed'] for x in (cr,ir,tr,dr))
    for p in (cr['capture'],cr['weights'],dr['payload']):check(p)
    with np.load(dr['payload']['path'],allow_pickle=False) as z:raw={k:z[k] for k in ('pre_mean','pre_m2','pre_m3','pre_m4')}
    return cr,ir,tr,dr,raw

def cell(info,l):
    cr,ir,tr,dr,raw=info
    ip=next(r['input'] for r in ir['records'] if r['layer']==l);tp=next(r['target'] for r in tr['records'] if r['layer']==l)
    check(ip);check(tp)
    with np.load(ip['path']) as z:c={k:z[k] for k in ('sigma','Wnext','gnext')}
    with np.load(cr['capture']['path']) as z:
        c.update({k:z[f'{k}_{l:02d}'] for k in ('C','K','mu','radial','alpha','pdf','gate')})
        c['gate_next']=z[f'gate_{l+1:02d}']
    s=c['sigma'];v=np.diag(c['C']);radial=c['radial'].item();assert np.all(v>=1e-10) and np.all(s>0)
    R=c['C']/np.outer(s,s);Rs=(R+R.T)/2;np.fill_diagonal(Rs,1.)
    m1,m2,m3,m4=[raw[k][l] for k in ('pre_mean','pre_m2','pre_m3','pre_m4')]
    true3=m3-3*m1*m2+2*m1**3
    true4=m4-4*m1*m3+6*m1*m1*m2-3*m1**4-3*(m2-m1*m1)**2
    k4=(true4-radial*v*v)/s**4
    t=np.load(tp['path']);S=t[0]*(v/(s*s))[:,None];q31=t[1].copy();q22=t[2].copy();del t
    np.fill_diagonal(S,(true3-np.diag(c['K']))/s**3)
    D,projection=project(Rs,k4,q31,(q22+q22.T)/2)
    c.update(D=D,Rs=Rs,R=R,S=S,k4=k4,q31=q31,q22=q22,projection=projection,
             pins=[cr['capture'],cr['weights'],dr['payload'],ip,tp],layer=l)
    return c

def consumer(c):
    fn=independent_fourth_map();s=c['sigma'];v=np.diag(c['C']);radial=c['radial'].item()
    base4=radial*v*v;base22=radial*(np.outer(v,v)+2*c['C']**2)/3
    base31=radial*c['R'];np.fill_diagonal(base31,0.)
    m,baseC,*_=fn(c['mu'],c['C'],c['K'],base4,base22,'covariance',q31_override=base31)
    mg=c['mu']*c['gate']+s*c['pdf'];common=(c['R'],s,c['alpha'],c['pdf'],c['gate'],mg/s,m/s)
    def apply(S=None,q4=None,q31=None,q22=None):
        dq31=None if q31 is None else q31.copy();dq22=None if q22 is None else q22.copy()
        if dq31 is not None:np.fill_diagonal(dq31,0.)
        if dq22 is not None:np.fill_diagonal(dq22,0.)
        dm,dc=exact_change(common,dS=S,dQ22=dq22,dQ31=dq31,dq4=q4)
        K=c['K'] if S is None else c['K']+S*np.outer(s*s,s)
        k4=base4 if q4 is None else base4+q4*s**4
        k22=base22 if dq22 is None else base22+dq22*np.outer(s*s,s*s)
        T31=base31 if dq31 is None else base31+dq31
        mm,cc,*_=fn(c['mu'],c['C'],K,k4,k22,'covariance',q31_override=T31)
        error=max(float(np.max(np.abs(mm-m-dm))),float(np.max(np.abs((cc-baseC+cc.T-baseC.T)/2-dc))))
        assert error<=2e-11,(c['layer'],error)
        W=c['Wnext'].astype(np.float64);nv=np.sum(W*(dc@W),axis=0)*c['gnext'];nt=nv+c['gate_next']*(W.T@dm)
        return dict(mean=dm,covariance=dc,next_variance_mean=nv,next_total_mean=nt),error
    return apply
