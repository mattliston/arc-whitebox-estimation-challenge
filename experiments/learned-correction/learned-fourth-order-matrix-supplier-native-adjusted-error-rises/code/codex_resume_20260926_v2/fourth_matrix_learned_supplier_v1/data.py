from common import *
import sys
import functools
import numpy as np
sys.path[:0]=[str(PROJECTION),str(ASTRA)]
from projection import project,faces,vector
from current_pair_oracle_math_v1 import exact_change,independent_fourth_map

@functools.lru_cache(maxsize=1)
def references(ordinal):
    assert ordinal in TRAIN+VALID
    if ordinal in VALID:
        frozen=read(HERE/'model_frozen.json');check(frozen['model']);check(frozen['normalizers'])
    cr=read(OLD/f'capture_mini{ordinal:04d}.json');ir=read(OLD/f'inputs_mini{ordinal:04d}.json')
    tr=read(JOINT/f'targets_mini{ordinal:04d}.json');dr=read(OLD/f'download_mini{ordinal:04d}.json')
    assert all(r['passed'] for r in (cr,ir,tr,dr))
    check(cr['capture']);check(dr['payload'])
    with np.load(dr['payload']['path'],allow_pickle=False) as z:
        raw={k:z[k] for k in ('pre_mean','pre_m2','pre_m3','pre_m4')}
    return cr,ir,tr,dr,raw

def cell(ordinal,layer):
    cr,ir,tr,dr,raw=references(ordinal)
    ip=next(r['input'] for r in ir['records'] if r['layer']==layer)
    tp=next(r['target'] for r in tr['records'] if r['layer']==layer)
    check(ip);check(tp)
    with np.load(ip['path'],allow_pickle=False) as z:
        c={k:z[k] for k in ('F','R','S','sigma','Wnext','gnext')}
    with np.load(cr['capture']['path'],allow_pickle=False) as z:
        c.update({k:z[f'{k}_{layer:02d}'] for k in ('C','K','mu','radial','alpha','pdf','gate')})
        c['gate_next']=z[f'gate_{layer+1:02d}']
        assert np.array_equal(c['sigma'],z[f'sigma_{layer:02d}'])
    s=c['sigma'];v=np.diag(c['C']);radial=c['radial'].item()
    assert np.all(v>=1e-10)
    rawR=c['C']/np.outer(s,s);R=(rawR+rawR.T)/2
    assert np.max(np.abs(np.diag(R)-1))<=1e-10;np.fill_diagonal(R,1.)
    m1,m2,m3,m4=[raw[k][layer] for k in ('pre_mean','pre_m2','pre_m3','pre_m4')]
    k4=m4-4*m1*m3+6*m1*m1*m2-3*m1**4-3*(m2-m1*m1)**2
    dk4=(k4-radial*v*v)/s**4
    t=np.load(tp['path'],allow_pickle=False);q31=t[1].copy();q22=t[2].copy()
    D,info=project(R,dk4,q31,(q22+q22.T)/2)
    c.update(projected_D=D,projection_R=R,raw_R=rawR,targets=(dk4,q31,q22),projection_info=info,
             external_pins=[cr['capture'],dr['payload'],ip,tp],ordinal=ordinal,layer=layer)
    return c

def assess(c,D):
    fn=independent_fourth_map();s=c['sigma'];v=np.diag(c['C']);radial=c['radial'].item()
    k4=radial*v*v;k22=radial*(np.outer(v,v)+2*c['C']**2)/3
    q31=radial*c['raw_R'];np.fill_diagonal(q31,0.)
    m,C,*_=fn(c['mu'],c['C'],c['K'],k4,k22,'covariance',q31_override=q31)
    mg=c['mu']*c['gate']+s*c['pdf']
    common=(c['raw_R'],s,c['alpha'],c['pdf'],c['gate'],mg/s,m/s)
    predicted=faces(c['projection_R'],D);values={};errors=[];W=c['Wnext'].astype(np.float64)
    for name,planes in (('target',c['targets']),('prediction',predicted)):
        d4,d31,d22=[np.array(z,copy=True) for z in planes]
        fv=vector(d4,d31,(d22+d22.T)/2)
        np.fill_diagonal(d31,0.);np.fill_diagonal(d22,0.)
        dm,dc=exact_change(common,dQ22=d22,dQ31=d31,dq4=d4)
        mm,cc,*_=fn(c['mu'],c['C'],c['K'],k4+d4*s**4,k22+d22*np.outer(s*s,s*s),
                   'covariance',q31_override=q31+d31)
        err=max(float(np.max(np.abs(mm-m-dm))),float(np.max(np.abs((cc-C+cc.T-C.T)/2-dc))))
        assert err<=2e-11,(c['ordinal'],c['layer'],name,err);errors.append(err)
        nv=np.sum(W*(dc@W),axis=0)*c['gnext']
        nt=nv+c['gate_next']*(W.T@dm)
        values[name]=dict(joint_faces=fv,mean=dm,covariance=dc,next_variance_mean=nv,next_total_mean=nt)
    metrics={}
    for name in values['target']:
        y=values['target'][name];p=values['prediction'][name]
        base=float(np.sum(y*y));err=float(np.sum((y-p)**2))
        assert np.isfinite(base+err) and base>0
        metrics[name]=dict(baseline_energy=base,residual_energy=err,ratio=err/base)
    return metrics,max(errors)
