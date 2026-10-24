"""Independent binomial centering, directed r-loop and full-congruence judgment."""
import math,resource,time
from common import *
import numpy as np
from threadpoolctl import threadpool_info
from head import predict_planes
from data import load_cell
verify();verify('fit_sources_sealed.json');verify('replay_sources_sealed.json')
assert all(x['num_threads']==1 for x in threadpool_info());start=time.monotonic()
f=read(HERE/'model_frozen.json');r=read(HERE/'information_result.json');assert f['training_completed'] and r['completed']
for key in ('model','normalizers','training'):check(f[key])
t=dict(np.load(f['model']['path'],allow_pickle=False));energy=np.load(f['normalizers']['path'],allow_pickle=False)['energies'];records=[]
for ordinal in VALID:
    dr=read(OLD/f'download_mini{ordinal:04d}.json');check(dr['payload'])
    with np.load(dr['payload']['path'],allow_pickle=False) as z:raw={k:z[k] for k in ('pre_mean','pre_m2','pre_m3','pre_m4','pre_M11','pre_M21','pre_M31','pre_M22')}
    caprec=read(OLD/f'capture_mini{ordinal:04d}.json');check(caprec['capture']);caps=np.load(caprec['capture']['path'],allow_pickle=False)
    original=next(x for x in r['records'] if x['ordinal']==ordinal);normalized=[];layers=[]
    for l in LAYERS:
        c=load_cell(ordinal,l);mu=raw['pre_mean'][l];raw11=raw['pre_M11'][l].astype(np.float64)
        moments={(0,0):1.,(1,0):mu[:,None],(0,1):mu[None,:],(2,0):raw['pre_m2'][l][:,None],(0,2):raw['pre_m2'][l][None,:],
                 (3,0):raw['pre_m3'][l][:,None],(0,3):raw['pre_m3'][l][None,:],(1,1):raw11,(2,1):raw['pre_M21'][l].astype(np.float64),
                 (1,2):raw['pre_M21'][l].T.astype(np.float64),(3,1):raw['pre_M31'][l].astype(np.float64),(2,2):raw['pre_M22'][l].astype(np.float64)}
        def centered(a,b):
            value=np.zeros((1024,1024),np.float64)
            for i in range(a+1):
                for j in range(b+1):value+=math.comb(a,i)*math.comb(b,j)*(-mu[:,None])**(a-i)*(-mu[None,:])**(b-j)*moments[i,j]
            return value
        Ct=raw11-np.outer(mu,mu);vt=raw['pre_m2'][l]-mu*mu;np.fill_diagonal(Ct,vt)
        K21=centered(2,1);K31=centered(3,1)-3*vt[:,None]*Ct;K22=centered(2,2)-np.outer(vt,vt)-2*Ct*Ct
        C=caps[f'C_{l:02d}'];v=np.diag(C);s=caps[f'sigma_{l:02d}'];a=caps[f'alpha_{l:02d}'];p=caps[f'pdf_{l:02d}'];g=caps[f'gate_{l:02d}'];radial=caps[f'radial_{l:02d}'].item()
        R=C/np.outer(s,s);k4=radial*v*v
        targets=np.stack([(K21-caps[f'K_{l:02d}'])/(v[:,None]*s[None,:]),K31/(s[:,None]**3*s[None,:])-(k4/(v*v))[:,None]*R,
                          K22/np.outer(v,v)-(radial*(np.outer(v,v)+2*C*C)/3)/np.outer(v,v)])
        for x in targets:np.fill_diagonal(x,0)
        target_errors=[float(np.linalg.norm(x-y)/np.linalg.norm(y)) for x,y in zip(targets,c['target'])];assert max(target_errors)<=1e-9,target_errors
        pred=np.stack(predict_planes(np,c['F'],c['R'],c['S'],t)).astype(np.float64);A=[g,p,-a*p,(a*a-1)*p,-(a**3-3*a)*p]
        np.fill_diagonal(R,0);terms=[];field_errors=[];W=c['Wnext'].astype(np.float64)
        for e in (-c['target'],pred-c['target']):
            cov=np.zeros_like(C)
            for k in range(3):
                part=(e[0]*np.outer(A[k+1],A[k])+e[0].T*np.outer(A[k],A[k+1]))/2
                part+=(e[1]*np.outer(A[k+2],A[k])+e[1].T*np.outer(A[k],A[k+2]))/6
                part+=e[2]*np.outer(A[k+1],A[k+1])/4
                cov+=(R**k/math.factorial(k))*part
            cov*=np.outer(s,s);np.fill_diagonal(cov,0)
            response=np.diag(W.T@cov@W)*c['gnext']
            values=[float(np.dot(x.ravel(),(w*x).ravel()))/(1024*1023) for x,w in zip(e,c['weight'])]
            values.extend([float(np.vdot(cov,cov))/(1024*1023),math.fsum(float(x)*float(x) for x in response)/1024]);terms.append(values)
        terms=np.array(terms);old=next(x for x in original['layers'] if x['layer']==l);expected=np.array([old['zero_energies'],old['model_energies']])
        loss_error=float(np.max(np.abs(terms-expected)/expected));assert loss_error<=5e-11,loss_error
        normalized.append(terms/energy[l-8]);layers.append(dict(layer=l,target_relative_errors=target_errors,loss_relative_error=loss_error))
    totals=np.array([[math.fsum(float(x[arm,k]) for x in normalized) for k in range(5)] for arm in range(2)])
    ratios=totals[1]/totals[0];relative=float(np.max(np.abs(ratios-original['ratios'])/ratios));assert relative<=5e-11
    passed=bool(np.all(ratios[3:]<=[.9,.9]));assert passed==original['passed']
    records.append(dict(ordinal=ordinal,ratios=ratios.tolist(),passed=passed,relative=relative,layers=layers));caps.close();del raw
    print(records[-1],flush=True)
assert all(x['passed'] for x in records)==r['passed']
write(HERE/'replay_result.json',dict(passed=True,information_gate_passed=r['passed'],records=records,model=f['model'],no_refit=True,
      public1B_labels_decoded=False,wall_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
