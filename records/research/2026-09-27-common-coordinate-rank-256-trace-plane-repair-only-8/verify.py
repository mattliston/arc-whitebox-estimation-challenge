"""Independent polarized-pair replay, without importing the analysis program."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

HERE=Path(__file__).resolve().parent
record=json.loads((HERE/'result.json').read_text())
for pin in json.loads((HERE/'seal.json').read_text())['files']:
    assert hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest()==pin['sha256']
errors=[]
for row in record['rows']:
    z=np.load(row['path']);U=z['U'].astype('float64');lam=z['lam'].astype('float64')
    local=z['local_rows'];alpha=z['alpha'];sigma=z['sigma'];gate=z['gate'];alive=z['alive']
    tails=set(map(int,z['tail']))
    scores=[min(float(gate[i]),1-float(gate[i])) if alive[i] else -1. for i in range(1024)]
    n=min(1024,32*((sum(x>=.01 for x in scores)+31)//32))
    chosen=sorted(range(1024),key=lambda i:(-scores[i],i))[:n]
    maprow={int(i):j for j,i in enumerate(local)}
    ii=np.array([maprow[i] for i in sorted(chosen) if i in maprow and i not in tails])
    jj=np.array([maprow[i] for i in range(1024) if alive[i] and i in maprow])
    keep=np.array([j for j,i in enumerate(local) if int(i) not in tails])
    assert (len(ii),len(jj),len(keep))==(row['response_rows'],row['live_columns'],row['kept_rows'])
    full=np.zeros((len(ii),len(jj)));short=full.copy();repair=full.copy()
    df=np.zeros(len(keep));ds=df.copy();dr=df.copy();trace_error=0.
    split=int(z['split'])
    blocks=[(U,z['F'],0,len(lam))] if split<0 else [
        (U@z['mapping'].astype('float64'),z['base'],0,split),
        (U,z['join'],split,len(lam))]
    for X,Fraw,lo,hi in blocks:
        F=Fraw.astype('float64');t=np.zeros(F.shape[0]);t_reference=np.zeros_like(t)
        for start,width in z['cohorts']:
            start,width=int(start),int(width)
            if not width or start<lo or start+width>hi:continue
            h=width//2;a=start-lo
            coeff=lam[start:start+h]
            assert np.array_equal(coeff,-lam[start+h:start+width])
            u=(F[:,a:a+h]+F[:,a+h:a+width])/2
            v=(F[:,a:a+h]-F[:,a+h:a+width])/2
            u0=u.copy();u0[256:]=0
            ut=u-u0
            t += 4*(u0@(coeff*np.sum(ut*v,axis=0)) + ut@(coeff*np.sum(u*v,axis=0))) + 2*v@(coeff*np.sum(ut*ut,axis=0))
            f0=u0+v;f1=u0-v
            t_reference+=F[:,a:a+h]@(coeff*np.sum(F[:,a:a+h]**2,axis=0))
            t_reference-=F[:,a+h:a+width]@(coeff*np.sum(F[:,a+h:a+width]**2,axis=0))
            t_reference-=f0@(coeff*np.sum(f0*f0,axis=0))-f1@(coeff*np.sum(f1*f1,axis=0))
            pv=X@v
            for ub,out,dout in ((u,full,df),(u0,short,ds)):
                pu=X@ub
                out+=(pu[ii]**2+pv[ii]**2)@(2*coeff*pv[jj]).T
                out+=(pu[ii]*pv[ii])@(4*coeff*pu[jj]).T
                dout+=np.sum((6*pu[keep]**2*pv[keep]+2*pv[keep]**3)*coeff,axis=1)
        trace_error=max(trace_error,float(np.max(abs(t-t_reference))))
        np.testing.assert_allclose(t,t_reference,rtol=5e-11,atol=5e-12)
        # Build the rank-two expression by explicit outer products and Gram.
        radius=np.einsum('ij,ij->i',X[ii],X[ii]);values=X@t
        repair+=(2*np.diag(values[ii])@(X[ii]@X[jj].T)+np.outer(radius,values[jj]))/(X.shape[1]+2)
        dr+=3*np.einsum('ij,ij->i',X[keep],X[keep])*values[keep]/(X.shape[1]+2)
    pdf=np.array([math.exp(-float(a)**2/2)/math.sqrt(2*math.pi) for a in alpha])
    w=np.outer(pdf[local[ii]]/sigma[local[ii]],gate[local[jj]])
    for a,global_i in enumerate(local[ii]):
        w[a,local[jj]==global_i]=0
    wd=-alpha[local[keep]]*pdf[local[keep]]/(6*sigma[local[keep]]**2)
    den=float(np.sum((w*full)**2));d_den=float(np.sum((wd*df)**2))
    proposed=dict(plane_den=den,plane_error_before=float(np.sum((w*(short-full))**2)),
        plane_error_after=float(np.sum((w*(short+repair-full))**2)),
        diagonal_relative_before=float(np.sum((wd*(ds-df))**2))/d_den,
        diagonal_relative_after=float(np.sum((wd*(ds+dr-df))**2))/d_den)
    max_rel=0.
    for key,value in proposed.items():
        rel=abs(value-row[key])/max(abs(row[key]),1e-30);max_rel=max(max_rel,rel)
        assert rel<2e-9,(key,value,row[key],rel)
    errors.append(dict(rank=row['rank'],layer=row['layer'],trace_max_error=trace_error,score_max_relative_error=max_rel))
    print(json.dumps(errors[-1]),flush=True)
ratio=math.fsum(r['plane_error_after'] for r in record['rows'])/math.fsum(r['plane_error_before'] for r in record['rows'])
assert abs(ratio-record['aggregate_repair_error_ratio'])<2e-15
assert record['gates']==dict(absolute=True,repair=False) and not record['passed']
with (HERE/'verification.json').open('x') as f:json.dump(dict(passed=True,cells=errors,ratio=ratio),f,indent=2)
print('INDEPENDENT_REPLAY_PASS',flush=True)
