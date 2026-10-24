"""FP64 information screen, with an independently formed tiny tensor oracle."""
from pathlib import Path
import hashlib
import json
import math
import sys
import numpy as np

HERE = Path(__file__).resolve().parent

def truncate(F, cohorts, lo, hi, ru):
    out = np.empty_like(F)
    covered=0
    for start,width in cohorts:
        start,width=int(start),int(width)
        if width and start >= lo and start+width <= hi:
            a=start-lo; h=width//2
            assert width%2==0
            u=(F[:,a:a+h]+F[:,a+h:a+width])*0.5
            v=(F[:,a:a+h]-F[:,a+h:a+width])*0.5
            u[ru:]=0
            out[:,a:a+h]=u+v;out[:,a+h:a+width]=u-v
            covered+=width
    assert covered==hi-lo
    return out

def trace(F,lam):
    return F @ (lam*np.sum(F*F,axis=0))

def correction(U,t,ii,jj):
    x,y=U[ii],U[jj]
    tx,ty=x@t,y@t
    norm=np.sum(x*x,axis=1)
    return (2*(x@y.T)*tx[:,None]+norm[:,None]*ty[None,:])/(U.shape[1]+2)

def unit():
    rng=np.random.default_rng(2026092745)
    reports=[]
    for r in (3,7):
        F=rng.normal(size=(r,10)); lam=np.r_[np.ones(3),-np.ones(3),[.3,.7,-.3,-.7]]
        cohorts=np.asarray([[0,6],[6,4]])
        G=truncate(F,cohorts,0,10,2)
        T=np.zeros((r,r,r));T0=T.copy()
        for c in range(10):
            for a in range(r):
                for b in range(r):
                    for d in range(r):
                        T[a,b,d]+=lam[c]*F[a,c]*F[b,c]*F[d,c]
                        T0[a,b,d]+=lam[c]*G[a,c]*G[b,c]*G[d,c]
        lost=T-T0;t=trace(F,lam)-trace(G,lam)
        direct=np.array([sum(lost[a,b,b] for b in range(r)) for a in range(r)])
        np.testing.assert_allclose(t,direct,atol=2e-13,rtol=2e-13)
        Q=np.zeros_like(T)
        for a in range(r):
            for b in range(r):
                for c in range(r):Q[a,b,c]=(t[a]*(b==c)+t[b]*(a==c)+t[c]*(a==b))/(r+2)
        np.testing.assert_allclose(np.einsum('abb->a',Q),t,atol=2e-13,rtol=2e-13)
        assert abs(np.sum((lost-Q)*Q)) < 1e-10
        U=rng.normal(size=(8,r));ii=np.asarray([1,5,2]);jj=np.asarray([6,0,1,4])
        expected=np.einsum('ia,ib,jc,abc->ij',U[ii],U[ii],U[jj],Q)
        np.testing.assert_allclose(correction(U,t,ii,jj),expected,rtol=2e-13,atol=2e-13)
        diag=3*np.sum(U*U,axis=1)*(U@t)/(r+2)
        np.testing.assert_allclose(diag,np.einsum('ia,ib,ic,abc->i',U,U,U,Q),rtol=2e-13,atol=2e-13)
        # Independent sum of two associated blocks, including a nonorthogonal map.
        V=U@rng.normal(size=(r,r)); expected2=np.einsum('ia,ib,jc,abc->ij',V[ii],V[ii],V[jj],Q)
        np.testing.assert_allclose(correction(U,t,ii,jj)+correction(V,t,ii,jj),expected+expected2,rtol=4e-13,atol=2e-12)
        reports.append(dict(rank=r,trace_max_error=float(np.max(abs(t-direct))),plane_max_error=float(np.max(abs(correction(U,t,ii,jj)-expected)))))
    return reports

def cell(path):
    z=np.load(path);U=z['U'].astype(np.float64);lam=z['lam'].astype(np.float64)
    local=z['local_rows'];alpha=z['alpha'];sig=z['sigma'];gate=z['gate'];alive=z['alive'];tail=z['tail']
    # Reproduce the parent's select() padded32 roster via its scalar specification.
    score=np.where(alive,np.minimum(gate,1-gate),-1.)
    count=min(1024,((int(np.count_nonzero(score>=.01))+31)//32)*32)
    response=np.sort(np.argsort(-score,kind='stable')[:count])
    inv={int(x):i for i,x in enumerate(local)}
    ii=np.array([inv[int(i)] for i in response if int(i) in inv and i not in tail],dtype=int)
    jj=np.array([inv[int(i)] for i in np.flatnonzero(alive) if int(i) in inv],dtype=int)
    keep=np.array([i for i in range(len(local)) if local[i] not in tail],dtype=int)
    assert len(ii)>0 and len(jj)>0
    split=int(z['split']);cohorts=z['cohorts']
    if split<0: blocks=[(U,z['F'].astype(np.float64),lam,0,len(lam))]
    else: blocks=[(U@z['mapping'].astype(np.float64),z['base'].astype(np.float64),lam[:split],0,split),
                 (U,z['join'].astype(np.float64),lam[split:],split,len(lam))]
    full=np.zeros((len(ii),len(jj)));short=full.copy();repair=full.copy()
    d_full=np.zeros(len(keep));d_short=d_full.copy();d_repair=d_full.copy()
    for Ub,F,lb,lo,hi in blocks:
        G=truncate(F,cohorts,lo,hi,min(256,F.shape[0]))
        P=Ub@F;P0=Ub@G;t=trace(F,lb)-trace(G,lb)
        full+=(P[ii]**2)@(P[jj]*lb).T
        short+=(P0[ii]**2)@(P0[jj]*lb).T
        repair+=correction(Ub,t,ii,jj)
        d_full+=np.sum(P[keep]**3*lb,axis=1)
        d_short+=np.sum(P0[keep]**3*lb,axis=1)
        d_repair+=3*np.sum(Ub[keep]**2,axis=1)*(Ub[keep]@t)/(Ub.shape[1]+2)
        del P,P0,F,G
    rowpdf=np.exp(-.5*alpha[local[ii]]**2)/math.sqrt(2*math.pi)
    w=(rowpdf/sig[local[ii]])[:,None]*gate[local[jj]][None,:]
    w[local[ii,None]==local[jj][None,:]]=0
    wd=-alpha[local[keep]]*np.exp(-.5*alpha[local[keep]]**2)/math.sqrt(2*math.pi)/(6*sig[local[keep]]**2)
    plane_den=float(np.sum((w*full)**2));diag_den=float(np.sum((wd*d_full)**2))
    e0=float(np.sum((w*(short-full))**2));e1=float(np.sum((w*(short+repair-full))**2))
    d0=float(np.sum((wd*(d_short-d_full))**2));d1=float(np.sum((wd*(d_short+d_repair-d_full))**2))
    assert plane_den>0 and diag_den>0
    return dict(path=str(path),response_rows=len(ii),live_columns=len(jj),kept_rows=len(keep),
        plane_den=plane_den,plane_error_before=e0,plane_error_after=e1,
        plane_relative_before=e0/plane_den,plane_relative_after=e1/plane_den,
        diagonal_relative_before=d0/diag_den,diagonal_relative_after=d1/diag_den,
        passes_absolute=e1/plane_den<=.01 and d1/diag_den<=.01)

if __name__=='__main__':
    u=unit()
    if sys.argv[1]=='unit':
        result=dict(unit=u,passed=True)
        target=HERE/'unit.json'
    else:
        capture=Path(sys.argv[1]);manifest=json.loads((capture/'capture_manifest.json').read_text())
        assert manifest['outputs_and_bills_equal']
        rows=[]
        for rec in manifest['records']:
            path=Path(rec['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==rec['sha256']
            row=cell(path);row.update(rank=rec['rank'],layer=rec['layer']);rows.append(row)
            print(json.dumps(row),flush=True)
        ratio=sum(r['plane_error_after'] for r in rows)/sum(r['plane_error_before'] for r in rows)
        result=dict(unit=u,rows=rows,aggregate_repair_error_ratio=ratio,
            gates=dict(absolute=all(r['passes_absolute'] for r in rows),repair=ratio<=.75),
            passed=all(r['passes_absolute'] for r in rows) and ratio<=.75)
        target=HERE/'result.json'
    with target.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(event='complete',passed=result['passed'])),flush=True)
