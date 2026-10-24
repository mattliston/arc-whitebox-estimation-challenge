import itertools,resource
import numpy as np
from common import *
from feature import planes,summaries,relative
seal_check();rng=np.random.default_rng(7501);records=[]
for case in range(6):
    n=5;k=6
    cap=dict(P=rng.normal(size=(n,k)).astype(np.float32),lam=rng.normal(size=k).astype(np.float32),
        W=rng.normal(size=(n,n)).astype(np.float32),A=rng.normal(size=(n,n)).astype(np.float32),
        C=rng.normal(size=(n,n)),b=rng.normal(size=n))
    if case==1:cap['P']=np.zeros((n,0),np.float32);cap['lam']=np.zeros(0,np.float32)
    if case==2:
        cap['P'][:,1::2]=cap['P'][:,::2];cap['lam']=np.tile([1,-1],3).astype(np.float32)
    if case==3:cap['A']*=0;cap['b']*=0
    if case==4:cap['C']*=0;cap['P']*=0;cap['A']*=0;cap['b']*=0
    s=np.exp(rng.normal(size=n)*.2);alpha=rng.normal(size=n)
    B,K=planes(cap);Z=summaries(B,K,s,alpha)
    C=cap['C'].astype(np.float32).astype(np.float64);C=(C+C.T)/2;W=cap['W'].astype(float);A=cap['A'].astype(float)
    Bref=np.zeros((n,n));Uref=np.zeros((n,n))
    for i,j in itertools.product(range(n),repeat=2):
        for a in range(n):
            Uref[i,j]+=W[a,i]*A[a,j]
            for b in range(n):Bref[i,j]+=W[a,i]*C[a,b]*W[b,j]
    tensor=np.zeros((n,n,n))
    for i,j,k in itertools.product(range(n),repeat=3):
        value=0.
        for a in range(len(cap['lam'])):
            value+=float(cap['lam'][a])*float(cap['P'][i,a])*float(cap['P'][j,a])*float(cap['P'][k,a])
        for a in range(n):
            value+=(Uref[i,a]*W[a,j]*W[a,k]+W[a,i]*Uref[j,a]*W[a,k]+W[a,i]*W[a,j]*Uref[k,a])/3
        value+=cap['b'][i]*Bref[j,k]+cap['b'][j]*Bref[i,k]+cap['b'][k]*Bref[i,j]
        tensor[i,j,k]=value
    Kref=np.array([[tensor[i,i,j] for j in range(n)] for i in range(n)])
    R=Bref/np.outer(s,s);T=Kref/(s[:,None]**2*s[None,:]);Zref=[]
    for i in range(n):
        Zref.append(s[i]*np.array([sum(R[i,j] for j in range(n))/n,
            (sum(R[i,j]**2 for j in range(n))/n)**.5,sum(abs(R[i,j]) for j in range(n))/n,
            sum(T[i,j]**2 for j in range(n))**.5,sum(T[j,i]**2 for j in range(n))**.5,
            sum(T[i,j] for j in range(n))/n**.5,sum(T[j,i] for j in range(n))/n**.5,
            sum(R[i,j]*alpha[j] for j in range(n))/n]))
    errors=dict(covariance=relative(B,Bref),plane=relative(K,Kref),features=relative(Z,np.array(Zref)))
    permutation=np.array([2,4,0,3,1]);c2=dict(cap);c2['P']=cap['P'][permutation];c2['W']=cap['W'][:,permutation];c2['b']=cap['b'][permutation]
    B2,K2=planes(c2);Z2=summaries(B2,K2,s[permutation],alpha[permutation])
    errors['permutation']=relative(Z2,Z[permutation]);assert max(errors.values())<=2e-10,errors
    Br,Kr=planes(cap);Zr=summaries(Br,Kr,s,alpha)
    assert B.tobytes()==Br.tobytes() and K.tobytes()==Kr.tobytes() and Z.tobytes()==Zr.tobytes()
    records.append(dict(case=case,errors=errors,repeat_bytes_exact=True))
write(HERE/'unit.json',dict(passed=True,records=records,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(passed=True,max_error=max(max(r['errors'].values()) for r in records)),flush=True)
