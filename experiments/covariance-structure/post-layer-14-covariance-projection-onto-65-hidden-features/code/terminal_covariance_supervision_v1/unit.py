import time,resource
from common import HERE,write,verify,project
def main():
    import numpy as np
    from threadpoolctl import threadpool_info
    assert all(x['num_threads']==1 for x in threadpool_info())
    verify();started=time.monotonic();rng=np.random.Generator(np.random.PCG64DXSM(2026092054));rows=[]
    for n,k in [(7,1),(7,4),(7,7),(11,5)]:
        Q=np.linalg.qr(rng.standard_normal((n,k)))[0]
        for mode in ['signed','in_span','zero']:
            E=rng.standard_normal((n,n));E=(E+E.T)*.5
            if mode=='in_span':
                t=rng.standard_normal((k,k));E=Q@((t+t.T)*.5)@Q.T
            if mode=='zero':E*=0
            p,t=project(E,Q);r=E-p;den=max(float(np.linalg.norm(E)),1.)
            pp,_=project(p,Q)
            errors=dict(symmetry=float(np.linalg.norm(p-p.T))/den,idempotence=float(np.linalg.norm(pp-p))/den,
                orthogonality=float(np.linalg.norm(Q.T@r@Q))/den,
                pythagorean=abs(float(np.sum(E*E)-np.sum(p*p)-np.sum(r*r)))/max(float(np.sum(E*E)),1.))
            if mode=='in_span':errors['in_span_identity']=float(np.linalg.norm(r))/den
            assert max(errors.values())<=1e-10,errors
            rows.append(dict(n=n,k=k,mode=mode,errors=errors))
    write(HERE/'unit.json',dict(passed=True,rows=rows,truth_accessed=False,wall_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print('12 projector fixtures passed',flush=True)
if __name__=='__main__':main()
