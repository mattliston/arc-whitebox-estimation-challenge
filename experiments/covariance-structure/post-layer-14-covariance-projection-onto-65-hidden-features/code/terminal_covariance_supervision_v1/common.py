import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
D=HERE.parent;P2=D.parent;A=P2/'astra_phase2_reexamination_v1'
TCI=D/'terminal_covariance_information_v1'
PARENT=D/'lazy_rotation_objective_v1/candidate.py'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def verify():
    seal=read(HERE/'sources_sealed.json')
    for x in seal['pins']:assert sha(x['path'])==x['sha256'],x['path']
    return seal
def basis(C,H,indices):
    import numpy as np
    diagonal=np.diag(C)[indices]
    assert np.isfinite(diagonal).all() and np.all(diagonal>0)
    X=np.sqrt(diagonal)[:,None]*np.column_stack([H[indices].astype(np.float64),np.ones(len(indices))])
    rms=np.maximum(np.sqrt(np.mean(X*X,axis=0)),1e-12)
    U,s,_=np.linalg.svd(X/rms,full_matrices=False)
    assert s[0]>0
    Q=U[:,s>1e-7*s[0]]
    return Q,s,rms
def project(E,Q):
    import numpy as np
    E=(E+E.T)*.5
    theta=Q.T@E@Q
    theta=(theta+theta.T)*.5
    projected=Q@theta@Q.T
    return projected,theta
