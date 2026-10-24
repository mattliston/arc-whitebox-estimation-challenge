"""Dense certificates, not target accuracy or production metering."""
import json
from pathlib import Path
import numpy as np
from positive_path_resummation_v1 import factors,positive_factor,birth_difference_slices,dense_difference
from exact_heavy_source_slices_v1 import dense as heavy_dense
from full_edge_gaussian_bond3_v1 import dense as gaussian_dense

HERE=Path(__file__).resolve().parent;OUT=HERE/'positive_path_resummation_certificate_v1.json'
assert not OUT.exists()
rng=np.random.default_rng(26951931);records=[]
for n,q in ((8,3),(12,4)):
    x=rng.normal(size=(n,n));C=x@x.T/n+np.eye(n)
    g=rng.uniform(.1,.9,n);d=rng.uniform(.05,.3,n);eta=rng.normal(size=n)*.3
    K21=rng.normal(size=(n,n))*.03;K22=rng.normal(size=(n,n))*.02;K22=(K22+K22.T)/2
    radial=.05;B,E=factors(C,g,d,eta,K21,K22,radial);Z,stats=positive_factor(B,E)
    maximum=0.
    for a in range(n):
        Q=np.outer(B[:,a],B[:,a])+np.outer(B[:,a],E[:,a])+np.outer(E[:,a],B[:,a])
        ev,U=np.linalg.eigh(Q);target=max(0,ev[-1])*np.outer(U[:,-1],U[:,-1])
        maximum=max(maximum,float(np.max(np.abs(np.outer(Z[:,a],Z[:,a])-target))))
    # Linearized birth differences contain exactly 6 Gaussian doubled paths
    # and the 18 independently certified non-Gaussian heavy terms.
    I=np.eye(n);D=E*(1+radial/3)
    T=np.einsum('ia,ja,ka,a->ijk',B,D,I,d)
    T=T+T.transpose(1,0,2);T=T+T.transpose(0,2,1)+T.transpose(2,1,0)
    ii,jj,kk=np.indices(T.shape);T*=((ii!=jj)&(ii!=kk)&(jj!=kk))
    expected=heavy_dense(C,g,d,eta,K21,K22)+gaussian_dense(C,g,d,eta*d)-gaussian_dense(C,g,d,np.zeros(n))
    relation=float(np.max(np.abs(T-expected)))
    H=rng.normal(size=(n,q));t=d*(1+radial/3)*rng.integers(0,2,n)
    actual=birth_difference_slices(B,Z,t,H)
    target=np.einsum('ia,ijk->ajk',H,dense_difference(B,Z,t))
    query=float(np.max(np.abs(actual-target)))
    assert max(maximum,relation,query)<2e-13,(maximum,relation,query)
    records.append(dict(width=n,positive_component_max_error=maximum,primary_class_relation_max_error=relation,
                        query_max_error=query,zero_diagonal_max=float(np.max(np.abs(np.diag(Z))))))
record=dict(passed=True,records=records,qualification='Rank-two positive eigencomponent and exact slice algebra; approximation quality is not asserted.')
OUT.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
