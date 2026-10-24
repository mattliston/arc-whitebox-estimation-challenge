"""Dense exact identities for the signed pair representation and diagonal."""
import hashlib
import json
from pathlib import Path
import numpy as np
from signed_path_birth_reference_v1 import signed_factors
from positive_path_resummation_v1 import positive_factor

HERE = Path(__file__).resolve().parent
OUT = HERE / 'signed_path_birth_certificate_v1.json'
assert not OUT.exists()
rng = np.random.default_rng(40572815)
records = []
for n in (8, 12):
    B = rng.normal(size=(n,n))/n
    E = rng.normal(size=(n,n))/n
    np.fill_diagonal(B, 0)
    np.fill_diagonal(E, 0)
    P, stats = positive_factor(B, E)
    F = E-B*stats['parallel'][None,:]
    neg = stats['negative']
    N = (neg[None,:]*B+F)*(np.sqrt(-neg)/np.sqrt(neg*neg+stats['perpendicular2']))[None,:]
    Q = np.einsum('ia,ja->aij',B,B)+np.einsum('ia,ja->aij',B,E)+np.einsum('ia,ja->aij',E,B)
    qerr = float(np.max(np.abs(np.einsum('ia,ja->aij',P,P)-np.einsum('ia,ja->aij',N,N)-Q)))
    u = rng.normal(size=n)
    delta = rng.uniform(.4, 1.6, size=n)
    birth = rng.uniform(.1, .5, size=n)
    delta_p = np.sqrt(2.)*delta
    p1,p2,n1,n2 = (v.copy() for v in (P,P,N,N))
    for v, diag in ((p1,(u+delta_p)/2),(p2,(u-delta_p)/2),
                    (n1,delta/2),(n2,-delta/2)):
        np.fill_diagonal(v,diag)
    p = np.concatenate((p1,p2,n1,n2),axis=1)
    lam = np.concatenate((birth/delta_p,-birth/delta_p,-birth/delta,birth/delta))
    T = np.einsum('ia,ja,ka,a->ijk',p,p,p,lam)
    I = np.eye(n)
    expected = np.einsum('ia,ajk,a->ijk',I,Q,birth)
    expected = expected+expected.transpose(1,0,2)+expected.transpose(2,1,0)
    repeated = np.einsum('ia,ja,ka,a->ijk',I,I,P,birth*u)
    expected += repeated+repeated.transpose(0,2,1)+repeated.transpose(2,1,0)
    expected[np.arange(n),np.arange(n),np.arange(n)] += birth*(3*u*u+delta*delta)/4
    terr = float(np.max(np.abs(T-expected)))
    diagerr = float(np.max(np.abs(T[np.arange(n),np.arange(n),np.arange(n)]-birth*(3*u*u+delta*delta)/4)))
    assert max(qerr,terr,diagerr) < 5e-14
    records.append(dict(width=n,quadratic_max_error=qerr,tensor_max_error=terr,diagonal_max_error=diagerr))
record = dict(passed=True,records=records,
              source_sha256=hashlib.sha256((HERE/'signed_path_birth_reference_v1.py').read_bytes()).hexdigest(),
              qualification='Complete dense cubic identity, including repeated entries and unchanged diagonal; no accuracy or bill claim.')
OUT.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
