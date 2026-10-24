"""Independent dense synthetic projection and sample-statistic certificates."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from two_high_fourth_v1 import batch_slice_statistic, pushed_planes, radial_slices, small_projection

HERE = Path(__file__).resolve().parent
OUT = HERE/'two_high_fourth_certificate_v1.json'
assert not OUT.exists()
rng = np.random.default_rng(930079)
records = []
for n, q in ((7, 2), (11, 4)):
    Q = np.linalg.qr(rng.normal(size=(n, q)))[0]
    raw = rng.normal(size=(n,)*4)
    T = sum(raw.transpose(p) for p in itertools.permutations(range(4)))/24
    U = np.einsum('ijkl,ia,jb->abkl', T, Q, Q, optimize=True)
    projection = small_projection(T, Q)
    W = rng.normal(size=(n, n))/np.sqrt(n)
    pushed = np.einsum('ijkl,ia,jb,kc,ld->abcd', projection, W, W, W, W, optimize=True)
    q22, q31, k4 = pushed_planes(Q, U, W)
    expected22 = np.einsum('iijj->ij', pushed)
    expected31 = np.einsum('iiij->ij', pushed)
    count = 173
    samples = rng.normal(size=(count, n))
    samples = samples + .2*(samples*samples-1)
    samples -= samples.mean(0)
    z = samples@Q
    S = samples.T@samples/count
    G, H = z.T@z/count, z.T@samples/count
    M = np.einsum('pi,pj,pk,pl->ijkl', samples, samples, samples, samples, optimize=True)/count
    wick = np.einsum('ij,kl->ijkl', S, S)+np.einsum('ik,jl->ijkl', S, S)+np.einsum('il,jk->ijkl', S, S)
    statistic = count*count/((count-1)*(count-2)*(count-3))*((count+1)*M-(count-1)*wick)
    slice_raw = np.einsum('pa,pb,pi,pj->abij', z, z, samples, samples, optimize=True)/count
    slices = batch_slice_statistic(S, G, H, slice_raw, count)
    r = .037
    expected_radial = np.einsum('ijkl,ia,jb->abkl', r*wick/3, Q, Q, optimize=True)
    errors = dict(
        idempotence=float(np.max(np.abs(small_projection(projection, Q)-projection))),
        orthogonality=abs(float(np.sum((T-projection)*projection))),
        projection_core=float(np.max(np.abs(np.einsum('ijkl,ia,jb->abkl', projection, Q, Q, optimize=True)-U))),
        q22=float(np.max(np.abs(q22-expected22))),
        q31=float(np.max(np.abs(q31-expected31))),
        k4=float(np.max(np.abs(k4-np.diag(expected22)))),
        plane_diagonal=float(np.max(np.abs(np.diag(q31)-k4))),
        sample_statistic=float(np.max(np.abs(slices-np.einsum('ijkl,ia,jb->abkl', statistic, Q, Q, optimize=True)))),
        radial_slices=float(np.max(np.abs(radial_slices(S, Q, r)-expected_radial))))
    records.append(dict(n=n, q=q, errors=errors, passed=max(errors.values()) <= 1e-10))
record = dict(passed=all(r['passed'] for r in records), records=records, target_access=False,
              source_sha256=hashlib.sha256((HERE/'two_high_fourth_v1.py').read_bytes()).hexdigest())
OUT.write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2), flush=True)
assert record['passed']
