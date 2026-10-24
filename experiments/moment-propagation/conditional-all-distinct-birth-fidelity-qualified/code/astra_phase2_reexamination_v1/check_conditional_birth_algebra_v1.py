"""Small direct-factor certificates, including deliberately clipped u."""
from pathlib import Path
import json
import numpy as np
from conditional_distinct_birth_v1 import coefficient_plane_shift,birth_scale

HERE = Path(__file__).resolve().parent
out = HERE/'conditional_birth_algebra_v1.json'
assert not out.exists()
rows = []
for n in (8,16):
    rng = np.random.default_rng(955000+n)
    bp = (rng.normal(size=(n,n))*.2).astype(np.float32)
    np.fill_diagonal(bp,0)
    # Force many clipping events; beta/birth is not generally the stored u.
    v = rng.uniform(.2,1,n)
    birth = rng.uniform(.1,.5,n)
    beta = rng.normal(size=n)*10
    u = np.clip(beta/birth,-8*v,8*v)
    gap = rng.uniform(.05,2,n)
    p1,p2 = bp.copy(),bp.copy()
    np.fill_diagonal(p1,(u+gap)/2)
    np.fill_diagonal(p2,(u-gap)/2)
    a0 = (birth/gap).astype(np.float32)
    eta = rng.uniform(.6,1.4,n).astype(np.float32)
    a1 = (a0*eta).astype(np.float32)
    births = (rng.uniform(size=n) > .25).astype(np.float64)
    def plane(a):
        f,g = p1.astype(np.float64),p2.astype(np.float64)
        aa = a.astype(np.float64)*births
        return (f*f)@(f*aa).T-(g*g)@(g*aa).T
    expected = plane(a1)-plane(a0)
    shift = coefficient_plane_shift(p1,p2,a0,a1,births)
    delta_star = -3*shift.T+2*np.diag(np.diag(shift))
    star_plane = delta_star.T/3
    np.fill_diagonal(star_plane,np.diag(delta_star))
    scale = max(np.linalg.norm(expected),1e-30)
    error = float(np.linalg.norm(shift-expected)/scale)
    restored = float(np.linalg.norm(expected+star_plane)/scale)
    zero = coefficient_plane_shift(p1,p2,a0,a0,births)
    h = rng.normal(size=(n,64)).astype(np.float32)
    zero_eta = birth_scale(8,h,birth,births,np.zeros((3,64),np.float32))
    row = dict(width=n,clipped_u_count=int(np.count_nonzero(u != beta/birth)),
               increment_relative=error,restored_plane_change_relative=restored,
               zero_increment_byte_exact=bool(np.array_equal(zero,np.zeros_like(zero))),
               zero_head_eta_byte_exact=bool(np.array_equal(zero_eta,np.ones(n,np.float32))))
    row['passed'] = max(error,restored) < 1e-10 and row['zero_increment_byte_exact'] and row['zero_head_eta_byte_exact']
    rows.append(row)
    assert row['passed'],row
result = dict(rows=rows,passed=all(r['passed'] for r in rows),
              qualification='Small synthetic stored-factor algebra only, including clipped diagonal cases. Not a full-card parity, derivative, fit, raw score or billed-cost result.')
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
