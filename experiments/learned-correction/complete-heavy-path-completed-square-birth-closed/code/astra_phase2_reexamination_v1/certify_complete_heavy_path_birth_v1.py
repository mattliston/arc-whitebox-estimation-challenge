"""Primary-code coefficient certificate at small widths only."""
from functools import lru_cache
from pathlib import Path
import json
import torch
from mlp_kprop.diagslice import eval_part, diagslice, zero_repeated
from mlp_kprop.harmonic import HTensor
from mlp_kprop.partitions import check_vec_partition
from mlp_kprop.wick import relu_wick_coef
import mlp_kprop.kprop_harmonic as kp

torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)
here = Path(__file__).resolve().parent
out = here/'complete_heavy_path_birth_certificate_v1.json'
assert not out.exists()
terms = []
for part, blocks in kp.get_all_terms(3, d_max=4, augment=True):
    if part != (1, 1, 1) or sum(kp.vec_block_cost(b) for b in blocks) != 3:
        continue
    supports = {tuple(i for i, t in enumerate(b) if t) for b in blocks}
    if len(supports) == 2 and all(len(s) == 2 for s in supports) and kp.is_hypertree(blocks):
        terms.append(blocks)
assert terms
rows = []
for n, seed in ((8, 94135), (16, 941100)):
    torch.manual_seed(seed)
    m = torch.randn(n)*.6
    x = torch.randn(n, n)
    c = x@x.T/n + .5*torch.eye(n)
    p = torch.randn(n, n+2)*.2
    lam = torch.randn(n+2)*.05
    third = torch.einsum('ir,jr,kr,r->ijk', p, p, p, lam)
    a = torch.randn(n, n)*.03
    a = (a+a.T)/2
    b = torch.eye(n)+.2*c
    tower = {1: HTensor(m, 0), 2: HTensor(c, 0), 3: HTensor(third, 0), 4: HTensor(a, 1, metric=b)}
    @lru_cache(None)
    def wick(k, power):
        return relu_wick_coef(m, c.diag(), k, power)
    target = torch.zeros((n, n, n))
    for blocks in terms:
        target += kp.multiply_wicks(eval_part(tower, blocks, 3), check_vec_partition(blocks, 3), (1, 1, 1), wick)
    target = zero_repeated(target)
    g, d = wick(1, 1), wick(2, 1)
    eta = -m/c.diag()
    t = diagslice(tower[3], (2, 1))
    k22 = diagslice(tower[4], (2, 2))
    original = g[:, None]*c
    shift = (.5*g[:, None]*t.T*eta[None, :]+.5*d[:, None]*t
             +d[:, None]*eta[None, :]*(k22/4+c*c/2))
    original = zero_repeated(original)
    shift = zero_repeated(shift)
    identity = torch.eye(n)
    def born(eps):
        p = original+eps*shift
        raw = torch.einsum('ia,ja,ka,a->ijk', p, p, identity, d)
        return zero_repeated(raw+raw.permute(0, 2, 1)+raw.permute(2, 1, 0))
    eps = 1e-3
    derivative = (born(eps)-born(-eps))/(2*eps)
    direct = torch.einsum('ia,ja,ka,a->ijk', original, shift, identity, d)
    direct = direct+direct.permute(1, 0, 2)
    direct = zero_repeated(direct+direct.permute(0, 2, 1)+direct.permute(2, 1, 0))
    relative = float(torch.linalg.vector_norm(derivative-target)/torch.linalg.vector_norm(target))
    explicit = float(torch.linalg.vector_norm(direct-target)/torch.linalg.vector_norm(target))
    row = dict(n=n, seed=seed, derivative_relative_error=relative,
               explicit_first_order_relative_error=explicit, passed=max(relative, explicit)<1e-9)
    rows.append(row)
    print(json.dumps(row), flush=True)
receipt = dict(rows=rows, n_primary_terms=len(terms), primary_terms=terms,
               passed=all(r['passed'] for r in rows),
               qualification='Small synthetic coefficient certificate. The finite square has an additional shift-squared term; no full-size accuracy claim.')
out.write_text(json.dumps(receipt, indent=2)+'\n')
assert receipt['passed'], receipt
