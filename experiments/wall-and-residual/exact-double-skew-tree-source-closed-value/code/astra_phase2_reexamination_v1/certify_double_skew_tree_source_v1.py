"""Primary diagram and dense tensor certificates; no wide truths or fits."""
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from mlp_kprop.diagslice import eval_part, diagslice, zero_repeated
from mlp_kprop.harmonic import HTensor
from mlp_kprop.partitions import check_vec_partition
from mlp_kprop.wick import relu_wick_coef
import mlp_kprop.kprop_harmonic as kp
from double_skew_tree_source_v1 import dense, plane

HERE = Path(__file__).resolve().parent
OUT = HERE/'double_skew_tree_source_certificate_v1.json'
assert not OUT.exists()
torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)
blocks = [(2,1,0), (1,2,0), (2,0,1), (1,0,2), (0,2,1), (0,1,2)]
terms = [t for t in combinations(blocks, 2)
         if len({tuple(i for i,v in enumerate(b) if v) for b in t}) == 2]
assert len(terms) == 12
assert all(sum(kp.vec_block_cost(b) for b in t) == 4 and kp.is_hypertree(t) for t in terms)
records = []
for n, seed in ((8, 9062144), (12, 9062145)):
    torch.manual_seed(seed)
    mu = torch.randn(n)*.7
    x = torch.randn(n,n)
    C = x@x.T/n + torch.eye(n)
    P = torch.randn(n, n+3)*.2
    T = torch.einsum('ir,jr,kr->ijk', P, P, P)*.3
    tower = {1: HTensor(mu, 0), 2: HTensor(C, 0), 3: HTensor(T, 0)}
    @lru_cache(None)
    def wick(k, power):
        return relu_wick_coef(mu, C.diag(), k, power)
    target = torch.zeros((n,n,n))
    for t in terms:
        target += kp.multiply_wicks(eval_part(tower, t, 3), check_vec_partition(t,3), (1,1,1), wick)
    target = zero_repeated(target).numpy()
    K = diagslice(tower[3], (2,1)).numpy()
    g,d,e,h = (wick(k,1).numpy() for k in range(1,5))
    predicted = dense(K,g,d,e,h)
    W = np.random.default_rng(seed).normal(size=(n,n))/np.sqrt(n)
    pushed = np.einsum('ia,jb,kc,ijk->abc', W,W,W,predicted, optimize=True)
    reference_plane = np.einsum('iij->ij', pushed)
    query = plane(K,g,d,e,h,W)
    relative = float(np.linalg.norm(predicted-target)/np.linalg.norm(target))
    query_relative = float(np.linalg.norm(query-reference_plane)/np.linalg.norm(reference_plane))
    derivative_error = float(np.max(np.abs(h - (mu.numpy()**2/C.diag().numpy()-1)*d/C.diag().numpy())))
    assert relative < 1e-11 and query_relative < 1e-11 and derivative_error < 1e-12
    records.append(dict(width=n, primary_relative=relative, query_relative=query_relative,
                        fourth_derivative_max_error=derivative_error))
receipt = dict(passed=True, primary_terms=terms, records=records,
               source_sha256=hashlib.sha256((HERE/'double_skew_tree_source_v1.py').read_bytes()).hexdigest(),
               qualification='Exact12 coefficient diagrams and complete dense next-plane query at small widths only; no wide-network accuracy claim.')
OUT.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2), flush=True)
