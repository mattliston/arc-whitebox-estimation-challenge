#!/usr/bin/env python3
"""n=32 world: contribution of the 4-PATH words (bond order 3) of the post kappa4 to the NEXT-layer (2,2) and (3,1)
pre-planes, split by a/b position pattern along the path: BLOCK patterns (a-vertices contiguous and b-vertices
contiguous: aabb/bbaa for (2,2), aaab/baaa for (3,1)) are GEMM-class at 1024; INTERLEAVED patterns are 1024^4.
Exact sums over distinct quadruples (n=32), random Gaussian-like state (m, v, C) and random W."""
import numpy as np, sys, itertools, json
from math import factorial
sys.path.insert(0, ".")
from k3gen import hermite_cols
rng = np.random.default_rng(5); n = 32
A = rng.normal(size=(n, n)) / np.sqrt(n); A += 0.4 * np.ones((n, n)) / np.sqrt(n); C = A @ A.T
m = rng.normal(0, 0.3, n); v = np.diag(C).copy(); sd = np.sqrt(v); rho = C / np.outer(sd, sd)
W = rng.normal(size=(n, n)) / np.sqrt(n)
c1, _, _, _, _, _ = hermite_cols(m, v)
def path_word(p, a, b, c_):
    """word for labeled path p=(p1,p2,p3,p4) with bond multiplicities (a,b,c)"""
    return (c1[a][p[0]] * c1[a + b][p[1]] * c1[b + c_][p[2]] * c1[c_][p[3]] * rho[p[0], p[1]] ** a * rho[p[1], p[2]] ** b * rho[p[2], p[3]] ** c_) / (factorial(a) * factorial(b) * factorial(c_))
# post kappa4 distinct-entry path words: T[j,k,l,m] = sum over labeled paths on {j,k,l,m} (12) of word
NB = 3
K22 = {"block": np.zeros((n, n)), "inter": np.zeros((n, n))}; K31 = {"block": np.zeros((n, n)), "inter": np.zeros((n, n))}
for quad in itertools.permutations(range(n), 4):
    j, k, l, mm = quad
    # each ordered quadruple as a PATH ordering (p1..p4) = (j,k,l,m): the labeled path {j-k-l-m}; each labeled path is
    # counted twice over orderings (reversal) -> weight 1/2
    wsum = 0.0
    for a in range(1, NB + 1):
        for b in range(1, NB + 2 - a):
            for c_ in range(1, NB + 3 - a - b):
                if a + b + c_ > NB: continue
                wsum += path_word(quad, a, b, c_)
    wsum *= 0.5
    if wsum == 0.0: continue
    # contribute to next-layer planes: (2,2): choose which 2 path positions -> a (rest -> b); (3,1): which position -> b
    for pos_a in itertools.combinations(range(4), 2):
        pos_b = tuple(q for q in range(4) if q not in pos_a)
        block = (sorted(pos_a) in ([0, 1], [2, 3]))
        Wa = np.prod([W[quad[q], :] for q in pos_a], axis=0); Wb = np.prod([W[quad[q], :] for q in pos_b], axis=0)
        K22["block" if block else "inter"] += wsum * np.outer(Wa, Wb)
    for pos_b in range(4):
        pos_a = tuple(q for q in range(4) if q != pos_b)
        block = pos_b in (0, 3)
        Wa = np.prod([W[quad[q], :] for q in pos_a], axis=0); Wb = W[quad[pos_b], :]
        K31["block" if block else "inter"] += wsum * np.outer(Wa, Wb)
off = ~np.eye(n, dtype=bool)
out = {}
for nm, D in (("K22", K22), ("K31", K31)):
    tot = D["block"] + D["inter"]
    out[nm] = {"block_share_of_norm": float(np.linalg.norm(D["block"][off]) / np.linalg.norm(tot[off])),
               "block_proj_share": float(np.dot(D["block"][off], tot[off]) / np.dot(tot[off], tot[off])),
               "corr_block_total": float(np.corrcoef(D["block"][off], tot[off])[0, 1])}
print(json.dumps(out, indent=1))
