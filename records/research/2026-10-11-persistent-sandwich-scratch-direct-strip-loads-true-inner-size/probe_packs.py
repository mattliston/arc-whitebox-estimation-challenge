"""TAG opus_engineer a59 (2026-10-11): how many elements the strip kernels' PACKS copy per network (the copy into the zero-interleaved
pack, then the tree load copies the pack again), by family, in process on the given ranks (last one reported), plus the bill each
family's products take. A direct load into the tree's level-1 slot would save about 2 x the pack's elements (its reshape + copy).
usage: probe_packs.py <candidate.py> <comma ranks>"""
from pathlib import Path
import collections
import json
import sys
import types
import numpy as np
import flopscope as flops
fnp = flops.numpy
P2 = Path(__file__).resolve().parents[3]
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
candidate, ranks = Path(sys.argv[1]).resolve(), [int(x) for x in sys.argv[2].split(',')]
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_probe_packs'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
packs = collections.Counter(); calls = collections.Counter(); bills = collections.Counter()
CTX = [None]


def numel(shape):
    n = 1
    for x in shape:
        n *= int(x)
    return n


DEPTH = collections.Counter()


def wrap(name, fam, size):
    orig = getattr(mod, name)
    def w(*a, **k):
        nested = DEPTH[fam] > 0          # _hz7_product calls itself for its full packets: count the outer call only
        DEPTH[fam] += 1
        try:
            before = CTX[0].summary_dict()['flops_used'] if CTX[0] is not None and not nested else 0
            out = orig(*a, **k)
            after = CTX[0].summary_dict()['flops_used'] if CTX[0] is not None and not nested else 0
        finally:
            DEPTH[fam] -= 1
        if not nested:
            bills[fam] += after - before; calls[fam] += 1
            for key, n in size(*a).items():
                packs[(fam, key)] += n
        return out
    setattr(mod, name, w)


def ks_size(a, b, out, m, k, n, plan):
    M, N, sizes, hzr, hzc, K = plan
    s = {}
    if hzr or K != k:
        s['A'] = M * K
    if hzc or K != k:
        s['B'] = N * K
    return s


def hzg_size(a, b, m, k, n):
    hzr, hzk, hzc = m % 64 == 32, k % 64 == 32, n % 64 == 32
    M, K, N = (m + 63) // 64 * 64, (k + 63) // 64 * 64, (n + 63) // 64 * 64
    s = {}
    if hzr or hzk:
        s['A'] = M * K
    if hzc or hzk:
        s['B'] = K * N
    return s


def hz7_size(a, b, m, k, n, dest=None):
    s = {}
    dm, dk = mod._hz7_dim(m), mod._hz7_dim(k)
    M, K = 128 * (dm[0] + (1 if dm[1] else 0)), 128 * (dk[0] + (1 if dk[1] else 0))
    full = n // mod._HZ7_PK * mod._HZ7_PK
    if dm[1] or dk[1]:
        s['A'] = M * K
    NP = (full + 127) // 128 * 128
    if dk[1] or NP != full:
        s['B'] = K * NP
    return s


wrap('_hz_ks_product', 'ks', ks_size)
wrap('_hzg_product', 'hzg', hzg_size)
if hasattr(mod, '_hz7_product'):
    wrap('_hz7_product', 'hz7', hz7_size)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
manifest = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
for rank in ranks:
    packs.clear(); calls.clear(); bills.clear()
    weights = np.load(STORE/Path(manifest[rank]['path']).parent/'weights.npy', allow_pickle=False)
    mlp = types.SimpleNamespace(width=1024, depth=16, weights=[weights[l] for l in range(16)], seed=0, name=f'phase2-{rank}')
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        CTX[0] = ctx
        est.predict(mlp, 2**41)
        bill = int(ctx.summary_dict()['flops_used'])
    CTX[0] = None
    print(json.dumps(dict(rank=rank, bill=bill)))
    for fam in sorted(calls):
        a, b = packs[(fam, 'A')], packs[(fam, 'B')]
        print(f'  {fam:4s} calls {calls[fam]:3d}  bill {100*bills[fam]/bill:6.2f}%  packed A {a/1e6:7.1f} M  B {b/1e6:7.1f} M elements'
              f'  -> 2 x packed = {100*2*(a+b)/bill:.3f}% of the bill')
    sys.stdout.flush()
