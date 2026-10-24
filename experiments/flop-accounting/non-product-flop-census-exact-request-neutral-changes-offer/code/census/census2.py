"""TAG opus_engineer product census: per product CALL attribution (top-level shapes, depth, bill split).

usage: census2.py <candidate.py> <comma ranks> <out.json>
For every charge the outermost candidate frame whose function is a product entry point (see ENTRY) is the
"product call"; its operand shapes are read from the frame locals. Output rows: one per (rank, layer, chain to
the entry, operand shapes) with total flops, leaf matmul flops, add/sub flops, other flops, number of calls and
the leaf matmul shapes seen (depth = number of leading 7s). Charges outside any product entry are aggregated by
(rank, layer, chain, op). Passive: only BudgetContext._charge_op is wrapped, after setup.
"""
from pathlib import Path
import collections
import hashlib
import json
import sys
import time
import types
import numpy as np
import flopscope as flops
from flopscope import _budget

P2 = Path(__file__).resolve().parents[3]
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
candidate = Path(sys.argv[1]).resolve()
ranks = [int(x) for x in sys.argv[2].split(',')]
out = Path(sys.argv[3]).resolve()
assert not out.exists()
manifest = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_census'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
est = mod.Estimator(); t0 = time.monotonic()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2',
                                scratch_dir=None, submission_dir=None, seed=0))
setup = dict(seconds=time.monotonic()-t0, pb_log=[list(x) for x in getattr(mod, '_PB_LOG', [])])
print(json.dumps(dict(event='setup', **setup)), flush=True)
SKIP = {'w', 'wrapped', '<lambda>', '<listcomp>', '<genexpr>', '<dictcomp>'}
ENTRY = {'_mm', '_mm_batch', '_tsr_mm', '_join_rows_mm', '_shared_old384_mm', '_psdb_sym_upper', '_psd_sym_upper',
         '_ps_sym_upper', '_sym_upper', '_bb_product', '_ks_product', '_r7_product', '_d7_product', '_d7_product_kpad',
         '_ipt_product_ab', '_packed_read5', '_packed_read6', '_cp_d6_shared', '_w5_port_product', '_square_product',
         '_ubc_product', '_ubc_transpose_product', '_resident_gram512', '_push5_product', '_push6_padded'}
cfile = str(candidate)
state = dict(rank=None, frame=None, serial=0)
prod = {}; other = collections.Counter(); other_n = collections.Counter()
original = _budget.BudgetContext._charge_op


def shape_of(x):
    try:
        return [int(v) for v in x.shape]
    except Exception:
        return None


def entry_shapes(frame):
    loc = frame.f_locals; name = frame.f_code.co_name
    if name == '_mm_batch':
        return [[shape_of(a), shape_of(b)] for a, b in loc.get('pairs', ())]
    for an, bn in (('a', 'b'), ('left', 'right'), ('A', 'B'), ('As', 'Bs')):
        if an in loc and bn in loc:
            a, b = loc[an], loc[bn]
            if isinstance(a, (list, tuple)):
                return [[shape_of(x), shape_of(y)] for x, y in zip(a, b)]
            return [shape_of(a), shape_of(b)]
    return None


def charged(self, op_name, flop_cost, subscripts, shapes, **kwargs):
    before = self._flops_used
    result = original(self, op_name, flop_cost, subscripts, shapes, **kwargs)
    paid = self._flops_used-before
    frame = sys._getframe(1); chain = []; owner = None
    while frame is not None:
        code = frame.f_code
        if code.co_name == '_predict_binary' and 'means' in frame.f_locals:
            owner = frame; break
        if code.co_filename == cfile and code.co_name not in SKIP:
            chain.append(frame)
        frame = frame.f_back
    layer = -1 if owner is None else owner.f_locals.get('l', -1)
    chain.reverse()
    names = [f.f_code.co_name for f in chain]
    entry = next((i for i, nm in enumerate(names) if nm in ENTRY), None)
    if entry is None:
        dedup = []
        for nm in names:
            if not dedup or dedup[-1] != nm: dedup.append(nm)
        key = (state['rank'], layer, tuple(dedup), op_name, '+'.join(sorted(set(str(d) for d in (kwargs.get('dtypes') or ())))))
        other[key] += paid; other_n[key] += 1
        return result
    ef = chain[entry]
    if state['frame'] is not ef:
        state['frame'] = ef; state['serial'] += 1
        dedup = []
        for nm in names[:entry+1]:
            if not dedup or dedup[-1] != nm: dedup.append(nm)
        prod[state['serial']] = dict(rank=state['rank'], layer=layer, chain=dedup, shapes=entry_shapes(ef),
                                     flops=0, matmul=0, addsub=0, rest=0, leaf=collections.Counter(),
                                     inner=collections.Counter(), ops=0)
    rec = prod[state['serial']]
    rec['flops'] += paid; rec['ops'] += 1
    if op_name == 'matmul':
        rec['matmul'] += paid
        rec['leaf'][json.dumps([[int(d) for d in sh] for sh in shapes])] += paid
    elif op_name in ('add', 'subtract'):
        rec['addsub'] += paid
    else:
        rec['rest'] += paid
    rec['inner'][names[-1]+':'+op_name] += paid
    return result


nets = []
try:
    _budget.BudgetContext._charge_op = charged
    for rank in ranks:
        state['rank'] = rank
        weights = np.load(STORE/Path(manifest[rank]['path']).parent/'weights.npy', allow_pickle=False)
        mlp = types.SimpleNamespace(width=1024, depth=16, weights=[weights[l] for l in range(16)], seed=0, name=f'phase2-{rank}')
        t0 = time.monotonic(); c0 = time.process_time()
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
        state['frame'] = None
        bill = int(ctx.summary_dict()['flops_used'])
        attributed = sum(r['flops'] for r in prod.values() if r['rank'] == rank)+sum(v for k, v in other.items() if k[0] == rank)
        assert attributed == bill, (attributed, bill)
        nets.append(dict(rank=rank, flops=bill, wall_s=time.monotonic()-t0, cpu_s=time.process_time()-c0,
                         pred_sha256=hashlib.sha256(y.tobytes()).hexdigest()))
        print(json.dumps(nets[-1]), flush=True)
        del weights, y
finally:
    _budget.BudgetContext._charge_op = original
products = [dict(serial=s, **{k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in r.items()}) for s, r in prod.items()]
others = [dict(rank=k[0], layer=k[1], chain=list(k[2]), op=k[3], dtypes=k[4], flops=v, ops=other_n[k]) for k, v in other.items()]
with out.open('x') as stream:
    json.dump(dict(candidate=str(candidate), sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(), ranks=ranks,
                   setup=setup, nets=nets, products=products, others=others), stream)
print(json.dumps(dict(done=True, products=len(products), others=len(others))), flush=True)
