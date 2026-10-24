"""TAG opus_engineer a62 module census (lead 10-11 RESTART, module ablation audit): every charge of one in-process prediction
attributed to the main loop's current line and the chain of candidate frames under it.

usage: module_census_v1.py <candidate.py> <comma ranks> <out.json>
Key per charge: (rank, layer, main-loop line, chain, flopscope op). The main loop = the _predict_binary frame that holds 'means';
chain = the candidate-file frames below it, outermost first, as (function, current line), cut at the first product entry (ENTRY,
kept as (name, 0)) so that kernel internals do not split the keys. Outside the main loop: layer -1, line 0, chain from the
outermost candidate frame. Value: flops paid, charged ops. Passive: only BudgetContext._charge_op is wrapped, after setup;
the per-network bill must equal the attributed sum.
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
mod = types.ModuleType('candidate_modules'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
est = mod.Estimator(); t0 = time.monotonic()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2',
                                scratch_dir=None, submission_dir=None, seed=0))
print(json.dumps(dict(event='setup', seconds=time.monotonic()-t0)), flush=True)
SKIP = {'w', 'wrapped', '<lambda>', '<listcomp>', '<genexpr>', '<dictcomp>'}
ENTRY = {'_mm', '_mm_batch', '_tsr_mm', '_join_rows_mm', '_shared_old384_mm', '_psdb_sym_upper', '_psd_sym_upper',
         '_ps_sym_upper', '_psdm_sym_upper', '_sym_upper', '_bb_product', '_ks_product', '_hz_ks_product', '_hzg_product',
         '_hz7_product', '_r7_product', '_d7_product', '_d7_product_kpad', '_ipt_product_ab', '_packed_read5', '_packed_read6',
         '_cp_d6_shared', '_w5_port_product', '_square_product', '_ubc_product', '_ubc_transpose_product', '_resident_gram512',
         '_push5_product', '_push6_padded', '_dcv_tiles'}
cfile = str(candidate)
state = dict(rank=None)
acc = collections.defaultdict(lambda: [0, 0])
original = _budget.BudgetContext._charge_op


def charged(self, op_name, flop_cost, subscripts, shapes, **kwargs):
    before = self._flops_used
    result = original(self, op_name, flop_cost, subscripts, shapes, **kwargs)
    paid = self._flops_used-before
    frame = sys._getframe(1); chain = []; main = None
    while frame is not None:
        code = frame.f_code
        if code.co_filename == cfile:
            if code.co_name == '_predict_binary' and 'means' in frame.f_locals:
                main = frame; break
            chain.append(frame)
        frame = frame.f_back
    keyc = []
    for f in reversed(chain):
        name = f.f_code.co_name
        if name in SKIP:
            continue
        if name in ENTRY:
            keyc.append((name, 0)); break
        keyc.append((name, f.f_lineno))
    if main is None:
        layer, line = -1, 0
    else:
        layer, line = int(main.f_locals.get('l', -1)), main.f_lineno
    rec = acc[(state['rank'], layer, line, tuple(keyc), op_name)]
    rec[0] += paid; rec[1] += 1
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
        bill = int(ctx.summary_dict()['flops_used'])
        attributed = sum(v[0] for k, v in acc.items() if k[0] == rank)
        assert attributed == bill, (attributed, bill)
        nets.append(dict(rank=rank, flops=bill, ops=sum(v[1] for k, v in acc.items() if k[0] == rank),
                         wall_s=time.monotonic()-t0, cpu_s=time.process_time()-c0,
                         pred_sha=hashlib.sha256(y.astype(np.float32).tobytes()).hexdigest()[:16]))
        print(json.dumps(nets[-1]), flush=True)
        del weights, y
finally:
    _budget.BudgetContext._charge_op = original
rows = [[k[0], k[1], k[2], [list(c) for c in k[3]], k[4], v[0], v[1]] for k, v in acc.items()]
with out.open('x') as stream:
    json.dump(dict(candidate=str(candidate), sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(), ranks=ranks,
                   nets=nets, fields=['rank', 'layer', 'line', 'chain', 'op', 'flops', 'ops'], rows=rows), stream)
print(json.dumps(dict(done=True, rows=len(rows))), flush=True)
