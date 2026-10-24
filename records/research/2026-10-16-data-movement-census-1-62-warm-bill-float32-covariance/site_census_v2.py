"""[opus-dmx] v2 = site_census_v1 with the op's resolved dtype in the site key. [opus-exact] native FLOP census by call site: every op record the local flopscope appends is tagged with the innermost frame of
the candidate file (line, function). Setup state = a captured boundary setup (CAPTURED_SETUP_MANIFEST), networks run in the given
order in one process (the last one is the warm network), exactly as native_probe_generic_v1.py does.
usage (OpenBLAS env python, BLAS 1 thread, under the guard): site_census_v1.py <candidate.py> <ranks> <outdir>"""
import collections, hashlib, json, os, sys, types
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope._budget as fb
sys.path.insert(0, '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline')
import captured_setup_replay_generic_v1 as replay

P2 = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16')
candidate = Path(sys.argv[1]).resolve(); ranks = [int(x) for x in sys.argv[2].split(',')]; out = Path(sys.argv[3]).resolve(); out.mkdir()
assert hashlib.sha256(candidate.read_bytes()).hexdigest() == os.environ['CANDIDATE_SHA256']
CF = str(candidate)
SITES = []          # parallel to the active context's op log: (line, function)
_append = fb.BudgetContext._append_op_record


def _append_site(self, record):
    f = sys._getframe(1)
    while f is not None and f.f_code.co_filename != CF:
        f = f.f_back
    SITES.append((f.f_lineno, f.f_code.co_name) if f is not None else (0, '<outside>'))
    return _append(self, record)


fb.BudgetContext._append_op_record = _append_site
store = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
manifest = json.loads((store/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_eval'); mod.__file__ = CF; sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), CF, 'exec'), mod.__dict__)
expected_setup = replay.install(mod, os.environ['CAPTURED_SETUP_MANIFEST'])


class MLP:
    def __init__(self, weights, rank):
        self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, f'phase2-{rank}'


est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
assert replay.verify(mod, expected_setup)['all_setup_counts_exact']
rows = []
for rank in ranks:
    weights = np.load(store/Path(manifest[rank]['path']).parent/'weights.npy')
    SITES.clear()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        y = np.asarray(est.predict(MLP([weights[l] for l in range(16)], rank), 2**41), dtype=np.float64)
        log = list(ctx.op_log); used = int(ctx.flops_used)
    assert len(log) == len(SITES), (len(log), len(SITES))
    by_op, by_site = collections.Counter(), collections.defaultdict(lambda: [0, 0, collections.Counter()])
    for rec, (line, func) in zip(log, SITES):
        by_op[rec.op_name] += rec.flop_cost
        s = by_site[f'{line} {func} {rec.op_name} {rec.resolved_dtype}']; s[0] += 1; s[1] += rec.flop_cost
        s[2][str(rec.shapes)[:80]] += 1
    np.save(out/f'pred_{rank}.npy', y)
    rows.append(dict(rank=rank, flops=used, ops=len(log), sum_cost=sum(r.flop_cost for r in log), by_op=dict(by_op.most_common()),
                     by_site={k: [v[0], v[1], dict(v[2].most_common(4))] for k, v in sorted(by_site.items(), key=lambda kv: -kv[1][1])}))
    print(json.dumps(dict(rank=rank, flops=used, ops=len(log))), flush=True)
(out/'census.json').write_text(json.dumps(dict(candidate=CF, sha256=os.environ['CANDIDATE_SHA256'], ranks=ranks, rows=rows)) + '\n')
print('CENSUS_DONE', flush=True)
