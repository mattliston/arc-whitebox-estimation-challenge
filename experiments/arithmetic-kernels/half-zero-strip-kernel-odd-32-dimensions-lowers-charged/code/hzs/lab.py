"""TAG opus_engineer hzs lab: load a candidate in process (native numpy backend), run its setup (arena + prebind), and expose
helpers to meter kernel calls on random operands. Import this from experiment scripts run under the guard."""
from pathlib import Path
import sys
import time
import types
import numpy as np
import flopscope as flops

P2 = Path(__file__).resolve().parents[3]
H = P2/'codex_resume_20260926_v2'
RRPI = H/'hill_climb_pair_rr_inverse_v1/candidate.py'


def load(candidate=RRPI, setup=True):
    candidate = Path(candidate)
    stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
    sys.modules['whestbench'] = stub
    mod = types.ModuleType('candidate_lab'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
    exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
    est = mod.Estimator()
    if setup:
        est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None,
                                        submission_dir=None, seed=0))
    return mod, est


class Meter:
    """with Meter() as m: ...  -> m.flops, m.seconds, m.ops (charged op count)"""
    def __enter__(self):
        self.ctx = flops.BudgetContext(flop_budget=2**44, quiet=True)
        self.ctx.__enter__()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter()-self.t0
        summary = self.ctx.summary_dict()
        self.flops = int(summary['flops_used'])
        self.ops = len(self.ctx._op_log)
        self.ctx.__exit__(*exc)
        return False


def rand(shape, seed):
    return flops.numpy.asarray(np.random.default_rng(seed).standard_normal(shape).astype(np.float32))
