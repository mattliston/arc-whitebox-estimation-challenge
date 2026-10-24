# TAG opus_engineer a58 (2026-10-11): the boundary's residual per request with (almost) no estimator Python: a stub Estimator whose
# predict makes N requests of ONE kind in a tight loop, each result proxy dropped at once, on tiny arrays (negligible server work).
# The kind follows the network the harness passes (its name ends in the rank): 35 = an in-place addition, 36 = a basic-slice view,
# 45 = a reshape view, 100 = nothing (the fixed part: two allocations and the fetch). residual_instructions / N = the per-request
# floor of flopscope's own client code outside its spans (wrapper, caller check, proxy finalizer) plus the loop's few bytecodes.
# Diagnostic only: never a submission.
import flopscope.numpy as fnp
from whestbench import BaseEstimator

N = 10000


class Estimator(BaseEstimator):
    def setup(self, context):
        return None

    def predict(self, mlp, budget):
        kind = int(str(mlp.name).rsplit('-', 1)[-1])
        x = fnp.zeros((16, 1024), dtype=fnp.float32)
        y = fnp.ones((16, 1024), dtype=fnp.float32)
        if kind == 35:
            for _ in range(N):
                fnp.add(x, y, out=x)
        elif kind == 36:
            for _ in range(N):
                v = x[1:3]
        elif kind == 45:
            for _ in range(N):
                v = fnp.reshape(x, (16, 32, 32))
        return x
