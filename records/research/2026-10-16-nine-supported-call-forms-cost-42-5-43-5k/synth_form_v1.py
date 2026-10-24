import os
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator
N = 8000
class Estimator(BaseEstimator):
    def setup(self, ctx): pass
    def predict(self, mlp, budget):
        f = os.environ.get('CC_FORM', 'add_out')
        x = fnp.ones((32, 32), dtype=fnp.float32); y = fnp.ones((32, 32), dtype=fnp.float32); z = fnp.empty((32, 32), dtype=fnp.float32)
        keep = []
        if f == 'add_out':
            for i in range(N): fnp.add(x, y, out=z)
        elif f == 'einsum_rowdot':
            for i in range(N): keep.append(fnp.einsum('ij,ij->i', x, y, optimize=False)); keep.clear() if len(keep) > 64 else None
        elif f == 'einsum_mm':
            for i in range(N): keep.append(fnp.einsum('ij,jk->ik', x, y, optimize=False)); keep.clear() if len(keep) > 64 else None
        elif f == 'getitem':
            for i in range(N): keep.append(x[0:16]); keep.clear() if len(keep) > 64 else None
        elif f == 'reshape':
            for i in range(N): keep.append(fnp.reshape(x, (16, 64))); keep.clear() if len(keep) > 64 else None
        elif f == 'matmul_new':
            for i in range(N): keep.append(fnp.matmul(x, y)); keep.clear() if len(keep) > 64 else None
        elif f == 'sum_axis':
            for i in range(N): keep.append(fnp.sum(x, axis=1)); keep.clear() if len(keep) > 64 else None
        elif f == 'astype':
            for i in range(N): keep.append(x.astype(fnp.float64)); keep.clear() if len(keep) > 64 else None
        elif f == 'add_new':
            for i in range(N): keep.append(fnp.add(x, y)); keep.clear() if len(keep) > 64 else None
        return fnp.zeros((16, 1024), dtype=fnp.float32)
