"""Independent scalar-only final read, including the carried trace tensor.

The prefix is the frozen source/full-pair host. No matrix pair output or
learned correction is used in constructing the current final-layer input.
This NumPy reference establishes accuracy, not a deployment cost claim.
"""
import radial_compact_feed_fullpair_v1 as base
from compact_scalar_final_v1 import nonlinear_scalar

source = base.PREDICT_SOURCE
old = "gate_floor=.001):"
assert source.count(old) == 1
source = source.replace(old, "gate_floor=.01, final_hook=None):")
old = "        Cpre = (w32.T @ C.astype(np.float32) @ w32).astype(np.float64) if cov32 else w.T @ C @ w"
new = '''        if l == depth-1:
            pushed = C.astype(np.float32)@w32
            v = np.sum(w32.astype(np.float64)*pushed.astype(np.float64),axis=0)
            metric_cov = C.copy()
            np.fill_diagonal(metric_cov,np.diag(C)*previous_alive)
            metric_push = metric_cov.astype(np.float32)@w32
            bv = np.sum(w32.astype(np.float64)*metric_push.astype(np.float64),axis=0)
            bpre = w.T@b
            pp = w32.T@p
            k3 = np.sum(pp.astype(np.float64)**3*lam.astype(np.float64),axis=1)
            k3 = k3+star_push(W[l],star,diagonal_only=True)+3*bv*bpre
            m,aux = nonlinear_scalar(mu,v,k3,radial*v*v)
            if final_hook is not None:
                m = m+np.asarray(final_hook(l,m,aux,W[l]),np.float64)
            means.append(m.copy())
            break
'''+old
assert source.count(old) == 1
source = source.replace(old,new)
env = dict(base.env, nonlinear_scalar=nonlinear_scalar)
exec(source,env)
predict = env["predict"]
