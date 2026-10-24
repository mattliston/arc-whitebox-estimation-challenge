"""Independent NumPy accuracy reference for the scalar-only final read."""
import inspect
import radial_binary_compact as base
from compact_scalar_final_v1 import nonlinear_scalar

source = inspect.getsource(base.predict)
old = "gate_floor=.001):"
assert source.count(old) == 1
source = source.replace(old,"gate_floor=.01, final_hook=None):")
old = "        Cpre = (w32.T @ C.astype(np.float32) @ w32).astype(np.float64) if cov32 else w.T @ C @ w"
new = '''        if l == depth-1:
            pushed = C.astype(np.float32)@w32
            v = np.sum(w32.astype(np.float64)*pushed.astype(np.float64),axis=0)
            pp = w32.T@p
            k3 = np.sum(pp.astype(np.float64)**3*lam.astype(np.float64),axis=1)+star_push(W[l],star,diagonal_only=True)
            m,aux = nonlinear_scalar(mu,v,k3,radial*v*v)
            if final_hook is not None:
                m = m+np.asarray(final_hook(l,m,aux,W[l]),np.float64)
            means.append(m.copy())
            break
'''+old
assert source.count(old) == 1
source = source.replace(old,new)
env = dict(vars(base), nonlinear_scalar=nonlinear_scalar)
exec(source,env)
predict = env["predict"]
