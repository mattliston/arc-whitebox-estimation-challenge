"""Leading non-Gaussian cubic-vertex contribution at fixed natural strength.

For Hermite H3_i, Cum(H3_i,X_j,X_k)=3(T_iij*C_ik+T_iik*C_ij).
Its coefficient e_i/6 therefore adds e_i/2 times that pair. Replacing the
birth covariance column C_i by C_i+(e_i/(2*d_i))*T_ii: captures this term
without adding factors. The completion also adds an O(T^2) term, explicitly
an approximation. All current repeated entries remain restored by star.
"""
import inspect
import radial_binary_compact as base

source=inspect.getsource(base.predict)
old="        bp = g[:, None]*Cpre"
new="        eta = -mu/(2*np.maximum(vv,1e-10))\n        bp = g[:,None]*(Cpre+K21.T*eta[None,:])"
assert source.count(old)==1
source=source.replace(old,new)
env=dict(vars(base))
exec(source,env)
predict=env["predict"]
