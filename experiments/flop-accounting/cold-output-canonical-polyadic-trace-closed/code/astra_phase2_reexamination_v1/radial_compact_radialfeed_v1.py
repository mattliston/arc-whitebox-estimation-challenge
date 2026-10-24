"""Compact K4-to-K3 feed with an explicitly active covariance metric.

Trace tensor is Sym(b,B), where B is the carried active covariance block
with cold diagonals excluded and b has active support. Both are pushed
linearly before reading its repeated plane. Deployment reuses the active
covariance product already required by the compact host; no extra GEMM.
"""
import inspect
import radial_binary_compact as base

PREDICT_SOURCE=inspect.getsource(base.predict)
edits={
    "    radial = 0.0":
        "    radial = 0.0\n    b = np.zeros(n)\n    previous_alive = np.ones(n,dtype=bool)",
    "        vv = np.diag(Cpre)":
        "        metric_cov = C.copy()\n        np.fill_diagonal(metric_cov,np.diag(C)*previous_alive)\n        Bpre = (w32.T@metric_cov.astype(np.float32)@w32).astype(np.float64)\n        bpre = w.T@b\n        vv = np.diag(Cpre)",
    "        K21 = cp_pre+star_pre":
        "        K21 = cp_pre+star_pre+np.diag(Bpre)[:,None]*bpre[None,:]+2*Bpre*bpre[:,None]",
    '        birth = aux["pdf"]/aux["sigma"]':
        '        d = aux["pdf"]/aux["sigma"]\n        birth = d*(1+radial/3)\n        b = (g*bpre+radial*d*vv/6)*alive',
    "        residual = target-inherited":
        "        Bpost = C.copy()\n        np.fill_diagonal(Bpost,np.diag(C)*alive)\n        trace_post = np.diag(Bpost)[:,None]*b[None,:]+2*Bpost*b[:,None]\n        residual = target-inherited-trace_post",
    "        rest = target-cp_post":
        "        rest = target-cp_post-trace_post",
    "        radial = radial_new":
        "        previous_alive = alive\n        radial = radial_new",
}
for old,new in edits.items():
    assert PREDICT_SOURCE.count(old)==1,old
    PREDICT_SOURCE=PREDICT_SOURCE.replace(old,new)
env=dict(vars(base))
exec(PREDICT_SOURCE,env)
predict=env["predict"]
