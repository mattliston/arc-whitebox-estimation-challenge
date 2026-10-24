"""Reuse the fixed fit, adding equation checks without changing its solve."""
import importlib.util
import numpy as np
from common import D

def load_library():
    path=D/'terminal_representation_v1/information.py'
    spec=importlib.util.spec_from_file_location('tlc_fixed_ridge',path)
    lib=importlib.util.module_from_spec(spec);spec.loader.exec_module(lib)
    equations=[]
    def ridge(x,y,w):
        gram=x.T@(x*w[:,None]);gram=(gram+gram.T)*.5
        gram.flat[::len(gram)+1]+=lib.RIDGE*np.sum(w)
        chol=np.linalg.cholesky(gram);rhs=x.T@(w*y)
        beta=np.linalg.solve(chol.T,np.linalg.solve(chol,rhs))
        residual=float(np.linalg.norm(gram@beta-rhs)/max(np.linalg.norm(rhs),1e-300))
        assert np.isfinite(beta).all() and residual<=1e-9,residual
        equations.append(dict(rows=x.shape[0],columns=x.shape[1],normal_equation_relative_residual=residual))
        return beta
    lib.ridge=ridge
    return lib,equations
