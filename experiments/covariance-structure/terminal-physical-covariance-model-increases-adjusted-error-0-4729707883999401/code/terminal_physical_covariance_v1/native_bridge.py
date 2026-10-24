"""Native covariance-diagonal forward with a declared algebraic Torch VJP.

The forward is the current paid FNP path, including its actual FP32 rounding.
Backward differentiates diag(W.T D W) in ordinary Torch FP64. It is a
straight-through derivative of the physical algebra, not a derivative of
the discrete rounded implementation. The frozen prefix C is not trained.
"""
import hashlib
import sys
import time
import types
from pathlib import Path
import numpy as np
import flopscope as flops

# Native NumPy/flops must win import resolution; Torch is loaded afterwards.
TORCH_SITE='/Users/matt/anaconda3/lib/python3.11/site-packages'
if TORCH_SITE not in sys.path:sys.path.append(TORCH_SITE)
import torch
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'candidate_thin_star_route_v1.py'
PARENT_SHA='25c22c219420c13db8eae758c16b96a322f24b4de5d40efb43890605b4d3b3ee'
_MODULE=None
_STATS=dict(initializations=0,calls=0,flops=0,setup_wall_s=0.,forward_wall_s=0.)

def initialize():
    global _MODULE
    if _MODULE is not None:return _MODULE
    blob=PARENT.read_bytes()
    assert hashlib.sha256(blob).hexdigest()==PARENT_SHA
    stub=types.ModuleType('whestbench')
    stub.BaseEstimator=type('BaseEstimator',(),{})
    sys.modules.setdefault('whestbench',stub)
    mod=types.ModuleType('physical_covariance_training_bridge')
    mod.__file__=str(PARENT);sys.modules[mod.__name__]=mod
    exec(compile(blob,str(PARENT),'exec'),mod.__dict__)
    t0=time.monotonic()
    est=mod.Estimator()
    est.setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,api_version='2',
               scratch_dir=None,submission_dir=None,seed=0))
    assert mod._WINOGRAD and mod._SCRATCH is not None and 7 in mod._IPT_REGION
    _STATS['setup_wall_s']+=time.monotonic()-t0
    _STATS['initializations']+=1
    _MODULE=mod
    return mod

def native_variance(C,W):
    mod=initialize()
    C=np.asarray(C,dtype=np.float64)
    W=np.asarray(W,dtype=np.float32)
    assert C.shape==W.shape==(1024,1024)
    t0=time.monotonic()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as budget:
        Cf=mod.fnp.asarray(C,dtype=mod.fnp.float64)
        Wf=mod.fnp.asarray(W,dtype=mod.fnp.float32)
        v,_=mod._flms_covariance_diagonal(Wf,Cf)
        result=np.array(v,dtype=np.float64,copy=True)
    assert np.isfinite(result).all()
    _STATS['calls']+=1
    _STATS['flops']+=int(budget.summary_dict()['flops_used'])
    _STATS['forward_wall_s']+=time.monotonic()-t0
    return result

class _NativeVariance(torch.autograd.Function):
    @staticmethod
    def forward(ctx,C,W):
        assert C.device.type==W.device.type=='cpu'
        ctx.save_for_backward(W.detach().double())
        ctx.c_dtype=C.dtype
        result=native_variance(C.detach().double().numpy(),W.detach().float().numpy())
        return torch.from_numpy(result)

    @staticmethod
    def backward(ctx,grad_v):
        (w,)=ctx.saved_tensors
        # Exact derivative of real arithmetic diag(W.T C W), declared STE
        # for the rounded native forward. W is a frozen network input.
        grad_c=(w*grad_v.double()[None,:])@w.T
        return grad_c.to(ctx.c_dtype),None

def variance(a,D):
    C=a['C'].detach().double()+D.double()
    return _NativeVariance.apply(C,a['W'].detach().float())

def stats():
    return dict(_STATS,parent_sha256=PARENT_SHA,numpy_version=np.__version__,
                forward='actual paid FNP covariance diagonal',
                backward='algebraic FP64 straight-through W diag(grad) W.T')
