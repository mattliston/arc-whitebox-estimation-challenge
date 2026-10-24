"""Pure differentiable predictor; no NumPy model, IO, loss, fit or main.

All inputs/parameters are supplied by the caller on one device and dtype.
No device selection, threadpoolctl, parameter detachment, checkpoint shadow,
target field or hidden training operation is present in this module.
"""
import math
import torch


PROTOCOL_NAME = "LEARNED_RESPONSE_PAIR_MEMORY_PROTOCOL_DRAFT_V1.md"
PROTOCOL_SHA256 = "1e277cd16c6596cca947172d0dcd2bd7c21c8962256596ca8ea070c65fc8754c"
DATA_KEYS = frozenset(("V", "b", "d", "g", "post_scale", "query_V", "K", "mask"))
PARAMETER_ORDER = (
    "theta_constant", "theta_u", "theta_b", "theta_u_b", "theta_d_sum",
    "theta_d_product", "theta_g_sum", "theta_g_product", "beta_K", "beta_KA",
)
RESET_INACTIVE_PARAMETER_INDICES = (1, 3)


def _qualified_tensors(data, parameters):
    if set(data) != DATA_KEYS:
        raise ValueError("Predict accepts exactly the eight causal data keys, no loss fields")
    p = parameters
    if not isinstance(p, torch.Tensor) or tuple(p.shape) != (10,) or \
            p.dtype not in (torch.float32, torch.float64):
        raise ValueError("parameters must be a float32/float64 tensor of shape(10,)")
    if not all(isinstance(value, torch.Tensor) for value in data.values()):
        raise TypeError("All data fields must be tensors, including boolean mask")
    k = data["K"]
    if k.ndim != 2 or k.shape[0] != k.shape[1] or k.shape[0] == 0:
        raise ValueError("K must be a nonempty square matrix")
    n = k.shape[0]
    expected = dict(V=(13, n, n), b=(13, n, n), d=(13, n), g=(13, n),
                    post_scale=(13, n), query_V=(n, n), K=(n, n), mask=(n, n))
    for name, value in data.items():
        if tuple(value.shape) != expected[name]:
            raise ValueError(f"Unexpected {name} shape: {tuple(value.shape)}")
        wanted = torch.bool if name == "mask" else p.dtype
        if value.dtype != wanted or value.device != p.device:
            raise ValueError(f"{name} must share parameter device and requested dtype {wanted}")
    return p, n


def _psi(x):
    # Same real-equivalent, finite-large-input spelling as the NumPy model.
    # Finite d_i*d_j belongs to input qualification, not a clipping repair.
    t = 1.0/(1.0+torch.abs(x))
    y = x*t
    return y/torch.sqrt(y*y+t*t)


def _update(u, b, d, g, theta):
    dsum = _psi(d[:, None]+d[None, :])
    dproduct = _psi(d[:, None]*d[None, :])
    gsum = g[:, None]+g[None, :]
    gproduct = g[:, None]*g[None, :]
    logit = (theta[0] + theta[1]*u + theta[2]*b + theta[3]*(u*b)
             + theta[4]*dsum + theta[5]*dproduct
             + theta[6]*gsum + theta[7]*gproduct)
    z = torch.sigmoid(logit)
    return z*u + (1.0-z)*b


def predict(data, parameters, memory=True, *, return_state=False):
    """Return deltaD, or (deltaD, {H,query_U,query_A,F0,F1}).

    State tensors remain attached to the same autograd graph. H is the
    final post12 state, not a saved trajectory. The caller owns all finite
    input/range and provenance checks, including V0=0, b=psi(B), and
    post_scale=g*post_alive*pre_sigma/post_sigma. RESET skips earlier
    independent zero-U updates; theta[1] and theta[3] have zero gradients.
    """
    if not isinstance(memory, bool):
        raise TypeError("memory must be boolean")
    p, n = _qualified_tensors(data, parameters)
    theta = p[:8]
    zero = torch.zeros_like(data["K"])
    if memory:
        h = _update(zero, data["b"][0], data["d"][0], data["g"][0], theta)
        for layer in range(1, 13):
            v = data["V"][layer]
            upre = (v.T @ h) @ v
            scale = data["post_scale"][layer]
            upost = (scale[:, None]*scale[None, :])*upre
            u = _psi(0.5*(upost+upost.T))
            h = _update(u, data["b"][layer], data["d"][layer], data["g"][layer], theta)
    else:
        h = _update(zero, data["b"][12], data["d"][12], data["g"][12], theta)
    v = data["query_V"]
    query_u = (v.T @ h) @ v
    query_a = _psi(0.5*(query_u+query_u.T))
    f0 = data["mask"]*data["K"]
    f1 = (data["mask"]*(data["K"] @ query_a))/math.sqrt(n)
    delta = p[8]*f0+p[9]*f1
    if return_state:
        return delta, dict(H=h, query_U=query_u, query_A=query_a, F0=f0, F1=f1)
    return delta
