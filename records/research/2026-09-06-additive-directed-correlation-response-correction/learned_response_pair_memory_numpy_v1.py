"""Pure NumPy predictor for the one accepted response-conditioned pair law.

No file access, data acquisition, fit, loss, thread/device selection or main.
The caller authenticates the pinned protocol and supplies finite, qualified
causal inputs. Shape/dtype checks here do not inspect data-dependent ranges.
"""
import numpy as np


PROTOCOL_NAME = "LEARNED_RESPONSE_PAIR_MEMORY_PROTOCOL_DRAFT_V1.md"
PROTOCOL_SHA256 = "1e277cd16c6596cca947172d0dcd2bd7c21c8962256596ca8ea070c65fc8754c"
DATA_KEYS = frozenset(("V", "b", "d", "g", "post_scale", "query_V", "K", "mask"))
PARAMETER_ORDER = (
    "theta_constant", "theta_u", "theta_b", "theta_u_b", "theta_d_sum",
    "theta_d_product", "theta_g_sum", "theta_g_product", "beta_K", "beta_KA",
)
RESET_INACTIVE_PARAMETER_INDICES = (1, 3)


def _qualified_arrays(data, parameters):
    if set(data) != DATA_KEYS:
        raise ValueError("Predict accepts exactly the eight causal data keys, no loss fields")
    p = np.asarray(parameters)
    if p.shape != (10,) or p.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("parameters must have shape(10,) and dtype float32 or float64")
    arrays = {name: np.asarray(data[name]) for name in DATA_KEYS}
    k = arrays["K"]
    if k.ndim != 2 or k.shape[0] != k.shape[1] or k.shape[0] == 0:
        raise ValueError("K must be a nonempty square matrix")
    n = k.shape[0]
    expected = dict(V=(13, n, n), b=(13, n, n), d=(13, n), g=(13, n),
                    post_scale=(13, n), query_V=(n, n), K=(n, n), mask=(n, n))
    for name, value in arrays.items():
        if value.shape != expected[name]:
            raise ValueError(f"Unexpected {name} shape: {value.shape}")
        wanted = np.dtype("bool") if name == "mask" else p.dtype
        if value.dtype != wanted:
            raise ValueError(f"{name} must have dtype {wanted}; no implicit whole-data cast")
    return arrays, p, n


def _psi(x):
    # Real-equivalent spelling avoids x*x overflow for a finite large x.
    # The caller must separately qualify finite d_i*d_j; no clipping here.
    t = 1.0/(1.0+np.abs(x))
    y = x*t
    return y/np.sqrt(y*y+t*t)


def _sigmoid(x):
    # Algebraically sigmoid, with no overflowing exp(+large) branch.
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0/(1.0+e), e/(1.0+e))


def _update(u, b, d, g, theta):
    dsum = _psi(d[:, None] + d[None, :])
    dproduct = _psi(d[:, None] * d[None, :])
    gsum = g[:, None] + g[None, :]
    gproduct = g[:, None] * g[None, :]
    logit = (theta[0] + theta[1]*u + theta[2]*b + theta[3]*(u*b)
             + theta[4]*dsum + theta[5]*dproduct
             + theta[6]*gsum + theta[7]*gproduct)
    z = _sigmoid(logit)
    return z*u + (1.0-z)*b


def predict(data, parameters, memory=True, *, return_state=False):
    """Return the standardized pre13 offdiagonal correction deltaD.

    Floating data must match parameters' float32/float64 dtype; mask is
    boolean. V0 is a caller-qualified zero matrix and is never consumed.
    b is already psi(B), not the raw source. post_scale is exactly
    g*post_alive*pre_sigma/post_sigma. No target or consumer is accepted.

    return_state=True returns (deltaD, state), where state contains final
    H, query_U before psi, query_A after sym/psi, and the two query features
    F0/F1. These are actual intermediate arrays, not copies or trajectories.
    The function never mutates inputs. RESET has no older-state dependence
    and is exactly the final layer12 update with u=0 under the same theta.
    """
    if not isinstance(memory, (bool, np.bool_)):
        raise TypeError("memory must be boolean")
    a, p, n = _qualified_arrays(data, parameters)
    theta = p[:8]
    zero = np.zeros((n, n), dtype=p.dtype)
    if memory:
        h = _update(zero, a["b"][0], a["d"][0], a["g"][0], theta)
        for layer in range(1, 13):
            v = a["V"][layer]
            upre = (v.T @ h) @ v
            scale = a["post_scale"][layer]
            upost = (scale[:, None]*scale[None, :])*upre
            u = _psi(0.5*(upost+upost.T))
            h = _update(u, a["b"][layer], a["d"][layer], a["g"][layer], theta)
    else:
        h = _update(zero, a["b"][12], a["d"][12], a["g"][12], theta)
    v = a["query_V"]
    query_u = (v.T @ h) @ v
    query_a = _psi(0.5*(query_u+query_u.T))
    f0 = a["mask"]*a["K"]
    # Parentheses fix product -> output mask -> scalar division ordering.
    f1 = (a["mask"]*(a["K"] @ query_a))/np.sqrt(np.asarray(n, dtype=p.dtype))
    delta = p[8]*f0 + p[9]*f1
    if return_state:
        return delta, dict(H=h, query_U=query_u, query_A=query_a, F0=f0, F1=f1)
    return delta
