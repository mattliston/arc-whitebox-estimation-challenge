"""Read fewer CP-plane rows, but keep every factor, active state, and diagonal.

This is not active-state or factor-rank pruning. In the exact positive-linear
ReLU limit, the CP plane cancels against inherited carry in the star residual.
Its surviving scalar fourth-moment source is restored by a factor contraction.
At gate >= .99 this is an approximation, tested at one fixed threshold only.
"""
import numpy as np
import radial_compact_feed_fullpair_v1 as base


def omitted_radial_source(pp, lam, diagonal, omitted, alive, aux, host):
    """Exact r=0 marginal-pair source sum; higher rho responses are not claimed."""
    p = pp.astype(np.float64)
    a = (aux["gate"] - host * aux["pdf"] / aux["sigma"]) * omitted
    beta = 2 * (aux["gaussian_mean"] - host * aux["gate"]) * alive
    row_energy = (p * p).T @ a
    column_sum = p.T @ beta
    contraction = np.sum(lam.astype(np.float64) * row_energy * column_sum)
    contraction -= np.sum(diagonal * a * beta)
    return 6 * contraction


source = base.PREDICT_SOURCE
edits = {
    "        pp = w32.T @ p":
        "        response_gate=ndtr(mu/np.sqrt(np.maximum(vv,1e-10)))\n"
        "        response_score=np.where(alive,np.minimum(response_gate,1-response_gate),-1.)\n"
        "        response_indices,response_mask=select(response_score,gate_floor)\n"
        "        omitted=alive & ~response_mask\n"
        "        pp = w32.T @ p",
    "            sub = pp[live_indices]\n            cp_pre[np.ix_(live_indices, live_indices)] = plane(sub, lam)":
        "            left=pp[response_indices]; right=pp[live_indices]\n"
        "            cp_pre[np.ix_(response_indices,live_indices)]=((left*left)@(right*lam).T).astype(np.float64)",
    "        host = m.copy()":
        "        host = m.copy()\n"
        "        if len(lam):\n"
        "            source_sum=omitted_radial_source(pp,lam,diagonal,omitted,alive,aux,host)\n"
        "            denominator=np.trace(C)**2+2*np.sum(C*C) if metric=='covariance' else n*(n+2)\n"
        "            radial_new=radial_new+source_sum/denominator",
    "states.append(dict(layer=l, active=int(alive.sum()), births=len(selected), columns=p.shape[1]))":
        "states.append(dict(layer=l, active=int(alive.sum()), births=len(selected), columns=p.shape[1],plane_rows=len(response_indices)))",
}
for old, new in edits.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new)
env = dict(base.env, omitted_radial_source=omitted_radial_source)
exec(source, env)
predict = env["predict"]
