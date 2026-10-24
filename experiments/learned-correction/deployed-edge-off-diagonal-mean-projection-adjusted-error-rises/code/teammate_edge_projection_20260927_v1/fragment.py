

# Fixed current-edge geometry ablation. All source weights remain in the
# byte-identical thin-star prefix. This is not a new fitted covariance law.
_EP_DIAGONAL_SOLVE_MIN = 2.0 ** -20
_ep_message_edge_parent = _message_edge_source


def _ep_project_edge(edge, m):
    """Preserve edge diagonal; project its off-diagonal part to D @ m == 0.

    The deployed parent edge is symmetric. The diagonal-plus-rank-one
    multiplier formula is the existing covariance-supplier projector. A
    vanishing mean makes the constraint vacuous. Outside its well-conditioned
    positive-diagonal domain, leave the original edge unchanged; no ridge,
    clipping, solve, or alternative learned model is introduced.
    """
    norm_sq = fnp.sum(m * m)
    if not bool(norm_sq > 0):
        return edge
    h = m / fnp.sqrt(norm_sq)
    d = 1 - 2 * h * h
    if not bool(fnp.all(d > _EP_DIAGONAL_SOLVE_MIN)):
        return edge
    diagonal = fnp.diag(edge).copy()
    A = edge.copy()
    fnp.fill_diagonal(A, 0)
    b = A @ h
    v = h / d
    bd = b / d
    u = 2 * (bd - v * ((h @ bd) / (1 + h @ v)))
    projected = A - (fnp.outer(u, h) + fnp.outer(h, u)) * 0.5
    fnp.fill_diagonal(projected, diagonal)
    return projected


def _message_edge_source(layer, m, K21, aux, w, state, P, context):
    edge = _ep_message_edge_parent(layer, m, K21, aux, w, state, P, context)
    return _ep_project_edge(edge, m)
