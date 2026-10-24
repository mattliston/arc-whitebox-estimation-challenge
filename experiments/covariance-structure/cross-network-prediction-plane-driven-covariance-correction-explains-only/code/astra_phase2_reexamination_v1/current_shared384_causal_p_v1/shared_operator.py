"""Pure shared-old384 proposal algebra; inert until passed an explicit torch API.

No imports, payloads, files, module caches, in-place tensor updates, or device
fallbacks. Production callers must fix rank=384, n=1024 and FP32 carrier tensors.
Other ranks/CPU FP64 exist only for the separate tiny mathematical qualifier.
This is a differentiable real-algebra model, NOT a native FP32 byte certificate.
"""

C2_SOURCE_SHA256 = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
PRODUCTION_RANK = 384
PRODUCTION_WIDTH = 1024
FIRST_JOIN = 5
LAST_JOIN = 14


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _tensor(torch, value, shape, anchor, name):
    _require(isinstance(value, torch.Tensor), name + ': explicit tensor')
    _require(tuple(value.shape) == tuple(shape), name + ': shape')
    _require(value.dtype == anchor.dtype and value.device == anchor.device,
             name + ': carrier dtype/device')


def _indices(torch, indices, n, anchor, name):
    _require(isinstance(indices, torch.Tensor) and indices.ndim == 1
             and indices.dtype == torch.int64 and indices.device == anchor.device,
             name + ': ordered same-device int64 vector')
    _require(int(indices.numel()) <= n, name + ': maximum size')
    # Current select() owns uniqueness/range. Do not sort, reconstruct from a
    # mask, or copy index values into Python during checkpoint recomputation.


def _finite(torch, value, name):
    # Detachment is confined to the Boolean diagnostic, never the returned path.
    _require(bool(torch.isfinite(value).all().detach().cpu()), name + ': finite')
    return value


def _qr(torch, Y):
    if Y.device.type == 'mps':
        # An explicit differentiable transfer is part of the declared proposal.
        # No .detach(), NumPy conversion, catch/retry, or implicit MPS fallback.
        Q_cpu, R_cpu = torch.linalg.qr(Y.to(device='cpu'), mode='reduced')
        Q = Q_cpu.to(device=Y.device)
        del Q_cpu, R_cpu
        return Q
    _require(Y.device.type == 'cpu', 'Only explicit CPU or MPS proposal devices')
    Q, R = torch.linalg.qr(Y, mode='reduced')
    del R
    return Q


def push(torch, layer, w32, p_full, lam, previous_indices,
         Q, F, core, cohort_stops, rank=384):
    """Return (pp, U, Fnew, corenew, old_columns), with all inputs unchanged.

    p_full contains every signed column, including BOTH explicit newest legs.
    cohort_stops is the tuple of cumulative signed-column stops after births
    0..layer-1. Q is the previous post-gate basis, F its original signed-factor
    coordinates, and core its lambda-squared Gram core. Before entry5 Q has
    shape(n,0), and F/core have shape(0,0). U is the full-row transported basis.
    """
    _require(type(layer) is int and 0 <= layer <= 15, 'Fixed sixteen-layer order')
    _require(type(rank) is int and rank > 0, 'Positive explicit mathematical rank')
    _require(isinstance(w32, torch.Tensor) and w32.ndim == 2
             and int(w32.shape[0]) == int(w32.shape[1])
             and w32.dtype in (torch.float32, torch.float64)
             and w32.device.type in ('cpu', 'mps'), 'Square real carrier weight')
    n = int(w32.shape[0])
    _require(rank <= n, 'Rank cannot exceed width')
    _require(isinstance(p_full, torch.Tensor) and p_full.ndim == 2,
             'Explicit full signed bank')
    total = int(p_full.shape[1])
    _tensor(torch, p_full, (n, total), w32, 'p_full')
    _tensor(torch, lam, (total,), w32, 'lam')
    _indices(torch, previous_indices, n, w32, 'previous_indices')
    _require(type(cohort_stops) is tuple and len(cohort_stops) == layer,
             'Immutable complete birth history')
    start = 0
    for stop in cohort_stops:
        _require(type(stop) is int and stop > start and (stop-start) % 2 == 0,
                 'Every cohort is an intact nonempty signed pair block')
        start = stop
    _require(start == total, 'Full explicit newest cohort and coefficient roster')

    wp = w32[previous_indices, :].T
    if layer < FIRST_JOIN:
        _tensor(torch, Q, (n, 0), w32, 'empty Q')
        _tensor(torch, F, (0, 0), w32, 'empty F')
        _tensor(torch, core, (0, 0), w32, 'empty core')
        return wp @ p_full[previous_indices, :], None, F, core, 0

    active = int(previous_indices.numel())
    _require(active >= rank, 'Enough actual previous active rows for fixed rank')
    old_columns = 0 if layer == FIRST_JOIN else cohort_stops[min(layer-6, 9)]
    if old_columns:
        _tensor(torch, Q, (n, rank), w32, 'post-gate Q')
        _tensor(torch, F, (rank, old_columns), w32, 'old F')
        _tensor(torch, core, (rank, rank), w32, 'old core')
        Qp = Q[previous_indices, :]
    else:
        _tensor(torch, Q, (n, 0), w32, 'first-join empty Q')
        _tensor(torch, F, (0, 0), w32, 'first-join empty F')
        _tensor(torch, core, (0, 0), w32, 'first-join empty core')
        Qp = None

    if layer <= LAST_JOIN:
        joining = layer-FIRST_JOIN
        stop = cohort_stops[joining]
        _require((0 if joining == 0 else cohort_stops[joining-1]) == old_columns,
                 'Exactly one next old cohort, no age/rank policy change')
        J = p_full[previous_indices, old_columns:stop]
        coefficient = lam[old_columns:stop]
        weight = coefficient*coefficient
        Omega = w32[previous_indices, :rank]
        Y = J @ (weight[:, None]*(J.T @ Omega))
        if old_columns:
            Y = Y + Qp @ (core @ (Qp.T @ Omega))
        Qn = _finite(torch, _qr(torch, Y), 'QR basis')
        _tensor(torch, Qn, (active, rank), w32, 'QR basis')
        Fj = Qn.T @ J
        joined_core = Fj @ (weight[:, None]*Fj.T)
        if old_columns:
            overlap = Qn.T @ Qp
            factors = torch.cat((overlap @ F, Fj), dim=1)
            next_core = (overlap @ (core @ overlap.T)) + joined_core
        else:
            factors, next_core = Fj, joined_core
        old_columns = stop
        Fnew = _finite(torch, factors, 'new F')
        corenew = _finite(torch, next_core, 'new core')
        basis = Qn
    else:
        # Layer15 has all fifteen birth cohorts but only the ten original joins.
        _require(old_columns == cohort_stops[9], 'No terminal join')
        Fnew, corenew, basis = F, core, Qp

    U = wp @ basis
    old = _finite(torch, U @ Fnew, 'full-row old reconstruction')
    # Dense explicit newest legs are only a real-algebra substitute for native
    # shared-newborn multiplication. All original young columns remain present.
    young = wp @ p_full[previous_indices, old_columns:]
    pp = torch.cat((old, young), dim=1)
    _tensor(torch, pp, (n, total), w32, 'complete pushed bank')
    _tensor(torch, Fnew, (rank, old_columns), w32, 'complete old F')
    return pp, U, Fnew, corenew, old_columns


def read(torch, pp, lam, response_indices, live_indices, U, F, old_columns):
    """Full requested response-row by active-column repeated CP plane.

    The complete diagonal and omitted-radial consumers still use full pp in the
    caller. This function neither overwrites that diagonal nor changes masks.
    """
    _require(isinstance(pp, torch.Tensor) and pp.ndim == 2
             and pp.dtype in (torch.float32, torch.float64), 'Full real pushed bank')
    n, total = map(int, pp.shape)
    _tensor(torch, lam, (total,), pp, 'read lambda')
    _indices(torch, response_indices, n, pp, 'response_indices')
    _indices(torch, live_indices, n, pp, 'live_indices')
    _require(type(old_columns) is int and 0 <= old_columns <= total,
             'Read old-column ownership')
    left, right = pp[response_indices, :], pp[live_indices, :]
    if old_columns == 0:
        _require(U is None, 'No factored read before first join')
        _tensor(torch, F, (0, 0), pp, 'pre-join empty read F')
        return (left*left) @ (right*lam).T
    _require(isinstance(F, torch.Tensor) and F.ndim == 2, 'Explicit old read factors')
    rank = int(F.shape[0])
    _tensor(torch, F, (rank, old_columns), pp, 'read F')
    _tensor(torch, U, (n, rank), pp, 'read U')
    old_left = left[:, :old_columns]
    weighted_factors = F*lam[:old_columns][None, :]
    inner = (old_left*old_left) @ weighted_factors.T
    old_plane = inner @ U[live_indices, :].T
    young_left, young_right = left[:, old_columns:], right[:, old_columns:]
    young_plane = (young_left*young_left) @ (young_right*lam[old_columns:]).T
    return young_plane + old_plane
