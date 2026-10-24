"""Independent manual JVP for the unchanged fifth CPU-FP64 operator fixture.

Inert: no imports, files, RNG, shared-helper calls, autograd or finite difference.
An explicitly supplied torch API performs ordinary arithmetic under no_grad.
The caller owns fixture/source authentication, comparison and graph retirement.
"""

SOURCE_OWNERS = {
    'shared_operator.py': '09136ff645869df9d8fae3e2e8c16ff65c777971c308d1ba8c124d35649d2ad2',
    'qualify_operator.py': 'ebc9a93402baefc96dc975d003ea0365c2e41e12223be1a4caa90d9d69df5c0d',
    'qualify_operator_diag.py': '2cbc8ea3aa04d3a2d8b53e95cb4679c48d5f7abfcd97820a16d9b99800a4e35c',
}


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def evaluate(torch, operands, directions, previous, response, live, stops, cot):
    """Return primal/tangent tensors and plain diagnostic metadata; no PASS flag.

    operands/directions each contain (w,p,lambda,Q,F,core). These are SIX
    INDEPENDENT leaves/tangents, even though the base fixture has p_old=QF and
    core=F diag(lambda_old**2) F.T. Only the fixed fifth layer6 fixture is legal.
    No value is compared to the previously observed derivative inside this file.
    """
    _require(len(operands) == len(directions) == 6, 'Six independent tensor blocks')
    _require(type(stops) is tuple and stops == (4, 8, 12, 16, 20, 24),
             'Original fifth fixture: layer6, four columns per cohort')
    shapes = ((12, 12), (12, 24), (24,), (12, 3), (3, 4), (3, 3))
    for values in (operands, directions):
        for value, shape in zip(values, shapes):
            _require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape
                     and value.dtype == torch.float64 and value.device.type == 'cpu',
                     'Exact original CPU-FP64 block shape')
    for indices, expected in ((previous, (9, 0, 7, 2, 11, 5, 4, 8, 1, 6)),
                              (response, (10, 1, 7, 3, 9)),
                              (live, (11, 4, 0, 6, 2, 7, 5))):
        _require(isinstance(indices, torch.Tensor) and indices.dtype == torch.int64
                 and indices.device.type == 'cpu' and indices.ndim == 1
                 and tuple(indices.tolist()) == expected, 'Original exact ordered index vector')
    _require(isinstance(cot, torch.Tensor) and tuple(cot.shape) == (12, 24)
             and cot.dtype == torch.float64 and cot.device.type == 'cpu',
             'Original fixed pp cotangent')

    with torch.no_grad():
        for value in (*operands, *directions, cot):
            _require(bool(torch.isfinite(value).all()), 'Finite supplied fixture/direction')
        w, p, lam, Q, F, core = operands
        dw, dp, dlam, dQ, dF, dcore = directions
        start, stop, rank = 4, 8, 3
        J, dJ = p[previous, start:stop], dp[previous, start:stop]
        Qp, dQp = Q[previous], dQ[previous]
        weight = lam[start:stop].square()
        dweight = 2*lam[start:stop]*dlam[start:stop]
        JW = J*weight
        dJW = dJ*weight + J*dweight
        H = Qp @ core
        dH = dQp @ core + Qp @ dcore

        # Independent DENSE previous-row Gram and full product-rule tangent.
        # dcore is deliberately not symmetrized or derived from F/lambda.
        A = JW @ J.T + H @ Qp.T
        dA = dJW @ J.T + JW @ dJ.T + dH @ Qp.T + H @ dQp.T
        Omega, dOmega = w[previous, :rank], dw[previous, :rank]
        Y = A @ Omega
        dY = dA @ Omega + A @ dOmega

        # Manual reduced-QR differential; torch QR is used only for its PRIMAL.
        N, R = torch.linalg.qr(Y, mode='reduced')
        diagonal = R.diagonal().abs()
        qr_ratio = float(diagonal.min()/diagonal.max())
        _require(qr_ratio > 1e-6, 'Original full-column QR diagonal-ratio guard')
        # B=dY R^{-1}, via a transposed triangular solve, not an explicit inverse.
        B = torch.linalg.solve_triangular(R.T, dY.T, upper=False).T
        M = N.T @ B
        lower = torch.tril(M, diagonal=-1)
        K = lower-lower.T
        dN = B-N @ M+N @ K
        dR = (M-K) @ R

        # F is independent of p_old. Never substitute projection @ p_old here.
        overlap = N.T @ Qp
        doverlap = dN.T @ Qp + N.T @ dQp
        Fj = N.T @ J
        dFj = dN.T @ J + N.T @ dJ
        Fnew = torch.cat((overlap @ F, Fj), dim=1)
        dFnew = torch.cat((doverlap @ F + overlap @ dF, dFj), dim=1)
        # Dense contraction is algebraically identical for independent core.
        AN = A @ N
        corenew = N.T @ AN
        dcorenew = dN.T @ AN + N.T @ (dA @ N) + N.T @ (A @ dN)
        wp, dwp = w[previous].T, dw[previous].T
        U = wp @ N
        dU = dwp @ N + wp @ dN
        young, dyoung = p[previous, stop:], dp[previous, stop:]
        pp = torch.cat((U @ Fnew, wp @ young), dim=1)
        dpp = torch.cat((dU @ Fnew + U @ dFnew,
                         dwp @ young + wp @ dyoung), dim=1)

        # Independent FULL signed-bank plane, not the helper's reassociation.
        left, right = pp[response], pp[live]
        dleft, dright = dpp[response], dpp[live]
        left2 = left.square()
        weighted = left2*lam
        dweighted = 2*left*dleft*lam + left2*dlam
        plane = weighted @ right.T
        dplane = dweighted @ right.T + weighted @ dright.T
        value = ((pp*cot).sum() + .07*Fnew.square().sum()
                 + .03*corenew.square().sum() + .11*plane.square().sum())
        components = dict(pp=(dpp*cot).sum(),
                          F=.14*(Fnew*dFnew).sum(),
                          core=.06*(corenew*dcorenew).sum(),
                          plane=.22*(plane*dplane).sum())
        directional = components['pp']+components['F']+components['core']+components['plane']

        def relative(residual, reference):
            return float(torch.linalg.vector_norm(residual) /
                         torch.linalg.vector_norm(reference).clamp_min(1e-20))

        diagnostics = dict(
            qr_diagonal_ratio=qr_ratio,
            gram_tangent_norm=float(torch.linalg.vector_norm(dA)),
            qr_primal_residual=relative(Y-N @ R, Y),
            triangular_solve_residual=relative(B @ R-dY, dY),
            qr_tangent_residual=relative(dY-dN @ R-N @ dR, dY),
            qr_orthogonality_tangent_norm=float(torch.linalg.vector_norm(N.T @ dN+dN.T @ N)),
            qr_lower_triangular_tangent_norm=float(torch.linalg.vector_norm(torch.tril(dR, diagonal=-1))),
            directional_components={key:float(v) for key,v in components.items()},
            autograd_used=False, finite_difference_used=False,
            old_state_tangents_independent=True, full_signed_plane_used=True)
        primal = dict(pp=pp, U=U, F=Fnew, core=corenew, plane=plane)
        tangent = dict(pp=dpp, U=dU, F=dFnew, core=dcorenew, plane=dplane)
        for v in (value, directional, *primal.values(), *tangent.values()):
            _require(not v.requires_grad and bool(torch.isfinite(v).all()),
                     'Finite manual, graph-free primal/tangent')
        return dict(value=value, directional=directional, primal=primal,
                    tangent=tangent, diagnostics=diagnostics)
