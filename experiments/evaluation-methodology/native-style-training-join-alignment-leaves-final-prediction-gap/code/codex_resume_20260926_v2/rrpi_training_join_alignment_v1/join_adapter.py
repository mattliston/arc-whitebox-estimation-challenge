"""Torch join specification of RRPI; ordinary products, not native rounding."""

def install(cd, joins):
    ranks = {7: 384, 10: 320, 13: 384}
    assert set(joins) == set(ranks)

    def push(t, layer, w32, p_full, lam, previous_indices, Q, F, core, cohort_stops, rank=384):
        wp = w32[previous_indices, :].T
        before = [c for L, cs in joins.items() if L < layer for c in cs]
        old_columns = cohort_stops[max(before)] if before else 0
        if layer in joins:
            rank = ranks[layer]
            start, stop = old_columns, cohort_stops[max(joins[layer])]
            J = p_full[previous_indices, start:stop]
            weight = lam[start:stop] ** 2
            Qp = Q[previous_indices, :] if old_columns else None
            d = (J * J) @ weight
            if old_columns:
                d = d + ((Qp @ core) * Qp).sum(1)
            rows = t.sort(t.argsort(-d.to(t.float64), stable=True)[:rank]).values
            Y = J @ (weight[:, None] * J[rows, :].T)
            if old_columns:
                Y = Y + Qp @ (core @ Qp[rows, :].T)
            assert Y.dtype == t.float32
            Qn, R = t.linalg.qr(Y, mode='reduced')
            if layer < 13:
                K = Y[rows, :].to(t.float64)
                K = (K + K.T) * .5
                R64 = R.to(t.float64)
                next_core = (R64 @ t.linalg.solve(K, R64.T)).to(t.float32)
                next_core = (next_core + next_core.T) * .5
            else:
                # Native uses None; the legacy card logs core.shape.
                next_core = t.empty((0, 0), dtype=t.float32)
            Fj = Qn.T @ J
            if old_columns:
                overlap = Qn.T @ Qp
                F = t.cat((overlap @ F, Fj), dim=1)
            else:
                F = Fj
            core, basis, old_columns = next_core, Qn, stop
        elif old_columns:
            basis = Q[previous_indices, :]
        else:
            return wp @ p_full[previous_indices, :], None, F, core, 0
        U = wp @ basis
        return t.cat((U @ F, wp @ p_full[previous_indices, old_columns:]), dim=1), U, F, core, old_columns

    cd.shared['push'] = push
