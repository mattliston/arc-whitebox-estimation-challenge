"""TAG fable-physicist (RF6-3): old-tier push with a per-layer JOIN SCHEDULE and a per-layer FRAME RANK, for the torch cards
(s2card / s6bcard; drop-in for card.shared['push']). Same algebra as fable_frame_v1/join_schedule_test.py make_push: at a join layer the
new frame Qn = qr(G Omega), Omega = W[previous, :rank], G = the lam^2-weighted Gram of the joining columns + the old core; old coordinates
rotate by Qn^T Qp; U = W[previous]^T Qn pushes the whole frame. If fewer than `rank` units are alive the frame is the whole alive space.
Presets (spec strings for make_from_spec):
  'every3'            {7: 0-2, 10: 3-5, 13: 6-8} at rank 384 (S6b; bit-exact with s2card.make_push)
  'every3:320'        the same at rank 320 (covhunter's milder old-tier cut)
  'fold<L>:<r0>/<r>'  every3's L7 join at rank r0 (if L > 7), then at layer L fold ALL young cohorts (every cohort born before L), and at
                      every later layer (to 15) fold the cohort born one layer earlier, at rank r: no exact young columns at layers >= L.
"""


def make_push(qr, joins, ranks):
    def push(torch, layer, w32, p_full, lam, previous_indices, Q, F, core, cohort_stops, rank=384):
        wp = w32[previous_indices, :].T
        before = [c for L_, cs in joins.items() if L_ < layer for c in cs]
        old_columns = cohort_stops[max(before)] if before else 0
        if layer in joins:
            r = ranks[layer]
            start, stop = old_columns, cohort_stops[max(joins[layer])]
            J = p_full[previous_indices, start:stop]; weight = lam[start:stop] ** 2
            Omega = w32[previous_indices, :r]
            Y = J @ (weight[:, None] * (J.T @ Omega))
            Qp = Q[previous_indices, :] if old_columns else None
            if old_columns: Y = Y + Qp @ (core @ (Qp.T @ Omega))
            Qn = qr(torch, Y)
            Fj = Qn.T @ J; joined = Fj @ (weight[:, None] * Fj.T)
            if old_columns:
                overlap = Qn.T @ Qp
                F = torch.cat((overlap @ F, Fj), dim=1); core = overlap @ (core @ overlap.T) + joined
            else:
                F, core = Fj, joined
            basis, old_columns = Qn, stop
        elif old_columns:
            basis = Q[previous_indices, :]
        else:
            return wp @ p_full[previous_indices, :], None, F, core, 0
        U = wp @ basis
        return torch.cat((U @ F, wp @ p_full[previous_indices, old_columns:]), dim=1), U, F, core, old_columns
    return push


def schedule(spec):
    """spec -> (joins {layer: [cohorts]}, ranks {layer: rank})."""
    if spec.startswith('every3'):
        r = int(spec.split(':')[1]) if ':' in spec else 384
        joins = {7: [0, 1, 2], 10: [3, 4, 5], 13: [6, 7, 8]}
        return joins, {l: r for l in joins}
    if spec.startswith('fold'):
        head, rr = spec[4:].split(':'); L = int(head); r0, r = (int(x) for x in rr.split('/'))
        joins, ranks = {}, {}
        if L > 7:
            joins[7] = [0, 1, 2]; ranks[7] = r0
        first = 3 if L > 7 else 0
        joins[L] = list(range(first, L)); ranks[L] = r          # every cohort born before L (cohort k is born at layer k)
        for l in range(L + 1, 16):
            joins[l] = [l - 1]; ranks[l] = r
        return joins, ranks
    raise ValueError(spec)


def make_from_spec(qr, spec):
    joins, ranks = schedule(spec)
    return make_push(qr, joins, ranks)
