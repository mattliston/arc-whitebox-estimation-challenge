"""TAG fable-engineer (E-69): torch mirror of S23's NYSTROM old-tier join (porter PT-42 build_join_nystrom.py; same math as the porter's
torch arm C in join_sketch_test.py). In the push's join block (s2card.make_push; native _shared_old384_push) the test matrix
Omega = W[live, :384] becomes the 384 live rows with the largest Gram diagonal d_i = sum_c w_c J_ic^2 (+ (Qp core Qp^T)_ii when an old tier
exists), ranked by a stable argsort of -d (native casts the float32 d to float64 first: exact, same order) and sorted; then
Y = J (w o J[rows]^T) [+ Qp core Qp[rows]^T]. QR, Fj = Qn^T J, the weighted core and the old-tier overlap/factor update are unchanged.
usage: import join_nystrom_torch as jn; jn.install(card, torch)   (after the card is built with fine_ckpt=False; works on the CPU card and
on mps_card cards). Certificate vs native S23 / S23pre: E-69 in RESULT.md."""
import s2card


def make_push(qr, joins):
    def push(torch, layer, w32, p_full, lam, previous_indices, Q, F, core, cohort_stops, rank=384):
        wp = w32[previous_indices, :].T
        before = [c for L_, cs in joins.items() if L_ < layer for c in cs]
        old_columns = cohort_stops[max(before)] if before else 0
        if layer in joins:
            start, stop = old_columns, cohort_stops[max(joins[layer])]
            J = p_full[previous_indices, start:stop]; weight = lam[start:stop] ** 2
            Qp = Q[previous_indices, :] if old_columns else None
            d = (J * J) @ weight                                   # Nystrom row choice (PT-42)
            if old_columns: d = d + ((Qp @ core) * Qp).sum(1)
            rows = torch.sort(torch.argsort(-d.to(torch.float64), stable=True)[:rank]).values
            Y = J @ (weight[:, None] * J[rows, :].T)
            if old_columns: Y = Y + Qp @ (core @ Qp[rows, :].T)
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


def install(card, torch):
    qn = getattr(card.shared['push'], '__qualname__', '')
    assert qn.startswith('make_push'), f'expected the plain s2card push (fine_ckpt=False), got {qn}'
    card.shared['push'] = make_push(card.shared['_qr'], s2card.EVERY3)
    return card
