"""TAG opus_engineer a33 v2 (2026-10-11, a62): = star_ss_torch.py (v1, unchanged and still imported by the trainer's engine) plus
rank -1, the native SS v3 form "one direction along the row sums": a = A 1, s2 = ((W o W)' A' a) (W' a)' / a'a (P1 = W' a,
T2 = (W o W)' (A' a) / a'a). In v1 a rank of -1 would have taken argsort[:-1], i.e. all but one column: a near-exact star.
NOT RUN BY THE ENGINEER (no torch in this lane): check it against the native prediction of a62_modules/cand_s47e_ss-1.py on
network 7100136 (a62_modules/gate_rdz_ss-1_1_pred.npz) before use.

v1 docstring follows.
TAG opus_engineer a33 (2026-10-02): torch drop-in of the native lever SS<rank> (O/engineer/a33_starsum/fragment_ss_v2.py), for the
s6b cards (trainer, physicist, lead). NOT RUN BY THE ENGINEER (no torch in this lane): whoever runs it should first check it
against the native predictions on a shared network (see 'parity' below).

What it changes. The card's star term is s2 = (W o W)' A' W / 3 with the star plane A in rank-32 factors Q (Q' A) (the porter's
star_nys_test.install_star, arm 'nysp'). Because squared weights are positive, s2 splits into m (W' A 1)' + R A' W (m = each unit's
mean squared weight): the first part needs only the ROW SUMS of A and holds 99.4-99.8% of s2's power at layers 4-14. The factors
miss 1-2% of the row sums; here that miss goes through W as one more column of the two factors, so the first part is exact, and
the rank of the factors is a free number (0 = the row-sum part alone).

usage, after the card is built with its usual hooks (jn.install, SN.install_star, gl.install, ps.install, ...):
    import star_ss_torch                       # sys.path: O/engineer/a33_starsum
    star_ss_torch.install(cd, torch, rank)     # rank 32, 16, 8, ... or 0; replaces card.env['_flms_star'] and ['_flms_star_diagonal']
The basis Q is built as the native file builds it (the `rank` columns of largest norm, one power step, no SVD truncation) and, like
the porter's hook, without gradient; the factors Q' A, the row sums and the weights carry gradients.
Note: the porter's hook builds Q from 48 columns and truncates to 32 by SVD; the native file (since the hill-climb step before
RRPI) takes 32 columns and no truncation. This drop-in follows the native file.

parity: native predictions of the S38 file and of the SS arms on fresh list a (ranks 651 378 424 328 287 641 548 557 316 390) are
O/engineer/logs/quick_s38_dials2_fresh_a_pred_<rank>.npy and quick_s38_ss<rank>_fresh_a_pred_<rank>.npy (float32, 16 x 1024,
post-activation means of every layer). The torch change (this hook against the card's own star, same dials: candidate 2) should
match the native change to the rounding level of two native builds (4-5e-6 rms at the final layer)."""


def install(card, t, rank):
    K = int(rank)

    def factors(wt, A32):
        rows = A32.sum(1)
        mean = (wt * wt).sum(1) / A32.shape[0]
        if K == 0:
            return (wt @ rows)[:, None], mean[:, None]
        if K < 0:   # v2: one direction along the row sums (native _ss_factors, K < 0)
            return (wt @ rows)[:, None], (((wt * wt) @ (rows @ A32)) / (rows @ rows))[:, None]
        with t.no_grad():
            Ad = A32.detach()
            cn = (Ad.to(t.float64) ** 2).sum(0)
            cols = t.sort(t.argsort(-cn, stable=True)[:K]).values
            Q = t.linalg.qr(Ad[:, cols])[0]
            Q = t.linalg.qr(Ad @ t.linalg.qr(Ad.T @ Q)[0])[0]
        B = Q.T @ A32
        miss = rows - Q @ B.sum(1)
        return t.cat([wt @ Q, (wt @ miss)[:, None]], 1), t.cat([(wt * wt) @ B.T, mean[:, None]], 1)

    def star(w, A):
        wt = w.T
        P1, T2 = factors(wt, A if A.dtype == t.float32 else A.to(t.float32))
        result = ((T2 @ P1.T) / 3).to(t.float64).clone()
        result.diagonal().copy_((P1 * T2).sum(dim=1, dtype=t.float64))
        return result

    def star_diagonal(w, A):
        P1, T2 = factors(w.T, A.to(t.float32))
        return (P1 * T2).sum(dim=1, dtype=t.float64)

    card.env['_flms_star'] = star
    card.env['_flms_star_diagonal'] = star_diagonal
    return card
