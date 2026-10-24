"""TAG opus-physicist F4: drop-in for the torch research card (HookCard built by s6bcard_loop5js / pcard.build / the trainer).
FOURTH-CUMULANT STAR ON THE BANK'S YOUNG COLUMNS, FED INTO BIRTHS BY AN IN-PLACE COLUMN SHIFT.

Physics. The ReLU at a hub c born at layer j makes, besides the third cumulant the card already banks, a fourth cumulant whose leading
part is the three-leg star  kappa4(x_i, x_j, x_k, x_c) = E f'''(z_c) p_ic p_jc p_kc  (p = the same birth legs). On the card's column
pair (legs + t e_c, third-order weights +-birth/gap) this is the fourth-order weight lam4 = lam * A with ONE scalar per pair,
     A = E f'''(z_c) / birth_c = -alpha_c / (sigma_c (1 + radial/3))                          (fixed at birth).
A fourth cumulant K4 = sum_r lam4_r q_r^(x)4 on pushed columns q_r feeds the next layer's third-order births as
     (d_c/2) g_i g_j Q_ijcc = sum_r lam4_r Sym3(u_r, u_r, w_r),   u_r = g o q_r (the carried post column),  w_r = (d/2) o q_r o q_r,
which is, to first order, an in-place shift of the carried column:  u_r -> u_r + (lam4_r / lam_r) w_r = u_r + theta A_r w_r.
The shift also changes the bank's post PLANE (entries with a repeated index), by P_ik = sum_r c_r (u_ri^2 w_rk + 2 u_ri w_ri u_rk),
c_r = theta lam_r A_r. That plane content is not wanted (the rule's K22 / K31 planes are poor; un-anchored the card is ruined), so it is
removed from the star ("anchor"):
     cheap : P ~ v_i beta_k + 2 C_ik beta_i,  beta_k = sum_i v_i P_ik / sum_i v_i^2  (matrix-vector passes over the shifted columns),
             diagonal exact: P_ii = 3 sum_r c_r u_ri^2 w_ri;   v = diag C, C = carried post covariance
     exact : P computed in full (costs a plane read; reference only)
Applied at layers 0..14 to the carried columns born at most `age` layers earlier and not yet joined into the old-tier frame.

install(cd, torch, theta=0.4, age=2, anchor='cheap', offset=0.0)   theta, offset: float or 0-dim tensor (trainable dials); returns cd.
   offset e: birth scalar A + e / sigma_c (open loop the true feed asks for e of about 0.15-0.25: the (2,1,1) birth coefficient; default 0).
install_bsrc(cd, torch, kappa)   scales only the SOURCE term radial*d*v/6 of the b carrier by kappa (the existing dial bscale multiplies
   the whole carried b every layer); kappa: float or 0-dim tensor. Install it after the card is built (it chains on the b_post hook).
Checkpoint-safe: the per-column A is kept per layer in a dict (recomputing a step rewrites the same entry) and is detached.
Certificates: t4_cert_shift.py (algebra, 1e-15), f4_cert.py (this module against t3_screen.py; backward pass finite).
Measured (2026-10-01, current torch card, no refit, learned on): see RESULT.md section T4."""


def install(cd, torch, theta=0.4, age=2, anchor='cheap', offset=0.0):
    t = torch; side = dict(base={-1: t.zeros(0, dtype=t.float32)}, isg={-1: t.zeros(0, dtype=t.float32)})
    push0 = cd.shared['push']

    def push(tt, layer, w, p, lam, previous, Q, F, core, stops, rank=384):
        r = push0(tt, layer, w, p, lam, previous, Q, F, core, stops, rank=rank)
        side['pp'], side['oc'] = r[0], int(r[4])
        return r
    cd.shared['push'] = push
    sel0 = cd.select

    def select(score):
        r = sel0(score); side['sel'] = r[0]
        return r
    cd.select = select
    prev = cd.hooks.get('state_out')

    def state_out(l, out, ctx):
        if prev is not None: out = prev(l, out, ctx)
        if l > 14: return out
        aux = ctx['aux']; C, p, lam, star, alive, stops = out[1], out[2], out[3], out[4], out[6], out[12]
        A = side['base'][l - 1] + offset * side['isg'][l - 1]; pp = side['pp']; oc = side['oc']; nc = int(pp.shape[1])
        assert A.numel() == nc, (l, A.numel(), nc)
        if l - age >= 1: oc = max(oc, int(stops[l - age - 1]))          # cohort j ends at column stops[j]; cohort j is l - j layers old
        if nc > oc:
            dh = ((aux['pdf'] / aux['sigma'] / 2) * alive).to(t.float32)
            u = p[:, oc:nc]; om = dh[:, None] * (pp[:, oc:] * pp[:, oc:])
            new = u + om * (theta * A[oc:])[None, :].to(t.float32)
            c = (lam[oc:nc] * theta * A[oc:]).to(t.float64)
            if anchor == 'cheap':
                v = t.diagonal(C); u64, o64 = u.to(t.float64), om.to(t.float64); uu = u64 * u64
                vP = o64 @ (c * (v @ uu)) + 2 * (u64 @ (c * (v @ (u64 * o64))))
                beta = vP / (v * v).sum()
                dP = t.diagonal_scatter(v[:, None] * beta[None, :] + 2 * C * beta[:, None], 3 * (uu * o64) @ c)
            elif anchor == 'exact':
                lm = lam[oc:nc].to(t.float64); o64, n64 = u.to(t.float64), new.to(t.float64)
                dP = ((n64 * n64) * lm[None, :]) @ n64.T - ((o64 * o64) * lm[None, :]) @ o64.T
            else: raise ValueError(anchor)
            star = star - (3 * dP.T - 2 * t.diag(t.diagonal(dP))).to(star.dtype)
            p = t.cat((p[:, :oc], new, p[:, nc:]), dim=1)
        sel = side['sel']
        An = (-aux['alpha'][sel] / (aux['sigma'][sel] * (1 + ctx['radial'] / 3))).detach()
        side['base'][l] = t.cat((side['base'][l - 1], An.to(t.float32), An.to(t.float32)))
        Is = (1.0 / aux['sigma'][sel]).detach().to(t.float32)
        side['isg'][l] = t.cat((side['isg'][l - 1], Is, Is))
        return out[:2] + (p, lam, star) + out[5:]
    cd.hooks['state_out'] = state_out
    return cd


def install_bsrc(cd, torch, kappa):
    import s6bcard_joint2t as JT                   # the card's dials: bscale multiplies b after the source is added
    prev = cd.hooks.get('b_post')

    def b_post(l, b, ctx):
        if prev is not None: b = prev(l, b, ctx)
        return b + JT.DIALS_T['bscale'] * (kappa - 1) * ctx['radial'] * ctx['d'] * ctx['vv'] / 6
    cd.hooks['b_post'] = b_post
    return cd
