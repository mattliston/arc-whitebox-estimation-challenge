#!/usr/bin/env python3
"""TAG fable-covhunter (CH-82 v3 = v2 + uz=r_u: SPLIT-RANK TIER at every join: after the join the common part f_u of every old birth
pair is zeroed beyond the first r_u frame directions (f_v kept on all r), F and the u Gram block updated to match; every consumer at
every later layer sees it (a native structured product U[:, :r_u] F_u +- U F_v / 2 would save (r - r_u) / (2r) of each old-tier
expansion and read). CH-82 v2 = v1 + su15o/sv15o: the u (v) part keeps its r directions ranked by that part's own energy AT L15,
e_k = |U[:, k]|^2 * sum_p w_p F_part[k, p]^2 (w_u = 2c^2, w_v = c^2/2), instead of the first r (L13 join order); O(n r + r n_pairs).
CH-82 r320_l15split = r320_vsel + SPLIT-RANK L15 k3 read (lead's L15 task 12:0x): su15=r_u, sv15=r_v compute
the old tier's L15 bank columns from the birth-pair split p+- = U(f_u +- f_v/2) with the common part f_u on the first r_u frame
directions and the difference part f_v on the first r_v (energy order); the tier itself is unchanged (L15 is the last layer), only
the final k3 read sees it. Native cost of the old part ~ (r_u + r_v) / (2 x rank). L15 plain truncation = cut13=15,cr=r.
CH-81 r320_vsel = r320_comp8 + VALUE-WEIGHTED ROW SELECTION at a join (lead's task 11:4x): vs<L>=code ranks the
Nystrom rows by d_i * v_i^2 instead of the Gram diagonal d_i (energy): code 1 v = gate of the row unit (layer L-1), 2 v^2 = |W_L[i,:]|^2
(how strongly the unit feeds layer L), 3 v^2 = sum_j W_L[i,j]^2 |z_j| phi(z_j) / var_j (loading on layer-L units whose mean is most
sensitive to their k3; z, var from the state entering L), 4 v^2 = sum_j W_L[i,j]^2 min(g_j, 1-g_j) (loading on kink units). wf<L>=1 adds
WEIGHTED Nystrom coordinates: basis D_v^-1 orth(D_v A[:, rows]), F = orth^T D_v J (the frame minimizes the v-weighted atom error; the
join diagnostics are then in the weighted metric). Same rank, same flop count in a native build (O(n^2) weights; code 3/4 reuse Cpre).
CH-80 v8 = v7 + arm name l10 = the L10-only schedule {7: 384, 10: 320, 13: 384}; v7 = v6 with kf CHAINED onto the card's own k3_15 hook (s6bcard.l15_rowcut, which rebuilds k3 from ctx bank/star/bterm): kf adds e x the tier's per-unit k3 to ctx['bank'] (v5/v6 kf REPLACED the rowcut hook: invalid); v6 = v5 + arm name l13 = the L13-only schedule {7: 384, 10: 384, 13: 320} (lead 10:13); v5 = v4 + CONSUMER-specific scales of the old tier (c): kd<L0>=e adds e x the tier's per-unit k3 to the pair map's k21 diagonal (layers L0..14; the mean path), kp<L0>=e scales the tier's plane read by 1+e (layers >= L0), kf=e adds e x the tier's per-unit k3 to the final k3 (L15); v4 = v3 + dg<L0>=g: the learned mean correction delta x g at layers >= L0 (a no-refit stopgap probe); v3 = v2 + cr=<rank>: the rank the cut<L>=l timing probe truncates to, default 320; v2 = v1 + dial overrides D_<key>=v per arm): cheap COMPENSATION for the content a rank-320 Nystrom join drops, on the torch card at the fixed schedule
384/320/320 (lead's task 09-25, RESULT sec 35). Card = the analyst's r320_features.py setup (AN-69b): s6bcard_loop4j (S23 structure) +
loopR3 params + loopR3 dials, float64 CPU, learned ON (or OFF with 'off': head/head_b and edge output/bias zeroed, for the shadow test).
Push = the analyst's energy-ordered Nystrom make_push (adaptive_join.py; nested truncations: rank r = the first r QR columns), with the old
tier's Gram carried SPLIT by birth pair: every cohort's columns are (positive atoms, negative atoms) of one birth set with weights (+c, -c),
p+- = u +- v/2, so c^2 (p+ p+^T + p- p-^T) = 2c^2 u u^T + (c^2/2) v v^T; core[0] = the u part, core[1] = the v part (the card's core = sum).
Knobs (per arm, 'sched:k=v,...'):
  a<L>=x    (c) the join at L builds its Nystrom Gram as u part + x * v part (row choice d and sketch Y); x = 1 is the card. Same flop count.
  fs<L>=s   (a) after the join at L: F <- s F (a scalar on the kept old-tier block; the cubic consumers see s^3)
  rw<L>=g   (b) after the join at L: frame rows scaled by (d_i / dhat_i)^(g/2), d = the card's Gram diagonal, dhat = diag(Q core Q^T) kept
  cut<L>=l  timing probe: the join at L keeps 384 and truncates to 320 at layer l (consumers L..l-1 see 384)
Arms: fixed384 | sched (= 384/320/320) [ ':knobs' ]. Per net and arm: final MSE vs truth688, per join the u/v Gram energy captured at 320/384.
usage: r320_comp.py <valid40|heldout40|train<k>|r1,r2,..> <threads> <tag> '<arm>;<arm>;...' [off]"""
import os, sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
for d in ('fable_engineer_v1', 'fable_covhunter_v1', 'fable_physicist_v1/refit6b', 'fable_analyst_v1'):
    sys.path.insert(0, str(P2 / d))
import math
import numpy as np, torch
import s6bcard_loop4j as LC
import s2card
JT = LC.JT
arg, THREADS, TAG, ARMS = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4].split(';')
OFF = len(sys.argv) > 5 and sys.argv[5] == 'off'
torch.set_num_threads(THREADS); torch.set_grad_enabled(False)
R3 = P2 / 'fable_physicist_v1/refit6b/s6b_loopR3_joint_s23'
_dv = json.load(open(R3 / 'dials_epoch0.json')); _dv = _dv.get('dials', _dv)
JT.DIALS_T = JT.make_dials(torch, {k: float(_dv[k]) for k in JT.KEYS}, trainable=())
p, a, c, e = LC.payload(torch); sv = torch.load(R3 / 'params_epoch0.pt')
with torch.no_grad():
    for k in p: p[k].copy_(sv['p'][k])
    a.copy_(sv['a']); c.copy_(sv['c'])
    for k in e: e[k].copy_(sv['e'][k])
    if OFF:
        for k in ('head', 'head_b'): p[k].zero_()
        for k in ('output', 'bias'): e[k].zero_()
inputs = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())
if arg == 'valid40': RANKS = inputs['valid'][:40]
elif arg == 'heldout40': RANKS = sorted(json.load(open(P2 / 'fable_analyst_v1/split_train640.json'))['new_ranks'])[::12][:40]
elif arg.startswith('train'): RANKS = [r for r in inputs['train'] if r not in (35, 100, 200)][::6][:int(arg[5:])]
else: RANKS = [int(x) for x in arg.split(',')]
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
f64 = torch.float64


def parse(arm):
    name, _, kv = arm.partition(':')
    ranks = {'fixed384': {7: 384, 10: 384, 13: 384}, 'sched': {7: 384, 10: 320, 13: 320}, 'l13': {7: 384, 10: 384, 13: 320}, 'l10': {7: 384, 10: 320, 13: 384}}[name]
    knobs = {k: float(v) for k, v in (x.split('=') for x in kv.split(',') if x)}
    return ranks, knobs


def pair_index(t, cohort_stops, cohorts, start):
    pos, neg = [], []
    for c_ in cohorts:
        lo = cohort_stops[c_ - 1] if c_ else 0; hi = cohort_stops[c_]; h = (hi - lo) // 2
        assert hi - lo == 2 * h
        pos.append(t.arange(lo, lo + h) - start); neg.append(t.arange(lo + h, hi) - start)
    return t.cat(pos), t.cat(neg)


STASH = {}
GATES, SIN = {}, {}


def value_w2(t, code, layer, w32, prev):
    W = w32[prev, :].to(f64)
    if code == 1: return GATES[layer - 1].to(f64)[prev] ** 2
    if code == 2: return (W * W).sum(1)
    m, C = SIN[layer]; Wf = w32.to(f64)
    mu = Wf.T @ m.to(f64); var = ((C.to(f64) @ Wf) * Wf).sum(0).clamp_min(1e-30); z = mu / var.sqrt()
    if code == 3: kap = z.abs() * t.exp(-z * z / 2) / math.sqrt(2 * math.pi) / var
    else: g = t.erfc(-z / math.sqrt(2)) / 2; kap = t.minimum(g, 1 - g)
    return (W * W) @ kap


def make_push(qr, joins, ranks, knobs, log):
    cut_at = {int(v): int(k[3:]) for k, v in knobs.items() if k.startswith('cut')}
    def push(t, layer, w32, p_full, lam, previous_indices, Q, F, core, cohort_stops, rank=384):
        wp = w32[previous_indices, :].T
        before = [c_ for L_, cs in joins.items() if L_ < layer for c_ in cs]
        old_columns = cohort_stops[max(before)] if before else 0
        if layer in joins:
            x = knobs.get(f'a{layer}', 1.0)
            start, stop = old_columns, cohort_stops[max(joins[layer])]
            J = p_full[previous_indices, start:stop]
            pos, neg = pair_index(t, cohort_stops, joins[layer], start)
            cl = lam[start:stop]
            assert bool(((cl[pos] + cl[neg]).abs() <= 1e-6 * cl[pos].abs()).all()), 'birth pairs must carry (+c, -c)'
            Ju, Jv = (J[:, pos] + J[:, neg]) / 2, J[:, pos] - J[:, neg]
            c2 = cl[pos] ** 2; wu, wv = 2 * c2, c2 / 2
            Qp = Q[previous_indices, :] if old_columns else None
            du, dv = (Ju * Ju) @ wu, (Jv * Jv) @ wv
            if old_columns:
                du = du + ((Qp @ core[0]) * Qp).sum(1); dv = dv + ((Qp @ core[1]) * Qp).sum(1)
            d = du + x * dv
            vs = knobs.get(f'vs{layer}'); Dv = None
            score = d.to(f64) if vs is None else d.to(f64) * value_w2(t, int(vs), layer, w32, previous_indices)
            if vs is not None and knobs.get(f'wf{layer}'):
                v_ = value_w2(t, int(vs), layer, w32, previous_indices).sqrt(); Dv = t.maximum(v_, 1e-3 * v_.max()).to(J.dtype)
            rows = t.argsort(-score, stable=True)[:384]                  # score order (nested truncations; energy = the card)
            Y = Ju @ (wu[:, None] * Ju[rows, :].T) + x * (Jv @ (wv[:, None] * Jv[rows, :].T))
            if old_columns: Y = Y + Qp @ ((core[0] + x * core[1]) @ Qp[rows, :].T)
            Qn = qr(t, Y if Dv is None else Dv[:, None] * Y)
            Fj = Qn.T @ (J if Dv is None else Dv[:, None] * J)
            Fu, Fv = (Fj[:, pos] + Fj[:, neg]) / 2, Fj[:, pos] - Fj[:, neg]
            ju, jv = Fu @ (wu[:, None] * Fu.T), Fv @ (wv[:, None] * Fv.T)
            if old_columns:
                overlap = Qn.T @ (Qp if Dv is None else Dv[:, None] * Qp)
                ou, ov = overlap @ (core[0] @ overlap.T), overlap @ (core[1] @ overlap.T)
            else:
                ou = ov = t.zeros_like(ju)
            eu, ev = t.cumsum(t.diagonal(ju + ou).to(f64), 0), t.cumsum(t.diagonal(jv + ov).to(f64), 0)
            tu, tv = float(du.to(f64).sum()), float(dv.to(f64).sum())
            log.append(dict(layer=layer, x=x, u320=float(eu[319]) / tu, u384=float(eu[383]) / tu, v320=float(ev[319]) / tv,
                            v384=float(ev[383]) / tv, vshare=tv / (tu + tv)))
            r = 384 if f'cut{layer}' in knobs else min(ranks[layer], Qn.shape[1])
            Qn, Fj = Qn[:, :r], Fj[:r]
            if Dv is not None: Qn = Qn / Dv[:, None]
            core_new = t.stack(((ju + ou)[:r, :r], (jv + ov)[:r, :r]))
            if old_columns:
                F = t.cat((overlap[:r] @ F, Fj), dim=1)
            else:
                F = Fj
            core = core_new
            s = knobs.get(f'fs{layer}')
            if s is not None: F = F * s; core = core * (s * s)
            g = knobs.get(f'rw{layer}')
            if g is not None:
                dhat = ((Qn @ (core[0] + core[1])) * Qn).sum(1)
                Qn = Qn * ((d.to(f64) - (x - 1) * dv.to(f64)) / dhat.to(f64).clamp_min(1e-30)).pow(g / 2).to(Qn.dtype)[:, None]
            if 'uz' in knobs:
                ru_ = int(knobs['uz'])
                pos_a, neg_a = pair_index(t, cohort_stops, [c_ for L_ in sorted(joins) if L_ <= layer for c_ in joins[L_]], 0)
                Fu_, Fv_ = (F[:, pos_a] + F[:, neg_a]) / 2, F[:, pos_a] - F[:, neg_a]
                Fu_[ru_:] = 0
                F = F.clone(); F[:, pos_a] = Fu_ + Fv_ / 2; F[:, neg_a] = Fu_ - Fv_ / 2
                core = core.clone(); core[0, ru_:, :] = 0; core[0, :, ru_:] = 0
            basis, old_columns = Qn, stop
        elif old_columns:
            basis = Q[previous_indices, :]
            if layer in cut_at:
                R_ = int(knobs.get("cr", 320)); basis, F, core = basis[:, :R_], F[:R_], core[:, :R_, :R_]
        else:
            return wp @ p_full[previous_indices, :], None, F, core, 0
        U = wp @ basis
        if layer == 15 and any(k_ in knobs for k_ in ('su15', 'sv15', 'su15o', 'sv15o')):
            cohorts = [c_ for L_ in sorted(joins) for c_ in joins[L_]]
            pos, neg = pair_index(t, cohort_stops, cohorts, 0)
            ru, rv = int(knobs.get('su15', U.shape[1])), int(knobs.get('sv15', U.shape[1]))
            Fu, Fv = (F[:, pos] + F[:, neg]) / 2, F[:, pos] - F[:, neg]
            if 'su15o' in knobs or 'sv15o' in knobs:
                c2 = lam[:old_columns][pos].to(f64) ** 2; un = (U.to(f64) ** 2).sum(0)
                ku = t.argsort(-(un * ((Fu.to(f64) ** 2) @ (2 * c2))), stable=True)[:int(knobs.get('su15o', U.shape[1]))]
                kv = t.argsort(-(un * ((Fv.to(f64) ** 2) @ (c2 / 2))), stable=True)[:int(knobs.get('sv15o', U.shape[1]))]
                pu, pv = U[:, ku] @ Fu[ku], U[:, kv] @ Fv[kv]
            else:
                pu, pv = U[:, :ru] @ Fu[:ru], U[:, :rv] @ Fv[:rv]
            old_pp = t.empty((U.shape[0], old_columns), dtype=U.dtype)
            old_pp[:, pos] = pu + pv / 2; old_pp[:, neg] = pu - pv / 2
        else:
            old_pp = U @ F
        STASH[layer] = ((old_pp.to(f64) ** 3) * lam[:old_columns].to(f64)).sum(1)
        return t.cat((old_pp, wp @ p_full[previous_indices, old_columns:]), dim=1), U, F, core, old_columns
    return push


out = HERE / f'r320_comp_{TAG}.jsonl'; preds = {}
with out.open('a') as fo:
    for r_ in RANKS:
        w = np.load(STORE / Path(man[r_]['path']).parent / 'weights.npy')
        tp = np.load(P2 / 'fable_refit_v1/truth688' / f'truthpost_rank{r_:04d}.npz')['mc_post'][15].astype(np.float64)
        for ai, arm in enumerate(ARMS):
            t0 = time.time(); log = []
            ranks, knobs = parse(arm); dv = {k: float(_dv[k]) for k in JT.KEYS}; dv.update({k[2:]: v for k, v in knobs.items() if k.startswith("D_")})
            JT.DIALS_T = JT.make_dials(torch, dv, trainable=())
            cd = LC.card(torch, w, p, a, c, e)
            cd.shared['push'] = make_push(cd.shared['_qr'], s2card.EVERY3, ranks, knobs, log)
            STASH.clear(); GATES.clear(); SIN.clear()
            _so0 = cd.hooks['state_out']
            def _so(l, st, ctx, _so0=_so0):
                if 'aux' in ctx: GATES[l] = ctx['aux']['gate'].detach()
                return _so0(l, st, ctx)
            cd.hooks['state_out'] = _so
            assert 'state_in' not in cd.hooks
            cd.hooks['state_in'] = lambda l, st, ctx: (SIN.__setitem__(l, (st[0], st[1])), st)[1]
            kd = [(int(k_[2:]), g_) for k_, g_ in knobs.items() if k_.startswith("kd")]
            kp = [(int(k_[2:]), g_) for k_, g_ in knobs.items() if k_.startswith("kp")]
            if kd:
                _pin0 = cd.hooks['pair_in']
                def _pin(l, pin, ctx, _pin0=_pin0, L0=kd[0][0], e_=kd[0][1]):
                    pin = _pin0(l, pin, ctx)
                    if l >= L0 and l in STASH:
                        pin = (pin[0], pin[1], torch.diagonal_scatter(pin[2], torch.diagonal(pin[2]) + e_ * STASH[l]), pin[3], pin[4], pin[5])
                    return pin
                cd.hooks['pair_in'] = _pin
            if kp:
                _read0 = cd.shared['read']
                def _read(t_, pp_, lam_, ri, li, U_, F_, oc, _read0=_read0, L0=kp[0][0], e_=kp[0][1]):
                    out_ = _read0(t_, pp_, lam_, ri, li, U_, F_, oc)
                    if oc and cd.call_layer >= L0:
                        left = pp_[ri, :oc]
                        out_ = out_ + e_ * (((left * left) @ (F_ * lam_[:oc][None, :]).T) @ U_[li, :].T)
                    return out_
                cd.shared['read'] = _read
            if 'kf' in knobs:
                _k30 = cd.hooks['k3_15']
                cd.hooks['k3_15'] = lambda l, k3, ctx, e_=knobs['kf'], _k30=_k30: _k30(l, k3 + e_ * STASH[15], dict(ctx, bank=ctx['bank'] + e_ * STASH[15]))
            for k_, g_ in knobs.items():
                if k_.startswith("dg"): cd.hooks["delta"] = (lambda L0, g: lambda l, d, ctx: d * g if l >= L0 else d)(int(k_[2:]), g_)
            pred = cd.run(False)[15].to(f64).numpy(); preds[f'{r_}|{arm}'] = pred
            rec = dict(net=r_, arm=arm, off=OFF, mse=float(((pred - tp) ** 2).mean()), wall=round(time.time() - t0, 1), joins=log)
            fo.write(json.dumps(rec) + '\n'); fo.flush()
            print(json.dumps(dict(net=r_, arm=arm, mse=rec['mse'], wall=rec['wall'],
                                  joins=[{k: (round(v, 6) if isinstance(v, float) else v) for k, v in j.items()} for j in log])), flush=True)
        np.savez(HERE / f'r320_comp_{TAG}_preds.npz', **preds)
