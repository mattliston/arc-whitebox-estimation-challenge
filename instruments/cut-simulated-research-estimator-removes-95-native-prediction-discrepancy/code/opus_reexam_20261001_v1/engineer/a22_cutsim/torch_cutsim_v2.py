"""TAG opus_engineer a22 (2026-10-02), v2: the native file's ROW CUTS simulated on the torch run M card, to price each cut without a
native session (v2 = v1 plus the native tail part of the fourth-order scalar's source for the old-tier rows, repair scales, and a
per-layer comparison of each repair with the exact value it stands in for). The torch card pushes every row; the native file skips rows to save bill and repairs the per-unit third cumulant
of the skipped rows with a trace formula. Here the same rule is applied to the torch card's pushed bank:
  dead   (layers 1-14)  rows with alpha < -dead are not pushed at all (kept count rounded up to 64); no repair       native 3.5
  ott    (layers 7-14)  OLD-tier columns: rows with alpha > ott are not reconstructed; repair = 3 rho U_i . F (lam o |F_r|^2)   native 2.5
  neg    (layers 7-14)  OLD-tier columns: dead rows (not alive) with alpha < -neg, same repair                        native 2.5
  ynt    (layers 7-14)  YOUNG columns: dead rows with alpha < -ynt are not pushed; repair = 3 |w_i|^2 / (n + 2) w_i . tau,
                        tau = sum_r c_r |p_r|^2 p_r (only when at least 64 rows go)                                   native 2.5
A threshold of 99 switches a cut off (the default: the torch card as it is). The repair is added to the diagonal of the bank's
plane (the third cumulant the pair map, the births and the learned parts see). The fourth-order scalar's source (the step's
omitted-row sum) is corrected as the native omitted_radial_source does for the old-tier rows it did not reconstruct: column sums
exactly through the factors, row energies by |F_r|^2 x mean(U_i^2), and the repaired diagonal in the subtracted term (rtail=0
leaves this out: v1's behaviour).
  orep / yrep   scale of the old-tier / young repair (1 = the native formulas, 0 = no repair)
The final layer's k3 row cut (|alpha| <= 3.5) is already part of the torch card (gn7_s6b.py, k3cut), so it is not simulated.
Differences from the native file that remain: its kept-row counts are rounded on the native alpha (float32 kernels), at layers
13-14 its old-tier repair is taken on two blocks, and the K4L level uses every row here (the native file leaves unpushed rows out).
arms: name:key=value,...   keys dead ott neg ynt and any dial family of the run M card (f4a1 f4a2 k4la1 k4la2 k22 bsrc rc ...)
usage: TRAINER_THREADS=3 OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3 TRAINER_PL=1 TRAINER_CARD=M \\
       python -B torch_cutsim.py <rank> <out.npz (new)> arm [arm ...]
Output: <out.npz> pred_<arm> (16 x 1024 float64); one JSON line per arm (per-layer MSE, rows cut per layer)."""
import json
import math
import sys
import time
from pathlib import Path

O = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(O/'trainer'))
import numpy as np
import otr_engine as E
from otr_common import data_paths

torch = t = E.torch
CUTS = ('dead', 'ott', 'neg', 'ynt')
KNOBS = dict(orep=1.0, yrep=1.0, rtail=1.0)
rank, out, arms = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
assert not out.exists() and E.M is not None and E.PL is not None
weights_path, truth_path = data_paths(rank)
weights = np.load(weights_path, allow_pickle=False)
truth = np.load(truth_path, allow_pickle=False)['mc_post'].astype(np.float64)
N = 1024


def top(score, step):
    """rows with score >= 0, their count rounded up to a multiple of `step`, in descending order of score (stable)"""
    count = min(int(score.numel()), ((int((score >= 0).sum()) + step - 1) // step) * step)
    order = t.argsort(-score, stable=True)
    return order[:count], order[count:]


def quality(exact, rep):
    """a repair against the exact value on the same rows: rows, rms of the exact value, rms left after the repair, correlation,
    and the scale of the repair that would fit best"""
    ex, rp = exact.to(t.float64), rep.to(t.float64)
    return dict(rows=int(ex.numel()), rms_exact=float((ex * ex).mean().sqrt()), rms_left=float(((ex - rp) ** 2).mean().sqrt()),
                corr=float((ex * rp).sum() / ((ex * ex).sum() * (rp * rp).sum()).clamp_min(1e-300).sqrt()),
                best_scale=float((ex * rp).sum() / (rp * rp).sum().clamp_min(1e-300)))


def install_cuts(cd, o, log):
    st = {}
    push0 = cd.shared['push']
    step0 = cd.step

    def step(layer, state):   # the incoming mean and covariance, for this layer's alpha (the run M card owns the 'state_in' hook)
        st['m'], st['C'] = state[0], state[1]
        return step0(layer, state)
    cd.step = step

    def push(tt, layer, w, p, lam, previous, Q, F, core, stops, rank=384):
        r = push0(tt, layer, w, p, lam, previous, Q, F, core, stops, rank=rank)
        st['repair'] = None
        if layer == 0 or layer == 15 or not lam.numel():
            return r
        pp, U, Fn, coren, oc = r
        mu = cd.ws[layer].T @ st['m']
        vv = (w.to(t.float64) * (st['C'].to(t.float32) @ w).to(t.float64)).sum(0)      # diag(w' C w), the step's own arithmetic
        a = mu / vv.clamp_min(1e-10).sqrt()
        cut = dict(layer=layer)
        gate = t.erfc(-a / math.sqrt(2)) / 2
        alive = t.zeros(N, dtype=t.bool)
        alive[top(gate - .01, 32)[0]] = True                                       # the card's select(gate)
        pp = pp.clone()
        keep = t.arange(N)
        if o['dead'] < 90:
            keep, gone = top(a + o['dead'], 64)
            pp[gone] = 0; cut['dead'] = int(gone.numel())
        ak, alivek, m = a[keep], alive[keep], int(keep.numel())
        repair = t.zeros(N, dtype=t.float64)
        rtail = dict(tail=None)
        if 7 <= layer <= 14 and oc > 0 and (o['ott'] < 90 or o['neg'] < 90):
            score = o['ott'] - ak
            negd = (ak < -o['neg']) & ~alivek
            score = t.where(negd, ak + o['neg'], score)
            kept, gone = top(score, 64)
            if gone.numel():
                tail = keep[gone]
                Uk = U[keep]
                norm2 = (Fn * Fn).sum(0)
                rho = (Uk * Uk).sum() / (Uk.shape[0] * Uk.shape[1])
                vector = Fn @ (lam[:oc] * norm2)
                rep = ((U[tail] @ vector) * (3 * rho)).to(t.float64)
                exact = ((pp[tail, :oc].to(t.float64) ** 3) * lam[:oc].to(t.float64)).sum(1)
                for tag, pick in (('old_pos', a[tail] > 0), ('old_neg', a[tail] < 0)):
                    if int(pick.sum()) > 1:
                        cut[tag] = quality(exact[pick], rep[pick])
                repair[tail] += o['orep'] * rep
                pp[tail, :oc] = 0
                rtail.update(tail=tail, Ut=U[tail], F=Fn, norm2=norm2, oc=oc)
        if 7 <= layer <= 14 and o['ynt'] < 90 and int(pp.shape[1]) > oc:
            neg = (ak < -o['ynt']) & ~alivek
            kept_count = min(m, ((int((~neg).sum()) + 63) // 64) * 64)
            if m - kept_count >= 64:
                score = t.where(neg, ak + o['ynt'], t.zeros_like(ak))
                order = t.argsort(-score, stable=True)
                tail = keep[order[kept_count:]]
                assert not bool(alive[tail].any())
                Y = p[:, oc:]
                tau = Y @ (lam[oc:] * (Y * Y).sum(0))
                wt = w[:, tail]
                rep = ((3 * (wt * wt).sum(0) / (N + 2)) * (wt.T @ tau)).to(t.float64)
                exact = ((pp[tail, oc:].to(t.float64) ** 3) * lam[oc:].to(t.float64)).sum(1)
                cut['young'] = quality(exact, rep)
                repair[tail] += o['yrep'] * rep
                pp[tail, oc:] = 0
        if len(cut) > 1:
            log.append(cut)
        st['repair'] = repair if bool((repair != 0).any()) else None
        if o['rtail'] and (rtail['tail'] is not None or st['repair'] is not None):
            rtail.update(pp=pp, lam=lam, repair=repair, oc=oc)
            st['rtail'] = rtail
        else:
            st.pop('rtail', None)
        return pp, U, Fn, coren, oc
    cd.shared['push'] = push

    pair0 = cd.hooks['pair_in']

    def pair_in(l, pin, ctx):
        pin = pair0(l, pin, ctx)
        rep = st.get('repair')
        if rep is not None:
            assert pin[2] is ctx['k21']
            ctx['cp'].diagonal().add_(rep)         # the step's own plane: the inherited diagonal of the births sees the repair
            ctx['k21'].diagonal().add_(rep)        # the third cumulant of the pair map, the learned parts and the birth update
        return pin
    cd.hooks['pair_in'] = pair_in

    cov0 = cd.hooks['cov_pm']

    def cov_pm(l, C, ctx):
        C = cov0(l, C, ctx)
        st['den'] = t.trace(C) ** 2 + 2 * (C * C).sum()      # the step divides the omitted-row sum by this, with this C
        return C
    cd.hooks['cov_pm'] = cov_pm
    out0 = cd.hooks['state_out']

    def state_out(l, out, ctx):
        out = out0(l, out, ctx)
        rt = st.pop('rtail', None)
        if rt is None or l > 14:
            return out
        aux, host, f64 = ctx['aux'], ctx['host'], t.float64
        aa = (aux['gate'] - host * aux['pdf'] / aux['sigma']) * (ctx['alive'] * (1 - ctx['response_mask']))
        beta = 2 * (aux['gaussian_mean'] - host * aux['gate']) * ctx['alive']
        p64, lam64 = rt['pp'].to(f64), rt['lam'].to(f64)
        energy, column = (p64 * p64).T @ aa, p64.T @ beta
        cpdiag = ((p64 * p64 * p64) * lam64).sum(1)
        extra0 = 6 * ((lam64 * energy * column).sum() - (cpdiag * aa * beta).sum())     # what the step itself added
        if rt['tail'] is not None:
            tail, Ut, oc = rt['tail'], rt['Ut'], rt['oc']
            rowvar = ((Ut * Ut).sum(1) / Ut.shape[1]).to(f64)
            energy = energy.clone(); column = column.clone()
            energy[:oc] += rt['norm2'].to(f64) * (aa[tail] * rowvar).sum()
            column[:oc] += ((Ut.T @ beta[tail].to(t.float32)) @ rt['F']).to(f64)
        extra1 = 6 * ((lam64 * energy * column).sum() - ((cpdiag + rt['repair']) * aa * beta).sum())
        return out[:7] + (out[7] + (extra1 - extra0) / st['den'],) + out[8:]
    cd.hooks['state_out'] = state_out


saved = {}
for spec in arms:
    name, _unused, opts = spec.partition(':')
    o = {key: 99.0 for key in CUTS}
    o.update(KNOBS)
    shifts = {}
    for item in filter(None, opts.split(',')):
        key, value = item.split('=')
        if key in CUTS or key in KNOBS:
            o[key] = float(value)
        else:
            shifts[key] = [float(value)] * 16
    E.PL.set_values(shifts)
    E.LC.JT.DIALS_T = E.DL
    log = []
    t0 = time.monotonic()
    with torch.no_grad():
        cd = E.LC.card(torch, weights, E.p, E.T['a'], E.T['c'], E.e, fine_ckpt=False)
        cd.hooks['k4_15'] = lambda layer, value, ctx: value
        E.install_star(cd)
        E.PL.install(E, cd)
        if any(o[key] < 90 for key in CUTS):
            install_cuts(cd, o, log)
        E.M.install(cd)
        prediction = cd.run(False).detach().to(torch.float64).numpy().copy()
    del cd
    E.collect()
    assert prediction.shape == (16, 1024) and np.isfinite(prediction).all()
    saved['pred_' + name] = prediction
    mse = ((prediction - truth) ** 2).mean(1)
    print(json.dumps(dict(rank=rank, arm=name, cuts={k: v for k, v in o.items() if k in CUTS and v < 90}, knobs={k: o[k] for k in KNOBS if o[k] != KNOBS[k]}, shifts={k: v[1] for k, v in shifts.items()},
                          final_x1e9=round(float(mse[15]) * 1e9, 4), layers_x1e9=[round(float(x) * 1e9, 3) for x in mse],
                          rows_cut=log, seconds=round(time.monotonic() - t0, 1))), flush=True)
np.savez(out, **saved)
