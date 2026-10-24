#!/usr/bin/env python3
"""TAG fable-covhunter (CH-26): kink-row CARRIER scales in the covariance output, on top of S2 + the v3 knobs (lead's request: per-band
(s_cp, s_sp, s_b) fitted by the quadratic model). Carriers of the card's K21 on kink rows (|mu/sd|<1.5, off-diagonal): cp (atoms and
old-tier read), sp (star push), bt (b terms v_a b_c + 2 C_ac b_a). Knob '<carrier><band>' adds th * carrier to the K21 of the covariance
call at the band's layers (E = L2-7, L = L8-14), on top of the v3 covariance knobs (same second call). Fit: one finite-difference run per
knob on TRAIN nets (quadratic model; gn_select.py); eval: S2+v3 vs S2+v3+carriers.
usage: gn4_carriers.py fit <ranks> <knobs e.g. cpE,cpL,spE,spL,btE,btL> <threads> <out.npz>
       gn4_carriers.py eval <ranks|valid40> <theta json> <threads> <out.jsonl>"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
import covstep as cs
import covstep_s2 as s2
n = 1024; eye = torch.eye(n, dtype=torch.bool)
V3 = {'iso': -0.094, 'k21n': -0.028, 'k31': -0.212, 'edge': 0.164, 'rc': -0.080}
STEP = {f'{c}{b}': st for c, st in (('cp', 0.01), ('sp', 0.2), ('bt', 0.05)) for b in 'EL'}
BANDS = dict(E=(2, 7), L=(8, 14))


def configure(card, car):
    s2.make_s2(card)
    def cov(l, C, ctx):
        if l < 2 or l > 14: return C
        mu, Cpre, k21, k4, k22, pair = ctx['mu'], ctx['Cpre'], ctx['k21'], ctx['k4'], ctx['k22'], ctx['pair']
        vv = torch.diagonal(Cpre); s = vv.sqrt(); al = mu / s
        kink = (al.abs() < 1.5).double(); km = (kink[:, None] * torch.ones(1, n, dtype=torch.float64)).masked_fill(eye, 0.)
        nm = ((1 - kink)[:, None] * torch.ones(1, n, dtype=torch.float64)).masked_fill(eye, 0.)
        k3 = torch.diagonal(k21); rho = (Cpre / torch.outer(s, s)).masked_fill(eye, 0.); bpre = ctx['bpre']
        comps = dict(cp=ctx['cp'], sp=ctx['sp'], bt=vv[:, None] * bpre[None, :] + 2 * Cpre * bpre[:, None])
        k2 = k21 + V3['iso'] * km * (k3[:, None] * Cpre / vv[:, None]) + V3['k21n'] * nm * k21
        for key, t in car.items():
            lo, hi = BANDS[key[-1]]
            if t and lo <= l <= hi: k2 = k2 + t * km * comps[key[:-1]]
        r31 = V3['k31'] * ((k4 / vv ** 2) * (0.85 + 0.5 * mu * k3 / (vv * vv)))[:, None] * rho
        return pair(mu, Cpre, k2, k4, k22, 'covariance', remainder31=r31)[1]
    card.hooks['cov_pm'] = cov
    card.hooks['edge'] = lambda l, v, ctx: v * (1 + V3['edge'])
    card.knobs['recenter'] = 1 + V3['rc']


def final_err(w, truth, car, pl):
    p, a, c, e = pl
    card = cs.HookCard(torch, w, p, a, c, e); configure(card, car)
    with torch.no_grad(): outs, _ = card.run_states()
    return outs[15].clamp_min(0).to(torch.float32).numpy().astype(np.float64) - truth[15]


if __name__ == '__main__':
    mode = sys.argv[1]; pl = cs.model.torch_payload(torch, np, False)
    if mode == 'fit':
        ranks = [int(x) for x in sys.argv[2].split(',')]; knobs = sys.argv[3].split(','); torch.set_num_threads(int(sys.argv[4])); out = sys.argv[5]
        E0, U = {}, {}
        for rk in ranks:
            w = cs.weights(rk); truth = cs.truthpost(rk); t0 = time.time()
            E0[rk] = final_err(w, truth, {}, pl); cols = []
            for k in knobs:
                ek = final_err(w, truth, {k: STEP[k]}, pl); cols.append((ek - E0[rk]) / STEP[k])
                print(json.dumps(dict(rank=rk, knob=k, step=STEP[k], ratio_fd=float((ek ** 2).mean() / (E0[rk] ** 2).mean()))), flush=True)
            U[rk] = np.stack(cols, 1)
            print(f'# rank {rk} done {time.time() - t0:.0f}s', flush=True)
            np.savez(out, knobs=np.array(knobs), ranks=np.array(list(E0)), **{f'e0_{r}': E0[r] for r in E0}, **{f'U_{r}': U[r] for r in U})
    else:
        arg, th, thr, out = sys.argv[2], json.loads(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5]); torch.set_num_threads(thr)
        ranks = json.loads((cs.P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())['valid'] if arg == 'valid40' else [int(x) for x in arg.split(',')]
        for rk in ranks:
            w = cs.weights(rk); truth = cs.truthpost(rk)
            e0 = final_err(w, truth, {}, pl); e1 = final_err(w, truth, th, pl)
            rec = dict(rank=rk, card='S2+v3', theta=th, base=float((e0 ** 2).mean()), knobs=float((e1 ** 2).mean()))
            rec['ratio'] = rec['knobs'] / rec['base']
            print(json.dumps(rec), flush=True)
            with out.open('a') as f: f.write(json.dumps(rec) + '\n')
