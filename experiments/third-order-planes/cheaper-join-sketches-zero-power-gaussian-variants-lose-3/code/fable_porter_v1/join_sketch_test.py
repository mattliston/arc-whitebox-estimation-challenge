#!/usr/bin/env python3
"""TAG fable-porter (PT-40, lead 21:3x GO "no-refit VALID16 test of the two cheaper join sketches", torch only): the S21-physics torch card
(fable_covhunter_v1/valid40_dials.py construction: s6bcard_joint2 = shared semantics, DIALS = round-1 dials_r1_ep2.json, ep2 literals
s6b_j05_ep2/params_epoch0.pt, the engineer's lrstar r32 hook; no gn_linear_hook, no pair series = covhunter's r1 rows; learned ON, no refit),
VALID16 (inputs.json valid[:16]), 3 torch threads. The old-tier JOIN (s2card.make_push, = native _shared_old384_push join block) in 4 arms:
  A      current: Y = J (w o (J^T Omega)) [+ Qp core Qp^T Omega], Omega = W[live, :384]      (control; must reproduce covhunter's fit rows)
  B_lit  the lead's formula: Y = J (w o G) [+ Qp core Qp^T Omega unchanged], G fixed seeded Gaussian (width x 384)
  B      scale-consistent q=0 sketch of A = [J w^1/2 | Qp core^1/2]: Y = J (w^1/2 o G1) [+ Qp core^1/2 G2]  (B_lit mixes a term linear in J
         with the Gram-form old-tier term, quadratic in J, so at the L10/L13 joins one term can swamp the other)
  C      Nystrom row selection: Omega = E_rows, the 384 rows of largest norm of A (largest Gram diagonal d_i = sum_c w_c J_ic^2
         [+ (Qp core Qp^T)_ii]); Y = J (w o J[rows]^T) [+ Qp core Qp[rows]^T] (a gather replaces J^T Omega)
QR, Fj = Qn^T J, the weighted core and the old-tier overlap/factor update are unchanged in every arm. w = lam^2 over the joining columns.
Gaussians: torch.Generator seed 20260924 + layer, float32. Gate: runs only while memory_pressure free >= 35% and no porter S22 chain is running.
usage: join_sketch_test.py <arms e.g. A,B_lit,B,C> <out.jsonl>   (resumable)"""
import sys, json, time, subprocess, re
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; CV = P2 / 'fable_covhunter_v1'
sys.path.insert(0, str(CV)); sys.path.insert(0, str(P2 / 'fable_analyst_v1')); sys.path.insert(1, str(P2 / 'fable_engineer_v1'))
import numpy as np, torch
import s6bcard_joint2 as SJ, lrstar_torch_hook as lh, s2card
torch.set_num_threads(3); torch.set_grad_enabled(False)
RANK = 384
def make_push(qr, joins, arm):
    def push(torch, layer, w32, p_full, lam, previous_indices, Q, F, core, cohort_stops, rank=384):
        wp = w32[previous_indices, :].T
        before = [c for L_, cs in joins.items() if L_ < layer for c in cs]
        old_columns = cohort_stops[max(before)] if before else 0
        if layer in joins:
            start, stop = old_columns, cohort_stops[max(joins[layer])]
            J = p_full[previous_indices, start:stop]; weight = lam[start:stop] ** 2
            Omega = w32[previous_indices, :rank]
            Qp = Q[previous_indices, :] if old_columns else None
            g = torch.Generator().manual_seed(20260924 + layer)
            if arm == 'A':
                Y = J @ (weight[:, None] * (J.T @ Omega))
                if old_columns: Y = Y + Qp @ (core @ (Qp.T @ Omega))
            elif arm == 'B_lit':
                G = torch.randn(J.shape[1], rank, generator=g, dtype=torch.float32).to(J.dtype)
                Y = J @ (weight[:, None] * G)
                if old_columns: Y = Y + Qp @ (core @ (Qp.T @ Omega))
            elif arm == 'B':
                G1 = torch.randn(J.shape[1], rank, generator=g, dtype=torch.float32).to(J.dtype)
                Y = J @ (weight.sqrt()[:, None] * G1)
                if old_columns:
                    G2 = torch.randn(rank, rank, generator=g, dtype=torch.float32).to(J.dtype)
                    ev, V = torch.linalg.eigh(0.5 * (core + core.T).to(torch.float64))
                    csq = ((V * ev.clamp_min(0).sqrt()) @ V.T).to(J.dtype)
                    Y = Y + Qp @ (csq @ G2)
            elif arm == 'C':
                d = (J * J) @ weight
                if old_columns: d = d + ((Qp @ core) * Qp).sum(1)
                rows = torch.sort(torch.argsort(-d.to(torch.float64), stable=True)[:rank]).values
                Y = J @ (weight[:, None] * J[rows, :].T)
                if old_columns: Y = Y + Qp @ (core @ Qp[rows, :].T)
            else:
                raise ValueError(arm)
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
def gate():
    while True:
        busy = subprocess.run(['pgrep', '-f', '[f]ire_chain_S22.sh'], capture_output=True, text=True).stdout.split()
        mp = subprocess.run(['memory_pressure'], capture_output=True, text=True).stdout
        free = int(re.search(r'free percentage: (\d+)%', mp).group(1)) if 'free percentage' in mp else 0
        if not busy and free >= 35: return free
        time.sleep(30)
if __name__ == '__main__':
    arms = sys.argv[1].split(','); OUT = HERE / sys.argv[2]
    done = set()
    if OUT.exists(): done = {(j['arm'], j['rank']) for j in map(json.loads, OUT.read_text().splitlines())}
    sv = torch.load(P2 / 'fable_physicist_v1/refit6b/s6b_j05_ep2/params_epoch0.pt'); p, a, c, e = SJ.payload(torch)
    for k in p: p[k].copy_(sv['p'][k])
    a.copy_(sv['a']); c.copy_(sv['c'])
    for k in e: e[k].copy_(sv['e'][k])
    D = json.load(open(CV / 'dials_r1_ep2.json'))
    nets = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())['valid'][:16]
    with OUT.open('a') as fo:
        for r in nets:
            w = SJ.weights(r); tp = torch.tensor(SJ.truthpost(r)[15]).to(torch.float64)
            for arm in arms:
                if (arm, r) in done: continue
                free = gate(); t0 = time.time()
                SJ.DIALS = dict(D); cd = SJ.card(torch, w, p, a, c, e); lh.install(cd, torch)
                cd.shared['push'] = make_push(cd.shared['_qr'], s2card.EVERY3, arm)
                mse = float(((cd.run(False)[15].to(torch.float64) - tp) ** 2).mean())
                rec = dict(tag='PT-40 join sketch (S21-physics torch card, no refit)', arm=arm, rank=r, mse=mse, wall=round(time.time() - t0, 1), mem_free=free)
                fo.write(json.dumps(rec) + '\n'); fo.flush(); print(json.dumps(rec), flush=True)
