#!/usr/bin/env python3
"""TAG fable-covhunter (CH-20): CERTIFICATE for the zero-cost native form of the covariance-only knobs. The torch validation computes
the knobbed covariance with a SECOND pair-map call (modified K21 and T31). Because the pair series is linear in T21 and T31, the same
change equals a closed-form delta on the covariance output built from the per-unit Hermite coefficients A_r (derivatives()):
  dcov = s s^T * [ (dT21 * t21a + dT21^T * t21b) / 2 + (dT31 * t31a + dT31^T * t31b) / 6 ]
  t21a = A2 A1^T + rho*(A3 A2^T) + rho^2/2*(A4 A3^T), t21b = its transpose pattern A1 A2^T + rho*(A2 A3^T) + rho^2/2*(A3 A4^T),
  t31a = A3 A1^T + rho*(A4 A2^T) + rho^2/2*(A5 A3^T), t31b = A1 A3^T + rho*(A2 A4^T) + rho^2/2*(A3 A5^T)   (outer products),
  dT21 = dK21 / (v_a s_c) (zero diagonal), dT31 = th31 * T31_row, rho = C/(s s^T) zero diagonal.
This script compares both forms on the card's own state (r35, several layers). usage: cert_delta.py <rank> <threads>"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
import covstep as cs
import gn2_knobs as g2
rank = int(sys.argv[1]); torch.set_num_threads(int(sys.argv[2]))
TH = dict(iso=-0.094, k21n=-0.028, k31=-0.212)
p, a, c, e = cs.model.torch_payload(torch, np, False)
card = cs.make(rank, p, a, c, e)
g2.configure(card, {}, base=True)            # K31 row factor on the main call only
n = 1024; eye = torch.eye(n, dtype=torch.bool)


def check(l, C, ctx):
    if l not in (2, 5, 9, 12, 14): return C
    mu, Cpre, k21, k4, k22, pair, env = ctx['mu'], ctx['Cpre'], ctx['k21'], ctx['k4'], ctx['k22'], ctx['pair'], card.env
    vv = torch.diagonal(Cpre); s = vv.sqrt(); al = mu / s
    kink = (al.abs() < 1.5).double(); km = (kink[:, None] * torch.ones(1, n, dtype=torch.float64)).masked_fill(eye, 0.)
    nm = ((1 - kink)[:, None] * torch.ones(1, n, dtype=torch.float64)).masked_fill(eye, 0.)
    k3 = torch.diagonal(k21); rho = (Cpre / torch.outer(s, s)).masked_fill(eye, 0.)
    r31 = g2.row_r31(mu, Cpre, k21, ctx['radial'])
    T31_row = (k4 / vv ** 2)[:, None] * rho + r31
    dK21 = TH['iso'] * km * (k3[:, None] * Cpre / vv[:, None]) + TH['k21n'] * nm * k21
    C_main = pair(mu, Cpre, k21, k4, k22, 'covariance', remainder31=r31)[1]
    C_two = pair(mu, Cpre, k21 + dK21, k4, k22, 'covariance', remainder31=r31 + TH['k31'] * T31_row)[1]
    # closed form
    _, al_, pdf, g, mg, *_ = env['gaussian_raw'](mu, vv)
    A, B = env['derivatives'](al_, pdf, g, mg / s, 6)
    O = torch.outer
    t21a = O(A[2], A[1]) + rho * O(A[3], A[2]) + rho ** 2 / 2 * O(A[4], A[3])
    t21b = O(A[1], A[2]) + rho * O(A[2], A[3]) + rho ** 2 / 2 * O(A[3], A[4])
    t31a = O(A[3], A[1]) + rho * O(A[4], A[2]) + rho ** 2 / 2 * O(A[5], A[3])
    t31b = O(A[1], A[3]) + rho * O(A[2], A[4]) + rho ** 2 / 2 * O(A[3], A[5])
    dT21 = (dK21 / (vv[:, None] * s[None, :])).masked_fill(eye, 0.)
    dT31 = (TH['k31'] * T31_row).masked_fill(eye, 0.)
    dcov = O(s, s) * ((dT21 * t21a + dT21.T * t21b) / 2 + (dT31 * t31a + dT31.T * t31b) / 6)
    dcov = dcov.masked_fill(eye, 0.)
    ref = C_two - C_main
    err = float(torch.linalg.norm(ref - dcov) / torch.linalg.norm(ref))
    print(f'L{l:02d} |second call - main| / |C_off| = {float(torch.linalg.norm(ref) / torch.linalg.norm(C_main.masked_fill(eye, 0.))):.2e}   '
          f'closed-form rel err = {err:.2e}   diag change {float(torch.diagonal(ref).abs().max()):.1e}', flush=True)
    return C


card.hooks['cov_pm'] = check
with torch.no_grad(): card.run_states()
