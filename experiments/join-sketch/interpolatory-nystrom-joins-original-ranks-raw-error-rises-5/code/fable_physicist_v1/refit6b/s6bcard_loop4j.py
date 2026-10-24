"""TAG fable-physicist (loopR3 card, 09-25): the S23 STRUCTURE with TRAINABLE dials, for the lead's JOINT refit (learned parts + dials).
= covhunter's s6bcard_joint2t (CH-71: the 12 dials as 0-dim float64 tensors, native-shared semantics, forward certified bit-exact vs
s6bcard_joint2) + join_nystrom_torch (E-69: S23's Nystrom old-tier join) + lrstar_s1zero_hook (star s1 = 0) + gn_linear_hook + the
pair_series loop form, installed in covhunter's round-3b order (gn_joint_fit.py hooked(): join, star, gl, ps). fine_ckpt=False always.
The driver sets s6bcard_joint2t.DIALS_T = make_dials(torch, values, trainable) ONCE (values = the round-1 dials, fable_covhunter_v1/
dials_r1_ep2.json) and trains those tensors. card() refuses to build while DIALS_T is unset (joint2t would fall back to BASE, not the
deployed dials)."""
import sys
from pathlib import Path
P2 = Path(__file__).resolve().parents[2]
for _d in ('fable_covhunter_v1', 'fable_engineer_v1', 'fable_analyst_v1', 'fable_physicist_v1/refit6b'):
    if str(P2 / _d) not in sys.path: sys.path.append(str(P2 / _d))
import s6bcard_joint2t as JT
import join_nystrom_torch as jn, gn_linear_hook as gl, pair_series_torch as ps
import lrstar_s1zero_hook as lz
payload, weights, model, truthpost = JT.payload, JT.weights, JT.model, JT.truthpost


def card(torch, w, p, a, c, e, fine_ckpt=False, **kw):
    assert JT.DIALS_T is not None, 'set s6bcard_joint2t.DIALS_T (the round-1 dials as tensors) before building the loopR3 card'
    cd = JT.card(torch, w, p, a, c, e, fine_ckpt=False, **kw)
    jn.install(cd, torch); lz.install(cd, torch); gl.install(cd, torch); ps.install(cd, torch)
    return cd
