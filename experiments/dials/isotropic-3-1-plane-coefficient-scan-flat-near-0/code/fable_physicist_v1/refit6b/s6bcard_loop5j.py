"""TAG fable-physicist (09-25, lead's c0 line search on the S24 STRUCTURE): s6bcard_loop4j with the star replaced by S24's star-Nystrom
range finder (the porter's PT-48 'nysp' arm: Q = qr(A[:, 48 largest-norm columns]) + one power step, rank-32 SVD, s1 = 0; installed with
fable_porter_v1/star_nys_test.py install_star(card, torch, 'nysp', stats), in the porter's order: join, star, gn_linear_hook, pair series).
lvfuse2, S24's other change, is bit-identical arithmetic (request diet), so this is the S24 torch card. Dials via s6bcard_joint2t.DIALS_T
(the driver sets them). fine_ckpt=False always."""
import sys
from pathlib import Path
P2 = Path(__file__).resolve().parents[2]
for _d in ('fable_covhunter_v1', 'fable_engineer_v1', 'fable_analyst_v1', 'fable_physicist_v1/refit6b', 'fable_porter_v1'):
    if str(P2 / _d) not in sys.path: sys.path.append(str(P2 / _d))
import s6bcard_joint2t as JT
import join_nystrom_torch as jn, gn_linear_hook as gl, pair_series_torch as ps
import star_nys_test as SN
payload, weights, model, truthpost = JT.payload, JT.weights, JT.model, JT.truthpost


def card(torch, w, p, a, c, e, fine_ckpt=False, **kw):
    assert JT.DIALS_T is not None, 'set s6bcard_joint2t.DIALS_T before building the card'
    cd = JT.card(torch, w, p, a, c, e, fine_ckpt=False, **kw)
    jn.install(cd, torch); SN.install_star(cd, torch, 'nysp', []); gl.install(cd, torch); ps.install(cd, torch)
    return cd
