"""TAG fable-covhunter (CH-64): ALTERNATING LOOP round 1 dials, FROZEN as literals, for the physicist's refit. Semantics: s6bcard_joint2
(CH-60, native-consistent: k4 x band at its definition incl. the GN recompute; c0/c1 shared with the GN r31 remainder). Fit: learned-ON
joint Gauss-Newton on the S18-HOOKED card with the j05 ep2 params (gn_joint_fit.py 8 3 gnjoint_r1_ep2_on_shared_s18 on shared, STAR=lr32,
PARAMS=s6b_j05_ep2/params_epoch0.pt): LOO -3.20%, real -4.12%; VALID16 hooked -2.10% (14/16). STAR='lr32' (default ON here) installs the
engineer's S18 low-rank star hook after the card; the refit wrapper re-wraps the star in s2card._ckpt and installs gn_linear_hook itself.
Drop-in card module (payload/model/card)."""
import sys
from pathlib import Path
sys.path.insert(1, str(Path(__file__).resolve().parent.parent / 'fable_engineer_v1'))
import s6bcard_joint2 as SJ
FIT = {'X1': 0.5590195295409507, 'X0': -0.006198436099218179, 'X2': 0.3584843180042713, 'X3': 0.15153807567151212, 'bscale': 1.0508502895680705, 's_carry': 1.0046882974947846, 's_new': 0.9931186860430007, 'c0': 0.7088619860998473, 'c1': 0.5026934507950427, 'k4a': 0.9889963697327108, 'k4b': 0.9626024746941586, 'k4c': 0.960122707694945}
STAR = 'lr32'
model, weights, truthpost, payload, PARAMS = SJ.model, SJ.weights, SJ.truthpost, SJ.payload, SJ.PARAMS


def card(torch, w, p, a, c, e, fine_ckpt=False, **kw):
    SJ.DIALS = dict(FIT)
    cd = SJ.card(torch, w, p, a, c, e, fine_ckpt=fine_ckpt, **kw)
    if STAR == 'lr32':
        import lrstar_torch_hook as lh
        lh.install(cd, torch)
    return cd
