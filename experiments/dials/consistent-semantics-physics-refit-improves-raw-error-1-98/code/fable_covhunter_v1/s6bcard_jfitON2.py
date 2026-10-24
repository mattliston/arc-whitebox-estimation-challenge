"""TAG fable-covhunter (CH-61): the j05 card with the LEARNED-ON joint Gauss-Newton dials re-fit on NATIVE-CONSISTENT semantics
(s6bcard_joint2, CH-60: k4 x band at its definition incl. the GN cov recompute; c0/c1 shared with the GN r31), FROZEN as literals
(gn_joint_fit.py 8 3 gnjoint_j05ep0_on_shared on shared -> 'chosen'). STAR='lr32' installs the engineer's S18 low-rank star hook.
Drop-in card module (payload/model/card); S19_fit_shared.json holds the same dict for the porter's build_s19_dials.py."""
import sys
from pathlib import Path
sys.path.insert(1, str(Path(__file__).resolve().parent.parent / 'fable_engineer_v1'))
import s6bcard_joint2 as SJ
FIT = {'X1': 0.5327927103206771, 'X0': 0.0005233752643266338, 'X2': 0.21341120600433072, 'X3': -0.22827770595122038, 'bscale': 1.0487528105836672, 's_carry': 1.004386516901868, 's_new': 0.9931997162515536, 'c0': 0.6526631556593073, 'c1': 1.6084004427565655, 'k4a': 0.9883072499548652, 'k4b': 0.9617904882420111, 'k4c': 0.963137292333831}
STAR = None
model, weights, truthpost, payload, PARAMS = SJ.model, SJ.weights, SJ.truthpost, SJ.payload, SJ.PARAMS


def card(torch, w, p, a, c, e, fine_ckpt=False, **kw):
    SJ.DIALS = dict(FIT)
    cd = SJ.card(torch, w, p, a, c, e, fine_ckpt=fine_ckpt, **kw)
    if STAR == 'lr32':
        import lrstar_torch_hook as lh
        lh.install(cd, torch)
    return cd
