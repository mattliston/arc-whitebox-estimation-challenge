"""TAG fable-covhunter (CH-58): the j05 card with the LEARNED-ON joint Gauss-Newton dials FROZEN at the fit (gn_joint_fit.py ... on ->
gnjoint_j05ep0_on.json 'chosen': full GN step, lam 0.01; VALID16 learned ON -2.23 pct, 13/16 vs j05 epoch 0, no refit). The dials are
literals (repr of the fit). Card = s6bcard_joint with DIALS = FIT; STAR='lr32' (default None) also installs the engineer's S18 low-rank
star hook (fable_engineer_v1/lrstar_torch_hook.py). Drop-in card module (payload/model/card)."""
import sys
from pathlib import Path
sys.path.insert(1, str(Path(__file__).resolve().parent.parent / 'fable_engineer_v1'))
import s6bcard_joint as SJ
FIT = {'X1': 0.5308918270970678, 'X0': 0.0008298924162001187, 'X2': 0.20924729592278657, 'X3': -0.1173526402020965, 'bscale': 1.0508288007945212, 's_carry': 1.004248449059205, 's_new': 0.9924920858839599, 'c0': 0.7026110527329645, 'c1': 1.1230804063883197, 'k4a': 0.9914374263037713, 'k4b': 0.9518799990980005, 'k4c': 0.9540390389423499}
STAR = None
model, weights, truthpost, payload, PARAMS = SJ.model, SJ.weights, SJ.truthpost, SJ.payload, SJ.PARAMS


def card(torch, w, p, a, c, e, fine_ckpt=False, **kw):
    SJ.DIALS = dict(FIT)
    cd = SJ.card(torch, w, p, a, c, e, fine_ckpt=fine_ckpt, **kw)
    if STAR == 'lr32':
        import lrstar_torch_hook as lh
        lh.install(cd, torch)
    return cd
