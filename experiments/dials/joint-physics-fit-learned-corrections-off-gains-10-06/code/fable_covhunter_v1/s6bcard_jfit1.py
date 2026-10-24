"""TAG fable-covhunter (CH-54): the j05 card with the L1 joint learned-OFF Gauss-Newton dials FROZEN at the fit (gn_joint_fit.py ->
gnjoint_j05ep0_off.json 'chosen': full GN step, lam 0.01; VALID16 learned-OFF -10.06 pct, 16/16 vs the j05 off-control). The dials are
literals (repr of the fit). Card = s6bcard_joint with DIALS = FIT. Drop-in card module (payload/model/card) for refit6b_aux.py --card."""
import s6bcard_joint as SJ
FIT = {'X1': 0.6826247483386554, 'X0': -0.06129994241408997, 'X2': 0.2947519471787973, 'X3': -0.4154392465020926, 'bscale': 1.0264213658457995, 's_carry': 1.0075582586805119, 's_new': 0.9870938211535366, 'c0': 0.45797436656942137, 'c1': 0.39562594149143887, 'k4a': 0.9537722153259945, 'k4b': 0.9533706763865709, 'k4c': 0.9153811702539767}
model, weights, truthpost, payload, PARAMS = SJ.model, SJ.weights, SJ.truthpost, SJ.payload, SJ.PARAMS


def card(torch, w, p, a, c, e, fine_ckpt=False, **kw):
    SJ.DIALS = dict(FIT)
    return SJ.card(torch, w, p, a, c, e, fine_ckpt=fine_ckpt, **kw)
