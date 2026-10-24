"""TAG fable-covhunter (CH-49): the physicist's s6bcard_j05 pattern (fable_physicist_v1/refit6b/s6bcard_j05.py, RF6-13; copied, their file
untouched) at j1 strength J1 = 0.3 instead of 0.5 (lead's hedge): the analyst's s6bcard with _bd_update replaced by
carry/variant_bd.make_bd(torch, 'phys', J1) = the born-unit-side plane-to-distinct feed -(g_i/2) eta_a K21[a,i], unorthogonalized, at
coefficient J1, in place of the learned c1 * orth(dir1). BSCALE None. Drop-in for refit6b_aux.py --card s6bcard_j03 (payload, model, card)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'fable_physicist_v1/refit6b/carry'))
import s6bcard
from s6bcard import payload, model, ZC
J1 = 0.3


def card(torch, w, p, a, c, e, fine_ckpt=False, zc=ZC, knobs=True):
    import variant_bd
    cd = s6bcard.card(torch, w, p, a, c, e, fine_ckpt=fine_ckpt, zc=zc, knobs=knobs)
    bd = variant_bd.make_bd(torch, 'phys', J1)
    if fine_ckpt:
        import s2card
        bd = s2card._ckpt(torch, bd)
    cd.env['_bd_update'] = bd
    return cd
