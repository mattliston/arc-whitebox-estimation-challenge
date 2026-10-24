"""TAG fable-physicist (RF6-13): the analyst's s6bcard with the born-unit-side plane-to-distinct feed j1 at PHYSICAL direction and strength
J1 (unorthogonalized -(g_i/2) eta_a K21[a,i] at coefficient J1, replacing the learned c1 * orth(dir1); carry/variant_eval.py make_bd, bit-exact
when off). Optional BSCALE multiplies the carried b (b_post hook). Basis: leadergap/RESULT.md section L (learned-off VALID16 J1 0.5: -6.3%,
15/16). Drop-in for refit_s6b.py --card s6bcard_j05 (exports payload, model, card)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'carry'))
import s6bcard
from s6bcard import payload, model, ZC
J1, BSCALE = 0.5, None


def card(torch, w, p, a, c, e, fine_ckpt=False, zc=ZC, knobs=True):
    import variant_bd
    cd = s6bcard.card(torch, w, p, a, c, e, fine_ckpt=fine_ckpt, zc=zc, knobs=knobs)
    bd = variant_bd.make_bd(torch, 'phys', J1)
    if fine_ckpt:
        import s2card
        bd = s2card._ckpt(torch, bd)
    cd.env['_bd_update'] = bd
    if BSCALE is not None: cd.hooks['b_post'] = lambda l, b, ctx: b * BSCALE
    return cd
