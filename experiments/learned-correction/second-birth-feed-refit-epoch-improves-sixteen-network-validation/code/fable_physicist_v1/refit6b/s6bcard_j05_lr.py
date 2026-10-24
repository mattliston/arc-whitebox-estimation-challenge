"""TAG fable-physicist (RF6-15): the j05 card (s6bcard_j05: j1 physical at 0.5) on the S18 physics. The engineer's certified low-rank star
plane (fable_engineer_v1/lrstar_torch_hook.py: rank 32, seeded Omega, 1 power step, detached basis) is installed AFTER the card is built,
as the lead directs for all refits from 09-24 evening on. It replaces env['_flms_star'] and env['_flms_star_diagonal'] only (no overlap
with the j05 _bd_update). With fine_ckpt the new star is re-wrapped in s2card._ckpt (engineer's recipe), so checkpointing is kept.
Drop-in for refit_s6b.py --card."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'fable_engineer_v1'))
import s6bcard_j05
from s6bcard_j05 import payload, model, ZC
import lrstar_torch_hook as lh


def card(torch, w, p, a, c, e, fine_ckpt=False, zc=ZC, knobs=True):
    cd = s6bcard_j05.card(torch, w, p, a, c, e, fine_ckpt=fine_ckpt, zc=zc, knobs=knobs)
    lh.install(cd, torch)
    if fine_ckpt:
        import s2card
        cd.env['_flms_star'] = s2card._ckpt(torch, cd.env['_flms_star'])
    return cd
