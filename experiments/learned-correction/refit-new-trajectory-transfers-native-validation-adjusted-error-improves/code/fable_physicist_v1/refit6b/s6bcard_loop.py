"""TAG fable-physicist (RF6-16): the refit card for the lead's ALTERNATING LOOP (dials <-> refit). It wraps covhunter's frozen-dial card
module (env LOOP_DIALMOD, e.g. s6bcard_jfitON3, looked up in fable_covhunter_v1; its dials are literals), forces the engineer's S18
low-rank star on (module STAR = 'lr32'), installs the engineer's gn_linear_hook (the native _gn_dcov linear GN-knob correction; lead GO for
all refits; reads ctx['k4'] and _ROWF so dialed k4 bands and c0/c1 carry over), and, with fine_ckpt, re-wraps the S18 star in s2card._ckpt
(the hook itself is not checkpointed). Drop-in for refit_s6b.py --card s6bcard_loop (exports payload, model, card)."""
import os, sys, importlib
from pathlib import Path
P2 = Path(__file__).resolve().parents[2]
for d in ('fable_covhunter_v1', 'fable_engineer_v1'):
    if str(P2 / d) not in sys.path: sys.path.insert(1, str(P2 / d))
DM = importlib.import_module(os.environ['LOOP_DIALMOD'])
DM.STAR = 'lr32'
import gn_linear_hook as gl
payload, model = DM.payload, DM.model


def card(torch, w, p, a, c, e, fine_ckpt=False, **kw):
    cd = DM.card(torch, w, p, a, c, e, fine_ckpt=fine_ckpt, **kw)
    gl.install(cd, torch)
    if fine_ckpt:
        import s2card
        cd.env['_flms_star'] = s2card._ckpt(torch, cd.env['_flms_star'])
    return cd
