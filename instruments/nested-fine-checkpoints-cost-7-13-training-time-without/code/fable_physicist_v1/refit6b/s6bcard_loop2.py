"""TAG fable-physicist (RF6-17, engineer E-67, lead GO; for ROUND-2 refits on): s6bcard_loop.py (RF6-16: covhunter's frozen-dial module via
env LOOP_DIALMOD with STAR forced to 'lr32', + gn_linear_hook) with the E-67 refit-step drop-in: nested fine checkpoints OFF (the harness's
per-layer checkpoint bounds memory; gradients bit-identical) and the engineer's loop-form pair series (pair_series_torch; the same float32
Hermite series, only the summation order differs: predictions ~1e-7 relative, final MSE ~1e-4). No star re-wrap (no fine checkpoints).
Every run using it states "fine_ckpt=False + pair_series loop form" in its TAG (lead's condition). s6bcard_loop.py is left unchanged
(it is the card of the running round-1 refit s6b_loopR1). Drop-in for refit_s6b.py --card s6bcard_loop2."""
import os, sys, importlib
from pathlib import Path
P2 = Path(__file__).resolve().parents[2]
for d in ('fable_covhunter_v1', 'fable_engineer_v1'):
    if str(P2 / d) not in sys.path: sys.path.insert(1, str(P2 / d))
DM = importlib.import_module(os.environ['LOOP_DIALMOD'])
DM.STAR = 'lr32'
import gn_linear_hook as gl
import pair_series_torch as ps
payload, model = DM.payload, DM.model


def card(torch, w, p, a, c, e, fine_ckpt=False, **kw):
    cd = DM.card(torch, w, p, a, c, e, fine_ckpt=False, **kw)   # E-67: nested fine checkpoints off
    gl.install(cd, torch)
    ps.install(cd, torch)                                          # E-67: loop-form pair series
    return cd
