"""TAG fable-physicist (RF6-18, lead 22:2x): the loopR2 refit card = s6bcard_loop2 (RF6-17: covhunter's frozen-dial module via LOOP_DIALMOD,
gn_linear_hook, engineer E-67 fine_ckpt off + pair_series loop form) with the S18 star taken from refit6b/lrstar_s1zero_hook.py (S1 = 'zero':
the s1 term dropped, engineer2 / lead) instead of the engineer's lrstar hook. The dial module's own STAR install is switched off (STAR=None)
and the s1-zero star is installed after the build. Tag every run "fine_ckpt=False + pair_series loop form + s1=0". Drop-in --card."""
import os, sys, importlib
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parents[1]
for d in ('fable_covhunter_v1', 'fable_engineer_v1'):
    if str(P2 / d) not in sys.path: sys.path.insert(1, str(P2 / d))
if str(HERE) not in sys.path: sys.path.insert(1, str(HERE))
DM = importlib.import_module(os.environ['LOOP_DIALMOD'])
DM.STAR = None
import gn_linear_hook as gl
import pair_series_torch as ps
import lrstar_s1zero_hook as lz
payload, model = DM.payload, DM.model


def card(torch, w, p, a, c, e, fine_ckpt=False, **kw):
    cd = DM.card(torch, w, p, a, c, e, fine_ckpt=False, **kw)   # E-67: nested fine checkpoints off
    lz.install(cd, torch)                                          # S18 low-rank star with s1 = 0
    gl.install(cd, torch)
    ps.install(cd, torch)                                          # E-67: loop-form pair series
    return cd
