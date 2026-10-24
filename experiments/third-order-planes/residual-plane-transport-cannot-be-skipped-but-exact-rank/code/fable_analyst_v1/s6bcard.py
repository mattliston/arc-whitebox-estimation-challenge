"""TAG fable-analyst (AN-42): the FIRED S6b physics (#332005, fable_fire_v1/fire_candS6b/candS6b_mem.py, sha a514c21b) as a torch research card.
  S2 physics (this directory's s2card.py: K31 row .85/.5 at pair layers 1-14, gain b-fix gamma .1 partition mean m, every3 joins) on
  covhunter's HookCard (fable_covhunter_v1/covstep.py + covstep_s2.make_s2, bit-exact vs s2card.py)
  + the v3 covariance knobs (fable_covhunter_v1/gn3_knobs.configure(V3): iso -.094, k21n -.028, k31 -.212, edge +.164, rc -.080
    = fable_porter_v1/gn_knobs/v3.json)
  + the physicist's L15 k3 row cut (fable_physicist_v1/leadergap/build_l15_rowcut.py, zc 3.5): the CP and star parts of the final k3 only on
    rows select(zc - |mu/sigma|, 0) (rounded up to 32 by the card's select, stable order); every row keeps the 3 v b term
  + the refit literals fable_analyst_v1/refit_s2_full_lr2e4_v2/params_epoch2.pt (all four learned groups p, a, c, e).
The kernel layers of the fired file (HEB, push5 depth rule, f32 third, evict5, batch, mem/yright/arena diet) are rounding-level and not
modelled. payload(torch) -> (p, a, c, e) = params_epoch2 on the parent's payload tensors (float32 leaves, requires_grad False);
card(torch, w, p, a, c, e, fine_ckpt=False, zc=3.5) -> the configured HookCard. run(True) checkpoints each layer; fine_ckpt adds nested
non-reentrant checkpoints (pair map, _bd_update, star push, tier push, plane read) for backward passes (memory only, same values)."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
for _p in (str(HERE), str(P2 / 'fable_covhunter_v1')):
    if _p not in sys.path: sys.path.insert(0, _p)
import numpy as np
import covstep as cs
import gn3_knobs as gn3
import s2card
PARAMS = HERE / 'refit_s2_full_lr2e4_v2' / 'params_epoch2.pt'
ZC = 3.5
V3 = dict(gn3.V3)
model = cs.model
weights, truthpost = cs.weights, cs.truthpost


def payload(torch, params=PARAMS):
    """(p, a, c, e) = the parent's learned payload with the refit values copied in (same float32 leaves as torch_payload)."""
    p, a, c, e = model.torch_payload(torch, np, False)
    if params is not None:
        sv = torch.load(params)
        assert set(sv) == {'p', 'a', 'c', 'e'} and set(sv['p']) == set(p) and set(sv['e']) == set(e)
        with torch.no_grad():
            for k in p: p[k].copy_(sv['p'][k])
            a.copy_(sv['a']); c.copy_(sv['c'])
            for k in e: e[k].copy_(sv['e'][k])
    return p, a, c, e


def l15_rowcut(torch, zc):
    def hook(layer, k3, ctx):
        score = zc - (ctx['mu'] / ctx['vv'].clamp_min(1e-10).sqrt()).abs()
        count = min(1024, ((int((score.detach() >= 0).sum()) + 31) // 32) * 32)
        keep = torch.zeros_like(score)
        keep[torch.argsort(-score.detach(), stable=True)[:count]] = 1.0
        return keep * (ctx['bank'] + ctx['star']) + ctx['bterm']
    return hook


def card(torch, w, p, a, c, e, fine_ckpt=False, zc=ZC, knobs=True):
    cd = cs.HookCard(torch, w, p, a, c, e)
    gn3.configure(cd, V3 if knobs else {})             # make_s2 (K31 row, b-fix .1, every3) + the v3 knob hooks
    if zc is not None: cd.hooks['k3_15'] = l15_rowcut(torch, zc)
    if fine_ckpt:
        for k in ('_nonlinear_pair_batch', '_bd_update', '_flms_star'): cd.env[k] = s2card._ckpt(torch, cd.env[k])
        for k in ('push', 'read'): cd.shared[k] = s2card._ckpt(torch, cd.shared[k])
    return cd


NATIVE = {'S6b': {35: 1.5653664131268201e-08, 100: 1.5702289026456161e-08},     # fable_porter_v1/stack/parity_S6b_vs_S6.txt
          'S5e': {35: 1.572180e-08, 100: 1.583176e-08}}                           # fable_porter_v1/stack/parity_S6.txt (S6 vs S5e)

if __name__ == '__main__':
    # usage: s6bcard.py parity <ranks> <threads>  -> torch S6b vs native S6b, plus the no-refit card (= S5e physics) vs native S5e
    import json, time, torch
    ranks = [int(x) for x in sys.argv[2].split(',')]; torch.set_num_threads(int(sys.argv[3]))
    pl_refit, pl_base = payload(torch), payload(torch, params=None)
    for rank in ranks:
        w, tp = weights(rank), truthpost(rank)
        for label, pl, nat in (('S6b', pl_refit, 'S6b'), ('S5e-physics (no refit)', pl_base, 'S5e')):
            t0 = time.time()
            with torch.no_grad(): pred = card(torch, w, *pl).run(False).numpy().astype(np.float64)
            mse = float(((pred[15] - tp[15]) ** 2).mean()); ref = NATIVE[nat].get(rank)
            print(json.dumps(dict(rank=rank, card=label, torch_mse=mse, native=ref, torch_vs_native=None if ref is None else mse / ref - 1,
                                  wall=round(time.time() - t0, 1))), flush=True)
