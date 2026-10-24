#!/usr/bin/env python3
"""TAG fable-porter (PT-25, lead GO 17:3x "lever 1"): no-refit torch test of the STAR (K21-plane) TRANSPORT on the S6b card (fable_analyst_v1/s6bcard.py:
HookCard + v3 knobs + L15 row cut) with the B2 epoch-1 params, 8 VALID nets (inputs.json valid[:8]). The star plane leaving layer l (state tuple
index 4) feeds ONLY layer l+1's star term sp = _flms_star(w, star) (and, leaving L14, the final layer's star diagonal). Arms via the 'state_out'
hook (unused by s6bcard; any existing hook is chained): base; zero the star term at layers L1-4 / L5-8 / L9-11 / L12-14 (= zero the plane leaving
layers 0-3 / 4-7 / 8-10 / 11-13); rank-r truncation (SVD, float64) of the plane leaving every layer 0-14, r = 32 and 128.
usage: star_ablation_s6b.py <arms e.g. base,z1_4,z5_8,z9_11,z12_14,r32,r128> <threads> <out.jsonl>"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; sys.path.insert(0, str(P2 / 'fable_analyst_v1'))
import numpy as np, torch
import s6bcard
ZERO = {'z1_4': range(0, 4), 'z5_8': range(4, 8), 'z9_11': range(8, 11), 'z12_14': range(11, 14)}
def make_hook(arm, prev):
    def hook(layer, out, ctx):
        if prev is not None: out = prev(layer, out, ctx)
        if layer > 14 or arm == 'base': return out
        star = out[4]
        if arm in ZERO:
            if layer not in ZERO[arm]: return out
            star = torch.zeros_like(star)
        else:
            r = int(arm[1:]); U, S, Vh = torch.linalg.svd(star.to(torch.float64), full_matrices=False)
            star = ((U[:, :r] * S[:r]) @ Vh[:r]).to(star.dtype)
        return tuple(star if i == 4 else v for i, v in enumerate(out))
    return hook
if __name__ == '__main__':
    arms = sys.argv[1].split(','); torch.set_num_threads(int(sys.argv[2])); OUT = HERE / sys.argv[3]
    B2 = P2 / 'fable_physicist_v1/refit6b/s6b_B2_fullnet_lr1e4_trunk5e5/params_epoch1.pt'
    pl = s6bcard.payload(torch, params=B2)
    NETS = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())['valid'][:8]
    truth = lambda r: np.load(P2 / 'fable_refit_v1/truth688' / f'truthpost_rank{r:04d}.npz')['mc_post'][15].astype(np.float64)
    with OUT.open('a') as fo:
        for r in NETS:
            w, tp = s6bcard.weights(r), truth(r)
            for arm in arms:
                t0 = time.time(); cd = s6bcard.card(torch, w, *pl); cd.hooks['state_out'] = make_hook(arm, cd.hooks.get('state_out'))
                with torch.no_grad(): pred = cd.run(False).numpy().astype(np.float64)[15]
                rec = dict(tag='PT-25 star ablation (s6bcard, B2 ep1, no refit)', rank=r, arm=arm, mse=float(((pred - tp) ** 2).mean()), wall=round(time.time() - t0, 1))
                fo.write(json.dumps(rec) + '\n'); fo.flush(); print(json.dumps(rec), flush=True)
