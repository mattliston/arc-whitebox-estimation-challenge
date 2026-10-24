#!/usr/bin/env python3
"""TAG fable-porter (PT-29, lead GO 18:25 "torch VALID40 at r = 32 and 64, plus r=32 on the j05 card; bar raw <= +0.8%, CI upper <= +1.5%"):
star (K21-plane) RANK TRUNCATION on VALID40 (inputs.json valid[:40]), torch, no refit. Hook = star_ablation_s6b.make_hook (PT-25: float64 SVD of
the plane leaving every layer 0-14, keep the top r). Cards: 'b2' = fable_analyst_v1/s6bcard + B2 epoch-1 params (the S11-S16 literals);
'j05' = fable_physicist_v1/refit6b/s6bcard_j05 (j1 leg physical, 0.5) + s6b_B2ep1_j05_1ep/params_epoch0.pt (the S17 literals).
Base rows = the physicist's eval_n40.json files (this harness reproduces the b2 ones to <= 3.3e-16 relative on valid[:8]; the j05 ones are
checked here on 2 nets by the 'j05:base' jobs). MSE = mean over layer 15 of (pred - truthpost)^2, as refit_s6b.py.
Threads: MAXT (argv[2], default 3), or 1 while any S17 native process (command line containing stack_S17p_j05_prebind_mem; was stack_S17_j05_mem until 18:12, which
also matched a teammate's process) is alive (porter
budget <= 3 threads total). usage: star_valid40_s6b.py <out.jsonl> [MAXT]   (resumable: skips (card, arm, rank) rows already in out.jsonl)"""
import sys, json, time, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
sys.path.insert(0, str(P2 / 'fable_analyst_v1')); sys.path.insert(0, str(P2 / 'fable_physicist_v1/refit6b')); sys.path.insert(0, str(HERE))
import numpy as np, torch
import s6bcard, s6bcard_j05
from star_ablation_s6b import make_hook
RF = P2 / 'fable_physicist_v1/refit6b'
CARDS = {'b2': (s6bcard, RF / 's6b_B2_fullnet_lr1e4_trunk5e5/params_epoch1.pt'), 'j05': (s6bcard_j05, RF / 's6b_B2ep1_j05_1ep/params_epoch0.pt')}
def threads():
    busy = subprocess.run(['pgrep', '-f', 'stack_S17p_j05_prebind_mem'], capture_output=True, text=True).stdout.split()
    return 1 if busy else (int(sys.argv[2]) if len(sys.argv) > 2 else 3)
if __name__ == '__main__':
    OUT = HERE / sys.argv[1]
    NETS = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())['valid'][:40]
    done = set()
    if OUT.exists(): done = {(j['card'], j['arm'], j['rank']) for j in map(json.loads, OUT.read_text().splitlines())}
    PL = {k: s6bcard.payload(torch, params=v[1]) for k, v in CARDS.items()}
    truth = lambda r: np.load(P2 / 'fable_refit_v1/truth688' / f'truthpost_rank{r:04d}.npz')['mc_post'][15].astype(np.float64)
    with OUT.open('a') as fo:
        for i, r in enumerate(NETS):
            todo = [('b2', 'r32'), ('b2', 'r64'), ('j05', 'r32')] + ([('j05', 'base')] if i < 2 else [])
            todo = [(c, a) for c, a in todo if (c, a, r) not in done]
            if not todo: continue
            w, tp = s6bcard.weights(r), truth(r)
            for card, arm in todo:
                nt = threads(); torch.set_num_threads(nt); t0 = time.time(); mod = CARDS[card][0]
                cd = mod.card(torch, w, *PL[card]); cd.hooks['state_out'] = make_hook(arm, cd.hooks.get('state_out'))
                with torch.no_grad(): pred = cd.run(False).numpy().astype(np.float64)[15]
                rec = dict(tag='PT-29 star rank VALID40 (no refit)', card=card, arm=arm, rank=r, mse=float(((pred - tp) ** 2).mean()),
                           threads=nt, wall=round(time.time() - t0, 1))
                fo.write(json.dumps(rec) + '\n'); fo.flush(); print(json.dumps(rec), flush=True)
