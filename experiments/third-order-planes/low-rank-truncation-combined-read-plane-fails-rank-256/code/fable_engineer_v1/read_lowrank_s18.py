#!/usr/bin/env python3
"""TAG fable-engineer (E-66, lead: "rank-truncate the BANK READ plane"): no-refit torch test on the S18 physics (s6bcard_j05 + S17
literals + lrstar_torch_hook). The bank read (shared['read']: young + old tiers, the (response rows x live columns) plane that goes into
cp and so into K21 before the pair map) is replaced at every layer by its exact top-r SVD (float64), r = 64 / 128 / 256. Per net:
final-layer MSE vs the 8M truth; per plane: shape and the rank-r relative Frobenius tail. Nets: the porter's 8 VALID nets.
usage (python with torch): read_lowrank_s18.py <arms e.g. base,r64,r128,r256> <threads> <out.jsonl>"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
sys.path.insert(0, str(P2 / 'fable_analyst_v1')); sys.path.insert(0, str(P2 / 'fable_physicist_v1/refit6b')); sys.path.insert(0, str(HERE))
import numpy as np, torch
import s6bcard, s6bcard_j05
import lrstar_torch_hook as lh
if __name__ == '__main__':
    arms = sys.argv[1].split(','); torch.set_num_threads(int(sys.argv[2])); OUT = HERE / sys.argv[3]
    if OUT.exists(): raise SystemExit(f"refusing to overwrite {OUT}")
    PL = s6bcard.payload(torch, params=P2 / 'fable_physicist_v1/refit6b/s6b_B2ep1_j05_1ep/params_epoch0.pt')
    NETS = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())['valid'][:8]
    truth = lambda r: np.load(P2 / 'fable_refit_v1/truth688' / f'truthpost_rank{r:04d}.npz')['mc_post'][15].astype(np.float64)
    with OUT.open('a') as fo:
        for rk in NETS:
            w, tp = s6bcard.weights(rk), truth(rk)
            for arm in arms:
                t0 = time.time(); cd = lh.install(s6bcard_j05.card(torch, w, *PL), torch)
                tails = []
                if arm != 'base':
                    r = int(arm[1:]); read0 = cd.shared['read']
                    def read(t, pp, *args, _r=r, _read0=read0, **kw):
                        P = _read0(t, pp, *args, **kw)
                        P64 = P.to(torch.float64); U, S, Vh = torch.linalg.svd(P64, full_matrices=False)
                        k = min(_r, S.numel()); e = S * S
                        tails.append([list(P.shape), float((e[k:].sum() / e.sum()).sqrt())])
                        return ((U[:, :k] * S[:k]) @ Vh[:k]).to(P.dtype)
                    cd.shared['read'] = read
                with torch.no_grad(): pred = cd.run(False).numpy().astype(np.float64)[15]
                rec = dict(tag='E-66 bank-read plane rank truncation (S18 physics, no refit)', rank=rk, arm=arm, mse=float(((pred - tp) ** 2).mean()),
                           wall=round(time.time() - t0, 1), planes=tails)
                fo.write(json.dumps(rec) + '\n'); fo.flush()
                print(json.dumps({k: v for k, v in rec.items() if k != 'planes'} | ({'tail_mean': round(float(np.mean([x[1] for x in tails])), 4)} if tails else {})), flush=True)
