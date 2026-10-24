#!/usr/bin/env python3
"""TAG fable-refit (RF-13): per-unit fourth cumulant from quantities the card already carries: k4_i = radial v_i^2 + beta (k3_i^2/v_i - s v_i^2),
s = sum(k3^2/v)/sum(v^2) (zero-sum correction: layer total of k4 unchanged). Injected into the pair map (marginals, t4/T31, K22 diagonal;
optional K22 off-diagonal rescale by sqrt(r_i r_j)) at layers >= FROM and into the final scalar map. No refit.
Truth-regression motivation: R2 of 8M truth k4 on [v^2, k3^2/v] = 0.71-0.89 at L12-15 vs 0.58-0.76 for v^2 alone, coef ~0.7.
usage: k4_from_k3_test.py <ranks> <betas e.g. 0.3,0.5,0.7> <from_layer> <offdiag 0|1> <threads>"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
from common import *
import model_bank_research as model
ranks = [int(x) for x in sys.argv[1].split(',')]; betas = [float(x) for x in sys.argv[2].split(',')]; FROM = int(sys.argv[3]); OFF = int(sys.argv[4]); torch.set_num_threads(int(sys.argv[5])); LEARNED_OFF = len(sys.argv) > 6 and sys.argv[6] == 'off'; DECOUPLE = len(sys.argv) > 7 and sys.argv[7] == 'decouple'
P2 = HERE.parent; STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
p, a, c, e = model.torch_payload(torch, np, False)
if LEARNED_OFF:
    with torch.no_grad(): p['head'].zero_(); p['head_b'].zero_(); e['output'].zero_(); e['bias'].zero_()
def k4_new(k3, v, k4_iso, beta):
    t = torch
    q = k3 * k3 / v
    s = q.sum() / (v * v).sum()
    return k4_iso + beta * (q - s * v * v)
for beta in [0.0] + betas:
    for rank in ranks:
        w = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
        truth = torch.tensor(np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"][15].astype(np.float64))
        card = model.Card(torch, w, p, a, c, e)
        f = card.env; orig_pair = f['_nonlinear_pair_batch']; orig_scalar = f['nonlinear_scalar']
        def pair(mu, C, K21, k4, k22_override=None, metric='identity', card=card, beta=beta):
            if beta and card.call_layer >= FROM:
                v = torch.diagonal(C); k3 = torch.diagonal(K21)
                kn = k4_new(k3, v, k4, beta)
                kn = torch.maximum(kn, 0.1 * k4) if bool((k4 > 0).all()) else kn
                if k22_override is not None:
                    if OFF:
                        r = torch.sqrt(torch.clamp(kn / k4, min=0.05)); k22_override = k22_override * r[:, None] * r[None, :]
                    k22_override = torch.diagonal_scatter(k22_override, kn)
                k4_old = k4; k4 = kn
                out = orig_pair(mu, C, K21, k4, k22_override, metric)
                if DECOUPLE:
                    _m0, aux0 = orig_scalar(mu, torch.diagonal(C), torch.diagonal(K21), k4_old)
                    out[4]['pre_k4'] = aux0['pre_k4']; out[4]['post_k4'] = aux0['post_k4']
                return out
            return orig_pair(mu, C, K21, k4, k22_override, metric)
        def scalar(mu, v, k3, k4, beta=beta):
            if beta and 15 >= FROM:
                kn = k4_new(k3, v, k4, beta); kn = torch.maximum(kn, 0.1 * k4) if bool((k4 > 0).all()) else kn
                m_, aux_ = orig_scalar(mu, v, k3, kn)
                if DECOUPLE:
                    _m0, aux0 = orig_scalar(mu, v, k3, k4)
                    aux_['pre_k4'] = aux0['pre_k4']; aux_['post_k4'] = aux0['post_k4']
                return m_, aux_
            return orig_scalar(mu, v, k3, k4)
        f['_nonlinear_pair_batch'] = pair; f['nonlinear_scalar'] = scalar
        t0 = time.time()
        with torch.no_grad(): pred = card.run(False)
        print(json.dumps(dict(decouple=DECOUPLE, learned='off' if LEARNED_OFF else 'on', beta=beta, from_layer=FROM, offdiag=OFF, rank=rank, mse=float(((pred[15].to(torch.float64) - truth) ** 2).mean()), wall=round(time.time() - t0, 1))), flush=True)
