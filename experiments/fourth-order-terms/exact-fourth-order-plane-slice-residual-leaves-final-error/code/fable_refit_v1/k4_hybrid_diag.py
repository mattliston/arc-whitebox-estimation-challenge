#!/usr/bin/env python3
"""TAG fable-refit (RF-1): does the (2,2)+(4)-slice residual of the post-activation fourth cumulant (post22, which the card computes
and collapses to the scalar 'radial') carry the per-unit structure of the next layer's pre-activation k4?  On the research torch card:
inject a pair map that also returns post22/cov, form Delta = post22 - radial_new*(cov_aa cov_bb + 2 cov_ab^2)/3 (diagonal: post_k4 -
radial_new*cov_aa^2), and at the next layer compute k4_hyb = k4_iso + 3 (X^T Delta X)_jj - 2 sum_a X_aj^2 Delta_aa with X = W.W.
Compare k4_iso and k4_hyb with the 8M truth per layer (rel rms error, corr). Optionally (mode=apply) feed k4_hyb into the card.
usage: k4_hybrid_diag.py <rank> <threads> <mode diag|apply>"""
import sys, os, time, json, ast
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
from common import *
import model_bank_research as model
rank, threads, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
torch.set_num_threads(threads)
P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
w = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
truth = torch.tensor(np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64))
tr = np.load(P2 / "public_g19_attribution_v1" / f"mps_truth_rank{rank:04d}_P8M.npz"); k4_true = tr["mc_pre_k4"].astype(np.float64)
planes_path = P2 / "public_g19_attribution_v1" / f"mps_planes_rank{rank:04d}_new.npz"
planes = np.load(planes_path) if planes_path.exists() else None

# --- inject a pair map that also exports post22 and cov ---
text = model.SOURCE.read_text(); tree = ast.parse(text)
node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_nonlinear_pair_batch'][0]
src = ast.get_source_segment(text, node)
old = "aux = dict(sigma=s, alpha=a, pdf=pdf, gate=g, gaussian_mean=mg, pre_v=v, pre_cov=C, pre_k3=k3, pre_k4=fnp.broadcast_to(k4, (n,)).copy(), post_v=var, post_k3=post_k3, post_k4=post_k4)"
assert src.count(old) == 1
src = src.replace(old, old[:-1] + ", post22=post22, cov22=cov)")
p, a, c, e = model.torch_payload(torch, np, False)
card = model.Card(torch, w, p, a, c, e)
mod_tree = model.Translate().visit(ast.parse(src)); ast.fix_missing_locations(mod_tree)
exec(compile(mod_tree, 'injected_pair', 'exec'), card.env)
orig = card.env['_nonlinear_pair_batch']
state = dict(delta=None, records={})
def wrapped(mu, C, K21, k4, k22_override=None, metric='identity'):
    l = card.call_layer
    k4_in = k4
    if state['delta'] is not None:
        W = card.w[l].to(torch.float64); X = W * W; D = state['delta']
        XD = X.T @ D                                  # n x n
        k4_hyb = k4_in + 3 * (XD * X.T).sum(1) - 2 * (X * X).T @ torch.diagonal(D)
        rec = dict(layer=l)
        for name, val in (("iso", k4_in), ("hyb", k4_hyb)):
            v = val.detach().numpy(); t = k4_true[l]
            rec[name] = dict(rel_err=float(np.sqrt(np.mean((v - t) ** 2) / np.mean(t ** 2))), corr=float(np.corrcoef(v, t)[0, 1]), mean_ratio=float(v.mean() / t.mean()))
        if planes is not None and f"K22_{l}" in planes.files:
            K22t = planes[f"K22_{l}"].astype(np.float64); vv = torch.diagonal(C)
            K22_iso = (state['radial'] * (torch.outer(vv, vv) + 2 * C * C) / 3).numpy()
            K22_hyb = K22_iso + (XD @ X).numpy()
            off = ~np.eye(1024, dtype=bool)
            for name, val in (("iso", K22_iso), ("hyb", K22_hyb)):
                rec[f"K22_{name}_off_rel_err"] = float(np.sqrt(np.mean((val[off] - K22t[off]) ** 2) / np.mean(K22t[off] ** 2)))
                rec[f"K22_{name}_off_corr"] = float(np.corrcoef(val[off], K22t[off])[0, 1])
        state['records'][l] = rec; print(json.dumps(rec), flush=True)
        if mode == 'apply':
            k4 = k4_hyb
            if k22_override is not None:
                k22_override = k22_override.clone(); k22_override = torch.diagonal_scatter(k22_override, k4_hyb)
    m, cov, post21, radial, aux = orig(mu, C, K21, k4, k22_override, metric)
    post22, cov22 = aux['post22'], aux['cov22']
    cd = torch.diagonal(cov22)
    iso = radial * (torch.outer(cd, cd) + 2 * cov22 * cov22) / 3
    state['delta'] = (post22 - iso).detach(); state['radial'] = float(radial)
    return m, cov, post21, radial, aux
card.env['_nonlinear_pair_batch'] = wrapped
t0 = time.time()
with torch.no_grad():
    pred = card.run(False)
mse = float(((pred[15].to(torch.float64) - truth[15]) ** 2).mean())
print(json.dumps(dict(stage="done", mode=mode, rank=rank, final_mse=mse, wall=round(time.time() - t0, 1))), flush=True)
json.dump(dict(records=state['records'], final_mse=mse, mode=mode), open(HERE / f"k4_hybrid_diag_r{rank}_{mode}.json", "w"), indent=1)
