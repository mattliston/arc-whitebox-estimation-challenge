"""TRAIN90-only fit of the gate readout design plus new feature blocks (pv = previous layer's hidden2, hg = hidden2 x gate,
pvg = prev hidden2 x gate). No folds (fold0 held-18 pricing lives in lc_blocks_fold0_v1); same custody/loader/solver
(ridge/damping via ARC_RIDGE/ARC_DAMP). Writes full90_BLOCKS_COEFFICIENTS_v1.json {design, coef_by_layer (15 x W)}.
usage: ARC_BLOCKS=pv,hg [ARC_RIDGE=0.1 ARC_DAMP=0.5] fit_blocks_full90_v1.py OUT_DIR   (linux/VM only)
"""
import importlib.util, json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('lb', HERE / ('lc_blocks2_fold0_v1.py' if os.environ.get('ARC_BLOCKSET', '1') == '2' else 'lc_blocks_fold0_v1.py')); lb = importlib.util.module_from_spec(spec); spec.loader.exec_module(lb)
fg, fs, cf, W0, W1 = lb.fg, lb.fs, lb.cf, lb.W0, lb.W1
cf.RIDGE = float(os.environ.get('ARC_RIDGE', '0.1')); cf.DAMPING = float(os.environ.get('ARC_DAMP', '0.5'))
ONE_LAYER = getattr(lb, 'one_layer_k', None) or lb.lb.one_layer_k
read, write, utc, sha = fs.read, fs.write, fs.utc, fs.sha

def main():
    assert sys.platform == 'linux'
    design = tuple(b for b in os.environ.get('ARC_BLOCKS', 'pv,hg').split(',') if b); assert all(b in lb.BLOCKS for b in design)
    out = Path(sys.argv[1]); out.mkdir(exist_ok=False)
    recipe = read(HERE / 'SECOND_FIT_RECIPE_v1.json'); pre, dsg, all_info, proof = fs.custody(recipe)
    pp = fs.D / 'PLAN_CAPTURE_v1.json'; assert sha(pp) == 'a6e841b1c5a09613e1d580f8eb49f68889eaddd1bed6f2ae73e8208f8d3e5375'; plan = read(pp)
    info = all_info['full90']; selected = info['family']['train']; P = info['P']; assert len(selected) == 90
    ix = lb.cols(design); W = len(ix); U = cf.nullspace(P['enc4_w']); n = len(selected) * fs.N
    G = np.zeros((15, W, W)); b = np.zeros((15, W)); loaded = []; num = np.zeros(15); den = np.zeros(15)
    rows_cache = []
    for r in (getattr(lb, "with_prev", lambda x: x))(fg.extend(fg.data_rows(selected, info, dsg, plan, 'TRAIN'))):
        F = lb.full_features(r)[:, ix]; l = r['layer'] - 1; G[l] += F.T @ F; b[l] += F.T @ (.5 * (r['yA'] + r['yB']))
        rows_cache.append(dict(seed=r['seed'], layer=r['layer'], F=F, yA=r['yA'], yB=r['yB'], mask=r['mask'], physical=r['physical'], saved=r['saved']))
        if r['layer'] == 15: loaded.append(r['seed'])
    assert loaded == list(selected)
    coef = np.stack([ONE_LAYER(G[l], b[l], n, P['head'][l + 1], P['head_b'][l + 1], P['hidden_head'][l + 1], P['first_hidden_head'][l + 1], U) for l in range(15)])
    per = {}
    for r in rows_cache:
        l = r['layer'] - 1; corr = (r['F'] @ coef[l]) * r['mask']
        a = per.setdefault(r['seed'], [0.0, 0.0]); a[0] += float(r['yA'] @ r['yB']); a[1] += float((r['yA'] - corr) @ (r['yB'] - corr))
        ch = corr * r['physical']; num[l] += float(ch @ ch); sv = r['saved'] * r['mask']; den[l] += float(sv @ sv)
    pcts = [100 * (v[1] / v[0] - 1) for v in per.values()]; ratios = np.sqrt(num / den)
    trust = dict(correction_to_parent_head_rms=float(np.sqrt(num.sum() / den.sum())), per_layer_ratios=ratios.tolist(), max_layer_ratio=float(ratios.max()), passed=bool(np.sqrt(num.sum() / den.sum()) <= .25 and ratios.max() <= 1))
    seal = out / 'full90_BLOCKS_COEFFICIENTS_v1.json'
    write(seal, dict(utc=utc(), family='full90', train=selected, design=list(design), block_order=[b for b in lb.BLOCKS if b in design], blockset=os.environ.get('ARC_BLOCKSET', '1'), columns=ix.tolist(), width=W, ridge=cf.RIDGE, damping=cf.DAMPING,
                     coef_by_layer=coef.tolist(), training=dict(mean_pct=float(np.mean(pcts)), lower=int(sum(p < 0 for p in pcts)), n=len(pcts)), physical=trust,
                     source_sha256=info['family']['science']['sha256']))
    print(json.dumps(dict(utc=utc(), design=design, width=W, training_mean_pct=float(np.mean(pcts)), lower=int(sum(p < 0 for p in pcts)), physical=trust, seal=str(seal))), flush=True)
    print('FIT_DONE', flush=True)

if __name__ == '__main__': main()
