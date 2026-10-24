"""Second pricing round of readout feature blocks on fold0 (fit TRAIN72, score held 18), on top of the 705-col gate design + hg:
fh = first128 x hidden2, h3 = hidden2^3, f3 = first128^3, hs = hidden2 x sigma, fs = first128 x sigma. Same solver/custody as lc_blocks_fold0_v1.
usage: lc_blocks2_fold0_v1.py OUT_DIR   (linux/VM only)
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('lb', HERE / 'lc_blocks_fold0_v1.py'); lb = importlib.util.module_from_spec(spec); spec.loader.exec_module(lb)
fg, fs, cf, W0, W1 = lb.fg, lb.fs, lb.cf, lb.W0, lb.W1
read, write, utc, sha = fs.read, fs.write, fs.utc, fs.sha
BLOCKS = ('hg', 'fh', 'h3', 'f3', 'hs', 'fs'); WF = W1 + 128 * len(BLOCKS)
DESIGNS = {'base705': (), 'hg': ('hg',), 'hg+fh': ('hg', 'fh'), 'hg+h3+f3': ('hg', 'h3', 'f3'), 'hg+hs+fs': ('hg', 'hs', 'fs'), 'hg+fh+h3+f3': ('hg', 'fh', 'h3', 'f3'), 'all': BLOCKS}

def full_features(r):
    m = r['mask'][:, None]; g = r['gate'][:, None]; s = r['sigma'][:, None]; h = r['hidden']; f = r['first']
    return np.column_stack([r['X2'], (h * g) * m, (f * h) * m, (h ** 3) * m, (f ** 3) * m, (h * s) * m, (f * s) * m])

def cols(design):
    idx = list(range(W1))
    for j, b in enumerate(BLOCKS):
        if b in design: idx += list(range(W1 + 128 * j, W1 + 128 * (j + 1)))
    return np.array(idx)

def main():
    assert sys.platform == 'linux'
    out = Path(sys.argv[1]); out.mkdir(exist_ok=False)
    recipe = read(HERE / 'SECOND_FIT_RECIPE_v1.json'); pre, design, all_info, proof = fs.custody(recipe)
    pp = fs.D / 'PLAN_CAPTURE_v1.json'; assert sha(pp) == 'a6e841b1c5a09613e1d580f8eb49f68889eaddd1bed6f2ae73e8208f8d3e5375'; plan = read(pp)
    key = 'fold0'; info = all_info[key]; selected = info['family']['train']; held = info['family']['validation']; P = info['P']
    U = cf.nullspace(P['enc4_w']); n = len(selected) * fs.N
    G = np.zeros((15, WF, WF)); b = np.zeros((15, WF)); loaded = []
    for r in fg.extend(fg.data_rows(selected, info, design, plan, 'TRAIN')):
        F = full_features(r); l = r['layer'] - 1; G[l] += F.T @ F; b[l] += F.T @ (.5 * (r['yA'] + r['yB']))
        if r['layer'] == 15: loaded.append(r['seed'])
    assert loaded == list(selected)
    seal = out / 'seal_dummy_v1.json'; write(seal, dict(family=key, excluded_validation=held))
    cache = [dict(seed=r['seed'], layer=r['layer'], F=full_features(r), yA=r['yA'], yB=r['yB'], mask=r['mask'], physical=r['physical'], saved=r['saved']) for r in fg.extend(fg.data_rows(held, info, design, plan, 'HELD', seal))]
    results = {}
    for name, dsg in DESIGNS.items():
        ix = cols(dsg); coef = np.stack([lb.one_layer_k(G[l][np.ix_(ix, ix)], b[l][ix], n, P['head'][l + 1], P['head_b'][l + 1], P['hidden_head'][l + 1], P['first_hidden_head'][l + 1], U) for l in range(15)])
        per = {}; num = np.zeros(15); den = np.zeros(15)
        for r in cache:
            l = r['layer'] - 1; corr = (r['F'][:, ix] @ coef[l]) * r['mask']
            a = per.setdefault(r['seed'], [0.0, 0.0]); a[0] += float(r['yA'] @ r['yB']); a[1] += float((r['yA'] - corr) @ (r['yB'] - corr))
            ch = corr * r['physical']; num[l] += float(ch @ ch); sv = r['saved'] * r['mask']; den[l] += float(sv @ sv)
        pcts = [100 * (v[1] / v[0] - 1) for v in per.values()]
        results[name] = dict(ncols=int(len(ix)), held18_mean_pct=float(np.mean(pcts)), lower=int(sum(p < 0 for p in pcts)), physical_rms=float(np.sqrt(num.sum() / den.sum())), max_layer_ratio=float(np.sqrt(num / den).max()))
        print(json.dumps(dict(utc=utc(), design=name, **results[name])), flush=True)
    write(out / 'BLOCKS2_FOLD0_v1.json', dict(utc=utc(), fold=key, results=results, blocks=BLOCKS, designs={k: list(v) for k, v in DESIGNS.items()}))
    print('BLOCKS2_DONE', flush=True)

if __name__ == '__main__': main()
