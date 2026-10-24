"""Price new readout feature blocks on the proxy (fold0: fit on TRAIN72, score on held 18) before any runtime work.
Blocks on top of the 705-col gate design: pv = previous layer's hidden2 (zeros at layer 1), hg = hidden2 x gate,
pvg = prev hidden2 x gate. Same custody/loader/solver as fit_extended_gate_v1 (ridge .1, damping .5), f8 scoring.
usage: lc_blocks_fold0_v1.py OUT_DIR   (linux/VM only)
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fg', HERE / 'fit_extended_gate_v1.py'); fg = importlib.util.module_from_spec(spec); spec.loader.exec_module(fg)
fs, cf, W0, W1 = fg.fs, fg.cf, fg.W0, fg.W1
read, write, utc, sha = fs.read, fs.write, fs.utc, fs.sha
BLOCKS = ('pv', 'hg', 'pvg'); WF = W1 + 128 * len(BLOCKS)
DESIGNS = {'base705': (), 'pv': ('pv',), 'hg': ('hg',), 'pv+hg': ('pv', 'hg'), 'pv+hg+pvg': ('pv', 'hg', 'pvg')}

def with_prev(rows):
    prev = None
    for r in rows:
        if r['layer'] == 1: prev = np.zeros_like(r['hidden'])
        r['prev'] = prev; yield r; prev = r['hidden']

def full_features(r):
    m = r['mask'][:, None]; g = r['gate'][:, None]
    return np.column_stack([r['X2'], r['prev'] * m, (r['hidden'] * g) * m, (r['prev'] * g) * m])

def cols(design):
    idx = list(range(W1))
    for j, b in enumerate(BLOCKS):
        if b in design: idx += list(range(W1 + 128 * j, W1 + 128 * (j + 1)))
    return np.array(idx)

def one_layer_k(G, b, n, head, bias, hidden, first, U):
    W = G.shape[0]; nnew = W - W0
    base = np.zeros(W); base[:W0] = fs.one_layer(G[:W0, :W0], b[:W0], n, head, bias, hidden, first, U)
    Q = np.zeros((W, 257)); Q[:65, :65] = np.eye(65); Q[65:193, 65:129] = U; Q[193:W0, 129:] = np.eye(128)
    T = np.zeros((W, nnew)); T[W0:] = np.eye(nnew)
    gg, cross = Q.T @ G @ Q, Q.T @ G @ T
    scale = np.maximum(np.sqrt(np.maximum(np.diag(gg), 0) / n), 1e-8)
    A = np.linalg.solve(gg / (scale[:, None] * scale[None, :]) + 1e-10 * n * np.eye(257), cross / scale[:, None]) / scale[:, None]
    R = T - Q @ A
    gres = R.T @ G @ R; gres = .5 * (gres + gres.T)
    k = cf.solve(gres, R.T @ (b - G @ base), n)
    c = base + R @ k
    c[:W0] = fs.frozen(c[:W0], head, bias, hidden, first)
    assert np.isfinite(c).all()
    return c

def main():
    assert sys.platform == 'linux'
    out = Path(sys.argv[1]); out.mkdir(exist_ok=False)
    recipe = read(HERE / 'SECOND_FIT_RECIPE_v1.json'); pre, design, all_info, proof = fs.custody(recipe)
    pp = fs.D / 'PLAN_CAPTURE_v1.json'; assert sha(pp) == 'a6e841b1c5a09613e1d580f8eb49f68889eaddd1bed6f2ae73e8208f8d3e5375'; plan = read(pp)
    key = 'fold0'; info = all_info[key]; selected = info['family']['train']; held = info['family']['validation']; P = info['P']
    U = cf.nullspace(P['enc4_w']); n = len(selected) * fs.N
    G = np.zeros((15, WF, WF)); b = np.zeros((15, WF)); loaded = []
    for r in with_prev(fg.extend(fg.data_rows(selected, info, design, plan, 'TRAIN'))):
        F = full_features(r); l = r['layer'] - 1; G[l] += F.T @ F; b[l] += F.T @ (.5 * (r['yA'] + r['yB']))
        if r['layer'] == 15: loaded.append(r['seed'])
    assert loaded == list(selected)
    seal = out / 'seal_dummy_v1.json'; write(seal, dict(family=key, excluded_validation=held))
    cache = []
    for r in with_prev(fg.extend(fg.data_rows(held, info, design, plan, 'HELD', seal))):
        cache.append(dict(seed=r['seed'], layer=r['layer'], F=full_features(r), yA=r['yA'], yB=r['yB'], mask=r['mask'], physical=r['physical'], saved=r['saved']))
    results = {}
    for name, dsg in DESIGNS.items():
        ix = cols(dsg); coef = np.stack([one_layer_k(G[l][np.ix_(ix, ix)], b[l][ix], n, P['head'][l + 1], P['head_b'][l + 1], P['hidden_head'][l + 1], P['first_hidden_head'][l + 1], U) for l in range(15)])
        per = {}; num = np.zeros(15); den = np.zeros(15)
        for r in cache:
            l = r['layer'] - 1; corr = (r['F'][:, ix] @ coef[l]) * r['mask']
            o = float(r['yA'] @ r['yB']); nw = float((r['yA'] - corr) @ (r['yB'] - corr))
            a = per.setdefault(r['seed'], [0.0, 0.0]); a[0] += o; a[1] += nw
            ch = corr * r['physical']; num[l] += float(ch @ ch); sv = r['saved'] * r['mask']; den[l] += float(sv @ sv)
        pcts = [100 * (v[1] / v[0] - 1) for v in per.values()]
        results[name] = dict(ncols=int(len(ix)), held18_mean_pct=float(np.mean(pcts)), lower=int(sum(p < 0 for p in pcts)), pooled_pct=float(100 * (sum(v[1] for v in per.values()) / sum(v[0] for v in per.values()) - 1)),
                             physical_rms=float(np.sqrt(num.sum() / den.sum())), max_layer_ratio=float(np.sqrt(num / den).max()))
        print(json.dumps(dict(utc=utc(), design=name, **results[name])), flush=True)
    write(out / 'BLOCKS_FOLD0_v1.json', dict(utc=utc(), fold=key, results=results, blocks=BLOCKS, designs={k: list(v) for k, v in DESIGNS.items()}))
    print('BLOCKS_DONE', flush=True)

if __name__ == '__main__': main()
