"""Learning curve of the extended+gate on-policy fit (fold0 only): fit on the first 18/36/54/72 of fold0's TRAIN72 seeds
(same 705-column design, ridge .1 / damping .5, same custody + loader as fit_extended_gate_v1), score each on fold0's held 18.
Decides whether more training captures (protected held25 / fresh) could pay before the deadline.
usage: lc_gate_fold0_v1.py OUT_DIR   (linux/VM only)
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fg', HERE / 'fit_extended_gate_v1.py'); fg = importlib.util.module_from_spec(spec); spec.loader.exec_module(fg)
fs, cf, W1 = fg.fs, fg.cf, fg.W1
read, write, utc, sha = fs.read, fs.write, fs.utc, fs.sha

def main():
    assert sys.platform == 'linux'
    out = Path(sys.argv[1]); out.mkdir(exist_ok=False)
    recipe = read(HERE / 'SECOND_FIT_RECIPE_v1.json')
    pre, design, all_info, proof = fs.custody(recipe)
    pp = fs.D / 'PLAN_CAPTURE_v1.json'; assert sha(pp) == 'a6e841b1c5a09613e1d580f8eb49f68889eaddd1bed6f2ae73e8208f8d3e5375'; plan = read(pp)
    key = 'fold0'; info = all_info[key]; selected = info['family']['train']; held = info['family']['validation']; P = info['P']
    assert len(selected) == 72 and len(held) == 18
    cuts = (18, 36, 54, 72); nb = len(cuts)
    Gb = np.zeros((nb, 15, W1, W1)); bb = np.zeros((nb, 15, W1)); loaded = []
    for r in fg.extend(fg.data_rows(selected, info, design, plan, 'TRAIN')):
        i = selected.index(r['seed']); blk = i * nb // 72; l = r['layer'] - 1
        Gb[blk, l] += r['X2'].T @ r['X2']; bb[blk, l] += r['X2'].T @ (.5 * (r['yA'] + r['yB']))
        if r['layer'] == 15: loaded.append(r['seed'])
    assert loaded == list(selected)
    models = {}
    for k, cut in enumerate(cuts):
        G = Gb[:k + 1].sum(0); b = bb[:k + 1].sum(0)
        coef = fg.coefficients(G, b, cut, P); models[cut] = fg.deploy(P, coef)
        write(out / ('seal_n%d_v1.json' % cut), dict(utc=utc(), family=key, train=selected[:cut], excluded_validation=held, coef_by_layer=coef.tolist(), width=W1))
        print(json.dumps(dict(utc=utc(), fitted=cut)), flush=True)
    seal = out / 'seal_n72_v1.json'
    cache = []
    for r in fg.data_rows(held, info, design, plan, 'HELD', seal):
        r.pop('X'); cache.append(r)   # evaluate() uses the raw features only
    results = {}
    for cut in cuts:
        rows, trust = fg.evaluate(cache, P, models[cut], held)
        s = cf.summary(rows); results[str(cut)] = dict(summary=s, physical=trust, per_network=rows)
        print(json.dumps(dict(utc=utc(), n_train=cut, held18=s, physical_rms=trust['correction_to_parent_head_rms'], max_layer_ratio=trust['max_layer_ratio'])), flush=True)
    write(out / 'LEARNING_CURVE_v1.json', dict(utc=utc(), fold=key, cuts=list(cuts), held=held, train_order=selected, results=results))
    print('LC_DONE', flush=True)

if __name__ == '__main__': main()
