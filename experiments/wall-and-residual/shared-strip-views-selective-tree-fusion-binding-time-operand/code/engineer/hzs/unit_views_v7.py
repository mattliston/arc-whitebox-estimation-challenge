"""TAG opus_engineer hzs ks v7 unit check (light, no estimator): the leaf views of fragment v7 (one index expression per view, the
operand views shared between the two output trees) against v6's (chained indexing) on arrays with the trees' axis layout: every
view must be the same region (same shape, same contents, same memory offset), in every combination of strips; and the number of
index requests each version makes.
usage: unit_views_v7.py"""
from pathlib import Path
import itertools
import numpy as np
HERE = Path(__file__).resolve().parent


class Counted(np.ndarray):
    count = [0]

    def __getitem__(self, key):
        Counted.count[0] += 1
        return np.ndarray.__getitem__(self, key)


def load(name):
    text = (HERE/name).read_text()
    lines = text.split('\n')
    ns = {}
    consts = [x for x in lines if x.startswith('_HZ_ROW_FULL') or x.startswith('_HZ_K_FULL') or x.startswith('_HZ_COL_FULL')]
    start = text.index('def _hz_slices(chunks):'); end = text.index('def _hz_leaf(v):')
    exec('\n'.join(consts) + '\n' + text[start:end], ns)
    return ns


old, new = load('fragment_ks_v6.py'), load('fragment_ks_v7.py')
rng = np.random.default_rng(0)
ra, kq, cb = 5, 4, 6
worst = True
totals = [0, 0]
for hzr, hzk, hzc in itertools.product((False, True), repeat=3):
    TA = rng.standard_normal((7, 3, 2, ra + 1, kq + 1)).view(Counted); TB = rng.standard_normal((7, 3, 2, kq + 1, cb + 1)).view(Counted)
    TC = rng.standard_normal((7, 3, 2, ra + 1, cb + 1)).view(Counted); ACC = rng.standard_normal((7, 3, 2, ra + 1, cb + 1)).view(Counted)
    Counted.count[0] = 0
    a = [old['_hz_leaf_views'](TA, TB, T, ra, kq, cb, hzr, hzk, hzc) for T in (TC, ACC)]
    n_old = Counted.count[0]; Counted.count[0] = 0
    first = new['_hz_leaf_views'](TA, TB, TC, ra, kq, cb, hzr, hzk, hzc)
    b = [first, new['_hz_leaf_views'](TA, TB, ACC, ra, kq, cb, hzr, hzk, hzc, share=first)]
    n_new = Counted.count[0]
    same = True
    for x, y in zip(a, b):
        assert len(x['mats']) == len(y['mats']) and len(x['fills']) == len(y['fills'])
        for p, q in zip([v for m in x['mats'] for v in m] + x['fills'], [v for m in y['mats'] for v in m] + y['fills']):
            same &= p.shape == q.shape and np.array_equal(p, q) and p.__array_interface__['data'][0] == q.__array_interface__['data'][0] and p.strides == q.strides
    worst &= same
    totals[0] += n_old; totals[1] += n_new
    print('strips rows %d inner %d columns %d: same views %s; index requests v6 %d, v7 %d' % (hzr, hzk, hzc, same, n_old, n_new))
print('all eight combinations: v6 %d requests, v7 %d' % tuple(totals))
print('PASS' if worst else 'FAIL')
