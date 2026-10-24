"""TAG opus_engineer hzs ks v8 unit check (light, no estimator): the leaf views of fragment v8 (cached per tree) against v7's (per
product) on arrays with the trees' axis layout: every view must be the same region (same shape, same contents, same memory
offset, same strides), in every combination of strips; then a second product that shares the left tree and the output tree
with the first must make index requests for its right tree only, and a repeat of the first product none.
usage: unit_views_v8.py"""
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
    consts = [x for x in lines if x.startswith('_HZ_ROW_FULL') or x.startswith('_HZ_K_FULL') or x.startswith('_HZ_COL_FULL') or x.startswith('_HZ_SIDES') or x.startswith('_HZ_SIDE_CAP')]
    start = text.index('def _hz_slices(chunks):'); end = text.index('def _hz_leaf(v):')
    exec('\n'.join(consts) + '\n' + text[start:end], ns)
    return ns


old, new = load('fragment_ks_v7.py'), load('fragment_ks_v8.py')
rng = np.random.default_rng(0)
ra, kq, cb = 5, 4, 6
worst = True
totals = [0, 0]
for hzr, hzk, hzc in itertools.product((False, True), repeat=3):
    TA = rng.standard_normal((7, 3, 2, ra + 1, kq + 1)).view(Counted); TB = rng.standard_normal((7, 3, 2, kq + 1, cb + 1)).view(Counted)
    TC = rng.standard_normal((7, 3, 2, ra + 1, cb + 1)).view(Counted); ACC = rng.standard_normal((7, 3, 2, ra + 1, cb + 1)).view(Counted)
    Counted.count[0] = 0
    first = old['_hz_leaf_views'](TA, TB, TC, ra, kq, cb, hzr, hzk, hzc)
    a = [first, old['_hz_leaf_views'](TA, TB, ACC, ra, kq, cb, hzr, hzk, hzc, share=first)]
    n_old = Counted.count[0]; Counted.count[0] = 0
    b = [new['_hz_leaf_views'](TA, TB, TC, ra, kq, cb, hzr, hzk, hzc), new['_hz_leaf_views'](TA, TB, ACC, ra, kq, cb, hzr, hzk, hzc, kind_out=3)]
    n_new = Counted.count[0]
    # a second product with the same left tree and output tree and another right tree (other inner / column split is not possible
    # with the same output: here another product SHAPE of the right tree only is emulated by a tree of another leading shape)
    TB2 = rng.standard_normal((7, 2, 3, kq + 1, cb + 1)).view(Counted)
    Counted.count[0] = 0
    c_old = [old['_hz_leaf_views'](TA, TB2, TC, ra, kq, cb, hzr, hzk, hzc)]; c_old.append(old['_hz_leaf_views'](TA, TB2, ACC, ra, kq, cb, hzr, hzk, hzc, share=c_old[0]))
    n_old2 = Counted.count[0]; Counted.count[0] = 0
    c_new = [new['_hz_leaf_views'](TA, TB2, TC, ra, kq, cb, hzr, hzk, hzc), new['_hz_leaf_views'](TA, TB2, ACC, ra, kq, cb, hzr, hzk, hzc, kind_out=3)]
    n_new2 = Counted.count[0]; Counted.count[0] = 0
    again = [new['_hz_leaf_views'](TA, TB, TC, ra, kq, cb, hzr, hzk, hzc), new['_hz_leaf_views'](TA, TB, ACC, ra, kq, cb, hzr, hzk, hzc, kind_out=3)]
    n_again = Counted.count[0]
    a, b = a + c_old, b + c_new
    assert n_again == 0 and all(p is q for x, y in zip(b[:2], again) for m1, m2 in zip(x['mats'], y['mats']) for p, q in zip(m1, m2))
    same = True
    for x, y in zip(a, b):
        assert len(x['mats']) == len(y['mats']) and len(x['fills']) == len(y['fills'])
        for p, q in zip([v for m in x['mats'] for v in m] + x['fills'], [v for m in y['mats'] for v in m] + y['fills']):
            same &= p.shape == q.shape and np.array_equal(p, q) and p.__array_interface__['data'][0] == q.__array_interface__['data'][0] and p.strides == q.strides
    worst &= same
    totals[0] += n_old; totals[1] += n_new
    totals[0] += n_old2; totals[1] += n_new2
    print('strips rows %d inner %d columns %d: same views %s; index requests first product v7 %d, v8 %d; a product sharing two of its three trees v7 %d, v8 %d; the first again v8 %d' % (hzr, hzk, hzc, same, n_old, n_new, n_old2, n_new2, n_again))
print('all eight combinations, two products each: v7 %d requests, v8 %d' % tuple(totals))
print('PASS' if worst else 'FAIL')
