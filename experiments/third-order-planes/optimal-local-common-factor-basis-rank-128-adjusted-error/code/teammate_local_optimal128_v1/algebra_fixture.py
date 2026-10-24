"""Tiny pure-Python checks of roster, projector read, and scaling algebra."""
import json
import math
from pathlib import Path


def transpose(a):
    return [list(row) for row in zip(*a)]


def mm(a, b):
    bt = transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def maximum_error(a, b):
    return max(abs(x - y) for ar, br in zip(a, b) for x, y in zip(ar, br))


u = [[1., 2., -1.], [3., -2., .5], [.7, .8, .9]]
v = [[.3, .2, -.4], [.1, .5, .7], [-.2, .8, .6]]
lam_pairs = [.3, -.7, .2]
blocks = ((0, 2), (4, 1))
basis = [[2 ** -.5, 2 ** -.5], [2 ** -.5, -(2 ** -.5)], [0., 0.]]
projected = mm(transpose(basis), u)
approx_u = mm(basis, projected)
F = [[0.] * 6 for _ in range(3)]
lam = [0.] * 6
offset = 0
for start, half in blocks:
    for j in range(half):
        for i in range(3):
            F[i][start+j] = approx_u[i][offset+j] + v[i][offset+j]
            F[i][start+half+j] = approx_u[i][offset+j] - v[i][offset+j]
        lam[start+j], lam[start+half+j] = lam_pairs[offset+j], -lam_pairs[offset+j]
    offset += half
left_sq = [[.1, .7, .3, .8, .9, 1.3], [1., .2, .5, .7, .1, .3]]
weighted = [[x * w for x, w in zip(row, lam)] for row in left_sq]
su, dv = [[0.] * 3 for _ in range(2)], [[0.] * 3 for _ in range(2)]
offset = 0
for start, half in blocks:
    for row in range(2):
        for j in range(half):
            su[row][offset+j] = weighted[row][start+j] + weighted[row][start+half+j]
            dv[row][offset+j] = weighted[row][start+j] - weighted[row][start+half+j]
    offset += half
common, difference = mm(mm(su, transpose(projected)), transpose(basis)), mm(dv, transpose(v))
factored_read = [[x+y for x, y in zip(a, b)] for a, b in zip(common, difference)]
read_error = maximum_error(factored_read, mm(weighted, transpose(F)))
U = [[.8, .3, -.1], [-.2, .5, .6]]
direct_recon = mm(U, F)
pu, pv = mm(mm(U, basis), projected), mm(U, v)
factored_recon = [[0.] * 6 for _ in range(2)]
offset = 0
for start, half in blocks:
    for row in range(2):
        for j in range(half):
            factored_recon[row][start+j] = pu[row][offset+j] + pv[row][offset+j]
            factored_recon[row][start+half+j] = pu[row][offset+j] - pv[row][offset+j]
    offset += half
recon_error = maximum_error(factored_recon, direct_recon)


def gram(uu, vv, ll):
    weights = [ll[j]**2 * sum(row[j]**2 for row in uu) * sum(row[j]**2 for row in vv) for j in range(3)]
    return [[sum(weights[j] * uu[a][j] * uu[b][j] for j in range(3)) for b in range(3)] for a in range(3)]


q = [2., .5, -3.]
scaled_u = [[x*q[j] for j, x in enumerate(row)] for row in u]
scaled_v = [[x*q[j] for j, x in enumerate(row)] for row in v]
scale_error = maximum_error(gram(u, v, lam_pairs), gram(scaled_u, scaled_v, [x/q[j]**3 for j, x in enumerate(lam_pairs)]))
assert read_error < 1e-12 and recon_error < 1e-12 and scale_error < 1e-11
result = dict(passed=True, read_max_abs=read_error, reconstruction_max_abs=recon_error,
              rescaling_gram_max_abs=scale_error, runtime='pure Python; no NumPy or estimator execution')
(Path(__file__).resolve().parent / 'algebra_fixture_result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
