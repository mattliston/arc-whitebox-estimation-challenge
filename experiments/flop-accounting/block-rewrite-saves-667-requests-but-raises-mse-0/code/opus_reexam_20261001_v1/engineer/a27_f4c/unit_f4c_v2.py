"""TAG opus_engineer a27 F4c v2 unit check (light, small arrays, single thread, no estimator): fragment v2 (the step's cohort loop
as one block of operations) against the physicist's plain-array reference O/physicist/f4c_native_ref.py (numpy float64) and
against fragment v1 on the same synthetic data.
  birth: the table (A, B, C) x (+ columns, - columns) per age against exact_constants + F4's rule written in the same basis
         (own entry o, Delta = o - partner), for two cohorts (born at layers 0 and 1; the first has alpha = 0 as on the card).
         Cohort 1 holds three degenerate pairs (gap 0, a common leg of norm 0, a nearly singular system).
  step:  step 1 (one cohort, age 1) and step 2 (two cohorts, ages 2 and 1): shift, beta and the plane diagonal against the
         reference and against v1; requests (logged operations + views, as the platform counts round trips) and bill of both.
usage: unit_f4c_v2.py [<fragment file>] [<theta age 1> <theta age 2> <c1 age 1> <c2 age 1> <c1 age 2> <c2 age 2>]"""
from pathlib import Path
import importlib.util
import math
import sys
import numpy as np
import flopscope as flops
from flopscope import _ndarray
fnp = flops.numpy
HERE = Path(__file__).resolve().parent
O = HERE.parents[1]
spec = importlib.util.spec_from_file_location('f4c_native_ref', O/'physicist/f4c_native_ref.py'); ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)

VIEWS = [0]
_getitem = _ndarray.FlopscopeArray.__getitem__


def counted_getitem(self, key):
    VIEWS[0] += 1
    return _getitem(self, key)


_ndarray.FlopscopeArray.__getitem__ = counted_getitem


class _E2Hold(list):
    def __call__(self, x):
        self.append(x)
        return x


def requests(ctx, start):
    """Logged operations + views since `start` (an advanced index is both a logged operation and a counted __getitem__: once)."""
    ops = len(ctx._op_log) - start[0]
    gathers = sum(1 for x in list(ctx._op_log)[start[0]:] if 'getitem' in x.op_name)
    return ops + (VIEWS[0] - start[1]) - gathers


args = sys.argv[1:]
fragment = args.pop(0) if args and args[0].endswith('.py') else 'fragment_f4c_v2.py'
S19 = (1.0032287195131746, 0.9932375877606482)
spaces = {}
for name in (fragment, 'fragment_f4c_v1.py'):
    ns = dict(fnp=fnp, _E2Hold=_E2Hold, _S19_LAM=S19)
    exec(compile((HERE/name).read_text(), name, 'exec'), ns)
    if args:
        v6 = [float(x) for x in args]
        ns['_F4C_THETA'] = (v6[0], v6[1]); ns['_F4C_C'] = ((v6[2], v6[3]), (v6[4], v6[5]))
    spaces[name] = ns
new, old = spaces[fragment], spaces['fragment_f4c_v1.py']
theta = new['_F4C_THETA'] if isinstance(new['_F4C_THETA'], tuple) else (new['_F4C_THETA'],) * 2
strength = new['_F4C_C'] if isinstance(new['_F4C_C'][0], tuple) else (new['_F4C_C'],) * 2
rng = np.random.default_rng(11)
n, nl = 512, 448
rel = lambda x, y: float(np.linalg.norm(np.asarray(x, dtype=np.float64) - y) / max(np.linalg.norm(y), 1e-300))
births, state, start = {}, dict(columns=0, cohorts=[]), 0
worst = 0.0
birth_cost = []
for l, K in ((0, 64), (1, 96)):
    alpha = rng.standard_normal(n) * (0.0 if l == 0 else 1.2)
    sigma = rng.uniform(0.6, 1.8, n)
    gate = np.array([0.5 * math.erfc(-x / math.sqrt(2)) for x in alpha]); pdf = np.exp(-alpha ** 2 / 2) / math.sqrt(2 * math.pi)
    d = pdf / sigma
    host = sigma * (alpha * gate + pdf) * (1 + 0.03 * rng.standard_normal(n))
    t = rng.uniform(0.4, 3.0, n) * sigma ** 2; s = -rng.uniform(0.4, 3.0, n) * sigma ** 2
    sel = rng.permutation(n)[:K]
    base = (rng.standard_normal((n, K)) * rng.uniform(0.02, 0.4, K)).astype(np.float32)
    base[sel, np.arange(K)] = 0.0                                   # the hub's own entry of the common leg
    if l == 1:
        s[sel[3]] = t[sel[3]]                                       # gap 0
        base[:, 7] = 0.0                                            # a leg of norm 0
        base[:, 11] *= 1e7                                          # |s| far above the gap: det / (G00 G11) below 1e-8
    weight = rng.uniform(0.5, 2.0, K).astype(np.float32)
    radial = 0.0 if l == 0 else 0.04
    aux = dict(alpha=fnp.asarray(alpha), sigma=fnp.asarray(sigma), gate=fnp.asarray(gate), pdf=fnp.asarray(pdf))
    for name, ns in spaces.items():
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            mark = (len(ctx._op_log), VIEWS[0])
            ns['_f4c_birth'](l, aux, fnp.asarray(d), fnp.asarray(host), fnp.asarray(sel), fnp.asarray(t), fnp.asarray(s), fnp.asarray(base), fnp.asarray(weight), radial)
            birth_cost.append((name, l, int(ctx.summary_dict()['flops_used']), requests(ctx, mark) - 9, K))   # 9 = the asarray calls of this line
    al, sg, gt, ph, m, tp, tm = alpha[sel], sigma[sel], gate[sel], pdf[sel], host[sel], t[sel], s[sel]
    S1 = -al * ph / (sg * sg); S2 = 2.0 * (gt - gt * gt - m * ph / sg)
    s2 = np.maximum((base.astype(np.float64) ** 2).sum(0), 1e-30)
    A = -al / (sg * (1 + radial / 3))
    per_age = []
    for i in (0, 1):                                                # age i + 1
        n_p, n_m = ref.exact_constants(np, tp, tm, S1, S2, s2, strength[i][0], strength[i][1])
        masked = np.flatnonzero(np.abs(n_p).sum(0) + np.abs(n_m).sum(0) == 0)
        lam_out = weight.astype(np.float64) * S19[1] * S19[0] ** (i + 1)
        ap, bp, cp = theta[i] * A + n_p[0] / lam_out, n_p[1] / lam_out, n_p[2] / lam_out
        am, bm, cm = theta[i] * A - n_m[0] / lam_out, -2 * theta[i] * A - n_m[1] / lam_out, theta[i] * A - n_m[2] / lam_out
        want = np.stack([np.concatenate([ap, am]), np.concatenate([bp, -2 * am - bm]), np.concatenate([cp, am + bm + cm])])
        got = new['_F4C_COEF'][l][i]
        if isinstance(got, (list, tuple)):                          # v1's six vectors (fragment v3 keeps them): compared in their own basis (p^2, p D, D^2)
            got = np.stack([np.concatenate([np.asarray(got[q], dtype=np.float64), np.asarray(got[q + 3], dtype=np.float64)]) for q in range(3)])
            want = np.stack([np.concatenate([ap, am]), np.concatenate([bp, bm]), np.concatenate([cp, cm])])
        got = np.asarray(got, dtype=np.float64)
        assert got.shape == (3, 2 * K), got.shape
        errors = [rel(got[q], want[q]) for q in range(3)]
        worst = max(worst, max(errors))
        on_masked = float(np.abs(got[:, np.concatenate([masked, masked + K])] - want[:, np.concatenate([masked, masked + K])]).max()) if len(masked) else 0.0
        six = np.stack([np.asarray(x, dtype=np.float64) for x in old['_F4C_COEF'][l][i]])
        if isinstance(new['_F4C_COEF'][l][i], (list, tuple)):
            v1_table = np.stack([np.concatenate([six[q], six[q + 3]]) for q in range(3)])
        else:
            v1_table = np.stack([np.concatenate([six[0], six[3]]), np.concatenate([six[1], -2 * six[3] - six[4]]), np.concatenate([six[2], six[3] + six[4] + six[5]])])
        print(f'birth layer {l}, age {i + 1}: table rows (A, B, C) against the reference, relative difference {" ".join("%.1e" % e for e in errors)}; '
              f'against v1\'s six vectors in the same basis {rel(got, v1_table):.1e}; pairs without an exact part in the reference: {list(masked)} (largest absolute difference there {on_masked:.1e})')
        per_age.append((n_p, n_m))
    births[l] = dict(A=A, n=per_age, K=K, weight=weight)
    state['cohorts'].append((start, 2 * K)); start += 2 * K
ok = worst < 5e-6
for l in (1, 2):
    # step l: the cohort born at j is l - j layers old; lam = the weights carried into the step
    young = [j for j in range(max(l - 2, 0), l)]
    nc = sum(2 * births[j]['K'] for j in young)
    lam = np.concatenate([np.concatenate([x, -x]) for x in [births[j]['weight'] * np.float32(S19[1]) * np.float32(S19[0]) ** (l - j - 1) for j in young]]).astype(np.float32)
    alpha = rng.standard_normal(n) * 1.2; sigma = rng.uniform(0.6, 1.8, n)
    gate = np.array([0.5 * math.erfc(-x / math.sqrt(2)) for x in alpha]); d = np.exp(-alpha ** 2 / 2) / math.sqrt(2 * math.pi) / sigma
    live = np.sort(rng.permutation(n)[:nl]); alive = np.zeros(n); alive[live] = 1
    C = rng.standard_normal((n, n)); C = C @ C.T / n + np.eye(n); v = np.diag(C).copy()
    right = (rng.standard_normal((nl, nc)) * 0.4).astype(np.float32)
    for j in young:                                                 # the pair's two columns differ by a multiple of the hub's edge: D is small against p
        k0, K = state['cohorts'][j][0], births[j]['K']
        right[:, k0 + K:k0 + 2 * K] = right[:, k0:k0 + K] + (rng.standard_normal((nl, K)) * 0.03).astype(np.float32)
    pp = np.zeros((n, nc), dtype=np.float32); pp[live] = right
    u = (alive * gate).astype(np.float32)[:, None] * pp             # the outgoing gated columns, as the card builds them
    outs = {}
    for name, ns in spaces.items():
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            operands = (fnp.asarray(right), fnp.asarray(gate), fnp.asarray(d), fnp.asarray(live), fnp.asarray(v), fnp.asarray(lam))
            mark = (len(ctx._op_log), VIEWS[0]); used = int(ctx.summary_dict()['flops_used'])
            out = ns['_f4c_step'](l, operands[0], operands[1], operands[2], operands[3], operands[4], operands[5], state)
            outs[name] = (out, int(ctx.summary_dict()['flops_used']) - used, requests(ctx, mark))
    (shift, beta, diagonal), used, reqs = outs[fragment]
    (shift1, beta1, diagonal1), used1, reqs1 = outs['fragment_f4c_v1.py']
    lam_out = lam.astype(np.float64) * S19[0]
    shift_ref = np.zeros((n, nc)); beta_ref = np.zeros(n); diag_ref = np.zeros(n)
    for j in young:
        k0, width = state['cohorts'][j]; K = width // 2; P, Mi = slice(k0, k0 + K), slice(k0 + K, k0 + 2 * K)
        i = l - j - 1
        n_p, n_m = births[j]['n'][i]
        new_p, new_q, be, dg = ref.f4c_step(np, pp[:, P], pp[:, Mi], u[:, P], u[:, Mi], lam_out[P], lam_out[Mi], births[j]['A'], n_p, n_m, theta[i], alive, d, v)
        shift_ref[:, P] = new_p.astype(np.float64) - u[:, P]; shift_ref[:, Mi] = new_q.astype(np.float64) - u[:, Mi]
        beta_ref += be; diag_ref += dg
    e = (rel(shift, shift_ref[live]), rel(beta, beta_ref), rel(diagonal, diag_ref))
    e1 = (rel(shift1, shift_ref[live]), rel(beta1, beta_ref), rel(diagonal1, diag_ref))
    between = (rel(shift, np.asarray(shift1, dtype=np.float64)), rel(beta, np.asarray(beta1)), rel(diagonal, np.asarray(diagonal1)))
    same_bits = all(bool((np.asarray(x) == np.asarray(y)).all()) and str(x.dtype) == str(y.dtype) for x, y in ((shift, shift1), (beta, beta1), (diagonal, diagonal1)))
    print(f'step {l} ({len(young)} cohort{"s" if len(young) > 1 else ""}, {nc} shifted columns, live rows {nl}):')
    print(f'   against the reference (which rounds u + shift to float32): shift {e[0]:.2e}, beta {e[1]:.2e}, plane diagonal {e[2]:.2e}   (v1: {e1[0]:.2e}, {e1[1]:.2e}, {e1[2]:.2e})')
    print(f'   against v1: shift {between[0]:.2e}, beta {between[1]:.2e}, plane diagonal {between[2]:.2e}' + (' (all three bit-identical, same dtypes)' if same_bits else ''))
    print(f'   requests {reqs} (v1 {reqs1}); billed {used / (nl * nc):.2f} flops per element of live rows x shifted columns (v1 {used1 / (nl * nc):.2f})')
    ok = ok and e[1] < 5e-6 and e[2] < 5e-6 and e[0] < 2 * max(e1[0], 1e-7) and between[0] < 5e-6
print('strengths: rule', theta, 'exact (c1, c2) per age', strength)
for name, l, f, o, K in birth_cost:
    print(f'birth at layer {l} ({name}): {o} requests, {f} flops for {K} pairs = {f / K:.0f} per pair')
print('worst table difference', f'{worst:.1e}', 'PASS' if ok else 'FAIL')
