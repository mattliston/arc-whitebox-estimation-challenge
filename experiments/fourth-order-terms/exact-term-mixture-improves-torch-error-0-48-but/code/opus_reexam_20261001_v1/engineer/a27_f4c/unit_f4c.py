"""TAG opus_engineer a27 F4c unit check (light, small arrays, single thread, no estimator): the fragment against the physicist's
plain-array reference O/physicist/f4c_native_ref.py (numpy float64) on synthetic data.
  birth: the six coefficient vectors per pair and age against exact_constants + F4's rule, for two cohorts (born at layers 0 and 1;
         the first has alpha = 0 as on the card: its rule part is exactly 0, its S2 is not). Cohort 1 holds three degenerate pairs
         (gap 0, a common leg of norm 0, a nearly singular system): the reference gives them no exact part.
  step:  at step 2 both cohorts are shifted (ages 2 and 1): shift, beta and the plane diagonal against f4c_step per cohort.
usage: unit_f4c.py [<fragment file>] [<theta age 1> <theta age 2> <c1 age 1> <c2 age 1> <c1 age 2> <c2 age 2>]"""
from pathlib import Path
import importlib.util
import math
import sys
import numpy as np
import flopscope as flops
fnp = flops.numpy
HERE = Path(__file__).resolve().parent
O = HERE.parents[1]
spec = importlib.util.spec_from_file_location('f4c_native_ref', O/'physicist/f4c_native_ref.py'); ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)


class _E2Hold(list):
    def __call__(self, x):
        self.append(x)
        return x


args = sys.argv[1:]
fragment = args.pop(0) if args and args[0].endswith('.py') else 'fragment_f4c_v1.py'
S19 = (1.0032287195131746, 0.9932375877606482)
ns = dict(fnp=fnp, _E2Hold=_E2Hold, _S19_LAM=S19)
exec(compile((HERE/fragment).read_text(), fragment, 'exec'), ns)
if args:
    v6 = [float(x) for x in args]
    ns['_F4C_THETA'] = (v6[0], v6[1]); ns['_F4C_C'] = ((v6[2], v6[3]), (v6[4], v6[5]))
theta = ns['_F4C_THETA'] if isinstance(ns['_F4C_THETA'], tuple) else (ns['_F4C_THETA'],) * 2
strength = ns['_F4C_C'] if isinstance(ns['_F4C_C'][0], tuple) else (ns['_F4C_C'],) * 2
rng = np.random.default_rng(11)
n, nl = 512, 448
rel = lambda x, y: float(np.linalg.norm(np.asarray(x, dtype=np.float64) - y) / max(np.linalg.norm(y), 1e-300))
births, state, lam_parts, start = {}, dict(columns=0, cohorts=[]), [], 0
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
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        ns['_f4c_birth'](l, aux, fnp.asarray(d), fnp.asarray(host), fnp.asarray(sel), fnp.asarray(t), fnp.asarray(s), fnp.asarray(base), fnp.asarray(weight), radial)
        birth_cost.append((int(ctx.summary_dict()['flops_used']), len(ctx._op_log), K))
    al, sg, gt, ph, m, tp, tm = alpha[sel], sigma[sel], gate[sel], pdf[sel], host[sel], t[sel], s[sel]
    S1 = -al * ph / (sg * sg); S2 = 2.0 * (gt - gt * gt - m * ph / sg)
    s2 = np.maximum((base.astype(np.float64) ** 2).sum(0), 1e-30)
    A = -al / (sg * (1 + radial / 3))
    per_age, masked = [], None
    for i in (0, 1):                                                # age i + 1
        n_p, n_m = ref.exact_constants(np, tp, tm, S1, S2, s2, strength[i][0], strength[i][1])
        masked = np.flatnonzero(np.abs(n_p).sum(0) + np.abs(n_m).sum(0) == 0)
        lam_out = weight.astype(np.float64) * S19[1] * S19[0] ** (i + 1)
        want = [theta[i] * A + n_p[0] / lam_out, n_p[1] / lam_out, n_p[2] / lam_out,
                theta[i] * A - n_m[0] / lam_out, -2 * theta[i] * A - n_m[1] / lam_out, theta[i] * A - n_m[2] / lam_out]
        got = ns['_F4C_COEF'][l][i]
        errors = [rel(got[q], want[q]) for q in range(6)]
        worst = max(worst, max(errors))
        on_masked = max([float(np.abs(np.asarray(got[q], dtype=np.float64)[masked] - want[q][masked]).max()) for q in range(6)]) if len(masked) else 0.0
        print(f'birth layer {l}, age {i + 1}: coefficient vectors, relative difference {" ".join("%.1e" % e for e in errors)}; pairs without an exact part in the reference: {list(masked)} (largest absolute difference there {on_masked:.1e})')
        per_age.append((n_p, n_m))
    births[l] = dict(A=A, n=per_age, K=K, weight=weight)
    state['cohorts'].append((start, 2 * K)); start += 2 * K
nc = start
# step 2: cohort 0 is 2 layers old, cohort 1 is 1 layer old; lam = the weights carried into the step
lam = np.concatenate([np.concatenate([x, -x]) for x in (births[0]['weight'] * np.float32(S19[1]) * np.float32(S19[0]), births[1]['weight'] * np.float32(S19[1]))]).astype(np.float32)
l = 2
alpha = rng.standard_normal(n) * 1.2; sigma = rng.uniform(0.6, 1.8, n)
gate = np.array([0.5 * math.erfc(-x / math.sqrt(2)) for x in alpha]); d = np.exp(-alpha ** 2 / 2) / math.sqrt(2 * math.pi) / sigma
live = np.sort(rng.permutation(n)[:nl]); alive = np.zeros(n); alive[live] = 1
C = rng.standard_normal((n, n)); C = C @ C.T / n + np.eye(n); v = np.diag(C).copy()
right = (rng.standard_normal((nl, nc)) * 0.4).astype(np.float32)
pp = np.zeros((n, nc), dtype=np.float32); pp[live] = right
u = (alive * gate).astype(np.float32)[:, None] * pp                 # the outgoing gated columns, as the card builds them
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    out = ns['_f4c_step'](l, fnp.asarray(right), fnp.asarray(gate), fnp.asarray(d), fnp.asarray(live), fnp.asarray(v), fnp.asarray(lam), state)
    used = int(ctx.summary_dict()['flops_used']); ops = len(ctx._op_log)
shift, beta, diagonal = out
lam_out = lam.astype(np.float64) * S19[0]
shift_ref = np.zeros((n, nc)); beta_ref = np.zeros(n); diag_ref = np.zeros(n)
for j in (0, 1):
    k0, width = state['cohorts'][j]; K = width // 2; P, Mi = slice(k0, k0 + K), slice(k0 + K, k0 + 2 * K)
    i = l - j - 1
    n_p, n_m = births[j]['n'][i]
    new_p, new_q, be, dg = ref.f4c_step(np, pp[:, P], pp[:, Mi], u[:, P], u[:, Mi], lam_out[P], lam_out[Mi], births[j]['A'], n_p, n_m, theta[i], alive, d, v)
    shift_ref[:, P] = new_p.astype(np.float64) - u[:, P]; shift_ref[:, Mi] = new_q.astype(np.float64) - u[:, Mi]
    beta_ref += be; diag_ref += dg
print('strengths: rule', theta, 'exact (c1, c2) per age', strength)
print(f'step {l}: shift (live rows, {int(shift.shape[1])} of {nc} columns): relative difference {rel(shift, shift_ref[live]):.2e} (the reference rounds u + shift to float32); reference zero off the live rows: {bool(np.abs(np.delete(shift_ref, live, axis=0)).max() == 0)}')
print(f'step {l}: beta {rel(beta, beta_ref):.2e}; plane diagonal {rel(diagonal, diag_ref):.2e}')
print(f'step billed {used / (nl * nc):.2f} flops per element of live rows x shifted columns, {ops} charged ops for two cohorts; stats {ns["_F4C_STATS"]}')
for (f, o, K), l in zip(birth_cost, (0, 1)):
    print(f'birth at layer {l}: {o} charged ops, {f} flops for {K} pairs = {f / K:.0f} per pair ({2 * n} of them for the leg norms)')
print('worst coefficient difference', f'{worst:.1e}', 'PASS' if worst < 5e-6 and rel(beta, beta_ref) < 5e-6 and rel(diagonal, diag_ref) < 5e-6 else 'FAIL')
