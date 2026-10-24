"""dissect337259 agent 2026-10-20: per-network MAGNITUDE + sign model for the common-mode coefficient c (CMODE: Cpre += c mean(v)/|mu|^2 mu mu^T
at L8-14) from the 26-net store screen (cmode_store_{A,B}.json: U = [base-debug, base, c=+0.1, c=-0.1]; eval_cmode_store_{A,B}.log CMODE_OBS lines
L6-14: mumode radial_prev meanv mu2 kink1 gate) plus the state dumps stateprobe/sp_<rank>.npz (GNN delta, cumulants, per layer).
Per net: f(c) in % = a c + b c^2 from f(+-0.1); c* = -a/2b; gain(c) = a c + b c^2 (true quadratic, c clipped to +-CMAX).
Rules are fitted leave-one-net-out; the chance rate is the fraction of 2000 feature permutations whose LOO mean gain is at least as good."""
import json, re, sys
import numpy as np
K = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1/'
CMAX = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
res = {}
for tag in 'AB':
    d = json.load(open(K + 'cmode_store_%s.json' % tag))
    for r, v in d['results'].items():
        if len(v['U']) >= 4: res[int(r)] = v['U']
obs = {}
for tag in 'AB':
    buf = []
    for ln in open(K + 'eval_cmode_store_%s.log' % tag).read().split('\n'):
        if ln.startswith('CMODE_OBS'): buf.append(ln)
        m = re.match(r'  rank (\d+) (\{.*\}) U', ln)
        if m and '"_CMODE_DEBUG": true' in m.group(2): obs[int(m.group(1))] = buf; buf = []
        elif m: buf = []
ranks = sorted(r for r in res if r in obs and len(obs[r]) >= 8)
feat = {}; resp = {}
for r in ranks:
    U = res[r]; fp = 100 * (U[2] / U[1] - 1); fm = 100 * (U[3] / U[1] - 1)
    a = (fp - fm) / 0.2; b = (fp + fm) / 0.02
    resp[r] = dict(fp=fp, fm=fm, a=a, b=b, U0=U[1])
    per = {}
    for ln in obs[r]:
        p = ln.split(); l = int(p[1][1:]); per[l] = {p[i]: float(p[i + 1]) for i in range(2, len(p), 2)}
    L = np.array([l for l in range(8, 15) if l in per], float)
    def col(name, log=False):
        v = np.array([per[int(l)][name] for l in L]); return np.log(np.maximum(v, 1e-12)) if log else v
    f = {}
    for name, log in (('mumode', True), ('radial_prev', True), ('meanv', True), ('mu2', True), ('kink1', False), ('gate', False)):
        v = col(name, log); f['lvl_' + name] = float(v.mean()); f['slp_' + name] = float(np.polyfit(L, v, 1)[0]); f['L14_' + name] = float(v[-1]); f['L8_' + name] = float(v[0])
    f['slp_diff'] = f['slp_mumode'] - f['slp_radial_prev']; f['lvl_Q'] = f['lvl_mumode'] - f['lvl_radial_prev']; f['L14_Q'] = f['L14_mumode'] - f['L14_radial_prev']
    # state-dump features (GNN delta, cumulants) at L8-14
    try:
        d = np.load(K + 'stateprobe/sp_%d.npz' % r)
        g_proj, g_eff, k3n, k4n, kink15, s_mu = [], [], [], [], [], []
        for l in range(8, 15):
            dm = d['delta_%d' % l].astype(float); m0 = d['m0_%d' % l].astype(float); pdf = d['pdf_%d' % l].astype(float); sg = d['sigma_%d' % l].astype(float)
            mu = d['mu_%d' % l].astype(float); pv = d['pre_v_%d' % l].astype(float); a_ = d['alpha_%d' % l].astype(float)
            rr = (pdf / (2 * sg)) * pv.mean() * mu ** 2 / np.sum(mu ** 2)      # CMODE first-order mean response per unit (c = 1)
            g_proj.append(float(np.sum(dm * rr) / np.sum(rr * rr))); g_eff.append(float(np.sqrt(np.mean(dm ** 2)) / np.sqrt(np.mean(m0 ** 2))))
            k3n.append(float(np.mean(d['pre_k3_%d' % l] / sg ** 3))); k4n.append(float(np.mean(d['pre_k4_%d' % l] / sg ** 4)))
            kink15.append(float(np.mean(np.abs(a_) < 1.5)))
        for name, v in (('G', g_proj), ('gnn_eff', g_eff), ('k3n', k3n), ('k4n', k4n), ('kink15', kink15)):
            v = np.array(v); f['lvl_' + name] = float(v.mean()); f['slp_' + name] = float(np.polyfit(np.arange(8, 15), v, 1)[0]); f['L14_' + name] = float(v[-1])
        # base error per layer from the dump (NOT available at predict time; diagnostic only)
        y, tr = d['y'].astype(float), d['truth'].astype(float)
        f['diag_err15'] = float(np.mean((y[15] - tr[15]) ** 2)); f['diag_err_slope'] = float(np.polyfit(np.arange(8, 16), np.log([np.mean((y[l] - tr[l]) ** 2) for l in range(8, 16)]), 1)[0])
    except Exception as e:
        print('no dump for', r, e)
    feat[r] = f
names = sorted(set().union(*[set(f) for f in feat.values()]))
X = np.array([[feat[r].get(n, np.nan) for n in names] for r in ranks]); A = np.array([resp[r]['a'] for r in ranks]); B = np.array([resp[r]['b'] for r in ranks])
print('n =', len(ranks), 'ranks', ranks)
print('slope a (pct per c): mean %.2f sd %.2f; curvature b (pct per c^2): mean %.1f sd %.1f, b<=0 on %d nets' % (A.mean(), A.std(), B.mean(), B.std(), int((B <= 0).sum())))
def gain(a, b, c): return a * c + b * c * c
copt = np.clip(np.where(B > 0, -A / (2 * np.maximum(B, 1e-9)), np.sign(-A) * CMAX), -CMAX, CMAX)
print('oracle per-net gain (quadratic, |c|<=%.2f): mean %.3f%%  best-of-(+-0.1): %.3f%%' % (CMAX, gain(A, B, copt).mean(), np.minimum(np.array([resp[r]['fp'] for r in ranks]), np.array([resp[r]['fm'] for r in ranks])).clip(max=0).mean()))
print('--- corr of each feature with a (slope) and with c*')
rows = []
for j, n in enumerate(names):
    x = X[:, j]; ok = np.isfinite(x)
    if ok.sum() < 10 or np.std(x[ok]) == 0: continue
    rows.append((n, np.corrcoef(x[ok], A[ok])[0, 1], np.corrcoef(x[ok], copt[ok])[0, 1]))
for n, ca, cc in sorted(rows, key=lambda t: -abs(t[1]))[:25]: print('  %-22s corr(a) %+.2f  corr(c*) %+.2f' % (n, ca, cc))
rng = np.random.default_rng(0)
def loo_linear(xcols, A, B, shrink=0.0):
    """rule: c_i = clip(-(w0 + w.x_i) / (2 bbar), +-CMAX) with (w0, w) = LOO least squares of a on x and bbar = LOO mean of b (b clipped >= 5)."""
    n = len(A); g = np.zeros(n); cs = np.zeros(n)
    for i in range(n):
        tr = np.arange(n) != i
        Xd = np.column_stack([np.ones(tr.sum()), xcols[tr]]); w = np.linalg.lstsq(Xd.T @ Xd + shrink * np.eye(Xd.shape[1]), Xd.T @ A[tr], rcond=None)[0]
        a_hat = w[0] + xcols[i] @ w[1:]; bbar = max(np.mean(np.maximum(B[tr], 5.0)), 5.0)
        c = float(np.clip(-a_hat / (2 * bbar), -CMAX, CMAX)); cs[i] = c; g[i] = gain(A[i], B[i], c)
    return g, cs
def perm_rate(xcols, A, B, stat, nperm=2000, shrink=0.0):
    cnt = 0
    for _ in range(nperm):
        p = rng.permutation(len(A)); g, _ = loo_linear(xcols[p], A, B, shrink); cnt += (g.mean() <= stat)
    return cnt / nperm
print('--- LOO linear rules (c = -a_hat/(2 bbar), |c| <= %.2f): mean gain (negative = better), lower share, chance rate (2000 perms)' % CMAX)
cands = [n for n in names if not n.startswith('diag_')]
single = []
for n in cands:
    x = X[:, [names.index(n)]]; ok = np.isfinite(x[:, 0])
    if ok.sum() < len(A) or np.std(x) == 0: continue
    g, cs = loo_linear(x, A, B); single.append((g.mean(), n, (g < 0).mean()))
single.sort()
for gm, n, lo in single[:8]:
    x = X[:, [names.index(n)]]; pr = perm_rate(x, A, B, gm, nperm=500)
    print('  %-22s LOO gain %+.3f%%  lower %.2f  chance %.3f' % (n, gm, lo, pr))
for pair in (('slp_mumode', 'slp_radial_prev'), ('slp_diff', 'lvl_kink1'), ('slp_diff', 'lvl_G'), ('slp_mumode', 'lvl_G')):
    if all(p in names for p in pair):
        x = X[:, [names.index(p) for p in pair]]
        if np.isfinite(x).all():
            g, cs = loo_linear(x, A, B); pr = perm_rate(x, A, B, g.mean(), nperm=500)
            print('  %-40s LOO gain %+.3f%%  lower %.2f  chance %.3f' % ('+'.join(pair), g.mean(), (g < 0).mean(), pr))
# constant rules for reference
for c in (0.1, -0.1, 0.05, -0.05):
    print('  const c=%+.2f: mean gain %+.3f%%' % (c, gain(A, B, c).mean()))
np.savez(K + 'agent_ideas_v1/cmode_magnitude_v1.npz', ranks=np.array(ranks), names=np.array(names), X=X, A=A, B=B, copt=copt)
