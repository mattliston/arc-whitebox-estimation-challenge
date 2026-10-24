"""TAG fable-lead 2026-10-20: per-network optimum of the common-mode coefficient c (quadratic through c = -0.1, 0, +0.1 from the
store screen cmode_store_{A,B}.json) regressed on card-observable per-network features: the CMODE_OBS lines of the screen's base
run (mumode, radial, mean v, |mu|^2, kink fraction, gate per layer L6-14, from eval_cmode_store_{A,B}.log) and stateprobe summary
features (sp_<rank>.npz). Leave-one-net-out linear fits; reports the MSE gain of the predicted per-net c vs c = 0 (clipped to
[-0.1, 0.1] and also the sign-only rule c = +-0.05). v2: the regression target is the SLOPE at 0, s = (U(+0.1) - U(-0.1)) / (0.2 U(0)) (linear in the data; the vertex only sizes the gain);
Q and kink1_late are the two PRE-DECLARED features, everything else exploratory; a label-permutation chance rate is printed for
the LOO sign rule. v3 adds the stateprobe summary features (sp_features_v1.json) as exploratory regressors. usage: cmode_sign_fit_v3.py"""
import glob, json, re
import numpy as np
K = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1/'
res = {}
for tag in 'AB':
    try:
        d = json.load(open(K + 'cmode_store_%s.json' % tag))
    except Exception:
        continue
    for r, v in d['results'].items():
        res[int(r)] = v
obs = {}
for tag in 'AB':
    cur = None
    try:
        lines = open(K + 'eval_cmode_store_%s.log' % tag).read().split('\n')
    except Exception:
        continue
    # the base (debug) setting prints CMODE_OBS lines BEFORE its '  rank N {"_CMODE": {}, "_CMODE_DEBUG": true}' summary line
    buf = []
    for ln in lines:
        if ln.startswith('CMODE_OBS'):
            buf.append(ln)
        m = re.match(r'  rank (\d+) (\{.*\}) U', ln)
        if m and '"_CMODE_DEBUG": true' in m.group(2):
            rank = int(m.group(1)); obs[rank] = buf; buf = []
        elif m:
            buf = []
SP = json.load(open(K + 'sp_features_v1.json'))
def feats(rank):
    f = {}
    for k, v in SP.get(str(rank), {}).items():
        f['sp_' + k] = v
    for ln in obs.get(rank, []):
        p = ln.split(); l = int(p[1][1:])
        for i in range(2, len(p), 2):
            f['%s_L%d' % (p[i], l)] = float(p[i + 1])
    for name in ('mumode', 'radial_prev', 'meanv', 'mu2', 'kink1', 'gate'):
        vals = [f[k] for k in f if k.startswith(name + '_L') and int(k.split('_L')[1]) >= 10]
        if vals: f[name + '_late'] = float(np.mean(vals))
    if 'mumode_late' in f and 'radial_prev_late' in f and f['radial_prev_late'] > 0:
        f['Q'] = f['mumode_late'] / f['radial_prev_late']
    return f
rows = []
for rank, v in sorted(res.items()):
    U = v['U']   # settings: [base(debug), base, +0.1, -0.1]
    if len(U) < 4: continue
    u0, up, um = U[1], U[2], U[3]
    a = (up + um - 2 * u0) / 2 / 0.01; b = (up - um) / 0.2   # U(c) ~ u0 + b c + a c^2
    cstar = -b / (2 * a) if a > 0 else (-0.1 if up > um else 0.1)
    cstar = float(np.clip(cstar, -0.1, 0.1)); ustar = u0 + b * cstar + a * cstar ** 2
    f = feats(rank); f['best_pm'] = 100 * (min(up, um) / u0 - 1); f['gain_star'] = 100 * (ustar / u0 - 1); f['cstar'] = cstar; f['dplus'] = 100 * (up / u0 - 1); f['dminus'] = 100 * (um / u0 - 1)
    rows.append((rank, f))
print('nets with results:', len(rows))
for rank, f in rows:
    print('rank %d  +0.1 %+.3f%%  -0.1 %+.3f%%  c* %+.3f gain %+.3f%% | Q %s mumode_late %s kink1_late %s gate_late %s radial %s' % (
        rank, f['dplus'], f['dminus'], f['cstar'], f['gain_star'], f.get('Q', 'na'), f.get('mumode_late', 'na'), f.get('kink1_late', 'na'), f.get('gate_late', 'na'), f.get('radial_prev_late', 'na')))
names = [n for n in ('Q', 'kink1_late', 'mumode_late', 'radial_prev_late', 'gate_late', 'meanv_late', 'mu2_late', 'sp_kink15_late', 'sp_k3n_late', 'sp_k4n_late', 'sp_delta_rms_late', 'sp_alpha_sd_late', 'sp_crow2_late', 'sp_k21row2_late', 'sp_dcovrow2_late', 'sp_base_mse15_truth') if all(n in f for _, f in rows)]
if len(rows) >= 6 and names:
    X = np.array([[f[n] for n in names] for _, f in rows]); dp = np.array([f['dplus'] for _, f in rows]); dm = np.array([f['dminus'] for _, f in rows])
    y = (dp - dm) / 0.2   # slope of U(c)/U(0) in % per unit c at c = 0: negative slope -> c* > 0
    print('slope target s = (U(+.1)-U(-.1))/(0.2 U0) per net:', ' '.join('%d:%+.1f' % (r, v) for (r, _), v in zip(rows, y)))
    print('corr(slope, feature):', ' '.join('%s %+.2f' % (n, np.corrcoef(y, X[:, j])[0, 1]) for j, n in enumerate(names)))
    for sub in ([names[0]], [names[1]], names[:2], [n for n in names if n.startswith('sp_') and 'truth' not in n][:6], [n for n in names if 'truth' not in n]):
        Xd = np.c_[np.ones(len(y)), np.array([[f[n] for n in sub] for _, f in rows])]
        pred = np.zeros_like(y)
        for i in range(len(y)):
            m = np.ones(len(y), bool); m[i] = False
            beta = np.linalg.lstsq(Xd[m], y[m], rcond=None)[0]; pred[i] = Xd[i] @ beta
        # realised gain of the predicted sign at +-0.1 (sign rule) and of the clipped prediction (linear interpolation of the measured deltas)
        sign_gain = np.where(pred < 0, dp, dm)   # predicted negative slope -> use c = +0.1, else c = -0.1
        rng = np.random.default_rng(0); chance = []
        for _ in range(400):
            yp = rng.permutation(y); pp = np.zeros_like(y)
            for i in range(len(y)):
                m = np.ones(len(y), bool); m[i] = False
                beta = np.linalg.lstsq(Xd[m], yp[m], rcond=None)[0]; pp[i] = Xd[i] @ beta
            chance.append(np.where(pp < 0, dp, dm).mean())
        chance = np.array(chance)
        print('LOO %-44s R2(slope) %.2f | sign rule +-0.1 mean %+.3f%% (lower %d/%d) | permutation chance: P(mean <= observed) %.2f (perm mean %+.3f%%)' % (sub, 1 - np.mean((y - pred) ** 2) / np.var(y), sign_gain.mean(), int((sign_gain < 0).sum()), len(y), float((chance <= sign_gain.mean()).mean()), chance.mean()))
    U0 = np.array([res[r]['U'][1] for r, _ in rows]); Up = np.array([res[r]['U'][2] for r, _ in rows]); Um = np.array([res[r]['U'][3] for r, _ in rows])
    pooled = lambda Ux: 100 * (Ux.sum() / U0.sum() - 1)
    print('oracle (best of +-0.1 per net): mean %+.3f%% pooled %+.3f%%; always +0.1: mean %+.3f%% pooled %+.3f%%; always -0.1: mean %+.3f%% pooled %+.3f%%; corr(base U, slope) %+.2f' % (
        np.minimum(dp, dm).mean(), pooled(np.minimum(Up, Um)), dp.mean(), pooled(Up), dm.mean(), pooled(Um), np.corrcoef(U0, y)[0, 1]))
