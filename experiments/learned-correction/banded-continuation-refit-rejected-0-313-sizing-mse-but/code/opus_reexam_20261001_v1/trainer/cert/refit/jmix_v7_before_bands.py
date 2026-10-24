"""TAG opus-trainer (2026-10-10): JOINT REFIT, one step = a mixture of four arms (lead 10-10: joint refit of the learned arrays with the batch-6 head
columns as zero-initialised trainable inputs at layers 1-14; early stopping on the 16 line-search ranks; judge once on the 50). The two kinds of
parameters have very different natural step sizes, so a step is not one line search but a small joint solve, as in the lead's Gauss-Newton note:
from the mean of the exact gradients of a batch of training networks, three directions (elements whose mean is beyond 2.5 standard errors, each moved
against its sign): ARR = the learned arrays (by the same fraction of each array's rms), H6 = column 6's head weights, HB3 = the head weights of the
batch-6 columns m2 (21), eold (28), qb (29), HQ = those of qold (27) alone (lead 03:30Z: qold costs +0.58% of bill natively, so it is judged with and
without), H20 (from step 2 of jmB) = those of column 20 (K4L value on the response rows), H17 (jmC, lead 04:15Z) = those of column 17 at
layer 15 (kept rows; started from the physicist's offline weights physicist/t67_U_17.npy), KC (jmC, lead 05:32Z) = the dial kappa_C: a shift of
the family kc at layers 1-14 by KC_TRIAL (otr_m comment kc; no gradient needed, the solve sizes it, either sign). LIVE ARMS (jmC step 2): a head
arm runs only if at least 3 x the chance count of its weights (P(|t| > 2.5) = 0.017 with 35 degrees of freedom) is beyond 2.5 standard errors
(below that the arm is noise and its first-order trial size explodes: column 6 had 1 of 128, column 17 4 of 128); the list goes to arms.json
'live', the driver runs those, the solve and the point use them. jmD (10-10 08:00Z): env JM_SIZING = the ranks file the arms are run and sized on
(default the 16 line-search ranks); env JM_KC_SPLIT=1 splits the kappa_C arm into kce (layers 1-9) and kcl (layers 10-14). jmE (10-10 09:20Z):
33 head rows (otr_m HSPLIT: row 31 = column 6 at layers 10-14, row 32 = column 20 at layers 10-14); arms h6e / h6l and h20e / h20l; each other arm is run at a trial size whose first-order change is -0.5% of the error, with the final
predictions kept, on the 16 line-search ranks; the final-layer prediction is close to linear in each arm's size, so the sizes that minimise the pooled
error are a 4 x 4 least-squares problem (ridge 0.02 of each arm's energy); the new point is then run for real on the same ranks.
  arms <gradient label> <params.pt> <hu.json> <out dir>      writes the arms' points (arm_<name>.pt / arm_<name>_hu.json) and their trial sizes
  solve <out dir> <label>                                    sizes from the kept predictions (results a_<label>_<arm>_r<rank>, arm = base / arr / h6 / hb3 / hq)
  point <out dir> <new params.pt> <new hu.json>              the new point at the solved sizes
usage (from O/trainer): python -B cert/refit/jmix.py <arms|solve|point> ..."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

T = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(T))
from otr_common import FROZEN_P, data_paths, load_params, load_pl, save_params

R = T / 'run_M/results'
LS16 = [int(x) for x in (T / os.environ.get('JM_SIZING', 'cert/ls16_ranks.txt')).read_text().split(',')]
GROUPS = dict(h6e=(6,), h6l=(31,), hb3=(21, 28, 29), h20e=(20,), h20l=(32,), h17=(17,))      # jmE; qold (row 27) is out (lead 07:50Z)
NROW = 33
HU_SCALE = 0.03
KC_TRIAL = 0.3
KC_ARMS = dict(kce=range(1, 10), kcl=range(10, 15)) if os.environ.get('JM_KC_SPLIT') == '1' else dict(kc=range(1, 15))
mode = sys.argv[1]


def hu_of(path):
    U = np.zeros((NROW, 128)); given = np.array(json.loads(Path(path).read_text())); U[:given.shape[0]] = given
    return U


if mode == 'arms':
    label, P0, H0, OUT = sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [torch.load(f, map_location='cpu', weights_only=False) for f in sorted(R.glob('a_%s_r*.pt' % label))]
    rows = [r for r in rows if 'hu_grads' in r and r['hu_grads'].shape[0] >= 30]
    N = len(rows); mse = float(np.mean([r['mse'] for r in rows])) * 1e8
    arrays0, dials = load_params(torch, P0); pl = load_pl(torch, P0); U0 = hu_of(H0)
    d_arr, first_arr, report = {}, 0.0, {}
    for key in rows[0]['grads']:
        G = torch.stack([r['grads'][key].double() * (1e8 / r['scale']) for r in rows])
        m, se = G.mean(0), G.std(0) / N ** .5
        t = torch.where(se > 0, m / se.clamp_min(1e-300), torch.zeros_like(m)); keep = (t.abs() > 2.5).double()
        rms = float(arrays0[key].double().pow(2).mean().sqrt())
        d_arr[key] = -torch.sign(m) * keep * rms
        g = float((m.abs() * (1 - 1 / (t * t).clamp_min(1e-12)) * keep).sum() * rms); first_arr += g
        report[key] = dict(moving=int(keep.sum()), of=int(m.numel()))
    assert not any(('p.' + key) in d_arr for key in FROZEN_P)
    d_hu, first_hu = {}, {}
    for name, cols in GROUPS.items():
        D = np.zeros((NROW, 128)); f = 0.0
        for row in cols:
            G = np.stack([r['hu_grads'][row].double().numpy() * (1e8 / r['scale']) for r in rows])
            m, se = G.mean(0), G.std(0, ddof=1) / N ** .5
            t = np.where(se > 0, m / np.maximum(se, 1e-300), 0.0); keep = np.abs(t) > 2.5
            D[row] = -np.sign(m) * keep * HU_SCALE
            f += float((np.abs(m) * (1 - 1 / np.maximum(t * t, 1e-12)) * keep).sum() * HU_SCALE)
            report['head row %d' % row] = dict(moving=int(keep.sum()), of=128)
        d_hu[name], first_hu[name] = D, f
    size = dict(arr=0.5 / 100 * mse / max(first_arr, 1e-300), **{name: 0.5 / 100 * mse / max(first_hu[name], 1e-300) for name in GROUPS})
    live = ['arr'] + [name for name in GROUPS if int((d_hu[name] != 0).sum()) >= 3 * 0.017 * 128 * len(GROUPS[name])] + list(KC_ARMS)
    save_params(torch, OUT / 'arm_base.pt', arrays0, dials, pl); (OUT / 'arm_base_hu.json').write_text(json.dumps(U0.tolist()))
    save_params(torch, OUT / 'arm_arr.pt', {k: (v.double() + size['arr'] * d_arr[k]).to(torch.float32) if k in d_arr else v for k, v in arrays0.items()}, dials, pl)
    (OUT / 'arm_arr_hu.json').write_text(json.dumps(U0.tolist()))
    for name in GROUPS:
        save_params(torch, OUT / ('arm_%s.pt' % name), arrays0, dials, pl)
        (OUT / ('arm_%s_hu.json' % name)).write_text(json.dumps((U0 + size[name] * d_hu[name]).tolist()))
    for name, layers in KC_ARMS.items():
        pl_kc = {k: [float(x) for x in v] for k, v in pl.items()}
        pl_kc['kc'] = [x + (KC_TRIAL if l in layers else 0.0) for l, x in enumerate(pl_kc.get('kc', [0.0] * 16))]
        save_params(torch, OUT / ('arm_%s.pt' % name), arrays0, dials, pl_kc); (OUT / ('arm_%s_hu.json' % name)).write_text(json.dumps(U0.tolist()))
        size[name] = KC_TRIAL
    torch.save(dict(d_arr=d_arr, d_hu=d_hu, size=size, base=str(P0), base_hu=str(H0)), OUT / 'direction.pt')
    (OUT / 'arms.json').write_text(json.dumps(dict(gradient_label=label, networks=N, base=str(P0), base_hu=str(H0), mean_mse_x1e8=mse, size=size, live=live,
                                                    first_order=dict(arr=first_arr, **first_hu), report=report), indent=1))
    print('live arms: %s' % ' '.join(live))
    print('%d networks; trial sizes %s; first-order slopes (1e8 x MSE per unit) arr %.4g, %s' % (N, {k: '%.4g' % v for k, v in size.items()}, first_arr,
                                                                                               ', '.join('%s %.4g' % kv for kv in first_hu.items())))
    for key, value in report.items():
        if value['moving']:
            print('  %-16s moving %5d of %5d' % (key, value['moving'], value['of']))
elif mode == 'solve':
    OUT, label = Path(sys.argv[2]), sys.argv[3]
    arms = json.loads((OUT / 'arms.json').read_text()).get('live') or ['arr'] + list(GROUPS) + list(KC_ARMS)
    L = lambda arm, r: json.loads((R / ('a_%s_%s_r%04d.json' % (label, arm, r))).read_text())
    D, res, alone = [], [], {a: [] for a in arms}
    for r in LS16:
        with np.load(data_paths(r)[1], allow_pickle=False) as saved:
            truth = np.asarray(saved['mc_post'], dtype=np.float64)[15]
        b = L('base', r); p0 = np.array(b['final_prediction'])
        D.append(np.stack([np.array(L(a, r)['final_prediction']) - p0 for a in arms], 1)); res.append(truth - p0)
        for a in arms:
            alone[a].append((L(a, r)['mse'], b['mse']))
    E0 = sum((x ** 2).sum() for x in res)
    live = [k for k in range(len(arms)) if sum(float((d[:, k] ** 2).sum()) for d in D) > 0]      # an arm that moved nothing (no weight beyond 2.5 se) stays at 0
    D = [d[:, live] for d in D]
    fit = lambda idx: np.linalg.solve(sum(D[i].T @ D[i] for i in idx) + 0.02 * np.diag(np.diag(sum(D[i].T @ D[i] for i in idx))), sum(D[i].T @ res[i] for i in idx))
    w_live = fit(range(len(LS16))); w = np.zeros(len(arms)); w[live] = w_live
    model = sum(((res[i] - D[i] @ w_live) ** 2).sum() for i in range(len(LS16))) / E0 - 1
    loo = sum(((res[i] - D[i] @ fit([j for j in range(len(LS16)) if j != i])) ** 2).sum() for i in range(len(LS16))) / E0 - 1
    for k, a in enumerate(arms):
        x = np.array(alone[a]); ch = 100 * (x[:, 0] / x[:, 1] - 1)
        print('%-4s alone %+.3f%% (se %.3f, better on %d/16)   size multiple %+.3f' % (a, 100 * (x[:, 0].mean() / x[:, 1].mean() - 1), ch.std(ddof=1) / 4, int((x[:, 0] < x[:, 1]).sum()), w[k]))
    print('model at its optimum %+.3f%%, leave-one-network-out %+.3f%%' % (100 * model, 100 * loo))
    (OUT / 'solve.json').write_text(json.dumps(dict(weights=dict(zip(arms, map(float, w))), model_pct=100 * model, loo_pct=100 * loo), indent=1))
else:
    OUT, out_params, out_hu = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    S = json.loads((OUT / 'solve.json').read_text())['weights']; Dd = torch.load(OUT / 'direction.pt', map_location='cpu', weights_only=False)
    arrays0, dials = load_params(torch, Path(Dd['base'])); pl = load_pl(torch, Path(Dd['base'])); U0 = hu_of(Dd['base_hu'])
    a = S['arr'] * Dd['size']['arr']
    arrays = {k: (v.double() + a * Dd['d_arr'][k]).to(torch.float32) if k in Dd['d_arr'] else v for k, v in arrays0.items()}
    U = U0 + sum(S[name] * Dd['size'][name] * Dd['d_hu'][name] for name in GROUPS if name in S)
    pl = {k: [float(x) for x in v] for k, v in pl.items()}
    for name, layers in dict(kc=range(1, 15), kce=range(1, 10), kcl=range(10, 15)).items():
        if S.get(name):
            pl['kc'] = [x + (S[name] * Dd['size'][name] if l in layers else 0.0) for l, x in enumerate(pl.get('kc', [0.0] * 16))]
    assert not out_params.exists()
    save_params(torch, out_params, arrays, dials, pl); out_hu.write_text(json.dumps(U.tolist()))
    print('new point: arrays at %.3g x the trial step, head weights %s, kc at layers 1 / 14 %+.4f / %+.4f -> %s, %s' % (
        S['arr'], {k: round(S[k], 3) for k in GROUPS if k in S}, pl.get('kc', [0.0] * 16)[1], pl.get('kc', [0.0] * 16)[14], out_params, out_hu))
