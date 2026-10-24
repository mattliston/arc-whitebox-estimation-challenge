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
33 head rows (otr_m HSPLIT: row 31 = column 6 at layers 10-14, row 32 = column 20 at layers 10-14); arms h6e / h6l and h20e / h20l (no signal).
jmF (09:40Z): 36 rows (otr_m HBANDS: rows 33 / 34 / 35 = band biases of the head's first layer at layers 1-9 / 10-14 / 15); arms b19 / b1014 / b15.
jmG (10:25Z): 42 rows (otr_m B7: rows 36-41 = the physicist's batch 7 fold kmu kg qk e34 s1); arms b7k (kmu kg qk), b7es (e34 s1), b7f (fold, +1.34% bill
natively: judged with and without). jmH (13:40Z): DIAL ARMS (env JM_DIALS, e.g. "X1,bscale,c0,k31"): a uniform shift of that per-layer family on
all layers (= a change of the global dial / native knob), against the sign of the summed gradient, trial size from the same -0.5% first-order rule
(zero-dial table cert/cv/dial_table_0cvg.out: X1 mean t^2 17, bscale 6.3, c0 4.6, k31 4.3); an entry family:lo:hi shifts layers lo..hi only (arm
d<family>_<lo>_<hi>; jmI: the layer 10-14 band). JM_ARR_SPLIT=1 (jmL, 10-10 22:00Z; staleness screen at jmIv: the learned EDGE arrays e.* carry
strong gradients, e.rho_scale mean t^2 27, e.bias 10.7, e.output 8.3, e.left 7.1, while the head's p.* are at 1-2.5): the array arm is split into arrp
(p.*, a, c) and arre (e.*), each with its own trial size; each other arm is run at a trial size whose first-order change is -0.5% of the error, with the final
predictions kept, on the 16 line-search ranks; the final-layer prediction is close to linear in each arm's size, so the sizes that minimise the pooled
error are a 4 x 4 least-squares problem (ridge 0.02 of each arm's energy); the new point is then run for real on the same ranks.
  arms <gradient label> <params.pt> <hu.json> <out dir>      writes the arms' points (arm_<name>.pt / arm_<name>_hu.json) and their trial sizes
  solve <out dir> <label>                                    sizes from the kept predictions (results a_<label>_<arm>_r<rank>, arm = base / arr / h6 / hb3 / hq)
  point <out dir> <new params.pt> <new hu.json>              the new point at the solved sizes
v5 (trainer, 10-11 06:45Z, round Ne): env JM_RUN = the run directory whose results / claims hold the gradients (default run_M); env JM_EDGE_SPLIT=1
(with JM_ARR_SPLIT=1) splits the edge arm into ebias (e.bias), eout (e.output), erho (e.rho_scale), eleft (e.left) and erest (the other e.* arrays),
each with its own first-order trial size, applied the same way by 'point'; JM_EDGE_SPLIT=2 (lead 06:45Z) makes TWO edge arms instead: ebo (e.bias +
e.output) and erl (e.rho_scale + e.left + every other e.* array); JM_EDGE_SPLIT=3 (Ne, 08:00Z: at Nd's point e.cov_factor alone carries t^2 18):
ecov (e.cov_factor) and erest (every other e.* array). Env JM_DROP_ARMS (comma list) removes named arms from the live list (Ne: h20, at noise). 'solve' is not used in Ne (cert/Ne/solve_ridge64.py does the 64-network solve).
v6 (trainer, 10-11 10:10Z, round Nf): gradients may come from GENERATED networks read without a truth plan (analyst/fresh, the no-plan worker):
env JM_GRAD_NOPLAN=1 checks each gradient row as truth_plan None, data_source 'generated', rank in the allow file JM_GRAD_ALLOW, instead of the plan
check; JM_REQ must then carry no truth_plan (the requests carry none). Nf (10:40Z): Ne's arms failed at their trial steps although the generated
gradients agree in direction with the public ones (h17 cosine 0.93, c0 / k31 / iso same sign) at smaller t^2: the first-order trial rule overshot. Env
caps on the trial sizes: JM_CAP_ARR (array parts, fraction of each array's rms), JM_CAP_HU (head-row groups, multiples of the 0.03 unit move), with
JM_DIAL_CAP for dials; JM_EDGE_SPLIT=4 gives escal (e.cov_factor + e.third_scale, the two edge scalars whose signal agrees) and erest.
v7 (trainer, 10-11 15:10Z, refit-after-drop pipeline cert/RAD): env JM_FIXED = comma list of fixed FORWARD arms "family:lo:hi:shift" (arm
f<family>_<lo>_<hi>): a shift of that per-layer family on layers lo..hi, no gradient needed (the carrier modules' strengths, e.g. nfs / n6b / nkl2,
get none: their modules run under no_grad), sized by the solve with either sign; 'point' applies weight x shift. Everything else = v6.
usage (from O/trainer): python -B cert/refit/jmix_generated_v7.py <arms|point> ..."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

T = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(T))
from otr_common import FROZEN_P, data_paths as public_data_paths, load_params, load_pl, save_params
from otr_generated_v1 import make_data_paths
_generated_file = os.environ.get('TRAINER_GENERATED_ALLOW_FILE')
_generated_seeds = [int(x) for x in Path(_generated_file).read_text().strip().split(',')] if _generated_file else []
from otr_generated_manifest_v1 import TruthPlan, make_data_paths as manifest_data_paths
from otr_truth_records_v1 import validate_record
TRUTH_PLAN = TruthPlan(os.environ['TRAINER_TRUTH_PLAN'], os.environ['TRAINER_TRUTH_PLAN_SHA256'], _generated_seeds)
data_paths, truth_metadata = manifest_data_paths(public_data_paths, TRUTH_PLAN, T / 'run_M/generated_cache_v1')

RUN = T / os.environ.get('JM_RUN', 'run_M')
R = RUN / 'results'
EDGE_SPLIT = os.environ.get('JM_EDGE_SPLIT') in ('1', '2', '3', '4')
EDGE_TWO = os.environ.get('JM_EDGE_SPLIT') == '2'
EDGE_COV = os.environ.get('JM_EDGE_SPLIT') == '3'
EDGE_SCAL = os.environ.get('JM_EDGE_SPLIT') == '4'
CAP_ARR = float(os.environ.get('JM_CAP_ARR', 'inf')); CAP_HU = float(os.environ.get('JM_CAP_HU', 'inf')); CAP_EDGE = float(os.environ.get('JM_CAP_EDGE', 'inf'))
EDGE_PARTS = dict(ebias='e.bias', eout='e.output', erho='e.rho_scale', eleft='e.left')


def part_of(key):
    if not key.startswith('e.'):
        return 'arrp'
    if not EDGE_SPLIT:
        return 'arre'
    if EDGE_TWO:
        return 'ebo' if key in ('e.bias', 'e.output') else 'erl'
    if EDGE_COV:
        return 'ecov' if key == 'e.cov_factor' else 'erest'
    if os.environ.get('JM_EDGE_SPLIT') == '4':
        return 'escal' if key in ('e.cov_factor', 'e.third_scale') else 'erest'
    for name, k in EDGE_PARTS.items():
        if key == k:
            return name
    return 'erest'
LS16 = [int(x) for x in (T / os.environ.get('JM_SIZING', 'cert/ls16_ranks.txt')).read_text().split(',')]
GROUPS = dict(h6=(6,), hb3=(21, 28, 29), h20=(20,), h17=(17,), b19=(33,), b1014=(34,), b15=(35,), b7k=(37, 38, 39), b7es=(40, 41), b7f=(36,))      # jmG; qold out
NROW = 42
HU_SCALE = 0.03
KC_TRIAL = 0.3
KC_ARMS = dict(kce=range(1, 10), kcl=range(10, 15)) if os.environ.get('JM_KC_SPLIT') == '1' else dict(kc=range(1, 15))
if os.environ.get('JM_NO_KC') == '1':      # Ne (lead 06:40Z): no kappa_C arms (Nd's 32-network LOO dropped them)
    KC_ARMS = {}
DIALS = [x for x in os.environ.get('JM_DIALS', '').split(',') if x]
DIAL_SPEC = {('d' + x.replace(':', '_')): (x.split(':')[0], range(int(x.split(':')[1]), int(x.split(':')[2]) + 1) if ':' in x else range(16)) for x in DIALS}
FIXED = {('f' + x.rsplit(':', 1)[0].replace(':', '_')): (x.split(':')[0], range(int(x.split(':')[1]), int(x.split(':')[2]) + 1), float(x.split(':')[3]))
         for x in os.environ.get('JM_FIXED', '').split(',') if x}      # v7: fixed forward arms
EXPECTED_CUTSIM = json.loads(os.environ.get('JM_REQ') or '{}').get('cutsim')
mode = sys.argv[1]


def hu_of(path):
    U = np.zeros((NROW, 128)); given = np.array(json.loads(Path(path).read_text())); U[:given.shape[0]] = given
    return U


if mode == 'arms':
    label, P0, H0, OUT = sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
    OUT.mkdir(parents=True, exist_ok=True)
    gradient_files = sorted(R.glob('a_%s_r*.pt' % label))
    rows = []
    expected_params = P0.resolve()
    expected_common = dict(f4c=1, ss=0, hf=2, b6=1)
    expected_common.update(json.loads(os.environ.get('JM_REQ') or '{}'))
    for file in gradient_files:
        row = torch.load(file, map_location='cpu', weights_only=False)
        expected_rank = int(file.stem.rsplit('_r', 1)[1])
        assert row.get('id') == file.stem and row.get('rank') == expected_rank
        assert row.get('mode') == 'grad'
        if os.environ.get('JM_GRAD_NOPLAN') == '1':
            allowed = {int(x) for x in Path(os.environ['JM_GRAD_ALLOW']).read_text().strip().split(',')}
            assert row.get('truth_plan') is None and row.get('data_source') == 'generated' and expected_rank in allowed, 'no-plan generated gradient check failed'
        else:
            validate_record(TRUTH_PLAN, row, expected_rank)
        assert np.isfinite(row['mse']) and all(torch.isfinite(v).all() for family in ('grads', 'pl_grads', 'dial_grads') for v in row.get(family, {}).values()), 'Nonfinite production gradient'
        assert torch.isfinite(row['hu_grads']).all(), 'Nonfinite head gradient'
        assert Path(row.get('params', '')).resolve() == expected_params
        assert row.get('cutsim') == EXPECTED_CUTSIM, 'gradient simulator mismatch'
        assert float(row.get('scale', 0)) == 1e8, 'gradient scale mismatch'
        claims = sorted((RUN / 'claimed').glob(file.stem + '.w*.json'))
        assert claims, 'missing gradient claim'
        for claim in claims:
            request = json.loads(claim.read_text())
            assert request.get('id') == file.stem and request.get('rank') == expected_rank
            assert request.get('mode') == 'grad' and request.get('scale') == 1e8
            assert Path(request.get('params', '')).resolve() == expected_params
            assert request.get('hu') == json.loads(H0.read_text())
            assert all(request.get(k) == v for k, v in expected_common.items())
        rows.append(row)
    assert all(r.get('cutsim') == EXPECTED_CUTSIM for r in rows), 'gradient simulator mismatch'
    rows = [r for r in rows if 'hu_grads' in r and r['hu_grads'].shape[0] >= 30]
    N = len(rows); mse = float(np.mean([r['mse'] for r in rows])) * 1e8
    arrays0, dials = load_params(torch, P0); pl = load_pl(torch, P0); U0 = hu_of(H0)
    d_arr, first_arr, report = {}, 0.0, {}
    SPLIT = os.environ.get('JM_ARR_SPLIT') == '1'; first_part = dict(arrp=0.0, arre=0.0, ebias=0.0, eout=0.0, erho=0.0, eleft=0.0, erest=0.0, ebo=0.0, erl=0.0, ecov=0.0, escal=0.0)
    for key in rows[0]['grads']:
        if key in ['p.' + k for k in FROZEN_P]:      # gradient runs made with 'unfreeze' carry them; they stay frozen here
            continue
        G = torch.stack([r['grads'][key].double() * (1e8 / r['scale']) for r in rows])
        m, se = G.mean(0), G.std(0) / N ** .5
        t = torch.where(se > 0, m / se.clamp_min(1e-300), torch.zeros_like(m)); keep = (t.abs() > 2.5).double()
        rms = float(arrays0[key].double().pow(2).mean().sqrt())
        d_arr[key] = -torch.sign(m) * keep * rms
        g = float((m.abs() * (1 - 1 / (t * t).clamp_min(1e-12)) * keep).sum() * rms); first_arr += g
        first_part[part_of(key)] += g
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
    size = dict(arr=min(0.5 / 100 * mse / max(first_arr, 1e-300), CAP_ARR), **{name: min(0.5 / 100 * mse / max(first_hu[name], 1e-300), CAP_HU) for name in GROUPS})
    live = ['arr'] + [name for name in GROUPS if int((d_hu[name] != 0).sum()) >= 3 * 0.017 * 128 * len(GROUPS[name])] + list(KC_ARMS)
    if SPLIT:
        parts = ['arrp'] + ((['ebo', 'erl'] if EDGE_TWO else ['ecov', 'erest'] if EDGE_COV else ['escal', 'erest'] if os.environ.get('JM_EDGE_SPLIT') == '4' else list(EDGE_PARTS) + ['erest']) if EDGE_SPLIT else ['arre'])
        live = parts + live[1:]
        for part in parts:
            size[part] = min(0.5 / 100 * mse / max(first_part[part], 1e-300), CAP_ARR if part == 'arrp' else CAP_EDGE)
            moved = {k: (v.double() + size[part] * d_arr[k]).to(torch.float32) if k in d_arr and part_of(k) == part else v for k, v in arrays0.items()}
            save_params(torch, OUT / ('arm_%s.pt' % part), moved, dials, pl); (OUT / ('arm_%s_hu.json' % part)).write_text(json.dumps(U0.tolist()))
    for name, (fam, layers) in DIAL_SPEC.items():      # dial arms: uniform shift of the family on its layers
        slope = float(np.mean([r['pl_grads'][fam].double().numpy() * (1e8 / r['scale']) for r in rows], 0)[list(layers)].sum())
        shift = -np.sign(slope) * min(0.5 / 100 * mse / max(abs(slope), 1e-300), float(os.environ.get('JM_DIAL_CAP', '0.05')))      # cap (jmI step 2: uncapped trial shifts of flat dials blew up)
        pl_d = {k: [float(x) for x in v] for k, v in pl.items()}
        pl_d[fam] = [x + (shift if l in layers else 0.0) for l, x in enumerate(pl_d.get(fam, [0.0] * 16))]
        save_params(torch, OUT / ('arm_%s.pt' % name), arrays0, dials, pl_d); (OUT / ('arm_%s_hu.json' % name)).write_text(json.dumps(U0.tolist()))
        size[name] = float(shift); live.append(name); report['dial ' + name] = dict(moving=len(layers), of=16)
    for name, (fam, layers, shift) in FIXED.items():      # v7: fixed forward arms (comment v7)
        pl_f = {k: [float(x) for x in v] for k, v in pl.items()}
        pl_f[fam] = [x + (shift if l in layers else 0.0) for l, x in enumerate(pl_f.get(fam, [0.0] * 16))]
        save_params(torch, OUT / ('arm_%s.pt' % name), arrays0, dials, pl_f); (OUT / ('arm_%s_hu.json' % name)).write_text(json.dumps(U0.tolist()))
        size[name] = shift; live.append(name); report['fixed ' + name] = dict(moving=len(layers), of=16)
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
    live = [a for a in live if a not in [x for x in os.environ.get('JM_DROP_ARMS', '').split(',') if x]]
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
    def L(arm, r):
        value = json.loads((R / ('a_%s_%s_r%04d.json' % (label, arm, r))).read_text())
        assert value.get('cutsim') == EXPECTED_CUTSIM, 'sizing simulator mismatch'
        validate_record(TRUTH_PLAN, value, r)
        assert r < 7_000_000 or value.get('data_source') == 'generated', 'generated sizing not run by qualified worker'
        return value
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
    a = S.get('arr', 0.0) * Dd['size']['arr']
    step = {k: a for k in Dd['d_arr']}
    for k in Dd['d_arr']:      # split array arms (JM_ARR_SPLIT, JM_EDGE_SPLIT)
        part = part_of(k)
        if part in S:
            step[k] = step[k] + S[part] * Dd['size'][part]
    arrays = {k: (v.double() + step[k] * Dd['d_arr'][k]).to(torch.float32) if k in Dd['d_arr'] else v for k, v in arrays0.items()}
    U = U0 + sum(S[name] * Dd['size'][name] * Dd['d_hu'][name] for name in GROUPS if name in S)
    pl = {k: [float(x) for x in v] for k, v in pl.items()}
    for name, layers in dict(kc=range(1, 15), kce=range(1, 10), kcl=range(10, 15)).items():
        if S.get(name):
            pl['kc'] = [x + (S[name] * Dd['size'][name] if l in layers else 0.0) for l, x in enumerate(pl.get('kc', [0.0] * 16))]
    for name in S:      # dial arms (the size stored is the signed trial shift)
        if name in DIAL_SPEC and S[name]:
            fam, layers = DIAL_SPEC[name]
            pl[fam] = [x + (S[name] * Dd['size'][name] if l in layers else 0.0) for l, x in enumerate(pl.get(fam, [0.0] * 16))]
    for name in S:      # v7: fixed forward arms
        if name in FIXED and S[name]:
            fam, layers, shift = FIXED[name]
            pl[fam] = [x + (S[name] * shift if l in layers else 0.0) for l, x in enumerate(pl.get(fam, [0.0] * 16))]
    assert not out_params.exists()
    save_params(torch, out_params, arrays, dials, pl); out_hu.write_text(json.dumps(U.tolist()))
    print('new point: arrays at %.3g x the trial step, head weights %s, kc at layers 1 / 14 %+.4f / %+.4f -> %s, %s' % (
        S.get('arr', 0.0), dict({k: round(S[k], 3) for k in GROUPS if k in S}, **{k: round(S[k], 3) for k in ('arrp', 'arre', 'ebias', 'eout', 'erho', 'eleft', 'erest', 'ebo', 'erl', 'ecov', 'escal') if k in S}), pl.get('kc', [0.0] * 16)[1], pl.get('kc', [0.0] * 16)[14], out_params, out_hu))
