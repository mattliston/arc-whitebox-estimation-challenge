"""TAG opus-trainer (2026-10-02): cert/law_ls/linesearch_law.py = m_linesearch.py with the gradient files and the base named on the command line
(the learned-array step test on the card WITH THE PAIR-CARRIED LAW, lead 15:09Z): prepare takes the label of the gradient results
(run/results/a_<label>_r<rank>.pt, gradients of scale x final-layer MSE) and the base parameter file. Original header:
ONE big-batch line-search step for the learned arrays, from the gradients a run has already computed.
Why: a gradient costs 75 s and is noisy; Adam at 1e-4 / 5e-5 per update spends each batch of 12 on a tiny step. The mean of ALL
network gradients computed so far (the parameters have barely moved) is a far better direction, and how far to go along it can
be measured with forward runs, which cost a third of a gradient.
prepare: reads <run>/results/g*.pt (gradients of (1e8 / batch) x final-layer MSE; rescaled to 1e8 x MSE), forms per element the
  mean m and its standard error over the networks, and the direction d = -sign(m) x (rms value of the element's array) for the
  elements with |t| > 2.5 (t = m / se), 0 for the rest: elements whose mean is not distinguishable from noise do not move, and
  every moving element moves by the same fraction eta of its array's typical size (plain steepest descent is useless here:
  one very sensitive scalar, e.cov_factor, takes 99% of it). Trainable arrays only; dials and physics dials untouched. Writes parameter files
  <out dir>/eta_<k>.pt = base + eta_k x d for the given first-order targets (a target of -0.5 means: eta such that the first-order
  change of the pooled error, in-sample, is -0.5%), base = the run's newest raw parameter file, shifts of the physics dials kept.
read: pooled change of the final-layer MSE of each eta against eta 0 on the networks evaluated (results a_<label>_eta<k>_r<rank>
  in the run), the parabola through them and its minimum.
usage (from O/trainer): python -B cert/law_ls/linesearch_law.py prepare <run dir> <out dir> <target %, target %, ...> <gradient label> <base params.pt>
       python -B cert/law_ls/linesearch_law.py read <run dir> <out dir> <label>"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from otr_common import FROZEN_P, load_params, load_pl, save_params

MODE, RUN, OUT = sys.argv[1], Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()
if MODE == 'prepare':
    targets = [float(x) for x in sys.argv[4].split(',')]
    files = sorted((RUN / 'results').glob('a_%s_r*.pt' % sys.argv[5]))
    rows = [torch.load(f, map_location='cpu', weights_only=True) for f in files]
    scale = [1e8 / row['scale'] for row in rows]
    base_path = Path(sys.argv[6]).resolve()
    arrays, dials = load_params(torch, base_path)
    pl = load_pl(torch, base_path)
    mse = float(np.mean([row['mse'] for row in rows])) * 1e8
    direction, first_order, report = {}, 0.0, {}
    for key in rows[0]['grads']:
        G = torch.stack([row['grads'][key].double() * s for row, s in zip(rows, scale)])
        m, se = G.mean(0), G.std(0) / len(rows) ** .5
        t = torch.where(se > 0, m / se.clamp_min(1e-300), torch.zeros_like(m))
        keep = (t.abs() > 2.5).double()
        rms = float(arrays[key].double().pow(2).mean().sqrt())
        direction[key] = -torch.sign(m) * keep * rms
        gain = float((m.abs() * (1 - 1 / (t * t).clamp_min(1e-12)) * keep).sum() * rms)   # first-order decrease per unit eta, shrunk for selection
        first_order += gain
        report[key] = dict(elements=int(m.numel()), moving=int((keep > 0).sum()), beyond_3=int((t.abs() > 3).sum()),
                           norm_direction=float(direction[key].norm()), first_order_share=gain)
    assert not any(('p.' + key) in direction for key in FROZEN_P)
    OUT.mkdir(parents=True, exist_ok=True)
    etas = [0.0] + [-target / 100 * mse / first_order for target in targets]
    for k, eta in enumerate(etas):
        new = {key: (value.double() + eta * direction[key]).to(torch.float32) if key in direction else value for key, value in arrays.items()}
        save_params(torch, OUT / ('eta_%d.pt' % k), new, dials, pl)
    for key in report:
        report[key]['first_order_share'] = round(report[key]['first_order_share'] / first_order, 4)
        report[key]['largest_move_at_eta1'] = float((etas[1] * direction[key]).abs().max())
        report[key]['rms_value'] = float(arrays[key].double().pow(2).mean().sqrt())
    note = dict(run=str(RUN), base=str(base_path), networks=len(rows), gradient_files=[files[0].name, files[-1].name], mean_mse_x1e8=mse,
                targets_pct=[0.0] + targets, etas=etas, first_order_decrease_per_unit_eta=first_order, arrays=report)
    (OUT / 'prepare.json').write_text(json.dumps(note, indent=1))
    print(json.dumps({k: v for k, v in note.items() if k != 'arrays'}, indent=1))
    for key, value in report.items():
        print('%-18s moving %5d of %5d, beyond 3 se %4d, share of the first-order gain %.3f, largest move at eta_1 %.2e (rms value %.3f)' % (
            key, value['moving'], value['elements'], value['beyond_3'], value['first_order_share'], value['largest_move_at_eta1'], value['rms_value']))
else:
    label = sys.argv[4]
    note = json.loads((OUT / 'prepare.json').read_text())
    R = RUN / 'results'
    got = {}
    for k in range(len(note['etas'])):
        for f in R.glob('a_%s_eta%d_r*.json' % (label, k)):
            got.setdefault(int(f.stem.rsplit('_r', 1)[1]), {})[k] = json.loads(f.read_text())
    ranks = sorted(r for r, d in got.items() if 0 in d)
    print('%d networks with the base evaluated' % len(ranks))
    xs, ys = [0.0], [0.0]
    for k in range(1, len(note['etas'])):
        both = [r for r in ranks if k in got[r]]
        if len(both) < 2:
            continue
        a, b = np.array([got[r][k]['mse'] for r in both]), np.array([got[r][0]['mse'] for r in both])
        la, lb = np.array([np.mean(got[r][k]['mse_layers']) for r in both]), np.array([np.mean(got[r][0]['mse_layers']) for r in both])
        ch = 100 * (a / b - 1)
        pooled = 100 * (a.mean() / b.mean() - 1)
        print('eta_%d = %.4g (first-order target %+.2f%%): %2d networks, pooled %+.3f%%, per-network mean %+.3f%% (se %.3f), better on %d/%d, all layers %+.3f%%' % (
            k, note['etas'][k], -note['targets_pct'][k] if note['targets_pct'][k] > 0 else note['targets_pct'][k], len(both), pooled, ch.mean(),
            ch.std(ddof=1) / len(ch) ** .5, int((a < b).sum()), len(both), 100 * (la.mean() / lb.mean() - 1)))
        xs.append(note['etas'][k]); ys.append(pooled)
    if len(xs) >= 3:
        coefficients = np.polyfit(xs, ys, 2)
        if coefficients[0] > 0:
            best = -coefficients[1] / (2 * coefficients[0])
            print('parabola: minimum at eta %.4g, %+.3f%%' % (best, np.polyval(coefficients, best)))
        else:
            print('parabola opens downward: no minimum inside; coefficients', coefficients)
