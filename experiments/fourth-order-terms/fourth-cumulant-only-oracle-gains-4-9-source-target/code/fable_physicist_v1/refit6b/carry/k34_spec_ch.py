#!/usr/bin/env python3
"""TAG fable-physicist (CR-24): versioned copy of fable_analyst_v1/k34_spec.py (AN-59) with a CHANNEL switch: eps<e>[k3|k4] and
noise<r>[k3|k4] inject only that channel (the other keeps the card's value). Everything else is unchanged; imports and the adjoint/jac
files come from fable_analyst_v1 (read-only), outputs go to this directory. Original docstring follows.
TAG fable-analyst (AN-59): SPEC for a per-unit k3+k4 source (lead's task after RESULT section 34). Torch s6bcard + B2 ep1, learned ON,
no refit; final MSE against the organizer truth (truth688); per-unit truth pre k3/k4 = our MC files (32M where they exist, else 8M), draws
independent of the scoring truth. k3 = the diagonal of the pre-activation K21 plane, k4 = the per-unit fourth cumulant.
Injection forms: 'pair' (as section 34) changes only the pair map's inputs; 'all' also writes the new k3 into the k21 diagonal in place, so
the message-edge source, the GNN features and the birth update at that layer see it too (k4 has no other consumer).
Variant = terms joined by '+', first term 'base' (or 'adjoint'):
  delta<s>@a-b        learned mean correction x s at layers a-b (several terms allowed)
  eps<e>@a-b          k3, k4 := card + e*(truth - card) at layers a-b, pair form;  epsall<e>@a-b: same, all form
  noise<r>@a-b        k3, k4 := truth + r*sigma_l*z (pair form), z ~ N(0,1) fixed per net/layer/channel, sigma_l = rms_i(truth - card)
  top<K><how>[e<x>|n<x>]@a-b   truth on the top-K units per layer only (pair form); how = adj (|adjoint| x the layer's rms error, per
                      channel, combined in quadrature), gain (first-order MSE decrease -(g3 d3 + g4 d4)), rand (fixed seed), jac (truth-free:
                      Jacobian norm of the L15 prediction x the layer's rms error), adjflip (adj units, but a random-sign fix s*d), fake
                      (gain-style selection and injection of a random-sign fix s*d: the leakage control); optional e<x> = card + x*(truth -
                      card) and n<x> = truth + x*sigma_l*z at the selected units; adj/gain/adjflip/fake read the adjoint file, jac the jac file
Mode 'adjoint' (e.g. 'adjoint+delta0.5@12-14'): d(final MSE)/d(k3_i) and d/d(k4_i) at the pair map's inputs, L12-14, by autograd through
L12-15 (L0-11 without grad); saved with the per-unit errors (truth - card) to k34_adj_rank<r>.npz. Mode 'jacobian<R>' (e.g.
'jacobian16+delta0.5@12-14'): per-unit Jacobian norms of the L15 prediction w.r.t. k3_i and k4_i at L12-14, from R Rademacher output
projections (no truth used), saved to k34_jac_rank<r>.npz.
Every run records the card's own k3/k4 relative rms error against truth at L1-14 (before any injection) and the units injected per layer.
usage: k34_spec.py <ranks> <threads> <out.jsonl> <variants>"""
import sys, json, time, re
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parents[2]; AN = P2 / 'fable_analyst_v1'; sys.path.insert(0, str(AN))
import numpy as np, torch
import s6bcard
ranks = [int(x) for x in sys.argv[1].split(',')]; torch.set_num_threads(int(sys.argv[2])); OUT = HERE / sys.argv[3]; VARIANTS = sys.argv[4].split(',')
pl = s6bcard.payload(torch, params=P2 / 'fable_physicist_v1/refit6b/s6b_B2_fullnet_lr1e4_trunk5e5/params_epoch1.pt')
f64 = torch.float64


def parse(vname):
    terms = vname.split('+'); cfg = dict(delta={}, inj=None, adjoint=terms[0] == 'adjoint', jac=int(terms[0][8:]) if terms[0].startswith('jacobian') else 0)
    assert terms[0] in ('base', 'adjoint') or cfg['jac'], vname
    for term in terms[1:]:
        head, span = term.split('@'); a, b = (int(x) for x in span.split('-'))
        if head.startswith('delta'):
            cfg['delta'].update({l: float(head[5:]) for l in range(a, b + 1)}); continue
        assert cfg['inj'] is None, 'one injection term per variant'
        mc = re.fullmatch(r'(epsall|eps|noise)([0-9.]+)(k3|k4)?', head)
        if mc and mc.group(1) == 'epsall': inj = dict(kind='eps', e=float(mc.group(2)), all=True, ch=mc.group(3))
        elif mc and mc.group(1) == 'eps': inj = dict(kind='eps', e=float(mc.group(2)), all=False, ch=mc.group(3))
        elif mc and mc.group(1) == 'noise': inj = dict(kind='noise', r=float(mc.group(2)), all=False, ch=mc.group(3))
        elif head.startswith('top'):
            mt = re.fullmatch(r'top(\d+)(adjflip|adj|gain|rand|jac|fake)(?:([en])([0-9.]+))?', head)
            inj = dict(kind='top', K=int(mt.group(1)), how=mt.group(2), mod=mt.group(3), x=float(mt.group(4)) if mt.group(4) else None, all=False)
        else: raise ValueError(term)
        inj.update(a=a, b=b); cfg['inj'] = inj
    return cfg


def signs(r, l, ch):                                     # fixed random +-1 per net / layer / channel (adjflip and fake controls)
    return torch.randint(0, 2, (1024,), generator=torch.Generator().manual_seed(15485863 * r + 101 * l + ch)).to(f64) * 2 - 1


def top_units(inj, l, r, ADJ, JAC):
    if inj['how'] == 'rand':
        return torch.randperm(1024, generator=torch.Generator().manual_seed(104729 * r + l))[:inj['K']]
    g3, g4, d3, d4 = (torch.tensor(ADJ[f'{k}_{l}']) for k in ('g3', 'g4', 'd3', 'd4'))
    s3, s4 = d3.pow(2).mean().sqrt(), d4.pow(2).mean().sqrt()
    if inj['how'] in ('adj', 'adjflip'):
        score = ((g3 * s3) ** 2 + (g4 * s4) ** 2).sqrt()
    elif inj['how'] == 'jac':
        j3, j4 = torch.tensor(JAC[f'j3_{l}']), torch.tensor(JAC[f'j4_{l}'])
        score = ((j3 * s3) ** 2 + (j4 * s4) ** 2).sqrt()
    elif inj['how'] == 'fake':
        score = -(g3 * signs(r, l, 3) * d3 + g4 * signs(r, l, 4) * d4)
    else:
        score = -(g3 * d3 + g4 * d4)
    return torch.argsort(-score, stable=True)[:inj['K']]


def run(r, vname, w, tp, TR, fo, base_mse):
    cfg = parse(vname); inj = cfg['inj']; cd = s6bcard.card(torch, w, *pl); kerr, nunits, grads = {}, {}, {}
    ADJ = np.load(AN / f'k34_adj_rank{r:04d}.npz') if inj and inj['kind'] == 'top' and inj['how'] != 'rand' else None
    JAC = np.load(AN / f'k34_jac_rank{r:04d}.npz') if inj and inj['kind'] == 'top' and inj['how'] == 'jac' else None
    if cfg['delta']:
        cd.hooks['delta'] = lambda l, d, ctx: d * cfg['delta'][l] if l in cfg['delta'] else d
    def pin(l, v, ctx):
        mu, Cp, k21, k4, k22, r31 = v
        t3 = torch.tensor(TR['mc_pre_k3'][l].astype(np.float64)); t4 = torch.tensor(TR['mc_pre_k4'][l].astype(np.float64))
        c3 = torch.diagonal(k21).detach().clone(); c4 = (k4 * torch.ones_like(t4)).detach().clone()
        if 1 <= l <= 14:
            kerr[l] = dict(k3=float((c3 - t3).norm() / t3.norm()), k4=float((c4 - t4).norm() / t4.norm()),
                           k3scale=float((c3 * t3).sum() / (t3 * t3).sum()), k4scale=float((c4 * t4).sum() / (t4 * t4).sum()))
        if (cfg['adjoint'] or cfg['jac']) and 12 <= l <= 14:
            d3 = torch.zeros(1024, dtype=f64, requires_grad=True); d4 = torch.zeros(1024, dtype=f64, requires_grad=True)
            grads[l] = (d3, d4, (t3 - c3).numpy(), (t4 - c4).numpy())
            return (mu, Cp, k21 + torch.diag(d3), k4 * torch.ones_like(t4) + d4, k22, r31)
        if inj is None or not inj['a'] <= l <= inj['b']: return v
        if inj['kind'] == 'eps':
            n3, n4 = c3 + inj['e'] * (t3 - c3), c4 + inj['e'] * (t4 - c4); nunits[l] = 1024
        elif inj['kind'] == 'noise':
            g = torch.Generator().manual_seed(7919 * r + 31 * l)
            s3, s4 = (t3 - c3).pow(2).mean().sqrt(), (t4 - c4).pow(2).mean().sqrt()
            n3 = t3 + inj['r'] * s3 * torch.randn(1024, generator=g, dtype=f64)
            n4 = t4 + inj['r'] * s4 * torch.randn(1024, generator=g, dtype=f64); nunits[l] = 1024
        else:
            m = torch.zeros(1024, dtype=f64); m[top_units(inj, l, r, ADJ, JAC)] = 1
            e3, e4 = t3 - c3, t4 - c4
            if inj['how'] in ('adjflip', 'fake'): e3, e4 = signs(r, l, 3) * e3, signs(r, l, 4) * e4
            if inj['mod'] == 'e': e3, e4 = inj['x'] * e3, inj['x'] * e4
            if inj['mod'] == 'n':
                g = torch.Generator().manual_seed(7919 * r + 31 * l)
                e3 = e3 + inj['x'] * (t3 - c3).pow(2).mean().sqrt() * torch.randn(1024, generator=g, dtype=f64)
                e4 = e4 + inj['x'] * (t4 - c4).pow(2).mean().sqrt() * torch.randn(1024, generator=g, dtype=f64)
            n3, n4 = c3 + m * e3, c4 + m * e4; nunits[l] = int(m.sum())
        if inj.get('ch') == 'k4': n3 = c3                     # channel switch: k4 only (card k3 kept)
        if inj.get('ch') == 'k3': n4 = c4                     # k3 only (card k4 kept)
        if inj['all']:
            k21.diagonal().copy_(n3)                          # in place: every consumer of k21 at this layer sees the new k3
            return (mu, Cp, k21, n4, k22, r31)
        return (mu, Cp, torch.diagonal_scatter(k21, n3), n4, k22, r31)
    cd.hooks['pair_in'] = pin
    t0 = time.time()
    if cfg['adjoint']:
        with torch.no_grad(): _, st = cd.run_states(0, None, 12)
        with torch.enable_grad():
            outs, _ = cd.run_states(12, st, 16)
            pred = outs[-1].clamp_min(0).to(torch.float32).to(f64)
            loss = ((pred - torch.tensor(tp)) ** 2).mean(); loss.backward()
        mse = float(loss)
        save = {}
        for l, (d3, d4, e3, e4) in grads.items():
            save.update({f'g3_{l}': d3.grad.numpy(), f'g4_{l}': d4.grad.numpy(), f'd3_{l}': e3, f'd4_{l}': e4})
        np.savez(HERE / f'k34_adj_rank{r:04d}.npz', **save)
        lin = {l: float(-(save[f'g3_{l}'] * save[f'd3_{l}'] + save[f'g4_{l}'] * save[f'd4_{l}']).sum() / mse) for l in grads}
    elif cfg['jac']:
        with torch.no_grad(): _, st = cd.run_states(0, None, 12)
        with torch.enable_grad():
            outs, _ = cd.run_states(12, st, 16)
            pred = outs[-1].clamp_min(0).to(torch.float32).to(f64)
            acc = {l: [torch.zeros(1024, dtype=f64), torch.zeros(1024, dtype=f64)] for l in grads}
            gz = torch.Generator().manual_seed(2750159 * r)
            for q in range(cfg['jac']):
                z = torch.randint(0, 2, (1024,), generator=gz).to(f64) * 2 - 1
                for l, (d3, d4, _, _) in grads.items(): d3.grad = None; d4.grad = None
                (z * pred).sum().backward(retain_graph=q < cfg['jac'] - 1)
                for l, (d3, d4, _, _) in grads.items(): acc[l][0] += d3.grad ** 2; acc[l][1] += d4.grad ** 2
        mse = float(((pred.detach() - torch.tensor(tp)) ** 2).mean()); lin = None
        np.savez(HERE / f'k34_jac_rank{r:04d}.npz', **{f'j{c}_{l}': (acc[l][i] / cfg['jac']).sqrt().numpy() for l in acc for i, c in ((0, 3), (1, 4))})
    else:
        with torch.no_grad(): pred = cd.run(False).numpy().astype(np.float64)[15]
        mse = float(((pred - tp) ** 2).mean()); lin = None
    rec = dict(rank=r, variant=vname, mse=mse, left=mse / base_mse if base_mse else 1.0, wall=round(time.time() - t0, 1), kerr=kerr, nunits=nunits)
    if lin is not None: rec['first_order_gain_frac'] = lin
    fo.write(json.dumps(rec) + '\n'); fo.flush(); print(json.dumps({k: rec[k] for k in ('rank', 'variant', 'left', 'wall')}), flush=True)
    return mse


with OUT.open('a') as fo:
    for r in ranks:
        w = s6bcard.weights(r); tp = np.load(P2 / 'fable_refit_v1/truth688' / f'truthpost_rank{r:04d}.npz')['mc_post'].astype(np.float64)[15]
        fT = [P2 / 'public_g19_attribution_v1' / f'mps_truth_rank{r:04d}_{n}.npz' for n in ('P32M', 'P8M')]; fT = [x for x in fT if x.exists()]
        TR = np.load(fT[0]); base = None
        for vname in VARIANTS:
            m = run(r, vname, w, tp, TR, fo, base)
            if vname == 'base': base = m
