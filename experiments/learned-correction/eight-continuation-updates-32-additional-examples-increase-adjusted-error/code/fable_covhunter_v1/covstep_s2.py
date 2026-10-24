"""TAG fable-covhunter (CH-21): the S2 physics (fired #331947) on my HookCard, for knob fitting on the S2 card: K31 row factor inside the
pair map (all outputs, L1-14; the analyst's source patch of _nonlinear_pair_batch with _ROWF), the gain b-fix gamma 0.1 (partition,
mean m; HookCard knob 'bfix_gamma') and the every3 old-tier join schedule (the analyst's make_push). Parity with fable_analyst_v1/s2card.py
is checked by parity() (bit-exact expected).  make_s2(card, rowf=(c0, c1), gamma=0.1, every3=True)"""
import ast, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'fable_analyst_v1'))
import numpy as np, torch
import covstep as cs
import s2card                     # sets model_bank_bfix globals at import (does not touch model_bank_research)


def set_rowf(card, c0, c1):
    card.env['_ROWF'][0] = (lambda mu, k3, v, c0=c0, c1=c1: c0 + c1 * mu * k3 / (v * v))


def make_s2(card, rowf=(0.85, 0.5), gamma=0.1, every3=True):
    card.env['_ROWF'] = [None]; set_rowf(card, *rowf)
    t_ = cs.model.Translate().visit(ast.parse(s2card._src)); ast.fix_missing_locations(t_)
    exec(compile(t_, 's2pair', 'exec'), card.env)
    if every3: card.shared['push'] = s2card.make_push(card.shared['_qr'], s2card.EVERY3)
    card.knobs['bfix_gamma'] = gamma
    return card


def parity(rank=35):
    p, a, c, e = cs.model.torch_payload(torch, np, False)
    w = cs.weights(rank)
    with torch.no_grad():
        ref = s2card.card(torch, w, p, a, c, e).run(False)
        card = make_s2(cs.HookCard(torch, w, p, a, c, e))
        outs, _ = card.run_states()
        mine = torch.stack(outs).clamp_min(0).to(torch.float32)
    truth = cs.truthpost(rank)[15]
    return float((ref - mine).abs().max()), float(((ref[15].numpy().astype(np.float64) - truth) ** 2).mean()), float(((mine[15].numpy().astype(np.float64) - truth) ** 2).mean())


if __name__ == '__main__':
    torch.set_num_threads(3)
    print('S2 HookCard vs s2card: max abs diff, mse ref, mse mine =', parity(int(sys.argv[1]) if len(sys.argv) > 1 else 35))
