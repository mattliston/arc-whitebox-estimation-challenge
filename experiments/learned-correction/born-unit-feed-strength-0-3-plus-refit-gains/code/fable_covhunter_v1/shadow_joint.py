#!/usr/bin/env python3
"""TAG fable-covhunter (CH-55): SHADOW (additivity) test of the L1 joint dials (s6bcard_jfit1.py) against the learned parts, on the j05
card with the j05 refit's epoch-0 params. Per VALID16 net: m(ON, base), m(OFF, base), m(OFF, fit), m(ON, fit) (ON = learned parts on;
OFF = head/head_b/edge output/bias zeroed; base = the j05 card = s6bcard_joint BASE). Reports the learned-ON no-refit value of the fit,
the shadow S = m(ON, base) + [m(OFF, fit) - m(OFF, base)] and the pooled best shadow scale beta*. Certificate: m(OFF, fit) per net must
equal gnjoint_j05ep0_off.json valid16 'chosen' (same env) bit-exactly. usage: shadow_joint.py <threads> -> shadow_joint_n16.json"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(P2 / 'fable_analyst_v1'))
import numpy as np, torch
import s6bcard_joint as SJ, s6bcard_jfit1 as JF
torch.set_num_threads(int(sys.argv[1])); torch.set_grad_enabled(False)
inputs = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text()); nets = inputs['valid'][:16]
sv = torch.load(P2 / 'fable_physicist_v1/refit6b/s6b_B2ep1_j05_1ep/params_epoch0.pt')
REF = json.load(open(HERE / 'gnjoint_j05ep0_off.json'))['valid16']
def literals(off):
    p, a, c, e = SJ.payload(torch)
    for k in p: p[k].copy_(sv['p'][k])
    a.copy_(sv['a']); c.copy_(sv['c'])
    for k in e: e[k].copy_(sv['e'][k])
    if off:
        for k in ('head', 'head_b'): p[k].zero_()
        for k in ('output', 'bias'): e[k].zero_()
    return p, a, c, e
ON, OFF = literals(False), literals(True)
def base_card(lit, w): SJ.DIALS = dict(SJ.BASE); return SJ.card(torch, w, *lit).run(False)[15].to(torch.float64)
def fit_card(lit, w): return JF.card(torch, w, *lit).run(False)[15].to(torch.float64)
rows = {}; t0 = time.time()
for r in nets:
    w = SJ.weights(r); tp = torch.tensor(SJ.truthpost(r)[15]).to(torch.float64)
    on_b, off_b, off_f, on_f = base_card(ON, w), base_card(OFF, w), fit_card(OFF, w), fit_card(ON, w)
    P = off_f - off_b; e_on = on_b - tp; m = lambda x: float(((x - tp) ** 2).mean())
    rows[r] = dict(on_base=m(on_b), off_base=m(off_b), off_fit=m(off_f), on_fit=m(on_f), shadow=float(((e_on + P) ** 2).mean()),
                   eP=float((e_on * P).mean()), PP=float((P * P).mean()), cert=m(off_f) == REF['chosen'][str(r)])
    print(json.dumps(dict(net=r, **rows[r], wall=round(time.time() - t0))), flush=True)
tot = lambda k: sum(rows[r][k] for r in nets)
beta = -tot('eP') / tot('PP')
S = dict(cert_all=all(rows[r]['cert'] for r in nets), on_base_mean=tot('on_base') / 16, off_fit_vs_off_base=tot('off_fit') / tot('off_base') - 1,
         on_fit_vs_on_base=tot('on_fit') / tot('on_base') - 1, on_fit_wins=sum(rows[r]['on_fit'] < rows[r]['on_base'] for r in nets),
         shadow_vs_on_base=tot('shadow') / tot('on_base') - 1, shadow_wins=sum(rows[r]['shadow'] < rows[r]['on_base'] for r in nets),
         beta_star=beta, shadow_beta_star=(tot('on_base') + 2 * beta * tot('eP') + beta ** 2 * tot('PP')) / tot('on_base') - 1)
print(json.dumps(dict(summary=S)), flush=True)
(HERE / 'shadow_joint_n16.json').write_text(json.dumps(dict(nets=nets, rows=rows, summary=S), indent=1))
