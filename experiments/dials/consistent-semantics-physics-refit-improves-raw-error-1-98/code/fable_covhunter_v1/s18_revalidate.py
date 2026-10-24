#!/usr/bin/env python3
"""TAG fable-covhunter (CH-59): re-validate the LEARNED-ON joint dials (s6bcard_jfitON1) on the S18 physics (the engineer's low-rank star
hook, lrstar_torch_hook.install) before shipping. j05 epoch-0 params, learned ON, VALID40: per net, base = the j05 card (s6bcard_joint BASE)
+ hook, fit = the frozen ON dials + hook. Also: hooked base vs the physicist's no-hook j05 VALID40 (3 threads) = the S18 hook's own raw
effect (lead: S18 raw +0.23% on VALID40), as a sanity check. usage: s18_revalidate.py <threads> [fit module, default s6bcard_jfitON1] -> s18_revalidate_n40.json (or s18_revalidate_<module>_n40.json)
v2: the fit module is an argument (v1 kept as s18_revalidate_v1.py)"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(P2 / 'fable_analyst_v1')); sys.path.insert(1, str(P2 / 'fable_engineer_v1'))
import numpy as np, torch
import importlib
import s6bcard_joint as SJ, lrstar_torch_hook as lh
MOD = sys.argv[2] if len(sys.argv) > 2 else 's6bcard_jfitON1'; JON = importlib.import_module(MOD)
torch.set_num_threads(int(sys.argv[1])); torch.set_grad_enabled(False)
inputs = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text()); nets = inputs['valid'][:40]
sv = torch.load(P2 / 'fable_physicist_v1/refit6b/s6b_B2ep1_j05_1ep/params_epoch0.pt')
p, a, c, e = SJ.payload(torch)
for k in p: p[k].copy_(sv['p'][k])
a.copy_(sv['a']); c.copy_(sv['c'])
for k in e: e[k].copy_(sv['e'][k])
NOHOOK = json.load(open(P2 / 'fable_physicist_v1/refit6b/s6b_B2ep1_j05_1ep_valid40/eval_n40.json'))['valid']
JON.STAR = 'lr32'; rows = {}; t0 = time.time()
for r in nets:
    w = SJ.weights(r); tp = torch.tensor(SJ.truthpost(r)[15]).to(torch.float64)
    SJ.DIALS = dict(SJ.BASE); cb = SJ.card(torch, w, p, a, c, e); lh.install(cb, torch)
    mb = float(((cb.run(False)[15].to(torch.float64) - tp) ** 2).mean())
    mf = float(((JON.card(torch, w, p, a, c, e).run(False)[15].to(torch.float64) - tp) ** 2).mean())
    rows[r] = dict(base_s18=mb, fit_s18=mf, base_s17_nohook=NOHOOK[str(r)])
    print(json.dumps(dict(net=r, **rows[r], wall=round(time.time() - t0))), flush=True)
def summ(ns):
    b = np.array([rows[r]['base_s18'] for r in ns]); f = np.array([rows[r]['fit_s18'] for r in ns]); n0 = np.array([rows[r]['base_s17_nohook'] for r in ns])
    rel = f / b - 1; se = rel.std(ddof=1) / np.sqrt(len(ns))
    boot = np.random.default_rng(0).integers(0, len(ns), (20000, len(ns))); bc = f[boot].sum(1) / b[boot].sum(1) - 1
    return dict(n=len(ns), fit_vs_base_pooled=float(f.sum() / b.sum() - 1), ci95=[float(np.percentile(bc, 2.5)), float(np.percentile(bc, 97.5))],
                wins=int((rel < 0).sum()), per_net_mean=float(rel.mean()), per_net_se=float(se), hook_effect_pooled=float(b.sum() / n0.sum() - 1),
                base_s18_mean=float(b.mean()), fit_s18_mean=float(f.mean()))
S = dict(valid16=summ(nets[:16]), valid40=summ(nets), other24=summ(nets[16:]))
print(json.dumps(dict(summary=S)), flush=True)
(HERE / ('s18_revalidate_n40.json' if MOD == 's6bcard_jfitON1' else f's18_revalidate_{MOD}_n40.json')).write_text(json.dumps(dict(nets=nets, rows=rows, summary=S), indent=1))
