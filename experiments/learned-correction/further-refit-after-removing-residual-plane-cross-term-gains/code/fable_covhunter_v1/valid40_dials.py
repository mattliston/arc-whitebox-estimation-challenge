#!/usr/bin/env python3
"""TAG fable-covhunter (CH-62): VALID40 of a dial set against a base dial set on the S18-hooked card (engineer's lrstar hook), learned ON,
native-consistent semantics (s6bcard_joint2), with given learned literals. For the alternating loop (dials <-> refit): base = the previous
round's state (its params and dials), fit = this round's params and dials. Rows use the s18_revalidate layout (rows[net] = base_s18,
fit_s18) for the porter. usage: valid40_dials.py <threads> <base params.pt> <base dials json | BASE> <fit params.pt> <fit dials json>
<tag> -> <tag>.json. v2 (v1 kept as valid40_dials_v1.py): env GL=1 also installs gn_linear_hook (native _gn_dcov). v3 (v2 kept as valid40_dials_v2_gl.py): env PS=1 installs
the loop-form pair series (E-67) on BOTH arms; the first log line states the card TAG. v4 (v3 kept as valid40_dials_v3_ps.py): env
STAR=s1zero (the physicist's lrstar_s1zero_hook; default lr32 = the engineer's lrstar) and JOIN=nystrom (join_nystrom_torch), both arms."""
import sys, json, time, os
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(P2 / 'fable_analyst_v1')); sys.path.insert(1, str(P2 / 'fable_engineer_v1'))
import numpy as np, torch
import s6bcard_joint2 as SJ, lrstar_torch_hook as lh, gn_linear_hook as gl, pair_series_torch as ps
GL = os.environ.get('GL') == '1'; PS = os.environ.get('PS') == '1'; STAR = os.environ.get('STAR', 'lr32'); JOIN = os.environ.get('JOIN')
import join_nystrom_torch as jn
sys.path.insert(2, str(P2 / 'fable_physicist_v1/refit6b')); import lrstar_s1zero_hook as lz
print(json.dumps(dict(TAG=f"card s6bcard_joint2{' + nystrom join' if JOIN else ''}{' + lrstar s1=0' if STAR == 's1zero' else ' + lrstar r32'}{' + gn_linear_hook' if GL else ''}{' + fine_ckpt=False + pair_series loop form' if PS else ''}; both arms")), flush=True)
torch.set_num_threads(int(sys.argv[1])); torch.set_grad_enabled(False)
BP, FP = Path(sys.argv[2]), Path(sys.argv[4]); BD = dict(SJ.BASE) if sys.argv[3] == 'BASE' else json.load(open(sys.argv[3]))
FD = dict(SJ.BASE) if sys.argv[5] == 'BASE' else json.load(open(sys.argv[5])); TAG = sys.argv[6]
inputs = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text()); nets = inputs['valid'][:40]
def literals(path):
    sv = torch.load(path); p, a, c, e = SJ.payload(torch)
    for k in p: p[k].copy_(sv['p'][k])
    a.copy_(sv['a']); c.copy_(sv['c'])
    for k in e: e[k].copy_(sv['e'][k])
    return p, a, c, e
LB, LF = literals(BP), literals(FP)
def run(D, lit, w, tp):
    SJ.DIALS = dict(D); cd = SJ.card(torch, w, *lit)
    if JOIN == 'nystrom': jn.install(cd, torch)
    if STAR == 's1zero': lz.install(cd, torch)
    else: lh.install(cd, torch)
    if GL: gl.install(cd, torch)
    if PS: ps.install(cd, torch)
    return float(((cd.run(False)[15].to(torch.float64) - tp) ** 2).mean())
rows = {}; t0 = time.time()
for r in nets:
    w = SJ.weights(r); tp = torch.tensor(SJ.truthpost(r)[15]).to(torch.float64)
    rows[r] = dict(base_s18=run(BD, LB, w, tp), fit_s18=run(FD, LF, w, tp)); print(json.dumps(dict(net=r, **rows[r], wall=round(time.time() - t0))), flush=True)
def summ(ns):
    b = np.array([rows[r]['base_s18'] for r in ns]); f = np.array([rows[r]['fit_s18'] for r in ns]); rel = f / b - 1
    boot = np.random.default_rng(0).integers(0, len(ns), (20000, len(ns))); bc = f[boot].sum(1) / b[boot].sum(1) - 1
    return dict(n=len(ns), fit_vs_base_pooled=float(f.sum() / b.sum() - 1), ci95=[float(np.percentile(bc, 2.5)), float(np.percentile(bc, 97.5))],
                wins=int((rel < 0).sum()), per_net_mean=float(rel.mean()), per_net_se=float(rel.std(ddof=1) / np.sqrt(len(ns))), base_mean=float(b.mean()), fit_mean=float(f.mean()))
S = dict(valid16=summ(nets[:16]), other24=summ(nets[16:]), valid40=summ(nets))
print(json.dumps(dict(summary=S)), flush=True)
(HERE / f'{TAG}.json').write_text(json.dumps(dict(gl=GL, ps=PS, star=STAR, join=JOIN, base_params=str(BP), fit_params=str(FP), base_dials=BD, fit_dials=FD, nets=nets, rows=rows, summary=S), indent=1))
