#!/usr/bin/env python3
"""TAG fable-covhunter (CH-53): the lead's L1 = JOINT learned-OFF Gauss-Newton fit of the physics dials of s6bcard_joint.py on the j05 card
with the j05 refit's epoch-0 params (learned OFF = head/head_b/edge output/bias zeroed, the physicist's definition).
1. certificate: s6bcard_joint at BASE == s6bcard_j03 at J1 0.5 (= the physicist's j05 card) on the first fit net, bit-exact;
2. Jacobian of the final means by one forward-difference run per dial (steps H) on the fit nets (inputs.json train[:NF]);
3. GN step in step units, Levenberg-Marquardt damped (lam in {0, .01, .1, 1}), leave-one-net-out check, pick the lam with the best LOO;
4. verify the full and half step by real runs on the fit nets, keep the better;
5. VALID16 (learned OFF, or ON with 'on'): BASE (= the j05 control in that mode) vs the chosen dials, per net.
usage: gn_joint_fit.py <n_fit> <threads> <tag> [on] -> <tag>.json   ('on' keeps the learned parts ON: the learned-ON joint fit;
default OFF = v1, kept as gn_joint_fit_v1_off.py); a 5th arg 'shared' uses s6bcard_joint2 (native-consistent k4 / c0c1 semantics, CH-60;
v2 without it kept as gn_joint_fit_v2_on.py). v4 env options (v3 kept as gn_joint_fit_v3_shared.py): PARAMS=<.pt> learned literals
(default the j05 refit's epoch 0); STAR=lr32 installs the engineer's S18 low-rank star hook on every card; BASE_DIALS=<json> is the
linearization point and the VALID16 comparison base (default SJ.BASE; the alternating loop's previous-round dials). v5 (v4 kept as
gn_joint_fit_v4_params_star.py): GL=1 also installs the engineer's gn_linear_hook (native _gn_dcov; exact on joint2 only). v6 (v5 kept as
gn_joint_fit_v5_gl.py): PS=1 installs the engineer's loop-form pair series (E-67, pair_series_torch; rounding-level, -18%/forward);
the run log's first line states the card TAG (fine_ckpt=False + pair_series loop form when PS=1, as the lead requires). v7 (v6 kept as
gn_joint_fit_v6_ps.py): STAR=s1zero installs the physicist's lrstar_s1zero_hook (S23: the star's s1 term dropped) instead of lrstar;
JOIN=nystrom installs the engineer's join_nystrom_torch (S23's Nystrom old-tier join) right after the card build."""
import sys, json, time, os
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(P2 / 'fable_analyst_v1'))
import numpy as np, torch
import s6bcard_j03 as J03
if len(sys.argv) > 5 and sys.argv[5] == 'shared':
    import s6bcard_joint2 as SJ
else:
    import s6bcard_joint as SJ
NF = int(sys.argv[1]); torch.set_num_threads(int(sys.argv[2])); TAG = sys.argv[3]; torch.set_grad_enabled(False)
LEARNED_ON = len(sys.argv) > 4 and sys.argv[4] == 'on'
inputs = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())
FIT, VAL = inputs['train'][:NF], inputs['valid'][:16]
PAR = Path(os.environ.get('PARAMS', str(P2 / 'fable_physicist_v1/refit6b/s6b_B2ep1_j05_1ep/params_epoch0.pt')))
STAR = os.environ.get('STAR'); GL = os.environ.get('GL') == '1'; PS = os.environ.get('PS') == '1'; JOIN = os.environ.get('JOIN'); BASE_D = json.load(open(os.environ['BASE_DIALS'])) if os.environ.get('BASE_DIALS') else dict(SJ.BASE)
if STAR or GL or PS or JOIN:
    sys.path.insert(1, str(P2 / 'fable_engineer_v1')); import lrstar_torch_hook as lh, gn_linear_hook as gl, pair_series_torch as ps, join_nystrom_torch as jn
    sys.path.insert(2, str(P2 / 'fable_physicist_v1/refit6b')); import lrstar_s1zero_hook as lz
assert not GL or SJ.__name__ == 's6bcard_joint2', 'gn_linear_hook is exact only on the native-consistent joint2 card'
def hooked(cd):
    if JOIN == 'nystrom': jn.install(cd, torch)
    if STAR == 'lr32': lh.install(cd, torch)
    if STAR == 's1zero': lz.install(cd, torch)
    if GL: gl.install(cd, torch)
    if PS: ps.install(cd, torch)
    return cd
H = dict(X1=0.05, X0=0.08, X2=0.08, X3=0.08, bscale=0.02, s_carry=0.005, s_new=0.02, c0=0.05, c1=0.1, k4a=0.1, k4b=0.1, k4c=0.1)
KEYS = list(H)
p, a, c, e = SJ.payload(torch); sv = torch.load(PAR)
for k in p: p[k].copy_(sv['p'][k])
a.copy_(sv['a']); c.copy_(sv['c'])
for k in e: e[k].copy_(sv['e'][k])
if not LEARNED_ON:
    for k in ('head', 'head_b'): p[k].zero_()
    for k in ('output', 'bias'): e[k].zero_()
LIT = (p, a, c, e); OUT = dict(tag=TAG, fit=FIT, steps=H, params=str(PAR), learned='on' if LEARNED_ON else 'off', module=SJ.__name__, star=STAR, gl=GL, ps=PS, join=JOIN, base_dials=BASE_D); t0 = time.time()
def say(**kw): print(json.dumps(dict(**kw, wall=round(time.time() - t0))), flush=True)
say(TAG=f"card {SJ.__name__}{' + nystrom join' if JOIN else ''}{' + lrstar r32' if STAR == 'lr32' else ''}{' + lrstar s1=0' if STAR == 's1zero' else ''}{' + gn_linear_hook' if GL else ''}{' + fine_ckpt=False + pair_series loop form' if PS else ''}; "
        f"learned {'ON' if LEARNED_ON else 'OFF'}; params {PAR.name} ({PAR.parent.name}); base dials {'BASE' if BASE_D == SJ.BASE else os.environ.get('BASE_DIALS')}")
def run(dials, nets):
    SJ.DIALS = dict(dials); out = {}
    for r in nets:
        out[r] = hooked(SJ.card(torch, SJ.weights(r), *LIT)).run(False)[15].to(torch.float64)
    return out
T = {r: torch.tensor(SJ.truthpost(r)[15]).to(torch.float64) for r in FIT + VAL}
mse = lambda M: {r: float(((M[r] - T[r]) ** 2).mean()) for r in M}
pool = lambda a_, b_: sum(a_.values()) / sum(b_.values()) - 1
# 1. certificate
r0 = FIT[0]; J03.J1 = 0.5; SJ.DIALS = dict(SJ.BASE)
ref = hooked(J03.card(torch, SJ.weights(r0), *LIT)).run(False); mine = hooked(SJ.card(torch, SJ.weights(r0), *LIT)).run(False)
OUT['cert_bit_exact'] = bool(torch.equal(ref, mine)); say(cert_bit_exact=OUT['cert_bit_exact'], net=r0)
assert OUT['cert_bit_exact']
# 2. Jacobian
base = run(BASE_D, FIT); mb = mse(base); OUT['base_mse'] = mb; say(base_pooled=sum(mb.values()) / NF)
D = {}
for k in KEYS:
    d = dict(BASE_D); d[k] += H[k]; M = run(d, FIT); D[k] = {r: M[r] - base[r] for r in FIT}
    OUT.setdefault('fd', {})[k] = pool(mse(M), mb); say(dial=k, h=H[k], fd_ratio=OUT['fd'][k])
# 3. GN in step units
R = {r: (base[r] - T[r]).numpy() for r in FIT}; A = {r: np.stack([D[k][r].numpy() for k in KEYS], 1) for r in FIT}
def solve(nets, lam):
    G = sum(A[r].T @ A[r] for r in nets); g = sum(A[r].T @ R[r] for r in nets)
    return -np.linalg.solve(G + lam * np.diag(np.diag(G)), g)
pred = lambda x, nets: sum(float(np.mean((R[r] + A[r] @ x) ** 2)) for r in nets) / sum(float(np.mean(R[r] ** 2)) for r in nets) - 1
best = None
for lam in (0.0, 0.01, 0.1, 1.0):
    x = solve(FIT, lam); loo = []
    for i in FIT:
        xi = solve([r for r in FIT if r != i], lam); loo.append((float(np.mean((R[i] + A[i] @ xi) ** 2)), float(np.mean(R[i] ** 2))))
    loo_pooled = sum(u for u, _ in loo) / sum(v for _, v in loo) - 1
    rec = dict(lam=lam, in_sample=pred(x, FIT), loo=loo_pooled, loo_wins=sum(u < v for u, v in loo), dial_delta={k: float(x[i] * H[k]) for i, k in enumerate(KEYS)})
    OUT.setdefault('gn', []).append(rec); say(**rec)
    if best is None or loo_pooled < best[0]: best = (loo_pooled, lam, x)
# 4. verify full and half step
cand = {}
for f in (1.0, 0.5):
    d = {k: BASE_D[k] + f * float(best[2][i]) * H[k] for i, k in enumerate(KEYS)}; M = mse(run(d, FIT))
    cand[f] = (pool(M, mb), d); say(verify_step=f, lam=best[1], fit_ratio=cand[f][0], wins=sum(M[r] < mb[r] for r in FIT))
f = min(cand, key=lambda z: cand[z][0]); OUT['chosen'] = dict(step=f, lam=best[1], fit_ratio=cand[f][0], dials=cand[f][1])
# 5. VALID16 learned OFF
vb, vc = mse(run(BASE_D, VAL)), mse(run(cand[f][1], VAL))
rel = np.array([vc[r] / vb[r] - 1 for r in VAL])
OUT['valid16'] = dict(base=vb, chosen=vc, pooled=pool(vc, vb), wins=int((rel < 0).sum()), per_net_mean=float(rel.mean()), per_net_se=float(rel.std(ddof=1) / 4))
say(valid16_pooled=OUT['valid16']['pooled'], wins=OUT['valid16']['wins'], base_mean=sum(vb.values()) / 16, chosen_mean=sum(vc.values()) / 16)
(HERE / f'{TAG}.json').write_text(json.dumps(OUT, indent=1))
