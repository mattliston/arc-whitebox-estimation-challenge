#!/usr/bin/env python3
"""TAG analyst A20 (2026-10-02): ERROR BUDGET OF THE S40 TORCH CARD, before and after the pair-carried law (lead's request of 14:40Z).
Card = the trainer's pool card (O/trainer/otr_engine.py, card M with the exact-term module) at candidate 3
(O/trainer/cert/mix3/mix3b_optimum_min5.pt), law 0 = candidate 3 alone, law 1 = + {'module': 'ncar6', 'forms': 3} (frozen copy
O/trainer/frozen/ncar6_t1339.py: level 'read', dfit 2, ca 0.75, arest False) = the trainer's judged arm a_c3q2zn6f3. No refit, forward only.

PART 1 (the physicist's T8 on this card, O/physicist/t8b_budget.py): one input of the per-unit mean formula replaced by Monte Carlo truth at
  layers 1-8 together (one arm) and at ONE layer L = 9..15 at a time (through the 'delta' hook: the shift is carried forward and the covariance is
  recentred on it; learned correction ON):
  v (truth variance, two independent 32M draws A and B), s (truth skew at the card's variance, 64M), k (truth kurtosis, 64M); plus the
  every-layer arms v A / v B / s / k / vsk A / vsk B. The final-layer prediction VECTOR of every arm is kept.
PART 2 (the physicist's T1, in CLOSED LOOP): at the pair map of layer l = 11..14 (it makes the covariance whose pushed diagonal is the
  per-unit variance of layer l + 1 = 12..15) ONE input is replaced by truth, all others the card's own:
     C (pre-activation covariance, 32M), mu (pre-activation mean, 32M), K21k / K21o (third-order plane, rows with |mu/sigma| < 1.5 / the other
     rows, 8M), K31 (8M), K22 (off-diagonal, 8M), k3, k4 (per-unit, 64M), PL (K21 + K31 + K22 together).
  The card's planes are taken as the pair map really reads them (with the covariance-output knobs iso / k21n / k31 and, with the law, the fold
  inside the bank read, which is K31 content: it stays in when K21 is replaced and is removed when K31 is replaced), in natural units.
  The change of the covariance output D_X = pair(inputs with X at truth) - pair(the card's inputs) is added to the card's own output.
  Kept per arm: the one-step change of the next layer's per-unit variance diag(W' D_X W), and the final-layer prediction when the change is
  carried to the end (closed loop).
PART 3: K21 / K31 / K22 / all three replaced by truth at EVERY layer 2..14 in closed loop (the 8M planes add their own noise at every layer:
  read the linear shares, not the MSE).
Mechanics: every arm restarts from the stored state of the base run before its first changed layer; the side tables of the modules (K4L and
exact-term births, the pair law's column weights) are restored from the base run. CERTIFICATES printed: restarts without a change reproduce
the base prediction (before and after the arms), the mean formula reproduces the card's host mean, the single-call pair map reproduces the
card's covariance output, and the all-truth one-step floor of layer 13 (T1 found 0.017% on 35 and 100).
  It also keeps the arm's per-unit variance at layers 3..15 (key allpm_<X>_v), for the share of the VARIANCE error each plane carries cumulatively.
usage: a20_budget.py <rank> <threads> <tag> <card> [parity|full|part3|part4|late]   -> a20_budget_rank{rank}_{tag}.npz, log on stdout (JSON lines)
  card: 0 = candidate 3, 1 = candidate 3 + the pair law (the S40 torch card), or a JSON file {params, ncar, ss, frozen, sha, ref} for any card of the trainer's pool
        (params: the parameter file relative to O; ncar, ss: the request's fields; frozen, sha: the folder of my lane holding the frozen otr_m.py and its sha256;
        ref: the trainer's stored judgement to compare the base run with, e.g. a_c3q4zj7judge). The S42 torch card: a20_card_s42.json.
  parity: base run only, compared with the trainer's stored judgement of the same rank when it exists (run_M/results).
  part3: base run, certificates and PART 3 only.
  part4: base run, certificates and PART 4: K31 / K21 kink rows / all three planes exact on the pair-layer bands 11-14, 8-10, 2-7 (keys band<name>_<X>), and, where a
         second independent 8M draw of the planes exists (network 35: layers 2, 5, 9, 12, 15), the single-layer arms of pair layer 12 with that draw (keys pm12n2_<X>):
         the product of the two draws' changes is free of plane noise.
  late (v6, lead 10-10 04:20Z, the real S42 card a21_card_s42.json with its head columns hf / hu): base run, certificates and the LATE arms: truth at ONE layer
         L = 12, 13, 14, 15 and on the band 12-15, closed loop, learned parts on: v / s / k / vsk in the mean formula only (as PART 1), and V = the per-unit variance
         at truth for EVERY consumer of the layer (the card's pre-activation covariance with its diagonal at truth, correlations kept, through the card's hook
         'cpre'; at layer 15 the variance through 'vv15'): what more compute at that layer could buy at most. v, vsk, V with draws A and B (keys late<L>_<arm>_<A|B>),
         the change of the final-layer MSE from the product of the two draws' changes (free of truth noise); s, k with the mean of both (64M)."""
import json, os, sys, time, math
from pathlib import Path
HERE = Path(__file__).resolve().parent; O = HERE.parent; P2 = O.parent
rank = int(sys.argv[1]); threads = str(int(sys.argv[2])); tag = sys.argv[3]; card = sys.argv[4]; mode = sys.argv[5] if len(sys.argv) > 5 else 'full'
_SHA1 = '8f8ea5bc378d31ce71a0101cb864f827677ba79e4bd03aa25f07f3884a789ef2'
CARDS = {'0': dict(params='trainer/cert/mix3/mix3b_optimum_min5.pt', ncar=None, ss=None, frozen='a20_frozen', sha=_SHA1, ref='a_c3min5'),
         '1': dict(params='trainer/cert/mix3/mix3b_optimum_min5.pt', ncar={'module': 'ncar6', 'forms': 3}, ss=None, frozen='a20_frozen', sha=_SHA1, ref='a_c3q2zn6f3')}
SPEC = CARDS[card] if card in CARDS else json.loads(Path(card).read_text())      # a card of the trainer's pool: parameter file, the request's 'ncar' and 'ss' fields, frozen otr_m.py
law = SPEC['ncar'] is not None                                                     # any carrier module installed (the pair law, K31L, ...)
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'TRAINER_THREADS'): os.environ[key] = threads
os.environ.update(TRAINER_PL='1', TRAINER_CARD='M', TRAINER_F4C='1')
sys.path = [x for x in sys.path if os.path.abspath(x or os.getcwd()) != str(HERE)]      # nothing of the card may resolve to a file of my lane
sys.path.insert(0, str(O / 'trainer'))
sys.path.insert(0, str(HERE / SPEC['frozen']))   # the trainer's card module otr_m.py, frozen (a20_frozen: 15:34Z, sha 8f8ea5bc; a20_frozen_4d219986: 16:41Z): the live file changes while the pool works
import hashlib
import numpy as np
import otr_engine as E
assert Path(E.M.__file__).resolve().parent == HERE / SPEC['frozen'], E.M.__file__
assert hashlib.sha256(Path(E.M.__file__).read_bytes()).hexdigest().startswith(SPEC['sha'])
from otr_common import load_params, load_pl, data_paths
torch = E.torch; torch.set_grad_enabled(False); t0 = time.time(); N = 1024
OUT = HERE / f'a20_budget_rank{rank:04d}_{tag}.npz'; assert mode == 'parity' or not OUT.exists(), f'{OUT} exists: choose another tag'
PARAMS = Path(SPEC['params']) if SPEC['params'].startswith('/') else O / SPEC['params']
NCAR = SPEC['ncar']
say = lambda **rec: print(json.dumps(dict(rec, wall=round(time.time() - t0, 1))), flush=True)
T64 = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64))

E.set_parameters(*load_params(torch, PARAMS))
shifts = {f_: list(v_) for f_, v_ in load_pl(torch, PARAMS).items()}
for f_, v_ in (SPEC.get('pl') or {}).items(): shifts[f_] = [a + b for a, b in zip(shifts.get(f_, [0.0] * 16), v_)]
E.PL.set_values(shifts)
E.M.F4C = bool(SPEC.get('f4c', 1)); E.M.NCAR = NCAR; E.STAR_ATTACH = False; E.STAR_SS = SPEC.get('ss'); E.LC.JT.DIALS_T = E.DL
if 'hf' in SPEC:   # the S42 card's head columns (request fields hf / hu) and readout, set as a21_dump.py sets them
    E.M.HEADF = int(SPEC['hf'] or 0); E.M.ROWS.clear(); E.M.RG = bool(SPEC.get('rg'))
    if E.M.RG or E.M.RG_W: E.M.readout_weights(SPEC.get('rgw'))
    if E.M.HEADF or E.M.HEADU is not None: E.M.head_weights(SPEC.get('hu'))
weights_path, truth_path = data_paths(rank)
W = np.load(weights_path, allow_pickle=False); tp = np.load(truth_path)['mc_post'].astype(np.float64)
cd = E.LC.card(torch, W, E.p, E.T['a'], E.T['c'], E.e, fine_ckpt=False)
cd.hooks['k4_15'] = lambda layer, value, ctx: value
E.install_star(cd)
if E.STAR_SS is not None: E._star_ss().install(cd, torch, E.STAR_SS)      # the engineer's star by row sums (request field 'ss'), as otr_engine.run_network installs it
E.PL.install(E, cd); E.M.install(cd)
cell = lambda fn, name: fn.__closure__[fn.__code__.co_freevars.index(name)].cell_contents
SLIST = []; read6 = read0 = SEEN = None
if law:   # the carriers' own states (the pair law's column weights) and the bank read under their folds, taken out of the modules' closures
    SLIST = [cell(o, 'S') for _, o in cell(cd.hooks['state_out'], 'parts') if o is not None and 'S' in o.__code__.co_freevars and {'w', 'x', 'partner'} <= set(cell(o, 'S'))]
    read6 = cd.shared['read']
    read0 = cell(read6, 'read0') if 'read0' in read6.__code__.co_freevars else cell(read6, 'read_plain')     # the trainer's replay wrapper (15:34Z) sits on the module's read
    SEEN = cell(cd.step, 'seen')     # the trainer's replay switch: a layer that ran before takes the modules' output from a record; my restarts are fresh runs, so the set is emptied
    assert isinstance(SEEN, set)


def wrap(name, fn):
    old = cd.hooks.get(name)
    cd.hooks[name] = fn if old is None else (lambda layer, value, ctx: fn(layer, old(layer, value, ctx), ctx))


NS = cd.env['nonlinear_scalar']; stash = {}; dev = []; REC = {}; S6SNAP = {-1: [dict(w=None, x=None, partner=None) for _ in SLIST]}
ctrl = dict(layers=(), arm='', tset=None, record=False); PM = dict(layer=None)
wrap('k4_15', lambda layer, value, ctx: stash.__setitem__('k4', value) or value)
wrap('k3_15', lambda layer, value, ctx: stash.__setitem__('k3', value) or value)


def delta(layer, d, ctx):
    host, mu = ctx['host'], ctx['mu']
    if layer == 15: v, k3, k4 = ctx['vv'], stash['k3'], stash['k4']
    else: v = torch.diagonal(ctx['Cpre']); k3 = torch.diagonal(ctx['k21']); k4 = ctx['k4']
    k4 = k4 if k4.ndim else k4.expand(N)
    if ctrl['record']: REC[layer] = dict(v=v.clone(), k3=k3.clone(), k4=k4.clone(), mu=mu.clone(), host=host.clone())
    PM['v_seen'] = {**PM.get('v_seen', {}), layer: v.clone()}
    if layer == 0 or layer not in ctrl['layers'] or not ctrl['arm'] or ctrl['arm'] == 'V': return d     # arm V: the card's own formula already reads the variance at truth
    dev.append(float((NS(mu, v, k3, k4)[0] - host).abs().max()))
    tt = TS[ctrl['tset']]; tv, tk3, tk4 = tt['v'][layer], tt['k3'][layer], tt['k4'][layer]; arm = ctrl['arm']
    v2 = tv if 'v' in arm else v
    g1 = tk3 / tv ** 1.5 if 's' in arm else k3 / v ** 1.5
    g2 = tk4 / tv ** 2 if 'k' in arm else k4 / v ** 2
    return d + (NS(mu, v2, g1 * v2 ** 1.5, g2 * v2 ** 2)[0] - host)


wrap('delta', delta)
CALLS = dict(cpre=set(), vv15=set())


def cpre_v(layer, C, ctx):      # arm V: the pre-activation covariance with its diagonal at truth, correlations kept, for every consumer of the layer
    CALLS['cpre'].add(layer)
    if layer not in ctrl['layers'] or ctrl['arm'] != 'V': return C
    r = (TS[ctrl['tset']]['v'][layer].to(C.dtype) / torch.diagonal(C)).sqrt()
    return C * r[:, None] * r[None, :]


wrap('cpre', cpre_v)
wrap('vv15', lambda layer, vv, ctx: CALLS['vv15'].add(layer) or (TS[ctrl['tset']]['v'][15].to(vv.dtype) if 15 in ctrl['layers'] and ctrl['arm'] == 'V' else vv))
active = lambda l: PM['layer'] is not None and (PM['layer'] == 'all' or PM['layer'] == l or (isinstance(PM['layer'], tuple) and l in PM['layer']))   # is a pair-map arm on at layer l
wrap('pair_in', lambda layer, pin, ctx: PM.__setitem__('pin', pin) or pin)
if law:
    def read_cap(tt, pp, lam, response_indices, indices, U, F, old_columns):
        plane = read6(tt, pp, lam, response_indices, indices, U, F, old_columns)
        if active(cd.call_layer) and PM.get('mode') in ('collect', 'live'):
            plain = read0(tt, pp, lam, response_indices, indices, U, F, old_columns)
            Ff = torch.zeros((N, N), dtype=torch.float64).index_put((response_indices[:, None], indices[None, :]), (plane - plain).to(torch.float64))
            PM['fold'] = Ff - torch.diag(torch.diagonal(Ff))
        return plane
    cd.shared['read'] = read_cap
    wrap('state_out', lambda layer, out, ctx: S6SNAP.__setitem__(layer, [dict(w=S['w'], x=S['x'], partner=S['partner']) for S in SLIST]) or out if ctrl['record'] else out)
eye = torch.eye(N, dtype=torch.bool); offm = ~eye
ARMS = ('C', 'mu', 'K21k', 'K21o', 'K31', 'K22', 'k3', 'k4', 'PL')


def pm_differences(l, C, ctx, arms, truth_state=False):
    """D_X = pair(the card's inputs with X at truth) - pair(the card's inputs), for X in arms; inputs as the pair map of layer l reads them."""
    mu, Cp, K21p, k4p, K22p, rem = PM['pin']; pair = ctx['pair']; rowf = cd.env['_ROWF'][0]
    assert float((ctx['k21'] - K21p).abs().max()) == 0.0 and float((ctx['Cpre'] - Cp).abs().max()) == 0.0
    v = torch.diagonal(Cp); s = v.sqrt(); a = mu / s; kink = (a.abs() < 1.5).double(); ss = torch.outer(s, s); rho = (Cp / ss).masked_fill(eye, 0.)
    k3c = torch.diagonal(K21p); k4c = k4p.expand(N) if k4p.ndim == 0 else k4p
    eps, V3 = E.PL.EPS, E.LC.JT.gn3.V3
    iso, k21n, k31 = [float(V3[k] + eps[k][l]) if 2 <= l <= 14 else 0.0 for k in ('iso', 'k21n', 'k31')]
    K21c = torch.diagonal_scatter(K21p + (iso * kink * k3c / v)[:, None] * Cp + (k21n * (1 - kink))[:, None] * K21p, k3c)
    T31c = ((1 + k31) * (k4c / v ** 2) * rowf(mu, k3c, v))[:, None] * rho
    if rem is not None: T31c = T31c + rem.to(torch.float64).masked_fill(eye, 0.)
    K31c = T31c * (v * s)[:, None] * s[None, :]
    k3t, k4t = TS['AB']['k3'][l], TS['AB']['k4'][l]
    K21t = torch.diagonal_scatter(T64(PLN[f'K21_{l}']), k3t); K22t = torch.diagonal_scatter(T64(PLN[f'K22_{l}']), k4t); K31t = T64(PLN[f'K31_{l}']).masked_fill(eye, 0.)
    fold = PM.get('fold') if law else None
    fold = torch.zeros((N, N), dtype=torch.float64) if fold is None else fold

    def call(mu, C, K21, k4, K22, K31):
        v_ = torch.diagonal(C); s_ = v_.sqrt(); rho_ = (C / torch.outer(s_, s_)).masked_fill(eye, 0.)
        T31 = (K31 / ((v_ * s_)[:, None] * s_[None, :])).masked_fill(eye, 0.)
        internal = ((k4 / v_ ** 2) * rowf(mu, torch.diagonal(K21), v_))[:, None] * rho_                 # what the pair map builds itself from (k4, k3)
        return pair(mu, C, K21, k4, K22, 'covariance', remainder31=(T31 - internal).masked_fill(eye, 0.))[1]

    card = dict(mu=mu, C=Cp, K21=K21c, k4=k4c, K22=K22p, K31=K31c)

    def swap(X):
        r = dict(card)
        if X in ('K21k', 'K21o'):
            msk = ((kink if X == 'K21k' else 1 - kink)[:, None] * torch.ones(1, N, dtype=torch.float64)).masked_fill(eye, 0.)
            r['K21'] = K21c * (1 - msk) + (K21t + fold) * msk
        elif X == 'K21': r['K21'] = torch.diagonal_scatter(K21t + fold, k3c)
        elif X == 'K31': r['K31'] = K31t; r['K21'] = K21c - fold
        elif X == 'K22': r['K22'] = torch.where(eye, K22p, K22t)
        elif X == 'k4': r['k4'] = k4t; r['K22'] = torch.diagonal_scatter(K22p, torch.diagonal(K22t))
        elif X == 'k3': r['K21'] = torch.diagonal_scatter(K21c, k3t)
        elif X == 'C': r['C'] = T64(MC[f'cov_pre_{l}'])
        elif X == 'mu': r['mu'] = T64(MC[f'mean_pre_{l}'])
        elif X == 'PL': r['K21'] = torch.diagonal_scatter(K21t, k3c); r['K31'] = K31t; r['K22'] = torch.where(eye, K22p, K22t)
        elif X == 'AT': r = dict(mu=mu, C=Cp, K21=K21t, k4=k4t, K22=K22t, K31=K31t)
        else: raise ValueError(X)
        return r

    if truth_state: return call(**swap('AT'))
    base = call(**card); D = {X: call(**swap(X)) - base for X in arms}
    info = dict(single_call_vs_card=float((base - C)[offm].norm() / C[offm].norm()), n_kink=int(kink.sum()),
                fold_over_K21=float(fold[offm].norm() / K21c[offm].norm()), knobs=[iso, k21n, k31],
                size={X: float(D[X][offm].norm() / C[offm].norm()) for X in arms})
    return D, info


def cov_pm(l, C, ctx):
    if not active(l) or not 2 <= l <= 14: return C
    if PM['mode'] == 'inject': return C + PM['D']
    if PM['mode'] == 'floor': PM['AT'] = pm_differences(l, C, ctx, (), truth_state=True); return C
    if PM['mode'] == 'collect': PM['DX'], PM['info'] = pm_differences(l, C, ctx, ARMS); return C
    D, info = pm_differences(l, C, ctx, (PM['X'],))          # 'live': the replacement made on the arm's own state at every layer
    return C + D[PM['X']]


wrap('cov_pm', cov_pm)
BASE = [cd.initial()]; M0 = {}; F0 = {}


def restore(start):
    cd.m_side['base'] = dict(M0['base']); cd.m_side['births'] = dict(M0['births']); cd.f4b_side['B'] = dict(F0)
    if law:
        for S, snap in zip(SLIST, S6SNAP[start - 1]): S.update(snap)
        SEEN.clear()


def final(start, state=None, stop=16):
    """the final-layer prediction of a run restarted before step `start` from the base run's state (side tables restored)."""
    restore(start); state = BASE[start] if state is None else state
    for layer in range(start, stop):
        try: state = cd.step(layer, state)
        finally: E.collect()
    return state[0].clamp_min(0).to(torch.float32).numpy().astype(np.float64) if stop == 16 else state


# ---- base run
ctrl.update(record=True); state = BASE[0]; preds = []
for layer in range(16):
    try: state = cd.step(layer, state)
    finally: E.collect()
    BASE.append(state); preds.append(state[0].clamp_min(0).to(torch.float32).numpy().astype(np.float64))
ctrl.update(record=False); preds = np.stack(preds); base = preds[15]; e0 = base - tp[15]; EE = float(e0 @ e0)
M0 = dict(base=dict(cd.m_side['base']), births=dict(cd.m_side['births'])); F0 = dict(cd.f4b_side['B'])
mse_layers = ((preds - tp) ** 2).mean(1)
out = dict(truth=tp[15], base=base, pred_all=preds, card_v=torch.stack([REC[l]['v'] for l in range(16)]).numpy(), card_k3=torch.stack([REC[l]['k3'] for l in range(16)]).numpy(),
           card_k4=torch.stack([REC[l]['k4'] for l in range(16)]).numpy(), card_mu=torch.stack([REC[l]['mu'] for l in range(16)]).numpy(),
           card_host=torch.stack([REC[l]['host'] for l in range(16)]).numpy())
ref = O / 'trainer/run_M/results' / (SPEC['ref'] + '_r%04d.json' % rank)
refrec = json.loads(ref.read_text()) if ref.exists() else None
say(TAG='analyst A20 budget', rank=rank, tag=tag, card=card, law=law, ncar=NCAR, ss=SPEC.get('ss'), otr_m=SPEC['sha'][:8], carrier_states=len(SLIST), params=str(PARAMS), threads=threads, mode=mode, final_x1e9=round(EE / N * 1e9, 5),
    layers_12_15_x1e9=[round(float(x) * 1e9, 4) for x in mse_layers[12:]],
    trainer_stored_final_x1e9=None if refrec is None else round(refrec['mse'] * 1e9, 5),
    parity_rel=None if refrec is None else float(abs(EE / N / refrec['mse'] - 1)),
    parity_layers_max_rel=None if refrec is None else float(np.max(np.abs(mse_layers / np.array(refrec['mse_layers']) - 1))),
    hook_calls={k: sorted(v) for k, v in CALLS.items()}, head=getattr(E.M, 'HEADF', None))
if mode == 'parity': sys.exit(0)
assert mode != 'late' or ({12, 13, 14} <= CALLS['cpre'] and CALLS['vv15'] == {15}), CALLS     # the card has the hook points arm V needs

# ---- truths
PUB = P2 / 'public_g19_attribution_v1'
A, B = np.load(PUB / f'mps_truth_rank{rank:04d}_P32M.npz'), np.load(PUB / f'mps_truth_rank{rank:04d}_indep32M.npz')
TS = {n: {k: T64(z[f'mc_pre_{k}']) for k in ('v', 'k3', 'k4')} for n, z in (('A', A), ('B', B))}
TS['AB'] = {k: (TS['A'][k] + TS['B'][k]) / 2 for k in ('v', 'k3', 'k4')}
MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz'); PLN = np.load(PUB / f'mps_planes_rank{rank:04d}.npz')
out.update(true_v_A=TS['A']['v'].numpy(), true_v_B=TS['B']['v'].numpy(), true_k3=TS['AB']['k3'].numpy(), true_k4=TS['AB']['k4'].numpy())

# ---- certificate 1: restarts without a change reproduce the base run
nulls = {L: float(np.abs(final(L) - base).max()) for L in (1, 6, 11, 13, 15)}
say(certificate='restart without change, max abs difference of the final prediction', by_start_layer=nulls)
assert max(nulls.values()) < 1e-9, nulls

# ---- LATE (mode late): truth at ONE late layer at a time and on the band 12-15; v / s / k / vsk in the mean formula, V for every consumer of the layer
if mode == 'late':
    f = lambda c: 100 * float(2 * (c @ e0) + (c @ c)) / EE
    two = lambda c, a: 100 * float((c[f'{a}_A'] + c[f'{a}_B']) @ e0 + c[f'{a}_A'] @ c[f'{a}_B']) / EE      # free of truth noise: the two draws' changes multiplied
    for name, layers in [(f'L{L}', (L,)) for L in (12, 13, 14, 15)] + [('B12-15', (12, 13, 14, 15))]:
        c = {}
        for arm, ts in (('v', 'A'), ('v', 'B'), ('s', 'AB'), ('k', 'AB'), ('vsk', 'A'), ('vsk', 'B'), ('V', 'A'), ('V', 'B')):
            ctrl.update(layers=layers, arm=arm, tset=ts); out[f'late{name}_{arm}_{ts}'] = final(layers[0]); c[f'{arm}_{ts}'] = out[f'late{name}_{arm}_{ts}'] - base
        say(late=name, final_mse_change_pct=dict(v=round(two(c, 'v'), 2), s=round(f(c['s_AB']), 2), k=round(f(c['k_AB']), 2), vsk=round(two(c, 'vsk'), 2), V=round(two(c, 'V'), 2)),
            V_vs_v_rms_over_base_error=round(float(np.sqrt(((c['V_A'] - c['v_A']) ** 2).sum() / EE)), 4))
        np.savez(OUT, **out)
    ctrl.update(layers=(), arm='', tset=None)
    say(certificate='mean formula against the host mean of the card, max abs difference', formula_check=max(dev) if dev else None)
    nulls = {L: float(np.abs(final(L) - base).max()) for L in (12, 15)}
    say(certificate='restart without change after all arms, max abs difference of the final prediction', by_start_layer=nulls)
    np.savez(OUT, **out); say(done=True, file=str(OUT)); sys.exit(0)

# ---- PART 1: mean-formula inputs, one layer at a time
PLAN = [('v', 'A'), ('v', 'B'), ('s', 'AB'), ('k', 'AB')]
f = lambda c: 100 * float(2 * (c @ e0) + (c @ c)) / EE
for name, layers in ([('B1-8', tuple(range(1, 9)))] + [(f'L{L}', (L,)) for L in range(9, 16)] if mode == 'full' else []):      # layers 1-8 as ONE arm (all eight layers together), then one layer at a time
    row = {}
    for arm, ts in PLAN:
        ctrl.update(layers=layers, arm=arm, tset=ts); out[f'{name}_{arm}_{ts}'] = final(layers[0]); row[f'{arm}_{ts}'] = out[f'{name}_{arm}_{ts}'] - base
    cv = (row['v_A'] + row['v_B']) / 2
    say(layers=name, share_v_pct=round(-100 * float(cv @ e0) / EE, 2), share_s_pct=round(-100 * float(row['s_AB'] @ e0) / EE, 2), share_k_pct=round(-100 * float(row['k_AB'] @ e0) / EE, 2),
        v_alone_pct=round(100 * float(2 * (cv @ e0) + (row['v_A'] @ row['v_B'])) / EE, 2), s_alone_pct=round(f(row['s_AB']), 2), k_alone_pct=round(f(row['k_AB']), 2))
for arm, ts in ([('v', 'A'), ('v', 'B'), ('s', 'AB'), ('k', 'AB'), ('vsk', 'A'), ('vsk', 'B')] if mode == 'full' else []):
    ctrl.update(layers=tuple(range(1, 16)), arm=arm, tset=ts); out[f'all_{arm}_{ts}'] = final(1)
    say(all_layers=arm, truth=ts, final_x1e9=round(float(((out[f'all_{arm}_{ts}'] - tp[15]) ** 2).mean()) * 1e9, 4))
ctrl.update(layers=(), arm='', tset=None)
say(certificate='mean formula against the host mean of the card, max abs difference', formula_check=max(dev) if dev else None)
np.savez(OUT, **out)

# ---- PART 2: pair-map inputs at layers 11..14, one at a time, closed loop
st = list(BASE[13]); st[0] = T64(MC['mean_post_12']); st[1] = T64(MC['cov_post_12']); PM.update(layer=13, mode='floor'); final(13, tuple(st), stop=14)
W14 = cd.ws[14]; vt14 = torch.diagonal(T64(MC['cov_pre_14'])); x = (W14 * ((PM['AT'] - T64(MC['cov_post_13'])) @ W14)).sum(0) / vt14
say(certificate='all-truth one-step floor at layer 13: next-layer variance, relative rms in % (T1: 0.017)', floor_v_pct=round(100 * float((x ** 2).mean().sqrt()), 4))
for l in ((11, 12, 13, 14) if mode == 'full' else ()):
    PM.update(layer=l, mode='collect'); PM.pop('fold', None); final(l, stop=l + 1); DX, info = PM['DX'], PM['info']
    Wn = cd.ws[l + 1]; vA, vB = TS['A']['v'][l + 1], TS['B']['v'][l + 1]; vc = REC[l + 1]['v']
    out[f'pm{l}_ev_A'] = ((vc - vA) / vA).numpy(); out[f'pm{l}_ev_B'] = ((vc - vB) / vB).numpy(); ev = (out[f'pm{l}_ev_A'] + out[f'pm{l}_ev_B']) / 2
    row = dict(pair_layer=l, variance_layer=l + 1, variance_error_rel_rms_pct=round(100 * math.sqrt(max(float(out[f'pm{l}_ev_A'] @ out[f'pm{l}_ev_B']), 0.0) / N), 4), info=info)
    for X in ARMS:
        dv = ((Wn * (DX[X] @ Wn)).sum(0) / ((vA + vB) / 2)).numpy(); out[f'pm{l}_{X}_dv'] = dv
        PM.update(layer=l, mode='inject', D=DX[X]); PM['v_seen'] = {}; out[f'pm{l}_{X}'] = final(l); c = out[f'pm{l}_{X}'] - base
        out[f'pm{l}_{X}_vnext'] = PM['v_seen'][l + 1].numpy()
        row[X] = dict(share_of_variance_error_pct=round(-100 * float(dv @ ev) / float(ev @ ev), 1), dv_rel_rms_pct=round(100 * float(np.sqrt((dv ** 2).mean())), 4),
                      share_of_final_error_pct=round(-100 * float(c @ e0) / EE, 2), final_alone_pct=round(f(c), 2))
    say(**row); del DX
PM.update(layer=None, mode=None); PM.pop('D', None)
np.savez(OUT, **out)

# ---- PART 3: one plane at truth at every layer 2..14, closed loop
for X in (('K21', 'K21k', 'K21o', 'K31', 'K22', 'PL') if mode in ('full', 'part3') else ()):
    PM.update(layer='all', mode='live', X=X); PM.pop('fold', None); PM['v_seen'] = {}; out[f'allpm_{X}'] = final(2); c = out[f'allpm_{X}'] - base
    out[f'allpm_{X}_v'] = torch.stack([PM['v_seen'][l] for l in range(3, 16)]).numpy()                       # the arm's per-unit variance at layers 3..15
    say(all_layers_plane=X, share_of_final_error_pct=round(-100 * float(c @ e0) / EE, 2), final_x1e9=round(float(((out[f'allpm_{X}'] - tp[15]) ** 2).mean()) * 1e9, 4))
PM.update(layer=None, mode=None)

# ---- PART 4 (mode part4): one plane exact on a BAND of pair layers, closed loop; and the single-layer arms of layer 12 with the second, independent 8M plane draw
if mode == 'part4':
    for name, band, arms in (('11-14', (11, 12, 13, 14), ('K31', 'K21k', 'PL')), ('8-10', (8, 9, 10), ('K31', 'K21k', 'PL')), ('2-7', tuple(range(2, 8)), ('K31', 'K21k', 'PL'))):
        for X in arms:
            PM.update(layer=band, mode='live', X=X); PM.pop('fold', None); PM['v_seen'] = {}; out[f'band{name}_{X}'] = final(band[0]); c = out[f'band{name}_{X}'] - base
            out[f'band{name}_{X}_v'] = torch.stack([PM['v_seen'][l] for l in range(12, 16)]).numpy()         # the arm's per-unit variance at layers 12..15
            say(band=name, plane=X, share_of_final_error_pct=round(-100 * float(c @ e0) / EE, 2), realized_change_pct=round(f(c), 2), size_pct=round(100 * float(c @ c) / EE, 2))
    PM.update(layer=None, mode=None)
    second = PUB / f'mps_planes_rank{rank:04d}_noise2.npz'
    if second.exists():
        PLN_first, PLN = PLN, np.load(second)
        PM.update(layer=12, mode='collect'); PM.pop('fold', None); final(12, stop=13); DX = PM['DX']; Wn = cd.ws[13]
        for X in ('K31', 'K21k', 'K21o', 'K22', 'PL'):
            out[f'pm12n2_{X}_dv'] = ((Wn * (DX[X] @ Wn)).sum(0) / ((TS['A']['v'][13] + TS['B']['v'][13]) / 2)).numpy()
            PM.update(layer=12, mode='inject', D=DX[X]); out[f'pm12n2_{X}'] = final(12); c = out[f'pm12n2_{X}'] - base
            say(second_draw_layer=12, plane=X, share_of_final_error_pct=round(-100 * float(c @ e0) / EE, 2), realized_change_pct=round(f(c), 2), size_pct=round(100 * float(c @ c) / EE, 2))
        PLN = PLN_first; PM.update(layer=None, mode=None); PM.pop('D', None); del DX
nulls = {L: float(np.abs(final(L) - base).max()) for L in (1, 12)}
say(certificate='restart without change after all arms, max abs difference of the final prediction', by_start_layer=nulls)
np.savez(OUT, **out)
say(done=True, file=str(OUT))
