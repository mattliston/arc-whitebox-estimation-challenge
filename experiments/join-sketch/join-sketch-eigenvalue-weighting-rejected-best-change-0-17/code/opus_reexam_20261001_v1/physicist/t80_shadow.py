#!/usr/bin/env python3
"""TAG opus-physicist T80: SHADOW TESTS of physics / capacity changes at the trainer's newest point (lead 10-10 16:0xZ): one of the trainer's requests (label) replayed
as his worker applies it on his current card module (frozen otr_frozen_d61b46d1/otr_m.py, as T79), with an arm applied on top, on pool ranks or on the analyst's
generated seeds (fresh weights / truths through data_paths, as T79). Arms (';'-joined parts):
  base                      the request as is
  sched:<r7>-<r10>-<r13>    the old tier's ranks at the joins of layers 7 / 10 / 13 (the card's 384-320-384)
  tier:<n>                  the cap on those ranks (covstep.TIER_RANK, 384)
  pl:<family>:<lo>-<hi>:<d> an extra per-layer dial shift d of one family at layers lo..hi (the trainer's otr_pl families), on top of the request's
  k3old15:<f>               the old tier's part of the final layer's per-unit k3 (sum over the old-tier columns of lam_r pp_ir^3) kept at fraction f (1 = card)
  unmask15:<group>          at layer 15 the head's masked features of the group (a = 13, 14, 15, 26 from the full C_pre15) computed and unmasked
  knob:<name>:<v>           a HookCard knob (cd.knobs, e.g. bfix_gamma 0.1)
  ncar:<module>:<arg>:<v>   an argument of one installed module in the request's ncar list (e.g. n6y ca 0.75, forms 3)
  joinsched:<L>=<c,c>/<L>=<c,c>...  the join schedule (layer = cohorts joined there; the card's 7=0,1,2/10=3,4,5/13=6,7,8); give 'sched' one rank per join
  sketch:<mode>:<layers>    the old tier's join at FIXED rank (T76's re-implementation): copy (certificate), rr, pow, exact, w<p> (the sketch weighted by |lam|^p
                            instead of lam^2; the projection and the core keep lam^2)
-> t80_<tag>.jsonl ; usage: t80_shadow.py <tag> <label> <ranks|seeds> <arm|arm|...> [threads]"""
import hashlib, json, os, sys, time, glob
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; O = HERE.parent
TAG, LABEL = sys.argv[1], sys.argv[2]; NETS = [int(x) for x in sys.argv[3].split(',')]; ARMS = sys.argv[4].split('|'); threads = sys.argv[5] if len(sys.argv) > 5 else '4'
FROZEN = HERE / 'otr_frozen_d61b46d1'; SHA = 'd61b46d101a590da20fb170b29c368b576a8bb43019c6fd8e78834701d5f946f'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'TRAINER_THREADS'): os.environ[key] = threads
os.environ.update(TRAINER_PL='1', TRAINER_CARD='M', TRAINER_F4C='1')
sys.path = [x for x in sys.path if os.path.abspath(x or os.getcwd()) != str(HERE)]
sys.path.insert(0, str(O / 'trainer')); sys.path.insert(0, str(FROZEN))
import otr_engine as E
assert Path(E.M.__file__).resolve().parent == FROZEN and hashlib.sha256(Path(E.M.__file__).read_bytes()).hexdigest() == SHA, E.M.__file__
from otr_common import load_params, load_pl
import covstep
torch = E.torch
sys.path.insert(0, str(O / 'analyst/fresh')); import loader as FL
SCRATCH = Path('/private/tmp/claude-501/-Users-matt-arc-white-box-estimation/fce1fde2-a6c3-4554-bf56-f6828c89d0f7/scratchpad/t79'); SCRATCH.mkdir(parents=True, exist_ok=True)
paths0 = E.data_paths


def fresh_paths(seed):
    w, tr = SCRATCH / f'w_{seed}.npy', SCRATCH / f't_{seed}.npz'
    if not w.exists(): np.save(w, FL.weights(seed))
    if not tr.exists(): np.savez(tr, mc_post=np.asarray(FL.truth(seed)['mc_post'], dtype=np.float64))
    return w, tr
E.data_paths = lambda rank: fresh_paths(rank) if rank >= 7_000_000 else paths0(rank)
REQ = json.loads(open(sorted(glob.glob(str(O / f'trainer/run_M/claimed/{LABEL}_r*.json')))[0]).read())
SCHED0, TIER0 = E.LC.SCHED, covstep.TIER_RANK
install0 = E.M.install
INFO = {}


UNMASK_GROUPS = dict(z=(), a=(13, 14, 15, 26), r=(13, 14, 15), q=(26,), f=(13, 24, 25, 26), b=(13, 14, 15, 22, 23, 24, 25, 26), c=(13, 14, 15, 22, 23, 24, 25, 26, 27, 28))
MASK15 = [1] * 13 + [0] * 3 + [1] * 6 + [0] * 7 + [1]


def sel(score):      # the card's selection (model_bank_bfix.select): the top scores, as many as reach .01, rounded up to a multiple of 32
    count = min(1024, ((int((score >= .01).sum()) + 31) // 32) * 32)
    ids = torch.argsort(-score, stable=True)[:count]
    return ids, torch.zeros_like(score).scatter(0, ids, torch.ones(count, dtype=score.dtype))


def install(cd):
    import s2card
    base_gnn = cd.env['gnn_step']; silu = lambda z: z / (1 + torch.exp(-z)); f64 = torch.float64
    read_base, pair_base = cd.shared['read'], cd.env['_nonlinear_pair_batch']      # the card's own read and pair map, below every module

    def feats_bc(Wd, Cpre):      # features 22-25 (row / column norms and sums of the K21 plane over v_i s_j) and 27-28 (rms / mean of the post-correlation row) at
        t = torch; n = Cpre.shape[0]      # layer 15, formed as the step forms them at layers 1-14: the bank's read on the response rows x live columns + its diagonal,
        mu, vv = ST15['mu'].to(f64), ST15['vv'].to(f64); s = vv.clamp_min(1e-12).sqrt()      # the full star, the b-form; the plain pair map on (mu, C_pre15, K21)
        gate = t.erfc(-(mu / vv.clamp_min(1e-10).sqrt()) / 2 ** 0.5) / 2
        ids, alive = sel(gate); rids, _ = sel(t.where(alive.bool(), t.minimum(gate, 1 - gate), -t.ones_like(gate)))
        S = ST15['state']; star, b, radial = S[4], S[5], S[7]; bpre = Wd.T @ b.to(f64); pp, lam = PUSH15['pp'], PUSH15['lam']
        cp = t.zeros((n, n), dtype=f64)
        if lam.numel():
            cp = cp.index_put((rids[:, None], ids[None, :]), read_base(t, pp, lam, rids, ids, PUSH15['U'], PUSH15['F'], PUSH15['old']).to(f64))
            p64 = pp.to(f64); cp = t.diagonal_scatter(cp, ((p64 * p64 * p64) * lam.to(f64)).sum(1))
        k21 = cp + cd.env['_flms_star'](cd.w[15], star).to(f64) + vv[:, None] * bpre[None, :] + 2 * Cpre * bpre[:, None]
        sk = k21 / (vv[:, None] * s[None, :]); sk2 = sk * sk
        out = {22: sk2.sum(1).sqrt(), 23: sk2.sum(0).sqrt(), 24: sk.sum(1) / n ** 0.5, 25: sk.sum(0) / n ** 0.5}
        k4 = radial * vv * vv; k22 = radial * (t.outer(vv, vv) + 2 * Cpre * Cpre) / 3
        _, Cp, _, _, auxp = pair_base(mu, Cpre, k21, k4, k22, 'covariance')
        ps = auxp['post_v'].to(f64).clamp_min(1e-12).sqrt(); prho = (Cp.to(f64) / t.outer(ps, ps)).clamp(-1, 1)
        out.update({27: (prho * prho).mean(1).sqrt(), 28: prho.mean(1)})
        INFO['n_alive15'], INFO['n_resp15'] = int(ids.numel()), int(rids.numel())
        return out

    def gnn(l, x, sigma, W, state, P):      # arm unmask15: at layer 15 the masked features of the group computed as at layers 1-14 and let through
        if l != 15 or (UNMASK[0] is None and DUMP[0] is None) or 'C' not in CP14:      # arm dump15: the same with nothing let through, everything saved
            return base_gnn(l, x, sigma, W, state, P)
        grp = UNMASK_GROUPS[UNMASK[0] or 'z']
        with torch.no_grad():
            Wd = cd.ws[15].to(torch.float64); Cpre = Wd.T @ (CP14['C'] @ Wd); s = torch.diagonal(Cpre).clamp_min(1e-12).sqrt()
            rho = Cpre / torch.outer(s, s); n = rho.shape[0]; alpha = x[:, 3].to(torch.float64)
            vals = {13: rho.mean(1), 14: (rho * rho).mean(1).sqrt(), 15: rho.abs().mean(1), 26: rho @ alpha / n}
            if DUMP[0] is not None or any(k > 15 and k != 26 for k in grp): vals.update(feats_bc(Wd, Cpre))
        x0 = x
        x = x.clone()
        for k in grp: x[:, k] = vals[k].to(x.dtype)
        mask = torch.tensor([1.0 if (MASK15[k] or k in grp) else 0.0 for k in range(30)], dtype=torch.float32)
        xn = torch.clamp((x.to(torch.float32) - P['centers'][15]) / P['scales'][15], -8, 8) * mask[None, :]
        g = xn[:, 5:6]; src = state @ P['msg_w'].T + P['msg_b']; cm = int(P['msg_w'].shape[0]) // 3
        w = W.to(torch.float32); w2 = w * w; w3 = w2 * w
        m1 = w.T @ src[:, :cm]; m2 = w2.T @ src[:, cm:2 * cm]; m3 = w3.T @ src[:, 2 * cm:]
        e1 = w2.T @ (src[:, :cm] ** 2); e2 = (w2 * w2).T @ (src[:, cm:2 * cm] ** 2); e3 = (w3 * w3).T @ (src[:, 2 * cm:] ** 2)
        msgs = torch.cat([g * m1, g * m2, g * m3, torch.log1p(torch.clamp(e1, min=0)), torch.log1p(torch.clamp(e2, min=0)), torch.log1p(torch.clamp(e3, min=0))], 1)
        h = torch.cat([xn, torch.clamp(msgs, -8, 8)], 1)
        pre = h @ P['enc0_w'].T + P['enc0_b']
        h = silu(pre); h = silu(h @ P['enc2_w'].T + P['enc2_b']); st = h @ P['enc4_w'].T + P['enc4_b']
        delta = (sigma.to(torch.float32) * P['rms'][15]) * (st @ P['head'][15] + P['head_b'][15]) * (1 + 0.25 * torch.tanh(P['gain'][15:16]))
        if DUMP[0] is not None:
            cpu = lambda z: z.detach().to(torch.float64).cpu().numpy()
            D15.update(x=cpu(x0), pre=cpu(pre), sigma=cpu(sigma), delta=cpu(delta), centers=cpu(P['centers'][15]), scales=cpu(P['scales'][15]), enc0_w=cpu(P['enc0_w']),
                       enc2_w=cpu(P['enc2_w']), enc2_b=cpu(P['enc2_b']), enc4_w=cpu(P['enc4_w']), enc4_b=cpu(P['enc4_b']), head=cpu(P['head'][15]), head_b=cpu(P['head_b'][15]),
                       rms=cpu(P['rms'][15]), gain=cpu(P['gain'][15]), **{'f%d' % k: cpu(v) for k, v in vals.items()})
        return delta, st
    cd.env['gnn_step'] = gnn
    cd.shared['push'] = make_push_q(cd.shared['_qr'], s2card.EVERY3, E.LC.SCHED, cd.shared['push'])      # below the modules' wrappers
    install0(cd); side = getattr(cd, 'm_side', {}); h0 = cd.hooks.get('k3_15')
    for k_, v_ in KNOBS.items():
        assert k_ in cd.knobs, (k_, sorted(cd.knobs)); cd.knobs[k_] = v_
    so0 = cd.hooks['state_out']

    def state_out_cap(l, out, ctx):
        out = so0(l, out, ctx)
        if l == 14 and (UNMASK[0] is not None or DUMP[0] is not None) and not side.get('again'): CP14['C'] = out[1].detach().to(torch.float64)
        if l == 15 and DUMP[0] is not None and not side.get('again'):
            D15.update(host=ctx['host'].detach().to(torch.float64).numpy(), m=out[0].detach().to(torch.float64).numpy(), mu=ctx['mu'].detach().to(torch.float64).numpy(),
                       vv=ctx['vv'].detach().to(torch.float64).numpy(), k3=ctx['k3'].detach().to(torch.float64).numpy())
        return out
    cd.hooks['state_out'] = state_out_cap
    si0 = cd.hooks.get('state_in')

    def state_in_cap(l, state, ctx):
        state = state if si0 is None else si0(l, state, ctx)
        if l == 15 and DUMP[0] is not None and not side.get('again'): ST15['state'] = state
        return state
    cd.hooks['state_in'] = state_in_cap
    CP14.clear()
    sel0 = cd.select; push1 = cd.shared['push']; slot = {}
    read1 = cd.shared['read']

    def read(*args):      # arms readsvd / readeig: the read's plane (response rows x live columns) projected on k column directions
        out = read1(*args); a = READP[0]; l = cd.call_layer
        if a is None or l not in a[2] or side.get('again'): return out
        with torch.no_grad():
            P = out.detach().to(torch.float64); k = min(a[1], P.shape[1])
            if a[0] == 'readsvd':
                U_, S_, Vh = torch.linalg.svd(P, full_matrices=False); Pk = (U_[:, :k] * S_[:k]) @ Vh[:k]
            else:
                idx = args[4]; C = cd.ncar6_state['C'].detach().to(torch.float64)[idx][:, idx]
                ev, V = torch.linalg.eigh(0.5 * (C + C.T)); V = V[:, -k:]; Pk = (P @ V) @ V.T
            INFO.setdefault('kept', []).append(float((Pk.norm() / P.norm().clamp_min(1e-300)) ** 2))
        return out + (Pk.to(out.dtype) - out.detach())
    cd.shared['read'] = read

    def select(score):
        l = cd.call_layer; k = slot.get(l, 0); slot[l] = k + 1
        r = sel0(score); COUNTS.setdefault(l, {})[{0: 'alive', 1: 'resp', 2: 'births'}.get(k, 'sel%d' % k)] = int(r[0].numel()); return r
    cd.select = select

    def push(tt, layer, w32, p_full, lam, previous_indices, Q, F, core, cohort_stops, rank=384):
        out = push1(tt, layer, w32, p_full, lam, previous_indices, Q, F, core, cohort_stops, rank=rank)
        COUNTS.setdefault(layer, {}).update(cols=int(out[0].shape[1]), old=int(out[4]), live_prev=int(len(previous_indices)), basis=(int(out[1].shape[1]) if out[1] is not None else 0))
        if layer == 15: PUSH15.update(pp=out[0], U=out[1], F=out[2], old=int(out[4]), lam=lam)
        return out
    cd.shared['push'] = push

    def k3_15(l, k3, ctx):
        k3 = k3 if h0 is None else h0(l, k3, ctx)
        ST15.update(mu=ctx['mu'].detach(), vv=ctx['vv'].detach())
        if K3OLD[0] is None or side.get('again'): return k3
        with torch.no_grad():
            pp = side['pp'].detach().to(torch.float64); lam = side['lam_in'].detach().to(torch.float64)[:pp.shape[1]]; old = int(side.get('old', 0) or 0)
            k3old = (pp[:, :old] ** 3) @ lam[:old] if old else torch.zeros(pp.shape[0], dtype=torch.float64)
            INFO['old'] = old; INFO['share'] = float(k3old.norm() / k3.detach().to(torch.float64).norm().clamp_min(1e-300))
        return k3 - (1.0 - K3OLD[0]) * k3old.to(k3.dtype)
    cd.hooks['k3_15'] = k3_15
E.M.install = install


def apply(req, extra_pl):
    E.set_parameters(*load_params(torch, req['params']))
    shifts = {f: list(v) for f, v in load_pl(torch, req['params']).items()}
    for family, values in (req.get('pl') or {}).items(): shifts[family] = [a + b for a, b in zip(shifts.get(family, [0.0] * 16), values)]
    for family, values in extra_pl.items(): shifts[family] = [a + b for a, b in zip(shifts.get(family, [0.0] * 16), values)]
    E.PL.set_values(shifts)
    E.STAR_ATTACH = bool(req.get('attach', E.STAR_ATTACH)); E.KEEP_FINAL = False; E.KEEP_ALL = False; E.STAR_SS = req.get('ss')
    E.M.F4C = bool(req.get('f4c', E.M.F4C)); E.M.NCAR = req.get('ncar'); E.M.HEADF = int(req.get('hf') or 0); E.M.ROWS.clear(); E.M.RG = bool(req.get('rg'))
    E.M.DUMP15 = req.get('dump15'); E.M.B6 = bool(req.get('b6')); E.M.HSPLIT = {int(k): tuple(v) for k, v in (req.get('hsplit') or {}).items()}
    E.M.HBANDS = bool(req.get('hbands')); E.M.B7 = bool(req.get('b7'))
    if hasattr(E.M, 'CV'): E.M.CV = bool(req.get('cv'))
    if E.M.RG or E.M.RG_W: E.M.readout_weights(req.get('rgw'))
    if E.M.HEADF or E.M.HEADU is not None: E.M.head_weights(req.get('hu'))


K3OLD = [None]
JOINS = [None]; COUNTS = {}; READP = [None]; KNOBS = {}; NCARX = []; UNMASK = [None]; CP14 = {}; DUMP = [None]; D15 = {}; PUSH15 = {}; ST15 = {}
import s2card as _s2
EVERY0 = dict(_s2.EVERY3)
SKETCH = [None]      # (mode, layers) of the current arm: copy (the card's arithmetic re-implemented: certificate), rr (Rayleigh-Ritz truncation of the sketch's
                     # span to the rank), pow (one power iteration of the joined matrix on the sketch), exact (top-rank eigenvectors of the joined matrix)


def make_push_q(qr, joins, sched_str, push_orig):
    """the card's join (s6bcard_loop5js.make_push, energy-ordered column sketch of M = J diag(w) J' + Qp core Qp') with the sketch-quality modes of SKETCH"""
    sched = dict(zip(sorted(joins), (int(x) for x in sched_str.split('-'))))

    def push(t, layer, w32, p_full, lam, previous_indices, Q, F, core, cohort_stops, rank=384):
        mode, mlayers = SKETCH[0] if SKETCH[0] else (None, ())
        if mode is None or layer not in joins or layer not in mlayers:
            return push_orig(t, layer, w32, p_full, lam, previous_indices, Q, F, core, cohort_stops, rank=rank)
        wp = w32[previous_indices, :].T
        before = [c_ for L_, cs in joins.items() if L_ < layer for c_ in cs]
        old_columns = cohort_stops[max(before)] if before else 0
        start, stop = old_columns, cohort_stops[max(joins[layer])]
        J = p_full[previous_indices, start:stop]; weight = lam[start:stop] ** 2
        Qp = Q[previous_indices, :] if old_columns else None
        wsk = weight if not mode.startswith('w') else lam[start:stop].abs() ** float(mode[1:])      # the sketch's weighting (w<p>: |lam|^p)
        d = (J * J) @ wsk
        if old_columns: d = d + ((Qp @ core) * Qp).sum(1)
        rows = t.argsort(-d.to(t.float64), stable=True)[:rank]
        Y = J @ (wsk[:, None] * J[rows, :].T)
        if old_columns: Y = Y + Qp @ (core @ Qp[rows, :].T)
        if mode == 'exact':
            M = (J @ (weight[:, None] * J.T)).to(t.float64)
            if old_columns: M = M + (Qp @ (core @ Qp.T)).to(t.float64)
            ev, V = t.linalg.eigh(0.5 * (M + M.T)); Qn = V.flip(1)[:, :rank].to(J.dtype)
        else:
            Qn = qr(t, Y)
            if mode == 'pow':
                Y2 = J @ (weight[:, None] * (J.T @ Qn))
                if old_columns: Y2 = Y2 + Qp @ (core @ (Qp.T @ Qn))
                Qn = qr(t, Y2)
        Fj = Qn.T @ J
        overlap = Qn.T @ Qp if old_columns else None
        r = min(sched[layer], Qn.shape[1])
        if mode == 'rr':
            Mp = (Fj @ (weight[:, None] * Fj.T)).to(t.float64)
            if old_columns: Mp = Mp + (overlap @ (core @ overlap.T)).to(t.float64)
            ev, V = t.linalg.eigh(0.5 * (Mp + Mp.T)); V = V.flip(1)[:, :r].to(Qn.dtype)
            Qn = Qn @ V; Fj = V.T @ Fj
            if old_columns: overlap = V.T @ overlap
        else:
            Qn, Fj = Qn[:, :r], Fj[:r]
            if old_columns: overlap = overlap[:r]
        joined = Fj @ (weight[:, None] * Fj.T)
        if old_columns:
            F = t.cat((overlap @ F, Fj), dim=1); core = overlap @ (core @ overlap.T) + joined
        else:
            F, core = Fj, joined
        basis, old_columns = Qn, stop
        U = wp @ basis
        return t.cat((U @ F, wp @ p_full[previous_indices, old_columns:]), dim=1), U, F, core, old_columns
    return push




def parse(arm):
    sched, tier, pl = SCHED0, TIER0, {}
    K3OLD[0] = None; SKETCH[0] = None; JOINS[0] = None; READP[0] = None; KNOBS.clear(); NCARX.clear(); UNMASK[0] = None; DUMP[0] = None
    for part in [p for p in arm.split(';') if p and p != 'base']:
        kind, rest = part.split(':', 1)
        if kind == 'sched': sched = rest
        elif kind == 'tier': tier = int(rest)
        elif kind == 'k3old15': K3OLD[0] = float(rest)
        elif kind == 'sketch': m_, ls_ = rest.split(':'); SKETCH[0] = (m_, tuple(int(x) for x in ls_.split('-')))
        elif kind == 'unmask15': UNMASK[0] = rest
        elif kind == 'dump15': DUMP[0] = rest
        elif kind == 'knob': k_, v_ = rest.split(':'); KNOBS[k_] = float(v_)
        elif kind == 'ncar': m_, k_, v_ = rest.split(':'); NCARX.append((m_, k_, float(v_) if v_.replace('.', '', 1).replace('-', '', 1).isdigit() else v_))
        elif kind in ('readsvd', 'readeig'): k_, ly_ = rest.split(':'); lo_, hi_ = (int(x) for x in ly_.split('-')); READP[0] = (kind, int(k_), set(range(lo_, hi_ + 1)))
        elif kind == 'joinsched': JOINS[0] = {int(g.split('=')[0]): [int(c) for c in g.split('=')[1].split(',')] for g in rest.split('/')}
        elif kind == 'pl':
            fam, rng, d = rest.split(':'); lo, hi = (int(x) for x in rng.split('-')); v = pl.setdefault(fam, [0.0] * 16)
            for l in range(lo, hi + 1): v[l] += float(d)
    return sched, tier, pl


out = HERE / f't80_{TAG}.jsonl'; done = set()
if out.exists():
    for line in out.read_text().splitlines(): d = json.loads(line); done.add((d['net'], d['arm']))
for net in NETS:
    for arm in ARMS:
        if (net, arm) in done: continue
        if (HERE / 'STOP_T80').exists(): print('STOP'); sys.exit(0)
        sched, tier, pl = parse(arm); E.LC.SCHED = sched; covstep.TIER_RANK = tier; _s2.EVERY3.clear(); _s2.EVERY3.update(EVERY0 if JOINS[0] is None else JOINS[0])
        assert len(sched.split('-')) == len(_s2.EVERY3), ('one rank per join', sched, _s2.EVERY3)
        req = json.loads(json.dumps(REQ))
        for m_, k_, v_ in NCARX:
            nc = req['ncar'] if isinstance(req['ncar'], list) else __import__('ast').literal_eval(req['ncar'])
            for e in nc:
                if e.get('module') == m_: e[k_] = int(v_) if k_ in ('forms', 'dfit') else v_
            req['ncar'] = nc
        apply(req, pl); COUNTS.clear(); D15.clear(); PUSH15.clear(); ST15.clear(); t0 = time.time()
        E.KEEP_FINAL = DUMP[0] is not None
        rec = E.run_network(net, 'eval')
        if DUMP[0] is not None:      # arm dump15: the layer-15 head's inputs, first-layer pre-activation, weights, the ten masked features, host / final / truth
            (HERE / 'l15j_dump').mkdir(exist_ok=True)
            truth = np.load(E.data_paths(net)[1])['mc_post'].astype(np.float64)[15]
            np.savez(HERE / 'l15j_dump' / f'{LABEL}_{net}.npz', truth=truth, final=np.array(rec['final_prediction'], dtype=np.float64), mse=np.array(rec['mse']), **D15)
        line = dict(net=net, arm=arm, label=LABEL, mse=rec['mse'], wall=round(time.time() - t0, 1), info=dict(INFO), sched=sched, joins={str(k): v for k, v in _s2.EVERY3.items()},
                    counts={str(k): v for k, v in sorted(COUNTS.items())}); INFO.clear()
        with open(out, 'a') as fh: fh.write(json.dumps(line) + '\n')
        print(json.dumps(dict(TAG='physicist T80', net=net, arm=arm, final_x1e9=round(rec['mse'] * 1e9, 6), wall=line['wall'])), flush=True)
E.LC.SCHED = SCHED0; covstep.TIER_RANK = TIER0; _s2.EVERY3.clear(); _s2.EVERY3.update(EVERY0)
