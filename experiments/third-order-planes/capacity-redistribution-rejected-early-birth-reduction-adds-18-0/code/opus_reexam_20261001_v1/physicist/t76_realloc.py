#!/usr/bin/env python3
"""TAG opus-physicist T76: BILL-NEUTRAL REALLOCATION of the bank's capacity from the early layers to layers 8-12 (lead 10-10 ~10:40Z), torch, learned parts on.
Card = the trainer's CURRENT torch card (frozen copy otr_frozen_482e996f/otr_m.py of O/trainer/otr_m.py, his otr_engine) at the point his jmG starts from (request
O/trainer/run_M/claimed/a_0mG1_base_r0005.w1.json: params cert/refit/jmD_p1.pt, b6 / b7 columns at zero, head weights hu, k31ly ages 1-3, pbase bsk 0.34), applied
exactly as his worker applies a request (worker.py serve()).
ARMS (comma list, each one run per network after the base):
  base                                  the card as is
  births:<lo>-<hi>:<floor>[;...]        the floor of the birth selection (min(gate, 1 - gate) >= floor; the card's 0.01) at layers lo..hi
  resp:<lo>-<hi>:<floor>[;...]          the same for the read's response rows (alive units with min(gate, 1 - gate) >= floor)
  sched:<r7>-<r10>-<r13>                the old tier's ranks at the joins of layers 7 / 10 / 13 (the card's 384-320-384)
  tier:<n>                              the cap on those ranks (covstep.TIER_RANK, the number of sketch rows; the card's 384)
  joins:<j1>-<j2>-<j3>                  the layers where cohorts 0-2 / 3-5 / 6-8 join the old tier (the card's 7-10-13; s2card.EVERY3)
  sketch:<mode>:<l1-l2..>               the join's sketch at FIXED rank at those join layers: copy (certificate), rr (Rayleigh-Ritz truncation), pow (one power
                                        iteration), exact (top eigenvectors of the joined matrix)
  (a floor is applied by scaling the score handed to the card's select: score x 0.01 / floor, same order, so its own rounding to 32 and its records stay)
Per network and arm: final-layer MSE against the 1e9 truth, and per layer the counts the bill follows: births, response rows, live units, columns pushed, old-tier columns.
-> t76_<tag>.jsonl (one line per network and arm) ; usage: t76_realloc.py <tag> <ranks> <arm;arm;...> [threads]   (arms separated by '|')"""
import hashlib, json, os, re, sys, time, subprocess
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; O = HERE.parent
TAG = sys.argv[1]; RANKS = [int(x) for x in sys.argv[2].split(',')]; ARMS = sys.argv[3].split('|'); threads = sys.argv[4] if len(sys.argv) > 4 else '2'
FROZEN = HERE / 'otr_frozen_482e996f'; SHA = '482e996f479fbb8d0b237c2a5f04982f604fad3e3994f9772db46569b332a226'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'TRAINER_THREADS'): os.environ[key] = threads
os.environ.update(TRAINER_PL='1', TRAINER_CARD='M', TRAINER_F4C='1')
sys.path = [x for x in sys.path if os.path.abspath(x or os.getcwd()) != str(HERE)]
sys.path.insert(0, str(O / 'trainer')); sys.path.insert(0, str(FROZEN))
import otr_engine as E
assert Path(E.M.__file__).resolve().parent == FROZEN and hashlib.sha256(Path(E.M.__file__).read_bytes()).hexdigest() == SHA, E.M.__file__
from otr_common import load_params, load_pl
import s2card, covstep
torch = E.torch; t = torch
REQ = json.loads((O / 'trainer/run_M/claimed/a_0mG1_base_r0005.w1.json').read_text())
# the worker's serve(), copied
E.set_parameters(*load_params(torch, REQ['params'])); shifts = {f: list(v) for f, v in load_pl(torch, REQ['params']).items()}
for family, values in (REQ.get('pl') or {}).items(): shifts[family] = [a + b for a, b in zip(shifts.get(family, [0.0] * 16), values)]
E.PL.set_values(shifts)
E.STAR_ATTACH = bool(REQ.get('attach', E.STAR_ATTACH)); E.KEEP_FINAL = False; E.KEEP_ALL = False; E.STAR_SS = REQ.get('ss')
E.M.F4C = bool(REQ.get('f4c', E.M.F4C)); E.M.NCAR = REQ.get('ncar'); E.M.HEADF = int(REQ.get('hf') or 0); E.M.ROWS.clear(); E.M.RG = bool(REQ.get('rg'))
E.M.DUMP15 = REQ.get('dump15'); E.M.B6 = bool(REQ.get('b6')); E.M.HSPLIT = {int(k): tuple(v) for k, v in (REQ.get('hsplit') or {}).items()}
E.M.HBANDS = bool(REQ.get('hbands')); E.M.B7 = bool(REQ.get('b7'))
if E.M.RG or E.M.RG_W: E.M.readout_weights(REQ.get('rgw'))
if E.M.HEADF or E.M.HEADU is not None: E.M.head_weights(REQ.get('hu'))
SCHED0 = E.LC.SCHED
EVERY0 = dict(s2card.EVERY3)
TIER0 = covstep.TIER_RANK
FLOORS = {}; COUNTS = {}


def parse(arm):
    fl = {}; sched = SCHED0
    for part in [p for p in arm.split(';') if p and p != 'base']:
        kind, rest = part.split(':', 1)
        if kind == 'sched': sched = rest; continue
        if kind == 'tier': fl[('tier', 0)] = int(rest); continue
        if kind == 'joins': fl[('joins', 0)] = tuple(int(x) for x in rest.split('-')); continue
        if kind == 'sketch': m_, ls_ = rest.split(':'); fl[('sketch', 0)] = (m_, tuple(int(x) for x in ls_.split('-'))); continue
        rng, val = rest.split(':'); lo, hi = (int(x) for x in rng.split('-'))
        for l in range(lo, hi + 1): fl[(kind, l)] = float(val)
    return fl, sched


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
        d = (J * J) @ weight
        if old_columns: d = d + ((Qp @ core) * Qp).sum(1)
        rows = t.argsort(-d.to(t.float64), stable=True)[:rank]
        Y = J @ (weight[:, None] * J[rows, :].T)
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


install0 = E.M.install


def install(cd):
    cd.shared['push'] = make_push_q(cd.shared['_qr'], s2card.EVERY3, E.LC.SCHED, cd.shared['push'])      # BELOW the modules' wrappers of the push
    install0(cd)
    sel0 = cd.select; push0 = cd.shared['push']; slot = {}

    def select(score):      # the step's selections in order: alive (gate), response rows, births (min(g, 1 - g))
        l = cd.call_layer; k = slot.get(l, 0); slot[l] = k + 1
        kind = {0: 'alive', 1: 'resp', 2: 'births'}.get(k)
        f = FLOORS.get((kind, l))
        r = sel0(score * (0.01 / f)) if f else sel0(score)
        COUNTS.setdefault(l, {})[kind or 'sel%d' % k] = int(r[0].numel())
        return r
    cd.select = select

    def push(tt, layer, w32, p_full, lam, previous_indices, Q, F, core, cohort_stops, rank=384):
        out = push0(tt, layer, w32, p_full, lam, previous_indices, Q, F, core, cohort_stops, rank=rank)
        COUNTS.setdefault(layer, {}).update(cols=int(out[0].shape[1]), old=int(out[4]), live_prev=int(len(previous_indices)))
        return out
    cd.shared['push'] = push
E.M.install = install


def gate():
    while True:
        if (HERE / 'STOP_T76').exists(): return False
        L = [x for x in (O / 'TEAM_LOG.md').read_text().splitlines() if x.startswith('[lead] ')]
        o = max([i for i, x in enumerate(L) if 'QUIET WINDOW OPEN' in x], default=-1); c = max([i for i, x in enumerate(L) if 'QUIET WINDOW CLOSED' in x], default=-1)
        pr = [x for x in L if 'MACHINE PRIORITY UNTIL' in x]; busy = o > c
        if pr:
            mm = re.search(r'MACHINE PRIORITY UNTIL (\d\d):(\d\d)Z', pr[-1]); now = time.gmtime()
            if mm and now.tm_hour * 60 + now.tm_min < int(mm.group(1)) * 60 + int(mm.group(2)) and pr[-1][7:17] == time.strftime('%Y-%m-%d', now): busy = True
        if not busy and subprocess.run(['bash', str(O / 'idle_gate.sh'), '30'], capture_output=True).returncode == 0: return True
        time.sleep(20)


out = HERE / f't76_{TAG}.jsonl'
done = set()
if out.exists():
    for line in out.read_text().splitlines():
        d = json.loads(line); done.add((d['rank'], d['arm']))
for r in RANKS:
    for arm in ARMS:
        if (r, arm) in done: continue
        if not gate(): print('STOP'); sys.exit(0)
        FLOORS.clear(); f, sched = parse(arm); FLOORS.update(f); E.LC.SCHED = sched; covstep.TIER_RANK = int(f.get(('tier', 0), TIER0)); jl = f.get(('joins', 0)); s2card.EVERY3.clear(); s2card.EVERY3.update(EVERY0 if jl is None else {jl[0]: [0, 1, 2], jl[1]: [3, 4, 5], jl[2]: [6, 7, 8]}); SKETCH[0] = f.get(('sketch', 0)); COUNTS.clear(); t0 = time.time()
        rec = E.run_network(r, 'eval')
        line = dict(rank=r, arm=arm, sched=sched, mse=rec['mse'], mse_layers=rec['mse_layers'], counts={str(k): v for k, v in sorted(COUNTS.items())}, wall=round(time.time() - t0, 1))
        with open(out, 'a') as fh: fh.write(json.dumps(line) + '\n')
        print(json.dumps(dict(TAG='physicist T76', rank=r, arm=arm, final_x1e9=round(rec['mse'] * 1e9, 6), wall=line['wall'])), flush=True)
E.LC.SCHED = SCHED0; covstep.TIER_RANK = TIER0; s2card.EVERY3.clear(); s2card.EVERY3.update(EVERY0)
