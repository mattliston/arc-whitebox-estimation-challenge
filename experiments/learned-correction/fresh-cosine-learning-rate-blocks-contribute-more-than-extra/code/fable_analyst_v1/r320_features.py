#!/usr/bin/env python3
"""TAG fable-analyst (AN-69b): RUNTIME FEATURES for the per-net R320 router, on the S25-like torch card: the physicist's S23-structure
refit6b/s6bcard_loop4j.py with the loopR3 params + loopR3 dials (S24's literals; S25 adds only j3cut, treuse and R320), learned ON, no refit,
float64 CPU. The join is the energy-ordered Nystrom join of adaptive_join.py (make_push copied verbatim; fixed384 = the card to rounding).
Per net: per join (L7/L10/L13) old columns before, joined columns, live rows, missed energy 1 - E(r) for r = 256/320/352; per layer the
old columns and rank; the card's own selection counts per layer (alive, response rows, births; floor .01 unchanged); the final MSE at
fixed384 and fixed320 (the torch per-net R320 penalty, a transfer check against the native target).
v2: optional 4th argument = arms (default fixed384,fixed320; sched<a>-<b>-<c> = per-join ranks at L7/L10/L13); truth from
fable_refit_v1/truth688 (covers TRAIN nets); 'heldout40' = 40 TRAIN640 nets outside TRAIN160 and VALID40 (every 12th of the split's new_ranks).
v4 (v3 kept as r320_features_v3.py): a join rank above 384 (raise arms, e.g. sched384-320-448) takes that many Nystrom rows; ranks <= 384
are unchanged. v3 (co-adaptation evals; v2 kept as r320_features_v2.py): env R320_PARAMS = params/dials dir relative to the experiment root (default the
loopR3 dir) and R320_EPOCH (default 0) load params_epoch<N>.pt + dials_epoch<N>.json instead; rows then carry params and epoch.
usage: r320_features.py <ranks|valid40|heldout40> <threads> <out.jsonl> [arms]"""
import os, sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
for d in ('fable_engineer_v1', 'fable_covhunter_v1', 'fable_physicist_v1/refit6b', 'fable_analyst_v1'):
    sys.path.insert(0, str(P2 / d))
import numpy as np, torch
import s6bcard_loop4j as LC
import s2card
JT = LC.JT
arg, THREADS, OUT = sys.argv[1], int(sys.argv[2]), HERE / sys.argv[3]
ARMS = sys.argv[4].split(',') if len(sys.argv) > 4 else ['fixed384', 'fixed320']
torch.set_num_threads(THREADS)
R3 = P2 / os.environ.get('R320_PARAMS', 'fable_physicist_v1/refit6b/s6b_loopR3_joint_s23'); EP = os.environ.get('R320_EPOCH', '0')
_dv = json.load(open(R3 / f'dials_epoch{EP}.json')); _dv = _dv.get('dials', _dv)
JT.DIALS_T = JT.make_dials(torch, {k: float(_dv[k]) for k in JT.KEYS}, trainable=())
p, a, c, e = LC.payload(torch); sv = torch.load(R3 / f'params_epoch{EP}.pt')
with torch.no_grad():
    for k in p: p[k].copy_(sv['p'][k])
    a.copy_(sv['a']); c.copy_(sv['c'])
    for k in e: e[k].copy_(sv['e'][k])
inputs = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())
if arg == 'valid40': RANKS = inputs['valid'][:40]
elif arg == 'heldout40': RANKS = sorted(json.load(open(HERE / 'split_train640.json'))['new_ranks'])[::12][:40]
else: RANKS = [int(x) for x in arg.split(',')]
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
GRID = (256, 320, 352, 384)


def make_push(qr, joins, arm, log):
    fixed = int(arm[5:]) if arm.startswith('fixed') else None; tau = float(arm[5:]) if arm.startswith('adapt') else None
    sched = dict(zip(sorted(joins), (int(x) for x in arm[5:].split('-')))) if arm.startswith('sched') else None
    def push(t, layer, w32, p_full, lam, previous_indices, Q, F, core, cohort_stops, rank=384):
        wp = w32[previous_indices, :].T
        before = [c_ for L_, cs in joins.items() if L_ < layer for c_ in cs]
        old_columns = cohort_stops[max(before)] if before else 0
        if layer in joins:
            start, stop = old_columns, cohort_stops[max(joins[layer])]
            J = p_full[previous_indices, start:stop]; weight = lam[start:stop] ** 2
            Qp = Q[previous_indices, :] if old_columns else None
            d = (J * J) @ weight
            if old_columns: d = d + ((Qp @ core) * Qp).sum(1)
            want = fixed if fixed is not None else (sched[layer] if sched is not None else rank)
            rows = torch.argsort(-d.to(t.float64), stable=True)[:max(rank, want)]   # ENERGY order (nested truncations); v4: > 384 rows for raise arms
            Y = J @ (weight[:, None] * J[rows, :].T)
            if old_columns: Y = Y + Qp @ (core @ Qp[rows, :].T)
            Qn = qr(t, Y)
            Fj = Qn.T @ J
            ek = (Fj.to(t.float64) ** 2) @ weight.to(t.float64)
            if old_columns:
                overlap = Qn.T @ Qp
                ek = ek + ((overlap.to(t.float64) @ core.to(t.float64)) * overlap.to(t.float64)).sum(1)
            tot = float(d.to(t.float64).sum()); cum = t.cumsum(ek, 0)
            E = {r: float(cum[min(r, len(cum)) - 1]) / tot for r in GRID}
            if fixed is not None: r = fixed
            elif sched is not None: r = sched[layer]
            else: r = next((x for x in (256, 320, 384) if E[x] >= tau), 384)
            r = min(r, Qn.shape[1])
            Qn, Fj = Qn[:, :r], Fj[:r]
            joined = Fj @ (weight[:, None] * Fj.T)
            if old_columns:
                overlap = overlap[:r]
                F = t.cat((overlap @ F, Fj), dim=1); core = overlap @ (core @ overlap.T) + joined
            else:
                F, core = Fj, joined
            basis, old_columns = Qn, stop
            log['joins'].append(dict(layer=layer, r=int(r), E={str(k): round(v, 6) for k, v in E.items()}, live=int(len(previous_indices)),
                                     joined=int(stop - start)))
        elif old_columns:
            basis = Q[previous_indices, :]
        else:
            log['layers'].append(dict(layer=layer, old=0, rank=0))
            return wp @ p_full[previous_indices, :], None, F, core, 0
        log['layers'].append(dict(layer=layer, old=int(old_columns), rank=int(basis.shape[1])))
        U = wp @ basis
        return t.cat((U @ F, wp @ p_full[previous_indices, old_columns:]), dim=1), U, F, core, old_columns
    return push




def counting_select(cd, counts):
    orig = cd.select
    def sel(score):
        l, slot = cd.call_layer, cd.slot
        ids, alive = orig(score)
        counts.setdefault(str(l), {})[('alive', 'resp', 'births')[slot] if slot < 3 else f's{slot}'] = int(len(ids))
        return ids, alive
    cd.select = sel


with OUT.open('a') as fo:
    for r_ in RANKS:
        w = np.load(STORE / Path(man[r_]['path']).parent / 'weights.npy')
        tp = np.load(P2 / 'fable_refit_v1/truth688' / f'truthpost_rank{r_:04d}.npz')['mc_post'][15].astype(np.float64)
        for arm in ARMS:
            t0 = time.time(); cd = LC.card(torch, w, p, a, c, e); log = dict(joins=[], layers=[]); counts = {}
            cd.shared['push'] = make_push(cd.shared['_qr'], s2card.EVERY3, arm, log)
            if arm == 'fixed384': counting_select(cd, counts)
            with torch.no_grad(): pred = cd.run(False)[15].to(torch.float64).numpy()
            rec = dict(rank=r_, arm=arm, params=R3.name, epoch=int(EP), mse=float(((pred - tp) ** 2).mean()), wall=round(time.time() - t0, 1), counts=counts, **log)
            fo.write(json.dumps(rec) + '\n'); fo.flush()
            print(json.dumps(dict(rank=r_, arm=arm, mse=rec['mse'], wall=rec['wall'])), flush=True)
