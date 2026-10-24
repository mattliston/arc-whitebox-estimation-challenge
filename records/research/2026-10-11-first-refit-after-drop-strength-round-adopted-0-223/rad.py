"""TAG opus-trainer RAD (lead 10-11 ~15:00Z): REFIT-AFTER-DROP. The learned parts are re-fitted to an ABLATED card (a module dropped by the
physicist's audit and priced by the engineer), the Nd way: generated gradients (A36 7100264-327) -> arms -> sizing on 7100232-263 (A30 truths) ->
ONE frozen point -> check on fresh A37 7100360-391; the final look on the sealed 7100200-231 is the physicist's.
One config per round, cert/RAD/<name>/config.json:
  name       letters / digits only (ids a_R<name>...; run dirs run_RAD_<name>_gen (no-plan workers: gradients, check) and run_RAD_<name>_a30 (A30-plan
             workers: base, arms))
  patch      request fields set on EVERY request of the round (null removes the field): the ablation, e.g. a shorter 'ncar' list or 'pl' shifts
  params, hu the start point (default: Nd's frozen point)
  bill_saving_pct   the engineer's price of the drop (% of bill; 0 for a pure refit)
  reuse      optional {"grads": [run dir, label], "base": [run dir, id prefix]}: results already run with exactly these requests (e.g. Nd's)
  grads_without optional list of patch fields the round's gradients were run WITHOUT (added to the patch after the gradients; not checked on them)
  grad_extra optional request fields added to the GRADIENT requests only (e.g. {"edgex": [1, 7]}: the early edge's zero scales, gradients ride along)
  check_seeds, decide   optional: the fresh check set [first, last] (default A37 7100360-391) and the decision's reference ('nd' default, 'ab')
Stages (from O/trainer): python -B cert/RAD/rad.py <stage> cert/RAD/<name>/config.json [arms dir] [...]
  grads                     queue 64 gradient runs at the ablated start point on A36 (a_R<name>g_r<seed>)
  base                      queue the ablated start point on 7100232-263, final predictions kept (a_R<name>_base_r<rank>)
  build <arms dir> <dials> <fixed>   jmix_generated_v7.py arms from the round's gradients (env written to <arms dir>/arms_env.sh; dials / fixed
                            = JM_DIALS / JM_FIXED strings, '-' for none; gradient arms by the Nf settings)
  arms <arms dir>           queue the live arms on 7100232-263 (a_R<name>_<arm>_r<rank>)
  freeze <arms dir>[,<arms dir>...]   ridge least squares on the 32 sizing networks (ridge and arm set by leave-one-network-out among the
                            pre-declared candidates all / no dials / no fixed / fixed only) -> the ONE point = start + sum of weight x (arm file - start)
                            -> FROZEN_POINT_v1.json, before any check
  check                     queue A37: the frozen point and the ablated start (both patched) and Nd unpatched (a_R<name>C_<pt|ab|nd>_r<rank>)
  verdict                   pooled final-layer changes on A37: frozen vs Nd (the decision), frozen vs ablated (what the refit recovered), ablated vs
                            Nd (the raw drop); net score = (1 + frozen vs Nd)(1 - bill saving) - 1; ACCEPT iff net < -SE (per-network % changes)
Workers: bash cert/RAD/launch.sh <gen|a30> <index> <name>."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '.')
from otr_common import enqueue, publish, run_dirs
from otr_generated_manifest_v1 import TruthPlan
from otr_truth_records_v1 import validate_record

PY = '/Users/matt/anaconda3/bin/python'
A = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/analyst/'
P30 = TruthPlan(A + 'a30_edband_selection_and_final_20261011T0057Z/plan_v2.json', 'b35e8a1c7c83d4018ea44b8b01464f96069cab02f1583f2e0bfbfbc9ee758f4a',
                list(range(7100232, 7100264)))
SIZING = list(range(7100232, 7100264))
GRAD = [int(x) for x in Path('cert/Nf/gen_allow_grad_v1.txt').read_text().split(',')]
GRID = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc = lambda: os.popen('date -u +%FT%TZ').read().strip()
read = lambda p: json.loads(Path(p).read_text())

stage, CFG = sys.argv[1], Path(sys.argv[2]).resolve()
cfg, HOME = read(CFG), CFG.parent
NAME = cfg['name']
assert NAME.isalnum()
ND = read('cert/Nd/FROZEN_POINT_v1.json')
assert sha(ND['params']) == ND['params_sha256'] and sha(ND['head']) == ND['head_sha256']
P0, H0 = str(Path(cfg.get('params', ND['params'])).resolve()), str(Path(cfg.get('hu', ND['head'])).resolve())
GEN, A30 = Path('run_RAD_%s_gen' % NAME).resolve(), Path('run_RAD_%s_a30' % NAME).resolve()
SZ, SZKIND = (GEN, 'gen') if cfg.get('sizing_source') == 'fresh' else (A30, 'a30')      # where the sizing evals run and how their records are checked
reuse = cfg.get('reuse', {})
span = lambda x: list(range(int(x.split('-')[0]), int(x.split('-')[1]) + 1))      # 'lo-hi' (inclusive) -> ranks
CHECK = list(range(*(lambda a, b: (a, b + 1))(*cfg.get('check_seeds', [7100360, 7100391]))))      # fresh check set (inclusive range)
if 'sizing' in cfg:      # optional: the round's sizing set, [first, last] or a list of such ranges (e.g. A39 + A41)
    _sz = cfg['sizing'] if isinstance(cfg['sizing'][0], list) else [cfg['sizing']]
    SIZING = [r for lo, hi in _sz for r in range(lo, hi + 1)]
FRESH = cfg.get('sizing_source') == 'fresh'      # optional: size on fresh no-plan truths (analyst/fresh, e.g. A39) with the gen workers
DECIDE = cfg.get('decide', 'nd')      # the decision's reference: 'nd' = Nd unpatched, 'ab' = the round's own start point (a follow-up round)


def patched(request):
    out = dict(request)
    for key, value in cfg.get('patch', {}).items():
        if value is None:
            out.pop(key, None)
        else:
            out[key] = value
    return out


def template(kind):      # Nf's gradient request at Nd (no plan) or Nd's stage-1 sizing request (A30 plan; the check drops the plan)
    t = read(next(Path('run_Nf/claimed').glob('a_0mNf1g_r7100264.w*.json'))) if kind == 'grad' else read(next(Path('run_Nd/claimed').glob('a_0mNdC_pt_r7100232.w*.json')))
    return {k: v for k, v in t.items() if k != 'truth_plan'} if kind == 'check' else t


CUTSIM = patched(template('a30')).get('cutsim')


def record(path, params, kind, rank):
    value = read(path)
    assert value.get('cutsim') == CUTSIM and value['rank'] == rank, ('simulator / rank mismatch', path)
    if kind == 'a30':
        validate_record(P30, value, rank)
    else:
        assert value.get('truth_plan') is None and value.get('data_source') == 'generated', path
    assert Path(value['params']).resolve() == Path(params).resolve(), ('params mismatch', path, value['params'])
    return value


if stage == 'grads':
    assert 'grads' not in reuse
    t = template('grad'); run_dirs(GEN)
    n = sum(enqueue(GEN, dict(patched(dict(t, id='a_R%sg_r%d' % (NAME, r), rank=r, params=P0, hu=read(H0))), **cfg.get('grad_extra', {}))) for r in GRAD)
    print(n, 'gradient runs queued in', GEN.name)
elif stage == 'base':
    assert 'base' not in reuse
    t = template('check' if FRESH else 'a30'); run_dirs(SZ)
    ranks = span(sys.argv[3]) if len(sys.argv) > 3 else SIZING      # optional 'lo-hi': a part of the sizing set (two-stage sizing)
    n = sum(enqueue(SZ, patched(dict(t, id='a_R%s_base_r%d' % (NAME, r), rank=r, params=P0, hu=read(H0), keep_final=1))) for r in ranks)
    print(n, 'base evals queued in', SZ.name)
elif stage == 'fixed':      # fixed forward arms only, 'family:lo:hi:shift,...' (the jmix_generated_v7 JM_FIXED format), from the start point
    import torch
    from otr_common import load_params, load_pl, save_params
    D, spec = Path(sys.argv[3]).resolve(), sys.argv[4]
    assert not D.exists(), D
    D.mkdir(parents=True)
    arrays0, dials0 = load_params(torch, P0); pl0 = load_pl(torch, P0); live, size = [], {}
    for x in spec.split(','):
        fam, lo, hi, shift = x.split(':'); lo, hi, shift = int(lo), int(hi), float(shift)
        name = 'f%s_%d_%d' % (fam, lo, hi)
        pl = {k: [float(y) for y in v] for k, v in pl0.items()}
        pl[fam] = [y + (shift if lo <= l <= hi else 0.0) for l, y in enumerate(pl.get(fam, [0.0] * 16))]
        save_params(torch, D / ('arm_%s.pt' % name), arrays0, dials0, pl); (D / ('arm_%s_hu.json' % name)).write_text(json.dumps(read(H0)))
        live.append(name); size[name] = shift
    (D / 'arms.json').write_text(json.dumps(dict(live=live, size=size, base=P0, base_hu=H0, spec=spec), indent=1))
    print(len(live), 'fixed arms written to', D, ':', ' '.join(live))
elif stage == 'build':
    D, dials, fixed = Path(sys.argv[3]).resolve(), sys.argv[4], sys.argv[5]
    assert not D.exists(), D
    run, label = reuse.get('grads', [GEN.name, 'R%sg' % NAME])
    common = {k: v for k, v in patched(template('grad')).items() if k in ('f4c', 'ss', 'hf', 'b6', 'ncar', 'cutsim') or k in cfg.get('patch', {})}
    common = {k: v for k, v in common.items() if v is not None and k not in cfg.get('grads_without', [])}      # patch fields added after the
    # gradients ran (e.g. 'rdc' joining the S48 card before the base / arms stage): the arm directions come from the near-identical card without them
    env = dict(JM_RUN=run, JM_GRAD_NOPLAN='1', JM_GRAD_ALLOW='cert/Nf/gen_allow_grad_v1.txt', JM_ARR_SPLIT='1', JM_EDGE_SPLIT='4', JM_KC_SPLIT='1',
               JM_NO_KC='1', JM_DIALS='' if dials == '-' else dials, JM_FIXED='' if fixed == '-' else fixed, JM_DIAL_CAP='0.02', JM_CAP_ARR='0.002',
               JM_CAP_EDGE='0.05', JM_CAP_HU='1.0', JM_DROP_ARMS=cfg.get('drop_arms', 'erest,h20,h6'), JM_REQ=json.dumps(common),
               TRAINER_GENERATED_ALLOW_FILE='cert/refit/gen_worker_allow_v5.txt', TRAINER_TRUTH_PLAN=A + 'a27_fast_truth_20261011T0023Z/plan.json',
               TRAINER_TRUTH_PLAN_SHA256='71f455d1d7f5f7239503468cacd5710f4db9d595c7f1be389c8d937e4f64a979')
    subprocess.run([PY, '-B', 'cert/refit/jmix_generated_v7.py', 'arms', label, P0, H0, str(D)], env=dict(os.environ, **env), check=True)
    (D / 'arms_env.sh').write_text(''.join("%s=%s\n" % (k, "'%s'" % v if k == 'JM_REQ' else v) for k, v in env.items()))
elif stage == 'arms':
    D = Path(sys.argv[3]).resolve()
    t = template('check' if FRESH else 'a30'); run_dirs(SZ); live = read(D / 'arms.json')['live']
    ranks = span(sys.argv[4]) if len(sys.argv) > 4 else SIZING      # optional 'lo-hi' and an optional comma list of the arms to run
    live = [a for a in live if len(sys.argv) <= 5 or a in sys.argv[5].split(',')]
    n = sum(enqueue(SZ, patched(dict(t, id='a_R%s_%s_r%d' % (NAME, a, r), rank=r, params=str(D / ('arm_%s.pt' % a)), hu=read(D / ('arm_%s_hu.json' % a)), keep_final=1)))
            for a in live for r in ranks)
    print(n, 'arm evals queued in', SZ.name, ':', len(live), 'arms x %d:' % len(ranks), ' '.join(live))
elif stage in ('freeze', 'prune'):      # prune <dirs> <lo-hi> <out json>: stage 1 of a two-stage sizing; freeze <dirs> [<prune json>]: the solve on all 32
    DIRS = [Path(x).resolve() for x in sys.argv[3].split(',')]      # one or more arms dirs (their live arms, names unique)
    FROZEN, SOLVE, P = HOME / 'FROZEN_POINT_v1.json', HOME / 'solve_v1.json', HOME / 'point_v1'
    PRUNE = stage == 'prune'
    RANKS = span(sys.argv[4]) if PRUNE else SIZING
    KEEP = None if PRUNE or len(sys.argv) <= 4 else read(sys.argv[4])['keep']
    assert PRUNE or (not FROZEN.exists() and not SOLVE.exists() and not P.exists())
    HOMEDIR = {a: d for d in DIRS for a in read(d / 'arms.json')['live']}
    ARMS = [a for d in DIRS for a in read(d / 'arms.json')['live'] if KEEP is None or a in KEEP]
    assert len(set(ARMS)) == len(ARMS) and len(HOMEDIR) == sum(len(read(d / 'arms.json')['live']) for d in DIRS), 'arm names repeat across the arms dirs'
    brun, bpre = reuse.get('base', [SZ.name, 'a_R%s_base' % NAME])
    D_, res, inputs = [], [], []
    for r in RANKS:
        if FRESH:      # the no-plan worker's truth: analyst/fresh/truth_seed<r>.npz (16M samples or more)
            entry = dict(path=A + 'fresh/truth_seed%d.npz' % r)
            with np.load(entry['path'], allow_pickle=False) as z:
                assert int(z['N']) >= 15990784, (r, int(z['N']))
        else:
            entry, expected, _ = P30.resolve(r)
            assert sha(entry['path']) == expected
        with np.load(entry['path'], allow_pickle=False) as z:
            truth = np.asarray(z['mc_post'], dtype=np.float64)[15]
        bpath = Path(brun) / 'results' / ('%s_r%d.json' % (bpre, r))
        p0 = np.array(record(bpath, P0, SZKIND, r)['final_prediction']); inputs.append(dict(rank=r, arm='base', path=str(bpath), sha256=sha(bpath)))
        cols = []
        for a in ARMS:
            path = SZ / 'results' / ('a_R%s_%s_r%d.json' % (NAME, a, r))
            cols.append(np.array(record(path, HOMEDIR[a] / ('arm_%s.pt' % a), SZKIND, r)['final_prediction']) - p0)
            inputs.append(dict(rank=r, arm=a, path=str(path), sha256=sha(path)))
        D_.append(np.stack(cols, 1)); res.append(truth - p0)
    N, E0 = len(RANKS), sum(float((x ** 2).sum()) for x in res)
    AA = [d.T @ d for d in D_]; BB = [d.T @ x for d, x in zip(D_, res)]; A_all, B_all = sum(AA), sum(BB)

    def fit(cols, lam, leave=None):
        Am = A_all - (AA[leave] if leave is not None else 0); Bm = B_all - (BB[leave] if leave is not None else 0)
        Am = Am[np.ix_(cols, cols)]; Bm = Bm[cols]
        return np.linalg.solve(Am + lam * np.diag(np.diag(Am)), Bm)

    def loo(cols, lam):
        return sum(float(((res[i] - D_[i][:, cols] @ fit(cols, lam, i)) ** 2).sum()) for i in range(N)) / E0 - 1

    moved = [a for k, a in enumerate(ARMS) if float(A_all[k, k]) > 0]      # an arm that moved nothing (no element beyond 2.5 se) stays at 0
    if PRUNE:      # pre-declared rule (lead 19:3xZ): keep an arm iff its single-arm ridge LOO (best over the grid) on these networks is < 0
        alone = {a: 100 * min(loo([ARMS.index(a)], x) for x in GRID) for a in moved}
        keep = [a for a in moved if alone[a] < 0]
        out = Path(sys.argv[5]); assert not out.exists()
        publish(out, lambda stream: json.dump(dict(schema='trainer_RAD_prune_v1', utc=utc(), round=NAME, ranks=RANKS, arms=ARMS, alone_loo_pct=alone, keep=keep,
                                                  rule='keep iff single-arm ridge LOO < 0 on these networks'), stream, indent=1))
        print('prune on %d networks: single-arm LOO (%%): %s' % (N, ', '.join('%s %+.3f' % kv for kv in sorted(alone.items(), key=lambda kv: kv[1]))))
        print('KEEP (%d of %d): %s' % (len(keep), len(ARMS), ' '.join(keep)))
        raise SystemExit(0)
    cand = dict(all=moved, no_dials=[a for a in moved if not a.startswith('d')], no_fixed=[a for a in moved if not a.startswith('f')],
                fixed_only=[a for a in moved if a.startswith('f')])
    idx = {}
    for name, arms in cand.items():      # pre-declared candidates, duplicates and empty sets dropped
        cols = [ARMS.index(a) for a in arms]
        if cols and cols not in idx.values():
            idx[name] = cols
    tables = {name: {str(lam): 100 * loo(cols, lam) for lam in GRID} for name, cols in idx.items()}
    best = {name: min(GRID, key=lambda x: t[str(x)]) for name, t in tables.items()}
    chosen = min(idx, key=lambda n: (tables[n][str(best[n])], n != 'all'))
    null = tables[chosen][str(best[chosen])] >= 0      # no candidate beats the start point out of sample: the frozen point is the start point
    w = np.zeros(len(ARMS))
    if not null:
        w[idx[chosen]] = fit(idx[chosen], best[chosen])
    model = sum(float(((res[i] - D_[i] @ w) ** 2).sum()) for i in range(N)) / E0 - 1
    alone = {a: 100 * min(loo([ARMS.index(a)], x) for x in GRID) for a in moved}
    s = dict(schema='trainer_RAD_solve_v1', utc=utc(), round=NAME, networks=SIZING, arms=ARMS, arm_set=chosen, null=bool(null),
             candidates={n: dict(arms=[ARMS[k] for k in idx[n]], ridge=best[n], loo_pct=tables[n][str(best[n])], grid=tables[n]) for n in idx},
             ridge=best[chosen], weights=dict(zip(ARMS, map(float, w))), model_pct=100 * model, loo_pct=tables[chosen][str(best[chosen])],
             arm_alone_loo_pct=alone, cutsim=CUTSIM, truth_plan=None if FRESH else P30.spec(), inputs=inputs,
             arm_files={a: dict(params=str(HOMEDIR[a] / ('arm_%s.pt' % a)), params_sha256=sha(HOMEDIR[a] / ('arm_%s.pt' % a)),
                                hu=str(HOMEDIR[a] / ('arm_%s_hu.json' % a)), hu_sha256=sha(HOMEDIR[a] / ('arm_%s_hu.json' % a))) for a in ARMS})
    publish(SOLVE, lambda stream: json.dump(s, stream, indent=1))
    print('candidates (best LOO at ridge): ' + '; '.join('%s %+.3f%% at %s' % (n, tables[n][str(best[n])], best[n]) for n in idx) + ' -> FROZEN SET: '
          + chosen + (' (NULL: frozen point = start point)' if null else ''))
    print('model %+.3f%%, LOO %+.3f%%; weights: %s' % (100 * model, s['loo_pct'], ', '.join('%s %+.3f' % (a, x) for a, x in zip(ARMS, w))))
    print('arm alone LOO (%): ' + ', '.join('%s %+.3f' % kv for kv in alone.items()))
    # the point = start + sum over arms of weight x (arm file - start): every arm file is the start plus its trial step (arrays, per-layer shifts,
    # head weights), so this is the solved mixture whatever the arm's kind (array part, head group, gradient dial, fixed forward arm)
    import torch
    from otr_common import load_params, load_pl, save_params
    P.mkdir()
    arrays0, dials0 = load_params(torch, P0); pl0 = load_pl(torch, P0); U0 = np.array(read(H0), dtype=np.float64)
    arrays = {k: v.double() for k, v in arrays0.items()}; pl = {k: np.array(v, dtype=np.float64) for k, v in pl0.items()}; U = U0.copy()
    for a, x in zip(ARMS, w):
        if x == 0:
            continue
        arrays_a, dials_a = load_params(torch, HOMEDIR[a] / ('arm_%s.pt' % a)); pl_a = load_pl(torch, HOMEDIR[a] / ('arm_%s.pt' % a))
        assert dials_a == dials0, ('an arm moved a global dial', a)
        for k in arrays:
            arrays[k] = arrays[k] + x * (arrays_a[k].double() - arrays0[k].double())
        for f in set(pl_a) | set(pl0):
            pl[f] = pl.get(f, np.zeros(16)) + x * (np.array(pl_a.get(f, [0.0] * 16)) - np.array(pl0.get(f, [0.0] * 16)))
        Ua = np.array(read(HOMEDIR[a] / ('arm_%s_hu.json' % a)), dtype=np.float64)
        U = U + x * (Ua - U0)
    save_params(torch, P / 'params.pt', {k: v.to(torch.float32) for k, v in arrays.items()}, dials0, {f: v.tolist() for f, v in pl.items()})
    (P / 'hu.json').write_text(json.dumps(U.tolist()))
    moved = {f: [round(float(y), 4) for y in pl[f] - np.array(pl0.get(f, [0.0] * 16))] for f in pl if np.abs(pl[f] - np.array(pl0.get(f, [0.0] * 16))).max() > 0}
    print('point: per-layer shifts moved', moved, '| arrays moved', sum(int((arrays[k].to(torch.float32) != arrays0[k]).sum()) for k in arrays),
          'elements | head weights moved', int((U != U0).sum()))
    rec = dict(schema='trainer_RAD_frozen_point_v1', utc=utc(), round=NAME, config=str(CFG), config_sha256=sha(CFG), patch=cfg.get('patch', {}),
               bill_saving_pct=float(cfg.get('bill_saving_pct', 0.0)), params=str(P / 'params.pt'), params_sha256=sha(P / 'params.pt'),
               head=str(P / 'hu.json'), head_sha256=sha(P / 'hu.json'), start_params=P0, start_params_sha256=sha(P0), start_head=H0,
               start_head_sha256=sha(H0), solve=dict(path=str(SOLVE), sha256=sha(SOLVE)), arm_set=chosen, null=bool(null), ridge=s['ridge'],
               weights=s['weights'], model_pct=s['model_pct'], loo_pct=s['loo_pct'],
               rule='check on fresh %d-%d: ACCEPT iff net score change (1 + frozen vs %s)(1 - bill saving) - 1 < -SE of the per-network %% changes' % (
                   CHECK[0], CHECK[-1], 'Nd' if DECIDE == 'nd' else 'the start point'))
    publish(FROZEN, lambda stream: json.dump(rec, stream, indent=1))
    print('FROZEN', rec['params_sha256'][:12], rec['head_sha256'][:12], 'set', chosen, 'ridge', rec['ridge'], 'LOO %+.3f%%' % rec['loo_pct'], '(null)' if null else '')
elif stage == 'check':
    F = read(HOME / 'FROZEN_POINT_v1.json')
    assert sha(F['params']) == F['params_sha256'] and sha(F['head']) == F['head_sha256'] and F['config_sha256'] == sha(CFG)
    t = template('check'); run_dirs(GEN); n = 0
    ablated = bool(cfg.get('patch')) or Path(P0) != Path(ND['params']).resolve()
    for r in CHECK:
        n += enqueue(GEN, patched(dict(t, id='a_R%sC_pt_r%d' % (NAME, r), rank=r, params=F['params'], hu=read(F['head']), keep_final=1)))
        if ablated:
            n += enqueue(GEN, patched(dict(t, id='a_R%sC_ab_r%d' % (NAME, r), rank=r, params=P0, hu=read(H0), keep_final=1)))
        if cfg.get('check_nd', True):      # optional (config 'check_nd': false): the Nd reference arm, not needed when the decision is against 'ab'
            n += enqueue(GEN, dict(t, id='a_R%sC_nd_r%d' % (NAME, r), rank=r, params=ND['params'], hu=read(ND['head']), keep_final=1))
    print(n, 'check evals queued in', GEN.name)
elif stage == 'verdict':
    F = read(HOME / 'FROZEN_POINT_v1.json'); OUT = HOME / 'CHECK_VERDICT_v1.json'
    assert not OUT.exists()
    ablated = bool(cfg.get('patch')) or Path(P0) != Path(ND['params']).resolve()
    rows = []
    for r in CHECK:
        nd_cutsim = template('check').get('cutsim')
        get = lambda kind, params: record(GEN / 'results' / ('a_R%sC_%s_r%d.json' % (NAME, kind, r)), params, 'gen', r)
        pt = get('pt', F['params'])
        if cfg.get('check_nd', True):
            nd = read(GEN / 'results' / ('a_R%sC_nd_r%d.json' % (NAME, r)))
            assert nd.get('cutsim') == nd_cutsim and nd.get('truth_plan') is None and Path(nd['params']).resolve() == Path(ND['params']).resolve()
        else:
            nd = dict(mse=float('nan'))      # no Nd arm: pt / ab vs Nd are not computed (nan)
        ab = get('ab', P0) if ablated else nd
        rows.append(dict(rank=r, nd=nd['mse'], ab=ab['mse'], pt=pt['mse']))

    def change(a, b):
        x, y = np.array([q[a] for q in rows]), np.array([q[b] for q in rows]); ch = 100 * (x / y - 1)
        return dict(pooled_pct=100 * (x.sum() / y.sum() - 1), se_pct=float(ch.std(ddof=1) / len(ch) ** .5), better=int((x < y).sum()))
    v = dict(pt_vs_nd=change('pt', 'nd'), pt_vs_ab=change('pt', 'ab'), ab_vs_nd=change('ab', 'nd'))
    bill = float(cfg.get('bill_saving_pct', 0.0))
    ref = 'pt_vs_nd' if DECIDE == 'nd' else 'pt_vs_ab'
    net = 100 * ((1 + v[ref]['pooled_pct'] / 100) * (1 - bill / 100) - 1)
    verdict = dict(schema='trainer_RAD_check_v1', utc=utc(), round=NAME, ranks=CHECK, bill_saving_pct=bill, net_score_change_pct=net,
                   accepted=bool(net < -v[ref]['se_pct']), decided_against=ref, rule=F['rule'], frozen=dict(path=str(HOME / 'FROZEN_POINT_v1.json'),
                   sha256=sha(HOME / 'FROZEN_POINT_v1.json')), rows=rows, **v)
    publish(OUT, lambda stream: json.dump(verdict, stream, indent=1))
    f = lambda k: '%+.3f%% (se %.3f, better %d/32)' % (v[k]['pooled_pct'], v[k]['se_pct'], v[k]['better'])
    print('RAD %s CHECK on %d-%d: frozen vs Nd %s; frozen vs start %s; start vs Nd %s; bill saving %.2f%% -> net score %+.3f%% (vs %s) -> %s' % (
        NAME, CHECK[0], CHECK[-1], f('pt_vs_nd'), f('pt_vs_ab'), f('ab_vs_nd'), bill, net, ref, 'ACCEPTED' if verdict['accepted'] else 'NOT ACCEPTED'))
else:
    raise SystemExit('unknown stage ' + stage)
