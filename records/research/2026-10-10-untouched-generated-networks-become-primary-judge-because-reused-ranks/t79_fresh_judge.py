#!/usr/bin/env python3
"""TAG opus-physicist T79: CLEAN FRESH-NETWORK JUDGEMENT of the trainer's torch points (lead 10-10 13:3xZ): each candidate is one of the trainer's own requests
(O/trainer/run_M/claimed/<label>_r*.json: params, head weights, flags), replayed exactly as his worker applies it (worker.py serve()) on his current card module
(frozen copy otr_frozen_d61b46d1/otr_m.py = O/trainer/otr_m.py of 13:15Z, his otr_engine / otr_pl), on GENERATED networks that no refit batch, sizing, check or
judging set has touched: the analyst's fresh seeds (O/analyst/fresh/loader.py: weights = PCG64(seed) normals x sqrt(2/1024) as float32, the store's layout; truth =
64,094,208-sample Monte Carlo, two halves; final-layer truth noise about 1.2e-9). The engine reads weights and truth through data_paths(rank); for a seed it is
pointed at a scratch copy (weights.npy + truthpost npz with mc_post). Parity mode: 'parity' replays the label on its own first pool rank and prints the stored MSE.
-> t79_<tag>.jsonl ; usage: t79_fresh_judge.py <tag> <label,label,...> <seeds or 'parity'> [threads]"""
import hashlib, json, os, sys, time, re, glob, subprocess
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; O = HERE.parent
TAG = sys.argv[1]; LABELS = sys.argv[2].split(','); SEEDS = sys.argv[3]; threads = sys.argv[4] if len(sys.argv) > 4 else '2'
FROZEN = HERE / 'otr_frozen_d61b46d1'; SHA = 'd61b46d101a590da20fb170b29c368b576a8bb43019c6fd8e78834701d5f946f'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'TRAINER_THREADS'): os.environ[key] = threads
os.environ.update(TRAINER_PL='1', TRAINER_CARD='M', TRAINER_F4C='1')
sys.path = [x for x in sys.path if os.path.abspath(x or os.getcwd()) != str(HERE)]
sys.path.insert(0, str(O / 'trainer')); sys.path.insert(0, str(FROZEN))
import otr_engine as E
assert Path(E.M.__file__).resolve().parent == FROZEN and hashlib.sha256(Path(E.M.__file__).read_bytes()).hexdigest() == SHA, E.M.__file__
from otr_common import load_params, load_pl
torch = E.torch
sys.path.insert(0, str(O / 'analyst/fresh')); import loader as FL
SCRATCH = Path(os.environ.get('T79_SCRATCH', '/private/tmp/claude-501/-Users-matt-arc-white-box-estimation/fce1fde2-a6c3-4554-bf56-f6828c89d0f7/scratchpad/t79'))
SCRATCH.mkdir(parents=True, exist_ok=True)
paths0 = E.data_paths


def fresh_paths(seed):
    w, tr = SCRATCH / f'w_{seed}.npy', SCRATCH / f't_{seed}.npz'
    if not w.exists(): np.save(w, FL.weights(seed))
    if not tr.exists(): np.savez(tr, mc_post=np.asarray(FL.truth(seed)['mc_post'], dtype=np.float64))
    return w, tr
E.data_paths = lambda rank: fresh_paths(rank) if rank >= 7_000_000 else paths0(rank)


def apply(req):
    """the worker's serve(), copied (O/trainer/worker.py)"""
    E.set_parameters(*load_params(torch, req['params']))
    shifts = {f: list(v) for f, v in load_pl(torch, req['params']).items()}
    for family, values in (req.get('pl') or {}).items(): shifts[family] = [a + b for a, b in zip(shifts.get(family, [0.0] * 16), values)]
    E.PL.set_values(shifts)
    E.STAR_ATTACH = bool(req.get('attach', E.STAR_ATTACH)); E.KEEP_FINAL = False; E.KEEP_ALL = False; E.STAR_SS = req.get('ss')
    E.M.F4C = bool(req.get('f4c', E.M.F4C)); E.M.NCAR = req.get('ncar'); E.M.HEADF = int(req.get('hf') or 0); E.M.ROWS.clear(); E.M.RG = bool(req.get('rg'))
    E.M.DUMP15 = req.get('dump15'); E.M.B6 = bool(req.get('b6')); E.M.HSPLIT = {int(k): tuple(v) for k, v in (req.get('hsplit') or {}).items()}
    E.M.HBANDS = bool(req.get('hbands')); E.M.B7 = bool(req.get('b7'))
    if hasattr(E.M, 'CV'): E.M.CV = bool(req.get('cv'))
    if E.M.RG or E.M.RG_W: E.M.readout_weights(req.get('rgw'))
    if E.M.HEADF or E.M.HEADU is not None: E.M.head_weights(req.get('hu'))


REQS = {lab: json.loads(open(sorted(glob.glob(str(O / f'trainer/run_M/claimed/{lab}_r*.json')))[0]).read()) for lab in LABELS}
out = HERE / f't79_{TAG}.jsonl'; done = set()
if out.exists():
    for line in out.read_text().splitlines(): d = json.loads(line); done.add((d['label'], d['seed']))
if SEEDS == 'parity':
    for lab, req in REQS.items():
        apply(req); rec = E.run_network(req['rank'], 'eval')
        stored = json.loads((O / f"trainer/run_M/results/{req['id']}.json").read_text())['mse']
        print(json.dumps(dict(TAG='physicist T79 parity', label=lab, rank=req['rank'], mse=rec['mse'], stored=stored, rel=rec['mse'] / stored - 1)), flush=True)
    sys.exit(0)
seeds = [int(x) for x in SEEDS.split(',')]
for s in seeds:
    for lab, req in REQS.items():
        if (lab, s) in done: continue
        if (HERE / 'STOP_T79').exists(): print('STOP'); sys.exit(0)
        apply(req); t0 = time.time(); rec = E.run_network(s, 'eval')
        line = dict(label=lab, seed=s, mse=rec['mse'], mse_layers=rec['mse_layers'], wall=round(time.time() - t0, 1))
        with open(out, 'a') as fh: fh.write(json.dumps(line) + '\n')
        print(json.dumps(dict(TAG='physicist T79', label=lab, seed=s, final_x1e9=round(rec['mse'] * 1e9, 6), wall=line['wall'])), flush=True)
