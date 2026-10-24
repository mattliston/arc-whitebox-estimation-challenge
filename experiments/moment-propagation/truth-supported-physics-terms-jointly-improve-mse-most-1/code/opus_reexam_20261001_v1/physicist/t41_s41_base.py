#!/usr/bin/env python3
"""TAG opus-physicist T41: THE PHYSICS BASE ON THE TORCH CARD OF THE FILE ON THE BOARD (lead 18:01Z: 'first dials + table'). S41's torch card = the trainer's pool card (run M card
with the exact-term module, star by row sums) at his FIRST joint dial solve (label c3q4zd6, parameter file O/trainer/cert/fold/c3q4zd6_optimum.pt, request fields f4c 1, ss 0,
ncar {'module': 'ncar6', 'forms': 3} = the pair law at level 'read' with the per-layer TABLE (dfit 2) and three forms) + {'module': 'k31l'} (the young-pair plane on cohorts of ages
1-2 at strength 1, level 'read'). The card is built and run by the trainer's own engine (O/trainer/otr_engine.py run_network, forward only); otr_m.py is a frozen copy in this lane
(otr_frozen_cbe85587, the live file of 17:55Z). The physics base is one more module of the request, {'module': 'pbase', <keywords>} (physicist/pbase_frozen_s3x.py); its entry in the
module table is made in this process only (nothing of the trainer's lane is written).
LEARNED MEAN CORRECTION OFF = the trainer's leaf ds at -1 on every layer (delta x 0 at layers 1-15, what pcard.build(learned=False) does); note the first dials already carry
ds = -0.2156 (the learned correction at 0.784 of its trained size).
arms: name:key=value,...
   off=1        learned mean correction off          ds=<x>   the learned correction times x (in place of the file's 0.784)
   law=0, k3l=0 without the pair law / the young-pair plane (parity arms)
   mod=1        the pbase module installed with every keyword at 0 (must change nothing)
   so2m so2c so2t so2f bsk gsk gk4 gbk x1 x2      pbase's keywords
   pl.<family>=<x>   a shift of one of the pool card's per-layer dial families on all layers, added to the file's (e.g. pl.k31=0.456)
output: one JSON line per arm: final_x1e9 = final-layer MSE x 1e9 against the organizers' truth (1e9 samples), per_layer_x1e9 = the 16 layers.
usage: t41_s41_base.py <rank> <threads> <tag> arm [arm ...]"""
import hashlib, json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; O = HERE.parent
rank = int(sys.argv[1]); threads = str(int(sys.argv[2])); tag = sys.argv[3]; ARMS = sys.argv[4:]
FROZEN = HERE / 'otr_frozen_cbe85587'; SHA = 'cbe85587357d9a89c986e53fae80199266588e69e27d7fa6fde87ada5a38bb90'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'TRAINER_THREADS'): os.environ[key] = threads
os.environ.update(TRAINER_PL='1', TRAINER_CARD='M', TRAINER_F4C='1')
sys.path = [x for x in sys.path if os.path.abspath(x or os.getcwd()) != str(HERE)]      # nothing of the card may resolve to a file of my lane
sys.path.insert(0, str(O / 'trainer')); sys.path.insert(0, str(FROZEN))
import otr_engine as E
assert Path(E.M.__file__).resolve().parent == FROZEN and hashlib.sha256(Path(E.M.__file__).read_bytes()).hexdigest() == SHA, E.M.__file__
from otr_common import load_params, load_pl
torch = E.torch; t0 = time.time()
PBKEYS = ('so2m', 'so2c', 'so2t', 'so2f', 'bsk', 'gsk', 'gk4', 'gbk', 'x1', 'x2')
E.M.MODULES['pbase'] = ('pbase_frozen_s3x', dict({k: 0.0 for k in PBKEYS}, lo=1, hi=14))
assert E.M._ncar('pbase').ARGS == E.M.MODULES['pbase'][1]
PARAMS = O / 'trainer/cert/fold/c3q4zd6_optimum.pt'
E.set_parameters(*load_params(torch, PARAMS)); PL0 = {f: [float(x) for x in v] for f, v in load_pl(torch, PARAMS).items()}
print(json.dumps(dict(TAG='physicist T41 S41 torch card + pbase', rank=rank, tag=tag, params=str(PARAMS), otr_m=SHA[:8], threads=threads, file_ds=PL0.get('ds', [0.0])[1])), flush=True)
for spec in ARMS:
    name, _, opts = spec.partition(':'); o = dict(off=0.0, law=1.0, k3l=1.0, mod=0.0); pl = {}
    for kv in filter(None, opts.split(',')):
        k, v = kv.split('=')
        if k.startswith('pl.'): pl[k[3:]] = float(v)
        else: o[k] = float(v)
    assert set(o) <= {'off', 'law', 'k3l', 'mod', 'ds'} | set(PBKEYS), o
    shifts = {f: list(v) for f, v in PL0.items()}
    if o['off']: shifts['ds'] = [-1.0] * 16
    elif 'ds' in o: shifts['ds'] = [o['ds'] - 1.0] * 16
    for f, x in pl.items(): shifts[f] = [a + x for a in shifts.get(f, [0.0] * 16)]
    E.PL.set_values(shifts)
    ncar = ([{'module': 'ncar6', 'forms': 3}] if o['law'] else []) + ([{'module': 'k31l'}] if o['k3l'] else [])
    pb = {k: o[k] for k in PBKEYS if o.get(k)}
    if pb or o['mod']: ncar.append(dict(module='pbase', **pb))
    E.STAR_ATTACH = False; E.KEEP_FINAL = False; E.STAR_SS = 0; E.M.F4C = True; E.M.NCAR = ncar or None; E.M.HEADF = False
    t1 = time.time(); rec = E.run_network(rank, 'eval')
    print(json.dumps(dict(rank=rank, tag=tag, arm=name, opts=o, pl=pl, ncar=ncar, final_x1e9=rec['mse'] * 1e9, per_layer_x1e9=[x * 1e9 for x in rec['mse_layers']],
                          seconds=round(time.time() - t1, 1), wall=round(time.time() - t0, 1))), flush=True)
