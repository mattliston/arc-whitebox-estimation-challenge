"""TAG opus_engineer a20 (2026-10-02): torch reference PREDICTIONS of the run M card (O/trainer/otr_m.py through O/trainer/otr_engine.py)
at chosen dial shifts, one network, for the native parity rehearsal of the run M export. Forward only (no_grad).
Parameters: RRPI's untrained ones (the engine's default), or --params <file.pt> (its arrays and dials; its stored 'pl' shifts are
added to every arm's).
arms: name:family=shift,family=shift,...   one shift per family, applied on all 16 layers, ADDED to the M card's base values
      (F4 strength 0.6 per age: f4a1, f4a2; K4L strength 1 per age: k4la1, k4la2; k22, bsrc, rc, ... = the families of otr_pl.py / otr_m.py)
The forward is run_network's eval path of otr_engine.py with the prediction kept (the engine returns only the MSE).
usage: TRAINER_THREADS=3 OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3 TRAINER_PL=1 TRAINER_CARD=M \
       python -B torch_arms.py <rank> <out.npz (new)> [--params <file.pt>] arm [arm ...]
Output: <out.npz> with pred_<arm> (16 x 1024 float64) per arm; one JSON line per arm on stdout (per-layer MSE against the truth)."""
import json
import sys
import time
from pathlib import Path

O = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(O/'trainer'))
import numpy as np
import otr_engine as E
from otr_common import data_paths, load_params, load_pl

torch = E.torch
args = sys.argv[1:]
stored = {}
if '--params' in args:
    i = args.index('--params'); path = args[i + 1]; del args[i:i + 2]
    arrays, dials = load_params(torch, path)
    E.set_parameters(arrays, dials)
    stored = {family: [float(x) for x in values] for family, values in load_pl(torch, path).items()}
rank, out, arms = int(args[0]), Path(args[1]), args[2:]
assert not out.exists() and E.M is not None and E.PL is not None
weights_path, truth_path = data_paths(rank)
weights = np.load(weights_path, allow_pickle=False)
truth = np.load(truth_path, allow_pickle=False)['mc_post'].astype(np.float64)
saved = {}
for spec in arms:
    name, _unused, opts = spec.partition(':')
    shifts = {family: list(values) for family, values in stored.items()}
    for item in filter(None, opts.split(',')):
        family, value = item.split('=')
        base = shifts.get(family, [0.0] * 16)
        shifts[family] = [x + float(value) for x in base]
    E.PL.set_values(shifts)
    E.LC.JT.DIALS_T = E.DL
    t0 = time.monotonic()
    with torch.no_grad():
        cd = E.LC.card(torch, weights, E.p, E.T['a'], E.T['c'], E.e, fine_ckpt=False)
        cd.hooks['k4_15'] = lambda layer, value, ctx: value
        E.install_star(cd)
        E.PL.install(E, cd)
        E.M.install(cd)
        prediction = cd.run(False).detach().to(torch.float64).numpy().copy()
    del cd
    E.collect()
    assert prediction.shape == (16, 1024) and np.isfinite(prediction).all()
    saved['pred_' + name] = prediction
    mse = ((prediction - truth) ** 2).mean(1)
    print(json.dumps(dict(rank=rank, arm=name, shifts={family: values[1] for family, values in shifts.items() if any(values)},
                          final_x1e9=round(float(mse[15]) * 1e9, 4), layers_x1e9=[round(float(x) * 1e9, 3) for x in mse],
                          seconds=round(time.monotonic() - t0, 1))), flush=True)
np.savez(out, **saved)
