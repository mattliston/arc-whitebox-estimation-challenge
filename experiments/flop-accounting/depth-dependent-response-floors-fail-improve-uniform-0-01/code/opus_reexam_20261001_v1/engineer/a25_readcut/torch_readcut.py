"""TAG opus_engineer a25 (2026-10-02): does the third-order plane READ need all alive columns? Torch run M card, forward only.
The read (27% of the native bill) forms cp[i, j] = sum_r lam_r pp_ir^2 pp_jr for the response rows i and ALL alive units j
(gate_j >= 0.01). The alive roster is sorted by gate, so a native build could read a leading slice of it at no extra cost. Here
the columns of the plane whose unit has gate below a floor are set to 0 (the units stay alive for everything else):
  fc    read-column floor on the gate (0.01 = today's alive roster)
  lo    first layer the floor applies to (default 1)      hi    last layer (default 14)
arms: name:fc=<floor>[,lo=<l>,hi=<l>][,family=shift ...]
usage: TRAINER_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 TRAINER_PL=1 TRAINER_CARD=M \\
       python -B torch_readcut.py <rank> <out.npz (new)> arm [arm ...]
Output: pred_<arm> per arm; one JSON line per arm (final-layer MSE; per layer the alive columns and the columns read)."""
import json
import sys
import time
from pathlib import Path

O = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(O/'trainer'))
import numpy as np
import otr_engine as E
from otr_common import data_paths

torch = t = E.torch
rank, out, arms = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
assert not out.exists() and E.M is not None and E.PL is not None
weights_path, truth_path = data_paths(rank)
weights = np.load(weights_path, allow_pickle=False)
truth = np.load(truth_path, allow_pickle=False)['mc_post'].astype(np.float64)


def install_readcut(cd, o, log):
    st = {}
    select0, read0 = cd.select, cd.shared['read']

    def select(score):                      # the first selection of a step is the alive roster, scored by the gate
        if cd.slot == 0:
            st['gate'] = score.detach()
        return select0(score)
    cd.select = select

    def read(tt, pp, lam, response_indices, live_indices, U, F, old_columns):
        plane = read0(tt, pp, lam, response_indices, live_indices, U, F, old_columns)
        layer = cd.call_layer
        keep = st['gate'][live_indices] >= o['fc']
        if not (o['lo'] <= layer <= o['hi']):
            keep = t.ones_like(keep)
        log.append([layer, int(keep.numel()), int(keep.sum())])
        return plane * keep.to(plane.dtype)[None, :]
    cd.shared['read'] = read


saved = {}
for spec in arms:
    name, _unused, opts = spec.partition(':')
    o = dict(fc=0.0, lo=1.0, hi=14.0)
    shifts = {}
    for item in filter(None, opts.split(',')):
        key, value = item.split('=')
        if key in o:
            o[key] = float(value)
        else:
            shifts[key] = [float(value)] * 16
    E.PL.set_values(shifts)
    E.LC.JT.DIALS_T = E.DL
    log = []
    t0 = time.monotonic()
    with torch.no_grad():
        cd = E.LC.card(torch, weights, E.p, E.T['a'], E.T['c'], E.e, fine_ckpt=False)
        cd.hooks['k4_15'] = lambda layer, value, ctx: value
        E.install_star(cd)
        E.PL.install(E, cd)
        if o['fc'] > 0:
            install_readcut(cd, o, log)
        E.M.install(cd)
        prediction = cd.run(False).detach().to(torch.float64).numpy().copy()
    del cd
    E.collect()
    assert prediction.shape == (16, 1024) and np.isfinite(prediction).all()
    saved['pred_' + name] = prediction
    mse = ((prediction - truth) ** 2).mean(1)
    print(json.dumps(dict(rank=rank, arm=name, opts={k: v for k, v in o.items() if v != dict(fc=0.0, lo=1.0, hi=14.0)[k]},
                          final_x1e9=round(float(mse[15]) * 1e9, 4), columns=log, seconds=round(time.monotonic() - t0, 1))), flush=True)
np.savez(out, **saved)
