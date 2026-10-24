"""TAG opus-trainer (2026-10-01): persistent worker of the parallel refit trainer. Loads torch + the card once (otr_engine), then
serves requests from <run>/queue until <run>/STOP exists. A request is one JSON file {id, mode: grad|eval, rank, params, scale};
the worker claims it by renaming it into <run>/claimed/<id>.w<k>.json (atomic: one winner), runs ONE network and publishes
<run>/results/<id>.pt (grad: gradients of the 23 arrays + 10 dials, final-layer MSE, all-layer MSE) or <id>.json (eval: MSE of all
16 layers). A request that raises publishes <id>.error (the traceback) and the worker carries on. On start it first finishes
requests its previous incarnation claimed but never answered. It leaves with code 3 when its resident memory at rest exceeds
7.5 GB (the supervisor restarts it behind the machine gate).
usage: TRAINER_THREADS=1 OMP_NUM_THREADS=1 ... python -B worker.py <run dir> <worker index>   (started by supervisor.py)"""
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

RUN, INDEX = Path(sys.argv[1]).resolve(), int(sys.argv[2])
TAG = 'w%d' % INDEX
RSS_LIMIT = int(float(os.environ.get('TRAINER_RSS_LIMIT', 7.5e9)))   # raise it when TRAINER_RELIEF=0 keeps freed memory resident

import psutil
import otr_engine as E
from otr_common import error_path, load_params, load_pl, publish, result_path

torch = E.torch
process = psutil.Process()
loaded, loaded_pl = None, {}
# Astra versioned offline adapter: the live worker.py and engine remain unchanged.
from otr_generated_v1 import make_data_paths
from otr_cut_runtime_v2 import enable as enable_cutsim
allowed_file = os.environ.get('TRAINER_GENERATED_ALLOW_FILE')
allowed_seeds = ([int(x) for x in Path(allowed_file).read_text().strip().split(',')] if allowed_file else [])
E.data_paths = make_data_paths(E.data_paths, allowed_seeds, RUN / 'generated_cache_v1')
configure_cutsim = enable_cutsim(E)
ATTACH = E.STAR_ATTACH   # this worker's default for the star basis in the backward pass; a request may carry its own 'attach'
F4C = E.M.F4C if E.M is not None else False   # default from env TRAINER_F4C; a request may carry its own 'f4c'


def say(**record):
    print(json.dumps(dict(worker=TAG, pid=os.getpid(), time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **record)),
          flush=True)


def serve(claim):
    global loaded, loaded_pl
    request = json.loads(claim.read_text())
    out, failed = result_path(RUN, request), error_path(RUN, request)
    if out.exists() or failed.exists():
        return
    try:
        configure_cutsim(request.get('cutsim'))
        E.UNFREEZE = bool(request.get('unfreeze'))     # the September-frozen arrays may differ / get gradients (otr_engine.py, UNFREEZE)
        if request['params'] != loaded:
            loaded = None
            E.set_parameters(*load_params(torch, request['params']))
            loaded_pl = load_pl(torch, request['params'])
            loaded = request['params']
        shifts = {family: list(values) for family, values in loaded_pl.items()}   # stored with the parameters ...
        for family, values in (request.get('pl') or {}).items():                  # ... plus this request's own (scans)
            shifts[family] = [a + b for a, b in zip(shifts.get(family, [0.0] * 16), values)]
        assert E.PL is not None or not shifts, 'per-layer dial shifts need a worker started with TRAINER_PL=1'
        if E.PL is not None:
            E.PL.set_values(shifts)
        E.STAR_ATTACH = bool(request.get('attach', ATTACH))
        E.KEEP_FINAL = bool(request.get('keep_final', False))
        E.KEEP_ALL = bool(request.get('keep_all', False))
        E.STAR_SS = request.get('ss')                 # None, or the factor rank of the engineer's star by row sums (0 = S39's star)
        if E.M is not None:   # the exact-term module in place of the F4 block (otr_m.py), per request
            E.M.F4C = bool(request.get('f4c', F4C))
            E.M.NCAR = request.get('ncar')            # None, or the carrier's install arguments (otr_m.NCAR_ARGS overridden by these)
            E.M.HEADF = int(request.get('hf') or 0)   # new head-input columns on (otr_m.py, HEAD INPUTS): 1 all, 2 the diagonal columns only
            E.M.ROWS.clear()
            E.M.RG = bool(request.get('rg'))          # the three zero-weight readouts (otr_m.py, READOUTS)
            E.M.DUMP15 = request.get('dump15')        # path of the layer-15 dump (otr_m.py, DUMP15) or None
            E.M.B6 = bool(request.get('b6'))          # the physicist's batch-6 head columns (O/physicist/b6cols.py)
            E.M.HSPLIT = {int(k): tuple(v) for k, v in (request.get('hsplit') or {}).items()}      # banded head weights (otr_m.py, HSPLIT)
            E.M.HBANDS = bool(request.get('hbands'))  # band biases of the head's first layer (otr_m.py, HBANDS)
            E.M.B7 = bool(request.get('b7'))          # the physicist's batch-7 head columns (O/physicist/b7cols.py)
            E.M.FEATS = bool(request.get('featstats')); E.M.FEATSTATS.clear()      # head-feature statistics (otr_m.py, FEATS)
            E.M.CV = bool(request.get('cv'))          # the low-rank covariance corrections computed at 0 for their gradients (otr_m.py, cvdd ...)
            E.LC.SCHED = request.get('sched') or '384-320-384'   # the old tier's ranks at the joins of layers 7 / 10 / 13 (physicist T80; the card's 384-320-384)
            if E.M.RG or E.M.RG_W:
                E.M.readout_weights(request.get('rgw'))
            if E.M.HEADF or E.M.HEADU is not None:
                E.M.head_weights(request.get('hu'))   # their first-layer weights: the request's 6 x 128 values or zero
        E.LOSS_W = request.get('lossw')                   # all-layer training loss of grad runs (otr_engine.py, LOSS_W)
        record = E.run_network(request['rank'], request['mode'], request.get('scale'))
    except Exception:
        publish(failed, lambda stream: stream.write(traceback.format_exc()))
        say(event='error', id=request['id'], error=traceback.format_exc().splitlines()[-1])
        E.collect()
        return
    if E.M is not None and E.M.HEADF:
        record['head_rows'] = [E.M.ROWS.get(l, 0) for l in range(16)]      # response rows of the bank read per layer
    if E.M is not None and E.M.FEATS:
        record['featstats'] = {str(l): v for l, v in E.M.FEATSTATS.items()}
    record['cutsim'] = request.get('cutsim')
    record['data_source'] = 'generated' if request['rank'] >= 7_000_000 else 'public'
    record.update(id=request['id'], params=request['params'], worker=TAG, threads=E.THREADS, f4c=bool(E.M is not None and E.M.F4C),
                  ncar=([dict(zip(('module', 'arguments'), E.M.carrier_arguments(x))) for x in (E.M.NCAR if isinstance(E.M.NCAR, list) else [E.M.NCAR])]
                        if E.M is not None and E.M.NCAR is not None else None),
                  peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if request['mode'] == 'grad':
        publish(out, lambda stream: torch.save(record, stream), 'wb')
    else:
        publish(out, lambda stream: json.dump(record, stream))
    say(event='done', id=request['id'], mse=record['mse'], seconds=round(record['seconds'], 2),
        total_seconds=round(record['total_seconds'], 2), rss_gb=round(process.memory_info().rss / 1e9, 2),
        peak_rss_gb=round(record['peak_rss'] / 1e9, 2))


say(event='start', run=str(RUN), threads=E.THREADS, torch=torch.__version__, card=E.CARD, per_layer=E.PL is not None)
for orphan in sorted((RUN / 'claimed').glob('*.%s.json' % TAG)):
    serve(orphan)
while not (RUN / 'STOP').exists():
    claim = None
    for request in sorted((RUN / 'queue').glob('*.json')):
        # Generated sizing/check jobs may queue before the analyst finishes their
        # truths. Keep them unclaimed, allowing ready public gradients to run.
        queued = json.loads(request.read_text())
        seed = int(queued['rank'])
        if seed >= 7_000_000 and seed in allowed_seeds:
            from otr_common import O
            if not (O / 'analyst/fresh' / ('truth_seed%d.npz' % seed)).exists():
                continue
        target = RUN / 'claimed' / ('%s.%s.json' % (request.stem, TAG))
        try:
            os.rename(request, target)
        except FileNotFoundError:
            continue
        claim = target
        break
    if claim is None:
        time.sleep(1.0)
        continue
    serve(claim)
    if process.memory_info().rss > RSS_LIMIT:
        say(event='recycle', rss=process.memory_info().rss)
        sys.exit(3)
say(event='stop')
