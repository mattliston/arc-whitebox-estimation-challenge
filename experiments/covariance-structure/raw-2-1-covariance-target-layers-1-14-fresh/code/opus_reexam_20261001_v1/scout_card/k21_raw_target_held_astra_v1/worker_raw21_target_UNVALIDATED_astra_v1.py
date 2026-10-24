# Version16 (trainer, written 2026-10-13T10:40:52Z; lead 10:3xZ): identical to v15, but the card module is otr_m_v7.py (+ K4L ages 3-4, families k4la3 / k4la4,
# default OFF when absent; certified bit for bit vs v15 without them before any use).
# Version15 (trainer, written 2026-10-12T23:46:25Z; lead 23:3xZ): identical to v14, but otr_pl is otr_pl_v2.py (+ the family k21k, the K21 scale on the
# kink rows), aliased as otr_pl before the card module and the engine import it. All new families 0 = v13 bit for bit (certified before use).
# Version14 (trainer, written 2026-10-12T23:30:31Z; lead 23:2xZ): identical to v13, but the card module is otr_m_v6.py (+ the per-layer families pe3 / pe4 on the
# pair map's local skew / kurtosis terms; both 0 = v13 bit for bit, certified on one request before any use).
# Version13 (trainer, written 2026-10-12T15:57:54Z; lead 15:2xZ): = worker_generated_v12.py with the generated networks' weights regenerated in memory
# from their seed (otr_generated_v3.make_data_paths_mem: the analyst's generator, checked against the truth file's weights_sha256 on first
# use) instead of the run's generated_cache_v1 on disk: identical weights and numerics, no disk growth. Truth-plan branch unchanged.
# Version12 (trainer, 10-12 03:25Z): identical to v11, but a queue file that another worker claimed between the glob and the read is skipped (v5-v11
# crashed with FileNotFoundError there: 3 of 4 S50 workers died at 4 workers).
# Version11 (trainer, 10-11 21:08Z): identical to v10, but the card module is otr_m_v5.py (the early edge's scale also from the per-layer family 'edx').
# Version10 (trainer, 10-11 18:50Z): identical to v9, but the card module is otr_m_v4.py (adds the STAR STRENGTH family 'ssb', request field 'ssb': 1 with 'ss' -1).
# Version9 (trainer, 10-11 17:40Z): identical to v8, but the card module is otr_m_v3.py (adds the READ COLUMN CUT twin, request field 'rdc'), and the
# engineer's star drop-in is star_ss_torch_v2.py (= v1 plus 'ss' -1, the native one-direction star; v1 would read -1 as all but one column).
# Version8 (trainer, 10-11 15:52Z): identical to v7, but the card module is the versioned otr_m_v2.py (imported under the name otr_m before the engine),
# so the live otr_m.py stays at bba0dea0 for the pinned tools of every lane (lead 15:4xZ).
# Version7 (trainer, 10-11 15:40Z): identical to v6 plus the request fields 'edgex' / 'exs' (otr_m.py, EARLY EDGE) and their gradient 'ex_grads'.
# Version6 (trainer, 10-11 11:30Z): identical to v5 plus the request fields 'edgein' / 'eu' (otr_m.py, EDGE INPUTS) and their gradient 'eu_grads'.
# Version5 (trainer, 10-11 10:08Z): identical to v4 except the no-plan loader otr_generated_v2 (minimum truth N from env).
# Version4: exact immutable truth-plan routing; legacy v3 arithmetic unchanged.
# Source parent SHA256: 7debd2cd0555e24a6d4fa9123093ec7e6783188725f4ce441415a4fc4c5cddf0
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

# HELD private copy; original worker/engine/module files remain immutable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'trainer'))
RUN, INDEX = Path(sys.argv[1]).resolve(), int(sys.argv[2])
TAG = 'w%d' % INDEX
RSS_LIMIT = int(float(os.environ.get('TRAINER_RSS_LIMIT', 7.5e9)))   # raise it when TRAINER_RELIEF=0 keeps freed memory resident

import psutil
from otr_common import O as _O      # v9: the engineer's star drop-in v2 under v1's name (the engine imports star_ss_torch lazily, path at the END)
sys.path.append(str(_O / 'engineer' / 'a33_starsum'))
import star_ss_torch_v2
sys.modules['star_ss_torch'] = star_ss_torch_v2
import otr_pl_v2
sys.modules['otr_pl'] = otr_pl_v2      # v15: + k21k; aliased BEFORE anything imports otr_pl (otr_m_v6, the engine)
import otr_m_v7
sys.modules['otr_m'] = otr_m_v7      # v11: + the early edge's per-layer family 'edx'
import otr_engine as E
assert E.M is otr_m_v7
from torch_raw21_target_UNVALIDATED_astra_v3 import enable as _astra_raw21_enable
from otr_common import error_path, load_params, load_pl, publish, result_path

torch = E.torch
process = psutil.Process()
loaded, loaded_pl = None, {}
# Astra versioned offline adapter: the live worker.py and engine remain unchanged.
from otr_generated_v2 import make_data_paths      # v5: configurable minimum truth N (TRAINER_GENERATED_MIN_N)
from otr_generated_v3 import make_data_paths_mem      # v13: generated weights regenerated in memory, no disk cache
from otr_cut_runtime_v2 import enable as enable_cutsim
allowed_file = os.environ.get('TRAINER_GENERATED_ALLOW_FILE')
allowed_seeds = ([int(x) for x in Path(allowed_file).read_text().strip().split(',')] if allowed_file else [])
truth_plan_path = os.environ.get('TRAINER_TRUTH_PLAN')
truth_plan_sha = os.environ.get('TRAINER_TRUTH_PLAN_SHA256')
truth_plan, truth_metadata = None, {}
assert bool(truth_plan_path) == bool(truth_plan_sha), 'Truth plan path and SHA must be supplied together'
if truth_plan_path:
    from otr_generated_manifest_v1 import TruthPlan, make_data_paths as manifest_data_paths
    truth_plan = TruthPlan(truth_plan_path, truth_plan_sha, allowed_seeds)
    E.data_paths, truth_metadata = manifest_data_paths(E.data_paths, truth_plan, RUN / 'generated_cache_v1')
else:
    E.data_paths = make_data_paths_mem(E.data_paths, allowed_seeds)      # v13 (lead 15:2xZ: the weight caches grew ~12 GB a round)
configure_cutsim = enable_cutsim(E)
configure_raw21_target = _astra_raw21_enable(E)
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
        assert request.get('truth_plan') == (truth_plan.spec() if truth_plan else None), 'Request/worker truth plan mismatch'
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
            E.M.EDGEIN = bool(request.get('edgein'))  # new zero-weight inputs of the covariance edge (otr_m.py, EDGE INPUTS)
            if E.M.EDGEIN or E.M.EDGEU is not None:
                E.M.edge_weights(request.get('eu'))
            E.M.RDC = request.get('rdc')              # the read column cut (otr_m_v3.py, READ COLUMN CUT): None = off
            E.M.SSB = bool(request.get('ssb'))        # the star strength (otr_m_v4.py, STAR STRENGTH); needs the one-direction star
            assert not E.M.SSB or request.get('ss') == -1, "'ssb' needs 'ss' -1"
            E.M.EDGEX = request.get('edgex')          # the learned edge also at layers lo..hi outside 8-14 (otr_m.py, EARLY EDGE)
            if E.M.EDGEX or E.M.EDGEXS is not None:
                E.M.edgex_scales(request.get('exs'))
        E.LOSS_W = request.get('lossw')                   # all-layer training loss of grad runs (otr_engine.py, LOSS_W)
        configure_raw21_target(request)
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
    if E.M is not None and E.M.EDGEIN and request['mode'] == 'grad':
        record['eu_grads'] = E.M.EDGEU.grad.clone() if E.M.EDGEU.grad is not None else torch.zeros_like(E.M.EDGEU)
    if E.M is not None and E.M.EDGEX and request['mode'] == 'grad':
        record['ex_grads'] = E.M.EDGEXS.grad.clone() if E.M.EDGEXS.grad is not None else torch.zeros_like(E.M.EDGEXS)
    record['truth_plan'] = truth_plan.spec() if truth_plan else None
    record['truth_metadata'] = truth_metadata.get(int(request['rank']))
    record['cutsim'] = request.get('cutsim')
    import hashlib
    record['astra_raw21_target'] = request.get('astra_raw21_target', 0)
    record['astra_raw21_route'] = configure_raw21_target.route_report()
    record['astra_raw21_provenance'] = dict(
        request_sha256=hashlib.sha256(claim.read_bytes()).hexdigest(),
        worker_path=str(Path(__file__).resolve()),
        worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256((Path(__file__).parent/'torch_raw21_target_UNVALIDATED_astra_v3.py').read_bytes()).hexdigest(),
        executable=sys.executable,
        thread_environment={key:os.environ.get(key) for key in ('TRAINER_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')})

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
        try:
            queued = json.loads(request.read_text())
        except FileNotFoundError:      # v12: claimed by another worker after the glob
            continue
        seed = int(queued['rank'])
        if seed >= 7_000_000 and seed in allowed_seeds:
            from otr_common import O
            ready = truth_plan.ready(seed) if truth_plan else (O / 'analyst/fresh' / ('truth_seed%d.npz' % seed)).exists()
            if not ready:
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
