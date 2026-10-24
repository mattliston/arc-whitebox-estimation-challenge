"""TAG opus-trainer (2026-10-01): the torch card of the current estimator for ONE network, forward (+ backward).
Importing this module loads torch and the card once (heavy). The card-loading, j3cut exec patch, install_star (rank-32 star, basis
under no_grad), collect and the per-network forward/backward are copied from the reference trainer
H/hill_climb_rrpi_detached_star_v1/train_segment.py (lines 54-168) without its protocol ceremony; certify.py proves the copy
reproduces the reference losses and gradients. Parameters are the 28 arrays + 12 dials of the research parent RRPI until
set_parameters() is called. Trainable: 23 arrays (full + trunk) and the 10 dials of DIAL_LR; frozen: p.centers/scales/rms/gain/first,
dials X3 and c1. Threads: env TRAINER_THREADS (default 1); the four BLAS thread variables must equal it.
With env TRAINER_PL=1 the per-layer dial leaves of otr_pl.py are installed on every card (values unchanged) and a gradient result
also carries 'pl_grads' (per family, 16 layers). cert/otr_engine_as_certified.py is this file as it ran pilot A (the certified engine
plus the TRAINER_RELIEF switch, whose default leaves the behaviour unchanged).
With env TRAINER_CARD=M (needs TRAINER_PL=1) every card also gets F4 + K4L with trainable strengths (otr_m.py): the run M card.
STAR_ATTACH (env TRAINER_STAR_ATTACH=1, or set per request by the worker): the rank-32 star basis is a differentiable function of
the star in the backward pass instead of a constant; the forward values are the same. cert/otr_engine_as_run_L.py is this file
as run L used it."""
import ast
import ctypes
import gc
import importlib.util
import inspect
import os
import sys
import time

from otr_common import DIAL_LR, FROZEN_DIALS, FROZEN_P, P2, THREAD_ENV, TRUNK, data_paths, read_source

THREADS = int(os.environ.get('TRAINER_THREADS', '1'))
assert all(os.environ.get(k) == str(THREADS) for k in THREAD_ENV)
os.environ['JOIN_SCHED'] = '384-320-384'
for rel in ('fable_analyst_v1', 'fable_covhunter_v1', 'fable_engineer_v1',
            'fable_physicist_v1/refit6b', 'fable_porter_v1'):
    sys.path.insert(0, str(P2 / rel))
import numpy as np
import torch

torch.set_num_threads(THREADS)
torch.set_num_interop_threads(1)
torch.manual_seed(0)
arrays, initial_dials = read_source(np)
T = {key: torch.from_numpy(np.array(value, copy=True)) for key, value in arrays.items()}
p = {key[2:]: value for key, value in T.items() if key.startswith('p.')}
e = {key[2:]: value for key, value in T.items() if key.startswith('e.')}
full = [p['head'], p['head_b'], e['output'], e['bias'], T['a'], T['c']]
full += [e[key] for key in e if key not in ('output', 'bias')]
trunk = [p[key] for key in TRUNK]
assert len(e) == 11 and len(full + trunk) == 23 and len(T) == 28
assert sum(x.numel() for x in full) == 3437 and sum(x.numel() for x in trunk) == 44144
for value in full + trunk:
    value.requires_grad_(True)
assert all(not p[key].requires_grad for key in FROZEN_P)
TRAINABLE = [key for key, value in T.items() if value.requires_grad]
DL = {key: torch.tensor(value, dtype=torch.float64, requires_grad=key in DIAL_LR)
      for key, value in initial_dials.items()}

card_path = P2 / 'fable_physicist_v1/refit6b/s6bcard_loop5js.py'
spec = importlib.util.spec_from_file_location('s33_two_update_card', card_path)
LC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LC)
torch.set_num_threads(THREADS)
torch.set_grad_enabled(True)
assert torch.is_grad_enabled() and torch.get_num_threads() == THREADS
assert LC.SCHED == '384-320-384'
bd_source = inspect.getsource(LC.JT.make_bd)
assert bd_source.count('for j in range(4):') == 1
exec(compile(ast.parse(bd_source.replace('for j in range(4):', 'for j in range(3):')),
             str(__file__) + ':j3cut', 'exec'), LC.JT.__dict__)
relief = ctypes.CDLL('/usr/lib/libSystem.B.dylib').malloc_zone_pressure_relief
relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
relief.restype = ctypes.c_size_t
# The card directories sit in front of O/trainer on sys.path; nothing of the card may resolve to a file of this lane.
HERE = os.path.dirname(os.path.abspath(__file__))
assert not [name for name, module in sys.modules.items() if not name.startswith(('otr_', '__m'))
            and os.path.dirname(getattr(module, '__file__', None) or '') == HERE]


RELIEF = os.environ.get('TRAINER_RELIEF', '1') == '1'   # 0 = keep freed memory in the process (bench_grad.py measures the trade)


def collect():
    gc.collect()
    if RELIEF:
        relief(None, 0)


STAR_ATTACH = os.environ.get('TRAINER_STAR_ATTACH', '0') == '1'
KEEP_FINAL = False   # set per request by the worker: a forward result then also carries the final layer's 1024 predictions
KEEP_ALL = False     # set per request by the worker: a forward result then carries the predictions of all 16 layers (parity targets)
STAR_SS = None       # set per request by the worker: an int = the engineer's star by row sums with that many factor columns
                     # (O/engineer/a33_starsum/star_ss_torch.py; 0 = the S39 file's star) in place of the card's rank-32 star
_SS = []


def _star_ss():
    """The engineer's drop-in, loaded on first use (its folder goes to the END of sys.path: nothing of the card may resolve to it)."""
    if not _SS:
        from otr_common import O
        sys.path.append(str(O / 'engineer' / 'a33_starsum'))
        import star_ss_torch
        _SS.append(star_ss_torch)
    return _SS[0]


def install_star(cd):
    # Same forward operations; only the basis gradient follows Opus's detached rule (unless STAR_ATTACH).
    def factor(A):
        if STAR_ATTACH and torch.is_grad_enabled():
            cn = (A.detach().to(torch.float64) ** 2).sum(0)
            cols = torch.sort(torch.argsort(-cn, stable=True)[:32]).values
            Q = torch.linalg.qr(A[:, cols])[0]
            Q = torch.linalg.qr(A @ torch.linalg.qr(A.T @ Q)[0])[0]
            return Q, Q.T @ A
        with torch.no_grad():
            Ad = A.detach()
            cn = (Ad.to(torch.float64) ** 2).sum(0)
            cols = torch.sort(torch.argsort(-cn, stable=True)[:32]).values
            Q = torch.linalg.qr(Ad[:, cols])[0]
            Q = torch.linalg.qr(Ad @ torch.linalg.qr(Ad.T @ Q)[0])[0]
        return Q, Q.T @ A
    def star(w, A):
        wt = w.T
        Q, B = factor(A.to(torch.float32))
        P = wt @ Q
        V = (wt * wt) @ B.T
        result = ((V @ P.T) / 3).to(torch.float64)
        return torch.diagonal_scatter(result, (P * V).sum(1, dtype=torch.float64))
    def diagonal(w, A):
        wt = w.T
        Q, B = factor(A.to(torch.float32))
        pushed = (wt @ Q) @ B
        return ((pushed * wt) * wt).sum(1, dtype=torch.float64)
    cd.env['_flms_star'] = star
    cd.env['_flms_star_diagonal'] = diagonal


PL = None   # per-layer physics dials (otr_pl.py): extra zero leaves whose gradients come out of the same backward pass
UNFREEZE = False   # 10-10 20:50Z (lead 20:26Z, staleness screen): request field 'unfreeze' = the arrays frozen since September (FROZEN_P: centers,
                   # scales, rms, gain, first) get gradients in 'grad' runs and may differ from the source in set_parameters
LOSS_W = None   # 10-10 19:30Z: {layer: weight} = the backward pass of 'grad' runs uses sum_l weight_l x MSE_l (all-layer truths) instead of the
                # final-layer MSE; the record's 'mse' stays the final-layer MSE (request field 'lossw')
if os.environ.get('TRAINER_PL') == '1':
    import otr_pl as PL
CARD = os.environ.get('TRAINER_CARD', 'L')   # L = today's card; M = today's card + F4 + K4L with trainable strengths (otr_m.py)
assert CARD in ('L', 'M')
M = None
if CARD == 'M':
    assert PL is not None, 'TRAINER_CARD=M needs TRAINER_PL=1'
    import otr_m as M


def set_parameters(new_arrays, new_dials):
    """Copy a parameter set into the leaves the card reads; the frozen arrays and dials must be unchanged."""
    assert set(new_arrays) == set(T) and set(new_dials) == set(DL)
    for key in FROZEN_P:
        assert UNFREEZE or new_arrays['p.' + key].numpy().tobytes() == arrays['p.' + key].tobytes()
    for key in FROZEN_DIALS:
        assert float(new_dials[key]) == initial_dials[key]
    with torch.no_grad():
        for key in T:
            assert new_arrays[key].dtype == torch.float32 and new_arrays[key].shape == T[key].shape
            T[key].copy_(new_arrays[key])
        for key in DL:
            DL[key].copy_(torch.tensor(float(new_dials[key]), dtype=torch.float64))
    assert all(torch.isfinite(value).all() for value in list(T.values()) + list(DL.values()))


def run_network(rank, mode, scale=None):
    """mode 'grad': forward with layer checkpoints + backward of scale * (final-layer MSE); gradients of the 23 arrays and 10 dials.
    mode 'eval': forward only under no_grad. Both return the MSE of all 16 layers against the 1e9-sample truth."""
    assert mode in ('grad', 'eval')
    begun = time.monotonic()
    collect()
    weights_path, truth_path = data_paths(rank)
    weights = np.load(weights_path, allow_pickle=False)
    assert weights.shape == (16, 1024, 1024) and np.isfinite(weights).all()
    with np.load(truth_path, allow_pickle=False) as saved:
        truth = torch.from_numpy(np.asarray(saved['mc_post'], dtype=np.float64).copy())
    assert truth.shape == (16, 1024) and torch.isfinite(truth).all()
    target = truth[15]
    for key in FROZEN_P:
        p[key].requires_grad_(UNFREEZE)
        p[key].grad = None
    for value in full + trunk + list(DL.values()):
        value.grad = None
    LC.JT.DIALS_T = DL
    with torch.set_grad_enabled(mode == 'grad'):
        cd = LC.card(torch, weights, p, T['a'], T['c'], e, fine_ckpt=False)
        cd.hooks['k4_15'] = lambda layer, value, ctx: value
        install_star(cd)
        if STAR_SS is not None:
            _star_ss().install(cd, torch, STAR_SS)
        if PL is not None:
            PL.install(sys.modules[__name__], cd)
        if M is not None:
            M.install(cd)
        original_step = cd.step
        def step(layer, state):
            try:
                return original_step(layer, state)
            finally:
                collect()  # Includes checkpoint recomputation/early stop.
        cd.step = step
        started = time.monotonic()
        prediction = cd.run(mode == 'grad')
        assert prediction.shape == (16, 1024) and torch.isfinite(prediction).all()
        loss = ((prediction[15].to(torch.float64) - target) ** 2).mean()
        assert torch.isfinite(loss)
        record = dict(rank=rank, mode=mode, card='L' if M is None else 'M', star_attach=STAR_ATTACH, star_ss=STAR_SS,
                      mse=float(loss.detach()),
                      mse_layers=((prediction.detach().to(torch.float64) - truth) ** 2).mean(1).tolist())
        if KEEP_FINAL and mode == 'eval':
            record['final_prediction'] = prediction[15].detach().to(torch.float64).tolist()
        if KEEP_ALL and mode == 'eval':
            record['prediction'] = prediction.detach().to(torch.float64).tolist()
        if mode == 'grad':
            train = loss if not LOSS_W else sum(float(w) * ((prediction[int(l)].to(torch.float64) - truth[int(l)]) ** 2).mean() for l, w in LOSS_W.items())
            (train * scale).backward()
            if M is not None and M.HEADF:      # gradient of the new head-input columns' first-layer weights (6 x 128)
                record['hu_grads'] = M.HEADU.grad.clone()
            if M is not None and M.RG:         # gradients of the three zero-weight readouts (row gain of the bank read, carried b, radial)
                record['rg_grads'] = {name: (leaf.grad.clone() if leaf.grad is not None else torch.zeros_like(leaf)) for name, leaf in M.RG_W.items()}
        record['seconds'] = time.monotonic() - started
        cd.step = original_step
    del prediction, loss, cd, original_step, step, weights, truth, target
    collect()
    if mode == 'grad':
        grads = {key: T[key].grad.clone() for key in TRAINABLE}
        if UNFREEZE:
            grads.update({'p.' + key: (p[key].grad.clone() if p[key].grad is not None else torch.zeros_like(p[key])) for key in FROZEN_P})
        dial_grads = {key: DL[key].grad.clone() for key in DIAL_LR}
        assert all(torch.isfinite(value).all() for value in list(grads.values()) + list(dial_grads.values()))
        record.update(scale=scale, grads=grads, dial_grads=dial_grads)
        if PL is not None:
            record['pl_grads'] = PL.collect()
    record['total_seconds'] = time.monotonic() - begun
    return record
