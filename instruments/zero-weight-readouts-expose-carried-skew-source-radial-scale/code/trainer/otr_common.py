"""TAG opus-trainer (2026-10-01): torch-free shared pieces of the parallel long-horizon refit trainer in O/trainer/.
Paths, the train / VAL-A / VAL-B split, per-network data paths, reading the 28 learned arrays + 12 dials back out of a native candidate
source (the research parent RRPI by default), parameter files and the file queue that master/eval drivers and workers talk through.
Constants and the parameter decoding are copied (not imported) from H/hill_climb_rrpi_detached_star_v1/{train_segment.py,
native_parameters.py}; the sealed protocol ceremony is not reused. Module names carry the otr_ prefix because the card imports a module
called `common` from fable_refit_v1. Always run python with -B: importing the card modules must not write bytecode outside O/trainer/."""
from pathlib import Path
import ast
import base64
import hashlib
import json
import os
import time
import zlib

HERE = Path(__file__).resolve().parent
O = HERE.parent
P2 = O.parent
H = P2 / 'codex_resume_20260926_v2'
SOURCE = H / 'hill_climb_pair_rr_inverse_v1/candidate.py'
SOURCE_SHA = 'a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
TRUTH = P2 / 'fable_refit_v1/truth688'
SPLIT = P2 / 'fable_analyst_v1/split_train640.json'
TRUNK = ('msg_w', 'msg_b', 'enc0_w', 'enc0_b', 'enc2_w', 'enc2_b', 'enc4_w', 'enc4_b')
DIAL_LR = dict(X1=.002, X0=.0032, X2=.0032, bscale=.0008,
               s_carry=.0002, s_new=.0008, c0=.002, k4a=.004, k4b=.004, k4c=.004)
DIAL_KEYS = ('X1', 'X0', 'X2', 'X3', 'bscale', 's_carry', 's_new', 'c0', 'c1', 'k4a', 'k4b', 'k4c')
FROZEN_P = ('centers', 'scales', 'rms', 'gain', 'first')
FROZEN_DIALS = ('X3', 'c1')
LITERALS = ('_GNN_LITERAL', '_BDNH_A_LITERAL', '_BD_COEFFICIENT_LITERAL',
            '_EDGE_LITERAL', '_MESSAGE_EDGE_LITERAL')
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_nodes(text):
    """name -> list of top-level single-target assignments, in file order."""
    nodes = {}
    for node in ast.parse(text).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            nodes.setdefault(node.targets[0].id, []).append(node)
    return nodes


def read_source(np, path=SOURCE):
    """(28 float32 arrays keyed p.* / a / c / e.*, 12 float dials) decoded from the candidate source itself."""
    nodes = source_nodes(Path(path).read_text())
    literals = {}
    for name in LITERALS:
        assert len(nodes[name]) == 1
        literals[name] = ast.literal_eval(nodes[name][0].value)
    arrays = {}
    for key, (shape, encoded) in literals['_GNN_LITERAL'].items():
        raw = zlib.decompress(base64.b64decode(encoded, validate=True))
        arrays['p.' + key] = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    arrays['a'] = np.asarray(literals['_BDNH_A_LITERAL'], dtype=np.float32)
    arrays['c'] = np.asarray(literals['_BD_COEFFICIENT_LITERAL'], dtype=np.float32)
    for key, value in literals['_EDGE_LITERAL'].items():
        arrays['e.' + key] = np.asarray(value, dtype=np.float32)
    for key, value in literals['_MESSAGE_EDGE_LITERAL'].items():
        arrays['e.context_' + key] = np.asarray(value, dtype=np.float32)
    assert len(arrays) == 28
    assert all(value.dtype == np.float32 and np.isfinite(value).all() for value in arrays.values())
    last = lambda name: ast.literal_eval(nodes[name][-1].value)
    x0, zero, x2, x3 = last('_S19_BD_ADD')
    assert zero == 0.0
    s_carry, s_new = last('_S19_LAM')
    k4a, k4b, k4c = last('_S19_K4BAND')
    table = nodes['_K31R_TABLE'][-1].value
    assert isinstance(table, ast.DictComp)
    c0, c1 = ast.literal_eval(table.value)
    dials = dict(X1=last('_BD_J1_PHYS'), X0=x0, X2=x2, X3=x3, bscale=last('_S19_BSCALE'), s_carry=s_carry,
                 s_new=s_new, c0=c0, c1=c1, k4a=k4a, k4b=k4b, k4c=k4c)
    assert tuple(dials) == DIAL_KEYS and all(type(value) is float for value in dials.values())
    return arrays, dials


def splits():
    """train = 641 ranks; VAL-A / VAL-B = even / odd positions of the 40 validation ranks (A selects checkpoints, B is report-only)."""
    saved = json.loads(SPLIT.read_text())
    train, valid = saved['train'], saved['valid40']
    assert len(train) == len(set(train)) == 641 and len(valid) == len(set(valid)) == 40
    assert not set(train) & set(valid)
    assert not (set(train) | set(valid)) & (set(saved['panel8']) - {150})   # panel net 150 sits in the validation ranks
    return dict(train=train, val_a=valid[0::2], val_b=valid[1::2])


_ROWS = None


def data_paths(rank):
    global _ROWS
    if _ROWS is None:
        _ROWS = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
    row = _ROWS[rank]
    assert row['role_rank'] == rank
    return STORE / Path(row['path']).parent / 'weights.npy', TRUTH / ('truthpost_rank%04d.npz' % rank)


def publish(path, writer, mode='w'):
    """Create `path` atomically (write a private temp file, then rename); never replaces an existing file."""
    path = Path(path)
    assert not path.exists(), path
    temp = path.with_name('.%s.tmp%d' % (path.name, os.getpid()))
    with temp.open('x' + mode[1:]) as stream:
        writer(stream)
    os.rename(temp, path)


def save_params(torch, path, arrays, dials, pl=None):
    """arrays: 28 float32 tensors; dials: 12 floats; pl (optional): shifts of the per-layer physics dials of otr_pl.py,
    {family: 16 floats}, which the workers apply to every request that uses this file."""
    assert len(arrays) == 28 and tuple(dials) == DIAL_KEYS
    payload = dict(arrays={key: value.detach().clone() for key, value in arrays.items()},
                   dials={key: float(value) for key, value in dials.items()})
    if pl:
        payload['pl'] = {family: [float(x) for x in values] for family, values in pl.items()}
    assert all(value.dtype == torch.float32 for value in payload['arrays'].values())
    publish(path, lambda stream: torch.save(payload, stream), 'wb')


def load_params(torch, path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    assert len(saved['arrays']) == 28 and tuple(saved['dials']) == DIAL_KEYS
    return saved['arrays'], saved['dials']


def load_pl(torch, path):
    """The per-layer dial shifts stored with a parameter file ({} for the files written before they existed)."""
    return torch.load(path, map_location='cpu', weights_only=True).get('pl', {})


def run_dirs(run):
    run = Path(run)
    for name in ('queue', 'claimed', 'results', 'params', 'ckpt', 'workers'):
        (run / name).mkdir(parents=True, exist_ok=True)
    return run


def enqueue(run, request):
    """request: dict(id, mode='grad'|'eval', rank, params, scale). Skipped when the result or a pending copy already exists."""
    run, name = Path(run), request['id']
    if result_path(run, request).exists() or (run / 'queue' / (name + '.json')).exists():
        return False
    if any((run / 'claimed').glob(name + '.w*.json')):
        return False
    publish(run / 'queue' / (name + '.json'), lambda stream: json.dump(request, stream))
    return True


def result_path(run, request):
    return Path(run) / 'results' / (request['id'] + ('.pt' if request['mode'] == 'grad' else '.json'))


def error_path(run, request):
    return Path(run) / 'results' / (request['id'] + '.error')


def wait_results(run, requests, note=None, poll=0.2, heartbeat=300):
    """Block until every request has a result file; `note(text)` is called every `heartbeat` seconds while waiting.
    A worker-side failure (<id>.error) or a STOP file in the run directory ends the caller."""
    pending = list(requests)
    started = last = time.monotonic()
    while True:
        pending = [request for request in pending if not result_path(run, request).exists()]
        if not pending:
            return
        for request in pending:
            if error_path(run, request).exists():
                raise SystemExit('request %s failed in a worker:\n%s' % (request['id'], error_path(run, request).read_text()))
        if (Path(run) / 'STOP').exists():
            raise SystemExit('STOP file present in %s: leaving (checkpoints and results stay on disk)' % run)
        now = time.monotonic()
        if note is not None and now - last >= heartbeat:
            note('waiting %.0f s for %d results, first missing %s' % (now - started, len(pending), pending[0]['id']))
            last = now
        time.sleep(poll)
