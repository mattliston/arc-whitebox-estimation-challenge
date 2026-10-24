"""Inert fixed pair-metric qualifier. No estimator, fixture file or target access.

Local mode owns two tiny FNP contexts. The separately hashed client delegates
remote mode; server mode only attests then calls the unchanged official server.
All imports at module scope are stdlib. No call/stage exists without --run.
"""
import argparse
import ast
import base64
import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import struct
import sys
import time
import weakref

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
SHARED = NEW / 'current_comp_shared_old384_v1'
OPENBLAS = NEW / 'openblas_runtime_env_v1'
CLIENT = P2 / 'data_gate/.client-venv'
OSITE = OPENBLAS / 'lib/python3.11/site-packages'
CSITE = CLIENT / 'lib/python3.11/site-packages'
PARENT = NEW / 'current_comp_shared_gain_slice_v1/build_v1/estimator.py'
PARENT_SHA = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
METRIC_SHA = '52731186b0ee2ecd932fb9b3a7a44bd23a305dd17d3afbf58f8949019df5c44e'
SCHEMA = 'current_comp_shared_pair256_tiny_v1'
CASES = ('dyadic', 'rounded', 'signed_zero')
REJECTIONS = ('opposition', 'dtype', 'odd', 'nonfinite', 'underflow', 'PSD', 'zero_range')
THREADS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
CALL_BUDGET = 10_000_000
PINS = {
    PARENT: PARENT_SHA,
    HERE / 'metric_fnp.py': METRIC_SHA,
    HERE / 'compose.py': '1d73f6b9c951917a5af0bbc57ec2cd020e01cff471cddbfd37720f73995b1b3e',
    HERE / 'DESIGN.md': '5290ec27913961c438768730320f1641bf0d7fcd530b70fbce4e5dc6d30a6653',
    SHARED / 'fnp.py': '980853fe0b794efcf046b360ea03bbfaa671bd48383312e8e766c9b546021d81',
    SHARED / 'scalar_semantics_probe_v3.py': 'e6336483af82f75364efe64a95eaa0855a676c19ceaf878e5d74d9908580afca',
    SHARED / 'SCALAR_SEMANTICS_PROTOCOL_V3.md': '9ff60bc1ab796df0e9911119a0f0bc24a1ee063226a21f6c6bea0e5dec489f6f',
    OSITE / 'numpy/__init__.py': '92a46f791e453926d3292af2582b89995a289475f0eaaea71a949823200b838a',
    OSITE / 'numpy/__config__.py': '97bedd19442e4714a9147f49cd93357032d21d791d87ca1367f44766db0771e6',
    OSITE / 'numpy/_core/_multiarray_umath.cpython-311-darwin.so': 'b9a53273d646f3ffbf8307725caa306bd81c7ade0815ae999af98e0cf3e8ebec',
    OSITE / 'numpy/linalg/_linalg.py': 'e47d0e2aed21361291cd6651999a59e6479ba5866abb8eabb1dbea5b0d1618ac',
    OSITE / 'numpy/linalg/_umath_linalg.cpython-311-darwin.so': '2dfac87008f01faaaea501c8e316f59f8dc083366f6f8648072cb842d5da7f4b',
    OSITE / 'numpy/.dylibs/libscipy_openblas64_.dylib': 'bd3ad3014a7977f69921ceb2f5d7b0a8fb4b96dfb0653f87bd3bfba5af806d78',
    OSITE / 'threadpoolctl.py': '12fb9526b6a74d2e686b7ec148dc165c3587999b14fa86984aa180e10802400b',
    OSITE / 'flopscope/__init__.py': 'cd3d8b325339717390cee951c5cbeee58c9521aa0e8d642ae62f63c39a2ccdcb',
    OSITE / 'flopscope/numpy/__init__.py': '33edacfc6ca01866d4f558e3a43f8fadfd6a2c0efdbc4e4712acc1d839ad4e2f',
    OSITE / 'flopscope/_pointwise.py': '7d376515446f1b174bef375c48116a552098bd514f5552450876f56822beb0c7',
    OSITE / 'flopscope/_ndarray.py': 'fa5bbecb48b123ab77f31ced3da877d2e333aa5b403a324c33da591b967d3e74',
    OSITE / 'flopscope/_array_ops.py': 'e45e927f18c54352503a4951d3d925a5f31f92ecea245e32360825d71a05da6c',
    OSITE / 'flopscope/_dtype_billing.py': 'd6f4df8b2fcf162f41afde2dae23a66d894eba62f173e57d89051e5873569cb8',
    OSITE / 'flopscope/_budget.py': 'ad9e78172bd0e8175533d4aae75b67df749d96ecc1ca868135eb3d781e542d01',
    OSITE / 'flopscope/data/default_weights.json': 'f729bc47f07c080b8c7b9d7b045e5663f0af313669b4c2a962b5a9044a94cb7a',
    OSITE / 'flopscope/numpy/linalg/_decompositions.py': '0cd6f08ac3fe3cef97d506be6713d0aa85f6ced74e8fd35016a420bfe6d8d942',
    OSITE / 'flopscope_server/__main__.py': '3ec966a9749af2c8fb912f04f2c5447e562ea5b64b82eacf30169052e290bff4',
    OSITE / 'flopscope_server/_request_handler.py': 'c6157a3512193573c425c93514c44cdb9438f91b7d52ced86e2ed2cff8348fde',
    OSITE / 'flopscope_server/_session.py': '76c22d0b368b03c98a33e97c5e33af42acaf509e7ec100b3dbe692e5775d9ee1',
    CSITE / 'flopscope/__init__.py': '86179621dc63a6e3e162da3c54817dfdf2f678b2b4d1b66aa536ac18bdf57be4',
    CSITE / 'flopscope/_remote_array.py': '8c74bfdcd08502d1b53daac4aa0b8c24809aaa1408c17e6178deb83a9876fd54',
    CSITE / 'flopscope/_budget.py': '8468c0cf31344ab28e470e3b40422e5c4eb0a361af24f51e918a6f6e1ebadc47',
    CSITE / 'flopscope/_connection.py': '6e1e822bdf1fec065d045dfb9da3e67b73e592b993d822aeacf250a218b78e82',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path, tick=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.is_relative_to(P2), 'In-scope regular source/JSON')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
            if tick:
                tick()
    return h.hexdigest()


def read(path):
    path = Path(path)
    require(path.suffix == '.json' and path.stat().st_size < 3 * 1024**2 and not path.is_symlink(),
            'Only bounded regular tiny JSON')
    return json.loads(path.read_text())


def write(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    require(len(raw) < 3 * 1024**2, 'Bounded tiny JSON publication')
    with Path(path).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def numerical_modules():
    return sorted(k for k in sys.modules if k.split('.')[0] in
                  ('numpy', 'scipy', 'torch', 'flopscope', 'flopscope_server', 'threadpoolctl'))


def owner_map(args):
    return {**PINS, HERE / 'tiny_worker.py': args.source_sha256,
            HERE / 'tiny_client.py': args.client_sha256,
            HERE / 'TINY_PROTOCOL.md': args.protocol_sha256,
            HERE / 'watch_tiny.py': args.watcher_sha256,
            HERE / 'TINY_WATCH_SPEC.md': args.watch_spec_sha256,
            HERE / 'TINY_SOURCE_PREPARATION.md': args.decision_sha256}


def admit(args, guard=None):
    require(not numerical_modules(), 'No numerical imports before actual source admission')
    owners = owner_map(args)
    for path, digest in owners.items():
        require(re.fullmatch('[0-9a-f]{64}', digest or '') and sha(path, guard) == digest,
                'Exact reviewed tiny owner: ' + str(path))
    return {str(p): d for p, d in owners.items()}


def _node_bytes(raw, node):
    offsets = [0]
    for line in raw.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return raw[offsets[node.lineno - 1] + node.col_offset:
               offsets[node.end_lineno - 1] + node.end_col_offset]


def helper_source():
    """Only four exact existing functions and six new inert helpers, no card."""
    raw = PARENT.read_bytes()
    fragment = (HERE / 'metric_fnp.py').read_bytes()
    require(hashlib.sha256(raw).hexdigest() == PARENT_SHA
            and hashlib.sha256(fragment).hexdigest() == METRIC_SHA, 'Fresh primitive owners')
    tree = ast.parse(raw.decode())
    names = ('_mm', '_shared_old384_mm', '_shared_old384_finite', '_shared_old384_read')
    chunks = []
    for name in names:
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        require(len(nodes) == (4 if name == '_mm' else 1), 'Exact original primitive ownership: ' + name)
        chunks.append(_node_bytes(raw, nodes[-1]))
    mm = chunks[0]
    require(b'    if m % 32 or k % 32 or n % 32:\n        return a @ b\n' in mm,
            'Exact nonaligned tiny dispatch branch')
    require(all(isinstance(n, ast.FunctionDef) or
                (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))
                for n in ast.parse(fragment.decode()).body), 'Only inert metric definitions')
    text = b'\n\n'.join(chunks) + b'\n\n' + fragment
    proof = dict(parent_sha256=PARENT_SHA, metric_sha256=METRIC_SHA,
                 exact_existing_function_names=list(names),
                 extracted_source_sha256=hashlib.sha256(text).hexdigest(),
                 primitive_bytes_unchanged=True, no_estimator_module_executed=True,
                 no_composed_candidate_materialized=True, tiny_large_dispatch_forbidden=True)
    return text.decode(), proof


def primitive_namespace(fnp):
    source, proof = helper_source()
    def forbidden(*unused, **unused_kw):
        raise RuntimeError('Tiny attempted a large production kernel')
    namespace = dict(fnp=fnp, _WINOGRAD=True, _PACKED_TILE=8192,
                     _square_product=forbidden, _packed_product_4=forbidden, _SCRATCH=None)
    exec(compile(source, 'pair256:exact-tiny-primitives', 'exec'), namespace)
    return namespace, proof


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def fixture(name):
    require(name in CASES, 'Fixed tiny case')
    middle = [[0.0] * 3 for _ in range(8)]
    difference = [[0.0] * 3 for _ in range(8)]
    middle[0][0] = 1.0
    difference[1][0] = 1.0 / 256
    middle[2][1] = middle[3][1] = 0.5
    difference[2][1], difference[3][1] = 1.0 / 32, 1.0 / 64
    middle[4][2], middle[5][2] = 1.0, 0.25
    left = [[middle[i][j] + difference[i][j] for j in range(3)] for i in range(8)]
    right = [[middle[i][j] - difference[i][j] for j in range(3)] for i in range(8)]
    a = [0.25, -0.125, 0.375]
    if name == 'rounded':
        left = [[f32((((i + 1) * (j + 2)) % 11 - 5) / 7) for j in range(3)] for i in range(8)]
        right = [[f32(left[i][j] + (((i + 2) * (j + 1)) % 5 - 2) / 113)
                  for j in range(3)] for i in range(8)]
        a = list(map(f32, (0.2, -0.3, 0.4)))
    elif name == 'signed_zero':
        a = [0.0, -0.0, 0.0]
    return dict(J=[x + y for x, y in zip(left, right)], coefficient=a + [-x for x in a],
        Omega=[[float(i == j) + ((3*i + 5*j + 1) % 13 - 6) / 32 for j in range(4)] for i in range(8)],
        Qp=[[(1., .75, .5, .25)[j] if i == j else 0. for j in range(4)] for i in range(8)],
        S=[[1., .125, 0., 0.], [.125, .75, .0625, 0.],
           [0., .0625, .5, .125], [0., 0., .125, .375]],
        Fold=[[((i+1)*(j+2) % 9 - 4) / 16 for j in range(6)] for i in range(4)],
        lambda_old=[.5, .25, -.125, -.5, -.25, .125],
        W=[[.75*float(i == j) + ((i+2)*(j+3) % 7 - 3) / 32 for j in range(8)] for i in range(8)])


def snapshot(value, mode):
    dtype, shape = str(value.dtype), tuple(value.shape)
    require(dtype in ('float32', 'float64') and 1 <= len(shape) <= 2
            and 0 < math.prod(shape) <= 2048, 'Bounded typed tiny array snapshot')
    if mode == 'remote':
        require(hasattr(value, '_fetch_data'), 'Actual remote ARRAY snapshot, never RemoteScalar')
        raw, actual_shape, actual_dtype = value._fetch_data()
        require(tuple(actual_shape) == shape and str(actual_dtype) == dtype, 'Exact wire metadata')
    else:
        raw = memoryview(value).tobytes()
    code, width = ('f', 4) if dtype == 'float32' else ('d', 8)
    require(len(raw) == width * math.prod(shape), 'Exact snapshot payload length')
    require(all(math.isfinite(x) for x in struct.unpack('<' + str(math.prod(shape)) + code, raw)),
            'All successful snapshot values finite')
    return dict(dtype=dtype, shape=list(shape), raw_sha256=hashlib.sha256(raw).hexdigest(),
                raw_base64=base64.b64encode(raw).decode('ascii'))


def circuit(fnp, ns, ctx, name, data, mode):
    """Fixed old-core + entry operator sequence, no network/card function."""
    sections = []
    def paid(label, action):
        before = int(ctx.flops_used)
        result = action()
        after = int(ctx.flops_used)
        sections.append(dict(name=label, flops=after-before))
        return result
    arrays = paid('all_FP32_inputs', lambda: {k: fnp.asarray(v, dtype=fnp.float32) for k, v in data.items()})
    live = paid('index_input', lambda: fnp.asarray(list(range(8)), dtype=fnp.int64))
    J, coefficient, Omega = (arrays[k] for k in ('J', 'coefficient', 'Omega'))
    Qp, S, Fold, W = (arrays[k] for k in ('Qp', 'S', 'Fold', 'W'))
    mm, finite = ns['_shared_old384_mm'], ns['_shared_old384_finite']
    before_snapshots = {k: snapshot(arrays[k], mode) for k in ('J', 'coefficient', 'Fold')}
    baseline_weight = paid('baseline_weight', lambda: coefficient * coefficient)
    baseline_range = paid('baseline_range', lambda: mm(J, baseline_weight[:, None] * mm(J.T, Omega)))
    metric_basis, kernel = paid('metric_construct', lambda: ns['_pair256_metric'](J, coefficient))
    entry_range = paid('metric_range', lambda: mm(metric_basis, ns['_pair256_apply_kernel'](
        mm(metric_basis.T, Omega), kernel)))
    Y = paid('old_core_range_plus_entry', lambda: entry_range + mm(Qp, mm(S, mm(Qp.T, Omega))))
    paid('metric_Y_guard', lambda: ns['_pair256_nonzero_range'](Y))
    Qn, unused_R = paid('reduced_QR', lambda: fnp.linalg.qr(Y, mode='reduced'))
    del unused_R
    paid('original_Q_guard', lambda: finite(Qn, 'QR basis'))
    require(tuple(Qn.shape) == (8, 4), 'Tiny reduced QR shape')
    Fj = paid('original_Fj_projection', lambda: mm(Qn.T, J))
    baseline_core = paid('baseline_core', lambda: mm(Fj, baseline_weight[:, None] * Fj.T))
    def new_core():
        transformed = ns['_pair256_coordinates'](Fj)
        answer = mm(transformed, ns['_pair256_apply_kernel'](transformed.T, kernel))
        return transformed, answer
    metric_factors, joined_core = paid('metric_core', new_core)
    overlap = paid('original_overlap', lambda: mm(Qn.T, Qp))
    rotated = paid('original_old_factor_rotation', lambda: mm(overlap, Fold))
    core = paid('original_old_core_congruence_plus_join', lambda: mm(overlap, mm(S, overlap.T)) + joined_core)
    F = paid('original_factor_concatenation', lambda: fnp.concatenate([rotated, Fj], axis=1))
    paid('original_F_guard', lambda: finite(F, 'old coefficient factors'))
    paid('original_core_guard', lambda: finite(core, 'weighted Gram core'))
    U = paid('original_basis_push', lambda: mm(W.T, Qn))
    P = paid('original_all_row_reconstruction', lambda: mm(U, F))
    paid('original_reconstruction_guard', lambda: finite(P, 'all-row reconstructed old bank'))
    lam = paid('original_lambda_concatenation', lambda: fnp.concatenate([arrays['lambda_old'], coefficient]))
    state = dict(columns=12, F=F, pre_basis=U)
    plane = paid('original_factorized_plane', lambda: ns['_shared_old384_read'](P, P, lam, live, state))
    direct_plane = paid('diagnostic_direct_plane', lambda: mm(P*P, (P*lam).T))
    dense_gram = paid('diagnostic_dense_metric_Gram', lambda: mm(metric_basis,
        ns['_pair256_apply_kernel'](metric_basis.T, kernel)))
    retained = dict(arrays, metric_basis=metric_basis, kernel00=kernel[0], kernel01=kernel[1],
        kernel11=kernel[2], entry_range=entry_range, Y=Y, Qn=Qn, Fj=Fj,
        metric_factors=metric_factors, joined_core=joined_core, overlap=overlap,
        rotated=rotated, core=core, F=F, U=U, P=P, lam=lam, plane=plane,
        direct_plane=direct_plane, dense_gram=dense_gram,
        baseline_range=baseline_range, baseline_core=baseline_core)
    start = int(ctx.flops_used)
    saved = {k: snapshot(v, mode) for k, v in retained.items()}
    require(int(ctx.flops_used) == start, 'All snapshots add zero hidden FLOPs')
    require(all(saved[k] == value for k, value in before_snapshots.items()), 'Original input/factor bytes unchanged')
    by_name = {x['name']: x['flops'] for x in sections}
    metric_cost = sum(by_name[k] for k in ('metric_construct', 'metric_range', 'metric_Y_guard', 'metric_core'))
    baseline_cost = sum(by_name[k] for k in ('baseline_weight', 'baseline_range', 'baseline_core'))
    allowance = 24*8*3 + 30*3*4 + 256*3 + 4*8*4 + 512
    require(0 <= metric_cost-baseline_cost <= allowance, 'Complete additional metric cost within fixed allowance')
    references = [weakref.ref(v) for v in retained.values()] + [weakref.ref(live), weakref.ref(baseline_weight)]
    return dict(name=name, snapshots=saved, sections=sections,
                original_inputs_byte_exact=True, metric_complete_flops=metric_cost,
                baseline_complete_flops=baseline_cost, metric_additional_flops=metric_cost-baseline_cost,
                metric_allowance_flops=allowance, metric_allowance_passed=True), references


def rejection_fixtures():
    """Host-side fixed test data, prepared before the participant context."""
    data = fixture('dyadic')
    rows = []
    for name in REJECTIONS:
        Jdata = [row[:] for row in data['J']]
        adata = list(data['coefficient'])
        if name == 'opposition':
            adata[-1] += .125
            expected = 'actual coefficient opposition failed'
        elif name == 'dtype':
            expected = 'entry shape/dtype contract'
        elif name == 'odd':
            Jdata, adata = [row[:5] for row in Jdata], adata[:5]
            expected = 'not a complete signed cohort'
        elif name == 'nonfinite':
            Jdata[0][0] = float('inf')
            expected = 'nonfinite or wrong dtype'
        elif name == 'underflow':
            Jdata = [[x*(2.0**-40) for x in row] for row in Jdata]
            adata = [x*(2.0**-40) for x in adata]
            expected = 'metric coefficient underflow'
        elif name == 'PSD':
            expected = 'metric exceeds fixed PSD rounding envelope'
        else:
            expected = 'identically zero metric range'
        rows.append((name,expected,Jdata,adata))
    return rows


def rejected_cases(fnp, ns, ctx, fixtures):
    records = []
    for name,expected,Jdata,adata in fixtures:
        before = int(ctx.flops_used)
        try:
            if name == 'PSD':
                ns['_pair256_psd'](*(fnp.asarray([x], dtype=fnp.float64)
                                    for x in (1., 2., 1., 1.0000000000000142)), 'explicit rejected block')
            elif name == 'zero_range':
                ns['_pair256_nonzero_range'](fnp.zeros((8, 4), dtype=fnp.float32))
            else:
                ns['_pair256_metric'](fnp.asarray(Jdata, dtype=fnp.float64 if name == 'dtype' else fnp.float32),
                                      fnp.asarray(adata, dtype=fnp.float32))
        except RuntimeError as error:
            require(expected is not None and expected in str(error), 'Only the prescribed guard failure counts')
            records.append(dict(name=name, expected_error=expected, error=str(error),
                                flops=int(ctx.flops_used)-before, passed=True))
        else:
            raise RuntimeError('Expected fixed rejection was not enforced: ' + name)
        gc.collect()
    return records


def decode(item):
    shape, dtype = tuple(item['shape']), item['dtype']
    require(dtype in ('float32', 'float64') and 1 <= len(shape) <= 2 and math.prod(shape) <= 2048,
            'Fixed small reference snapshot')
    raw = base64.b64decode(item['raw_base64'], validate=True)
    require(hashlib.sha256(raw).hexdigest() == item['raw_sha256'], 'Reference snapshot hash')
    values = list(struct.unpack('<' + str(math.prod(shape)) + ('f' if dtype == 'float32' else 'd'), raw))
    require(all(math.isfinite(x) for x in values), 'Reference snapshot finite')
    return values if len(shape) == 1 else [values[i*shape[1]:(i+1)*shape[1]] for i in range(shape[0])]


def transpose(a):
    return [list(row) for row in zip(*a)]


def multiply(a, b):
    bt = transpose(b)
    return [[math.fsum(x*y for x, y in zip(row, column, strict=True)) for column in bt] for row in a]


def plus(a, b):
    return [[x+y for x, y in zip(row, other, strict=True)] for row, other in zip(a, b, strict=True)]


def comparison(actual, expected, label, *, absolute=1e-5, relative=5e-5):
    flat_a = [x for row in actual for x in row]
    flat_b = [x for row in expected for x in row]
    require(len(flat_a) == len(flat_b) and all(math.isfinite(x) for x in flat_a+flat_b), 'Reference size/finite: '+label)
    differences = [a-b for a, b in zip(flat_a, flat_b, strict=True)]
    maximum = max(map(abs, differences), default=0.)
    norm = math.sqrt(math.fsum(x*x for x in flat_b))
    error_norm = math.sqrt(math.fsum(x*x for x in differences))
    ratio = error_norm / max(norm, 1e-30)
    require((all(x == 0. for x in flat_a) if norm == 0. else maximum <= absolute and ratio <= relative),
            'Fixed independent reference tolerance: ' + label)
    return dict(name=label, maxabs=maximum, relative_frobenius=ratio,
                reference_norm=norm, denominator=max(norm, 1e-30), passed=True)


def direct_pair_grams(J, coefficient):
    """Independent FP64 third tensors, never the production block polynomial."""
    n, pairs = len(J), len(coefficient)//2
    answer, separate = [[0.]*n for _ in range(n)], []
    for pair in range(pairs):
        tensor = [[[coefficient[pair]*(J[i][pair]*J[j][pair]*J[k][pair]
                   - J[i][pair+pairs]*J[j][pair+pairs]*J[k][pair+pairs])
                   for k in range(n)] for j in range(n)] for i in range(n)]
        gram = [[math.fsum(tensor[i][j][k]*tensor[t][j][k]
                          for j in range(n) for k in range(n)) for t in range(n)] for i in range(n)]
        answer = plus(answer, gram)
        separate.append(gram)
    return answer, separate


def reference_checks(record, tick=None):
    tick = tick or (lambda: None)
    for key,expected in fixture(record['name']).items():
        is_matrix = isinstance(expected[0],list)
        shape = [len(expected),len(expected[0])] if is_matrix else [len(expected)]
        values = [x for row in expected for x in row] if is_matrix else expected
        raw = struct.pack('<'+str(len(values))+'f',*values)
        item = record['snapshots'][key]
        require(item['dtype'] == 'float32' and item['shape'] == shape
                and base64.b64decode(item['raw_base64'],validate=True) == raw,
                'Exact prospective source fixture bytes: '+key)
    s = {k: decode(v) for k, v in record['snapshots'].items()}
    J, lam = s['J'], s['coefficient']
    def exact_fp32(name, values):
        item = record['snapshots'][name]
        matrix = isinstance(values[0], list)
        shape = [len(values), len(values[0])] if matrix else [len(values)]
        flat = [x for row in values for x in row] if matrix else values
        require(item['dtype'] == 'float32' and item['shape'] == shape
                and base64.b64decode(item['raw_base64'], validate=True)
                    == struct.pack('<'+str(len(flat))+'f', *flat),
                'Exact original-coordinate/metric-copy bytes: '+name)
    def metric_copy(rows):
        return [[f32(f32(row[j]+row[j+3])*.5) for j in range(3)]
                + [f32(f32(row[j]-row[j+3])*.5) for j in range(3)] for row in rows]
    exact_fp32('metric_basis', metric_copy(J))
    exact_fp32('metric_factors', metric_copy(s['Fj']))
    exact_fp32('F', [a+b for a,b in zip(s['rotated'], s['Fj'], strict=True)])
    exact_fp32('lam', s['lambda_old']+lam)
    G, per_pair = direct_pair_grams(J, lam)
    checks = []
    expected_kernel = [[], [], []]
    for p in range(3):
        B, D = [row[p] for row in s['metric_basis']], [row[p+3] for row in s['metric_basis']]
        u, v, w = math.fsum(x*x for x in B), math.fsum(x*x for x in D), math.fsum(x*y for x,y in zip(B,D))
        values = (8*lam[p]**2*(u*v+w*w), 8*lam[p]**2*w*(u+v),
                  4*lam[p]**2*(u*u+v*v+2*w*w))
        for row,value in zip(expected_kernel, values, strict=True):
            row.append(f32(value))
    for name,expected in zip(('kernel00','kernel01','kernel11'), expected_kernel, strict=True):
        checks.append(comparison([s[name]], [expected], 'independent stored '+name))
    by_original = [[0.]*8 for _ in range(8)]
    by_middle = [[0.]*8 for _ in range(8)]
    for p in range(3):
        x, y = [row[p] for row in J], [row[p+3] for row in J]
        xx, xy, yy = math.fsum(v*v for v in x), math.fsum(a*b for a,b in zip(x,y)), math.fsum(v*v for v in y)
        K = [[lam[p]**2*xx**2, -lam[p]**2*xy**2], [-lam[p]**2*xy**2, lam[p]**2*yy**2]]
        X = [[a,b] for a,b in zip(x,y)]
        by_original = plus(by_original, multiply(multiply(X,K),transpose(X)))
        B, D = [(a+b)/2 for a,b in zip(x,y)], [(a-b)/2 for a,b in zip(x,y)]
        u,v,w = math.fsum(z*z for z in B),math.fsum(z*z for z in D),math.fsum(a*b for a,b in zip(B,D))
        Kp = [[4*lam[p]**2*2*(u*v+w*w),4*lam[p]**2*2*w*(u+v)],
              [4*lam[p]**2*2*w*(u+v),4*lam[p]**2*(u*u+v*v+2*w*w)]]
        BD = [[a,b] for a,b in zip(B,D)]
        by_middle = plus(by_middle,multiply(multiply(BD,Kp),transpose(BD)))
        require(Kp[0][0] >= 0 and Kp[1][1] >= 0
                and Kp[0][1]**2 <= Kp[0][0]*Kp[1][1]*(1+64*2**-52), 'Independent exact-polynomial PSD')
    checks += [comparison(by_original,G,'FP64 original K/direct tensor',absolute=2e-12,relative=2e-12),
               comparison(by_middle,G,'FP64 midpoint/direct tensor',absolute=2e-12,relative=2e-12),
               comparison(s['dense_gram'],G,'actual entry metric/direct tensor')]
    tick()
    expected_entry = multiply(G,s['Omega'])
    expected_Y = plus(expected_entry,multiply(s['Qp'],multiply(s['S'],multiply(transpose(s['Qp']),s['Omega']))))
    Q = s['Qn']
    expected_Fj = multiply(transpose(Q),J)
    expected_overlap = multiply(transpose(Q),s['Qp'])
    expected_rotated = multiply(expected_overlap,s['Fold'])
    expected_joined = multiply(multiply(transpose(Q),G),Q)
    expected_core = plus(multiply(multiply(expected_overlap,s['S']),transpose(expected_overlap)),expected_joined)
    expected_F = [a+b for a,b in zip(expected_rotated,expected_Fj)]
    expected_U = multiply(transpose(s['W']),Q)
    expected_P = multiply(expected_U,expected_F)
    squared = [x*x for x in lam]
    weighted_Jt = [[weight*x for x in row] for row,weight in zip(transpose(J),squared,strict=True)]
    weighted_Fjt = [[weight*x for x in row] for row,weight in zip(transpose(s['Fj']),squared,strict=True)]
    checks += [comparison(s['baseline_range'], multiply(J,multiply(weighted_Jt,s['Omega'])), 'paid baseline range'),
               comparison(s['baseline_core'], multiply(s['Fj'],weighted_Fjt), 'paid baseline core')]
    for key, expected in (('entry_range',expected_entry),('Y',expected_Y),('Fj',expected_Fj),
                          ('overlap',expected_overlap),('rotated',expected_rotated),('joined_core',expected_joined),
                          ('core',expected_core),('F',expected_F),('U',expected_U),('P',expected_P)):
        checks.append(comparison(s[key],expected,key))
    identity = [[float(i == j) for j in range(4)] for i in range(4)]
    checks.append(comparison(multiply(transpose(Q),Q),identity,'QR orthogonality',relative=1e-5))
    plane = [[math.fsum(s['lam'][k]*s['P'][i][k]**2*s['P'][j][k] for k in range(12))
              for j in range(8)] for i in range(8)]
    checks += [comparison(s['plane'],plane,'original x/y factorized cubic plane'),
               comparison(s['direct_plane'],plane,'paid direct cubic plane'),
               comparison(s['plane'],s['direct_plane'],'same-interface physical read')]
    if record['name'] == 'dyadic':
        require(all(x == 0. for row in per_pair[2] for x in row), 'Exact cancelled third pair')
        epsilon, a = 2**-8, .25
        toy = [[8*a*a*epsilon**2,0.],[0.,4*a*a*epsilon**2*(1+epsilon**4)]]
        checks.append(comparison([[s['dense_gram'][i][j] for j in (0,1)] for i in (0,1)],toy,'paired derivative toy'))
    if record['name'] == 'signed_zero':
        expected = struct.pack('<6f',0.,-0.,0.,-0.,0.,-0.)
        require(base64.b64decode(record['snapshots']['coefficient']['raw_base64']) == expected,
                'Original signed zero coefficient bytes retained')
    tick()
    return checks


def arithmetic(flops, fnp, ns, mode, index, guard):
    kwargs = dict(flop_budget=CALL_BUDGET, quiet=True, namespace='pair256-tiny-'+mode)
    if mode == 'local':
        kwargs['wall_time_limit_s'] = 120.
    inputs = [(name,fixture(name)) for name in CASES]
    rejection_inputs = rejection_fixtures()
    rows, references = [], []
    with flops.BudgetContext(**kwargs) as ctx:
        for name,data in inputs:
            row, refs = circuit(fnp, ns, ctx, name, data, mode)
            rows.append(row); references.extend(refs)
            del refs
            gc.collect()
            require(all(ref() is None for ref in references), 'All successful tiny arrays retired inside context')
            guard()
        rejected = rejected_cases(fnp,ns,ctx,rejection_inputs)
        gc.collect()
        require(all(ref() is None for ref in references), 'No retained physical/metric arrays before close')
    summary = ctx.summary_dict()
    complete = int(summary['flops_used'])
    explained = sum(s['flops'] for row in rows for s in row['sections']) + sum(x['flops'] for x in rejected)
    require(complete == explained and 0 < complete < CALL_BUDGET, 'Every actual input/operator/guard FLOP reconciles')
    ledger = [dict(op._asdict()) for op in ctx.op_log] if mode == 'local' else None
    if ledger is not None:
        require(sum(int(x['flop_cost']) for x in ledger) == complete, 'Actual complete native operation ledger')
    del ctx
    gc.collect()
    limit = 120. if mode == 'local' else 60.
    require(all(math.isfinite(summary[k]) and summary[k] >= 0
                for k in ('wall_time_s','residual_wall_time_s','flopscope_backend_time_s','flopscope_overhead_time_s'))
            and summary['wall_time_s'] < limit, 'Actual original context wall/timing fields')
    require(mode == 'local' or summary['residual_wall_time_s'] <= .4, 'Original actual RPC residual<=.4s')
    checks = [dict(case=row['name'], checks=reference_checks(row,guard)) for row in rows]
    return dict(schema=SCHEMA, mode=mode, call=index, passed=True, completed=True,
                context_retired=True, all_array_weakrefs_retired=True, rows=rows, rejected_cases=rejected,
                reference_checks=checks, complete_flops=complete, bill=summary,
                native_operation_ledger=ledger, remote_sections_are_actual_budget_deltas=(mode == 'remote'),
                own_complete_bill=True, original_physical_coordinates_verified=True,
                metric_allowance_passed=True, model_calls=0, target_accessed=False)


def backend_attestation(mode, sources, guard=None):
    root = CSITE if mode == 'remote' else OSITE
    if mode == 'remote':
        require(importlib.util.find_spec('numpy') is None and not any(k.split('.')[0] == 'numpy' for k in sys.modules),
                'NumPy-free actual client')
    else:
        importlib.import_module('numpy.linalg._linalg')
        importlib.import_module('numpy.linalg._umath_linalg')
        pools = importlib.import_module('threadpoolctl').threadpool_info()
        blas = [p for p in pools if p.get('user_api') == 'blas']
        require(len(blas) == 1 and blas[0]['internal_api'] == 'openblas'
                and blas[0]['num_threads'] == 1 and blas[0]['version'] == '0.3.31.188.0'
                and blas[0]['threading_layer'] == 'pthreads'
                and Path(blas[0]['filepath']).resolve() == OSITE/'numpy/.dylibs/libscipy_openblas64_.dylib',
                'Actual sole production OpenBLAS, no Accelerate')
    modules = {}
    for name,module in tuple(sys.modules.items()):
        if name.split('.')[0] not in ('numpy','flopscope','flopscope_server','threadpoolctl'):
            continue
        value = getattr(module,'__file__',None)
        if value is None:
            continue
        path = Path(value).resolve()
        require(path.is_relative_to(root), 'No mixed numerical installation')
        digest = sha(path,guard)
        require(str(path) not in sources or sources[str(path)] == digest, 'Actual imported source remains pinned')
        sources[str(path)] = digest
        modules[name] = dict(path=str(path),sha256=digest)
    require(not any(k.split('.')[0] in ('scipy','torch') for k in sys.modules), 'No other numerical backend')
    return dict(mode=mode,site=str(root),modules=modules,
                threadpools=[] if mode == 'remote' else pools,
                actual_origins_qualified=True,production_OpenBLAS_qualified=(mode != 'remote'),
                numpy_free=(mode == 'remote'))


def verify_call(record, mode, index, tick=None):
    require(record['schema'] == SCHEMA and record['mode'] == mode and record['call'] == index
            and all(record.get(k) is True for k in ('passed','completed','context_retired',
                'all_array_weakrefs_retired','own_complete_bill','original_physical_coordinates_verified','metric_allowance_passed'))
            and record['model_calls'] == 0 and record['target_accessed'] is False
            and [r['name'] for r in record['rows']] == list(CASES)
            and [r['name'] for r in record['rejected_cases']] == list(REJECTIONS)
            and all(r['passed'] and r['flops'] >= 0 for r in record['rejected_cases']), 'Complete fixed tiny call')
    explained = sum(s['flops'] for row in record['rows'] for s in row['sections']) + sum(x['flops'] for x in record['rejected_cases'])
    require(explained == record['complete_flops'] == record['bill']['flops_used'] and 0 < explained < CALL_BUDGET,
            'Actual complete tiny bill reconciliation')
    if mode == 'local':
        require(isinstance(record['native_operation_ledger'],list)
                and sum(int(x['flop_cost']) for x in record['native_operation_ledger']) == explained,
                'Independent complete native ledger sum')
    else:
        require(record['native_operation_ledger'] is None and record['remote_sections_are_actual_budget_deltas'],
                'No invented client operation ledger')
    for item,(name,expected,_,_) in zip(record['rejected_cases'],rejection_fixtures(),strict=True):
        require(item['name'] == name and item['expected_error'] == expected and expected in item['error'],
                'Exact prescribed rejected case, not an unrelated exception')
    for row in record['rows']:
        costs = {x['name']:x['flops'] for x in row['sections']}
        require(len(costs) == len(row['sections']) and all(type(x) is int and x >= 0 for x in costs.values()),
                'Unique nonnegative actual operator sections')
        metric = sum(costs[k] for k in ('metric_construct','metric_range','metric_Y_guard','metric_core'))
        baseline = sum(costs[k] for k in ('baseline_weight','baseline_range','baseline_core'))
        require(row['original_inputs_byte_exact'] and row['metric_allowance_passed']
                and row['metric_allowance_flops'] == 2344 and 0 <= row['metric_additional_flops'] <= 2344
                and metric == row['metric_complete_flops'] and baseline == row['baseline_complete_flops']
                and metric-baseline == row['metric_additional_flops'],
                'Original coordinates and fixed complete metric allowance')
        reference_checks(row,tick)
    limit = 120. if mode == 'local' else 60.
    require(all(math.isfinite(record['bill'][k]) and record['bill'][k] >= 0 for k in
                ('wall_time_s','residual_wall_time_s','flopscope_backend_time_s','flopscope_overhead_time_s'))
            and 0 <= record['bill']['wall_time_s'] < limit
            and (mode == 'local' or 0 <= record['bill']['residual_wall_time_s'] <= .4), 'Original actual context limits')


def call_records(result, tick=None):
    out = HERE/'tiny_qualification_v1'/result['mode']
    require(len(result['calls']) == 2, 'Two freshly repeated contexts')
    records = []
    for index,meta in enumerate(result['calls']):
        path = out/f'call_{index:02d}.json'
        require(meta['path'] == str(path) and sha(path,tick) == meta['sha256'] == result['artifacts'][str(path)],
                'Actual exclusive context receipt')
        record = read(path)
        verify_call(record,result['mode'],index,tick)
        records.append(record)
    return records


def repeat_check(records):
    first, second = records
    require(first['complete_flops'] == second['complete_flops'], 'Exact own repeated complete F')
    for a,b in zip(first['rows'],second['rows'],strict=True):
        require(a['name'] == b['name'] and a['snapshots'] == b['snapshots'] and a['sections'] == b['sections'],
                'Exact own repeated input/output bytes and every operator section')
    require(first['rejected_cases'] == second['rejected_cases'], 'Exact own repeated rejection costs/categories')
    return True


def compare(local, remote, tick=None):
    require(all(x.get(k) is True for x in (local,remote) for k in
                ('passed','completed','own_complete_bill','source_hashes_unchanged','two_contexts_completed'))
            and local['mode'] == 'local' and remote['mode'] == 'remote'
            and local['admitted_sources'] == remote['admitted_sources']
            and local['primitive_proof'] == remote['primitive_proof'], 'Two actual fresh matched mode chains')
    a,b = call_records(local,tick),call_records(remote,tick)
    repeat_check(a);repeat_check(b)
    checks = []
    for left,right in zip(a,b,strict=True):
        require(left['complete_flops'] == right['complete_flops'], 'Exact native/RPC complete operator bill')
        require(left['rejected_cases'] == right['rejected_cases'], 'Same native/RPC expected errors and actual costs')
        for x,y in zip(left['rows'],right['rows'],strict=True):
            require(x['name'] == y['name'] and x['sections'] == y['sections'], 'All actual native/RPC section tariffs')
            for name,item in x['snapshots'].items():
                other = y['snapshots'][name]
                require((item['dtype'],item['shape']) == (other['dtype'],other['shape']), 'Same interface dtype/shape')
                if name in fixture(x['name']):
                    require(item == other, 'Byte-exact same paid inputs across interfaces')
                values1,values2 = decode(item),decode(other)
                if len(item['shape']) == 1:
                    values1,values2 = [values1],[values2]
                detail = comparison(values1,values2,'native/RPC '+name,relative=1e30)
                checks.append(dict(call=left['call'],case=x['name'],**detail))
    return dict(passed=True,four_contexts_completed=True,both_interfaces_repeated=True,
        native_RPC_complete_F_exact=True,all_native_RPC_arrays_within_1e_minus5=True,
        all_independent_references_passed=True,original_physical_coordinates_verified=True,
        complete_metric_allowance_passed=True,expected_rejections_passed=True,
        details=checks,model_calls=0,truth_accessed=False,full_estimator_fidelity_established=False,
        raw_value_measured=False,automatic_next_stage=False)


def main(forced_mode='local'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--mode',choices=('local','remote','server'))
    for key in ('source','client','protocol','watcher','watch-spec','decision'):
        parser.add_argument('--'+key+'-sha256')
    parser.add_argument('--server-url')
    args = parser.parse_args()
    if not args.run:
        parser.print_help();return 0
    require(args.mode in (('local','server') if forced_mode == 'local' else ('remote',)), 'Exact worker/client entry ownership')
    began = time.monotonic()
    out = HERE/'tiny_qualification_v1'/args.mode
    require(not out.exists() and not HERE.is_symlink(),'Exclusive fresh tiny mode')
    def guard():
        require(time.monotonic()-began < 270 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 1_000_000_000,
                'Tiny mode270s/self1GB including final publication')
    def alarm(*unused):
        raise TimeoutError('Tiny current context/whole deadline')
    require(sys.platform == 'darwin' and sys.byteorder == 'little' and sys.flags.optimize == 0
            and sys.dont_write_bytecode and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
            and all(os.environ.get(k) == '1' for k in THREADS)
            and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
            and not os.environ.get('PYTHONPATH') and not os.environ.get('PYTHONOPTIMIZE')
            and os.environ.get('FLOPSCOPE_MAX_ARRAY_BYTES') == '4294967296', 'Exact tiny runtime environment')
    os.nice(max(0,10-os.getpriority(os.PRIO_PROCESS,0)))
    require(Path(sys.prefix).resolve() == (CLIENT if args.mode == 'remote' else OPENBLAS).resolve(), 'Actual selected interpreter')
    if args.mode != 'local':
        require(args.server_url and args.server_url.startswith('ipc:///private/tmp/')
                and args.server_url.endswith('/flops.sock') and os.environ.get('FLOPSCOPE_SERVER_URL') == args.server_url,
                'Only the parent-owned local IPC endpoint')
    else:
        require(not args.server_url and not os.environ.get('FLOPSCOPE_SERVER_URL'), 'Actual in-process native mode')
    versions = {'flopscope-client':'0.12.0'} if args.mode == 'remote' else {
        'numpy':'2.4.6','flopscope':'0.12.0','flopscope-server':'0.12.0','threadpoolctl':'3.6.0'}
    require(all(importlib.metadata.version(k) == v for k,v in versions.items()), 'Actual fixed package versions')
    if args.mode == 'remote':
        require(importlib.util.find_spec('numpy') is None, 'NumPy-free production client before imports')
    admitted = admit(args,guard)
    source,proof = helper_source()
    require(not numerical_modules(), 'Source extraction did not import numerics')
    del source
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,270.)
    if args.mode == 'server':
        # The unchanged official entry point owns all server behavior after
        # this no-array backend attestation. Parent owns its process group.
        import flopscope
        import flopscope.numpy
        from flopscope_server.__main__ import main as server_main
        sources = dict(admitted)
        backend = backend_attestation('server',sources,guard)
        path = HERE/'tiny_qualification_v1/server_attestation.json'
        write(path,dict(schema=SCHEMA,mode='server',passed=True,backend_attestation=backend,
            source_bindings=sources,admitted_sources=admitted,pid=os.getpid(),url=args.server_url,
            no_fixture_arrays_created=True,model_calls=0,truth_accessed=False))
        sys.argv = ['flopscope_server','--url',args.server_url,'--timeout','60']
        server_main()
        return 0
    out.mkdir(parents=True,exist_ok=False)
    result = dict(schema=SCHEMA,mode=args.mode,passed=False,completed=False,model_calls=0,
        truth_accessed=False,candidate_changed=False,full_estimator_fidelity_established=False,
        raw_value_measured=False,automatic_next_stage=False,source_bindings=dict(admitted),
        admitted_sources=admitted,input_bindings={},artifacts={},calls=[],primitive_proof=proof,
        package_versions=versions)
    try:
        import flopscope as flops
        import flopscope.numpy as fnp
        require(Path(flops.__file__).resolve() == (CSITE if args.mode == 'remote' else OSITE)/'flopscope/__init__.py',
                'Actual FNP origin')
        result['backend_attestation'] = backend_attestation(args.mode,result['source_bindings'],guard)
        ns,actual_proof = primitive_namespace(fnp)
        require(actual_proof == proof, 'Same exact primitive source after import')
        records = []
        for index in range(2):
            limit = 120. if args.mode == 'local' else 60.
            signal.setitimer(signal.ITIMER_REAL,min(limit,270-(time.monotonic()-began)))
            record = arithmetic(flops,fnp,ns,args.mode,index,guard)
            signal.setitimer(signal.ITIMER_REAL,270-(time.monotonic()-began))
            records.append(record)
            path = out/f'call_{index:02d}.json'
            digest = write(path,record)
            result['calls'].append(dict(path=str(path),sha256=digest))
            result['artifacts'][str(path)] = digest
            print(json.dumps(dict(event='retired',mode=args.mode,call=index,path=str(path),
                                  complete_sha256=digest,complete_flops=record['complete_flops'])),flush=True)
            guard()
        repeat_check(records)
        if args.mode == 'remote':
            from flopscope._connection import reset_connection
            reset_connection()
        del ns,fnp,flops,records
        gc.collect()
        backend_attestation(args.mode,result['source_bindings'],guard)
        for path,digest in result['source_bindings'].items():
            require(sha(Path(path),guard) == digest, 'Fresh final actual source union')
        result.update(passed=True,completed=True,own_complete_bill=True,source_hashes_unchanged=True,
            two_contexts_completed=True,retired_contexts=2,all_array_weakrefs_retired=True,
            all_independent_references_passed=True,metric_allowance_passed=True,expected_rejections_passed=True,
            original_physical_coordinates_verified=True,repeat_byte_exact=True,
            numerical_imports='native_OpenBLAS_FNP' if args.mode == 'local' else 'NumPy_free_RPC_client')
    except BaseException as error:
        result.update(passed=False,completed=False,error_type=type(error).__name__,error=str(error)[:1000])
    finally:
        if not result['passed']:
            signal.setitimer(signal.ITIMER_REAL,2.)
        result.update(elapsed_s=time.monotonic()-began,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if result['passed']:
            guard()
        digest = write(out/'result.json',result)
        end = write(out/'completion.json',dict(schema=SCHEMA,passed=result['passed'],completed=result['completed'],result_sha256=digest))
        if result['passed']:
            guard()
        write(out/'finalization.json',dict(schema=SCHEMA,passed=result['passed'],completed=result['completed'],
              result_sha256=digest,completion_sha256=end,elapsed_s=time.monotonic()-began))
        if result['passed']:
            guard()
        signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps(dict(mode=args.mode,passed=result['passed'],result_sha256=digest)),flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
