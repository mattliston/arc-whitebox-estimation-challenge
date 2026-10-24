"""Target-free GNN compression on native teacher captures; no network targets.

Runs in judge Python with NumPy. Train ranks and hyperparameters are frozen in
code. Native full32 plus release gates are still required for every candidate.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import ast
import base64
import copy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import zlib
import numpy as np

TRAIN = (35, 150, 200, 250)
VALID = (259, 277, 287)
RIDGE = 1e-4
BANK_RANK35_BILL = 236823679610


def utc():
    return subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path, text):
    with Path(path).open('x') as f:
        f.write(text)


def jwrite(path, obj):
    write(path, json.dumps(obj, indent=2, allow_nan=False) + '\n')


def decode(literal):
    return {k: np.frombuffer(zlib.decompress(base64.b64decode(v)), dtype='<f4').reshape(sh).copy()
            for k, (sh, v) in literal.items()}


def encode(arrays):
    return {k: (tuple(v.shape), base64.b64encode(zlib.compress(np.asarray(v, dtype='<f4').tobytes())).decode())
            for k, v in arrays.items()}


def silu(x):
    return x / (1 + np.exp(-x))


def array(rec, key):
    a = rec[key]
    return np.asarray(a['data'], dtype=np.float32).reshape(a['shape'])


def ridge_affine(x, y):
    # Standardized feature ridge, intercept unpenalized. Fixed before capture.
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-8)
    z = (x - xm) / scale
    gram = z.T @ z
    coef = np.linalg.solve(gram + RIDGE * len(z) * np.eye(z.shape[1]), z.T @ (y - ym))
    coef = coef / scale[:, None]
    intercept = ym - xm @ coef
    return coef.astype(np.float32), intercept.astype(np.float32)


def choose(values, outgoing, count):
    score = values.astype(np.float64).var(axis=0) * np.sum(np.asarray(outgoing, dtype=np.float64)**2, axis=0)
    # Stable order and original coordinates retained; no targets used.
    selected = np.sort(np.argsort(-score, kind='stable')[:count])
    return selected, score


def literals_and_functions(source):
    t = ast.parse(source)
    lines = source.splitlines()
    literals, functions = {}, {}
    for node in t.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ('_GNN_LITERAL', '_FINAL_LITERAL', '_HC_U', '_HCF_U'):
                literals[name] = ast.literal_eval(node.value)
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = '\n'.join(lines[node.lineno-1:node.end_lineno])
    return literals, functions


def collect(root):
    rows, paths, metadata = [], [], None
    for path in sorted(root.rglob('rank*_scored_L*.json.gz')):
        if path.parent.name != 'gnn_teacher_v2':
            continue
        rec = json.load(gzip.open(path, 'rt'))
        if rec['rank'] not in TRAIN + VALID:
            continue
        meta = json.loads((path.parent / 'METADATA_v2.json').read_text())
        if metadata is None:
            metadata = meta
        assert meta['gnn_literal'] == metadata['gnn_literal']
        assert meta['hc_u'] == metadata['hc_u'] and meta['hcf_u'] == metadata['hcf_u']
        rows.append(rec)
        paths.append(dict(path=str(path), sha256=sha(path)))
    keys = [(x['rank'], x['layer']) for x in rows]
    assert len(set(keys)) == len(keys), 'duplicate scored captures'
    assert set(keys) == {(r, l) for r in TRAIN + VALID for l in range(1, 16)}, ('incomplete capture', len(keys))
    return rows, paths, metadata


def matrices(rows, P):
    H = np.concatenate([array(r, 'H') for r in rows])
    bias = np.concatenate([np.broadcast_to(array(r, 'bias'), (128, 128)) for r in rows])
    z1 = H @ P['enc0_w'].T + bias
    h1 = silu(z1)
    z2 = h1 @ P['enc2_w'].T + P['enc2_b']
    h2 = silu(z2)
    state = h2 @ P['enc4_w'].T + P['enc4_b']
    saved_h2 = np.concatenate([array(r, 'hidden2') for r in rows])
    saved_state = np.concatenate([array(r, 'state') for r in rows])
    # Host BLAS can differ at rounding level; disallow stale feature/bias captures.
    np.testing.assert_allclose(h2, saved_h2, atol=5e-5, rtol=3e-5)
    np.testing.assert_allclose(state, saved_state, atol=5e-5, rtol=3e-5)
    return dict(H=H, bias=bias, h1=h1, z2=z2, h2=h2, state=saved_state,
                messages=np.concatenate([array(r, 'messages').reshape(128, 48) for r in rows]),
                replay_h2_max_abs=float(np.max(np.abs(h2-saved_h2))),
                replay_state_max_abs=float(np.max(np.abs(state-saved_state))))


def state_error(rows, got, expected, P):
    diff = got.astype(np.float64) - expected.astype(np.float64)
    by_layer = []
    offset = 0
    all_deltas = []
    for rec in rows:
        dd = diff[offset:offset+128]
        l = rec['layer']
        sigma = array(rec, 'sigma').astype(np.float64)
        gain = 1 + .25 * np.tanh(float(P['gain'][l]))
        delta = sigma * float(P['rms'][l]) * (dd @ P['head'][l].astype(np.float64)) * gain
        by_layer.append(dict(rank=rec['rank'], layer=l, delta_rms=float(np.sqrt(np.mean(delta*delta)))))
        all_deltas.append(delta)
        offset += 128
    return dict(state_relative_l2=float(np.linalg.norm(diff) / max(np.linalg.norm(expected), 1e-30)),
                local_head_delta_rms=float(np.sqrt(np.mean(np.concatenate(all_deltas)**2))),
                local_head_delta_by_rank_layer=by_layer,
                limitation='Frozen teacher-input approximation error, not propagated estimator MSE or adjusted gain.')


def metric_function(original_E):
    return '''\n_VMFR_ORIGINAL_W3_ENCODER = %s
_VMFR_W3_ENCODER_ARRAY = None
_W3R_METRIC = None
def _w3r_metric():
    global _W3R_METRIC, _VMFR_W3_ENCODER_ARRAY
    if _W3R_METRIC is None:
        if _VMFR_W3_ENCODER_ARRAY is None:
            _VMFR_W3_ENCODER_ARRAY = fnp.asarray(_VMFR_ORIGINAL_W3_ENCODER, dtype=fnp.float64)
        E = _VMFR_W3_ENCODER_ARRAY
        values, vectors = fnp.linalg.eigh(fnp.matmul(E.T, E))
        scales = fnp.sqrt(fnp.maximum(values, 1e-30))
        root = fnp.matmul(vectors * scales[None, :], vectors.T)
        inverse = fnp.matmul(vectors / scales[None, :], vectors.T)
        _W3R_METRIC = (root, inverse)
        _W3R_STATS['metric_builds'] += 1
    return _W3R_METRIC
''' % repr(original_E.astype(float).tolist())


def export(out, name, parents, newP, literal_extra, code_extra, detail):
    tail = '\n# Target-free GNN compression; APPROXIMATION, not an exact runtime patch.\n'
    tail += 'assert _GNN_PARAMS is None and not _E2_GNN_STATE\n'
    tail += 'assert not _HC_CONST and not _HCF_CONST and _MESSAGE_EDGE_PARAMETERS is None\n'
    tail += '_GNN_LITERAL = ' + repr(encode(newP)) + '\n'
    tail += literal_extra + code_extra
    tail += '\n_FAST_RESCUE_FRAC = {}\n'
    tailpath = out / ('tail_' + name + '_v1.py')
    write(tailpath, tail)
    record = dict(name=name, tail=str(tailpath), tail_sha256=sha(tailpath), **detail)
    for role, parent in parents.items():
        text = parent.read_text() + tail
        path = out / ('candidate_' + name + ('_sci_v1.py' if role == 'science' else '_rel_v1.py'))
        compile(text, str(path), 'exec')
        write(path, text)
        record[role] = dict(path=str(path), sha256=sha(path), parent_sha256=sha(parent))
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--captures', type=Path, required=True)
    ap.add_argument('--parent-science', type=Path, required=True)
    ap.add_argument('--parent-release', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    jwrite(a.out/'PREREG_v1.json', dict(utc=utc(), train=TRAIN, valid=VALID,
            encoder_widths=[96, 64], message_widths=[12, 8], ridge=RIDGE,
            target_free=True, source_sha256=sha(__file__),
            reference_submission=337393,
            no_network_gain_inference_from_teacher_error=True))
    rows, paths, meta = collect(a.captures)
    literals, functions = literals_and_functions(a.parent_science.read_text())
    P = decode(literals['_GNN_LITERAL'])
    capturedP = decode(meta['gnn_literal'])
    for key in P:
        assert np.array_equal(P[key], capturedP[key]), ('teacher/source parameter mismatch', key)
    assert np.array_equal(np.asarray(literals['_HC_U']), np.asarray(meta['hc_u']))
    assert np.array_equal(np.asarray(literals['_HCF_U']), np.asarray(meta['hcf_u']))
    train_rows = [r for r in rows if r['rank'] in TRAIN]
    valid_rows = [r for r in rows if r['rank'] in VALID]
    train = matrices(train_rows, P)
    valid = matrices(valid_rows, P)
    parents = dict(science=a.parent_science, release=a.parent_release)
    candidates = []
    for width in (96, 64):
        keep1, score1 = choose(train['h1'], P['enc2_w'], width)
        keep2, score2 = choose(train['h2'], P['enc4_w'], width)
        w2, b2 = ridge_affine(train['h1'][:, keep1], train['z2'][:, keep2])
        new_h2 = silu(train['h1'][:, keep1] @ w2 + b2)
        w4, b4 = ridge_affine(new_h2, train['state'])
        q = {k: v.copy() for k, v in P.items()}
        q.update(enc0_w=P['enc0_w'][keep1], enc0_b=P['enc0_b'][keep1],
                 enc2_w=w2.T, enc2_b=b2, enc4_w=w4.T, enc4_b=b4)
        stats = {}
        for label, data, rr in [('train', train, train_rows), ('valid', valid, valid_rows)]:
            got = silu(data['h1'][:, keep1] @ w2 + b2) @ w4 + b4
            stats[label] = state_error(rr, got, data['state'], P)
        extra = '_HC_U = ' + repr(np.asarray(literals['_HC_U'])[:, keep1].tolist()) + '\n'
        extra += '_HCF_U = ' + repr(np.asarray(literals['_HCF_U'])[:, keep1].tolist()) + '\n'
        # Preserve the old W3 response metric while changing encoder width.
        code = metric_function(P['enc0_w'][:, 62:78])
        old = 15*1024*(128*(2*126-1)+128*(2*128-1)+64*(2*128-1)+40*128)
        new = 15*1024*(width*(2*126-1)+width*(2*width-1)+64*(2*width-1)+40*width)
        detail = dict(kind='encoder_hidden_distillation', hidden_width=width,
                      retained_first=keep1.tolist(), retained_second=keep2.tolist(),
                      teacher_errors=stats, first_scores=score1.tolist(), second_scores=score2.tolist(),
                      fixed_geometry_bill_saving_flops=old-new,
                      fixed_geometry_bill_pct_vs_best337393_rank35=-100*(old-new)/BANK_RANK35_BILL,
                      qualification='NOT NATIVE TESTED; parent full32 does not transfer to this approximation')
        candidates.append(export(a.out, 'enc%d_teacher' % width, parents, q, extra, code, detail))
    # Message pruning needs the current literal, including the EH14 columns.
    # Execute only its constant-only assignments in a sandboxed plain namespace.
    tree = ast.parse(a.parent_science.read_text())
    ns = {'__builtins__': {}}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('_MESSAGE_EDGE_LITERAL', '_O20_EDGE_HIDDEN_KEEP', '_O20_EDGE_HIDDEN'):
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<literal-only>', 'exec'), ns, ns)
    edge = {k:np.asarray(v, dtype=np.float32) for k,v in ns['_MESSAGE_EDGE_LITERAL'].items()}
    for width in (12, 8):
        keep = []
        for g in range(3):
            s = np.zeros(16)
            for block in (g, g+3):
                cols = slice(30+16*block, 30+16*(block+1))
                s += train['H'][:, cols].astype(float).var(axis=0) * np.sum(P['enc0_w'][:, cols].astype(float)**2, axis=0)
            keep.append(np.sort(np.argsort(-s, kind='stable')[:width]))
        message_indices = np.concatenate([16*g+keep[g] for g in range(3)])
        input_indices = np.concatenate([np.arange(30)] + [30+16*g+keep[g % 3] for g in range(6)])
        w0, add_b0 = ridge_affine(train['H'][:, input_indices], train['H'] @ P['enc0_w'].T)
        q = {k:v.copy() for k,v in P.items()}
        q.update(msg_w=P['msg_w'][message_indices], msg_b=P['msg_b'][message_indices],
                 enc0_w=w0.T, enc0_b=P['enc0_b']+add_b0)
        # Fold discarded message content into retained context rows without
        # changing the deterministic physical-feature or edge bias branches.
        mt = np.clip(train['messages'], -8, 8)
        xx = mt[:, message_indices].astype(float)
        gram = xx.T @ xx
        edge_new = {}
        for key, value in edge.items():
            edge_new[key] = np.linalg.solve(gram + RIDGE*len(xx)*np.eye(len(message_indices)), xx.T @ (mt @ value)).astype(np.float32)
        final = decode(literals['_FINAL_LITERAL'])
        final['w0'] = final['w0'][:, input_indices]
        extra = '_VMFR_CM = %d\n' % width
        extra += '_MESSAGE_EDGE_LITERAL = ' + repr({k:v.tolist() for k,v in edge_new.items()}) + '\n'
        extra += '_FINAL_LITERAL = ' + repr(encode(final)) + '\n'
        product = functions['_gop2_message_product'].replace('assert X.shape == (3, n, 16)', 'assert X.shape == (3, n, _VMFR_CM)').replace('(3, n, 16), dtype', '(3, n, _VMFR_CM), dtype')
        assert product != functions['_gop2_message_product']
        edgefun = functions['_g5_message_edge_base'].replace('cm != 16', 'cm != _VMFR_CM').replace('(48, _O20_EDGE_HIDDEN)', '(3 * _VMFR_CM, _O20_EDGE_HIDDEN)')
        assert edgefun != functions['_g5_message_edge_base']
        code = metric_function(P['enc0_w'][:, 62+keep[2]]) + '\n' + product + '\n' + edgefun
        code += '\n_gn_message_edge_source_base = _g5_message_edge_base\n'
        stats = {}
        for label, data, rr in [('train',train,train_rows), ('valid',valid,valid_rows)]:
            z = data['H'][:, input_indices] @ w0 + data['bias'] + add_b0
            hh = silu(silu(z) @ P['enc2_w'].T + P['enc2_b'])
            got = hh @ P['enc4_w'].T + P['enc4_b']
            stats[label] = state_error(rr, got, data['state'], P)
            mm = np.clip(data['messages'], -8, 8)
            stats[label]['edge_projection_relative_l2'] = {
                k:float(np.linalg.norm(mm[:,message_indices]@edge_new[k]-mm@edge[k]) / max(np.linalg.norm(mm@edge[k]),1e-30)) for k in edge}
        detail = dict(kind='whole_message_width_distillation', message_width=width,
                      retained_by_power=[k.tolist() for k in keep], teacher_errors=stats,
                      bill_pct_vs_best337393=None,
                      bill_note='Must price natively: current W3rank4 and GOP2 rank2 already active; no six-dense-products saving claim.',
                      qualification='NOT NATIVE TESTED; retained GOP2 subspaces change and require full propagation validation')
        candidates.append(export(a.out, 'msg%d_teacher' % width, parents, q, extra, code, detail))
    report = dict(utc=utc(), status='IMMUTABLE_TARGET_FREE_CANDIDATES_NO_NETWORK_SCORE',
                  train=TRAIN, valid=VALID, capture_metadata=meta, capture_files=paths,
                  candidates=candidates, reference_submission=337393)
    jwrite(a.out/'CANDIDATES_v1.json', report)
    print(json.dumps(dict(status=report['status'], manifest=str(a.out/'CANDIDATES_v1.json'),
          candidates=[dict(name=c['name'], science=c['science'], release=c['release'],
                          valid_teacher_error=c['teacher_errors']['valid']['local_head_delta_rms'])
                      for c in candidates]), indent=2))


if __name__ == '__main__':
    main()
