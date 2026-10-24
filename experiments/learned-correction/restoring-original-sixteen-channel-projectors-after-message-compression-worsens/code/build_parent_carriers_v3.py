"""Preserve original message-projector inputs while retaining a narrowed consumer.

No fit or dial. Original teacher constants and retained channels are verified
against the original frozen distillation manifest and both actual sources.
"""
import fit_frontier_v1 as f
import argparse
import ast
import json
from pathlib import Path
import numpy as np

PAIRS = {
    '3edbd193c74d7358f2a9a81da7b11ff0d45af204cb36cddb33b28e7ba3d1127a':
        ('ba7ac099cbb791438cd19661f8045dd2d6e6fdee747f5af9b2102797b4166980',
         '777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510', 8, 4, 'msg4_current_selfteacher'),
    '777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510':
        ('e41095e464be4f5973cae13252a716bc1d85a56bd3ba9d0837e28509c3d08f05',
         '09fb70c86fa245d083f1db5d9c06dcde90cd71cbc8aaeaabb1bcb134aef59ded', 16, 8, 'msg8_teacher'),
    'c92dfc72a436aa9d47ceccdc9476371d78bc2d42c7ebc62d144a859bf14c5967':
        ('71ed87404309ea5e559f30bf08f99cd30bdfd508bdaf56cda2fd9030dadfdecf',
         '777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510', 8, 4, 'msg4_current_selfteacher'),
}

OPERATORS = '''
def _vmc_parent_parameters():
    global _VMC_PARENT_PARAMETERS
    if _VMC_PARENT_PARAMETERS is None:
        p = _unpack_static_fp32(_VMC_PARENT_LITERAL)
        p['msg_wT'] = p['msg_w'].T
        p['keep'] = [fnp.asarray(indices, dtype=fnp.int64) for indices in _VMC_KEEP]
        _VMC_PARENT_PARAMETERS = p
    return _VMC_PARENT_PARAMETERS

def _vmc_parent_buffers(k):
    got = k.get('_vmc_parent_buffers')
    if got is None:
        n = int(k['H'].shape[0])
        source = fnp.empty((n, 3 * _VMC_PARENT_WIDTH), dtype=fnp.float32)
        got = dict(SRC=source, SRC3=source.reshape(n, 3, _VMC_PARENT_WIDTH).transpose(1, 0, 2),
                   SQ=fnp.empty((3, n, _VMC_PARENT_WIDTH), dtype=fnp.float32))
        k['_vmc_parent_buffers'] = got
    return got

def _vmc_parent_metric():
    global _VMC_PARENT_METRIC
    if _VMC_PARENT_METRIC is None:
        E = fnp.asarray(_VMC_PARENT_ENCODER, dtype=fnp.float64)
        values, vectors = fnp.linalg.eigh(fnp.matmul(E.T, E))
        scales = fnp.sqrt(fnp.maximum(values, 1e-30))
        root = fnp.matmul(vectors * scales[None, :], vectors.T)
        inverse = fnp.matmul(vectors / scales[None, :], vectors.T)
        _VMC_PARENT_METRIC = (root, inverse)
    return _VMC_PARENT_METRIC

def _vmc_parent_w3(T, X):
    root, inverse = _vmc_parent_metric()
    wide = X.astype(fnp.float64)
    gram = fnp.matmul(wide.T, wide)
    weighted = fnp.matmul(fnp.matmul(root, gram), root)
    _, vectors = fnp.linalg.eigh(weighted)
    V = vectors[:, -4:]
    L = fnp.matmul(root, V).astype(fnp.float32)
    B = fnp.matmul(V.T, inverse).astype(fnp.float32)
    coefficients = fnp.matmul(X, L)
    projected = fnp.matmul(T, coefficients)
    _W3R_STATS['calls'] += 1
    return fnp.matmul(projected, B)

def _gop2_messages(k, W, state):
    got = _E2_MSG.get('v')
    if got is not None and got[0] is state and got[1] is W:
        return got[2]
    p = _vmc_parent_parameters()
    carrier = _vmc_parent_buffers(k)
    fnp.add(state @ p['msg_wT'], p['msg_b'], out=carrier['SRC'])
    for power in range(3):
        fnp.copyto(k['SRC3'][power], fnp.take(carrier['SRC3'][power], p['keep'][power], axis=1))
    T = _e2_wpow(W)['T013']
    n = int(state.shape[0])
    M = fnp.empty((3, n, _VMFR_CM), dtype=fnp.float32)
    fnp.matmul(T[0:1], k['SRC3'][0:1], out=M[0:1])
    parent_w2 = _gop2_contract(T[1:2], carrier['SRC3'][1:2])
    fnp.copyto(M[1:2], fnp.take(parent_w2, p['keep'][1], axis=2))
    parent_w3 = _vmc_parent_w3(T[2], carrier['SRC3'][2])
    fnp.copyto(M[2], fnp.take(parent_w3, p['keep'][2], axis=1))
    _GOP2_STATS['message_misses'] += 1
    _E2_MSG['v'] = (state, W, M)
    return M

def _vmc_parent_energy(k, W):
    p = _vmc_parent_parameters()
    carrier = _vmc_parent_buffers(k)
    fnp.power(carrier['SRC3'], 2, out=carrier['SQ'])
    parent = _gop2_contract(_e2_wpow(W, need6=True)['T135'], carrier['SQ'])
    selected = fnp.empty((3, int(k['H'].shape[0]), _VMFR_CM), dtype=fnp.float32)
    for power in range(3):
        fnp.copyto(selected[power], fnp.take(parent[power], p['keep'][power], axis=1))
    return selected
'''


def main():
    parser = argparse.ArgumentParser()
    for key in ('fit-manifest', 'base-science', 'base-release', 'base-tail', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    science_sha = f.sha(args.base_science)
    release_sha, teacher_sha, parent_width, width, fit_name = PAIRS[science_sha]
    assert f.sha(args.base_release) == release_sha
    report = json.loads(args.fit_manifest.read_text())
    spec = next(row for row in report['candidates'] if row['name'] == fit_name)
    metadata = report['capture_metadata']
    assert metadata['source_sha256'] == teacher_sha
    teacher_source = Path(metadata['source'])
    assert f.sha(teacher_source) == teacher_sha
    teacher_literal, teacher_functions = f.literals_and_functions(teacher_source.read_text())
    original = f.decode(metadata['gnn_literal'])
    source_original = f.decode(teacher_literal['_GNN_LITERAL'])
    assert all(np.array_equal(original[key], source_original[key]) for key in original)
    base_literal, functions = f.literals_and_functions(args.base_science.read_text())
    current = f.decode(base_literal['_GNN_LITERAL'])
    assert original['msg_w'].shape == (3 * parent_width, 64)
    assert current['msg_w'].shape == (3 * width, 64)
    keep = spec['retained_by_power']
    assert len(keep) == 3 and all(len(indices) == width for indices in keep)
    selected = np.concatenate([parent_width * power + np.asarray(keep[power]) for power in range(3)])
    assert np.array_equal(current['msg_w'], original['msg_w'][selected])
    assert np.array_equal(current['msg_b'], original['msg_b'][selected])
    encoder = None
    for node in ast.parse(teacher_source.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == '_VMFR_ORIGINAL_W3_ENCODER':
                encoder = np.asarray(ast.literal_eval(node.value), dtype='f4')
    if parent_width == 16:
        assert encoder is None and "_GNN_PARAMS['enc0_w'][:, 62:78]" in teacher_functions['_w3r_metric']
        encoder = original['enc0_w'][:, 62:78]
    assert encoder.shape == (128, parent_width)
    step = functions['_gop2_gnn_step']
    lines = step.splitlines()
    power_lines = [line for line in lines if "fnp.power(k['SRC3'], 2, out=k['SQ'])" in line]
    energy_lines = [line for line in lines if line.startswith('    E = _gop2_contract(')]
    assert len(power_lines) == len(energy_lines) == 1
    step = step.replace(power_lines[0] + '\n', '').replace(energy_lines[0], '    E = _vmc_parent_energy(k, W)')
    for term in ("k = _e2_gnn_consts(P,", "M = _e2_messages(k, W, state)", ("_vmnl['enc0_wT']" if "_vmfr_nonlinear_parameters(l)" in step else "k['enc0_wT']")):
        assert term in step
    tail = '\n# Preserve original projector carriers; retained message consumers remain narrow.\n'
    tail += 'assert _GNN_PARAMS is None and not _E2_GNN_STATE and not _E2_MSG\n'
    tail += 'assert _O20_FLATEXTRA_ON is False and _VMFR_CM == ' + str(width) + '\n'
    tail += '_VMC_PARENT_WIDTH = ' + str(parent_width) + '\n_VMC_KEEP = ' + repr(keep) + '\n'
    tail += '_VMC_PARENT_LITERAL = ' + repr(f.encode({key: original[key] for key in ('msg_w', 'msg_b')})) + '\n'
    tail += '_VMC_PARENT_ENCODER = ' + repr(encoder.tolist()) + '\n'
    tail += '_VMC_PARENT_PARAMETERS = None\n_VMC_PARENT_METRIC = None\n'
    tail += OPERATORS + '\n' + step + '\n_FAST_RESCUE_FRAC = {}\n'
    args.out.mkdir(parents=True, exist_ok=False)
    name = ('msg4_firstaffine_parent8_carriers' if science_sha.startswith('3edbd193') else 'msg%d_parent%d_carriers' % (width, parent_width))
    append = args.out / ('tail_' + name + '_v1.py')
    fixture = args.out / ('composed_tail_' + name + '_v1.py')
    f.write(append, tail)
    f.write(fixture, args.base_tail.read_text() + tail)
    candidate = dict(name=name, message_width=width, parent_carrier_width=parent_width,
        retained_by_power=keep, teacher_source_sha256=teacher_sha,
        tail=str(fixture), tail_sha256=f.sha(fixture), append_tail=str(append), append_tail_sha256=f.sha(append),
        original_fit_manifest_sha256=f.sha(args.fit_manifest), no_new_fit=True,
        private_buffers_in_existing_k=True, current_GNN_parameters_unchanged=True,
        current_context_heads_physics_unchanged=True, no_global_width_mutation=True)
    for role, parent in [('science', args.base_science), ('release', args.base_release)]:
        target = args.out / ('candidate_' + name + ('_sci_v1.py' if role == 'science' else '_rel_v1.py'))
        source = parent.read_text() + tail
        compile(source, str(target), 'exec')
        f.write(target, source)
        candidate[role] = dict(path=str(target), sha256=f.sha(target), parent_path=str(parent), parent_sha256=f.sha(parent))
    f.jwrite(args.out / 'CANDIDATES_v1.json', dict(utc=f.utc(), candidates=[candidate], reference_submission=337422,
        status='IMMUTABLE_ORIGINAL_CARRIER_SOURCE_NO_NATIVE_SCORE', builder_sha256=f.sha(__file__),
        original_teacher_source=str(teacher_source), no_score_or_bill_gain_claim=True))
    print(json.dumps(candidate, indent=2))


if __name__ == '__main__':
    main()
