"""Zero-extra-FLOP teacher head repair for compressed GNN states.

Fixed ridge1e-4, train4 only, teacher-validation3. Targets are the original
teacher's existing per-layer head outputs, never network truth. Original
teacher sigma/rms/gain are used only to report local output error.
"""
import fit_frontier_v1 as f
import argparse
import ast
import json
from pathlib import Path
import numpy as np

EXPECTED_HELPER_SHA = 'b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363'
NAMES = ('enc96_msg8_teacher', 'enc64_msg8_teacher')


def states(data, original, student, spec):
    keep1 = np.asarray(spec.get('retained_first', list(range(128))), dtype=int)
    if 'retained_by_power' in spec:
        keep = [np.asarray(v, dtype=int) for v in spec['retained_by_power']]
        inputs = np.concatenate([np.arange(30)] + [30+16*g+keep[g%3] for g in range(6)])
    else:
        inputs = np.arange(126)
    bias = data['bias'][:, keep1] + student['enc0_b'] - original['enc0_b'][keep1]
    h = f.silu(data['H'][:, inputs] @ student['enc0_w'].T + bias)
    if spec.get('nonlinear_layers', 2) == 2:
        h = f.silu(h @ student['enc2_w'].T + student['enc2_b'])
    return h @ student['enc4_w'].T + student['enc4_b']


def head_error(rows, state, teacher, Q, P):
    errors = []
    by_layer = []
    for index, rec in enumerate(rows):
        sl = slice(128*index, 128*(index+1))
        layer = rec['layer']
        target = teacher[sl] @ P['head'][layer] + P['head_b'][layer]
        got = state[sl] @ Q['head'][layer] + Q['head_b'][layer]
        scale = f.array(rec, 'sigma') * P['rms'][layer] * (1+.25*np.tanh(P['gain'][layer]))
        delta = (got-target).astype(np.float64)*scale.astype(np.float64)
        errors.append(delta)
        by_layer.append(dict(rank=rec['rank'], layer=layer, delta_rms=float(np.sqrt(np.mean(delta*delta)))))
    return dict(local_head_delta_rms=float(np.sqrt(np.mean(np.concatenate(errors)**2))),
                local_head_delta_by_rank_layer=by_layer,
                limitation='Frozen teacher-input head repair error, not propagated estimator MSE or adjusted gain.')


def main():
    ap = argparse.ArgumentParser()
    for key in ('captures', 'input-manifest', 'base-science', 'base-release', 'out'):
        ap.add_argument('--'+key, type=Path, required=True)
    a = ap.parse_args()
    assert f.sha(f.__file__) == EXPECTED_HELPER_SHA
    a.out.mkdir(parents=True, exist_ok=False)
    f.jwrite(a.out/'PREREG_v1.json', dict(utc=f.utc(), train=f.TRAIN, valid=f.VALID,
              candidate_names=NAMES, ridge=f.RIDGE, source_sha256=f.sha(__file__),
              helper_sha256=EXPECTED_HELPER_SHA, target_free=True,
              output_target='Original teacher state times original learned head plus head bias',
              reference_submission=337393, extra_runtime_flops=0))
    manifest = json.loads(a.input_manifest.read_text())
    source_specs = {c['name']:c for c in manifest['candidates']}
    assert all(name in source_specs for name in NAMES)
    rows, paths, meta = f.collect(a.captures)
    P = f.decode(meta['gnn_literal'])
    base_literals, _ = f.literals_and_functions(a.base_science.read_text())
    base_release_literals, _ = f.literals_and_functions(a.base_release.read_text())
    assert base_literals['_GNN_LITERAL'] == base_release_literals['_GNN_LITERAL']
    baseP = f.decode(base_literals['_GNN_LITERAL'])
    for key in P:
        assert np.array_equal(P[key], baseP[key]), ('teacher/source mismatch', key)
    train_rows = [r for r in rows if r['rank'] in f.TRAIN]
    valid_rows = [r for r in rows if r['rank'] in f.VALID]
    train = f.matrices(train_rows, P)
    valid = f.matrices(valid_rows, P)
    layer_index = np.concatenate([np.full(128, r['layer'], dtype=int) for r in train_rows])
    candidates = []
    for name in NAMES:
        spec = source_specs[name]
        assert f.sha(a.base_science) == spec['science']['parent_sha256']
        assert f.sha(a.base_release) == spec['release']['parent_sha256']
        old_tail_path = Path(spec['tail'])
        assert f.sha(old_tail_path) == spec['tail_sha256']
        old_tail = old_tail_path.read_text()
        tail_literals, _ = f.literals_and_functions(old_tail)
        Q = f.decode(tail_literals['_GNN_LITERAL'])
        original_student = {k:v.copy() for k,v in Q.items()}
        train_state = states(train, P, Q, spec)
        valid_state = states(valid, P, Q, spec)
        for layer in range(1, 16):
            selected = layer_index == layer
            assert int(selected.sum()) == 512
            target = train['state'][selected] @ P['head'][layer] + P['head_b'][layer]
            coef, intercept = f.ridge_affine(train_state[selected], target[:, None])
            Q['head'][layer] = coef[:, 0]
            Q['head_b'][layer] = intercept[0]
        assert all(np.array_equal(Q[k], original_student[k]) for k in Q if k not in ('head', 'head_b'))
        assert all(np.isfinite(v).all() for v in Q.values())
        stats = {}
        for label, rr, data, state in [('train', train_rows, train, train_state), ('valid', valid_rows, valid, valid_state)]:
            stats[label] = dict(before=head_error(rr, state, data['state'], original_student, P),
                                after=head_error(rr, state, data['state'], Q, P))
        tree = ast.parse(old_tail)
        assignments = [n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets)==1
                       and isinstance(n.targets[0], ast.Name) and n.targets[0].id=='_GNN_LITERAL']
        assert len(assignments) == 1
        node = assignments[0]
        lines = old_tail.splitlines()
        lines[node.lineno-1:node.end_lineno] = ['_GNN_LITERAL = '+repr(f.encode(Q))]
        tail = '\n'.join(lines)+'\n_VMFR_HEAD_TEACHER = True\n_FAST_RESCUE_FRAC = {}\n'
        newname = name+'_headrepair'
        tailpath = a.out/('tail_'+newname+'_v1.py')
        f.write(tailpath, tail)
        record = {k:v for k,v in spec.items() if k not in ('science','release','tail','tail_sha256','name','teacher_errors')}
        record.update(name=newname, kind='zero_extra_flop_teacher_head_repair', original_student_name=name,
                      original_student_science_sha256=spec['science']['sha256'], tail=str(tailpath),
                      tail_sha256=f.sha(tailpath), head_repair_errors=stats, extra_runtime_flops=0,
                      head_weight_norm_before=np.linalg.norm(original_student['head'],axis=1).tolist(),
                      head_weight_norm_after=np.linalg.norm(Q['head'],axis=1).tolist(),
                      qualification='NOT NATIVE TESTED; teacher-input repair may not transfer to propagated student states')
        for role, parent in [('science',a.base_science),('release',a.base_release)]:
            source = parent.read_text()+tail
            target = a.out/('candidate_'+newname+('_sci_v1.py' if role=='science' else '_rel_v1.py'))
            compile(source,str(target),'exec');f.write(target,source)
            record[role] = dict(path=str(target),sha256=f.sha(target),parent_sha256=f.sha(parent))
        candidates.append(record)
    report = dict(utc=f.utc(), status='IMMUTABLE_TEACHER_HEAD_REPAIR_NO_NETWORK_SCORE',
                  train=f.TRAIN, valid=f.VALID, source_manifest_sha256=f.sha(a.input_manifest),
                  capture_files=paths, candidates=candidates, reference_submission=337393)
    f.jwrite(a.out/'CANDIDATES_v1.json', report)
    print(json.dumps(dict(status=report['status'],manifest=str(a.out/'CANDIDATES_v1.json'),
          candidates=[dict(name=c['name'],science=c['science'],release=c['release'],
                          valid_before=c['head_repair_errors']['valid']['before']['local_head_delta_rms'],
                          valid_after=c['head_repair_errors']['valid']['after']['local_head_delta_rms']) for c in candidates]),indent=2))


if __name__ == '__main__':
    main()
