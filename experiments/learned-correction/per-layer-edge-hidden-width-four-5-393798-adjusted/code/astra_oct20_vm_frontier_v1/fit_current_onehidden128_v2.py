"""One predeclared shared one-hidden128 student of actual777a native states.

Preserves all78 inputs, all128 first nonlinear units, message/projector paths,
heads and birth consumers. No network truth is opened. The second SiLU layer
is removed only if the frozen reconstruction gate passes.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import fit_frontier_v1 as f
import fit_current_msg8_messages_v2 as capture

SCI = '777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
REL = 'e41095e464be4f5973cae13252a716bc1d85a56bd3ba9d0837e28509c3d08f05'
FIELDS = ('state_mse', 'head_delta_mse', 'message_mse')


def measures(rows, original, prediction, parameters):
    error = prediction.astype('f8') - original.astype('f8')
    physical = []
    for i, row in enumerate(rows):
        layer = row['layer']
        scale = f.array(row, 'sigma').astype('f8') * float(parameters['rms'][layer])
        scale *= 1 + .25 * np.tanh(float(parameters['gain'][layer]))
        physical.append((error[128*i:128*(i+1)] @ parameters['head'][layer].astype('f8')) * scale)
    messages = error @ parameters['msg_w'].astype('f8').T
    return dict(state_mse=float(np.mean(error**2)),
                head_delta_mse=float(np.mean(np.concatenate(physical)**2)),
                message_mse=float(np.mean(messages**2)),
                state_relative_l2=float(np.linalg.norm(error)/np.linalg.norm(original)))


def main():
    ap = argparse.ArgumentParser()
    for key in ('captures', 'input-manifest', 'fidelity-reference', 'out'):
        ap.add_argument('--'+key, type=Path, required=True)
    args = ap.parse_args()
    assert f.sha(f.__file__) == 'b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363'
    assert f.sha(capture.__file__) == '56d3c63e1d766e6b1870e6aa07b88b84c02172d71edf28a6ad6a1724ad3f53ea'
    reference = json.loads(args.fidelity_reference.read_text())
    assert reference['candidate_science_sha256'] == '96b0e972f962500b5b2240afbb5623ee4ea25d2b5dcf8a103ddebefea1c84089'
    assert reference['teacher_source_sha256'] == SCI and reference['train'] == list(f.TRAIN) and reference['valid'] == list(f.VALID)
    args.out.mkdir(parents=True, exist_ok=False)
    f.jwrite(args.out/'PREREG_v1.json', dict(utc=f.utc(), source_sha256=f.sha(__file__),
        train=f.TRAIN, valid=f.VALID, ridge=f.RIDGE, first_hidden=128, message_width=8,
        fitting='One shared128to64 affine state map from unchanged native first activations',
        fidelity_gate='Aggregate valid state/head/message MSE each no greater than already frozen actual-current rank16 student',
        fidelity_reference_sha256=f.sha(args.fidelity_reference), no_network_truth=True,
        no_validation_tuning=True, no_hypergrid=True, reference_submission=337422))
    manifest = json.loads(args.input_manifest.read_text())
    spec = next(x for x in manifest['candidates'] if x['name'] == 'msg8_teacher_loaderfix')
    parents = {role: Path(spec[role]['path']) for role in ('science', 'release')}
    assert f.sha(parents['science']) == SCI and f.sha(parents['release']) == REL
    literals, functions = f.literals_and_functions(parents['science'].read_text())
    rl, rf = f.literals_and_functions(parents['release'].read_text())
    P = f.decode(literals['_GNN_LITERAL'])
    assert literals['_GNN_LITERAL'] == rl['_GNN_LITERAL']
    assert functions['_gop2_gnn_step'] == rf['_gop2_gnn_step']
    rows, paths, metadata = capture.collect(args.captures)
    captured = f.decode(metadata['gnn_literal'])
    for key in P: assert np.array_equal(P[key], captured[key]), key
    train_rows = [r for r in rows if r['rank'] in f.TRAIN]
    valid_rows = [r for r in rows if r['rank'] in f.VALID]
    train = capture.matrices(train_rows, P); valid = capture.matrices(valid_rows, P)
    assert train['h1'].shape == (7680, 128) and valid['h1'].shape == (5760, 128)
    w4, b4 = f.ridge_affine(train['h1'], train['state'])
    metrics = {}
    for label, rr, data in [('train', train_rows, train), ('valid', valid_rows, valid)]:
        got = data['h1'] @ w4 + b4
        metrics[label] = measures(rr, data['state'], got, P)
    ranks = np.concatenate([np.full(128, r['rank']) for r in train_rows])
    loo = []
    for held in f.TRAIN:
        mask = ranks != held
        cw, cb = f.ridge_affine(train['h1'][mask], train['state'][mask])
        rr = [r for r in train_rows if r['rank'] == held]
        loo.append(dict(rank=held, **measures(rr, train['state'][~mask], train['h1'][~mask]@cw+cb, P)))
    ratios = {key: metrics['valid'][key]/reference['valid_mse'][key] for key in FIELDS}
    passed = all(value <= 1 for value in ratios.values())
    result = dict(utc=f.utc(), status='ONEHIDDEN128_CURRENT_BANK_FIXED_RIDGE_NO_NATIVE_SCORE',
        metrics=metrics, leave_one_network_out=loo, valid_error_ratios_to_frozen_rank16=ratios,
        fixed_fidelity_gate_pass=passed, source_sha256=f.sha(__file__),
        capture_files=paths, capture_metadata=metadata, parent_spec=spec,
        replay_checks={label:{k:v for k,v in data.items() if k.startswith('replay')} for label,data in [('train',train),('valid',valid)]},
        fitted_state_affine=f.encode(dict(enc4_w=w4.T,enc4_b=b4)), reference_submission=337422)
    f.jwrite(args.out/'FIT_COMPLETE_v1.json', result)
    if not passed:
        print(json.dumps(dict(status='REJECTED_FIDELITY_BEFORE_NATIVE', metrics=metrics,
            valid_error_ratios_to_frozen_rank16=ratios)), flush=True)
        return
    Q = {k:v.copy() for k,v in P.items()}; Q.update(enc4_w=w4.T, enc4_b=b4)
    lines = functions['_gop2_gnn_step'].splitlines()
    removed = [line for line in lines if 'h = silu(' in line and "k['enc2_wT']" in line]
    assert len(removed) == 1
    step = '\n'.join(line for line in lines if line not in removed)
    assert "k['enc2_wT']" not in step
    name = 'msg8_current_onehidden128'
    candidate = f.export(args.out, name, parents, Q,
        '_VMFR_ONE_HIDDEN = True\nassert _VMFR_CM == 8 and _O20_FLATEXTRA_ON is False\n',
        step+'\n', dict(kind='current_bank_shared_onehidden128', message_width=8,
            first_hidden=128, state_width=64, second_hidden_removed=True,
            unchanged_projectors=True, teacher_fit_metrics=metrics, fidelity_gate_pass=True,
            actual_current_teacher=True, no_network_truth=True))
    append = Path(candidate['tail']); fixture = args.out/('composed_fixture_'+name+'_v1.py')
    oldtail = Path(spec['tail']); assert f.sha(oldtail) == spec['tail_sha256']
    f.write(fixture, oldtail.read_text()+append.read_text())
    candidate.update(append_tail=str(append),append_tail_sha256=candidate['tail_sha256'],
        tail=str(fixture),tail_sha256=f.sha(fixture), fixture_not_deployment=True)
    f.jwrite(args.out/'CANDIDATES_v1.json',dict(utc=f.utc(),candidates=[candidate],reference_submission=337422,
        fit_report_sha256=f.sha(args.out/'FIT_COMPLETE_v1.json')))
    f.jwrite(args.out/'TINY_WITH_EXACT_PARENT_v1.json',dict(candidates=[spec,candidate],reference_submission=337422))
    print(json.dumps(dict(status='FROZEN_FIDELITY_PASS_EXPORTED_NO_NATIVE',science=candidate['science'],
        valid_error_ratios_to_frozen_rank16=ratios)),flush=True)


if __name__ == '__main__':
    main()
