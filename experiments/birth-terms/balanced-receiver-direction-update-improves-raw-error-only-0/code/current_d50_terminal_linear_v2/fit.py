"""One fixed terminal-P65 ridge solve. Inert until explicit sealed admission.

Offline TRAIN data only. This never imports an estimator, Torch or flopscope.
Generated candidate source differs only in the existing final head literals.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
P2 = NEW.parent
BANK = NEW/'birth_direction_fit_v1/build_terminal_v1/estimator.py'
BANK_SHA = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'
PAYLOAD = NEW/'payload_response_rows_v1.npz'
PAYLOAD_SHA = '6b55c0d6071b2350e53e243bcd4137651e2c80a2cd55d04467bf194d0ea54a1c'
DATA = NEW/'joint_prefix_final_data_v1.py'
DATA_SHA = '22fff8465a13e412c68c1e5c204d7bfa0ce3d7c1cb9b99e1e5378098a9fd8b6b'
TARGET_INDEX = NEW/'joint_prefix_final_fit_v1/manifest.json'
TARGET_INDEX_SHA = 'd61757d28236e3d35d87415962d1a57d18ff38ae189651df52df5e3412b9a971'
CAP_ROOT = NEW/'strategic_reset_20260909_v1/cap_native_objective_v1'
CAP_RESULT = CAP_ROOT/'analysis_v2/result.json'
CAP_RESULT_SHA = '92853467117de93914097ac316670b771367dc3e29ea9e947a9d87e9180506bb'
CAP_WATCH_SHA = '9f4c7bb707ccecaf3946f3667f9e28fe68759f6e01e1543b396033b4bc5fa329'
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
STORE_SHA = '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50'
RIDGE = 1e-4
BUDGET = 2**41


def require(test, message):
    if not test:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def assignment(source, name):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    require(len(nodes) == 1 and len(nodes[0].targets) == 1, 'Unique literal '+name)
    node = nodes[0]
    lines = source.splitlines(keepends=True)
    begin = sum(map(len, lines[:node.lineno-1]))+node.col_offset
    end = sum(map(len, lines[:node.end_lineno-1]))+node.end_col_offset
    return begin, end, ast.literal_eval(node.value)


def run(args, out, result):
    pins, targets = {}, {}
    result.update(source_bindings=pins, target_bindings=targets)

    def pin(path, expected=None):
        path = Path(path).resolve()
        require('truthpost' not in path.parts and not
                (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()),
                'No truth-array hash/read through source or capture inventories')
        digest = pins.get(str(path))
        if digest is None:
            digest = sha(path)
            pins[str(path)] = digest
        require(expected is None or digest == expected, 'Pin mismatch '+str(path))
        return digest

    def bind(mapping):
        require(isinstance(mapping, dict), 'Concrete binding inventory')
        for path, item in mapping.items():
            if isinstance(item, dict):
                require(Path(path).resolve() == Path(item['path']).resolve(), 'Binding path owns its metadata')
                pin(item['path'], item['sha256'])
            else:
                pin(path, item)

    def completed(path, digest=None):
        path = Path(path).resolve()
        value = pin(path, digest)
        end_path = path.parent/'completion.json'
        pin(end_path)
        record, end = read(path), read(end_path)
        require(record['passed'] is True and record['completed'] is True
                and record['hashes_unchanged'] is True and end['passed'] is True
                and end['completed'] is True and end['result_sha256'] == value,
                'Completed PASS and exact completion seal: '+str(path))
        return record

    def metrics(record):
        require(type(record['flops']) is int and 0 < record['flops'] <= BUDGET
                and 0 <= record['operation_count'] and record['wall_s'] < 120
                and 0 < record['peak_rss_bytes'] < 8_000_000_000
                and all(type(record[k]) in (float,int) and math.isfinite(record[k]) and record[k] >= 0
                        for k in ('wall_s','context_wall_s','flopscope_backend_s',
                                  'flopscope_overhead_s','native_residual_s')),
                'Actual complete native metric/resource gates')

    def reset(record, name, bill_name, bill, folder, rank):
        meta, bill_meta = record[name+'_receipt'], record[bill_name]
        pin(meta['path'], meta['sha256'])
        pin(bill_meta['path'], bill_meta['sha256'])
        require(Path(meta['path']).resolve() == folder/(name+'.json')
                and Path(bill_meta['path']).resolve() == folder/(bill_name+'.json')
                and read(bill_meta['path']) == bill, 'Exact saved bill/reset files')
        value = read(meta['path'])
        require(value == record[name] and value['call'] == 0 and value['rank'] == rank
                and Path(value['saved_bill']).resolve() == Path(bill_meta['path']).resolve()
                and value['saved_bill_sha256'] == bill_meta['sha256']
                and value['reset_api'] == 'flopscope.budget_reset' and value['after_context_exit']
                and value['candidate_state_reset'] is False and value['complete']
                and value['context_released'] and value['closed_context_bill_unchanged_by_reset'],
                'Public harness-only reset after the saved closed bill')
        require(value['closed_metrics'] == dict(flops=bill['flops'],operation_count=bill['operation_count'],
                context_wall_s=bill['context_wall_s'],backend_s=bill['flopscope_backend_s'],
                overhead_s=bill['flopscope_overhead_s'],residual_s=bill['native_residual_s']),
                'Closed complete bill unchanged across reset')
        return datetime.fromisoformat(value['finished_utc'])

    pin(__file__, args.source_sha256)
    pin(HERE/'PROTOCOL.md', args.protocol_sha256)
    pin(BANK, BANK_SHA)
    pin(PAYLOAD, PAYLOAD_SHA)
    pin(DATA, DATA_SHA)
    pin(TARGET_INDEX, TARGET_INDEX_SHA)
    require(args.cap_result_sha256 == CAP_RESULT_SHA, 'Fixed completed CAP v2 outcome')
    cap = completed(CAP_RESULT, CAP_RESULT_SHA)
    require(cap['integrity_passed'] and
            not cap['advances_to_independent14'], 'Completed non-advancing CAP audit')
    # The already-scored CAP result contains VALID target bindings. Authenticate
    # only its pinned result/seal and independent watch, never traverse targets.
    cap_watch = completed(CAP_ROOT/'watch_v2/result.json', CAP_WATCH_SHA)
    bind(cap_watch['source_bindings']); bind(cap_watch['artifacts'])
    require(cap_watch['child_result_sha256'] == CAP_RESULT_SHA
            and cap_watch['child_completion_sha256'] == pin(CAP_RESULT.parent/'completion.json'),
            'Successful CAP offline supervisor binds the same completed result')
    valid = tuple(assignment(DATA.read_bytes(), 'VALID')[2])
    train = tuple(r for r in range(35, 235) if r not in valid)
    require(len(train) == 160 and len(valid) == 40, 'Fixed split')

    manifest_path = HERE/'capture_train_v1/manifest.json'
    pin(manifest_path, args.capture_sha256)
    manifest = read(manifest_path)
    require(manifest['completed'] and manifest['target_accessed'] is False and
            manifest['train_ranks'] == list(train) and
            manifest['candidate_sha256'] == BANK_SHA, 'Sealed full native TRAIN capture')
    require(set(manifest['captures']) == {str(r) for r in train}, 'No selected TRAIN rows')
    bind(manifest['source_bindings'])
    for path in (HERE/'native.py', HERE/'common.py', HERE/'PROTOCOL.md'):
        require(str(path) in manifest['source_bindings'], 'Manifest binds exact capture source: '+str(path))
    require(manifest['source_bindings'][str(HERE/'PROTOCOL.md')] == args.protocol_sha256,
            'Same prospective fit/capture protocol')
    watch_path = Path(manifest['watch_result_path']).resolve()
    require(watch_path == HERE/'capture_train_v1/result.json'
            and Path(manifest['watch_completion_path']).resolve() == watch_path.parent/'completion.json',
            'Fixed full-capture supervisor result/completion')
    watch = completed(watch_path)
    bind(watch['source_bindings']); bind(watch['artifacts'])
    require(watch['artifacts'][str(manifest_path)] == args.capture_sha256
            and all(watch['source_bindings'].get(path) == digest
                    for path,digest in manifest['source_bindings'].items()),
            'Passed full-capture watch seals this exact manifest and source roster')
    qitem = manifest['qualification']
    qpath = Path(qitem['path']).resolve()
    require(qpath == HERE/'qualify_v1/result.json', 'Fixed initial35/100/repeat qualification')
    qualification = completed(qpath, qitem['sha256'])
    bind(qualification['source_bindings']); bind(qualification['artifacts'])
    require(all(qualification['source_bindings'].get(str(path)) == manifest['source_bindings'][str(path)]
                for path in (HERE/'native.py', HERE/'common.py', HERE/'PROTOCOL.md')),
            'Initial qualification and full capture use the same reviewed sources')
    require(isinstance(qualification['captures'], list)
            and [item['rank'] for item in qualification['captures']] == [35,100,35],
            'Exact ordered initial capture/repeat roster')
    pin(STORE/'manifest.json', STORE_SHA)
    weight_rows = read(STORE/'manifest.json')['roles']['observer_fit']['row_manifests']

    def capture(item, rank):
        path = Path(item['path']).resolve()
        require(HERE in path.parents and path.name == 'result.json', 'Owned capture result')
        record = completed(path, item['sha256'])
        folder = path.parent
        require(record['schema'] == 'current_d50_terminal_linear_capture_v1'
                and record['role'] == 'native' and record['sources_unchanged']
                and record['truth_accessed'] is False and record['MPS_used'] is False
                and record['parameter_updates'] == 0 and record['gradient_calls'] == 0
                and record['network_calls'] == 1 and record['training_admitted'] is False
                and record['score_measured'] is False and record['rank'] == rank
                and record['candidate_sha256'] == BANK_SHA,
                'Completed exact native capture')
        bind(record['source_bindings']); bind(record['bindings']); bind(record['artifacts'])
        require(all(record['source_bindings'].get(str(p)) == manifest['source_bindings'][str(p)]
                    for p in (HERE/'native.py', HERE/'common.py', HERE/'PROTOCOL.md')),
                'Every captured row uses the manifest-authorized native source')
        expected_shapes = dict(host=([1024],'float64'), hidden=([1024,64],'float32'),
            sigma=([1024],'float64'), delta=([1024],'float32'), raw=([16,1024],'float64'),
            served=([16,1024],'float32'))
        require(set(record['captures']) == set(expected_shapes), 'Exact terminal capture field set')
        for name,value in record['captures'].items():
            require(Path(value['path']).resolve() == folder/(name+'.npy')
                    and (value['shape'],value['dtype']) == expected_shapes[name]
                    and record['artifacts'][value['path']] == value['sha256'], 'Capture artifact ownership')
            pin(value['path'], value['sha256'])
        head = record['head_replay']
        require(all(head[k] is True for k in ('delta_byte_exact','final_served_byte_exact',
            'final_raw_byte_exact','widened_hook_delta_byte_exact','original_payload_arrays_byte_exact',
            'original_P_unchanged','captured_parameter_bytes_exact','offline_only',
            'native_bill_already_saved_and_context_reset')), 'Qualified original terminal head grouping')
        proof = record['ownership_proof']
        require(proof['paid_blocks_ast_identical'] and proof['original_numeric_prefix_unchanged']
                and proof['reset_order_ast_identical'], 'Preserved complete paid blocks and reset ordering')
        require(record['cold'] and record['all_primary_cold'] and record['includes_initialization']
                and record['own_complete_bill'] and not record['initialization_subtracted']
                and not record['candidate_source_edited'] and not record['input_preparation_in_participant_bill']
                and record['observed_predict_calls'] == 1 and record['observed_gnn_calls'] == 16
                and record['served_reconstruction_byte_exact'], 'One unmodified complete cold current card')
        bill, prep = record['complete_bill'], record['input_preparation']
        metrics(bill); metrics(prep)
        require(bill['cold'] and bill['includes_initialization'] and bill['call'] == 0
                and bill['rank'] == rank and bill['multiplier'] == max(.1,bill['flops']/BUDGET)
                and bill['observed_predict_calls'] == 1 and bill['capture_serialization_inside_context'],
                'Actual complete cold prediction bill')
        matches = [row for row in weight_rows if row['role_rank'] == rank]
        require(len(matches) == 1, 'Unique canonical captured network')
        rp = (STORE/matches[0]['path']).resolve()
        pin(rp,matches[0]['sha256'])
        row = read(rp); wp = rp.parent/'weights.npy'
        require(row['role_rank'] == rank and row['role'] == 'observer_fit'
                and Path(record['weights']['path']).resolve() == wp
                and record['weights']['mlp_seed'] == int(row['mlp_seed'])
                and record['weights']['sha256'] == row['files']['weights.npy']['sha256']
                and record['bindings'][str(wp)]['sha256'] == record['weights']['sha256'],
                'Current canonical rank/seed/weight identity')
        pin(wp,record['weights']['sha256'])
        require(prep['call'] == 0 and prep['rank'] == rank and prep['flops'] == 16*1024*1024
                and prep['layers'] == 16 and prep['context_closed'] and not prep['charged_to_participant']
                and prep['official_fnp_type'] and prep['exact_bytes'] and prep['independent_weight_buffers']
                and prep['weight_file_sha256'] == record['weights']['sha256'], 'Official separately billed inputs')
        first = reset(record,'input_lifecycle','input_bill',prep,folder,rank)
        last = reset(record,'lifecycle','bill',bill,folder,rank)
        require(first <= last, 'Official inputs precede participant prediction/reset')
        resources = record['resources']
        require(0 < resources['peak_rss_bytes'] < 8_000_000_000
                and 0 <= resources['elapsed_s'] < 120 and 0 <= record['elapsed_s'] < 120,
                'Complete cold capture resource qualification')
        diag = record['diagnostics']
        require(record['all16_domains_present'] and record['all47_selectors_present']
                and diag['variance_count'] == 16 and diag['selector_count'] == 47
                and diag['added_participant_array_operations'] == 0
                and set(diag['variance_domain']) == {str(l) for l in range(16)}
                and all(v['finite'] and math.isfinite(v['minimum']) and v['minimum'] >= 1e-10
                        and v['below_floor_count'] == 0 for v in diag['variance_domain'].values()),
                'All native unfloored finite prevariance domains')
        keys = {f'{l}:{slot}' for l in range(16) for slot in range(2 if l == 15 else 3)}
        require(set(diag['selection_cells']) == keys, 'All47 ordered selectors')
        for key,cell in diag['selection_cells'].items():
            ids = cell['ordered_ids']
            require(cell['floor'] == .01 and cell['count'] == len(ids) and len(ids) % 32 == 0
                    and len(ids) == len(set(ids)) and sorted(ids) == cell['membership']
                    and all(type(i) is int and 0 <= i < 1024 for i in ids), 'Selector identity/domain')
        require(all(diag['selection_cells'][f'0:{slot}']['ordered_ids'] == list(range(1024)) for slot in range(3)),
                'Original Gaussian initial support')
        if rank in (35,100):
            require(all(record['archived_cold_reference'].values()), 'Exact archived cold35/100 served/raw/bill/selector replay')
        return record

    qualified = [capture(item,item['rank']) for item in qualification['captures']]
    for item in qualification['captures']:
        path = Path(item['path']).resolve()
        require(qualification['artifacts'][str(path)] == item['sha256']
                and qualification['artifacts'][str(path.parent/'completion.json')] == pin(path.parent/'completion.json'),
                'Initial supervisor binds every qualification result/completion')
    require(qualified[0]['captures']['served']['raw_sha256'] == qualified[2]['captures']['served']['raw_sha256']
            and qualified[0]['captures']['raw']['raw_sha256'] == qualified[2]['captures']['raw']['raw_sha256']
            and qualified[0]['complete_bill']['flops'] == qualified[2]['complete_bill']['flops'],
            'Initial fresh-cold35 repeat bytes and complete bill')
    for index,rank in enumerate((35,100)):
        require(manifest['captures'][str(rank)]['path'] == qualification['captures'][index]['path']
                and manifest['captures'][str(rank)]['sha256'] == qualification['captures'][index]['sha256'],
                'Reuse the qualified primaries without replacement or repeat inclusion')
    captures = {}
    for rank in train:
        item = manifest['captures'][str(rank)]
        captures[rank] = capture(item,rank)
        path = Path(item['path']).resolve()
        require(watch['artifacts'][str(path)] == item['sha256']
                and watch['artifacts'][str(path.parent/'completion.json')] == pin(path.parent/'completion.json'),
                'Passed full supervisor binds every160 primary result/completion')
    require(not any(x in sys.modules for x in ('numpy', 'torch', 'flopscope')),
            'Source and capture verification precedes numerical imports')
    write(out/'captures_sealed_before_targets.json', dict(source_bindings=pins,
        capture_manifest_sha256=args.capture_sha256, train_ranks=list(train),
        qualification_sha256=qitem['sha256'], capture_watch_sha256=pin(watch_path),
        validation_targets_accessed=False, target_accessed=False))
    result['capture_seal_sha256'] = sha(out/'captures_sealed_before_targets.json')

    import numpy as np
    with np.load(PAYLOAD, allow_pickle=False) as f:
        p = {k: f[k].copy() for k in f.files}
    old_source = BANK.read_bytes()
    begin, end, literal = assignment(old_source, '_GNN_LITERAL')
    require(set(literal) == set(p), 'Original full P keys')
    require(all(np.asarray(literal[k], dtype=p[k].dtype).shape == p[k].shape and
                np.asarray(literal[k], dtype=p[k].dtype).tobytes() == p[k].tobytes()
                for k in p), 'Current bank original P literal bytes')
    require(p['head'].shape == (16,64) and p['head_b'].shape == (16,), 'Existing P65 block')
    theta0 = np.concatenate((p['head'][15], p['head_b'][15:16])).astype(np.float64)
    truth_index = read(TARGET_INDEX)
    expected = {x['rank']: x['target_sha256'] for x in truth_index['target_sources']}
    require(set(expected) == set(train), 'Existing TRAIN-only target index')
    for rank in train:
        path = P2/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{rank:04d}.npz'
        require(sha(path) == expected[rank], 'Pinned TRAIN truth')
        targets[str(path)] = expected[rank]
    write(out/'train_target_bindings.json', targets)
    result['training_targets_accessed'] = True

    def load(meta):
        a = np.load(meta['path'], allow_pickle=False)
        require(type(a) is np.ndarray and list(a.shape) == meta['shape'] and
                str(a.dtype) == meta['dtype'] and np.isfinite(a).all() and
                hashlib.sha256(a.tobytes()).hexdigest() == meta['raw_sha256'], 'Captured array bytes')
        return a

    xs, ds, hosts, weights, native_before = [], [], [], [], []
    for rank in train:
        record = captures[rank]
        a = {k: load(record['captures'][k]) for k in ('host','hidden','sigma','served')}
        require(a['host'].shape == (1024,) and a['hidden'].shape == (1024,64) and
                a['sigma'].shape == (1024,) and a['served'].shape == (16,1024), 'Terminal shapes')
        # The fit is explicitly real-affine. Deployment/replay keeps separate FP32 products.
        left = a['sigma'].astype(np.float32)*p['rms'][15]
        right = 1 + 0.25*np.tanh(p['gain'][15])
        scale = left.astype(np.float64)*np.float64(right)
        x = scale[:,None]*np.column_stack((a['hidden'].astype(np.float64), np.ones(1024)))
        truth = P2/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{rank:04d}.npz'
        with np.load(truth, allow_pickle=False) as f:
            target = f['mc_post'][-1].astype(np.float64)
        require(target.shape == (1024,) and np.isfinite(target).all(), 'TRAIN final means')
        d = target - a['host'] - x@theta0
        multiplier = max(.1, record['complete_bill']['flops']/BUDGET)
        xs.append(x); ds.append(d); hosts.append(a['host']); weights.append(multiplier)
        native_before.append(float(np.mean((a['served'][-1].astype(np.float64)-target)**2)))
    x, d = np.concatenate(xs), np.concatenate(ds)
    w = np.repeat(np.asarray(weights, dtype=np.float64), 1024)
    w /= np.sum(w)
    rms = np.maximum(np.sqrt(np.sum(w[:,None]*x*x, axis=0)), 1e-12)
    z = x/rms
    g = z.T@(w[:,None]*z)
    g = .5*(g+g.T) + RIDGE*np.eye(65)
    rhs = z.T@(w*d)
    eigenvalues = np.linalg.eigvalsh(g)
    require(np.isfinite(g).all() and np.isfinite(rhs).all() and eigenvalues[0] > 0,
            'Finite positive definite fixed ridge system')
    chol = np.linalg.cholesky(g)
    gamma = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))
    residual = float(np.linalg.norm(g@gamma-rhs)/max(np.linalg.norm(rhs), 1e-30))
    require(residual <= 1e-9 and np.isfinite(gamma).all(), 'Qualified fixed linear solve')
    theta64 = theta0 + gamma/rms
    theta = theta64.astype(np.float32)
    require(np.isfinite(theta).all(), 'Finite sole rounded terminal candidate')
    changed = {k: value.copy() for k, value in p.items()}
    changed['head'][15], changed['head_b'][15] = theta[:64], theta[64]
    for k in p:
        if k == 'head':
            require(changed[k][:15].tobytes() == p[k][:15].tobytes(), 'Frozen earlier heads')
        elif k == 'head_b':
            require(changed[k][:15].tobytes() == p[k][:15].tobytes(), 'Frozen earlier biases')
        else:
            require(changed[k].tobytes() == p[k].tobytes(), 'Frozen P '+k)
    replacement = ('_GNN_LITERAL = '+repr({k: changed[k].tolist() for k in literal})).encode()
    candidate = old_source[:begin]+replacement+old_source[end:]
    new_begin, new_end, new_literal = assignment(candidate, '_GNN_LITERAL')
    require(candidate[:new_begin]+old_source[begin:end]+candidate[new_end:] == old_source,
            'Every non-literal source byte unchanged')
    require(all(np.asarray(new_literal[k], dtype=p[k].dtype).tobytes() == changed[k].tobytes()
                for k in p), 'FP32 literal roundtrip')
    for path, digest in pins.items():
        require(sha(path) == digest, 'Input changed during fit '+path)
    for path, digest in targets.items():
        require(sha(path) == digest, 'TRAIN truth changed during fit')
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2e9, 'Offline fit memory limit')
    # Mechanical literal-only build; no source interpretation or estimator import.
    with (out/'estimator.py').open('xb') as f:
        f.write(candidate)
    with (out/'terminal_payload.npz').open('xb') as f:
        np.savez(f, **changed)
    with (out/'theta.npz').open('xb') as f:
        np.savez(f, theta0=theta0, theta64=theta64, theta=theta, rms=rms, gram=g, rhs=rhs)
    affine_after = x@(theta.astype(np.float64)-theta0)
    result.update(passed=True, completed=True, hashes_unchanged=True,
        source_bindings=pins, target_bindings=targets, ridge=RIDGE,
        parameters_changed=65, parameter_search=False, validation_targets_accessed=False,
        normal_equation_relative_residual=residual, gram_min_eigenvalue=float(eigenvalues[0]),
        gram_max_eigenvalue=float(eigenvalues[-1]), gram_condition=float(eigenvalues[-1]/eigenvalues[0]),
        normalized_change_norm=float(np.linalg.norm(gamma)),
        training_affine_weighted_mse_before=float(np.sum(w*d*d)),
        training_affine_weighted_mse_after=float(np.sum(w*(d-affine_after)**2)),
        training_native_mean_raw_before=float(np.mean(native_before)),
        training_native_mean_adjusted_before=float(np.mean(np.asarray(native_before)*weights)),
        training_native_candidate_measured=False, official_score=False, advances_to_independent14=False,
        candidate=str(out/'estimator.py'), candidate_sha256=sha(out/'estimator.py'),
        payload=str(out/'terminal_payload.npz'), payload_sha256=sha(out/'terminal_payload.npz'),
        artifacts={str(out/name): sha(out/name) for name in
            ('estimator.py','terminal_payload.npz','theta.npz','captures_sealed_before_targets.json','train_target_bindings.json')},
        parent_sha256=BANK_SHA, inverse_source_byte_exact=True, other_P_elements_byte_exact=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    for field in ('source','protocol','capture','cap-result'):
        parser.add_argument('--'+field+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    require(all(isinstance(v,str) and len(v)==64 and set(v)<=set('0123456789abcdef')
                for k,v in vars(args).items() if k!='run'), 'Reviewed immutable admission hashes')
    require(Path(sys.prefix).resolve() == (P2/'data_gate/.venv').resolve() and
            importlib.metadata.version('numpy') == '2.4.6', 'Pinned OFFPY numerical convention')
    require(os.getpriority(os.PRIO_PROCESS,0) >= 10 and all(os.environ.get(k)=='1' for k in
            ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')), 'nice10 BLAS1')
    out = HERE/'fit_v1'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    result = dict(schema='current_d50_terminal_linear_fit_v1', passed=False, completed=False,
        hashes_unchanged=False, training_targets_accessed=False, validation_targets_accessed=False,
        started_utc=datetime.now(timezone.utc).isoformat(), requested_seals=vars(args))
    write(out/'started.json', result)
    def expired(signum, frame):
        raise TimeoutError('Fixed120-second offline fit limit')
    signal.signal(signal.SIGALRM, expired); signal.alarm(120)
    try:
        run(args, out, result)
    except BaseException:
        result.update(passed=False, completed=False, hashes_unchanged=False, error=traceback.format_exc())
    finally:
        signal.alarm(0)
        result.update(wall_s=time.monotonic()-started,
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            finished_utc=datetime.now(timezone.utc).isoformat())
        write(out/'result.json', result)
        write(out/'completion.json', dict(passed=result['passed'], completed=result['completed'],
            result_sha256=sha(out/'result.json')))
    print(json.dumps({k:result.get(k) for k in ('passed','completed','training_affine_weighted_mse_before',
        'training_affine_weighted_mse_after','normal_equation_relative_residual','error')}), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
