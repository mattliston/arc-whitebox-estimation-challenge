"""Conditional source preparation: one fixed cstar gradient/JVP information pass."""
import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import textwrap
import time
import traceback
import weakref

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
P1 = NEW/'current_d50_native_objective_p1_v1'
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
TRUTH = P2/'public_g22_operator_features_v1/truthpost'
INDEX = NEW/'joint_prefix_final_fit_v1/manifest.json'
OUT = HERE/'information_v1'
RANKS = tuple(r for r in range(35, 59) if r not in (37, 39, 42, 48))
MODEL_SHA = 'a34f7de73914b76aee295ad8dba488bbd6257e93cfa33aa9d58124c2610ed0f1'
QUAL_SHA = '894128f110b098ea531c930cd7498cf0922ad5227f0c3839f8b9cd7a841db420'
QUAL_PROTOCOL_SHA = 'b77df4b6d68bb98c7ec6112bf33c150b416c4c467717a427b4227669cd6610a9'
PINS = {
    HERE/'model.py': MODEL_SHA,
    HERE/'PROTOCOL.md': 'c3660d88e1a9e90d3ff040b5f9cf191a323b0cf00589d1167fe4ff60bf851682',
    HERE/'qualification.py': QUAL_SHA, HERE/'QUALIFICATION_PROTOCOL.md': QUAL_PROTOCOL_SHA,
    P1/'training_v1/result.json': 'aaaef3d5f5a3c60583af779895b8db4f769276c64cb0cc032e93755169dbab91',
    P1/'training_v1/completion.json': '6964011b2262a70e56fafd74b040432065513706c3fde5586341a09b85fd0253',
    P1/'worker.py': 'd1160bfb2f5ac0edf4d425f09282ec79bbca310cb2f88cf7b97b6c59acbd426c',
    INDEX: 'd61757d28236e3d35d87415962d1a57d18ff38ae189651df52df5e3412b9a971',
}
QUAL_FLAGS = ('execution_integrity_passed', 'numerical_qualification_passed', 'all_graphs_retired',
    'all16_domains_qualified', 'all47_selectors_qualified', 'same_backend_null_byte_exact',
    'repeat_gradient_byte_exact', 'strict_secant_gates_passed', 'strict_gradient_gates_passed',
    'served_vector_ad_identity_passed', 'frozen_P_A_cstar_weights_passed', 'hashes_unchanged', 'sources_unchanged')
_TARGET_ALLOWLIST = set()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    path = Path(path).resolve()
    require(not path.is_relative_to(TRUTH.resolve()) or path in _TARGET_ALLOWLIST,
            'No target hash/open before full qualification admission; only fixed20 TRAIN')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    require(not Path(path).resolve().is_relative_to(TRUTH.resolve()), 'JSON metadata only')
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Pinned inert module')
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def helper_source():
    """Exact qualification call ownership; only the prescribed information loss changes."""
    require(sha(HERE/'qualification.py') == QUAL_SHA, 'Frozen fresh-qualification worker')
    q = module(HERE/'qualification.py', 'cstar_information_qualified_call_owner')
    _, vector, prior, pins = q.source_helpers()
    source = (HERE/'qualification.py').read_text()
    tree, lines, parts = ast.parse(source), source.splitlines(keepends=True), {}
    for name in ('array', 'rawsha', 'frozen_hash', 'run_card', 'exact_state'):
        found = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
        require(len(found) == 1, 'One exact qualification function: '+name)
        node = found[0]
        parts[name] = textwrap.dedent(''.join(lines[node.lineno-1:node.end_lineno]))
    call_old = parts['run_card']
    changes = (
        ('        def scalar_value(raw):\n            return scalar(torch, m._CO, raw, cot, card.dtype)\n',
         "        def scalar_value(raw):\n"
         "            y = m._CO.served(raw)[-1]\n"
         "            target = torch.as_tensor(cot[-1], dtype=y.dtype, device=y.device)\n"
         "            return (y-target).square().mean()/2e-8\n"),
        ("                    and np.isfinite(gradient).all() and (gradient[0] == 0).all()\n"
         "                    and np.linalg.norm(gradient.astype(np.float64)) > 1e-12, 'Finite nontrivial full60/active56 gradient')",
         "                    and np.isfinite(gradient).all() and (gradient[0] == 0).all(),\n"
         "                    'Finite full60/active56 information gradient; zero allowed by protocol')"),
    )
    adapted = call_old
    for old, new in changes:
        require(adapted.count(old) == 1, 'Unique prescribed loss/zero-gradient scope span')
        adapted = adapted.replace(old, new)
    inverse = adapted
    for old, new in reversed(changes):
        require(inverse.count(new) == 1, 'Unique information call inverse')
        inverse = inverse.replace(new, old)
    require(inverse == call_old, 'Exact original call owner outside prescribed loss/scope')
    parts['run_card'] = adapted
    vector_old = vector
    vchanges = (
        ('    j64, q64 = vector.astype(np.float64), np.asarray(cot, dtype=np.float64)\n'
         '    tangent = float(np.sum(j64[-1]*q64[-1])/math.sqrt(card.n)\n'
         '              + .05*np.sum(j64[:-1]*q64[:-1])/math.sqrt((card.depth-1)*card.n))\n',
         '    j64, y64 = vector.astype(np.float64), array(yp).astype(np.float64)\n'
         '    q64 = np.asarray(cot, dtype=np.float32).astype(np.float64)\n'
         '    tangent = float(2*np.mean((y64[-1]-q64[-1])*j64[-1])/2e-8)\n'),
        ('    del dual_served, yp, yt, vector, j64, q64, raw_primal\n',
         '    del dual_served, yp, yt, vector, j64, y64, q64, raw_primal\n'),
    )
    for old, new in vchanges:
        require(vector.count(old) == 1, 'One prescribed FP32-target vector contraction')
        vector = vector.replace(old, new)
    inverse = vector
    for old, new in reversed(vchanges):
        require(inverse.count(new) == 1, 'Unique actual-vector readout inverse')
        inverse = inverse.replace(new, old)
    require(inverse == vector_old, 'Same genuine served vector/primal/retirement, changed loss cotangent only')
    packet = '\n\n'.join(parts.values())+'\n\n'+vector
    proof = dict(qualification_worker_sha256=QUAL_SHA, parent_readout_proof=prior,
        original_call_sha256=hashlib.sha256(call_old.encode()).hexdigest(),
        adapted_call_sha256=hashlib.sha256(adapted.encode()).hexdigest(),
        packet_sha256=hashlib.sha256(packet.encode()).hexdigest(),
        call_source_inverse_byte_exact=True, vector_readout_inverse_byte_exact=True,
        model_checkpoint_domain_input_identity_retirement_unchanged=True,
        loss='FP32-target final served MSE/2e-8', actual_served_vector_jvp=True,
        qualification_nontrivial_gate_not_reused_for_different_loss=True,
        frozen_information_zero_direction_case_preserved=True)
    return packet, proof, pins, q


def qualification_admission(args, pin, qowner, expected_helper_proof):
    """Stdlib only; full future qualification evidence before any target access."""
    qdir, wdir = HERE/'qualification_v1', HERE/'qualification_watch_v1'
    qdigest, wdigest = args.qualification_result_sha256, args.qualification_watch_result_sha256
    pin(qdir/'result.json', qdigest)
    pin(wdir/'result.json', wdigest)
    q, w = read(qdir/'result.json'), read(wdir/'result.json')
    seals = {}
    for folder, record, digest in ((qdir, q, qdigest), (wdir, w, wdigest)):
        endpath, finalpath = folder/'completion.json', folder/'finalization.json'
        eh, fh = sha(endpath), sha(finalpath)
        pin(endpath, eh)
        pin(finalpath, fh)
        end, final = read(endpath), read(finalpath)
        require(record['passed'] is True and end['passed'] is True and final['passed'] is True
                and end['result_sha256'] == digest and final['completion_sha256'] == eh,
                'Actual passed qualification three-seal chain')
        if folder == qdir:
            require(record['completed'] and end['completed'] and final['completed']
                    and final['result_sha256'] == digest and 0 < final['elapsed_s'] < 600, 'Worker finalization covers complete stage')
        else:
            require(0 < end['elapsed_through_result_hash_s'] < 600
                    and 0 < final['elapsed_through_completion_io_s'] < 600, 'Independent qualification seals within600s')
        seals[str(folder)] = dict(result_sha256=digest, completion_sha256=eh, finalization_sha256=fh)
    require(q['schema'] == 'current_comp_cstar_qualification_v1' and q['stage'] == 'qualification'
            and all(q.get(k) is True for k in QUAL_FLAGS)
            and q['full_model_calls'] == q['retired_call_count'] == 34 and q['tiny_model_calls'] == 0
            and q['cpu_calls'] == q['mps_calls'] == 17
            and q['full_model_call_roster'] == [list(x) for x in qowner.ROSTER]
            and q['native_gradient_qualified'] is False and q['parameter_updates'] == q['optimizer_updates'] == 0
            and not q['truth_accessed'] and not q['training_targets_accessed'] and not q['validation_targets_accessed']
            and 0 < q['elapsed_s'] < 600 and 0 <= q['maximum_sampled_mps_driver_bytes'] < 16_000_000_000,
            'Every newly required strict34-call gate passed without targets or updates')
    require(q['helper_source_proof'] == expected_helper_proof, 'Exact qualified scalar/vector source proof')
    model_proof = q['model_source_proof']
    require(all(model_proof.get(k) is True for k in ('original_comp_step_unchanged',
            'inherited_step_inverse_byte_exact', 'original_node_function_unchanged',
            'original_birth_action_and_stored_plane_unchanged', 'original_checkpoint_domain_replay_unchanged',
            'parameter_binding_only', 'supplied_cstar_identity_preserved', 'exact_exported_cstar_null',
            'no_retanh_or_inverse_tanh', 'frozen_P_and_A', 'node_amplitude_and_endogenous_state_derivatives_live')),
            'Exact actual frozen COMP/cstar leaf ownership')
    require(w['schema'] == 'current_comp_cstar_qualification_watch_v1' and w['stage'] == 'qualification'
            and w['retired_calls'] == 34 and w['whole_owned_tree_retired'] and w['cleanup_passed'] and w['sources_unchanged']
            and w['mps_slot_clear_attested'] and not w['numerical_imports'] and w['whole_cap_s'] == 600
            and 0 < w['elapsed_s'] < 600 and 0 < w['maximum_combined_rss_bytes'] < 12_000_000_000
            and 0 < w['maximum_interval_s'] <= 1 and w['observation_samples'] > 0
            and w['child_result_sha256'] == qdigest
            and w['child_completion_sha256'] == seals[str(qdir)]['completion_sha256']
            and w['child_finalization_sha256'] == seals[str(qdir)]['finalization_sha256'],
            'Independent full resource/zero-exit/source/retirement qualification')
    required = {HERE/'model.py': MODEL_SHA, HERE/'PROTOCOL.md': PINS[HERE/'PROTOCOL.md'],
        HERE/'qualification.py': QUAL_SHA, HERE/'QUALIFICATION_PROTOCOL.md': QUAL_PROTOCOL_SHA,
        HERE/'watch_qualification.py': args.qualification_watcher_sha256,
        HERE/'WATCH_SPEC.md': args.qualification_watch_spec_sha256}
    for record in (q, w):
        for path, digest in required.items():
            require(record['source_bindings'].get(str(path)) == digest, 'Same reviewed six qualification owners')
        for group in ('source_bindings', 'input_bindings', 'artifacts'):
            for path, digest in record.get(group, {}).items():
                require(not Path(path).resolve().is_relative_to(TRUTH.resolve()), 'Qualification binds no target arrays')
                if group == 'artifacts':
                    require(Path(path).resolve().is_relative_to(qdir.resolve()), 'Qualification artifact fixed folder')
                pin(path, digest)
    samples_path, log_path = wdir/'samples.jsonl', wdir/'worker.log'
    pin(samples_path, sha(samples_path))
    pin(log_path, sha(log_path))
    samples = [json.loads(line) for line in samples_path.read_text().splitlines()]
    require(len(samples) == w['observation_samples'] and samples[-1]['child_pids'] == [], 'All observed samples and terminal child absence')
    require(all(math.isfinite(s['elapsed_s']) and 0 <= s['elapsed_s'] < 600
                and 0 <= s['combined_rss_bytes'] < 12_000_000_000
                and s['swap_bytes'] <= w['initial_swap_bytes'] and 0 <= s['retired_calls'] <= 34 for s in samples)
            and all(a['elapsed_s'] < b['elapsed_s'] and a['retired_calls'] <= b['retired_calls'] for a, b in zip(samples, samples[1:]))
            and max(s['combined_rss_bytes'] for s in samples) == w['maximum_combined_rss_bytes'], 'Actual observed resource ledger')
    markers = []
    for line in log_path.read_text().splitlines():
        try:
            item = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(item, dict) and item.get('event') == 'retired':
            markers.append(item)
    require(len(markers) == len(q['calls']) == 34, 'Actual complete marker/evidence roster')
    for index, (device, rank, kind) in enumerate(qowner.ROSTER):
        call, marker = q['calls'][index], markers[index]
        require(call['call'] == marker['call'] == index and call['device'] == marker['device'] == device
                and call['rank'] == marker['rank'] == rank and call['kind'] == marker['kind'] == kind
                and call['passed'] and call['completed'] and call['graph_retired'], 'Exact declared and observed call identity')
        for observations in (call['live_resources'], call['resources'], marker):
            require(0 <= observations['mps_driver_bytes'] <= q['maximum_sampled_mps_driver_bytes'] < 16_000_000_000
                    and 0 < observations['host_peak_rss_bytes'] < 12_000_000_000, 'All actual call-resource observations')
        for key in ('raw', 'served', 'diagnostics') + (('gradient',) if kind.startswith('backward') else ()):
            item = call[key]
            require(q['artifacts'].get(item['path']) == item['sha256'], 'Every full-call evidence seal')
        d = read(call['diagnostics']['path'])
        require(set(d['variance_domain']) == {str(l) for l in range(16)}
                and set(d['selection_cells']) == {f'{l}:{s}' for l in range(16) for s in range(2 if l == 15 else 3)}
                and set(d['node_head_calls']) == {str(l) for l in range(1, 15)}
                and all(v['finite'] and v['below_floor_count'] == 0 and v['minimum'] >= 1e-10 for v in d['variance_domain'].values()),
                'Every qualified16-domain/47-selector record')
        if kind.startswith('backward'):
            require(max(d['variance_visits'].values()) >= 2 and max(d['selection_visits'].values()) >= 2, 'Actual reverse replay visits')
        if kind == 'jvp':
            vector = call['served_jvp']
            require(q['artifacts'].get(vector['path']) == vector['sha256']
                    and q['artifacts'].get(vector['served_primal']['path']) == vector['served_primal']['sha256']
                    and call['ad_identity']['absolute_error'] <= call['ad_identity']['tolerance'], 'Actual complete served-vector evidence and strict identity')
    return q, dict(seals=seals, all_qualification_evidence_rehashed=True,
        independent_resources_and_retirement_passed=True, before_any_TRAIN_target_access=True,
        source_helper_and_model_proofs_exact=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    for key in ('source', 'model', 'protocol', 'watcher', 'watch-spec', 'qualification-result',
                'qualification-watch-result', 'qualification-watcher', 'qualification-watch-spec'):
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    began = time.monotonic()
    def expired(*_):
        raise TimeoutError('600s complete fixed-state information stage')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(600)
    require(not _TARGET_ALLOWLIST, 'Fresh target-free admission; no process resume')
    OUT.mkdir(exist_ok=False)
    result = dict(schema='current_comp_cstar_information_v1', stage='information', passed=False, completed=False,
        source_bindings={}, input_bindings={}, target_bindings={}, artifacts={}, calls=[], ranks=list(RANKS),
        gradient_calls=0, jvp_calls=0, nonlinear_calls=0, tiny_model_calls=0, full_model_calls=0, retired_call_count=0,
        optimizer_updates=0, parameter_updates=0, parameter_proposals=0, evaluated_interventions=0,
        direction_definitions=0, training_targets_accessed=False, validation_targets_accessed=False,
        native_gradient_qualified=False, native_score_measured=False, no_native_gradient_claim=True,
        information_promising=False, maximum_sampled_mps_driver_bytes=0)
    try:
        def pin(path, digest, group=None):
            path = Path(path).resolve()
            require(isinstance(digest, str) and len(digest) == 64 and sha(path) == digest, 'Exact bound bytes: '+str(path))
            group = group or ('source_bindings' if path.suffix in ('.py', '.md') else 'input_bindings')
            require(not path.is_relative_to(TRUTH.resolve()) or group == 'target_bindings', 'Separate authorized20 target bindings')
            require(str(path) not in result[group] or result[group][str(path)] == digest, 'Consistent binding')
            result[group][str(path)] = digest
            require(time.monotonic()-began < 600, 'Admission/rehashing within whole cap')
        require(args.model_sha256 == MODEL_SHA and len(RANKS) == 20, 'Fixed model and TRAIN roster')
        for path, digest in {**PINS, Path(__file__).resolve(): args.source_sha256,
                HERE/'INFORMATION_PROTOCOL.md': args.protocol_sha256,
                HERE/'watch_information.py': args.watcher_sha256,
                HERE/'INFORMATION_WATCH_SPEC.md': args.watch_spec_sha256,
                HERE/'watch_qualification.py': args.qualification_watcher_sha256,
                HERE/'WATCH_SPEC.md': args.qualification_watch_spec_sha256}.items():
            pin(path, digest)
        packet, proof, helper_pins, qowner = helper_source()
        for path, digest in helper_pins.items():
            pin(path, digest)
        qual, admission = qualification_admission(args, pin, qowner, proof['parent_readout_proof'])
        result['qualification_admission'], result['helper_source_proof'] = admission, proof
        m = module(HERE/'model.py', 'cstar_fixed_information_model')
        for path, digest in m.authenticate().items():
            pin(path, digest)
        p1, end = read(P1/'training_v1/result.json'), read(P1/'training_v1/completion.json')
        require(p1['schema'] == 'current_d50_native_objective_p1_v1' and p1['stage'] == 'train'
                and p1['passed'] and p1['completed'] and p1['hashes_unchanged'] and p1['sources_unchanged']
                and end['passed'] and end['completed'] and end['result_sha256'] == PINS[P1/'training_v1/result.json']
                and not p1['validation_targets_accessed'] and all(r in p1['train_ranks'] for r in RANKS),
                'Completed P1 TRAIN label provenance only; no old error screening')
        for path, digest in p1['source_bindings'].items():
            if Path(path).suffix in ('.py', '.md'):
                pin(path, digest)
        index = read(INDEX)
        pin(STORE/'manifest.json', index['manifest_sha256'])
        rows = read(STORE/'manifest.json')['roles']['observer_fit']['row_manifests']
        paths, target_pins = {}, {}
        for rank in RANKS:
            wp = STORE/f'rows/observer_fit/{rank:04d}/weights.npy'
            rowpath, tp = wp.with_name('row_manifest.json'), TRUTH/f'truthpost_rank{rank:04d}.npz'
            items = [x for x in index['target_sources'] if x['rank'] == rank]
            entries = [x for x in rows if x['role_rank'] == rank]
            require(len(items) == len(entries) == 1, 'Unique fixed TRAIN metadata')
            item, row = items[0], read(rowpath)
            require(item['weight_file'] == str(wp) and item['target_file'] == str(tp)
                    and item['target_sha256'] == p1['target_bindings'][str(tp)]
                    and (STORE/entries[0]['path']).resolve() == rowpath.resolve()
                    and entries[0]['sha256'] == p1['weight_bindings'][str(rowpath)]
                    and row['role'] == 'observer_fit' and row['role_rank'] == rank
                    and row['files']['weights.npy']['sha256'] == p1['weight_bindings'][str(wp)], 'Canonical same1e9-label/weight lineage')
            pin(rowpath, p1['weight_bindings'][str(rowpath)])
            pin(wp, p1['weight_bindings'][str(wp)])
            paths[rank], target_pins[rank] = wp, (tp, item['target_sha256'])
        require(Path(sys.executable).resolve() == Path('/Users/matt/anaconda3/bin/python').resolve()
                and sys.dont_write_bytecode and sys.flags.optimize == 0 and os.getpriority(os.PRIO_PROCESS, 0) >= 10
                and all(os.environ.get(k) == '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
                and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
                and os.environ.get('PYTORCH_MPS_LOW_WATERMARK_RATIO') == '0.1', 'Fixed AP/nice10/BLAS/MPS environment')
        require(not any(k in sys.modules for k in ('numpy', 'torch', 'flopscope', 'threadpoolctl')),
                'Full qualification/source/target-metadata admission before numerical imports or target files')
        import numpy as np
        import torch
        require(np.__version__ == '1.26.4' and torch.__version__.split('+')[0] == '2.7.0'
                and torch.backends.mps.is_available() and 'threadpoolctl' not in sys.modules, 'Qualified runtime')
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        recommended = int(torch.mps.recommended_max_memory())
        require(recommended > 0, 'Positive MPS allocation recommendation')
        fraction = min(1., 16_000_000_000/recommended)
        torch.mps.set_per_process_memory_fraction(fraction)
        result['mps_allocator'] = dict(configured=True, api='torch.mps.set_per_process_memory_fraction',
            recommended_max_memory_bytes=recommended, configured_fraction=fraction, low_watermark_ratio='0.1')
        result['versions'] = dict(numpy=np.__version__, torch=torch.__version__)
        m.runtime()

        def guard():
            torch.mps.synchronize()
            driver, host = int(torch.mps.driver_allocated_memory()), int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            result['maximum_sampled_mps_driver_bytes'] = max(result['maximum_sampled_mps_driver_bytes'], driver)
            require(driver < 16_000_000_000 and host < 12_000_000_000 and time.monotonic()-began < 600,
                    'Whole600s/12GB host/16GB driver')
            return dict(mps_driver_bytes=driver, host_peak_rss_bytes=host, mps_initialized=True)

        def save(name, value):
            require(np.isfinite(value).all(), 'Finite saved evidence')
            path = OUT/(name+'.npy')
            with path.open('xb') as stream:
                np.save(stream, value, allow_pickle=False)
            digest = sha(path)
            result['artifacts'][str(path)] = digest
            return dict(path=str(path), sha256=digest, raw_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                dtype=str(value.dtype), shape=list(value.shape))

        def doc(name, value):
            path = OUT/(name+'.json')
            write(path, value)
            digest = sha(path)
            result['artifacts'][str(path)] = digest
            return dict(path=str(path), sha256=digest)

        def load_saved(item):
            require(result['artifacts'].get(item['path']) == item['sha256'] and sha(item['path']) == item['sha256'], 'Same-run array seal')
            value = np.load(item['path'], allow_pickle=False)
            require(list(value.shape) == item['shape'] and str(value.dtype) == item['dtype']
                    and hashlib.sha256(value.tobytes()).hexdigest() == item['raw_sha256'] and np.isfinite(value).all(), 'Full saved array identity')
            return value

        cstar0 = np.asarray(m._JOINT._LITERALS['_BD_COEFFICIENT_LITERAL'], dtype=np.float32)
        null = qual['fixtures']['payloads']['null']
        require(null['shape'] == [15, 4] and null['dtype'] == 'float32'
                and null['raw_sha256'] == hashlib.sha256(cstar0.tobytes()).hexdigest(), 'Exact newly qualified endpoint bytes')
        env = dict(np=np, torch=torch, m=m, math=math, gc=gc, weakref=weakref, hashlib=hashlib,
            require=require, save=save, guard=guard, OFFSETS=qowner.OFFSETS,
            payloads={'null': cstar0}, direction_np=np.zeros((15, 4), np.float32))
        exec(compile(packet, str(__file__)+':exact_qualified_call_adapter', 'exec'), env)
        run_card, exact_state = env['run_card'], env['exact_state']
        result['admission_seal'] = doc('qualification_and_inputs_admitted_before_targets', dict(
            qualification_admission=admission, helper_source_proof=proof,
            source_bindings=result['source_bindings'], input_bindings=result['input_bindings'],
            null_cstar=save('fixed_cstar_null', cstar0), ranks=list(RANKS),
            target_metadata={str(r): dict(path=str(p), sha256=h) for r, (p, h) in target_pins.items()},
            targets_opened=False, targets_hashed=False, full_model_calls=0, parameter_updates=0))
        _TARGET_ALLOWLIST.update(p.resolve() for p, _ in target_pins.values())
        require(len(_TARGET_ALLOWLIST) == 20, 'Only prescribed20 TRAIN target files become eligible')
        targets, baseline, gradients = {}, {}, {}

        def attempt(index, rank, kind):
            require(kind == ('backward' if index < 20 else 'jvp') and rank == RANKS[index % 20], 'Fixed40-call roster')
            wp = paths[rank]
            require(sha(wp) == result['input_bindings'][str(wp)], 'Original immutable input before each call')
            weights = np.load(wp, allow_pickle=False)
            require(weights.shape == (16, 1024, 1024) and weights.dtype == np.float32 and np.isfinite(weights).all(), 'Full canonical input')
            answer = run_card('mps', rank, kind, weights, targets[rank])
            del weights
            record = {k: v for k, v in answer.items() if k not in ('raw', 'served', 'gradient', 'diagnostics')}
            record.update(call=index, device='mps', rank=rank, kind=kind, graph_retired=True,
                raw=save(f'call_{index:02d}_raw', answer['raw']), served=save(f'call_{index:02d}_served', answer['served']),
                diagnostics=doc(f'call_{index:02d}_diagnostics', answer['diagnostics']))
            if answer['gradient'] is not None:
                record['gradient'] = save(f'call_{index:02d}_gradient', answer['gradient'])
            doc(f'call_{index:02d}_attempt', record)
            return answer, record

        def retired(index, record):
            record.update(passed=True, completed=True)
            doc(f'call_{index:02d}_complete', record)
            result['calls'].append(record)
            result['full_model_calls'] += 1
            result['retired_call_count'] += 1
            gc.collect()
            resources = guard()
            print(json.dumps(dict(event='retired', call=index, device='mps', rank=record['rank'], kind=record['kind'], **resources)), flush=True)

        result['target_original_arrays'] = {}
        for index, rank in enumerate(RANKS):
            tp, digest = target_pins[rank]
            pin(tp, digest, 'target_bindings')
            result['training_targets_accessed'] = True
            with np.load(tp, allow_pickle=False) as archive:
                values = archive['mc_post'].copy()
            require(values.shape == (16, 1024) and values.dtype.kind == 'f' and np.isfinite(values).all(), 'Original finite1e9 means')
            result['target_original_arrays'][str(rank)] = dict(dtype=str(values.dtype), shape=list(values.shape),
                raw_sha256=hashlib.sha256(values.tobytes()).hexdigest(), file_sha256=digest)
            targets[rank] = values.astype(np.float64)
            answer, record = attempt(index, rank, 'backward')
            record['target_original_dtype'] = str(values.dtype)
            record['original_label_raw_final_mse'] = float(np.mean((answer['served'][-1].astype(np.float64)-targets[rank][-1])**2))
            if rank == 35:
                qcall = next(x for x in qual['calls'] if x['device'] == 'mps' and x['rank'] == 35 and x['kind'] == 'backward')
                qdiag = read(qcall['diagnostics']['path'])
                require(all(record[k]['raw_sha256'] == qcall[k]['raw_sha256'] and record[k]['dtype'] == qcall[k]['dtype']
                            and record[k]['shape'] == qcall[k]['shape'] for k in ('raw', 'served'))
                        and all(answer['diagnostics'][k] == qdiag[k] for k in ('variance_domain', 'selection_cells'))
                        and answer['frozen_hashes'] == qcall['frozen_hashes'], 'Actual current35 endpoint matches newly qualified bytes')
                record['qualified_zero35_byte_exact'] = True
            baseline[rank], gradients[rank] = answer, answer['gradient'].astype(np.float64)
            result['gradient_calls'] += 1
            del values
            retired(index, record)
        pooled = np.zeros((15, 4), np.float64)
        for rank in RANKS:
            pooled += gradients[rank]
        pooled /= 20
        norm = float(np.linalg.norm(pooled))
        require(np.isfinite(pooled).all() and math.isfinite(norm) and (pooled[0] == 0).all(), 'Finite one pooled active56 gradient')
        direction = np.zeros((15, 4), np.float32) if norm == 0 else (-pooled/norm).astype(np.float32)
        direction[0] = 0.
        env['direction_np'] = direction
        result['direction_definitions'] = 1
        result['direction_seal'] = doc('single_direction_sealed', dict(ranks=list(RANKS), gradient_calls=20,
            gradient_artifacts={str(r): result['calls'][i]['gradient'] for i, r in enumerate(RANKS)},
            pooled_gradient=save('pooled_gradient64', pooled), pooled_norm=norm,
            direction=save('single_direction32', direction), zero_pooled_gradient=bool(norm == 0),
            direction_definitions=1, parameter_updates=0, parameter_proposals=0,
            qualification_result_sha256=args.qualification_result_sha256))
        rows = []
        for offset, rank in enumerate(RANKS):
            index = 20+offset
            answer, record = attempt(index, rank, 'jvp')
            old = baseline[rank]
            exact_state(old, answer)
            require(old['frozen_hashes'] == answer['frozen_hashes'], 'Same actual endpoint inputs in both passes')
            reverse = float(np.sum(gradients[rank]*direction.astype(np.float64)))
            tangent = answer['tangent']
            require(math.isfinite(reverse) and math.isfinite(tangent), 'Finite FP32-target loss directional values')
            identity = dict(reverse_dot=reverse, forward_ad=tangent, absolute_error=abs(reverse-tangent),
                tolerance=1e-10+1e-4*max(abs(reverse), abs(tangent)),
                fp32_target_cotangent=True, actual_served_jvp=answer['served_jvp'])
            doc(f'call_{index:02d}_directional_identity', identity)
            require(identity['absolute_error'] <= identity['tolerance'], 'Unchanged strict actual-vector FAD/RAD identity')
            j = load_saved(answer['served_jvp'])[-1].astype(np.float64)
            residual = old['served'][-1].astype(np.float64)-targets[rank][-1]
            row = dict(rank=rank, a=float(np.mean(residual*residual)), b=float(np.mean(residual*j)), c=float(np.mean(j*j)))
            require(all(math.isfinite(row[k]) for k in ('a', 'b', 'c')), 'Finite original-label scalar statistics')
            rows.append(row)
            record.update(ad_identity=identity, original_label_quadratic=row)
            result['jvp_calls'] += 1
            del j, residual
            retired(index, record)
        a, b, c = (math.fsum(row[k] for row in rows)/20 for k in ('a', 'b', 'c'))
        require(all(math.isfinite(x) for x in (a, b, c)) and a >= 0 and c >= 0, 'Finite nonnegative squared energies')
        degenerate = a == 0 or c == 0
        gain = 0. if degenerate else min(b, 0.)**2/(a*c)
        require(math.isfinite(gain), 'Finite unclipped prescribed Gline')
        result['statistics'] = dict(per_network=rows, a=a, b=b, c=c, Gline=gain,
            zero_a=bool(a == 0), zero_c=bool(c == 0), degenerate=bool(degenerate),
            equal_network_and_neuron_weighting=True, original_label_precision=True,
            native_cost_measured=False, native_gradient_claim=False, single_direction_not_capacity_bound=True,
            usefulness_gate=.08, Gline_is_linearized_not_nonlinear_gain=True)
        result['information_promising'] = bool(gain >= .08)
        require(result['gradient_calls'] == result['jvp_calls'] == 20 and result['full_model_calls'] == result['retired_call_count'] == 40
                and result['nonlinear_calls'] == result['tiny_model_calls'] == 0
                and result['direction_definitions'] == 1, 'Exact fixed40 calls and one direction; no finite intervention')
        for group in ('source_bindings', 'input_bindings', 'target_bindings', 'artifacts'):
            require(all(sha(path) == digest for path, digest in result[group].items()), 'Every source/input/label/evidence byte unchanged: '+group)
        result['model_source_proof'] = m.source_proof()
        del targets, baseline, gradients, answer
        gc.collect()
        guard()
        result.update(passed=True, completed=True, execution_integrity_passed=True, sources_unchanged=True,
            hashes_unchanged=True, all_graphs_retired=True, all16_domains_qualified=True,
            all47_selectors_qualified=True, same_endpoint_primal_domain_replay_passed=True,
            served_vector_ad_identity_passed=True, frozen_P_A_cstar_weights_passed=True,
            qualification_verified_before_targets=True, no_checkpoint_selection=True,
            no_finite_step_selected=True, no_optimizer_constructed=True)
    except BaseException:
        result.update(passed=False, completed=False, information_promising=False, error=traceback.format_exc())
    result['elapsed_s'] = time.monotonic()-began
    write(OUT/'result.json', result)
    digest = sha(OUT/'result.json')
    write(OUT/'completion.json', dict(passed=result['passed'], completed=result['completed'], result_sha256=digest))
    end = sha(OUT/'completion.json')
    elapsed = time.monotonic()-began
    ok = bool(result['passed'] and elapsed < 600)
    write(OUT/'finalization.json', dict(passed=ok, completed=ok, result_sha256=digest, completion_sha256=end, elapsed_s=elapsed))
    sha(OUT/'finalization.json')
    within = time.monotonic()-began < 600
    signal.alarm(0)
    return 0 if ok and within else 1


if __name__ == '__main__':
    raise SystemExit(main())
