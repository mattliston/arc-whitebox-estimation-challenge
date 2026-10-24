"""Fixed two-arm native CHECK4; no model, fitting or partial scoring."""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import sys
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
COMP = NEW/'current_p1_node_composition_v1'
RC = NEW/'current_d50_response10_candidate_v1'
CENSUS = NEW/'current_d50_response10_census_v1/watch.py'
CENSUS_SHA = '7f868ba0250f142822b2aa36503d1908b21849c47e6e7406320f979e5d93bbcf'
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
INDEX = NEW/'joint_prefix_final_judge_cached_v2/receipt.json'
TRUTH = P2/'public_g22_operator_features_v1/truthpost'
RANKS, ARMS, BUDGET = (37, 39, 42, 48), ('p_only', 'joint'), 2**41
PARENT_SHA = '2e13d40a21e62691155918719f8ec088f581d495ae17800c5503c3b56504918a'
BASE_SHA = 'a9071e5afa6dbf35e6a6871d6997a7dbc8a1105d2bf3e97bd172c7a1b0e0180f'
SCORE_SHA = '17bd92bb65abb5d113cf3cd57df175b865f03295a7e0d96eadfeb133b6658390'
MODEL_SHA = 'cb49d7d598a76e430f388e6f3dc84b6b79d89eaf8a0e12cd3c19f8caca61cc86'
SCHEMA = 'current_comp_joint_p_node_score_v1'
PINS = {
    HERE/'model.py': MODEL_SHA,
    HERE/'PROTOCOL.md': '330f744e26f5672748ce94fc20f6e0f617358f83883d6a6c0c4b25c185d11830',
    HERE/'DECISION.md': '4ef8567620f6b2efb2890fda32eb029082fc1a07e084ab2f37ec35b4d29e4118',
    HERE/'TRAIN_PROTOCOL.md': '53cb4c490f0a474e12fb35d18f1ecfa59f229ea0cb9c0ef8442a2ddf7d3bee16',
    COMP/'native.py': 'fafc28732701aacfa70ada15d30ed083af76e91747bc01dc840ec8ba14d23d01',
    COMP/'score.py': 'e1d4874068b6638c95b4b66063d0abfe2e5327a07cb0ebfef1c442105e037dfe',
    COMP/'build_v1/terminal_estimator.py': PARENT_SHA,
    COMP/'native_terminal_v2/result.json': BASE_SHA,
    COMP/'native_terminal_v2/completion.json': '6c050c04fd565d840273f2a917f2a106182cfed1b69f59a09cd5392497706772',
    COMP/'score_v3/result.json': SCORE_SHA,
    COMP/'score_v3/completion.json': '23d4e593af714034e7208a3a5947bef47386cf87a47dd5872c568a22c65415b4',
    RC/'score.py': '3bc2aeb4d66a74984c2801c281a52e11b75a52f36d9bff5c877cca3daccbf2b4',
    RC/'score_v2.py': 'e4a9ef56dd5309e33cde2d67ad983e8655918af12909a8fdec58db0292c27eaf',
    RC/'SCORE_DTYPE_V2.md': '9995cd8ae375ae0decef7615e1b08678941510f00f1bd3bcbb7436a649bf30ac',
    CENSUS: CENSUS_SHA,
    STORE/'manifest.json': '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50',
    INDEX: '4968610553b95011cabdc651dd65d4c9057054f9d2d2bf73711426fff695893c',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def extract(path, name):
    source = Path(path).read_text(); tree = ast.parse(source)
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    require(len(found) == 1, 'Unique preserved function: '+name)
    n = found[0]
    return ''.join(source.splitlines(keepends=True)[n.lineno-1:n.end_lineno])


def splice(source, changes):
    changed = source
    for before, after in changes:
        require(changed.count(before) == 1, 'Unique metadata-only splice')
        changed = changed.replace(before, after)
    inverse = changed
    for before, after in reversed(changes):
        require(inverse.count(after) == 1, 'Unique metadata inverse')
        inverse = inverse.replace(after, before)
    require(inverse == source, 'Exact original function restored')
    return changed


class Admission:
    def __init__(self, result, guard):
        self.result, self.guard = result, guard
        self.pins = result['input_bindings']

    def pin(self, path, digest):
        path = Path(path).resolve(); key = str(path)
        require(path.is_relative_to(P2) and 'truthpost' not in path.parts
            and not (path.suffix in ('.npy', '.npz') and 'truth' in path.name.lower()), 'No target IO before seal')
        require(re.fullmatch('[0-9a-f]{64}', digest or ''), 'Exact binding digest')
        previous = self.pins.get(key)
        require(previous is None or previous == digest, 'Consistent binding')
        if previous is None:
            require(sha(path) == digest, 'Immutable source/input: '+key)
            self.pins[key] = digest
        if path.suffix in ('.py', '.md'):
            self.result['source_bindings'][key] = digest
        self.guard()
        return digest

    def bind(self, mapping, ignored=None):
        for key, value in mapping.items():
            path, digest = (value['path'], value['sha256']) if isinstance(value, dict) else (key, value)
            self.pin(path, digest)

    def inventory(self, record):
        for group in ('source_bindings', 'input_bindings', 'weight_bindings', 'artifacts'):
            self.bind(record.get(group, {}))  # Deliberately no target/truth bindings.

    def chain(self, folder, digest, *, watch=False, limit=600, inventory=True):
        rp, ep, fp = (folder/n for n in ('result.json', 'completion.json', 'finalization.json'))
        self.pin(rp, digest); eh, fh = sha(ep), sha(fp); self.pin(ep, eh); self.pin(fp, fh)
        r, e, f = read(rp), read(ep), read(fp)
        require(r['passed'] and e['passed'] and f['passed'] and e['result_sha256'] == digest
            and f['completion_sha256'] == eh, 'Successful three-seal chain: '+str(folder))
        if watch:
            require(r['whole_owned_tree_retired'] and r['cleanup_passed'] and r['sources_unchanged']
                and r['elapsed_s'] < limit and r['maximum_interval_s'] <= 1
                and r['maximum_combined_rss_bytes'] < 8_000_000_000
                and e['elapsed_through_result_hash_s'] < limit
                and f['elapsed_through_completion_io_s'] < limit, 'Complete independent resource seal')
        else:
            require(r['completed'] and e['completed'] and f['completed']
                and f['result_sha256'] == digest and f['elapsed_s'] < limit, 'Complete worker final seal')
        if inventory:
            self.inventory(r)
        return r, e, dict(result_sha256=digest, completion_sha256=eh, finalization_sha256=fh)


def helpers(admit):
    """Exact selected functions only: no legacy module, model or watcher import."""
    env = dict(Path=Path, ast=ast, hashlib=hashlib, math=math, require=require, sha=sha, read=read,
        bind=admit.bind, BUDGET=BUDGET, RSS_LIMIT=8_000_000_000)
    life = extract(CENSUS, 'lifecycle')
    npy = extract(COMP/'native.py', 'npy_bytes')
    trial_original = extract(RC/'score.py', 'trial_output')
    trial = splice(trial_original, [('value.dtype == np.float64', 'value.dtype == np.float32')])
    verifier_original = extract(COMP/'native.py', 'verify_acquisition')
    verifier = splice(verifier_original, [
        ("record['builder_sha256'] == BUILDER_SHA", "record['source_bindings'][str(HERE/'build.py')] == BUILDER_SHA"),
        ("str(HERE/'PROTOCOL.md')", "str(HERE/'NATIVE_PROTOCOL.md')")])
    exec(compile(life+'\n'+npy+'\n'+trial, str(__file__)+':exact_saved_checks', 'exec'), env)
    env['load'] = lambda path, name, expected: lifecycle_owner(path, expected, env['lifecycle'])
    env.update(CENSUS=CENSUS, CENSUS_SHA=CENSUS_SHA)
    exec(compile(verifier, str(__file__)+':metadata_only_acquisition_adapter', 'exec'), env)
    proof = dict(acquisition_original_sha256=hashlib.sha256(verifier_original.encode()).hexdigest(),
        acquisition_adapted_sha256=hashlib.sha256(verifier.encode()).hexdigest(), acquisition_inverse_verified=True,
        saved_output_original_sha256=hashlib.sha256(trial_original.encode()).hexdigest(),
        saved_FP32_output_sha256=hashlib.sha256(trial.encode()).hexdigest(), saved_dtype_inverse_verified=True,
        lifecycle_exact_source_sha256=hashlib.sha256(life.encode()).hexdigest(),
        imported_legacy_modules=False, unbound_transitive_watch_helper=False)
    return env, proof


def lifecycle_owner(path, expected, fn):
    require(Path(path) == CENSUS and expected == CENSUS_SHA and sha(path) == expected, 'Only exact lifecycle function')
    return SimpleNamespace(lifecycle=fn)


def baseline(admit):
    folder = COMP/'native_terminal_v2'
    native, end = read(folder/'result.json'), read(folder/'completion.json')
    score, se = read(COMP/'score_v3/result.json'), read(COMP/'score_v3/completion.json')
    require(all(native.get(k) is True for k in ('passed', 'completed', 'hashes_unchanged', 'sources_unchanged',
        'own_complete_bills', 'all16_domains_qualified', 'all47_selectors_qualified', 'repeat_output_domain_byte_exact',
        'initialization_ownership_passed')) and native['candidate_sha256'] == PARENT_SHA
        and not native['truth_accessed'] and end['passed'] and end['completed'] and end['result_sha256'] == BASE_SHA
        and len(native['trials']) == len(native['input_preparation']) == len(native['diagnostic_records']) == 41
        and [t['rank'] for t in native['trials'][:4]] == list(RANKS) and native['trials'][-1]['rank'] == RANKS[0],
        'Retained current COMP native40 control and its own archived repeat')
    require(score['passed'] and score['completed'] and score['integrity_passed'] and score['hashes_unchanged']
        and score['candidate_sha256'] == PARENT_SHA and score['advances_to_independent14']
        and len(score['records']) == 40 and se['passed'] and se['completed'] and se['result_sha256'] == SCORE_SHA,
        'Exact completed COMP native score reference')
    rows = read(STORE/'manifest.json')['roles']['observer_fit']['row_manifests']; refs = {}
    for call, rank in enumerate(RANKS):
        found = [x for x in rows if x['role_rank'] == rank]; require(len(found) == 1, 'Unique canonical row')
        rp = (STORE/found[0]['path']).resolve(); require(rp.is_relative_to(STORE), 'Canonical STORE path')
        admit.pin(rp, found[0]['sha256']); row = read(rp); wp = rp.parent/'weights.npy'
        wh = row['files']['weights.npy']['sha256']; admit.pin(wp, wh)
        trial, prepared = native['trials'][call], native['input_preparation'][call]
        require(row['role_rank'] == rank and row['role'] == 'observer_fit'
            and trial['call'] == prepared['call'] == call and trial['rank'] == prepared['rank'] == rank
            and prepared['weight_file_sha256'] == wh, 'Target rank, physical weight and archived input match')
        refs[rank] = dict(trial, weight_file_sha256=wh, weight_file=str(wp))
    return folder, native, score, refs


def arm_admission(args, arm, admit, env, refs, builder):
    get = lambda name: getattr(args, arm+'_'+name+'_sha256')
    build_folder = HERE/f'build_{arm}_v1'
    build, _, bs = admit.chain(build_folder, get('build'), limit=60)
    built = builder.verify_build(arm, get('build'), get('candidate'), args.builder_sha256, args.build_protocol_sha256)
    require(built['build'] == build, 'Same complete build independently reverified without array decoding')
    admit.bind(built['source_bindings']); admit.bind(built['input_bindings'])
    require(build['schema'] == 'current_comp_joint_p_node_build_v1' and build['arm'] == arm
        and build['hashes_unchanged'] and build['sources_unchanged'] and build['peak_rss_bytes'] < 2_000_000_000
        and build['candidate'] == dict(path=str(build_folder/'estimator.py'), sha256=get('candidate'))
        and build['parent'] == dict(path=str(COMP/'build_v1/terminal_estimator.py'), sha256=PARENT_SHA)
        and build['source_bindings'][str(HERE/'build.py')] == args.builder_sha256
        and build['source_bindings'][str(HERE/'BUILD_PROTOCOL.md')] == args.build_protocol_sha256
        and all(build['proof'][k] for k in ('literal_inverse_byte_exact', 'literal_roundtrip_byte_exact',
            'frozen_nonP_and_inactive66_byte_exact', 'cstar_source_byte_exact', 'full_shape_FP32_finite')),
        'Exact complete fixed-arm literal build')
    ap = HERE/f'native_{arm}_admission_v1'
    # The completed native result and watch independently bind the context digest.
    nf = HERE/f'native_{arm}_v1'; wf = HERE/f'native_{arm}_watch_v1'
    native, end, ns = admit.chain(nf, get('native'))
    watch, _, ws = admit.chain(wf, get('watch'), watch=True)
    context_sha = native['admission_context_sha256']; admit.pin(ap/'context.json', context_sha)
    context = read(ap/'context.json')
    ar_sha = native['source_bindings'].get(str(ap/'result.json'), native.get('input_bindings', {}).get(str(ap/'result.json')))
    ar, _, ads = admit.chain(ap, ar_sha, limit=60)
    require(ar['arm'] == context['arm'] == arm and ar['context_sha256'] == context_sha
        and ar['sources_unchanged'] and not ar['truth_accessed'] and not ar['numerical_imports']
        and ar['elapsed_s'] < 60 and ar['peak_rss_bytes'] < 2_000_000_000
        and context['source_admission_complete'] and not context['truth_accessed'] and not context['numerical_imports']
        and context['candidate'] == build['candidate']['path'] and context['candidate_sha256'] == get('candidate')
        and context['build_sha256'] == native['build_sha256'] == get('build')
        and native['full_admission_in_retired_process'] and native['full_model_calls'] == native['retired_call_count'] == 5
        and native['arm'] == arm and native['proof'] == context['proof'], 'Same retired full source-admission and candidate')
    admit.inventory(context)
    owned = {str(HERE/'native.py'): args.native_source_sha256,
        str(HERE/'watch_native.py'): args.native_watch_source_sha256,
        str(HERE/'NATIVE_PROTOCOL.md'): args.native_protocol_sha256, str(HERE/'model.py'): MODEL_SHA}
    for item in (native, watch, context):
        require(all(item['source_bindings'].get(p) == h for p, h in owned.items()), 'Same caller-reviewed native owners')
    require(context['source_bindings'][str(HERE/'build.py')] == args.builder_sha256
        and context['source_bindings'][str(HERE/'BUILD_PROTOCOL.md')] == args.build_protocol_sha256
        and watch['source_bindings'][str(ap/'context.json')] == context_sha
        and watch['retired_calls'] == 5 and all(watch['child_'+key] == value for key, value in ns.items()),
        'Independent watch binds all native child seals and the admitted context')
    for rank in RANKS:
        prior = dict(refs[rank]); prior.pop('weight_file')
        require(context['reference_by_rank'][str(rank)] == prior, 'Same explicit COMP reference ownership')
    require(native['versions'] == dict(numpy='2.4.6', flopscope='0.12.0', whestbench='0.16.0', threadpoolctl='3.6.0'),
        'Original complete native runtime')
    for name in ('samples.jsonl', 'worker.log', 'started.json'):
        admit.pin(wf/name, sha(wf/name))
    samples = [json.loads(line) for line in (wf/'samples.jsonl').read_text().splitlines()]
    require(len(samples) == watch['observation_samples'] and samples and samples[-1]['retired_calls'] == 5
        and max(x['combined_rss_bytes'] for x in samples) == watch['maximum_combined_rss_bytes']
        and all(x['swap_bytes'] <= watch['initial_swap_bytes'] and x['combined_rss_bytes'] < 8_000_000_000 for x in samples),
        'Corroborated complete saved zero-swap resource observations')
    env.update(HERE=HERE, FIXED={arm: RANKS}, CANDIDATES={arm: get('candidate')},
        SCHEMA='current_comp_joint_p_node_native_v1', BUILD_SHA=get('build'),
        BUILDER_SHA=args.builder_sha256, PROTOCOL_SHA=args.native_protocol_sha256)
    env['verify_acquisition'](native, end, get('native'), arm, args.native_source_sha256,
                              args.native_watch_source_sha256, refs)
    return nf, native, dict(build=bs, admission=ads, native=ns, watch=ws, context_sha256=context_sha)


def original_control_checks(admit, env, folder, record, refs):
    selected = (0, 1, 2, 3, 40)
    for call in selected:
        trial, prepared = record['trials'][call], record['input_preparation'][call]
        require(prepared['rank'] == trial['rank'] and prepared['call'] == trial['call'] == call
            and prepared['weight_file_sha256'] == refs[trial['rank']]['weight_file_sha256']
            and trial['includes_initialization'] is (call == 0), 'Original own COMP input/cold history')
        for suffix, bill in (('', trial), ('_input', prepared)):
            require(type(bill['flops']) is int and 0 < bill['flops'] <= BUDGET
                and all(math.isfinite(bill[k]) and bill[k] >= 0 for k in
                    ('wall_s', 'context_wall_s', 'flopscope_backend_s', 'flopscope_overhead_s', 'native_residual_s'))
                and bill['wall_s'] < 120 and bill['context_wall_s'] < 120
                and 0 < bill['peak_rss_bytes'] < 8_000_000_000, 'Original full control input/participant resource bill')
            bp = folder/f'call_{call:02d}{suffix}_bill.json'
            lp = folder/f"call_{call:02d}_{'lifecycle' if not suffix else 'input_reset'}.json"
            for p in (bp, lp):
                admit.pin(p, record['artifacts'][str(p)])
            require(read(bp) == bill, 'Original complete saved control bill')
            env['lifecycle'](dict(bill, lifecycle_receipt=str(lp)), bp, lp)
        require(prepared['flops'] == 16*1024*1024 and prepared['layers'] == 16
            and prepared['charged_to_participant'] is False and all(prepared[k] is True for k in
                ('official_fnp_type', 'exact_bytes', 'independent_weight_buffers', 'context_closed')),
            'Original independent official weight materialization')
        dp = folder/f'call_{call:02d}_diagnostics.json'; admit.pin(dp, record['artifacts'][str(dp)])
        require(record['diagnostic_records'][call] == dict(path=str(dp), sha256=sha(dp)), 'Original diagnostic owner')
        diagnostic = read(dp); domain = diagnostic['domains']
        keys = {f'{l}:{s}' for l in range(16) for s in range(2 if l == 15 else 3)}
        require(diagnostic['call'] == call and diagnostic['rank'] == trial['rank']
            and not diagnostic['truth_accessed'] and diagnostic['offline_after_saved_bill_and_context_release']
            and domain['variance_count'] == 16 and domain['selector_count'] == 47
            and set(domain['variance_domain']) == {str(l) for l in range(16)} and set(domain['selection_cells']) == keys
            and all(v['finite'] and math.isfinite(v['minimum']) and v['minimum'] >= 1e-10
                and v['below_floor_count'] == 0 for v in domain['variance_domain'].values())
            and all(v['floor'] == .01 and type(v['count']) is int and 0 <= v['count'] <= 1024
                and v['count'] % 32 == 0 for v in domain['selection_cells'].values()), 'Original all16/all47 control qualification')
        out = Path(trial['output']); admit.pin(out, record['artifacts'][str(out)])
        require(trial['output_sha256'] == sha(out), 'Original control output SHA')
        env['npy_bytes'](out, trial['output_bytes_sha256'])
    first, repeat = (read(folder/f'call_{call:02d}_diagnostics.json') for call in (0, 40))
    measured = record['trials'][0]['flops']-record['trials'][40]['flops']
    require(first['domains'] == repeat['domains'] and measured == record['initialization_flops']
        == first['startup']['observed_startup_region_flops']-repeat['startup']['observed_startup_region_flops'],
        'Archived own COMP repeated domain and startup; no cross-history opcount comparison')


def score(args, out, result, guard):
    admit = Admission(result, guard)
    sources = {**PINS, Path(__file__).resolve(): args.source_sha256, HERE/'SCORE_PROTOCOL.md': args.protocol_sha256,
        HERE/'native.py': args.native_source_sha256, HERE/'watch_native.py': args.native_watch_source_sha256,
        HERE/'NATIVE_PROTOCOL.md': args.native_protocol_sha256, HERE/'build.py': args.builder_sha256,
        HERE/'BUILD_PROTOCOL.md': args.build_protocol_sha256}
    for p, h in sources.items():
        admit.pin(p, h)
    spec = importlib.util.spec_from_file_location('matched_score_stdlib_build_verifier', HERE/'build.py')
    builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
    env, proof = helpers(admit); result['helper_source_proof'] = proof
    bf, base, oldscore, refs = baseline(admit); original_control_checks(admit, env, bf, base, refs)
    acquisitions, lineage = {}, {}
    for arm in ARMS:
        folder, record, seals = arm_admission(args, arm, admit, env, refs, builder)
        acquisitions[arm] = (folder, record); lineage[arm] = seals
    require(not any(k in sys.modules for k in ('numpy', 'torch', 'flopscope', 'threadpoolctl')),
        'Complete source/byte admission without numerical import side effects')
    import numpy as np
    require(np.__version__ == '2.4.6', 'Original scoring precision/runtime')
    arrays = {'COMP': [env['trial_output'](np, bf, base['trials'][i], admit.pins) for i in (0, 1, 2, 3, 40)]}
    for arm, (folder, record) in acquisitions.items():
        arrays[arm] = [env['trial_output'](np, folder, t, admit.pins) for t in record['trials']]
    for role, values in arrays.items():
        require(values[0].tobytes() == values[-1].tobytes(), 'Own complete served repeat: '+role)
    for arm in ARMS:
        require(all(arrays[arm][i][0].tobytes() == arrays['COMP'][i][0].tobytes() for i in range(4)),
            'Untouched Gaussian first-layer bytes')
    index = read(INDEX)['records']; require(len(index) == 40 and len({r['rank'] for r in index}) == 40, 'Original40 truth index')
    expected = {row['rank']: row['target_sha256'] for row in index}
    require(all(expected[r] == oldscore['truth_bindings'][str(TRUTH/f'truthpost_rank{r:04d}.npz')] for r in RANKS),
        'Same original1e9 truth bytes as retained COMP scores')
    seal = out/'predictions_and_bills_sealed.json'
    write(seal, dict(truth_accessed=False, ranks=list(RANKS), all_ten_new_outputs_verified=True,
        all_native_bills_and_inputs_verified=True, repeat_excluded=True, own_complete_bills=True,
        baseline_native_sha256=BASE_SHA, baseline_score_sha256=SCORE_SHA, lineage=lineage,
        source_bindings=result['source_bindings'], input_bindings=dict(admit.pins)))
    result.update(all_predictions_verified_before_truth=True, lineage=lineage,
        artifacts={str(seal): sha(seal)}, predictions_and_bills_seal_sha256=sha(seal))
    records = []
    for call, rank in enumerate(RANKS):
        path = TRUTH/f'truthpost_rank{rank:04d}.npz'
        result['truth_accessed'] = True
        require(sha(path) == expected[rank], 'Original fixed1e9 target bytes')
        result['truth_bindings'][str(path)] = expected[rank]
        with np.load(path, allow_pickle=False) as archive:
            original = archive['mc_post']; dtype = str(original.dtype); target = original.astype(np.float64)
        require(np.issubdtype(original.dtype, np.floating) and target.shape == (16, 1024)
            and np.isfinite(target).all(), 'Finite original floating target frame')
        row = dict(call=call, rank=rank, target_sha256=expected[rank], target_original_dtype=dtype, per_role={})
        for role in ('COMP',)+ARMS:
            trial = base['trials'][call] if role == 'COMP' else acquisitions[role][1]['trials'][call]
            mse = np.mean((arrays[role][call].astype(np.float64)-target)**2, axis=1)
            raw = float(mse[-1]); f = trial['flops']; multiplier = max(.1, f/BUDGET)
            require(np.isfinite(mse).all() and (mse >= 0).all() and raw > 0 and multiplier == trial['multiplier'],
                'Finite actual raw paired with its own complete cost')
            row['per_role'][role] = dict(raw_mse=raw, flops=f, multiplier=multiplier,
                adjusted=raw*multiplier, layer_mse=mse.tolist())
        require(oldscore['records'][call]['rank'] == rank
            and row['per_role']['COMP'] == oldscore['records'][call]['per_role']['composition'],
            'Exact four-row original COMP score replay')
        row['comparisons'] = {}
        for left, right in (('joint', 'COMP'), ('p_only', 'COMP'), ('joint', 'p_only')):
            a, b = row['per_role'][left], row['per_role'][right]
            row['comparisons'][left+'_vs_'+right] = dict(raw_ratio=a['raw_mse']/b['raw_mse'],
                adjusted_ratio=a['adjusted']/b['adjusted'], raw_win=a['raw_mse'] < b['raw_mse'],
                adjusted_win=a['adjusted'] < b['adjusted'])
        records.append(row); guard()
    summary = {role: {key: sum(r['per_role'][role][key] for r in records)/4
        for key in ('raw_mse', 'flops', 'adjusted')} for role in ('COMP',)+ARMS}
    comparisons = {}
    for left, right in (('joint', 'COMP'), ('p_only', 'COMP'), ('joint', 'p_only')):
        a, b = summary[left], summary[right]
        wins = sum(r['per_role'][left]['adjusted'] < r['per_role'][right]['adjusted'] for r in records)
        improved = a['adjusted'] < b['adjusted']
        pareto = a['raw_mse'] <= b['raw_mse'] and a['flops'] <= b['flops'] and (a['raw_mse'] < b['raw_mse'] or a['flops'] < b['flops'])
        comparisons[left+'_vs_'+right] = dict(mean_adjusted_ratio=a['adjusted']/b['adjusted'],
            mean_raw_ratio=a['raw_mse']/b['raw_mse'], mean_complete_flops_ratio=a['flops']/b['flops'],
            adjusted_wins=wins, raw_wins=sum(r['per_role'][left]['raw_mse'] < r['per_role'][right]['raw_mse'] for r in records),
            mean_adjusted_improved=improved, at_least_three_wins=wins >= 3, gate_passed=improved and wins >= 3,
            classification='Pareto improvement' if pareto else 'beneficial tradeoff' if improved else 'not advanced')
    for path, digest in {**admit.pins, **result['truth_bindings'], **result['artifacts']}.items():
        require(sha(path) == digest, 'Final complete input/source/output/truth rehash'); guard()
    result.update(passed=True, completed=True, integrity_passed=True, hashes_unchanged=True, sources_unchanged=True,
        records=records, summary=summary, comparisons=comparisons,
        joint_learning_lead=comparisons['joint_vs_COMP']['gate_passed'] and comparisons['joint_vs_p_only']['gate_passed'],
        p_only_component=comparisons['p_only_vs_COMP']['gate_passed'],
        exact_COMP_score_replay=True, no_native_gradient_claim=True,
        aggregation='Mean of four per-net final raw*max(.1,own completeF/2^41); repeat excluded')


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', action='store_true')
    names = ['source', 'protocol', 'native-source', 'native-watch-source', 'native-protocol', 'builder', 'build-protocol']
    names += [arm+'-'+kind for arm in ('p-only', 'joint') for kind in ('build', 'candidate', 'native', 'watch')]
    for name in names:
        parser.add_argument('--'+name+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    require(all(re.fullmatch('[0-9a-f]{64}', getattr(args, n.replace('-', '_')+'_sha256') or '') for n in names),
        'Explicit reviewed source and completed artifact hashes')
    require(Path(sys.prefix).resolve() == (P2/'data_gate/.venv').resolve() and sys.flags.optimize == 0
        and sys.dont_write_bytecode and os.getpriority(os.PRIO_PROCESS, 0) >= 10
        and all(os.environ.get(k) == '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')), 'OFFPY nice10/BLAS1/bytecode0')
    began = time.monotonic(); out = HERE/'score_v1'; out.mkdir(exist_ok=False)
    result = dict(schema=SCHEMA, stage='score', passed=False, completed=False, integrity_passed=False,
        hashes_unchanged=False, sources_unchanged=False, source_bindings={}, input_bindings={}, truth_bindings={},
        truth_accessed=False, estimator_executed=False, training_or_refit_performed=False,
        MPS_score_used=False, official_score=False, production_qualified=False, independent_panel_qualified=False,
        automatic_next_stage=False, ranks=list(RANKS), repeat_excluded=True, requested_seals=vars(args))
    write(out/'started.json', result)
    def guard(*_):
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform == 'darwin' else 1024)
        require(time.monotonic()-began < 120 and rss < 1_000_000_000, 'Whole120s/1GB offline scoring bound')
        return rss
    signal.signal(signal.SIGALRM, guard); signal.setitimer(signal.ITIMER_REAL, .2, .2)
    try:
        require(not any(k in sys.modules for k in ('numpy', 'torch', 'flopscope', 'threadpoolctl')), 'No pre-admission numerical imports')
        score(args, out, result, guard)
        result['peak_rss_bytes'] = guard()
    except BaseException:
        result.update(passed=False, completed=False, integrity_passed=False, error=traceback.format_exc())
    result['elapsed_s'] = time.monotonic()-began
    write(out/'result.json', result); rh = sha(out/'result.json')
    ok = result['passed'] and time.monotonic()-began < 120
    write(out/'completion.json', dict(passed=ok, completed=ok, result_sha256=rh))
    eh = sha(out/'completion.json'); elapsed = time.monotonic()-began; ok = ok and elapsed < 120
    write(out/'finalization.json', dict(passed=ok, completed=ok, result_sha256=rh, completion_sha256=eh, elapsed_s=elapsed))
    signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps({k: result.get(k) for k in ('passed', 'summary', 'comparisons', 'joint_learning_lead', 'p_only_component', 'error')},
        allow_nan=False), flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
