"""Saved independent14 scoring only; no estimator, fitting or release authority."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import sys
import textwrap
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
APUSH = NEW/'current_comp_aligned_push_d6_v1'
OFFPY = P2/'data_gate/.venv/bin/python'
PUBLIC = (35,100,150,200,250,300,400,500)
VIRGIN = (3016500,3016501,3016502,3016503,3016504,3016505)
RANKS, SEQUENCE = PUBLIC+VIRGIN, PUBLIC+VIRGIN+(35,)
BUDGET, INIT, CAP, RSS = 2**41, 1944970298, 120., 2_000_000_000
CANDIDATE = '4d98fef2c11c571efe1eca945018bb9fe9ece1454d16b4abde2eb52d71e755c3'
PARENT = '33879ab58f3a5e4e1ad41deb5db21b73ba703438374225d30fd0ebfc7511b89a'
SCHEMA = 'current_comp_shared_old384_score_panel_v1'
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')
KEYS = ('source','protocol','watcher','watch-spec','native-source','native-watch-source',
        'native-protocol','native-watch-spec','native','native-watch',
        'native40','native40-watch','score40','score40-watch')
BASE = HERE/'score_probe.py'
TL = NEW/'current_d50_terminal_linear_v2/panel.py'
RC = NEW/'current_d50_response10_candidate_v1/score.py'
CENSUS = NEW/'current_d50_response10_census_v1/watch.py'
JOINT = NEW/'current_comp_joint_p_node_v1/score.py'
TRUTH_INDEX = NEW/'response_rows_panel_v1/score/result.json'
OLD_NATIVE = '8be856a31f70a7878286e42b1abf5c6bb71979b5a29853f870c7e00d8e96c733'
OLD_NATIVE_WATCH = 'e2d8f13623d507b0369331f079df50524b53369df3ffe527caee2009ec632828'
OLD_SCORE = '5cdd536b2bd055a177c310c0ec7fcc61b2f68448ef844bd3c32a21ba9a9e0abb'
OLD_SCORE_WATCH = '84ecb95216cf1ce93554f9f4c3522f20ef3309a9d7eedcf706e70157c2f44dda'
PINS = {
    BASE:'a279e9bf196c50ef3157ade817946f5f6a5f75521040f0ce03820649a8c1e7cb',
    TL:'0a004bb746a0cdee4345a1b7e5621c2048ef8cfe8e6b46dd2036adb2b581da19',
    RC:'3bc2aeb4d66a74984c2801c281a52e11b75a52f36d9bff5c877cca3daccbf2b4',
    CENSUS:'7f868ba0250f142822b2aa36503d1908b21849c47e6e7406320f979e5d93bbcf',
    JOINT:'7d4f6b3f919f73bce749b7a109bd5276e28cc79152355d54faad3be3c6bb5b7e',
    HERE/'check4_v2.py':'8d09d9b1f443e229b1641d277312b52c204392648b25445ffab63b82dba16058',
    HERE/'score40.py':'b99c9fef43da28a5f84ba6769d4d0f7b378480862752d0784ceb28bd8378fde7',
    HERE/'watch_score40.py':'5c110483447bcd1b5d8720429fa88038effe9ad929e14db73d238bb2d491a896',
    HERE/'SCORE40_PROTOCOL.md':'8a2c13dd68b50fcd962a9c917bb96d7e7c2c0182f92c1725ceeb5308b2022068',
    HERE/'native40.py':'0ac42cceec2bcba30813536deb33906a30cc2873ae99e5799d74c4c5e2a3472a',
    HERE/'watch_native40.py':'c4536d3c237f393efe5cef180057b4e72650c31e2d8589411ea3bb022fabd98f',
    HERE/'NATIVE40_PROTOCOL.md':'8dbf45273f4835e7dc60d942b59b9fa95143420b7e8ef752f4e91e2966734599',
    HERE/'NATIVE40_WATCH_SPEC.md':'fcf0104e3d9abfdffd9fa68ca85e5a25fbf1bc3f3ac88291b355bd0c4fa4beea',
    HERE/'compose.py':'fdf5bd87eb6411d008043f5e989a670979cde20965366dca212146c075ee1d4e',
    HERE/'fnp.py':'980853fe0b794efcf046b360ea03bbfaa671bd48383312e8e766c9b546021d81',
    HERE/'PROTOCOL.md':'ef2331d9adfe20c8bdea5c173d9f2d19ec523b85bc0cde4a653c93ead4351607',
    APUSH/'panel_score.py':'097b9d4fcccbaaaea3d5dfd93fa455fb06796678a2039d415a020d339db2a2db',
    APUSH/'panel_common.py':'b8ae22e9586f76cafd92763562d2a631efae6a8ea81d2417e6cad5ce6828129a',
    APUSH/'panel_watch_score.py':'4a424836ae3707d0be0c314bb428a6e372da0d45a1790b62b935def0dd9ef99e',
    TRUTH_INDEX:'a31981681305de1828c48d8b6e8830c87ed979789c7d600b949217814f479cdd',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size < 16*1024**2,
            'Bounded regular metadata')
    return json.loads(path.read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')


def truth_path(rank):
    require(rank in RANKS, 'Only original independent14 truth roster')
    return P2/'public_g19_attribution_v1'/(f'mps_truth_rank{rank:04d}_indep4M.npz'
        if rank in PUBLIC else f'fresh_{rank}_truth4M.npz')


def is_truth(path):
    path = Path(path)
    return 'truthpost' in path.parts or (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower())


def sha(path, tick=None, *, truth=False):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(P2), 'Regular in-scope file')
    require(not is_truth(path) or (truth and path in {truth_path(r) for r in RANKS}),
            'Target arrays forbidden outside explicitly allowlisted post-seal access')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            h.update(block)
            if tick is not None:
                tick()
    return h.hexdigest()


def extract(path, name):
    source = Path(path).read_text()
    nodes = [n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name == name]
    require(len(nodes) == 1, 'Unique exact source owner '+name)
    n = nodes[0]
    return ''.join(source.splitlines(keepends=True)[n.lineno-1:n.end_lineno])


def support():
    """Only exact stdlib definitions; no ancestor main/admission or estimator import."""
    for path in (BASE,RC,CENSUS,JOINT,HERE/'check4_v2.py'):
        require(sha(path) == PINS[path], 'Exact selected helper owner')
    names = ('Admission','linked','helpers','diagnostics')
    source = '\n\n'.join(extract(BASE,n) for n in names)
    source += '\n\n'+extract(HERE/'check4_v2.py','_v2_validate_ledgers')
    env = dict(globals())
    exec(compile(source,'<exact-SHARED-saved-metadata-helpers>','exec'),env)
    return env,dict(selected_helper_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        definitions_byte_exact=True,no_ancestor_main_or_admission=True)


def arithmetic_source():
    """Entire original independent14 numeric/summary body, role labels only."""
    require(sha(TL) == PINS[TL], 'Exact qualified original independent14 arithmetic')
    original = extract(TL,'score')
    start = original.index('    for call,rank in enumerate(p.RANKS):\n')
    stop = original.index("    require(all(sha(path) == digest for path,digest in pins.items()),'End source/output/bill seals')",start)
    original = textwrap.dedent(original[start:stop])
    edits = [("'current_d50'","'APUSH'"),("'terminal'","'shared'"),("['candidate']","['terminal']")]
    changed = original
    for before,after in edits:
        require(before in changed, 'Original scoring label '+before)
        changed = changed.replace(before,after)
    inverse = changed
    for before,after in reversed(edits):
        inverse = inverse.replace(after,before)
    require(inverse == original, 'Full independent14 arithmetic inverse, labels only')
    return changed,dict(owner_sha256=PINS[TL],whole_numeric_body_inverse=True,
        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        FP64_original_truth_mean_and_SE=True,unclipped_noise_subtraction=True,mean_of_own_products=True)


def actual_chain(admit, folder, digest, cap, rss, *, legacy=None, watched=False):
    """Actual legacy fields are explicit, never invented modern compatibility fields."""
    if legacy is None:
        return admit.chain(folder,digest,cap,rss,watch=watched)
    paths = [folder/n for n in ('result.json','completion.json','finalization.json')]
    hashes = [digest]+[sha(p,admit.tick) for p in paths[1:]]
    for p,h in zip(paths,hashes):
        admit.pin(p,h)
    r,e,f = map(read,paths)
    require(all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
        and e['result_sha256'] == f['result_sha256'] == digest and f['completion_sha256'] == hashes[1]
        and 0 <= f['elapsed_s'] < cap and r['hashes_unchanged'] and r['sources_unchanged'], 'Actual APUSH three seals')
    if watched:
        require(all(r[k] is True for k in ('whole_owned_tree_retired','observational_groups_retired',
            'actual_completed_samples','foreign_numerical_workers_checked')) and r['numerical_imports'] is False
            and r['maximum_swap_growth_bytes'] == 0 and 0 < r['maximum_sampled_rss_bytes'] < rss
            and 0 < r['maximum_sample_interval_s'] <= 1 and 0 <= r['elapsed_s'] < cap
            and r['experiment_groups_retired'] and r['child_execution_completed'],
            'Original actual APUSH sampled resources and retired groups')
    elif legacy == 'score':
        require(0 <= r['wall_s'] < cap and 0 < r['peak_rss_bytes'] < rss, 'Original actual APUSH score resources')
    else:
        require(legacy == 'native' and all(0 < t['peak_rss_bytes'] < rss and 0 <= t['wall_s'] < 120
            and 0 <= t['context_wall_s'] < 120 for t in r['trials']), 'Original actual APUSH per-call resources')
    return r,dict(zip(('result_sha256','completion_sha256','finalization_sha256'),hashes))


def baseline(admit, env):
    folder = APUSH/'panel_v1'
    n,ns = actual_chain(admit,folder/'terminal',OLD_NATIVE,1200,8_000_000_000,legacy='native')
    w,ws = actual_chain(admit,folder/'terminal_watch',OLD_NATIVE_WATCH,1200,8_000_000_000,legacy='native',watched=True)
    s,ss = actual_chain(admit,folder/'score',OLD_SCORE,120,2_000_000_000,legacy='score')
    sw,sws = actual_chain(admit,folder/'score_watch',OLD_SCORE_WATCH,120,2_000_000_000,legacy='score',watched=True)
    env['linked'](w,ns); env['linked'](sw,ss)
    require(n['schema'] == 'current_comp_aligned_push_d6_independent14_native_v1'
        and s['schema'] == 'current_comp_aligned_push_d6_independent14_score_v1'
        and n['candidate_sha256'] == s['candidate_sha256'] == PARENT
        and n['full_model_calls'] == n['retired_call_count'] == 15 and n['initialization_flops'] == INIT
        and len(n['artifacts']) == 90 and [t['rank'] for t in n['trials']] == list(SEQUENCE)
        and all(n[k] is True for k in ('all16_domains_qualified','all47_selectors_qualified','own_complete_bills',
            'repeat_output_domain_byte_exact')) and s['integrity_passed'] and s['repeat_excluded']
        and s['all_predictions_verified_before_truth'] and len(s['per_role']['terminal']) == 14
        and set(s['truth_bindings']) == {str(r) for r in RANKS}, 'Direct completed APUSH panel boundary, same frame')
    # The completed APUSH ancestry is a trust boundary. Rehash its actual15/90 and fixtures,
    # not all ancestor training/inference artifacts or any truth arrays.
    admit.bind(n['artifacts'])
    for binding in n['fixture_bindings'].values():
        admit.pin(binding['path'],binding['sha256'])
    for record in (s,w,sw):
        admit.bind(record['artifacts'])
    return n,s,dict(APUSH_native=ns,APUSH_native_watch=ws,APUSH_score=ss,APUSH_score_watch=sws)


def qualified40(args, admit, env):
    r,rs = actual_chain(admit,HERE/'native40_v1',args.native40_sha256,3000,8_000_000_000)
    w,ws = actual_chain(admit,HERE/'native40_watch_v1',args.native40_watch_sha256,3000,8_000_000_000,watched=True)
    s,ss = actual_chain(admit,HERE/'score40_v1',args.score40_sha256,120,1_000_000_000)
    sw,sws = actual_chain(admit,HERE/'score40_watch_v1',args.score40_watch_sha256,120,1_000_000_000,watched=True)
    env['linked'](w,rs); env['linked'](sw,ss)
    require(r['schema'] == 'current_comp_shared_old384_native40_v1'
        and s['schema'] == 'current_comp_shared_old384_score40_v1'
        and r['candidate_sha256'] == CANDIDATE and r['full_model_calls'] == r['retired_call_count'] == 41
        and len(r['artifacts']) == 287 and len(s['records']) == 40 and s['information_promising']
        and s['scientific_gate_passed'] and sw['information_promising']
        and s['comparison']['adjusted_ratio'] <= .97 and s['exact_APUSH_score_replay']
        and s['lineage']['probe'] == rs and s['lineage']['probe_watch'] == ws,
        'Separate actual native40 <=.97 advancement prerequisite, never guessed or rescored')
    for record in (r,w,s,sw):
        admit.inventory(record)
    require(all(r['source_bindings'].get(str(p)) == w['source_bindings'].get(str(p)) == PINS[p]
        for p in (HERE/'native40.py',HERE/'watch_native40.py',HERE/'NATIVE40_PROTOCOL.md',HERE/'NATIVE40_WATCH_SPEC.md'))
        and all(s['source_bindings'].get(str(p)) == PINS[p]
        for p in (HERE/'score40.py',HERE/'watch_score40.py',HERE/'SCORE40_PROTOCOL.md')), 'Exact reviewed40 source boundary')
    return dict(native40=rs,native40_watch=ws,score40=ss,score40_watch=sws)


def verify_calls(record, folder, admit, env, *, shared, watch=None):
    require(len(record['trials']) == len(record['input_preparation']) == len(record['diagnostic_records']) == 15
        and len(record['artifacts']) == (105 if shared else 90), 'Exact complete per-call artifact cardinality')
    for call,(rank,t,p,dm) in enumerate(zip(SEQUENCE,record['trials'],record['input_preparation'],record['diagnostic_records'])):
        bp,ip,rp,dp = (folder/f'call_{call:02d}_{n}.json' for n in ('bill','input_bill','input_reset','diagnostics'))
        require(t['call'] == p['call'] == call and t['rank'] == p['rank'] == rank
            and t['includes_initialization'] is (call == 0) and read(bp) == t and read(ip) == p
            and type(t['flops']) is int and 0 < t['flops'] <= BUDGET and t['multiplier'] == max(.1,t['flops']/BUDGET)
            and 0 <= t['wall_s'] < 120 and 0 <= t['context_wall_s'] < 120 and 0 < t['peak_rss_bytes'] < 8_000_000_000
            and t['input_bill'] == str(ip) and t['input_bill_sha256'] == sha(ip,admit.tick)
            and t['input_reset_receipt'] == str(rp) and t['input_reset_sha256'] == sha(rp,admit.tick), 'Own complete physical bill')
        env['lifecycle'](t,bp,Path(t['lifecycle_receipt']))
        env['lifecycle']({**p,'lifecycle_receipt':str(rp)},ip,rp)
        fixture = record['fixture_bindings'][f'weights_{rank}']
        require(p['weight_file_sha256'] == fixture['sha256'] and p['flops'] == 16777216
            and p['operation_count'] == p['layers'] == 16 and p['charged_to_participant'] is False
            and all(p[k] is True for k in ('official_fnp_type','exact_bytes','independent_weight_buffers','context_closed')),
            'Exact independent FNP input bytes; input preparation not borrowed or subtracted')
        require(dm == dict(path=str(dp),sha256=sha(dp,admit.tick)), 'Own diagnostic hash')
        packet = read(dp); require(packet['call'] == call and packet['rank'] == rank, 'Own diagnostic identity')
        env['diagnostics'](packet,call == 0)
        if shared:
            mp = folder/f'call_{call:02d}_complete.json'; m = read(mp)
            require(m['passed'] and m['completed'] and m['context_retired'] and m['call'] == call and m['rank'] == rank
                and m['bill_sha256'] == sha(bp,admit.tick) and m['lifecycle_sha256'] == sha(t['lifecycle_receipt'],admit.tick)
                and m['diagnostic_sha256'] == sha(dp,admit.tick) and m['output_sha256'] == t['output_sha256']
                and watch['markers'][call] == dict(call=call,rank=rank,path=str(mp),sha256=sha(mp,admit.tick)),
                'Actual post-release marker, no heartbeat or inferred model completion')
        admit.tick()
    require(record['trials'][0]['flops']-record['trials'][-1]['flops'] == INIT,
        'Each backend owns its actual cold initialization exactly once')
    if shared:
        require(env['_v2_validate_ledgers'](record), 'Actual post-context operation ledgers, not rebased counts')


def acquisition(args, admit, env, control, control_seals):
    n,ns = actual_chain(admit,HERE/'native_panel_v1',args.native_sha256,1200,8_000_000_000)
    w,ws = actual_chain(admit,HERE/'native_panel_watch_v1',args.native_watch_sha256,1200,8_000_000_000,watched=True)
    env['linked'](w,ns); admit.inventory(n); admit.inventory(w)
    require(n['schema'] == 'current_comp_shared_old384_native_panel_v1'
        and w['schema'] == 'current_comp_shared_old384_native_panel_watch_v1'
        and n['candidate_sha256'] == w['candidate_sha256'] == CANDIDATE
        and n['ranks'] == w['ranks'] == list(SEQUENCE)
        and n['full_model_calls'] == n['retired_call_count'] == w['retired_calls'] == 15
        and n['initialization_flops'] == INIT and n['truth_accessed'] is False and n['score_measured'] is False
        and n['parameter_updates'] == 0 and n['controls'] == control['trials']
        and n['fixture_bindings'] == control['fixture_bindings']
        and n['control_seals'] == dict(native=list(control_seals['APUSH_native'].values()),
            watch=list(control_seals['APUSH_native_watch'].values()))
        and all(n[k] is True for k in ('all16_domains_qualified','all47_selectors_qualified','own_complete_bills',
            'repeat_byte_exact','operation_ledgers_verified','operation_counts_reported_not_rebased',
            'native40_expansion_prerequisite_passed','APUSH_panel_comparator_qualified',
            'panel_repeat35_physics_exact','production_same_input_controls_sealed')),
        'Fixed same-candidate positive15 acquisition and actual APUSH control binding')
    required = {HERE/'native_panel.py':args.native_source_sha256,HERE/'watch_native_panel.py':args.native_watch_source_sha256,
        HERE/'NATIVE_PANEL_PROTOCOL.md':args.native_protocol_sha256,HERE/'NATIVE_PANEL_WATCH_SPEC.md':args.native_watch_spec_sha256,
        **{p:PINS[p] for p in (HERE/'compose.py',HERE/'fnp.py',HERE/'PROTOCOL.md')}}
    require(all(n['source_bindings'].get(str(p)) == w['source_bindings'].get(str(p)) == h for p,h in required.items()),
        'Actual reviewed acquisition, passive observer and unchanged physical composition owners')
    proof = n['composition_proof']
    observer = n['passive_observer_source_proof']
    require(proof == w['composition_proof'] and proof['composed_source_sha256'] == CANDIDATE
        and proof['parent_sha256'] == PARENT and proof['rank'] == 384
        and proof['first_join_layer'] == 5 and proof['last_join_layer'] == 14
        and proof['weight_rule'] == 'actual_FP32_lambda_squared' and proof['full_source_inverse_verified']
        and all(proof[k] is True for k in ('physical_function_inverse_verified','learned_literals_and_imports_byte_exact',
            'original_BD_stored_pair_repair_once','original_all_row_diagonal_source_unchanged',
            'original_omitted_radial_source_unchanged','parent_arena_and_dispatchers_unchanged','no_import_time_helper_array_work'))
        and proof['fitted'] is False and proof['extra_incumbent_plane_repair'] is False
        and n['paid_loop_proof']['whole_loop_inverse']
        and n['paid_loop_proof']['no_billing_input_reset_materialization_or_repeat_change']
        and n['paid_loop_proof']['owner_sha256'] == '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
        and all(observer[k] is True for k in ('broad_push_observer_inverse_verified','unchanged_all16_all47_capture',
            'exact_shape_specific_D6_and_seven_D5_tariffs','no_added_participant_array_operations')),
        'Same frozen one representation and complete paid loop; no new science')
    verify_calls(n,HERE/'native_panel_v1',admit,env,shared=True,watch=w)
    final_source = extract(HERE/'native_panel.py','panel_final_checks')
    final_env = dict(env,SEQUENCE=SEQUENCE)
    exec(compile(final_source,'<exact-retired-panel-metadata-checks>','exec'),final_env)
    final = final_env['panel_final_checks'](n,admit.tick)
    require(all(n[k] == value for k,value in final.items()), 'Exact whole panel repeat/ledger/production correspondence owner')
    return n,dict(native=ns,native_watch=ws)


def own_sources(args):
    return {**PINS,HERE/'score_panel.py':args.source_sha256,HERE/'watch_score_panel.py':args.watcher_sha256,
        HERE/'SCORE_PANEL_PROTOCOL.md':args.protocol_sha256,HERE/'SCORE_PANEL_WATCH_SPEC.md':args.watch_spec_sha256,
        HERE/'native_panel.py':args.native_source_sha256,HERE/'watch_native_panel.py':args.native_watch_source_sha256,
        HERE/'NATIVE_PANEL_PROTOCOL.md':args.native_protocol_sha256,HERE/'NATIVE_PANEL_WATCH_SPEC.md':args.native_watch_spec_sha256}


def watch_setup(args):
    """Pure metadata interface for the unchanged external resource-loop adapter."""
    child = {k.replace('-','_')+'_sha256':getattr(args,k.replace('-','_')+'_sha256')
        for k in KEYS if k not in ('source','watcher')}
    child.update(source_sha256=args.worker_sha256,watcher_sha256=args.source_sha256)
    ns = SimpleNamespace(**child)
    own = own_sources(ns)
    inputs = {HERE/'native_panel_v1/result.json':args.native_sha256,
        HERE/'native_panel_watch_v1/result.json':args.native_watch_sha256,
        HERE/'native40_v1/result.json':args.native40_sha256,
        HERE/'native40_watch_v1/result.json':args.native40_watch_sha256,
        HERE/'score40_v1/result.json':args.score40_sha256,
        HERE/'score40_watch_v1/result.json':args.score40_watch_sha256}
    return own,inputs,HERE/'score_panel_v1',HERE/'score_panel_watch_v1'


def watch_contract(args, result, folder, check_sha, tick):
    flags = ('integrity_passed','hashes_unchanged','sources_unchanged','complete_panel_acquisition_reverified',
        'all30_outputs_verified','all105_new_artifacts_verified','all90_APUSH_artifacts_verified',
        'all16_domains_and_all47_selectors_reverified','own_complete_bills','all_predictions_verified_before_truth',
        'exact_APUSH_score_replay','exact_APUSH_three_summary_replay','repeated_outputs_byte_exact',
        'no_model_imported','metadata_fresh_union_rehashed','native40_gate_passed')
    require(all(result.get(k) is True for k in flags) and result['schema'] == SCHEMA
        and result['candidate_sha256'] == CANDIDATE and result['ranks'] == list(RANKS)
        and result['full_model_calls'] == result['parameter_updates'] == 0 and result['repeat_excluded']
        and result['estimator_executed'] is False and result['training_or_refit_performed'] is False
        and result['submission_authorized'] is False and result['automatic_next_stage'] is False,
        'Exactly14 saved scores, no extra experiment or release authority')
    ratios = result['adjusted_candidate_over_parent']
    expected = {g:ratios[g] <= .97 for g in ('public8','virgin6','combined14')}
    both = expected['public8'] and expected['virgin6']
    below = result['summaries']['shared']['combined14']['mean_adjusted_noise_corrected'] < 4e-9
    require(result['group_three_percent_gain_gate'] == expected
        and result['both_panels_three_percent_gain'] is both
        and result['combined_noise_corrected_adjusted_below_4e9'] is below
        and type(result['information_promising']) is bool
        and result['information_promising'] == result['scientific_gate_passed'] == result['bank_scientific_gate_passed'] == (both and below),
        'Both independent groups >=3% and combined strictly<4e-9; scientific status is separate')
    prior = APUSH/'panel_v1/score/result.json'
    require(check_sha(prior,tick) == OLD_SCORE and result['truth_bindings'] == read(prior)['truth_bindings']
        and result['truth_accessed'] is True, 'Exactly the original14 truth bindings; no additional labels')
    seal = folder/'predictions_and_bills_sealed.json'; s = read(seal)
    require(check_sha(seal,tick) == result['predictions_and_bills_seal_sha256'] == result['artifacts'][str(seal)]
        and s['truth_accessed'] is False and s['all30_outputs_verified'] and s['all105_new_artifacts_verified']
        and s['all90_APUSH_artifacts_verified'] and s['own_complete_bills'] and s['lineage'] == result['lineage']
        and s['source_bindings'] == result['source_bindings'] and s['input_bindings'] == result['input_bindings'],
        'Actual all-output/bill seal precedes all14 truth accesses')


def score(args, out, result, tick):
    env,proof = support(); admit = env['Admission'](result,tick)
    admit.bind({str(p):h for p,h in own_sources(args).items()})
    call_helpers,call_proof = env['helpers'](admit); env.update(call_helpers)
    result.update(helper_source_proof=proof,saved_call_helper_proof=call_proof)
    prerequisite = qualified40(args,admit,env)
    base,old,lineage = baseline(admit,env)
    n,seals = acquisition(args,admit,env,base,lineage); lineage.update(seals,development=prerequisite)
    require(n['native40_prerequisite_seals'] == prerequisite, 'Same actual full40 advancement authority in score and acquisition')
    verify_calls(base,APUSH/'panel_v1/terminal',admit,env,shared=False)
    source,numeric_proof = arithmetic_source(); result['arithmetic_source_proof'] = numeric_proof
    require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules), 'Complete metadata admission before numerical imports')
    import numpy as np
    require(np.__version__ == '2.4.6', 'Original NumPy scoring version')
    outputs = {}
    for role,record,folder in (('APUSH',base,APUSH/'panel_v1/terminal'),('shared',n,HERE/'native_panel_v1')):
        outputs[role] = [env['trial_output'](np,folder,t,admit.pins) for t in record['trials']]
        require(outputs[role][0].tobytes() == outputs[role][-1].tobytes(), 'Every native repeat fully byte-exact')
    require(all(outputs['APUSH'][i][:5].tobytes() == outputs['shared'][i][:5].tobytes() for i in range(15)),
        'Actual unaffected first5 layers exactly reproduce APUSH for all15 calls')
    require(sum(a.nbytes for group in outputs.values() for a in group) == 30*16*1024*4, 'Exactly30 saved FP32 output arrays')
    seal = out/'predictions_and_bills_sealed.json'
    write(seal,dict(truth_accessed=False,all30_outputs_verified=True,all105_new_artifacts_verified=True,
        all90_APUSH_artifacts_verified=True,own_complete_bills=True,ranks=list(RANKS),repeat_excluded=True,
        lineage=lineage,source_bindings=result['source_bindings'],input_bindings=dict(admit.pins)))
    result['artifacts'][str(seal)] = sha(seal,tick)
    result.update(all_predictions_verified_before_truth=True,predictions_and_bills_seal_sha256=sha(seal,tick),
        lineage=lineage,fixture_bindings=n['fixture_bindings'],per_role={'APUSH':[],'shared':[]})
    index = read(TRUTH_INDEX)
    require(index['qualification_passed'] and index['hashes_unchanged']
        and set(index['truth_bindings']) == {str(r) for r in RANKS}, 'Original fixed independent4M truth index')
    result['truth_accessed'] = True
    for rank in RANKS:
        binding = old['truth_bindings'][str(rank)]; path = truth_path(rank)
        require(Path(binding['path']) == path and index['truth_bindings'][str(rank)]['sha256'] == binding['sha256']
            and sha(path,tick,truth=True) == binding['sha256'], 'Same original independent4M truth bytes after complete seal')
        result['truth_bindings'][str(rank)] = dict(binding)
    scope = dict(np=np,p=SimpleNamespace(PUBLIC=PUBLIC,VIRGIN=VIRGIN,RANKS=RANKS),h=SimpleNamespace(BUDGET=BUDGET),
        acquisition=n,context=dict(control=base,control_score=old),outputs=outputs,result=result,guard=tick,
        require=require,math=math,json=json)
    exec(compile(source,'<exact-independent14-APUSH-role-arithmetic>','exec'),scope)
    summaries,ratios = scope['summaries'],scope['ratios']
    gates = {g:ratios[g] <= .97 for g in ('public8','virgin6','combined14')}
    both = gates['public8'] and gates['virgin6']
    below = summaries['shared']['combined14']['mean_adjusted_noise_corrected'] < 4e-9
    for path,digest in {**admit.pins,**result['artifacts']}.items():
        require(sha(path,tick) == digest, 'Fresh final complete non-target union')
    for binding in result['truth_bindings'].values():
        require(sha(binding['path'],tick,truth=True) == binding['sha256'], 'Original truth bytes unchanged')
    require(not {'torch','flopscope','threadpoolctl'} & set(sys.modules), 'NumPy-only offline scoring, no model imports')
    result.update(passed=True,completed=True,integrity_passed=True,hashes_unchanged=True,sources_unchanged=True,
        summaries=summaries,adjusted_candidate_over_parent=ratios,group_three_percent_gain_gate=gates,
        both_panels_three_percent_gain=both,combined_noise_corrected_adjusted_below_4e9=below,
        bank_scientific_gate_passed=both and below,information_promising=both and below,scientific_gate_passed=both and below,
        exact_APUSH_score_replay=True,exact_APUSH_three_summary_replay=True,all30_outputs_verified=True,
        all105_new_artifacts_verified=True,all90_APUSH_artifacts_verified=True,complete_panel_acquisition_reverified=True,
        all16_domains_and_all47_selectors_reverified=True,own_complete_bills=True,repeated_outputs_byte_exact=True,
        no_model_imported=True,metadata_fresh_union_rehashed=True,native40_gate_passed=True,
        aggregation='Mean per-net unclipped (final observed MSE-mean(mc_se15**2))*max(.1,own complete F/2**41).',
        frame=old['frame'],production_fidelity_pending=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run',action='store_true')
    for key in KEYS:
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    began = time.monotonic()
    require(all(re.fullmatch('[0-9a-f]{64}',getattr(args,k.replace('-','_')+'_sha256') or '') for k in KEYS), 'Actual reviewed source/result hashes')
    require(Path(sys.executable).resolve() == OFFPY.resolve() and sys.dont_write_bytecode and sys.flags.optimize == 0
        and os.getpriority(os.PRIO_PROCESS,0) >= 10 and all(os.environ.get(k) == '1' for k in BLAS)
        and not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules)
        and importlib.metadata.version('numpy') == '2.4.6', 'Original OFFPY nice10/sixBLAS1; no pre-admission numerical imports')
    out = HERE/'score_panel_v1'; out.mkdir(exist_ok=False)
    result = dict(schema=SCHEMA,passed=False,completed=False,integrity_passed=False,source_bindings={},input_bindings={},
        artifacts={},truth_bindings={},truth_accessed=False,ranks=list(RANKS),repeat_excluded=True,candidate_sha256=CANDIDATE,
        parent_sha256=PARENT,full_model_calls=0,parameter_updates=0,estimator_executed=False,training_or_refit_performed=False,
        production_qualified=False,submission_authorized=False,uploaded=False,automatic_next_stage=False,
        information_promising=False,bank_scientific_gate_passed=False,requested_seals=vars(args))
    def guard(*unused):
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        require(time.monotonic()-began < CAP and value < RSS, 'Whole120s/2GB saved panel scoring')
        return value
    write(out/'started.json',result)
    signal.signal(signal.SIGALRM,guard); signal.setitimer(signal.ITIMER_REAL,.2,.2)
    try:
        score(args,out,result,guard); guard()
    except BaseException:
        result.update(passed=False,completed=False,integrity_passed=False,error=traceback.format_exc())
    signal.setitimer(signal.ITIMER_REAL,0)
    result.update(elapsed_s=time.monotonic()-began,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    ok = bool(result['passed'] and result['elapsed_s'] < CAP and result['peak_rss_bytes'] < RSS)
    result['passed'] = result['completed'] = ok
    write(out/'result.json',result); rh = sha(out/'result.json'); ok = bool(ok and time.monotonic()-began < CAP)
    write(out/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh)); ch = sha(out/'completion.json')
    elapsed = time.monotonic()-began; ok = bool(ok and elapsed < CAP)
    write(out/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,completion_sha256=ch,elapsed_s=elapsed))
    sha(out/'finalization.json'); ok = bool(ok and time.monotonic()-began < CAP)
    print(json.dumps({k:result.get(k) for k in ('passed','bank_scientific_gate_passed','summaries','error')},allow_nan=False),flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
