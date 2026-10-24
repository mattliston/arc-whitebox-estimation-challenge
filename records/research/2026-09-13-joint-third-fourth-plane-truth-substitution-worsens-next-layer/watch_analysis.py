"""DRAFT: independent 60s/1GB watch for the fixed sealed-cell analysis.

Only JSON metadata and explicitly admitted immutable bytes are read here.
The numerical worker owns the fixed analysis; a scientific negative is not
an operational failure. No estimator, fit, new control, or automatic stage.
"""
import hashlib
from pathlib import Path
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
OWNER = NEW/'current_comp_shared_old384_v1/watch_probe.py'
OWNER_SHA = '7ade1c9025a2932e7929498062f894b72a3817d50027e7c5d1cfbf22a1f0fd18'
CANDIDATE_SHA = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
CONSUMER_ROSTER = ('tiny','N','T','Q','TQ','ALL-local')
KEYS = ('source','worker','protocol','spec','consumer-source','consumer-protocol',
        'consumer-watch-source','consumer-watch-spec','consumer','consumer-watch')
B = NEW/'covariance_replica_truth_v1/rank_0035_8M_v1.npz'
ARCHIVES = {
    P2/'public_g19_attribution_v1/mps_cov2_rank0035.npz':
        '7e6b97166885de0cdd870a835ae536830fb6da41ca345d8a66e2eb8d0d6380e6',
    P2/'public_g19_attribution_v1/mps_planes_rank0035.npz':
        '64bbf8f8b17115252a186535c45a15295a0869135a547faab04242fb72453398',
    B:'e6a7787d974c6b56f39a4444bd54c793ff3efa6e62acb55054774ddb2a2e3358'}
B_BYTE_ALLOWED = False


def worker_namespace(args):
    """Exactly the namespace parsed from child_command's ten SHA flags."""
    return SimpleNamespace(source_sha256=args.worker_sha256,protocol_sha256=args.protocol_sha256,
        watcher_sha256=args.source_sha256,watch_spec_sha256=args.spec_sha256,
        consumer_source_sha256=args.consumer_source_sha256,
        consumer_protocol_sha256=args.consumer_protocol_sha256,
        consumer_watch_source_sha256=args.consumer_watch_source_sha256,
        consumer_watch_spec_sha256=args.consumer_watch_spec_sha256,
        consumer_result_sha256=args.consumer_sha256,consumer_watch_sha256=args.consumer_watch_sha256)


def child_command(args, python):
    values = vars(worker_namespace(args))
    command = [str(python),'-B','-u',str(HERE/'analysis.py'),'--run']
    for key,value in values.items():
        command += ['--'+key.replace('_','-'),value]
    return command


def path_allowed(path):
    if not path.is_relative_to(P2) or path.name == 'key.txt' or 'truthpost' in path.parts:
        return False
    if path.suffix == '.npz':
        return path in ARCHIVES and (path != B or B_BYTE_ALLOWED)
    return not (path.suffix == '.npy' and 'truth' in str(path).lower())


def merge(target, values, require):
    require(isinstance(values,dict), 'Flat immutable binding map')
    for path,digest in values.items():
        require(path not in target or target[path] == digest, 'No contradictory authority')
        target[path] = digest


def consumer_gate(read, sha, require, args, context, result, tick):
    """Independent direct positive six-arm seal/resource boundary, before spawn."""
    records,chains = [],[]
    for folder,key in (('consumer_v2','consumer_sha256'),('consumer_watch_v2','consumer_watch_sha256')):
        paths = [HERE/folder/n for n in ('result.json','completion.json','finalization.json')]
        hashes = [sha(p,tick) for p in paths]
        r,e,f = [read(p) for p in paths]
        require(hashes[0] == getattr(args,key) and all(x.get('passed') is True and
            x.get('completed') is True for x in (r,e,f)) and e['result_sha256'] == f['result_sha256'] == hashes[0]
            and f['completion_sha256'] == hashes[1], 'Actual positive linked consumer three seals')
        merge(result['input_bindings'],{str(p):h for p,h in zip(paths,hashes)},require)
        records.append((r,f)); chains.append(hashes)
    (r,rf),(w,wf) = records
    require(r['schema'] == 'current_comp_joint_cell_consumer_v2'
        and w['schema'] == 'current_comp_joint_cell_consumer_watch_v2'
        and r['candidate_sha256'] == w['candidate_sha256'] == CANDIDATE_SHA
        and r['roster'] == w['roster'] == list(CONSUMER_ROSTER) and r['ranks'] == w['ranks'] == [35]*6
        and r['retired_call_count'] == w['retired_calls'] == 6
        and [w['child_'+k] for k in ('result_sha256','completion_sha256','finalization_sha256')] == chains[0]
        and context['consumer_seals'] == chains[0] and context['consumer_watch_seals'] == chains[1],
        'Same actual six-arm stage, not another positive source or saved negative attempt')
    require(all(r[k] is True for k in ('tiny_qualified','null_byte_exact','own_complete_bills',
        'all_parameter_and_estimator_owners_retired','all_outputs_sealed_before_evaluation_B',
        'hashes_unchanged','sources_unchanged')) and r['evaluation_B_accessed'] is False
        and r['score_measured'] is False and r['full_model_calls'] == r['parameter_updates'] == 0
        and r['partial_model_calls'] == 5 and r['tiny_contexts'] == 1,
        'All fixed consumers retired and outputs sealed before any B evaluation')
    require(all(w[k] is True for k in ('whole_owned_tree_retired','cleanup_passed','observational_groups_retired',
        'actual_completed_samples','hashes_unchanged','sources_unchanged')) and w['numerical_imports'] is False
        and 0 < w['maximum_combined_rss_bytes'] < 8_000_000_000
        and 0 < w['maximum_interval_s'] <= 1 and w['maximum_swap_growth_bytes'] == 0
        and 0 < rf['elapsed_s'] < 600 and 0 < wf['elapsed_s'] < 600,
        'Actual consumer full-tree resource and retirement proof')
    require(context['all_six_retired_before_evaluation'] is True and context['negative_v1_preserved'] is True
        and context['prior_total_flops'] == r['capture_plus_all_consumer_flops'] == 710_007_601_555
        and context['prior_tiny_flops'] == 1_427_192
        and context['prior_saved_array_bytes'] == 494_273_728,
        'Failed v1, capture and positive v2 costs retained in original aggregate')
    result.update(candidate_sha256=CANDIDATE_SHA,consumer_seals=chains[0],consumer_watch_seals=chains[1],
        all_six_retired_before_evaluation=True,negative_v1_preserved=True,
        prior_total_flops=context['prior_total_flops'],prior_tiny_flops=context['prior_tiny_flops'],
        prior_saved_array_bytes=context['prior_saved_array_bytes'])


def analysis_gate(read, sha, require, api, record, result, context, files, digests, tick):
    """Check saved scalar/bill/lifecycle metadata; never decode numerical arrays."""
    global B_BYTE_ALLOWED
    require(record['schema'] == 'current_comp_joint_cell_analysis_v1'
        and record['candidate_sha256'] == CANDIDATE_SHA and record['rank'] == 35
        and record['layer'] == 8 and record['consumer_layer'] == 9
        and record['roster'] == ['analysis'] and record['ranks'] == [35]
        and record['consumer_roster'] == list(CONSUMER_ROSTER)
        and record['analysis_contexts'] == record['retired_call_count'] == 1
        and record['full_model_calls'] == record['partial_model_calls'] == record['parameter_updates'] == 0,
        'Exactly one saved-output analysis, no model or fitted intervention')
    require(all(record[k] is True for k in ('own_complete_bills','all_numerical_array_owners_retired',
        'all_parameter_and_estimator_owners_retired','all_six_retired_before_evaluation',
        'evaluation_B_accessed','negative_v1_preserved','scientific_information_measured',
        'no_automatic_rescue_or_followup','hashes_unchanged','sources_unchanged'))
        and all(record[k] is False for k in ('automatic_next_stage','deployable_correction_demonstrated',
        'raw_gain_measured','estimator_J_measured','bank_or_submission_authorized',
        'current_cell_centering_error_bound_available','covariance_joint_noise_standard_error_available')),
        'Honest information-only scope and fully retired numerical owners')
    for key in ('consumer_seals','consumer_watch_seals','prior_total_flops','prior_tiny_flops',
                'prior_saved_array_bytes','prior_elapsed_s_including_metadata_reserve'):
        require(record[key] == context[key], 'Exact pre-spawn authority and attempted-cost join')
    folder = HERE/'analysis_v1'
    expected = {str(folder/n) for n in ('pre_evaluation.json','call_00_bill.json','call_00_lifecycle.json',
        'call_00_diagnostics.json','call_00_complete.json','vectors.bin','vectors.json','statistics.json')}
    require(set(record['artifacts']) == expected, 'Exact eight analysis artifacts, no undisclosed numerical outputs')
    pre,bill,life,diag,done,stats,vm = [read(folder/n) for n in ('pre_evaluation.json','call_00_bill.json',
        'call_00_lifecycle.json','call_00_diagnostics.json','call_00_complete.json','statistics.json','vectors.json')]
    require(pre['passed'] is True and pre['completed'] is True and pre['evaluation_B_accessed'] is False
        and pre['all_six_retired_before_evaluation'] is True and pre['consumer_seals'] == context['consumer_seals']
        and pre['consumer_watch_seals'] == context['consumer_watch_seals']
        and all(record['source_bindings'].get(p) == h for p,h in pre['source_bindings'].items())
        and all(pre['source_bindings'].get(p) == h for p,h in context['source_bindings'].items())
        and pre['input_bindings'] == record['input_bindings'],
        'Actual pre-evaluation seal preserves complete source/input authority')
    actual_modules = {x['path']:x['sha256'] for x in record['backend_attestation']['modules'].values()}
    require(all(actual_modules.get(p) == h for p,h in record['source_bindings'].items()
        if p not in pre['source_bindings']), 'Post-seal source additions are actual imported numerical owners only')
    require(all(x['call'] == 0 and x['rank'] == 35 and x['arm'] == 'analysis' for x in (bill,life,diag,done))
        and done['passed'] is True and done['completed'] is True and done['context_retired'] is True
        and done['all_parameter_and_estimator_owners_retired'] is True
        and life['context_released'] is True and life['closed_context_bill_unchanged_by_reset'] is True
        and life['all_numerical_array_owners_retired'] is True
        and life['all_parameter_and_estimator_owners_retired'] is True
        and diag['all_numerical_array_owners_retired'] is True and diag['no_estimator_or_parameter_owners_created'] is True,
        'The one retired marker owns the complete paid calculation and released storage')
    for suffix,key in (('bill','bill_sha256'),('lifecycle','lifecycle_sha256'),('diagnostics','diagnostic_sha256')):
        require(sha(folder/f'call_00_{suffix}.json',tick) == done[key], 'Actual retired-call links')
    api.ledger(bill)
    require(api.ANALYSIS_FLOP_CAP == 1_000_000_000
        and 0 < bill['flops'] <= api.ANALYSIS_FLOP_CAP and bill['input_flops'] == 0
        and bill['complete_research_flops'] == record['analysis_flops'] == bill['flops']
        and bill['all_numerical_work_in_single_owned_context'] is True and bill['not_a_complete_estimator_score'] is True
        and 0 < bill['wall_s'] < 60 and 0 < bill['context_wall_s'] < 60
        and record['total_research_flops'] == context['prior_total_flops']+bill['flops'] <= 1_425_000_000_000
        and record['tiny_plus_analysis_flops'] == context['prior_tiny_flops']+bill['flops'] <= 10_000_000_000,
        'Complete actual analysis bill, failed work and unchanged original ceilings')
    require(vm['path'] == str(folder/'vectors.bin') and vm['sha256'] == sha(folder/'vectors.bin',tick)
        and vm['bytes'] == (folder/'vectors.bin').stat().st_size < 16*1024**2
        and context['prior_saved_array_bytes']+vm['bytes'] < 1024**3
        and diag['statistics_sha256'] == sha(folder/'statistics.json',tick)
        and diag['vector_manifest_sha256'] == sha(folder/'vectors.json',tick)
        and record['primary'] == diag['primary'] == stats['primary']
        and record['numerical_owner_proof'] == diag['numerical_owner_proof'], 'Actual sealed statistics and detached vectors')
    primary = record['primary']
    require(primary['arm'] == 'TQ' and primary['baseline'] == 'N'
        and primary['covariance_stage'] == 'end8.C' and primary['mean_stage'] == 'served_FP32_L9'
        and primary['outcome'] in ('provisional_witness','negative','inconclusive')
        and type(primary['provisional_joint_supply_witness']) is bool
        and record['provisional_joint_supply_witness'] == primary['provisional_joint_supply_witness'],
        'Only the prospectively fixed primary; scientific failure is a valid measured result')
    expected_evaluation = {str(p):h for p,h in ARCHIVES.items() if p != P2/'public_g19_attribution_v1/mps_planes_rank0035.npz'}
    require(record['evaluation_bindings'] == expected_evaluation, 'Exactly fixed A covariance and B replica bytes')
    B_BYTE_ALLOWED = True
    for path,digest in expected_evaluation.items():
        require(sha(path,tick) == digest, 'Independent byte authentication only, after positive analysis seals')
    merge(result['input_bindings'],expected_evaluation,require)
    merge(result['input_bindings'],{str(p):h for p,h in zip(files,digests)},require)
    result.update(analysis_flops=bill['flops'],total_research_flops=record['total_research_flops'],
        tiny_plus_analysis_flops=record['tiny_plus_analysis_flops'],own_complete_bills=True,
        all_numerical_array_owners_retired=True,evaluation_B_byte_authenticated=True,
        scientific_outcome_measured_by_worker=True,primary=primary,
        provisional_joint_supply_witness=primary['provisional_joint_supply_witness'],
        scientific_success_required_for_operational_pass=False,bank_or_submission_authorized=False)


def source_and_proof():
    raw = OWNER.read_bytes()
    if hashlib.sha256(raw).hexdigest() != OWNER_SHA:
        raise RuntimeError('Exact independently qualified external watch owner')
    original = raw.decode('utf8')
    own_start = original.index('    own = {HERE/')
    own_end = original.index('    require(all(re.fullmatch',own_start)
    prep_start = original.index("        probe = load(HERE/'probe.py'")
    prep_end = original.index('        env = dict(os.environ',prep_start)
    gate_start = original.index("        require(record['candidate_sha256'] == result['candidate_sha256']")
    gate_end = original.index("        for group in ('source_bindings'",gate_start)
    edits = [
        ('CAP, RSS, PROGRESS = 300.,8_000_000_000,120.','CAP, RSS, PROGRESS = 60.,1_000_000_000,60.'),
        ("KEYS = ('source','worker','protocol','spec','compose','fnp','design')",'KEYS = _ANALYSIS_KEYS'),
        ('began = previous = last_progress = time.monotonic()','began = previous = last_progress = _ANALYSIS_BEGAN'),
        (original[own_start:own_end],
         "    own = {HERE/'watch_analysis.py':args.source_sha256,HERE/'analysis.py':args.worker_sha256,\n"
         "        HERE/'ANALYSIS_PROTOCOL.md':args.protocol_sha256,HERE/'ANALYSIS_WATCH_SPEC.md':args.spec_sha256,\n"
         "        HERE/'consumer_v2.py':args.consumer_source_sha256,HERE/'CONSUMER_PROTOCOL_V2.md':args.consumer_protocol_sha256,\n"
         "        HERE/'watch_consumer_v2.py':args.consumer_watch_source_sha256,\n"
         "        HERE/'CONSUMER_WATCH_SPEC_V2.md':args.consumer_watch_spec_sha256,\n"
         "        TRANSPORT:TRANSPORT_SHA,_ANALYSIS_OWNER:_ANALYSIS_OWNER_SHA}\n"),
        ("HERE/'probe_v1',HERE/'probe_watch_v1'","HERE/'analysis_v1',HERE/'analysis_watch_v1'"),
        ("schema='current_comp_shared_old384_native37_watch_v1'","schema='current_comp_joint_cell_analysis_watch_v1'"),
        ('numerical_imports=False,truth_accessed=False,score_measured=False,uploaded=False,',
         'numerical_imports=False,numerical_archives_decoded=False,readout_arrays_decoded=False,uploaded=False,\n'
         '        evaluation_B_byte_authenticated=False,scientific_outcome_measured_by_worker=False,\n'
         '        watch_source_proof=WATCH_SOURCE_PROOF,'),
        ('retired_calls=0,ranks=[37,37]','retired_calls=0,ranks=[35],roster=[\'analysis\']'),
        ("'Whole300s/next retired call120s'","'Whole60s/one fully retired analysis within60s'"),
        ("'Zero added swap/full owned tree8GB'","'Zero added swap/full owned tree1GB'"),
        ("raise RuntimeError('Owned fixed shared-basis stage interrupted')","raise RuntimeError('Owned fixed analysis stage interrupted')"),
        ("        require(not classifier(sample['rows'],scope.pop('allowed')),'No foreign numerical worker')",
         "        allowed = scope.pop('allowed')\n"
         "        require(not classifier(sample['rows'],allowed),'No foreign numerical worker')\n"
         "        require(not any(int(row['pid']) not in allowed and 'flopscope_server' in str(row.get('command',''))\n"
         "            for row in sample['rows']), 'No foreign underscore-named RPC server')"),
        ("index < 2 and item['call'] == index and item['rank'] == 37,'Exact ordered37/37'",
         "index == 0 and item['call'] == 0 and item['rank'] == 35 and item['arm'] == 'analysis','One fixed analysis marker'"),
        ("record['call'] == index and record['rank'] == 37 and record['context_retired']",
         "record['call'] == 0 and record['rank'] == 35 and record['arm'] == 'analysis' and record['context_retired']\n"
         "                        and record['all_parameter_and_estimator_owners_retired'] is True"),
        ('sha256=digest,call=index,rank=37)','sha256=digest,call=index,rank=35,arm=\'analysis\')'),
        (original[prep_start:prep_end],
         "        probe = load(HERE/'analysis.py','joint_cell_fixed_analysis_metadata',args.worker_sha256)\n"
         "        ns = _analysis_worker_namespace(args)\n"
         "        context = probe.admission(ns,tick)\n"
         "        for group in ('source_bindings','input_bindings'):\n"
         "            require(isinstance(context[group],dict) and context[group], 'Complete inert analysis admission')\n"
         "            _analysis_merge(result[group],context[group],require)\n"
         "        _analysis_consumer_gate(read,sha,require,args,context,result,tick)\n"
         "        require(not any(k.split('.')[0] in ('numpy','torch','flopscope','threadpoolctl')\n"
         "            for k in sys.modules),'No supervisor numerical imports')\n"
         "        argv = _analysis_child_command(args,OFFPY)\n"),
        ("        env.update({k:'1' for k in BLAS})",
         "        env.update({k:'1' for k in BLAS})\n"
         "        for key in ('PYTHONOPTIMIZE','PYTHONPATH','FLOPSCOPE_SERVER_URL'):\n"
         "            env.pop(key,None)"),
        ("result['retired_calls'] == 2,'Two complete calls and exit0'",
         "result['retired_calls'] == 1,'Exactly one completely retired analysis and exit0'"),
        (original[gate_start:gate_end],
         '        _analysis_gate(read,sha,require,probe,record,result,context,files,digests,tick)\n'),
        ("                result[group][path] = digest",
         "                _analysis_merge(result[group],{path:digest},require)"),
        ('if p != TRANSPORT)','if p not in (TRANSPORT,_ANALYSIS_OWNER))'),
        ('    path = Path(path).resolve()',
         "    path = Path(path)\n    require(not path.is_symlink(),'No symlink byte owner')\n    path = path.resolve()"),
        ("require(path.is_relative_to(P2) and 'truthpost' not in path.parts and not\n"
         "        (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()), 'No supervisor target access')",
         "require(_analysis_path_allowed(path),'Only explicitly admitted bytes; no numerical archive decoding')")]
    changed,spans = original,[]
    for old,new in edits:
        if changed.count(old) != 1:
            raise RuntimeError('Unique analysis-watch splice: '+old[:100])
        pos = changed.index(old)
        changed = changed[:pos]+new+changed[pos+len(old):]
        spans.append((pos,old,new))
    inverse = changed
    for pos,old,new in reversed(spans):
        if inverse[pos:pos+len(new)] != new:
            raise RuntimeError('Exact positional inverse of analysis-watch adapter')
        inverse = inverse[:pos]+old+inverse[pos+len(new):]
    if inverse.encode('utf8') != raw:
        raise RuntimeError('Complete frozen external-watch source inverse')
    return changed,dict(owner_sha256=OWNER_SHA,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),splices=len(spans),
        resource_transport_scope_cleanup_and_completion_unchanged=True,
        one_analysis_context=True,whole_cap_s=60,combined_rss_limit_bytes=1_000_000_000)


def main():
    global B_BYTE_ALLOWED
    began = time.monotonic()
    B_BYTE_ALLOWED = False
    text,proof = source_and_proof()
    namespace = dict(__name__='joint_cell_qualified_analysis_watch',__file__=str(HERE/'watch_analysis.py'),
        _ANALYSIS_KEYS=KEYS,_ANALYSIS_BEGAN=began,_ANALYSIS_OWNER=OWNER,_ANALYSIS_OWNER_SHA=OWNER_SHA,
        _analysis_worker_namespace=worker_namespace,_analysis_child_command=child_command,
        _analysis_merge=merge,_analysis_path_allowed=path_allowed,
        _analysis_consumer_gate=consumer_gate,_analysis_gate=analysis_gate,WATCH_SOURCE_PROOF=proof)
    exec(compile(text,'<one-analysis-qualified-external-watch>','exec'),namespace)
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
