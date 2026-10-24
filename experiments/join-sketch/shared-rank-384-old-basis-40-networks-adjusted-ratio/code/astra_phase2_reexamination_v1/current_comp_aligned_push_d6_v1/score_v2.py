"""Versioned metadata repair only; preserves the completed native run and failed watch."""
import hashlib
import json
from pathlib import Path

_here=Path(__file__).resolve().parent
_parent=_here/'score.py'
_parent_sha='4804a14f9442d431234d724d9de3ac15fb483aa08ed39b7df339286dbf781074'
_watch_seals=('94b6e66e317cb458ebb6000370f4451fac9052f15264aae2d3ad1d49a8da388d',
    '595c8d3a2cf1da3902bba456bbdc3b667ad1880b1bab582c3f2124c74a400804',
    'b13a7bff299a26d87589ef1cddb0337a53494e882b07bb5d171fa55aef87be69')
_native_seals=('b9372a024b1257ead34fe8e36fcd85014735f73e0b2a6bbb83bfaca2b8f3c1e8',
    'ed9222dc1cfdc50409746642a8102c426b314e447e920d3b2e1a0e57e62ebba1',
    'b6ccb318ef0763fb2fb09f5266238de7ad633cb711c237e13b07230b135be5c3')
_watch_source='dca03cd8830f02faee4757383e541e759bc0170cb9299cc60d1283326af31064'
_local={
    'samples.jsonl':'89c1fcee7c909b774ed3bde83af949c68a0f0b8359ac5c16fb4716dbca5c0857',
    'worker.log':'4c3092690ae41ffc69c9b013b8fdbe0a4d4ca801fe1a2d65988589f1194dbb65',
    'started.json':'7024561681bce6992c1830239507142260d44e37e7059e4f83def29c3391d62b',
    'worker_started.json':'8c4f43cb532cde2ab508e94062c3abd66a8c83dff80ddad0a5d4e5aff9287804',
    'admission.log':'c6fcb36fec24afad1d612c8afbcbff42aab011a4cc942718b890286f8b9ad122'}


def _require(ok,message):
    if not ok: raise RuntimeError(message)


def _read(path): return json.loads(Path(path).read_text())


def _failed_watch_chain(admit,folder,digest):
    _require(folder==_here/'native_watch_v1' and digest==_watch_seals[0],
        'Only the exact completed run with the known metadata-field failure')
    paths=[folder/n for n in ('result.json','completion.json','finalization.json')]
    for path,bound in zip(paths,_watch_seals): admit.pin(path,bound)
    record,end,final=map(_read,paths)
    _require(all(v.get('passed') is False and v.get('completed') is False for v in (record,end,final))
        and end['result_sha256']==final['result_sha256']==_watch_seals[0]
        and final['completion_sha256']==_watch_seals[1]
        and 0<record['elapsed_s']<=final['elapsed_s']<600
        and 0<=final['terminal_publication_interval_s']<=1,
        'Original negative three seals remain negative, with original whole/tail bounds')
    _require(record['error'].endswith("KeyError: 'input_bindings'\n")
        and 'line 219, in main\n    inventory(record)' in record['error']
        and 'cleanup_error' not in record,'Exact post-reap/post-native-validation inventory bug only')
    admit.inventory(record)
    for name,bound in _local.items():
        path=folder/name
        _require(record['artifacts'][str(path)]==bound,'Exact original saved external evidence')
        admit.pin(path,bound)
    return record,end,dict(zip(('result_sha256','completion_sha256','finalization_sha256'),_watch_seals))


def _verify_saved_watch(admit,watch,args,ns,ads,context_sha):
    _require(args.native_watch_sha256==_watch_seals[0] and args.native_watch_source_sha256==_watch_source
        and tuple(ns[k] for k in ('result_sha256','completion_sha256','finalization_sha256'))==_native_seals,
        'Exact independently completed positive native packet and negative observer packet')
    roster=[dict(call=i,rank=r) for i,r in enumerate((37,39,42,48,37))]
    _require(watch['schema']=='current_comp_aligned_push_d6_native_watch_v1'
        and all(watch.get(k) is True for k in ('whole_owned_tree_retired','observational_groups_retired',
            'cleanup_passed','admission_producer_reaped'))
        and watch['retired_call_count']==5 and watch['retired_calls']==roster
        and watch['context_sha256']==context_sha and watch['admission_seals']==ads
        and watch['candidate_sha256']==args.candidate_sha256
        and watch['truth_arrays_accessed'] is False and watch['numerical_imports'] is False
        and watch['full_model_calls']==0 and watch['hashes_unchanged'] is False
        and watch['sources_unchanged'] is False and 'child_result_sha256' not in watch,
        'Preserve actual partial accounting; do not manufacture successful old-watch fields')
    _require(len(watch['phases'])==1 and watch['phases'][0]['phase']=='admission'
        and watch['phases'][0]['exited'] is True
        and watch['phases'][0]['elapsed_through_verification_s']<60,
        'Original complete source admission before native worker')
    folder=_here/'native_watch_v1'
    samples=[json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]
    intervals=watch['completed_sample_intervals_s']
    _require(len(samples)==len(intervals)==watch['observation_samples']==644
        and all(0<x<=1 for x in intervals)
        and max(intervals)==watch['maximum_interval_s']
        and abs(sum(intervals)-watch['last_completed_observation_elapsed_s'])<1e-8,
        'Every actual completed observation from the original whole clock')
    previous=last_progress=0.;last_count=0;phase='admission';elapsed_complete=0.
    for sample,interval in zip(samples,intervals):
        elapsed_complete+=interval;elapsed=sample['elapsed_s'];count=sample['retired_call_count']
        _require(previous<elapsed<=elapsed_complete and elapsed_complete-elapsed<.1
            and sample['phase'] in ('admission','worker')
            and sample['uncached_single_process_table'] is True,
            'Actual monotone sampled timestamps and completion interval, no synthesized sample')
        if phase=='admission' and sample['phase']=='worker':
            phase='worker';last_progress=previous
        # The complete scope also contains the live observational ps/sysctl children.
        _require(sample['phase']==phase and sample['combined_rss_bytes']>=
            sample['watcher_rss_bytes']+sample['child_tree_rss_bytes']
            and 0<sample['combined_rss_bytes']<(2_000_000_000 if phase=='admission' else 8_000_000_000)
            and sample['swap_growth_bytes']==sample['swap_bytes']-watch['initial_swap_bytes']<=0,
            'Exact original phase bounds and zero added swap throughout')
        _require(count in (last_count,last_count+1) and 0<=count<=5 and elapsed-last_progress<120,
            'All original per-retired-call deadlines and ordered markers')
        if count!=last_count: last_progress=elapsed
        if phase=='admission': _require(elapsed<60 and count==0,'Original60s admission deadline')
        commands=sample['transport']
        _require(len(commands)==2 and commands[0]['argv']==['/bin/ps','-axo','pid=,ppid=,pgid=,rss=,command=']
            and commands[1]['argv']==['/usr/sbin/sysctl','vm.swapusage']
            and all(0<row['finished_since_start_s']<=1 for row in commands),
            'Original completed uncached external observations')
        previous,last_count=elapsed,count;admit.guard()
    _require(not samples[-1]['child_pids'] and samples[-1]['child_tree_rss_bytes']==0 and last_count==5
        and watch['elapsed_s']-last_progress<120
        and watch['maximum_combined_rss_bytes']==max(s['combined_rss_bytes'] for s in samples)
        and all(watch['maximum_'+p+'_rss_bytes']==max(s['combined_rss_bytes'] for s in samples if s['phase']==p)
            for p in ('admission','worker')) and watch['maximum_swap_growth_bytes']==0,
        'Entire original model tree retired under unchanged complete resource limits')
    rows=[]
    for line in (folder/'worker.log').read_text().splitlines():
        try: value=json.loads(line)
        except ValueError: continue
        if isinstance(value,dict): rows.append(value)
    _require([v for v in rows if v.get('event')=='retired']==[dict(event='retired',call=i) for i in range(5)]
        and rows[-1]==dict(passed=True,trials=5,error=None), 'All original release markers and successful native exit summary')
    started=_read(folder/'worker_started.json')
    _require(started['pid']==watch['worker_pid'] and started['command']==watch['worker_command']
        and started['context_sha256']==context_sha,'Exact monitored native producer and context')
    # Frozen watch control flow requires child.wait()==0, cleanup, all native seals,
    # saved-acquisition verification and every native flag before failing line219.
    # Re-run those metadata/content checks in the original scorer below, not models.
    admit.result.update(saved_external_watch_revalidated=True,original_external_watch_still_failed=True,
        numerical_acquisition_repeated=False,original_observation_samples_verified=644,
        successful_child_exit_evidence='Pinned watch control flow before exact line219 KeyError plus native seals/log',
        repaired_metadata_only=True)


_require(hashlib.sha256(_parent.read_bytes()).hexdigest()==_parent_sha,'Frozen first scorer preserved')
_original=_parent.read_text()
_start=_original.index("    require(watch['schema'] == 'current_comp_aligned_push_d6_native_watch_v1'")
_last="        'Actual external native-watch success, full child/admission seals and complete resource bounds')\n"
_stop=_original.index(_last,_start)+len(_last)
_old_block=_original[_start:_stop]
_changes=(
    ("SCHEMA = 'current_comp_aligned_push_d6_score_v1'","SCHEMA = 'current_comp_aligned_push_d6_score_v2'"),
    ("PINS = {","PINS = {\n    HERE/'score.py': '"+_parent_sha+"',"),
    ("HERE/'SCORE_PROTOCOL.md'","HERE/'SCORE_PROTOCOL_V2.md'"),
    ("watch, _, ws = admit.chain(wf, args.native_watch_sha256, limit=600, inventory=True)",
     "watch, _, ws = failed_watch_chain(admit, wf, args.native_watch_sha256)"),
    (_old_block,"    verify_saved_watch(admit,watch,args,ns,ads,context_sha)\n"),
    ("out = HERE/'score_v1'","out = HERE/'score_v2'"),
)
_derived=_original
for _before,_after in _changes:
    _require(_derived.count(_before)==1,'Unique versioned metadata repair span')
    _derived=_derived.replace(_before,_after)
_inverse=_derived
for _before,_after in reversed(_changes):
    _require(_inverse.count(_after)==1,'Unique inverse metadata repair span')
    _inverse=_inverse.replace(_after,_before)
_require(_inverse==_original,'Entire original scorer restored; no numerical/selection/cost change')
exec(compile(_derived,str(_parent)+':metadata-repair-v2','exec'),
    dict(__name__=__name__,__file__=__file__,failed_watch_chain=_failed_watch_chain,verify_saved_watch=_verify_saved_watch))
