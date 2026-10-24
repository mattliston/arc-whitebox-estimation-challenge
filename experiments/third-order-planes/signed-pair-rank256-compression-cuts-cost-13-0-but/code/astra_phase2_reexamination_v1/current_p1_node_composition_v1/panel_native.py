"""One fixed14+repeat worker, unchanged candidate and complete BD paid loop."""
import importlib.metadata
import os
from pathlib import Path
import sys
import panel_common as c


def main():
    args = c.arguments('worker')
    if args is None: return 0
    c.environment()
    context = c.read_context(args,args.context_sha256); c.retired_admission(args,context)
    n = c.e.old()  # Metadata definitions only: no authenticate/construct in this process.
    versions = {k:importlib.metadata.version(k) for k in ('numpy','flopscope','whestbench','threadpoolctl')}
    c.require(versions == dict(numpy='2.4.6',flopscope='0.12.0',whestbench='0.16.0',threadpoolctl='3.6.0'),
              'Original qualified native frame')
    n.load_diagnostics(); legacy = n.legacy()
    observer,source,proof = c.paid_source(n)
    c.require(proof == context['proof'] and not c.OUTPUT.exists() and not c.ROOT.is_symlink(),
              'Exact retired source proofs and exclusive full acquisition')
    pins = {**context['source_bindings'],**context['input_bindings']}
    original_finish = c.finish_function(n)

    def after_bill(np,observed,trial,result,folder):
        # Original NODE-A/cstar and all16/all47 owner runs after bill/reset/release.
        n.after_bill(np,observed,trial,result,folder,context['reference_by_rank'])
        ref = context['reference_by_rank'][trial['rank']]
        c.require(c.sha(ref['output']) == ref['output_sha256'], 'Authenticated P1 reference before output-only decode')
        own,prior = np.load(trial['output'],allow_pickle=False),np.load(ref['output'],allow_pickle=False)
        exact = own[0].tobytes() == prior[0].tobytes()
        result['layer0_checks'].append(dict(call=trial['call'],rank=trial['rank'],byte_exact=bool(exact)))
        c.require(exact, 'Unchanged Gaussian first layer, later outputs/costs allowed to differ')
        del own,prior

    def finish(result):
        original_finish(result)
        c.require(len(result['layer0_checks']) == 15 and all(x['byte_exact'] for x in result['layer0_checks']),
                  'All first-layer checks, no partial acquisition')
        result.update(layer0_byte_exact=True)

    header = dict(schema=c.NATIVE_SCHEMA,mode='acquire',stage='native',role='terminal',completed=False,
        sources_unchanged=False,build_sha256=n.BUILD_SHA,builder_sha256=n.BUILDER_SHA,
        prerequisites=context['prerequisites'],versions=versions,truth_accessed=False,truth_files_accessed=False,
        no_fit=True,no_refit=True,score_measured=False,old_NODE_gate_still_failed=True,
        native_original_sha256=n.NATIVE_SHA,native_main_sha256=proof['native_original_main_sha256'],
        native_adapter_parent_sha256=n.RC_SHA,source_inverse_verified=True,paid_arithmetic_and_lifecycle_unchanged=True,
        original_TL_diagnostics_unchanged=True,response_floor=.01,active_floor=.01,birth_floor=.01,
        own_complete_bills=False,old_initialization_value_assumed=False,auto_launch_next_stage=False,
        admission_context_sha256=args.context_sha256,admission_watch_sha256=args.admission_watch_sha256,
        observer_event_filter_inverse_verified=True,proof=proof,full_admission_in_retired_process=True,
        fixture_bindings=context['fixtures'],layer0_checks=[],layer0_byte_exact=False,
        terminal_score_sha256=c.SCORE_SHA,control_result_sha256=c.CONTROL_SHA,pid=os.getpid(),
        nice=os.getpriority(os.PRIO_PROCESS,0),blas_env=dict.fromkeys(c.e.BLAS,'1'),
        executable=sys.executable,no_warmup=True,initialization_once=True,repeat_excluded=True)
    namespace = dict(vars(legacy),__file__=str(Path(__file__).resolve()),PARENT=c.NEW,CANDIDATE=c.CANDIDATE,
        NATIVE_PINS=pins,FIXED_RANKS=c.RANKS,PANEL_ROWS=context['rows'],PANEL_FIXTURES=context['fixtures'],
        Observer=observer,observed_predict=n.observed_predict,after_bill=after_bill,finish_native=finish,NATIVE_HEADER=header)
    os.chdir(c.NEW); previous = sys.argv
    try:
        sys.argv = [str(__file__),'--source',str(c.CANDIDATE),'--split','validation','--out',str(c.OUTPUT)]
        exec(compile(source,str(__file__)+':unchanged_paid_panel','exec'),namespace)
        return namespace['main']()
    finally:
        sys.argv = previous


if __name__ == '__main__':
    raise SystemExit(main())
