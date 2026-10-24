"""Fresh complete terminal acquisition using a retired full-admission owner.

No model/candidate/meter changes. Only an inverse-proved neutral event filter
and strengthened same-history checks follow the preserved complete paid loop.
"""
import hashlib
import importlib.metadata
import os
from pathlib import Path
import sys
import endpoint_v2 as e


def prepare(n):
    """Small harness-source parsing only; never parse an estimator P literal."""
    original = e.function_source(e.HERE/'native.py','Observer')
    filtered = e.splice(original,[
        ('        self.begin()\n', '        self.begin()\n'
         '        self.interested_code_ids = frozenset(id(x) for x in self.codes.values()) | frozenset(\n'
         "            id(getattr(self.small,k)) for k in ('predict_code','gaussian_code','select_code','gnn_code'))\n"
         "        require(len(self.interested_code_ids) == 8, 'Exactly eight original observed code identities')\n"),
        ('        self.small(frame,event,answer)\n',
         "        if event not in ('call','return') or id(frame.f_code) not in self.interested_code_ids:\n"
         '            return\n        self.small(frame,event,answer)\n')])
    namespace = dict(vars(n))
    exec(compile(filtered,str(__file__)+':neutral_observer_filter','exec'),namespace)
    paid, original_sha = n.derived_main()  # Existing exact inverse to original BD paid main.
    proof = dict(observer_inverse_verified=True,paid_loop_inverse_verified=True,interested_code_count=8,
        observer_original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        observer_filtered_sha256=hashlib.sha256(filtered.encode()).hexdigest(),
        derived_paid_main_sha256=hashlib.sha256(paid.encode()).hexdigest(),native_original_main_sha256=original_sha)
    return namespace['Observer'],paid,proof


def main():
    args = e.arguments('worker')
    if args is None: return 0
    e.environment()
    context = e.read_context(args,args.context_sha256)
    e.retired_admission(args,context)
    n = e.old()  # Defines metadata functions only; authenticate/construct are NOT called.
    versions = {k:importlib.metadata.version(k) for k in ('numpy','flopscope','whestbench','threadpoolctl')}
    e.require(versions == dict(numpy='2.4.6',flopscope='0.12.0',whestbench='0.16.0',threadpoolctl='3.6.0'),
              'Unchanged qualified official native versions')
    n.load_diagnostics(); legacy = n.legacy(); observer,source,proof = prepare(n)
    e.require(proof == context['proof'] and tuple(legacy.VALID) == n.VALID, 'Same admitted inverse proofs and VALID')
    e.require(not e.OUTPUT.exists() and not e.HERE.is_symlink(), 'Fresh complete roster; never resume/overwrite')
    candidate = e.HERE/'build_v1/terminal_estimator.py'
    e.require(e.sha(candidate) == n.CANDIDATES['terminal'], 'Unchanged frozen candidate')
    pins = {**context['source_bindings'],**context['input_bindings']}

    def after_bill(np, observed, trial, result, folder):
        n.after_bill(np,observed,trial,result,folder,context['reference_by_rank'])
        if e.compare22(trial,result['input_preparation'][-1],folder/f"call_{trial['call']:02d}_diagnostics.json",context):
            result['first22_exact_checks'] += 1
        if trial['call'] == 21:
            e.require(result['first22_exact_checks'] == 22, 'All old22 checks precede any later call')

    def finish(result):
        e.require(result['first22_exact_checks'] == 22, 'All22 same-history controls before completion')
        n.finish_native(result)
        result['first22_output_bill_domain_byte_exact'] = True

    header = dict(schema=e.NATIVE_SCHEMA,stage='native',role='terminal',completed=False,sources_unchanged=False,
        build_sha256=n.BUILD_SHA,builder_sha256=n.BUILDER_SHA,prerequisites=context['prerequisites'],
        versions=versions,truth_accessed=False,no_fit=True,score_measured=False,old_NODE_gate_still_failed=True,
        native_original_sha256=n.NATIVE_SHA,native_main_sha256=proof['native_original_main_sha256'],
        native_adapter_parent_sha256=n.RC_SHA,source_inverse_verified=True,paid_arithmetic_and_lifecycle_unchanged=True,
        original_TL_diagnostics_unchanged=True,response_floor=.01,active_floor=.01,birth_floor=.01,
        own_complete_bills=False,old_initialization_value_assumed=False,auto_launch_next_stage=False,
        admission_context_sha256=args.context_sha256,admission_watch_sha256=args.admission_watch_sha256,
        failed_v1_watch_sha256=e.FAIL_SHA,
        observer_event_filter_inverse_verified=True,proof=proof,first22_exact_checks=0,
        first22_output_bill_domain_byte_exact=False,full_admission_in_retired_process=True)
    namespace = dict(vars(legacy),__file__=str(Path(__file__).resolve()),PARENT=e.NEW,CANDIDATE=candidate,
        NATIVE_PINS=pins,FIXED_RANKS=n.VALID,Observer=observer,observed_predict=n.observed_predict,
        after_bill=after_bill,finish_native=finish,NATIVE_HEADER=header)
    os.chdir(e.NEW); saved = sys.argv
    try:
        sys.argv = [str(__file__),'--source',str(candidate),'--split','validation','--out',str(e.OUTPUT)]
        exec(compile(source,str(__file__)+':unchanged_paid_native','exec'),namespace)
        return namespace['main']()
    finally:
        sys.argv = saved


if __name__ == '__main__':
    raise SystemExit(main())
