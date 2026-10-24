"""Short-lived full frozen admission; publishes bounded metadata, never a card."""
import json
import signal
import time
import traceback
import endpoint_v2 as e


def main():
    args = e.arguments('admitter')
    if args is None: return 0
    began = time.monotonic(); e.environment()
    owners = e.owners(args)
    e.require(not e.ADMISSION.exists() and not e.HERE.is_symlink(), 'Exclusive admission, no retry')
    e.ADMISSION.mkdir(exist_ok=False)
    record = dict(schema=e.ADMIT_SCHEMA,passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,
        reviewed_owners=owners,source_bindings=owners,input_bindings={},artifacts={},
        estimator_executed=False,numerical_imports=False,truth_accessed=False)
    def guard(*unused):
        e.require(time.monotonic()-began < 60, 'Fixed admission60s, inside original whole/progress limits')
    previous = signal.signal(signal.SIGALRM,guard); signal.setitimer(signal.ITIMER_REAL,.2,.2)
    try:
        n = e.old()
        admitted = n.authenticate('terminal',n.BUILD_SHA,n.CANDIDATES['terminal'],n.BUILDER_SHA,
                                  e.OLD_NATIVE_SHA,e.OLD_WATCH_SHA,e.ZERO_SHA,e.ZERO_WATCH_SHA)
        native = e.load(e.HERE/'native_v2.py','node_p1_v2_source_proofs',args.worker_sha256)
        _observer,_paid,proof = native.prepare(n)
        sources = {**admitted['source_bindings'],**owners}
        inputs = dict(admitted['input_bindings'])
        preserved = e.preserved22(inputs)
        context = dict(schema=e.CONTEXT_SCHEMA,role='terminal',ranks=list(n.VALID),
            candidate_sha256=n.CANDIDATES['terminal'],build_sha256=n.BUILD_SHA,
            prerequisites=admitted['prerequisites'],reference_by_rank=admitted['reference_by_rank'],
            source_bindings=sources,input_bindings=inputs,reviewed_owners=owners,preserved22=preserved,
            failed_watch_sha256=e.FAIL_SHA,full_frozen_admission_passed=True,
            observer_inverse_verified=True,paid_loop_inverse_verified=True,proof=proof,
            estimator_executed=False,numerical_imports=False,truth_accessed=False)
        encoded = json.dumps(context,sort_keys=True,indent=2,allow_nan=False)+'\n'
        e.require(len(encoded.encode()) <= e.CONTEXT_LIMIT, 'Bounded context without source literals/arrays/modules')
        e.bind(sources,guard); e.bind(inputs,guard); e.environment(); guard()
        e.write(e.ADMISSION/'context.json',context)
        record.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,
            source_bindings=sources,input_bindings=inputs,context_sha256=e.sha(e.ADMISSION/'context.json'),
            context_size_bytes=(e.ADMISSION/'context.json').stat().st_size,proof=proof,
            full_frozen_admission_passed=True,preserved_completed_calls=22)
        record['artifacts'][str(e.ADMISSION/'context.json')] = record['context_sha256']
        guard()
    except BaseException:
        record.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,error=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,previous)
        record['elapsed_s'] = time.monotonic()-began
        if record['elapsed_s'] >= 60:
            record.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False)
        e.write(e.ADMISSION/'result.json',record)
        e.write(e.ADMISSION/'completion.json',dict(passed=record['passed'],completed=record['completed'],
            result_sha256=e.sha(e.ADMISSION/'result.json')))
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
