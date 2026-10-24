"""Inert cold-primary coordinator and isolated official-input native worker.

The public CLI matches ND1. Internal workers use this same reviewed source,
inherit the coordinator's process group, and execute the authenticated native-v2
paid blocks unchanged. Every scored primary owns its measured complete cold bill.
No target arrays, numerical library, or candidate are imported by the coordinator.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'current_d50_native_directional_search_v1/native.py'
OLD_SHA = '68f7ba7fb092fb057cc5db8fc0ec1998dd15ea3231aef3e07b9befb09153f40d'
PROTOCOL_SHA = '07339b5cd9ee06770145e36528a751d12e419ed514bcfba4d0465662a4368f8d'
FIT, CHECK = (35, 36, 38, 40), (41, 43, 44, 45)
INITIALIZATION = 1383913786
PRIMARY = dict(baseline=FIT + CHECK, plus=FIT, minus=FIT, candidate=FIT + CHECK)
EXECUTION = {case: ((35, 100, 35) if case == 'baseline' else (35, 35)) + ranks[1:]
             for case, ranks in PRIMARY.items()}
NUMERICAL = ('numpy', 'torch', 'flopscope', 'threadpoolctl')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    path = Path(path)
    require('truthpost' not in path.parts and not
            (path.suffix in ('.npz', '.npy') and 'truth' in path.name.lower()),
            'Native processes do not authenticate or decode target arrays')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def owner():
    require(sha(OLD) == OLD_SHA, 'Frozen ND1 utility source changed')
    require(not any(k in sys.modules for k in NUMERICAL), 'Stdlib-only owner admission')
    return module(OLD, 'nd2_frozen_native_utilities')


def schedules(case):
    return [EXECUTION[case][:3 if case == 'baseline' else 2]] + [
        (rank,) for rank in PRIMARY[case][1:]]


def prerequisites(stage, args, old):
    # The unchanged baseline branch authenticates the new target-free CPU stage.
    payload, sealed = old.prerequisites(stage, argparse.Namespace(
        case='baseline', qualification_result_sha256=args.qualification_result_sha256))
    if args.case == 'baseline':
        return payload, sealed
    baseline = stage.receipt('native_baseline_v1', args.baseline_result_sha256, 'native', 'baseline')
    require([t['rank'] for t in baseline['trials']] == list(EXECUTION['baseline'])
            and set(baseline['outputs']) == {str(r) for r in PRIMARY['baseline'] + (100,)}
            and baseline['initialization_flops'] == INITIALIZATION
            and baseline['truth_accessed'] is False and baseline['alltruth_accessed'] is False
            and all(baseline[k] is True for k in ('baseline_reference_passed', 'all_domain_checks_passed',
                'all_selector_checks_passed', 'repeat_byte_exact', 'repeat_diagnostics_byte_exact',
                'own_complete_bills', 'all_primary_cold'))
            and all(baseline['outputs'][str(r)]['includes_initialization'] is True
                    for r in PRIMARY['baseline']), 'New complete cold-primary baseline required')
    proposal = stage.receipt('proposal_v1', args.proposal_result_sha256, 'propose')
    require(proposal['prerequisites'] == (sealed | {'baseline_result_sha256': args.baseline_result_sha256})
            and proposal['parameter_updates'] == 1 and proposal['adam_steps'] == 1
            and proposal['curve_h'] == 5e-4 and proposal['curve_endpoint_byte_exact'] is True
            and proposal['frozen_payload_fields_byte_exact'] is True
            and proposal['proposal_sealed'] is True and proposal['surrogate_only'] is True
            and proposal['no_native_step_selection'] is True
            and proposal['check_targets_accessed'] is False and proposal['validation_targets_accessed'] is False,
            'Only the one fixed sealed prospective proposal')
    sealed.update(baseline_result_sha256=args.baseline_result_sha256,
                  proposal_result_sha256=args.proposal_result_sha256)
    if args.case in ('plus', 'minus'):
        payload = proposal['payloads'][args.case]
    else:
        interpolation = stage.receipt('interpolation_v1', args.interpolation_result_sha256, 'interpolate')
        require(interpolation['check_targets_accessed'] is False
                and interpolation['validation_targets_accessed'] is False,
                'Candidate sealed without CHECK/VALID arrays')
        for key, value in sealed.items():
            require(interpolation['prerequisites'][key] == value, 'Interpolation lineage: ' + key)
        payload = interpolation['payloads']['candidate']
        sealed['interpolation_result_sha256'] = args.interpolation_result_sha256
    return payload, sealed


def adapted_native_source(old):
    """Three reversible lifetime/roster edits after the frozen ND1 adapter."""
    source, proof = old.adapted_native_source()
    previous, edits = source, []

    def replace(before, after):
        nonlocal source
        require(source.count(before) == 1, 'Unique native lifetime span: ' + before[:80])
        source = source.replace(before, after)
        edits.append((before, after))

    replace('            for call,rank in enumerate(tuple(ranks)+(ranks[0],)):\n',
            '            for call,rank in enumerate(_cold_sequence):\n')
    replace("                if call==len(ranks): assert values.tobytes()==first_bytes,'Repeat identity'\n",
            "                if _cold_has_repeat and call==len(_cold_sequence)-1: assert values.tobytes()==first_bytes,'Repeat identity'\n")
    replace("                      initialization_flops=result['trials'][0]['flops']-result['trials'][-1]['flops'])\n",
            "                      initialization_flops=(result['trials'][0]['flops']-result['trials'][-1]['flops'] if _cold_has_repeat else None))\n")
    inverse = source
    for before, after in reversed(edits):
        require(inverse.count(after) == 1, 'Unique lifetime inverse')
        inverse = inverse.replace(after, before)
    require(inverse == previous, 'Exact ND1-adapted source inverse')
    original = old.ORIGINAL.read_text()
    blocks = {}
    for name, begin, end in (
        ('official_input', '                with flops.BudgetContext(flop_budget=BUDGET,wall_time_limit_s=120,quiet=True) as input_ctx:\n',
         '                input_record=dict('),
        ('participant', '                with flops.BudgetContext(flop_budget=BUDGET,wall_time_limit_s=120,quiet=True) as ctx:\n',
         '                trial=dict('),
        ('closed_bill_reset', '                context_ref=weakref.ref(ctx)\n', '                guard(trial)\n')):
        a = original[original.index(begin):original.index(end, original.index(begin))]
        b = source[source.index(begin):source.index(end, source.index(begin))]
        require(a == b, 'Paid/lifecycle block changed: ' + name)
        blocks[name] = hashlib.sha256(a.encode()).hexdigest()
    require(blocks == proof['paid_block_sha256'], 'Original paid block identities')
    proof.update(native_nd1_adapter_sha256=OLD_SHA,
        native_previous_adapted_sha256=hashlib.sha256(previous.encode()).hexdigest(),
        native_adapted_sha256=hashlib.sha256(source.encode()).hexdigest(),
        lifetime_source_inverse_verified=True, explicit_sequence_no_implicit_repeat=True,
        lifetime_edits=[dict(old_sha256=hashlib.sha256(a.encode()).hexdigest(),
                            new_sha256=hashlib.sha256(b.encode()).hexdigest()) for a, b in edits])
    return source, proof


class WorkerGates:
    def __init__(self, old, plan, references):
        self.old, self.plan, self.references = old, plan, references
        self.first = self.first_diagnostic = None
        self.checked, self.reference_checks = [], []

    def trial(self, trial, diagnostic):
        call = trial['call']
        sequence = self.plan['sequence']
        require(call == len(self.checked) and trial['rank'] == sequence[call], 'Exact worker sequence')
        self.old.check_diagnostics(diagnostic)
        require(trial['includes_initialization'] == (call == 0), 'Fresh worker owns one cold initialization')
        if call == 0:
            self.first, self.first_diagnostic = dict(trial), diagnostic
        if self.plan['has_repeat'] and call == len(sequence) - 1:
            require(trial['output_bytes_sha256'] == self.first['output_bytes_sha256']
                    and trial['output_sha256'] == self.first['output_sha256']
                    and diagnostic == self.first_diagnostic, 'Exact warm repeat output and observations')
            require(self.first['flops'] - trial['flops'] == INITIALIZATION,
                    'Measured per-case cold-minus-repeat initialization')
        if self.plan['case'] == 'baseline' and self.plan['worker_index'] == 0:
            ref = self.references[call]
            require(trial['rank'] == ref['rank'] and trial['flops'] == ref['flops']
                    and trial['includes_initialization'] == ref['includes_initialization']
                    and trial['output_bytes_sha256'] == ref['output_bytes_sha256']
                    and trial['output_sha256'] == ref['output_sha256'], 'Exact BR cold35/warm100/repeat35')
            self.reference_checks.append(dict(call=call, rank=trial['rank'], flops=trial['flops'],
                                              reference_call=ref['call'], passed=True))
        self.checked.append(call)

    def finish(self, result):
        count = len(self.plan['sequence'])
        require(self.checked == list(range(count)) and len(result['trials']) == count
                and len(result['input_preparation']) == count, 'All fixed worker calls and owned inputs')
        require(self.plan['worker_index'] == 0 or count == 1, 'Singleton cannot make an implicit repeat')
        require('torch' not in sys.modules, 'Native worker cannot import Torch')
        result.update(all_domain_checks_passed=True, all_selector_checks_passed=True,
            repeat_byte_exact=self.plan['has_repeat'], repeat_diagnostics_byte_exact=self.plan['has_repeat'],
            baseline_reference_passed=(len(self.reference_checks) == 3),
            baseline_reference_checks=self.reference_checks, own_complete_bills=True,
            all_primary_cold=True, primary_ranks=[self.plan['sequence'][0]],
            initialization_subtracted_from_primary=False, native_score_measured=False,
            alltruth_accessed=False, truth_accessed=False, MPS_used=False)


def authenticate_worker(c, args, old):
    """Metadata-only admission before any native import; no writable shared state."""
    require(not any(k in sys.modules for k in NUMERICAL), 'Worker admission before numerical imports')
    admission_path = HERE / 'ADMISSION.json'
    require(sha(admission_path) == args.admission_sha256, 'Reviewed worker admission digest')
    admission = c.read(admission_path)
    expected = {str(HERE / name) for name in c.SOURCES}
    require(admission.get('schema') == c.SCHEMA + '_admission' and admission.get('reviewed') is True
            and admission.get('numerical_work_begun') is False
            and set(admission['source_bindings']) == expected
            and admission['source_bindings'][str(HERE / 'PROTOCOL.md')] == PROTOCOL_SHA == c.PROTOCOL_SHA,
            'Exact reviewed worker source roster/protocol')
    for path, digest in admission['source_bindings'].items():
        require(sha(path) == digest, 'Worker reviewed source changed: ' + path)
    plan_path = Path(args.worker_plan).resolve()
    require(plan_path.parent == HERE / ('native_' + args.case + '_v1')
            and sha(plan_path) == args.worker_plan_sha256, 'Exclusive owned worker plan')
    plan = c.read(plan_path)
    index = plan['worker_index']
    require(type(index) is int and 0 <= index < len(schedules(args.case))
            and plan_path.name == f'worker_{index:02d}_plan.json'
            and plan['schema'] == c.SCHEMA + '_worker_plan' and plan['case'] == args.case
            and plan['admission_sha256'] == args.admission_sha256
            and plan['sequence'] == list(schedules(args.case)[index])
            and plan['has_repeat'] is (index == 0)
            and plan['coordinator_pid'] == os.getppid() and plan['process_group_id'] == os.getpgrp(),
            'Fixed sequence and inherited coordinator-owned process group')
    expected_out = plan_path.parent / f'worker_{index:02d}'
    require(Path(plan['directory']) == expected_out and not expected_out.exists(), 'Exclusive worker output')
    require(plan['pins'][str(OLD)] == OLD_SHA
            and plan['pins'][str(admission_path)] == args.admission_sha256,
            'Worker plan binds original adapter and reviewed admission')
    for path, digest in plan['pins'].items():
        require(sha(path) == digest, 'Changed worker input: ' + path)
    candidate = Path(plan['candidate'])
    require(candidate == plan_path.parent / 'estimator.py'
            and sha(candidate) == plan['candidate_sha256'], 'One fixed case source shared by all fresh workers')
    old.candidate_inverse(candidate, args.case)
    require(sys.flags.optimize == 0 and os.path.samefile(sys.executable, c.OFFPY)
            and os.getpriority(os.PRIO_PROCESS, 0) >= 10
            and all(os.environ.get(k) == '1' for k in c.BLAS)
            and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
            and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') == '0', 'Official nice10 BLAS1 worker frame')
    return plan, admission, expected_out


def worker_main(args, c, old):
    plan, admission, out = authenticate_worker(c, args, old)
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    record = dict(schema=c.SCHEMA, stage='native_worker', case=args.case,
        worker_index=plan['worker_index'], sequence=plan['sequence'], has_repeat=plan['has_repeat'],
        passed=False, completed=False, hashes_unchanged=False, sources_unchanged=False,
        admission_sha256=args.admission_sha256, source_bindings=dict(admission['source_bindings']),
        input_bindings=dict(plan['pins']), artifacts={}, target_bindings={},
        training_targets_accessed=False, check_targets_accessed=False, validation_targets_accessed=False,
        alltruth_accessed=False, truth_accessed=False, MPS_used=False, native_score_measured=False,
        candidate_sha256=plan['candidate_sha256'], coordinator_pid=os.getppid(),
        worker_pid=os.getpid(), process_group_id=os.getpgrp(), own_complete_bills=False)
    record['input_bindings'][str(Path(args.worker_plan).resolve())] = args.worker_plan_sha256
    c.write(out / 'started.json', record)
    try:
        source, proof = adapted_native_source(old)
        require(proof == plan['source_proof'], 'Worker source inverse exactly matches sealed coordinator proof')
        record.update(proof)
        gates = WorkerGates(old, plan, plan['references'])

        def candidate_admission(actual, inputs):
            require(actual == Path(plan['candidate']) and sha(actual) == plan['candidate_sha256'], 'Worker fixed source')
            for path, digest in record['input_bindings'].items():
                require(sha(path) == digest and (path not in inputs or inputs[path] == digest),
                        'Worker admission before estimator import: ' + path)
                inputs[path] = digest

        def environment(np):
            require(np.__version__ == '2.4.6' and 'torch' not in sys.modules, 'Official NumPy-only native frame')
            record['versions'] = dict(numpy=np.__version__, flopscope='0.12.0', whestbench='0.16.0')

        def metadata():
            return dict(stage='native', case=args.case, completed=False, sources_unchanged=False,
                native_adapter_sha256=sha(__file__), worker_index=plan['worker_index'],
                explicit_sequence=plan['sequence'], has_repeat=plan['has_repeat'],
                prerequisites=plan['prerequisites'], candidate_payload_sha256=plan['payload_sha256'],
                expected_initialization_flops=INITIALIZATION, alltruth_accessed=False,
                truth_accessed=False, **proof)

        namespace = dict(__name__='nd2_owned_official_meter', __file__=str(old.ORIGINAL),
            _nd_candidate_admission=candidate_admission, _nd_ranks=tuple(dict.fromkeys(plan['sequence'])),
            _cold_sequence=tuple(plan['sequence']), _cold_has_repeat=plan['has_repeat'],
            _nd_environment=environment, _nd_result_metadata=metadata,
            _nd_trial_gate=gates.trial, _nd_final_gate=gates.finish, Observer=old.Observer)
        exec(compile(source, str(old.ORIGINAL) + ':nd2-cold-admitted', 'exec'), namespace)
        require(namespace['HERE'] == old.BD and namespace['PARENT'] == old.NEW, 'Original native-v2 path ownership')
        gc.collect()
        previous_argv = sys.argv
        try:
            sys.argv = [str(old.ORIGINAL), '--source', plan['candidate'], '--split', 'development',
                        '--out', str(out / 'meter')]
            status = namespace['main']()
        finally:
            sys.argv = previous_argv
        rp, cp = out / 'meter/result.json', out / 'meter/completion.json'
        meter, completion = c.read(rp), c.read(cp)
        record['artifacts'][str(rp)], record['artifacts'][str(cp)] = sha(rp), sha(cp)
        record['native_meter_result'] = dict(path=str(rp), sha256=sha(rp))
        for path, digest in meter['source_bindings'].items():
            require(sha(path) == digest, 'Worker meter source/input changed')
            record['input_bindings'][path] = digest
        for path, digest in meter['artifacts'].items():
            require(sha(path) == digest, 'Worker meter artifact changed')
            record['artifacts'][path] = digest
        require(status == 0 and meter['passed'] is True and meter['completed'] is True
                and meter['hashes_unchanged'] is True and meter['sources_unchanged'] is True
                and not meter.get('error') and completion['passed'] is True and completion['completed'] is True
                and completion['result_sha256'] == sha(rp), 'Every worker native gate must pass')
        require(meter['initialization_flops'] == (INITIALIZATION if plan['has_repeat'] else None),
                'Singleton initialization is fully billed, never inferred or subtracted')
        record.update(own_complete_bills=True, all_primary_cold=True, threadpools=meter['threadpools'],
                      sources_unchanged=True, passed=True, completed=True)
    except BaseException:
        record.update(passed=False, completed=False, sources_unchanged=False,
                      hashes_unchanged=False, error=traceback.format_exc())
        # Keep the failed native bill/metrics if original native-v2 reached them.
        rp = out / 'meter/result.json'
        if rp.exists():
            record['native_meter_result'] = dict(path=str(rp), sha256=sha(rp))
            record['artifacts'][str(rp)] = sha(rp)
    finally:
        try:
            for group in ('source_bindings', 'input_bindings', 'artifacts'):
                for path, digest in record[group].items():
                    require(sha(path) == digest, 'Final worker provenance changed: ' + path)
            record['hashes_unchanged'] = True
            record['peak_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            record['elapsed_s'] = time.monotonic() - started
            require(record['peak_rss_bytes'] < 8_000_000_000,
                    'Worker lifetime RSS after native serialization and final provenance hashing')
        except BaseException:
            record.update(passed=False, completed=False, hashes_unchanged=False,
                          sources_unchanged=False, final_check_error=traceback.format_exc())
        c.write(out / 'result.json', record)
        c.write(out / 'completion.json', dict(passed=record['passed'], completed=record['completed'],
                                              result_sha256=sha(out / 'result.json')))
    return 0 if record['passed'] else 1


def collect_worker(stage, c, folder, worker_index, case):
    """Preserve attempted metrics before the coordinator tests worker PASS."""
    rp, cp = folder / 'result.json', folder / 'completion.json'
    require(rp.exists() and cp.exists(), 'Worker must seal execution or failure')
    digest = stage.pin(rp, group='artifacts')
    stage.pin(cp, group='artifacts')
    record, completion = c.read(rp), c.read(cp)
    require(completion['result_sha256'] == digest and completion['passed'] == record['passed']
            and completion['completed'] == record['completed']
            and record['case'] == case and record['worker_index'] == worker_index,
            'Worker completion identity')
    stage.inventory(record, targets=False)
    stage.record['worker_results'].append(dict(path=str(rp), sha256=digest,
        completion_sha256=sha(cp), sequence=record['sequence'], passed=record['passed'],
        completed=record['completed'], peak_rss_bytes=record.get('peak_rss_bytes')))
    info = record.get('native_meter_result')
    if info is None:
        return record, None
    stage.pin(info['path'], info['sha256'], group='artifacts')
    meter = c.read(info['path'])
    for path, value in meter.get('source_bindings', {}).items():
        stage.pin(path, value)
    for path, value in meter.get('artifacts', {}).items():
        stage.pin(path, value, group='artifacts')
    for old_trial in meter['trials']:
        local = old_trial['call']
        repeat = worker_index == 0 and local == len(schedules(case)[0]) - 1
        primary = local == 0
        ordinal = len(stage.record['trials'])
        prediction = dict(path=old_trial['output'], sha256=old_trial['output_sha256'],
                          raw_sha256=old_trial['output_bytes_sha256'], dtype='float32', shape=[16, 1024])
        trial = dict(old_trial)
        trial.update(call=ordinal, worker_call=local, worker_index=worker_index,
            prediction=prediction, effective_flops=old_trial['flops'], repeat=repeat, primary=primary,
            kind='repeat' if repeat else ('cold' if primary else 'warm_reference'))
        stage.record['trials'].append(trial)
        if not repeat:
            require(str(old_trial['rank']) not in stage.record['outputs'], 'Unique primary/reference output')
            stage.record['outputs'][str(old_trial['rank'])] = dict(prediction,
                effective_flops=old_trial['flops'], call=ordinal, worker_index=worker_index,
                includes_initialization=old_trial['includes_initialization'], primary=primary)
    for item in meter['input_preparation']:
        stage.record['input_preparation'].append(dict(item, worker_index=worker_index))
    return record, meter


def coordinator_main(args, c, old):
    out = HERE / ('native_' + args.case + '_v1')
    stage = c.Stage(__file__, args.admission_sha256, 'native', out)
    child = None
    try:
        stage.environment('OFFPY')
        require(c.PROTOCOL_SHA == PROTOCOL_SHA and c.FIT == FIT and c.CHECK == CHECK,
                'Compiled cold-primary protocol and split')
        stage.pin(OLD, OLD_SHA, 'source_bindings')
        stage.pin(old.ORIGINAL, old.ORIGINAL_SHA, 'source_bindings')
        stage.record.update(case=args.case, truth_accessed=False, alltruth_accessed=False,
            native_score_measured=False, own_complete_bills=False, all_primary_cold=False,
            primary_ranks=list(PRIMARY[args.case]), execution_roster=list(EXECUTION[args.case]),
            initialization_once=False, initialization_once_per_worker=True,
            initialization_subtracted_from_primary=False, repeat_excluded=True, frozen_cstar=True,
            trials=[], outputs={}, worker_results=[], worker_launches=[], input_preparation=[],
            baseline_reference_passed=False, all_domain_checks_passed=False, all_selector_checks_passed=False,
            repeat_byte_exact=False, repeat_diagnostics_byte_exact=False, sources_unchanged=False,
            long_lived_process_resource_qualified=False, coordinator_numerical_imports=False,
            process_group_id=os.getpgrp(), sequential_native_children=True)
        payload, seals = prerequisites(stage, args, old)
        stage.record.update(payload=payload, prerequisites=seals)
        references = old.reference_records(stage)
        candidate, build = old.build_candidate(stage, out, args.case, payload)
        _, proof = adapted_native_source(old)
        stage.record.update(candidate=str(candidate), candidate_sha256=build['candidate_sha256'],
                            build_sha256=sha(out / 'build.json'), **proof)
        first_meter = None
        for index, sequence in enumerate(schedules(args.case)):
            stage.recheck()
            require(not any(k in sys.modules for k in NUMERICAL), 'Coordinator stays stdlib-only')
            pins = {}
            for group in ('source_bindings', 'input_bindings', 'artifacts'):
                for path, digest in stage.record[group].items():
                    require(path not in pins or pins[path] == digest, 'Unambiguous worker provenance')
                    pins[path] = digest
            folder = out / f'worker_{index:02d}'
            plan = dict(schema=c.SCHEMA + '_worker_plan', admission_sha256=args.admission_sha256,
                case=args.case, worker_index=index, sequence=list(sequence), has_repeat=index == 0,
                coordinator_pid=os.getpid(), process_group_id=os.getpgrp(), directory=str(folder),
                candidate=str(candidate), candidate_sha256=build['candidate_sha256'],
                payload_sha256=payload['sha256'], prerequisites=seals, references=references,
                source_proof=proof, pins=pins)
            plan_info = stage.attempt(f'worker_{index:02d}_plan.json', plan)
            command = [str(c.OFFPY), '-u', str(Path(__file__).resolve()), '--run', '--case', args.case,
                '--admission-sha256', args.admission_sha256, '--worker-plan', plan_info['path'],
                '--worker-plan-sha256', plan_info['sha256']]
            log = out / f'worker_{index:02d}.log'
            launch = dict(worker_index=index, sequence=list(sequence), command=command,
                          inherited_process_group_id=os.getpgrp(), completed=False)
            stage.record['worker_launches'].append(launch)
            with log.open('x') as stream:
                # Deliberately no start_new_session/process_group override: the
                # outer watchdog owns this coordinator and every numerical child.
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT)
                launch['child_pid'] = child.pid
                while child.poll() is None:
                    require(time.monotonic() - stage.start < 1200, 'Case cannot exceed whole-sequence wall ceiling')
                    time.sleep(.25)
                status = child.wait(timeout=3)
                launch['exit_code'] = status
                child = None
            stage.pin(log, group='artifacts')
            record, meter = collect_worker(stage, c, folder, index, args.case)
            # Receipt() repeats full source/input/artifact binding before another
            # worker may start; no partial worker result is reusable for scoring.
            stage.receipt(folder, sha(folder / 'result.json'), 'native_worker', args.case)
            require(status == 0 and meter is not None and record['sources_unchanged'] is True
                    and record['own_complete_bills'] is True and meter['passed'] is True
                    and meter['completed'] is True and meter['hashes_unchanged'] is True
                    and meter['sources_unchanged'] is True and not meter.get('error'),
                    'Every isolated native worker must pass before continuation')
            require([t['rank'] for t in meter['trials']] == list(sequence)
                    and meter['initialization_flops'] == (INITIALIZATION if index == 0 else None),
                    'Exact worker roster and measured reference initialization')
            require(all(meter[k] is True for k in ('all_domain_checks_passed',
                    'all_selector_checks_passed', 'own_complete_bills', 'all_primary_cold')),
                    'Full native domain/selector/own-bill gates')
            if index == 0:
                first_meter = meter
                stage.record['versions'] = record['versions']
                stage.record['threadpools'] = meter['threadpools']
            else:
                require(record['versions'] == stage.record['versions']
                        and meter['threadpools'] == stage.record['threadpools'], 'Identical worker numerical frames')
            launch['completed'] = True
        require([t['rank'] for t in stage.record['trials']] == list(EXECUTION[args.case])
                and set(stage.record['outputs']) == {str(r) for r in PRIMARY[args.case] +
                    ((100,) if args.case == 'baseline' else ())}, 'Complete aggregate execution roster')
        require(all(stage.record['outputs'][str(r)]['includes_initialization'] is True
                    and stage.record['outputs'][str(r)]['primary'] is True for r in PRIMARY[args.case]),
                'Every scored primary owns its complete measured cold bill')
        require(first_meter['repeat_byte_exact'] is True and first_meter['repeat_diagnostics_byte_exact'] is True
                and (args.case != 'baseline' or first_meter['baseline_reference_passed'] is True),
                'Fixed reference child gates cannot be substituted by singletons')
        for key in ('input_representation', 'input_preparation_in_participant_bill', 'reset_scope',
                    'candidate_state_reset', 'baseline_reference_passed', 'baseline_reference_checks'):
            stage.record[key] = first_meter[key]
        stage.record.update(initialization_flops=INITIALIZATION, own_complete_bills=True,
            all_primary_cold=True, all_domain_checks_passed=True, all_selector_checks_passed=True,
            repeat_byte_exact=True, repeat_diagnostics_byte_exact=True, sources_unchanged=True,
            empty_threadpool_inventory_is_unknown=not stage.record['threadpools'],
            native_processes=len(schedules(args.case)), native_calls=len(EXECUTION[args.case]))
        stage.finish(passed=True, scientific=False)
    except BaseException:
        stage.record.update(passed=False, completed=False, sources_unchanged=False,
                            hashes_unchanged=False, error=traceback.format_exc())
    finally:
        if child is not None:
            try:
                child.terminate()
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait(timeout=3)
            except BaseException:
                stage.record['child_cleanup_error'] = traceback.format_exc()
        return stage.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--case', choices=tuple(PRIMARY))
    for name in ('admission-sha256', 'qualification-result-sha256', 'baseline-result-sha256',
                 'proposal-result-sha256', 'interpolation-result-sha256', 'worker-plan', 'worker-plan-sha256'):
        parser.add_argument('--' + name)
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    require(args.case in PRIMARY and isinstance(args.admission_sha256, str)
            and len(args.admission_sha256) == 64, 'Explicit case and admission digest')
    c = module(HERE / 'common.py', 'nd2_common_native')
    old = owner()
    if args.worker_plan is not None:
        require(isinstance(args.worker_plan_sha256, str) and len(args.worker_plan_sha256) == 64
                and all(getattr(args, k) is None for k in ('qualification_result_sha256',
                    'baseline_result_sha256', 'proposal_result_sha256', 'interpolation_result_sha256')),
                'Internal worker accepts only its sealed coordinator plan')
        return worker_main(args, c, old)
    require(args.worker_plan_sha256 is None, 'No unused worker plan digest')
    needed = [args.qualification_result_sha256]
    if args.case != 'baseline':
        needed += [args.baseline_result_sha256, args.proposal_result_sha256]
    if args.case == 'candidate':
        needed.append(args.interpolation_result_sha256)
    require(all(isinstance(x, str) and len(x) == 64 for x in needed), 'Complete case-specific prerequisite digests')
    require(args.case == 'candidate' or args.interpolation_result_sha256 is None, 'No unused interpolation')
    require(args.case != 'baseline' or (args.baseline_result_sha256 is None and args.proposal_result_sha256 is None),
            'Baseline does not depend on proposal')
    return coordinator_main(args, c, old)


if __name__ == '__main__':
    raise SystemExit(main())
