"""Fixed native-objective acquisition; inert unless explicitly admitted.

The candidate is the frozen d50 source with its one sealed P literal replaced.
The official-input native_v2 meter is adapted reversibly for admission, the
fixed rank roster, and read-only observations.  No surrogate model, target,
optimizer, covariance override, or candidate numerical operation is imported.
"""
import argparse
import ast
import gc
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import struct
import sys
import traceback
import zipfile

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
BD = NEW / 'birth_direction_fit_v1'
BR = NEW / 'current_d50_native_p_bridge_v1'
ORIGINAL = BD / 'native_v2.py'
ORIGINAL_SHA = '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
PARENT = BD / 'build_terminal_v1/estimator.py'
PARENT_SHA = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'
REFERENCE = BR / 'run_null_v1/result.json'
REFERENCE_SHA = '039c98a5acffa4e88440ef331b7efafd78b4c56163043dcbc670c0eedc1cd333'
REFERENCE_COMPLETION_SHA = '315a134c516dc3f44a243446c130306641a244dc0ff5d305604457ab5910bc48'
INITIALIZATION = 1383913786
FIT = (35, 36, 38, 40)
CHECK = (41, 43, 44, 45)
ROSTERS = dict(baseline=FIT + CHECK + (100,), plus=FIT, minus=FIT,
               candidate=FIT + CHECK)
PKEYS = ('msg_w', 'msg_b', 'enc0_w', 'enc0_b', 'enc2_w', 'enc2_b',
         'enc4_w', 'enc4_b', 'head', 'head_b', 'gain')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    path = Path(path)
    require('truthpost' not in path.parts and not
            (path.suffix in ('.npz', '.npy') and 'truth' in path.name.lower()),
            'Native acquisition cannot authenticate or read target arrays')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def assignment_span(source, name):
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]
    require(len(nodes) == 1 and len(nodes[0].targets) == 1, 'Unique literal assignment')
    node, lines = nodes[0], source.splitlines(keepends=True)
    return (sum(map(len, lines[:node.lineno - 1])) + node.col_offset,
            sum(map(len, lines[:node.end_lineno - 1])) + node.end_col_offset)


def candidate_inverse(candidate, case):
    """Check source bytes without evaluating any numerical payload literal."""
    require(sha(PARENT) == PARENT_SHA, 'Frozen d50 source changed')
    old, new = PARENT.read_text(), Path(candidate).read_text()
    a, b = assignment_span(old, '_GNN_LITERAL')
    c, d = assignment_span(new, '_GNN_LITERAL')
    require(a == c and new[:c] + old[a:b] + new[d:] == old,
            'Only the complete P literal may change')
    if case == 'baseline':
        require(new == old, 'Baseline source must be byte-exact d50')
    return dict(candidate_sha256=sha(candidate), parent_sha256=PARENT_SHA,
                inverse_source_byte_exact=True, all_other_source_byte_exact=True,
                frozen_cstar=True, only_candidate_edit='_GNN_LITERAL')


def literal_data(value):
    """Canonical FP32 bytes and shape of a nested Python literal; no NumPy."""
    if not isinstance(value, (list, tuple)):
        require(isinstance(value, (int, float)) and math.isfinite(value), 'Finite real literal')
        return (), struct.pack('<f', value)
    require(len(value) > 0, 'Nonempty parameter dimensions')
    parts = [literal_data(item) for item in value]
    shape = parts[0][0]
    require(all(item[0] == shape for item in parts), 'Rectangular parameter literal')
    return (len(value),) + shape, b''.join(item[1] for item in parts)


def unpack_f32_npy(data, expected):
    """Read the fixed sealed FP32 NPY format using only stdlib byte operations."""
    require(data[:6] == b'\x93NUMPY' and data[6:8] in (b'\x01\x00', b'\x02\x00'), 'NPY version')
    length_bytes = 2 if data[6] == 1 else 4
    length = int.from_bytes(data[8:8 + length_bytes], 'little')
    require(0 < length < 65536, 'Bounded NPY header')
    offset = 8 + length_bytes + length
    header = ast.literal_eval(data[8 + length_bytes:offset].decode('latin1'))
    require(set(header) == {'descr', 'fortran_order', 'shape'}
            and header['descr'] == '<f4' and isinstance(header['fortran_order'], bool),
            'Exact little-endian FP32 parameter storage')
    shape = header['shape']
    require(isinstance(shape, tuple) and len(shape) <= 3
            and all(isinstance(n, int) and n > 0 for n in shape)
            and expected['dtype'] == 'float32' and list(shape) == expected['shape'],
            'Sealed parameter shape/precision')
    size, raw = math.prod(shape), data[offset:]
    require(size <= 1000000 and len(raw) == 4 * size, 'Exact bounded FP32 byte length')
    if header['fortran_order'] and len(shape) > 1:
        strides = [math.prod(shape[:axis]) for axis in range(len(shape))]
        positions = (sum(i * stride for i, stride in zip(index, strides))
                     for index in itertools.product(*(range(n) for n in shape)))
        raw = b''.join(raw[4 * pos:4 * pos + 4] for pos in positions)
    require(hashlib.sha256(raw).hexdigest() == expected['raw_sha256'], 'Sealed parameter raw bytes')
    values = struct.unpack('<' + str(size) + 'f', raw)
    require(all(math.isfinite(v) for v in values), 'Finite supplied payload')

    def nested(axis, begin):
        if axis == len(shape):
            return values[begin]
        stride = math.prod(shape[axis + 1:])
        return [nested(axis + 1, begin + i * stride) for i in range(shape[axis])]

    return nested(0, 0), shape, raw


def build_candidate(stage, out, case, payload):
    """Literal-only private build, sealed before importing FNP or NumPy."""
    stage.pin(PARENT, PARENT_SHA)
    stage.pin(payload['path'], payload['sha256'])
    original = PARENT.read_text()
    nodes = [node for node in ast.parse(original).body if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == '_GNN_LITERAL' for t in node.targets)]
    require(len(nodes) == 1, 'Unique original P literal')
    old = ast.literal_eval(nodes[0].value)
    require(set(payload['arrays']) == set(old) and set(PKEYS).issubset(old), 'Complete payload keys')
    current, arrays = {}, {}
    with zipfile.ZipFile(payload['path']) as archive:
        require(len(archive.infolist()) == len(old)
                and set(archive.namelist()) == {key + '.npy' for key in old}
                and sum(item.file_size for item in archive.infolist()) < 16000000,
                'Exact bounded payload archive members')
        for key in old:
            value, shape, raw = unpack_f32_npy(archive.read(key + '.npy'), payload['arrays'][key])
            original_shape, original_raw = literal_data(old[key])
            require(shape == original_shape, 'Unchanged architecture shape: ' + key)
            if key not in PKEYS or case == 'baseline':
                require(raw == original_raw, 'Frozen field/null payload: ' + key)
            if key in ('head', 'head_b', 'gain'):
                require(shape[0] == 16, 'Inactive layer0 ownership')
                row_bytes = 4 * math.prod(shape[1:])
                require(raw[:row_bytes] == original_raw[:row_bytes], 'Inactive layer0 changed: ' + key)
            current[key] = value
            arrays[key] = dict(shape=list(shape), dtype='float32',
                               raw_sha256=hashlib.sha256(raw).hexdigest())
    require(sum(math.prod(arrays[k]['shape']) for k in PKEYS) == 45200, 'Fixed P parameter count')
    require(sha(payload['path']) == payload['sha256'], 'Payload changed during literal build')
    a, b = assignment_span(original, '_GNN_LITERAL')
    text = original if case == 'baseline' else original[:a] + '_GNN_LITERAL = ' + repr(current) + original[b:]
    candidate = out / 'estimator.py'
    with candidate.open('x') as stream:
        stream.write(text)
    proof = candidate_inverse(candidate, case)
    proof.update(schema='current_d50_native_directional_literal_v1', case=case,
                 passed=True, completed=True, payload=payload, arrays=arrays,
                 payload_literal_array_byte_exact=True, non_P_fields_byte_exact=True,
                 inactive_layer0_byte_exact=True, allocated_P_parameters=45200,
                 numerical_imports=False, candidate_calls=0,
                 payload_binding='Original FNP binding once in first participant context')
    build_path = out / 'build.json'
    write(build_path, proof)
    stage.pin(candidate, proof['candidate_sha256'], group='artifacts')
    stage.pin(build_path, group='artifacts')
    return candidate, proof


def reference_records(stage):
    stage.pin(REFERENCE, REFERENCE_SHA)
    stage.pin(REFERENCE.with_name('completion.json'), REFERENCE_COMPLETION_SHA)
    result = json.loads(REFERENCE.read_text())
    completion = json.loads(REFERENCE.with_name('completion.json').read_text())
    require(result['passed'] is True and result['hashes_unchanged'] is True and 'error' not in result
            and completion['passed'] is True and completion['result_sha256'] == REFERENCE_SHA
            and result['candidate_sha256'] == PARENT_SHA and result['truth_accessed'] is False
            and result['initialization_flops'] == INITIALIZATION
            and [t['rank'] for t in result['trials']] == [35, 100, 35],
            'Qualified unchanged d50 native reference')
    require([t['flops'] for t in result['trials']] ==
            [471955793128, 468515344292, 470571879342], 'Fixed cold/warm reference bills')
    for path, digest in result['artifacts'].items():
        stage.pin(path, digest)
    for path, digest in result['source_bindings'].items():
        if Path(path).name in ('weights.npy', 'row_manifest.json', 'manifest.json'):
            stage.pin(path, digest)
    return result['trials']


class Observer:
    """Retain existing small outputs; all decoding is after BudgetContext exit."""
    def __init__(self, module):
        self.module = module
        self.variances, self.selectors = [], []
        original_gaussian, original_select = module.gaussian_raw, module.select

        def gaussian(mu, variance):
            self.variances.append(variance)
            return original_gaussian(mu, variance)

        def select(score, floor):
            answer = original_select(score, floor)
            self.selectors.append((answer[0], answer[1], float(floor)))
            return answer

        module.gaussian_raw, module.select = gaussian, select

    def clear(self):
        self.variances.clear()
        self.selectors.clear()

    def materialize(self, np):
        variance_records, selector_records = {}, {}
        for layer, value in enumerate(self.variances):
            raw = value.tobytes()
            v = np.frombuffer(raw, dtype=value.dtype).reshape(value.shape)
            finite = bool(np.isfinite(v).all())
            variance_records[str(layer)] = dict(
                dtype=str(v.dtype), shape=list(v.shape), finite=finite,
                minimum=float(v.min()) if finite and v.size else None,
                below_floor_count=int((v < 1e-10).sum()),
                raw_sha256=hashlib.sha256(raw).hexdigest())
        keys = [f'{l}:{s}' for l in range(16) for s in range(2 if l == 15 else 3)]
        for ordinal, (indices, mask, floor) in enumerate(self.selectors):
            ib, mb = indices.tobytes(), mask.tobytes()
            ids = np.frombuffer(ib, dtype=indices.dtype).reshape(indices.shape)
            members = np.frombuffer(mb, dtype=mask.dtype).reshape(mask.shape)
            key = keys[ordinal] if ordinal < len(keys) else 'extra:' + str(ordinal)
            selector_records[key] = dict(
                ordered_ids=ids.tolist(), membership=np.flatnonzero(members).tolist(),
                count=int(ids.size), indices_dtype=str(ids.dtype),
                indices_shape=list(ids.shape), mask_dtype=str(members.dtype),
                mask_shape=list(members.shape), floor=floor,
                indices_sha256=hashlib.sha256(ib).hexdigest(),
                mask_sha256=hashlib.sha256(mb).hexdigest())
        record = dict(variance_domain=variance_records, selection_cells=selector_records,
                      variance_count=len(self.variances), selector_count=len(self.selectors),
                      ownership='Existing FNP outputs retained; decoded only after closed budget',
                      added_participant_array_operations=0)
        self.clear()
        return record


def check_diagnostics(record):
    v = record['variance_domain']
    require(record['variance_count'] == 16 and set(v) == {str(l) for l in range(16)},
            'All16 native variance entries required')
    require(all(x['dtype'] == 'float64' and x['shape'] == [1024] and x['finite']
                and x['minimum'] is not None and x['minimum'] >= 1e-10
                and x['below_floor_count'] == 0 for x in v.values()),
            'Native positive prevariance domain; no floor permitted')
    selectors = record['selection_cells']
    keys = {f'{l}:{s}' for l in range(16) for s in range(2 if l == 15 else 3)}
    require(record['selector_count'] == 47 and set(selectors) == keys,
            'Exact all-layer selector roster')
    for key, entry in selectors.items():
        ids = entry['ordered_ids']
        require(entry['indices_dtype'] in ('int32', 'int64')
                and entry['indices_shape'] == [entry['count']]
                and entry['mask_dtype'] == 'bool' and entry['mask_shape'] == [1024]
                and entry['floor'] == .01 and entry['count'] % 32 == 0
                and 0 <= entry['count'] <= 1024 and len(set(ids)) == len(ids)
                and sorted(ids) == entry['membership']
                and all(isinstance(i, int) and 0 <= i < 1024 for i in ids),
                'Native selector membership/order: ' + key)
    require(all(selectors[f'0:{s}']['ordered_ids'] == list(range(1024)) for s in range(3)),
            'Gaussian first layer must retain all coordinates')


def adapted_native_source():
    """Literal inverse for every harness edit; estimator source is not rewritten."""
    require(sha(ORIGINAL) == ORIGINAL_SHA, 'Frozen official-input meter changed')
    original = ORIGINAL.read_text()
    source, edits = original, []

    def replace(old, new):
        nonlocal source
        require(source.count(old) == 1, 'Unique frozen meter span: ' + old[:80])
        source = source.replace(old, new)
        edits.append((old, new))

    start = source.index('    if candidate==BANK:\n')
    stop = source.index('    manifest=json.loads(', start)
    replace(source[start:stop], '    _nd_candidate_admission(candidate,inputs)\n    ranks=_nd_ranks\n')
    replace("    write(args.out/'started.json',result)\n",
            "    result.update(_nd_result_metadata())\n    write(args.out/'started.json',result)\n")
    replace('        estimator=None\n',
            "        _nd_environment(np)\n        observer=Observer(module)\n        estimator=None\n")
    replace('                started=time.monotonic()\n',
            '                observer.clear()\n                started=time.monotonic()\n')
    replace("                trial['peak_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)\n",
            "                diagnostic=observer.materialize(np)\n"
            "                diagnostic_path=args.out/f'call_{call:02d}_state.json'\n"
            "                write(diagnostic_path,diagnostic)\n"
            "                result['artifacts'][str(diagnostic_path)]=sha(diagnostic_path)\n"
            "                trial.update(state_diagnostic=str(diagnostic_path),state_diagnostic_sha256=sha(diagnostic_path))\n"
            "                trial['peak_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)\n")
    replace('                guard(trial)\n',
            '                guard(trial)\n                _nd_trial_gate(trial,diagnostic)\n')
    replace('        result.update(passed=True,hashes_unchanged=True,\n',
            '        _nd_final_gate(result)\n'
            '        result.update(passed=True,completed=True,sources_unchanged=True,hashes_unchanged=True,\n')
    replace("    except BaseException:\n        result['error']=traceback.format_exc()\n",
            "    except BaseException:\n"
            "        result.update(passed=False,completed=False,sources_unchanged=False,hashes_unchanged=False)\n"
            "        result['error']=traceback.format_exc()\n")
    replace("    write(args.out/'completion.json',dict(passed=result['passed'],result_sha256=sha(args.out/'result.json')))\n",
            "    write(args.out/'completion.json',dict(passed=result['passed'],completed=result['completed'],result_sha256=sha(args.out/'result.json')))\n")
    restored = source
    for old, new in reversed(edits):
        require(restored.count(new) == 1, 'Unique inverse meter span')
        restored = restored.replace(new, old)
    require(restored == original, 'Whole native meter source inverse')
    # Explicit sub-block certificates prevent observer/admission edits from
    # changing numerical input preparation or the scored prediction boundary.
    blocks = {}
    for name, begin, end in (
        ('official_input', '                with flops.BudgetContext(flop_budget=BUDGET,wall_time_limit_s=120,quiet=True) as input_ctx:\n',
         '                input_record=dict('),
        ('participant', '                with flops.BudgetContext(flop_budget=BUDGET,wall_time_limit_s=120,quiet=True) as ctx:\n',
         '                trial=dict('),
        ('closed_bill_reset', '                context_ref=weakref.ref(ctx)\n',
         '                guard(trial)\n')):
        old_block = original[original.index(begin):original.index(end, original.index(begin))]
        new_block = source[source.index(begin):source.index(end, source.index(begin))]
        require(new_block == old_block, 'Changed paid/lifecycle block: ' + name)
        blocks[name] = hashlib.sha256(old_block.encode()).hexdigest()
    proof = dict(native_original_sha256=ORIGINAL_SHA,
                 native_adapted_sha256=hashlib.sha256(source.encode()).hexdigest(),
                 source_inverse_verified=True, paid_blocks_byte_exact=True,
                 paid_block_sha256=blocks, native_here=str(BD), native_file=str(ORIGINAL),
                 harness_edits=[dict(old_sha256=hashlib.sha256(a.encode()).hexdigest(),
                                     new_sha256=hashlib.sha256(b.encode()).hexdigest()) for a, b in edits])
    return source, proof


class TrialGates:
    def __init__(self, case, references):
        self.case, self.references = case, references
        self.first = self.first_diagnostic = None
        self.checked, self.reference_checks = [], []

    def trial(self, trial, diagnostic):
        call, rank = trial['call'], trial['rank']
        sequence = ROSTERS[self.case] + (35,)
        require(call == len(self.checked) and rank == sequence[call], 'Fixed ordered native calls')
        check_diagnostics(diagnostic)
        if call == 0:
            self.first, self.first_diagnostic = dict(trial), diagnostic
        if call == len(sequence) - 1:
            require(trial['output_bytes_sha256'] == self.first['output_bytes_sha256']
                    and trial['output_sha256'] == self.first['output_sha256'], 'Exact native repeat bytes')
            require(diagnostic == self.first_diagnostic, 'Exact repeat domain/selector bytes')
            require(self.first['flops'] - trial['flops'] == INITIALIZATION,
                    'Measured cold/warm initialization ownership')
        if self.case == 'baseline' and (call == 0 or rank == 100 or call == len(sequence) - 1):
            ref = self.references[0 if call == 0 else (1 if rank == 100 else 2)]
            require(trial['flops'] == ref['flops']
                    and trial['includes_initialization'] == ref['includes_initialization']
                    and trial['output_bytes_sha256'] == ref['output_bytes_sha256']
                    and trial['output_sha256'] == ref['output_sha256'],
                    'Exact baseline all16 output bytes and actual full cold/warm bill')
            self.reference_checks.append(dict(call=call, rank=rank, flops=trial['flops'],
                                              reference_call=ref['call'], passed=True))
        self.checked.append(call)

    def finish(self, result):
        require(self.checked == list(range(len(ROSTERS[self.case]) + 1)), 'Complete native roster')
        require(len(result['trials']) == len(self.checked) and len(result['input_preparation']) == len(self.checked),
                'Every prediction has an owned-input receipt')
        require(all(t['includes_initialization'] == (i == 0) for i, t in enumerate(result['trials'])),
                'Exactly one initialization')
        require(self.case != 'baseline' or len(self.reference_checks) == 3, 'Three exact BR reference checks')
        require('torch' not in sys.modules, 'Native acquisition cannot import Torch')
        result.update(all_domain_checks_passed=True, all_selector_checks_passed=True,
                      repeat_byte_exact=True, repeat_diagnostics_byte_exact=True,
                      baseline_reference_passed=(self.case == 'baseline'),
                      baseline_reference_checks=self.reference_checks,
                      own_complete_bills=True, native_score_measured=False,
                      alltruth_accessed=False, truth_accessed=False, MPS_used=False)


def prerequisites(stage, args):
    qualified = stage.receipt('qualification_v1', args.qualification_result_sha256, 'qualify')
    require(qualified['qualification_passed'] is True and qualified['qualified_cpu_function_only'] is True
            and qualified['qualified_ranks'] == [35, 100] and qualified['no_fit'] is True
            and qualified['all_task_states_released'] is True and qualified['parameter_updates'] == 0
            and qualified['native_gradient_qualified'] is False
            and all(qualified[k] is False for k in ('training_targets_accessed', 'validation_targets_accessed',
                                                  'check_targets_accessed')),
            'New target-free CPU-function qualification required, not a native gradient certificate')
    sealed = dict(qualification_result_sha256=args.qualification_result_sha256)
    payload = qualified['payloads']['baseline']
    if args.case != 'baseline':
        baseline = stage.receipt('native_baseline_v1', args.baseline_result_sha256, 'native', 'baseline')
        require(all(baseline[k] is True for k in ('baseline_reference_passed', 'all_domain_checks_passed',
                    'all_selector_checks_passed', 'repeat_byte_exact', 'own_complete_bills'))
                and baseline['initialization_flops'] == INITIALIZATION
                and baseline['truth_accessed'] is False and baseline['alltruth_accessed'] is False
                and [t['rank'] for t in baseline['trials']] == list(ROSTERS['baseline'] + (35,)),
                'Exact fresh baseline must precede every perturbed native case')
        proposal = stage.receipt('proposal_v1', args.proposal_result_sha256, 'propose')
        require(proposal['prerequisites'] == (sealed | {'baseline_result_sha256': args.baseline_result_sha256})
                and proposal['parameter_updates'] == 1 and proposal['adam_steps'] == 1
                and proposal['curve_h'] == 5e-4 and proposal['curve_endpoint_byte_exact'] is True
                and proposal['frozen_payload_fields_byte_exact'] is True
                and proposal['proposal_sealed'] is True and proposal['surrogate_only'] is True
                and proposal['no_native_step_selection'] is True
                and proposal['check_targets_accessed'] is False and proposal['validation_targets_accessed'] is False,
                'Only the one sealed prospective direction is admitted')
        sealed.update(baseline_result_sha256=args.baseline_result_sha256,
                      proposal_result_sha256=args.proposal_result_sha256)
        if args.case in ('plus', 'minus'):
            payload = proposal['payloads'][args.case]
        else:
            interpolation = stage.receipt('interpolation_v1', args.interpolation_result_sha256, 'interpolate')
            require(interpolation['check_targets_accessed'] is False
                    and interpolation['validation_targets_accessed'] is False,
                    'Candidate must be sealed without check/VALID target access')
            for key, value in sealed.items():
                require(interpolation['prerequisites'][key] == value, 'Interpolation lineage: ' + key)
            payload = interpolation['payloads']['candidate']
            sealed['interpolation_result_sha256'] = args.interpolation_result_sha256
    return payload, sealed


def collect_meter(stage, out, case):
    path = out / 'meter/result.json'
    if not path.exists():
        return None
    stage.pin(path)
    stage.pin(path.with_name('completion.json'))
    result = json.loads(path.read_text())
    require(result['case'] == case and result['source_inverse_verified'] is True,
            'Nested meter role/source ownership')
    for p, digest in result['source_bindings'].items():
        stage.pin(p, digest)
    for p, digest in result['artifacts'].items():
        stage.pin(p, digest, group='artifacts')
    trials, outputs = [], {}
    for old in result['trials']:
        trial = dict(old)
        prediction = dict(path=old['output'], sha256=old['output_sha256'],
                          raw_sha256=old['output_bytes_sha256'], dtype='float32', shape=[16, 1024])
        is_repeat = old['call'] == len(ROSTERS[case])
        trial.update(prediction=prediction, effective_flops=old['flops'], repeat=is_repeat,
                     kind='repeat' if is_repeat else ('cold' if old['call'] == 0 else 'warm'))
        trials.append(trial)
        if not trial['repeat']:
            outputs[str(old['rank'])] = dict(prediction, effective_flops=old['flops'], call=old['call'])
    stage.record.update(trials=trials, outputs=outputs,
                        native_meter_result=dict(path=str(path), sha256=sha(path)),
                        native_meter_passed=bool(result['passed']))
    for key in ('initialization_flops', 'input_preparation', 'input_representation',
                'input_preparation_in_participant_bill', 'reset_scope', 'candidate_state_reset',
                'threadpools', 'all_domain_checks_passed', 'all_selector_checks_passed',
                'baseline_reference_passed', 'baseline_reference_checks', 'repeat_byte_exact',
                'repeat_diagnostics_byte_exact', 'own_complete_bills'):
        if key in result:
            stage.record[key] = result[key]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--case', choices=tuple(ROSTERS))
    for name in ('admission-sha256', 'qualification-result-sha256', 'baseline-result-sha256',
                 'proposal-result-sha256', 'interpolation-result-sha256'):
        parser.add_argument('--' + name)
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    needed = [args.admission_sha256, args.qualification_result_sha256]
    if args.case != 'baseline':
        needed += [args.baseline_result_sha256, args.proposal_result_sha256]
    if args.case == 'candidate':
        needed.append(args.interpolation_result_sha256)
    require(args.case in ROSTERS and all(isinstance(x, str) and len(x) == 64 for x in needed),
            'Explicit fixed case and complete reviewed prerequisite digests')
    require(args.case == 'candidate' or args.interpolation_result_sha256 is None, 'No unused interpolation override')
    require(args.case != 'baseline' or (args.baseline_result_sha256 is None and args.proposal_result_sha256 is None),
            'Baseline may not depend on a proposal')
    require(sys.flags.optimize == 0, 'Assertions must remain enabled')
    spec = importlib.util.spec_from_file_location('native_directional_common_native', HERE / 'common.py')
    common = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = common
    spec.loader.exec_module(common)
    out = HERE / ('native_' + args.case + '_v1')
    stage = common.Stage(__file__, args.admission_sha256, 'native', out)
    try:
        stage.environment('OFFPY')
        require(common.FIT == FIT and common.CHECK == CHECK, 'Exact network split')
        stage.record.update(case=args.case, truth_accessed=False, alltruth_accessed=False,
                            native_score_measured=False, own_complete_bills=False,
                            initialization_once=True, repeat_excluded=True, frozen_cstar=True,
                            trials=[], outputs={}, baseline_reference_passed=False,
                            all_domain_checks_passed=False, all_selector_checks_passed=False,
                            repeat_byte_exact=False, sources_unchanged=False)
        stage.pin(ORIGINAL, ORIGINAL_SHA, group='source_bindings')
        payload, seals = prerequisites(stage, args)
        stage.record.update(prerequisites=seals, payload=payload)
        references = reference_records(stage)
        candidate, build = build_candidate(stage, out, args.case, payload)
        adapted, proof = adapted_native_source()
        stage.record.update(candidate=str(candidate), candidate_sha256=build['candidate_sha256'],
                            build_sha256=sha(out / 'build.json'), **proof)
        # Only metadata has been handled so far, including the literal build.
        require(not any(k in sys.modules for k in ('numpy', 'torch', 'flopscope', 'threadpoolctl')),
                'Complete source/payload/fixture admission before numerical imports')
        pins = {}
        for group in ('source_bindings', 'input_bindings', 'artifacts'):
            for path, digest in stage.record[group].items():
                require(path not in pins or pins[path] == digest, 'Conflicting native provenance')
                pins[path] = digest
        gates = TrialGates(args.case, references)

        def candidate_admission(actual, inputs):
            require(actual == candidate and sha(actual) == build['candidate_sha256'], 'Fixed private candidate')
            for path, digest in pins.items():
                require(sha(path) == digest and (path not in inputs or inputs[path] == digest),
                        'Changed admission before estimator import: ' + path)
                inputs[path] = digest

        def environment(np):
            require(np.__version__ == '2.4.6' and 'torch' not in sys.modules, 'Official native NumPy-only frame')
            stage.record['versions'] = dict(numpy=np.__version__, flopscope='0.12.0', whestbench='0.16.0')

        def metadata():
            return dict(stage='native', case=args.case, completed=False, sources_unchanged=False,
                        native_adapter_sha256=sha(Path(__file__)), prerequisites=seals,
                        candidate_payload_sha256=payload['sha256'], expected_initialization_flops=INITIALIZATION,
                        alltruth_accessed=False, truth_accessed=False, **proof)

        namespace = dict(__name__='nd_owned_official_meter', __file__=str(ORIGINAL),
                         _nd_candidate_admission=candidate_admission, _nd_ranks=ROSTERS[args.case],
                         _nd_environment=environment, _nd_result_metadata=metadata,
                         _nd_trial_gate=gates.trial, _nd_final_gate=gates.finish, Observer=Observer)
        exec(compile(adapted, str(ORIGINAL) + ':nd-admitted', 'exec'), namespace)
        require(namespace['HERE'] == BD and namespace['PARENT'] == NEW, 'Original native meter paths')
        gc.collect()
        previous_argv = sys.argv
        try:
            sys.argv = [str(ORIGINAL), '--source', str(candidate), '--split', 'development', '--out', str(out / 'meter')]
            status = namespace['main']()
        finally:
            sys.argv = previous_argv
        result = collect_meter(stage, out, args.case)
        require(status == 0 and result is not None and result['passed'] is True
                and result['completed'] is True and result['hashes_unchanged'] is True
                and result['sources_unchanged'] is True and 'error' not in result,
                'Every native prediction/resource/domain/repeat gate must pass')
        require(result['initialization_flops'] == INITIALIZATION
                and len(stage.record['outputs']) == len(ROSTERS[args.case]), 'Complete own-bill roster')
        stage.record.update(sources_unchanged=True, empty_threadpool_inventory_is_unknown=not result['threadpools'])
        stage.finish(passed=True, scientific=False)
    except BaseException:
        # A resource, source, or numerical failure is not a scientific no-advance
        # verdict and can never leave an outer PASS flag behind.
        stage.record.update(passed=False, completed=False, sources_unchanged=False,
                            hashes_unchanged=False, error=traceback.format_exc())
        try:
            collect_meter(stage, out, args.case)
        except BaseException:
            stage.record['meter_collection_error'] = traceback.format_exc()
    finally:
        return stage.close()


if __name__ == '__main__':
    raise SystemExit(main())
