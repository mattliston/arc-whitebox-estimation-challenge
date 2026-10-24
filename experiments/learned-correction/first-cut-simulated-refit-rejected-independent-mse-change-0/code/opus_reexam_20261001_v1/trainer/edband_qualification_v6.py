"""Evidence checks for the existing trainer_edge_band_fullcard_qualification_v1.

Source-only helper; imports are stdlib. A qualification is not constructed here.
Actual pinned worker outputs and native measurement evidence must exist first.
"""
import hashlib
import json
import math
from pathlib import Path
import struct


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_pin(item, parse=True):
    require(set(item) == {'path', 'sha256'} and Path(item['path']).is_absolute(), 'Qualification needs an exact absolute pin')
    require(Path(item['path']).name != 'key.txt', 'Forbidden credential path')
    raw = Path(item['path']).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == item['sha256'], 'Qualification artifact changed: ' + item['path'])
    return json.loads(raw) if parse else raw


def behavior(request):
    # Parameter identity is checked separately. All numerical flags and routing
    # metadata are retained, including ncar/ss/attach/unfreeze/lossw/truth_plan.
    return {k: v for k, v in request.items()
            if k not in {'id', 'rank', 'mode', 'scale', 'params', 'edge_band', 'keep_all', 'keep_final'}}


def prediction_bytes(record):
    value = record.get('prediction')
    require(isinstance(value, list) and len(value) == 16
            and all(isinstance(row, list) and len(row) == 1024
                    and all(type(x) in (int, float) and math.isfinite(x) for x in row) for row in value),
            'All16 finite actual predictions are required')
    return b''.join(struct.pack('<d', x) for row in value for x in row)


def verify_zero_control(qualification, config, expected_worker, require_identical=True):
    require(qualification.get('schema') == 'trainer_edge_band_fullcard_qualification_v1', 'Wrong qualification schema')
    require(all(key in qualification for key in ('latest_best_checkpoint', 'latest_best_request', 'zero_control', 'runtime_sources', 'native_evidence')),
            'Boolean-only qualification lacks retrievable zero/native/runtime evidence')
    latest_checkpoint = qualification['latest_best_checkpoint']
    read_pin(latest_checkpoint, parse=False)
    latest_request = read_pin(qualification['latest_best_request'])
    template = read_pin(config['request_template'])
    require(latest_request.get('params') == latest_checkpoint['path']
            and not latest_request.get('edge_band') and behavior(latest_request) == behavior(template),
            'Latest-best plain request differs from the proposed numerical behavior')
    control = qualification['zero_control']
    require(set(control) == {'plain_request', 'enabled_request', 'plain_result', 'enabled_result'}, 'Zero-control evidence fields')
    requests = [read_pin(control[name + '_request']) for name in ('plain', 'enabled')]
    records = [read_pin(control[name + '_result']) for name in ('plain', 'enabled')]
    plain, enabled = requests
    require(plain['rank'] == enabled['rank'] and plain['rank'] in (651, 378), 'Zero control must use a previously used parity rank')
    require(plain['id'] != enabled['id'], 'Distinct genuine zero-control request IDs required')
    require(plain['params'] == latest_checkpoint['path'] and not plain.get('edge_band')
            and enabled['params'] == config['checkpoint']['path']
            and enabled.get('edge_band') == template['edge_band'], 'Zero control checkpoint/spec binding')
    for request, record in zip(requests, records):
        require(request.get('mode') == 'eval' and request.get('keep_all') == 1
                and behavior(request) == behavior(template), 'Zero-control flags differ from the qualified template')
        require(not request.get('dump15') and not request.get('featstats'), 'Diagnostic side effects are not a fitting profile')
        for key in ('id', 'rank', 'mode', 'params', 'cutsim'):
            require(record.get(key) == request.get(key), 'Actual zero-control output has different ' + key)
        require(record.get('edge_band_worker') == expected_worker and record.get('threads') == config['runtime']['threads'],
                'Actual zero-control worker/source/thread identity mismatch')
        require(record.get('data_source') == 'public' and record.get('truth_metadata') is None
                and record.get('truth_plan') == request.get('truth_plan'), 'Zero-control public-data routing mismatch')
        require(type(record.get('mse')) in (int, float) and math.isfinite(record['mse']) and record['mse'] > 0
                and len(record.get('mse_layers', [])) == 16
                and all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in record['mse_layers']), 'Invalid zero-control MSEs')
        require('edge_band_grads' not in record, 'Eval control carries gradients')
    require(records[0].get('edge_band') == dict(schema='trainer_edge_band_provenance_v1', enabled=False,
        module_sha256=expected_worker['sources']['otr_edband_v1.py'], request=plain), 'Plain control did not use the disabled adapter path')
    identity = records[1].get('edge_band', {})
    expected_identity = dict(schema='trainer_edge_band_provenance_v1', enabled=True,
        design='shared_reference_13_14_two_offsets', edges=[[8, 9], [10, 12], [13, 14]],
        request=enabled, request_sha256=hashlib.sha256(json.dumps(enabled, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        spec=enabled['edge_band'], base_sha256=qualification['base_content_sha256'],
        payload_sha256=qualification['payload_sha256'])
    require(identity == expected_identity, 'Enabled control complete envelope identity mismatch')
    same = prediction_bytes(records[0]) == prediction_bytes(records[1])
    if require_identical:
        require(same and records[0]['mse'] == records[1]['mse'] and records[0]['mse_layers'] == records[1]['mse_layers'],
                'Actual all16 zero-control predictions/MSEs are not bit-identical')
    sources = qualification['runtime_sources']
    require(isinstance(sources, dict) and sources, 'Frozen runtime source closure required')
    for item in sources.values():
        read_pin(item, parse=False)
    require(all(any(Path(item['path']).name == name and item['sha256'] == sha for item in sources.values())
                for name, sha in expected_worker['sources'].items()), 'Qualified closure omits a pinned worker runtime dependency')
    return dict(control=control, enabled_identity=identity, rank=plain['rank'], latest_best_checkpoint=latest_checkpoint)



def json_value(value):
    # Match actual JSON publication: integer dictionary keys become strings
    # before comparison/sorting; retain raw objects separately for native AST.
    return json.loads(json.dumps(value, allow_nan=False))


def native_f32(item):
    """Read only the precise little-endian, C-order f32 NPY emitted by quick.py."""
    import ast
    raw = read_pin(item, parse=False)
    require(raw[:6] == b'\x93NUMPY' and raw[6:8] in (b'\x01\x00', b'\x02\x00'), 'Unexpected native NPY header')
    version = raw[6]
    offset = 10 if version == 1 else 12
    length = struct.unpack('<H' if version == 1 else '<I', raw[8:offset])[0]
    require(0 < length <= 4096, 'Unbounded NPY header')
    header = ast.literal_eval(raw[offset:offset+length].decode('latin1').strip())
    require(header == {'descr': '<f4', 'fortran_order': False, 'shape': (16, 1024)}, 'Native array must be f32 C(16,1024)')
    data = raw[offset+length:]
    require(len(data) == 16*1024*4, 'Native array byte count')
    values = struct.unpack('<16384f', data)
    require(all(math.isfinite(x) for x in values), 'Nonfinite native prediction')
    return [values[i*1024:(i+1)*1024] for i in range(16)], hashlib.sha256(data).hexdigest()


def verify_native(qualification, config, expected_worker, directory):
    import importlib.util
    directory = Path(directory)
    evidence = qualification['native_evidence']
    require(set(evidence) == {'export_complete', 'derived_build', 'native_source', 'quick_producer', 'parity_producer', 'cases'},
            'Native measurement evidence fields')
    require(evidence['quick_producer'] == dict(path=str(directory/'quick.py'),
        sha256='b75676cf6c158de8aeb6f2ef617df8bf10d6acbc5abd45f72b48941f9734c880'), 'Unreviewed native quick producer')
    require(evidence['parity_producer'] == dict(path=str(directory/'a22_cutsim/parity_report_s46_v1.py'),
        sha256='be338647c5655dd97d7a1a36ac25a9ced4ae9ed7477d104721f48a05f585c0e2'), 'Unreviewed parity producer')
    for name in ('quick_producer', 'parity_producer', 'native_source'):
        read_pin(evidence[name], parse=False)
    adapter_pin = dict(path=str(directory/'a56_edge_band/edband_s46_adapter_v3.py'),
        sha256='eb9b4519e67b7e9c5b80d2adfc4aaf884d22c324707e8b39deb382437a20ac54')
    read_pin(adapter_pin, parse=False)
    spec = importlib.util.spec_from_file_location('_edge_qualification_native_adapter', adapter_pin['path'])
    adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
    complete = read_pin(evidence['export_complete'])
    bundle = adapter.load_bundle(evidence['export_complete']['path'])
    build = read_pin(evidence['derived_build']); plan = read_pin(build['plan'])
    require(build.get('schema') == 'edband_s46_raw_parent_build_v1' and plan.get('schema') == 'edband_s46_raw_parent_plan_v1'
            and build['complete'] == plan['complete'] == evidence['export_complete']
            and build['output'] == evidence['native_source'] and plan['output'] == evidence['native_source']['path']
            and build['direct_parent'] == plan['direct_parent'] == complete['direct_parent']
            and plan['adapter'] == adapter_pin, 'Native source/export/build chain mismatch')
    prepared = bundle['prepared']
    require(prepared['originals']['checkpoint'] == config['checkpoint']
            and prepared['originals']['head'] == config['head']
            and complete['preserved_base_content_sha256'] == qualification['base_content_sha256']
            and complete['payload_sha256'] == qualification['payload_sha256'], 'Native build belongs to another checkpoint/head')
    export_request = read_pin(complete['required_request'])
    template = read_pin(config['request_template'])
    require(behavior(export_request) == behavior(template) and export_request['params'] == template['params']
            and export_request['edge_band'] == template['edge_band'], 'Native export request flags differ')
    for item in build['sources'] + build['assets']:
        read_pin(item, parse=False)
    base = read_pin(plan['base_geometry'])
    require(plan['variant'] in ('safe', 'engineer_twin'), 'Unsupported native variant')
    base['override'] = 'dz309' if plan['variant'] == 'safe' else 'kp10dz'
    require(plan['override']['path'] == str(directory/('a50_ovr/'+base['override']+'.json')), 'Native override binding')
    read_pin(plan['override']); read_pin(plan['stack_builder'], parse=False)
    token = plan['command'][-1]
    assets, expected = adapter.science_assets(bundle['physics'], bundle['layer'], bundle['head'], bundle['edge'], token, base)
    require(json_value(assets) == plan['assets']
            and json_value(expected) == plan['expected_final_literals'],
            'Declared native science assets do not reconstruct export')
    require({item['path'] for item in build['assets']} == {str(directory/name) for name in assets}, 'Native asset set mismatch')
    for item in build['assets']:
        require(read_pin(item) == json_value(assets[str(Path(item['path']).relative_to(directory))]),
                'Native asset value mismatch')
    checked = adapter.verify_build(bundle['parent_raw'], read_pin(evidence['native_source'], parse=False),
                                   read_pin(build['counted_build_receipt']), expected, bundle['edge'])
    require(checked == build['checks'], 'Native counted derivation/literal verification changed')
    cases = evidence['cases']
    require(isinstance(cases, list) and [case.get('rank') for case in cases] == [651, 378], 'Native evidence requires exactly651 then378')
    require(prepared['originals']['result'] in [case['torch_result'] for case in cases], 'Export result is not an actual measured enabled case')
    require(qualification['zero_control']['enabled_result'] in [case['torch_result'] for case in cases], 'Zero-control and native measurements are disconnected')
    output = []
    for case in cases:
        require(set(case) == {'rank', 'torch_request', 'torch_result', 'original_torch_request', 'original_torch_result',
                             'native_quick', 'native_guard', 'native_prediction', 'parity_report'}, 'Native case fields')
        q = dict(qualification, zero_control=dict(plain_request=case['original_torch_request'], plain_result=case['original_torch_result'],
                enabled_request=case['torch_request'], enabled_result=case['torch_result']))
        verify_zero_control(q, config, expected_worker, require_identical=False)
        request, result = read_pin(case['torch_request']), read_pin(case['torch_result'])
        original = read_pin(case['original_torch_result'])
        require(request['rank'] == original['rank'] == case['rank'], 'Native/Torch rank binding')
        quick, guard = read_pin(case['native_quick']), read_pin(case['native_guard'])
        require(quick.get('candidate') == evidence['native_source']['path'], 'Quick tested another native source')
        rows = quick.get('rows', [])
        require(rows and len({row.get('rank') for row in rows}) == len(rows), 'Duplicate native quick rows')
        ranks = [row['rank'] for row in rows]
        require(set(ranks) <= {651, 378} and case['rank'] in ranks, 'Unexpected native quick networks')
        row = next(row for row in rows if row['rank'] == case['rank'])
        require(type(row.get('flops')) is int and 0 < row['flops'] <= 2**41
                and type(row.get('ops')) is int and row['ops'] > 0
                and type(row.get('mse')) in (int, float) and math.isfinite(row['mse']) and row['mse'] > 0, 'Invalid actual flopscope quick row')
        require(guard.get('completed') is True and guard.get('exit_code') == 0 and guard.get('error') is None
                and guard.get('whole_group_retired') is True, 'Native guard did not cleanly complete')
        command = guard.get('command', [])
        quick_path = evidence['quick_producer']['path']
        require(command.count(quick_path) == 1, 'Guard does not name exact native producer')
        tail = command[command.index(quick_path):]
        require(tail == [quick_path, evidence['native_source']['path'], ','.join(map(str, ranks)), case['native_quick']['path']],
                'Native guard argv does not bind tested source/ranks/output exactly')
        native_path = Path(case['native_quick']['path'])
        expected_prediction = str(native_path.with_name(native_path.stem + '_pred_%d.npy' % case['rank']))
        require(case['native_prediction']['path'] == expected_prediction, 'Native prediction is not quick output')
        native, raw_sha = native_f32(case['native_prediction'])
        require(raw_sha[:16] == row.get('pred_sha'), 'Native raw-f32 quick prediction SHA mismatch')
        report = read_pin(case['parity_report'])
        require(report.get('rank') == case['rank'] and report.get('native') == case['native_prediction']['path']
                and len(report.get('layers', [])) == 16, 'Parity report lacks explicit tested native path/rank/layers')
        rms = []
        for layer, saved in enumerate(report['layers']):
            values = []
            for prediction in (original['prediction'], result['prediction']):
                value = math.sqrt(math.fsum((float(x)-float(y))**2 for x, y in zip(prediction[layer], native[layer])) / 1024)
                require(math.isfinite(value), 'Nonfinite native/Torch RMS')
                values.append(value)
            require(saved.get('layer') == layer and all(math.isclose(saved.get(key, float('nan')), value, rel_tol=1e-12, abs_tol=1e-18)
                    for key, value in zip(('original_rms', 'simulated_rms'), values)), 'Saved parity metrics disagree with pinned predictions')
            rms.append(values[1])
        require(rms[-1] <= 7e-6, 'Native-vs-Torch final RMS exceeds established7e-6 gate')
        output.append(dict(rank=case['rank'], rms_layers=rms, raw_native_f32_sha256=raw_sha))
    return dict(cases=output, final_rms_limit=7e-6, all16_finite=True, counted_native_build=checked,
                residual_power_and_cosine_used_for_acceptance=False)



def probe_context(qualification, config):
    probe = qualification.get('nonzero_probe')
    require(isinstance(probe, dict) and set(probe) == {'checkpoint', 'request_template', 'payload_sha256', 'prescription', 'native_evidence'},
            'Retrievable predeclared fixed-nonzero native/Torch probe is mandatory before fitting')
    prescription = read_pin(probe['prescription'])
    require(set(prescription) == {'fixed_before_measurement', 'uses_network_data', 'base_checkpoint', 'offsets'}
            and prescription['fixed_before_measurement'] is True and prescription['uses_network_data'] is False
            and prescription['base_checkpoint'] == qualification['latest_best_checkpoint'], 'Probe is not a fixed latest-best source-only prescription')
    offsets = prescription['offsets']
    require(set(offsets) == {'bias', 'output'}, 'Probe must exercise both offset families')
    require(all(len(offsets[f]) == 2 and all(len(row) == 16 and all(type(x) in (int, float) and math.isfinite(x) for x in row)
                and any(x != 0 for x in row) for row in offsets[f]) for f in offsets), 'Probe must exercise every early/middle family row')
    for key in ('checkpoint', 'request_template'):
        read_pin(probe[key], parse=False)
    request = read_pin(probe['request_template']); baseline = read_pin(config['request_template'])
    require(request['params'] == probe['checkpoint']['path'] and request['edge_band']['checkpoint_sha256'] == probe['checkpoint']['sha256']
            and behavior(request) == behavior(baseline), 'Fixed-nonzero probe changes unrelated request behavior')
    q = dict(qualification, checkpoint=probe['checkpoint'], request_template=probe['request_template'],
             payload_sha256=probe['payload_sha256'], native_evidence=probe['native_evidence'])
    first = probe['native_evidence']['cases'][0]
    q['zero_control'] = dict(plain_request=first['original_torch_request'], plain_result=first['original_torch_result'],
                            enabled_request=first['torch_request'], enabled_result=first['torch_result'])
    c = dict(config, checkpoint=probe['checkpoint'], request_template=probe['request_template'])
    return probe, q, c


def verify(qualification, config, expected_worker, engineer_directory):
    zero = verify_zero_control(qualification, config, expected_worker)
    native = verify_native(qualification, config, expected_worker, engineer_directory)
    probe, q, c = probe_context(qualification, config)
    nonzero_native = verify_native(q, c, expected_worker, engineer_directory)
    for case in probe['native_evidence']['cases']:
        require(prediction_bytes(read_pin(case['torch_result'])) != prediction_bytes(read_pin(case['original_torch_result'])),
                'Fixed-nonzero probe is numerically an identity on the tested network')
    return dict(zero_control=zero, zero_native=native, fixed_nonzero_native=nonzero_native,
                nonzero_probe_prescription=probe['prescription'], qualification_is_not_a_gain_claim=True)
