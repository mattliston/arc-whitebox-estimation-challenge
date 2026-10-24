"""Prepare six identifiable EDGE directions; source-only, no enqueue/worker code.

This deliberately does not accept legacy joint-refit directions. Each direction
moves one family (bias/output) and one coordinate (shared/early/middle). The late
band is the shared reference, so there is no third offset or redundant arm.
Only explicit, SHA-bound gradient requests can supply a direction. A separate
finite-step sizing/check round must qualify any proposed point.

CLI: python -B edband_six_arms_v3.py CONFIG.json NEW_DIRECTORY
CONFIG schema and required fields are defined by validate_config below. Preparing
a config is not launch authorization. All neural work stays outside this file.
"""
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys

import otr_edband_v1 as EB

SCHEMA = 'trainer_edge_six_arm_config_v1'
COORDINATES = ('shared', 'early', 'middle')
FAMILIES = ('bias', 'output')
WORKER_SHA = 'f1dae5b23057a08871e9691fbb42e0cbe7b0565e1d970c6eea273ed66618366f'
WORKER_PINS = {
    'worker_generated_v4.py': '308cc7b08d2291e8e817ac722ae3e7184dc9e95bdc3f13990659a5b958e244a1',
    'otr_engine.py': 'ea8a3bc6a8d178427470ef74389f2a5ecf94ee1a717e39710a87fcd184a7ec17',
    'otr_edband_v1.py': '24a7debb6f8994982fb4052d01b9f28ca07729a8a179a1628c8bac9b6bb4ffc6',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=EB.sha256(path))


def read_pin(item):
    require(set(item) == {'path', 'sha256'}, 'Expected exact file pin')
    path = Path(item['path'])
    require(path.is_absolute(), 'Absolute pinned path required')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == item['sha256'], 'Pinned file changed: ' + str(path))
    return raw


def write_json(path, value):
    EB._publish(path, lambda stream: json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False))


def rank_pin(item):
    values = [int(x) for x in read_pin(item).decode().strip().split(',')]
    require(values and len(values) == len(set(values)), 'Empty/duplicate rank file')
    return values


def validate_config(config):
    required = {'schema', 'tag', 'checkpoint', 'head', 'gradient_ranks', 'held_rank_files',
                'public_training_ranks', 'gradient_requests', 'request_template', 'worker',
                'qualification_receipt', 't_threshold', 'target_change_fraction', 'max_rms_multiple'}
    require(set(config) == required and config['schema'] == SCHEMA, 'Config schema/fields')
    require(isinstance(config['tag'], str) and config['tag'].replace('_', '').isalnum(), 'Safe tag required')
    ranks = rank_pin(config['gradient_ranks'])
    require(len(ranks) in (24, 36), 'This screen requires a prescribed24 or36 ranks')
    require(set(ranks) <= set(rank_pin(config['public_training_ranks'])), 'Gradient outside public training')
    require(all(0 <= rank < 7_000_000 for rank in ranks), 'No generated training ranks')
    require(config['held_rank_files'], 'Pinned held/diagnostic rank inventory required')
    for item in config['held_rank_files']:
        require(not set(ranks) & set(rank_pin(item)), 'Gradient overlaps held/diagnostic rank file')
    require(len(config['gradient_requests']) == len(ranks), 'Exact gradient request count required')
    require(config['worker']['sha256'] == WORKER_SHA, 'Unsupported EDBAND worker')
    read_pin(config['worker'])
    # Full-card zero-offset and native parity qualification must be supplied by
    # the lead after actual execution. A source fixture cannot satisfy this gate.
    qualification = json.loads(read_pin(config['qualification_receipt']))
    require(qualification.get('schema') == 'trainer_edge_band_fullcard_qualification_v1'
            and qualification.get('zero_offset_all16_exact') is True
            and qualification.get('native_parity_pass') is True
            and qualification.get('worker_sha256') == WORKER_SHA,
            'Full-card/native EDBAND qualification is required before arming')
    for name in ('t_threshold', 'target_change_fraction', 'max_rms_multiple'):
        require(type(config[name]) in (int, float) and math.isfinite(config[name]) and config[name] > 0,
                'Positive finite prescription required: ' + name)
    require(config['t_threshold'] >= 2.5, 'Cannot silently relax statistical direction filter')
    require(config['target_change_fraction'] <= 0.01 and config['max_rms_multiple'] <= 0.25,
            'Finite trial prescription exceeds this bounded screen')
    return ranks, qualification


def behavior(request):
    return {k: v for k, v in request.items()
            if k not in {'id', 'rank', 'mode', 'scale', 'keep_all', 'keep_final'}}


def expected_worker():
    import ast
    import worker_edband_v2 as worker
    worker.verify_sources()
    require(EB.sha256(worker.__file__) == WORKER_SHA and worker.PINS == WORKER_PINS, 'Worker source changed')
    source = (Path(__file__).resolve().parent / 'otr_engine.py').read_text()
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'run_network']
    require(len(nodes) == 1, 'Engine run_network is not unique')
    node = nodes[0]
    function = ''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    return dict(schema='trainer_edge_band_worker_v1', worker_sha256=WORKER_SHA, sources=WORKER_PINS,
                patched_run_sha256=hashlib.sha256(worker.engine_source(function).encode()).hexdigest())


def verify_result(torch, config, payload, expected_request, record):
    for key in ('id', 'rank', 'mode', 'params'):
        require(record.get(key) == expected_request[key], 'Result/request mismatch: ' + key)
    require(record.get('mode') == 'grad' and record.get('scale') == expected_request['scale'], 'Gradient scale/mode')
    require(record.get('truth_metadata') is None, 'Public gradient has generated truth metadata')
    require(record.get('cutsim') == expected_request['cutsim'], 'Simulator identity changed')
    identity = record.get('edge_band', {})
    require(identity.get('enabled') is True and identity.get('request') == expected_request
            and identity.get('request_sha256') == canonical(expected_request), 'Exact EDBAND request identity missing')
    require(identity.get('payload_sha256') == EB._fingerprint(torch, payload)
            and identity.get('base_sha256') == EB._fingerprint(torch, payload['base'])
            and identity.get('spec') == expected_request['edge_band'], 'EDBAND checkpoint identity mismatch')
    worker = record.get('edge_band_worker', {})
    require(worker == expected_worker(), 'Unexpected EDBAND worker/source/capture identity')
    gradients = record.get('edge_band_grads', {})
    require(gradients.get('schema') == 'trainer_edge_band_gradients_v1'
            and gradients.get('identity') == identity and gradients.get('scale') == record['scale']
            and set(gradients.get('values', {})) == set(FAMILIES), 'Missing offset gradients/provenance')
    require(math.isfinite(record['mse']) and record['mse'] > 0, 'Invalid gradient MSE')
    for family in FAMILIES:
        shared, offsets = record['grads']['e.' + family], gradients['values'][family]
        require(tuple(shared.shape) == (16,) and tuple(offsets.shape) == (2, 16)
                and bool(torch.isfinite(shared).all()) and bool(torch.isfinite(offsets).all()),
                'Missing/nonfinite shared or offset gradient: ' + family)


def direction(torch, gradients, parameter_rms, threshold):
    """Return a sign direction in shared-RMS units and shrunk first-order slope."""
    stack = torch.stack(gradients).double()
    mean = stack.mean(0)
    se = stack.std(0, unbiased=True) / math.sqrt(len(gradients))
    t = torch.where(se > 0, mean / se.clamp_min(1e-300), torch.zeros_like(mean))
    selected = t.abs() > threshold
    value = -mean.sign() * selected * parameter_rms
    slope = float((mean.abs() * (1 - 1 / t.square().clamp_min(1e-12)) * selected).sum() * parameter_rms)
    return value, dict(selected=int(selected.sum()), size=16, mean=mean.tolist(), se=se.tolist(),
                       t=t.tolist(), shrunk_decrease_per_multiple=slope, parameter_rms=parameter_rms)


def main():
    require(len(sys.argv) == 3, 'usage: edband_six_arms_v3.py CONFIG.json NEW_DIRECTORY')
    config_path, output = (Path(value).resolve() for value in sys.argv[1:])
    config = json.loads(config_path.read_text())
    ranks, qualification = validate_config(config)
    # Imports are deferred: source inspections do not load Torch or any network.
    import torch
    torch.set_num_threads(1)
    payload = EB.load_checkpoint(torch, **{'path': config['checkpoint']['path'],
        'expected_sha256': config['checkpoint']['sha256']})
    head = json.loads(read_pin(config['head']))
    require(len(head) == 42 and all(len(row) == 128 for row in head), 'Exact head shape required')
    require(all(math.isfinite(x) for row in head for x in row), 'Nonfinite head')
    template = json.loads(read_pin(config['request_template']))
    require(template.get('hu') == head and template.get('params') == config['checkpoint']['path'],
            'Template checkpoint/head mismatch')
    require(template.get('edge_band') == EB.request_spec(config['checkpoint']['path'],
        config['checkpoint']['sha256'], FAMILIES), 'Template EDBAND spec mismatch')
    require(qualification.get('base_content_sha256') == EB._fingerprint(torch, payload['base'])
            and qualification.get('cutsim') == template.get('cutsim'), 'Qualification does not cover this starting point')
    source_directory = Path(config['worker']['path']).parent
    for filename, digest in WORKER_PINS.items():
        require(EB.sha256(source_directory / filename) == digest, 'Pinned live source changed: ' + filename)
    rows, actual_ranks, seen_ids = [], [], set()
    for item in config['gradient_requests']:
        require(set(item) == {'request', 'result'}, 'Gradient entry fields')
        request = json.loads(read_pin(item['request']))
        require(request['mode'] == 'grad' and type(request.get('scale')) in (int, float)
                and math.isfinite(request['scale']) and request['scale'] > 0, 'Expected positive-scale gradient request')
        require(behavior(request) == behavior(template), 'Gradient behavior differs from frozen template')
        require(request['id'] not in seen_ids, 'Duplicate gradient ID')
        seen_ids.add(request['id']); actual_ranks.append(request['rank'])
        record = torch.load(io.BytesIO(read_pin(item['result'])), map_location='cpu', weights_only=False)
        verify_result(torch, config, payload, request, record)
        # Retain just96 scalars per network; discard the rest of each result.
        rows.append(dict(mse=record['mse'], values={
            family: dict(shared=record['grads']['e.' + family].detach().double() / record['scale'],
                         early=record['edge_band_grads']['values'][family][0].detach().double() / record['scale'],
                         middle=record['edge_band_grads']['values'][family][1].detach().double() / record['scale'])
            for family in FAMILIES}))
        del record
    require(actual_ranks == ranks, 'Exact prescribed gradient rank order required')
    mean_mse = sum(row['mse'] for row in rows) / len(rows)
    output.mkdir(exist_ok=False)
    proposals, report = {}, {}
    source = dict(schema='trainer_edge_six_arm_preparation_v1', config=pin(config_path),
        script=pin(__file__), module=pin(EB.__file__), parent_checkpoint=config['checkpoint'],
        qualification_receipt=config['qualification_receipt'])
    for family in FAMILIES:
        parameter_rms = float(payload['base']['arrays']['e.' + family].double().square().mean().sqrt())
        require(parameter_rms > 0, 'Shared parameter RMS is zero; explicit scale protocol required')
        for coordinate in COORDINATES:
            name = family + '_' + coordinate
            delta, summary = direction(torch, [row['values'][family][coordinate] for row in rows],
                                       parameter_rms, config['t_threshold'])
            slope = summary['shrunk_decrease_per_multiple']
            multiplier = min(config['target_change_fraction'] * mean_mse / slope,
                             config['max_rms_multiple']) if slope > 0 else 0.0
            summary.update(multiplier=multiplier, target_change_fraction=config['target_change_fraction'])
            report[name] = summary
            if not summary['selected'] or multiplier == 0:
                summary['inactive_reason'] = 'No selected nonzero direction'
                continue
            trial = EB._clone(torch, payload)
            if coordinate == 'shared':
                values = trial['base']['arrays']['e.' + family]
            else:
                values = trial['edge_band']['offsets'][family][COORDINATES.index(coordinate) - 1]
            moved = (values.double() + multiplier * delta).to(torch.float32)
            summary['realized_nonzero'] = int((moved != values).sum())
            if not summary['realized_nonzero']:
                summary['inactive_reason'] = 'Entire finite trial rounded away'
                continue
            values.copy_(moved)
            trial['source'] = dict(source, arm=name, prescription=summary)
            saved = EB.save_checkpoint(torch, output / ('arm_' + name + '.pt'), trial)
            request = dict(template, id=config['tag'] + '_' + name + '_template', params=saved['path'],
                           mode='eval', keep_final=1,
                           edge_band=EB.request_spec(saved['path'], saved['sha256'], FAMILIES))
            request.pop('scale', None); request.pop('keep_all', None)
            write_json(output / ('arm_' + name + '_request.json'), request)
            proposals[name] = dict(checkpoint=saved, request=pin(output / ('arm_' + name + '_request.json')),
                                   direction=delta.tolist(), multiplier=multiplier)
    manifest = dict(schema='trainer_edge_six_arms_v1', utc=subprocess.check_output(['date','-u','+%FT%TZ'],text=True).strip(),
        source=source, gradient_ranks=ranks, count=len(ranks), mean_mse=mean_mse,
        design=EB.DESIGN, head=config['head'], base_checkpoint=config['checkpoint'],
        request_template=config['request_template'], proposals=proposals, report=report,
        queued=False, candidate_accepted=False, sizing_and_independent_acceptance_required=True)
    write_json(output / 'arms.json', manifest)
    print(json.dumps(dict(manifest=str(output/'arms.json'), live=list(proposals), queued=False)))


if __name__ == '__main__':
    main()
