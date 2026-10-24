"""Finite-step sizing for EDBAND six arms; writes one unaccepted point only.

Consumes exactly16 prescribed sizing predictions. The independent acceptance
and untouched judge ranks are checked for overlap but their labels are never
opened here. There is no enqueue, worker launch, export, or submission operation.
CLI: python -B edband_six_fit_v3.py CONFIG.json NEW_DIRECTORY
"""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import otr_edband_v1 as EB
import edband_six_arms_v3 as A


def configuration(request):
    return {key: value for key, value in request.items()
            if key not in {'id', 'rank', 'mode', 'scale', 'keep_all', 'keep_final', 'params', 'edge_band'}}


def load_eval(item, expected_pin, template, rank, payload_fingerprint, base_fingerprint):
    A.require(set(item) == {'request', 'result'}, 'Expected pinned evaluation request and result')
    request = json.loads(A.read_pin(item['request']))
    record = json.loads(A.read_pin(item['result']))
    A.require(request['rank'] == rank and request['mode'] == 'eval'
              and request.get('keep_final') == 1, 'Exact sizing evaluation request required')
    A.require(request['params'] == expected_pin['path'] and request['edge_band'] ==
              EB.request_spec(expected_pin['path'], expected_pin['sha256'], A.FAMILIES),
              'Evaluation checkpoint spec mismatch')
    A.require(configuration(request) == configuration(template), 'Sizing behavior differs from base')
    for key in ('id', 'rank', 'mode', 'params', 'cutsim'):
        A.require(record.get(key) == request.get(key), 'Evaluation request/result mismatch: ' + key)
    identity = record.get('edge_band', {})
    A.require(identity.get('enabled') is True and identity.get('request') == request
              and identity.get('request_sha256') == A.canonical(request)
              and identity.get('spec') == request['edge_band']
              and identity.get('payload_sha256') == payload_fingerprint
              and identity.get('base_sha256') == base_fingerprint,
              'Evaluation EDBAND identity mismatch')
    worker = record.get('edge_band_worker', {})
    A.require(worker == A.expected_worker(), 'Sizing worker/source/capture identity mismatch')
    prediction = record.get('final_prediction', [])
    A.require(len(prediction) == 1024 and all(math.isfinite(x) for x in prediction), 'Invalid final prediction')
    A.require(type(record.get('mse')) in (int, float) and math.isfinite(record['mse']) and record['mse'] > 0,
              'Invalid sizing MSE')
    return record


def check_roles(plan, sizing, acceptance, judge):
    # Metadata only: do not widen plan.allowed or resolve held target files.
    for role, ranks in [('sizing', sizing), ('acceptance', acceptance), ('final', judge)]:
        A.require(all(rank in plan.entries and plan.entries[rank]['role'] == role for rank in ranks),
                  'Assigned rank does not have expected truth-plan role: ' + role)


def main():
    A.require(len(sys.argv) == 3, 'usage: edband_six_fit_v3.py CONFIG.json NEW_DIRECTORY')
    config_path, output = (Path(value).resolve() for value in sys.argv[1:])
    config = json.loads(config_path.read_text())
    fields = {'schema', 'arms', 'sizing_ranks', 'acceptance_ranks', 'judge_ranks',
              'truth_plan', 'evaluations', 'ridge_fraction', 'cache_directory', 'tag'}
    A.require(set(config) == fields and config['schema'] == 'trainer_edge_six_fit_config_v1', 'Fit config schema')
    A.require(config['ridge_fraction'] == 0.02, 'This version fixes the ridge prescription at0.02')
    sizing = A.rank_pin(config['sizing_ranks'])
    acceptance = A.rank_pin(config['acceptance_ranks'])
    judge = A.rank_pin(config['judge_ranks'])
    A.require(len(sizing) == len(acceptance) == 16 and len(judge) == 32, 'Expected16 sizing/16 acceptance/32 judge')
    A.require(not set(sizing) & (set(acceptance) | set(judge)) and not set(acceptance) & set(judge),
              'Sizing/acceptance/judge overlap')
    A.require(all(rank >= 7_000_000 for rank in sizing + acceptance + judge), 'Fresh generated sets required')
    arms = json.loads(A.read_pin(config['arms']))
    A.require(arms.get('schema') == 'trainer_edge_six_arms_v1' and len(arms['proposals']) <= 6
              and arms['proposals'], 'No active EDBAND arms')
    A.require(not set(arms['gradient_ranks']) & set(sizing + acceptance + judge), 'Gradient/holdout overlap')
    A.require(set(config['evaluations']) == {'base', *arms['proposals']}, 'Exact live-arm sizing groups required')
    A.require(all(len(values) == len(sizing) for values in config['evaluations'].values()), 'Exact sizing count required')
    for file_pin in arms['source'].values():
        if isinstance(file_pin, dict) and set(file_pin) == {'path', 'sha256'}:
            A.read_pin(file_pin)
    # These imports happen only on explicit execution; reading/compiling this
    # source cannot consume any generated labels or initialize Torch.
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from otr_common import data_paths as public_data_paths
    from otr_generated_manifest_v1 import TruthPlan, make_data_paths
    from otr_truth_records_v1 import validate_record
    truth_spec = config['truth_plan']
    A.require(set(truth_spec) == {'path', 'sha256'}, 'Explicit pinned truth plan required')
    plan = TruthPlan(truth_spec['path'], truth_spec['sha256'], sizing)
    check_roles(plan, sizing, acceptance, judge)
    data_paths, truth_metadata = make_data_paths(public_data_paths, plan, Path(config['cache_directory']))
    parent_pin = arms['base_checkpoint']
    parent = EB.load_checkpoint(torch, parent_pin['path'], parent_pin['sha256'])
    template = json.loads(A.read_pin(arms['request_template']))
    A.require(template.get('truth_plan') == plan.spec(), 'Sizing truth plan differs from request')
    head = json.loads(A.read_pin(arms['head']))
    A.require(template.get('hu') == head, 'Pinned head mismatch')
    names = list(arms['proposals'])
    checkpoints = {'base': parent_pin, **{name: value['checkpoint'] for name, value in arms['proposals'].items()}}
    identities = {}
    for name, item in checkpoints.items():
        trial = EB.load_checkpoint(torch, item['path'], item['sha256'])
        identities[name] = (EB._fingerprint(torch, trial), EB._fingerprint(torch, trial['base']))
    del trial
    D, residual, errors, ids = [], [], {name: [] for name in checkpoints}, set()
    for i, rank in enumerate(sizing):
        records = {}
        for name, checkpoint in checkpoints.items():
            record = load_eval(config['evaluations'][name][i], checkpoint, template, rank, *identities[name])
            A.require(record['id'] not in ids, 'Duplicate sizing result ID')
            ids.add(record['id'])
            validate_record(plan, record, rank)
            records[name] = record
        # TruthPlan denies every acceptance/final rank before path resolution.
        with np.load(data_paths(rank)[1], allow_pickle=False) as archive:
            truth = np.asarray(archive['mc_post'], dtype=np.float64)[15]
        baseline = np.asarray(records['base']['final_prediction'], dtype=np.float64)
        residual.append(truth - baseline)
        D.append(np.stack([np.asarray(records[name]['final_prediction'], dtype=np.float64) - baseline
                           for name in names], axis=1))
        for name, record in records.items():
            prediction = np.asarray(record['final_prediction'], dtype=np.float64)
            error = float(np.mean((truth - prediction) ** 2))
            A.require(abs(error / record['mse'] - 1) < 2e-5, 'Saved prediction/MSE disagreement')
            errors[name].append(error)
    energy = sum(float(value @ value) for value in residual)
    A.require(math.isfinite(energy) and energy > 0, 'Invalid baseline residual energy')
    nonzero = [i for i in range(len(names)) if sum(float(np.dot(d[:, i], d[:, i])) for d in D) > 0]
    A.require(nonzero, 'All finite trials are exact identity; no point can be fitted')
    selected = [matrix[:, nonzero] for matrix in D]

    def fit(indices):
        gram = sum(selected[i].T @ selected[i] for i in indices)
        rhs = sum(selected[i].T @ residual[i] for i in indices)
        return np.linalg.solve(gram + 0.02 * np.diag(np.diag(gram)), rhs)

    selected_weights = fit(list(range(len(sizing))))
    A.require(bool(np.isfinite(selected_weights).all()), 'Nonfinite finite-step solve')
    weights = {name: 0.0 for name in names}
    for index, value in zip(nonzero, selected_weights):
        weights[names[index]] = float(value)
    model = sum(float(np.dot(r - d @ selected_weights, r - d @ selected_weights))
                for r, d in zip(residual, selected)) / energy - 1
    loo = sum(float(np.dot(residual[i] - selected[i] @ fit([j for j in range(len(sizing)) if j != i]),
                           residual[i] - selected[i] @ fit([j for j in range(len(sizing)) if j != i])))
              for i in range(len(sizing))) / energy - 1
    point = EB._clone(torch, parent)
    for name, weight in weights.items():
        family, coordinate = name.split('_')
        A.require(family in A.FAMILIES and coordinate in A.COORDINATES, 'Unexpected arm coordinate')
        arm = arms['proposals'][name]
        direction = torch.tensor(arm['direction'], dtype=torch.float64)
        A.require(tuple(direction.shape) == (16,) and bool(torch.isfinite(direction).all()), 'Invalid arm direction')
        values = (point['base']['arrays']['e.' + family] if coordinate == 'shared' else
                  point['edge_band']['offsets'][family][A.COORDINATES.index(coordinate) - 1])
        values.copy_((values.double() + weight * arm['multiplier'] * direction).to(torch.float32))
    # Every unrelated array, dial, PL value and base field survives verbatim.
    before, after = EB._clone(torch, parent['base']), EB._clone(torch, point['base'])
    for family in A.FAMILIES:
        before['arrays'].pop('e.' + family); after['arrays'].pop('e.' + family)
    A.require(EB._fingerprint(torch, before) == EB._fingerprint(torch, after), 'Unrelated base field changed')
    point['source'] = dict(schema='trainer_edge_six_fit_source_v1', config=A.pin(config_path),
        arms=config['arms'], script=A.pin(__file__), parent_checkpoint=parent_pin,
        weights=weights, ridge_fraction=0.02)
    output.mkdir(exist_ok=False)
    saved = EB.save_checkpoint(torch, output/'point.pt', point)
    request = dict(template, id=config['tag'] + '_point_template', mode='eval', params=saved['path'],
                   edge_band=EB.request_spec(saved['path'], saved['sha256'], A.FAMILIES))
    request.pop('scale', None); request.pop('keep_all', None); request.pop('keep_final', None)
    A.write_json(output/'point_request.json', request)
    A.write_json(output/'head_42x128.json', head)
    report = dict(schema='trainer_edge_six_fit_point_v1', utc=subprocess.check_output(['date','-u','+%FT%TZ'],text=True).strip(),
        source=point['source'], point=saved, request=A.pin(output/'point_request.json'), head=A.pin(output/'head_42x128.json'),
        sizing_ranks=sizing, acceptance_ranks=acceptance, sealed_judge_ranks=judge,
        weights=weights, model_change_pct=100*model, leave_one_out_change_pct=100*loo,
        arm_change_pct={name: 100*(sum(value)/sum(errors['base'])-1) for name, value in errors.items() if name != 'base'},
        accepted=False, independent_acceptance_required=True, final_judge_read=False, queued=False)
    A.write_json(output/'point.json', report)
    print(json.dumps(dict(manifest=str(output/'point.json'), accepted=False, queued=False)))


if __name__ == '__main__':
    main()
