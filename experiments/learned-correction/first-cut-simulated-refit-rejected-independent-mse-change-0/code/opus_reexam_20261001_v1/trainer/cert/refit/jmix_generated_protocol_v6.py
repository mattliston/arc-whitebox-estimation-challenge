"""Exact96 protocolv5 plus explicit pinned truth-plan readiness and provenance.

The prescribed96 selection remains bound to unchanged protocolv5. This wrapper
does not read truth contents or change training/selection/acceptance rules.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
spec = importlib.util.spec_from_file_location('jmix_prescribed96_protocol_v5', HERE / 'jmix_generated_protocol_v5.py')
BASE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BASE)
from otr_generated_manifest_v1 import TruthPlan, require


def audit(args):
    result = BASE.audit(args)
    allowed = BASE._V1.read_ranks(os.environ['TRAINER_GENERATED_ALLOW_FILE'])
    plan = TruthPlan(os.environ['TRAINER_TRUTH_PLAN'], os.environ['TRAINER_TRUTH_PLAN_SHA256'], allowed)
    require(result['expected_request'].get('truth_plan') == plan.spec(), 'Request truth plan is not pinned to worker plan')
    require(not set(result['sets']['generated_judge32']) & plan.allowed, 'Plan allowlist exposes sealed judge seeds')
    readiness = {}
    for role, ranks in (('sizing', result['sets']['sizing']), ('check', result['sets']['check'])):
        ready, missing = [], []
        for seed in sorted(ranks):
            (ready if plan.ready(seed) else missing).append(seed)
        readiness[role] = dict(ready=ready, missing=missing, required_at_launch=role == 'sizing')
    require(not readiness['sizing']['missing'], 'Planned sizing truths not ready')
    result['protocol'] = 'jmix_generated_protocol_v6'
    result['files'].update(protocol_v6=BASE.fingerprint(__file__), truth_plan=BASE.fingerprint(plan.path),
                           truth_adapter=BASE.fingerprint(HERE.parents[1] / 'otr_generated_manifest_v1.py'))
    result['fresh_launch'].update(truth_readiness=readiness,
        truth_readiness_basis='explicit immutable plan paths plus pinned completion receipt when required; no truth contents read',
        sizing_check_truth_files_exist=not readiness['check']['missing'])
    result['truth_plan'] = plan.spec()
    result['truth_entries'] = [plan.entries[s] for s in sorted(set(result['sets']['sizing']) | set(result['sets']['check']))]
    result['verdict'] = 'PASS: unchanged prescribed96 protocol plus explicit pinned truth plan; %d check truths pending' % len(readiness['check']['missing'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sizing', 'check', 'judge', 'gradient-ranks', 'gradient-selection', 'out'):
        parser.add_argument('--' + name, required=True)
    for name in ('first-batch', 'steps'):
        parser.add_argument('--' + name, required=True, type=int)
    parser.add_argument('--gradient-label', default='')
    parser.add_argument('--params')
    parser.add_argument('--head')
    args = parser.parse_args()
    require(not Path(args.out).exists(), 'Protocol output already exists')
    result = audit(args)
    with Path(args.out).open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(result['verdict'])
