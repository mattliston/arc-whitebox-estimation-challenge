#!/bin/bash
# Astra v24: generated sizing/checking, checked cut-simulator requests, and exact
# request/result waits. Exactly one prescribed96-gradient point, then report/stop.
# All earlier versions are preserved; this is NOT a final judging/export driver.
# Run from O/trainer after simulator certification and all 16 sizing truths exist.
# Check truths may be pending; exact check results are required at acceptance:
# JM_CHECK=cert/refit/gen_check16_v4.txt \
# TRAINER_GENERATED_ALLOW_FILE=cert/refit/gen_worker_allow_v5.txt \
# JM_REQ='<certified request JSON>' bash cert/refit/jmix24.sh \
#   <fresh tag> <params.pt> <hu.json> <first step> <last step> <first batch> \
#   '<extra ncar JSON, normally []>' cert/refit/gen_size16_v4.txt
# Required: FIRST=LAST=1, FIRST_BATCH=0; JM_G1 and JM_PREV must be empty.
# JM_GRAD_RANKS defaults to cert/refit/astra_cut_first96_v1.txt.
# JM_GRAD_SELECTION defaults to cert/refit/astra_cut_first96_v1.selection.json.
# Optional: JM_HSPLIT, JM_DIALS, JM_ARR_SPLIT,
# JM_JUDGE (defaults to cert/refit/gen_judge32_v4.txt), JM_EXPORT_TOOL
# (defaults to export_dials_s46_v2.py). JM_DROP must be empty: freeze one variant
# before the round. No resume/overwrite mode.
# Repeated check16 is checkpoint-selection data. Public judge48 is secondary;
# untouched generated judging and native qualification remain downstream gates.
set -euo pipefail
PY=/Users/matt/anaconda3/bin/python
exec "$PY" -B - "$@" <<'PYEOF'
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import time
import traceback

sys.path.insert(0, '.')
from otr_common import enqueue, splits

PY = sys.executable
RUN = Path('run_M').resolve()
RESULTS = RUN / 'results'
REFIT = Path('cert/refit')
TEAM_LOG = Path('../TEAM_LOG.md')
REQUESTS = {}
TAG = None
TRUTH_PLAN = None
APPROVED_HU_ROWS = {6, 17, 20, 21, 28, 29}


def require(condition, detail):
    if not condition:
        raise RuntimeError(detail)


def utc():
    return subprocess.check_output(['date', '-u', '+%FT%TZ'], text=True).strip()


def log(message):
    stamp = utc()
    with TEAM_LOG.open('a') as stream:
        stream.write('[trainer] %s %s\n' % (stamp, message))
    print('%s %s' % (stamp, message), flush=True)


def ranks_file(path):
    values = [int(value) for value in Path(path).read_text().strip().split(',')]
    require(values and len(values) == len(set(values)), 'Empty or duplicate rank set: %s' % path)
    return values


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def require_file(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size > 0, 'Missing/empty artifact: %s' % path)
    return path


def fingerprint(path):
    path = require_file(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def head_values(path):
    values = json.loads(require_file(path).read_text())
    require(isinstance(values, list) and 1 <= len(values) <= 42, 'Invalid head row count: %s' % path)
    require(all(isinstance(row, list) and len(row) == 128
                and all(isinstance(x, (int, float)) and math.isfinite(x) for x in row)
                for row in values), 'Invalid/nonfinite head weights: %s' % path)
    require(all(all(x == 0 for x in row) for i, row in enumerate(values) if i not in APPROVED_HU_ROWS),
            'First-point HU contains unqualified rows outside 6/17/20/21/28/29: %s' % path)
    return values


def check_failures():
    require(not (RUN / 'STOP').exists(), 'run_M/STOP exists; driver exits preserving all work')
    prefixes = {request['id'].rsplit('_r', 1)[0] for request in REQUESTS.values()}
    for prefix in sorted(prefixes):
        errors = sorted(RESULTS.glob(prefix + '_r*.error'))
        require(not errors, 'Worker failed; inspect %s' % (errors[0] if errors else prefix))


def quiet_window_open():
    states = re.findall(r'^\[lead\] \S+ QUIET WINDOW (OPEN|CLOSED)\b',
                        TEAM_LOG.read_text(), re.MULTILINE)
    return bool(states and states[-1] == 'OPEN')


def await_window():
    announced = False
    while quiet_window_open():
        check_failures()
        if not announced:
            print('%s holding for release quiet window' % utc(), flush=True)
            announced = True
        time.sleep(20)
    check_failures()


def command(arguments, output, heavy=True):
    if heavy:
        await_window()
    else:
        check_failures()
    with Path(output).open('x') as stream:
        completed = subprocess.run([PY, '-B', *map(str, arguments)], stdout=stream, stderr=subprocess.STDOUT)
    require(completed.returncode == 0,
            'Command failed (exit %d); inspect %s: %s' % (completed.returncode, output, arguments[0]))


def eval_record(request):
    path = RESULTS / (request['id'] + '.json')
    record = json.loads(require_file(path).read_text())
    from otr_truth_records_v1 import validate_record
    validate_record(TRUTH_PLAN, record, request['rank'])
    require(record.get('id') == request['id'] and record.get('rank') == request['rank']
            and record.get('mode') == 'eval', 'Result identity mismatch: %s' % path)
    require(Path(record.get('params', '')).resolve() == Path(request['params']).resolve(),
            'Result parameter mismatch: %s' % path)
    require(record.get('cutsim') == request.get('cutsim'), 'Result simulator mismatch: %s' % path)
    if request['rank'] >= 7_000_000:
        require(record.get('data_source') == 'generated', 'Generated result lacks qualified worker identity: %s' % path)
    require(isinstance(record.get('mse'), (int, float)) and math.isfinite(record['mse'])
            and record['mse'] > 0, 'Invalid MSE: %s' % path)
    layers = record.get('mse_layers', [])
    require(len(layers) == 16 and all(math.isfinite(x) and x >= 0 for x in layers),
            'Invalid layer MSEs: %s' % path)
    if request.get('keep_final'):
        prediction = record.get('final_prediction', [])
        require(len(prediction) == 1024 and all(math.isfinite(x) for x in prediction),
                'Missing/invalid final prediction: %s' % path)
    if request.get('keep_all'):
        prediction = record.get('prediction', [])
        require(len(prediction) == 16 and all(len(row) == 1024 and all(math.isfinite(x) for x in row)
                                             for row in prediction),
                'Missing/invalid all-layer parity prediction: %s' % path)
    return record


def wait_exact(requests):
    expected = {request['id']: request for request in requests}
    require(len(expected) == len(requests), 'Duplicate IDs in an expected result group')
    pending = dict(expected)
    last_report = 0.0
    while True:
        check_failures()
        for name, request in list(pending.items()):
            suffix = '.pt' if request['mode'] == 'grad' else '.json'
            path = RESULTS / (name + suffix)
            if path.is_file():
                require_file(path)
                if request['mode'] == 'eval':
                    eval_record(request)
                del pending[name]
        if not pending:
            return
        if time.monotonic() - last_report >= 300:
            print('%s waiting for %d/%d exact results; first %s' %
                  (utc(), len(pending), len(expected), next(iter(pending))), flush=True)
            last_report = time.monotonic()
        time.sleep(20)


def behavior_config(request):
    # Normalize only documented worker defaults. attach inherits the worker's
    # startup setting, so absence stays distinct from either explicit boolean.
    flags = ('unfreeze', 'rg', 'b6', 'hbands', 'b7', 'featstats', 'cv', 'keep_final', 'keep_all')
    value = {key: bool(request.get(key)) for key in flags}
    value.update(attach=request.get('attach', '<worker-default>'),
                 f4c=request.get('f4c', '<worker-default>'), ss=request.get('ss'),
                 hf=int(request.get('hf') or 0), ncar=request.get('ncar'),
                 pl=request.get('pl') or {}, hsplit=request.get('hsplit') or {},
                 rgw=request.get('rgw'), sched=request.get('sched') or '384-320-384',
                 lossw=request.get('lossw') or None, cutsim=request.get('cutsim'),
                 dump15=request.get('dump15'))
    return value


def request_group(prefix, params, head, mode, ranks, common):
    require(ranks and len(ranks) == len(set(ranks)), 'Empty/duplicate request rank set')
    require_file(params)
    base = dict(common, hu=head_values(head))
    options = {'grad': {'mode': 'grad', 'scale': 1e8}, 'eval': {'mode': 'eval'},
               'evalkeep': {'mode': 'eval', 'keep_final': 1},
               'evalall': {'mode': 'eval', 'keep_all': 1}}[mode]
    return [dict(id='a_%s_r%04d' % (prefix, rank), rank=rank,
                 params=str(Path(params).resolve()), **options, **base) for rank in ranks]


def queue_group(prefix, params, head, mode, ranks, common):
    await_window()
    requests = request_group(prefix, params, head, mode, ranks, common)
    for request in requests:
        require(request['id'] not in REQUESTS, 'Duplicate planned request: %s' % request['id'])
        REQUESTS[request['id']] = request
        check_failures()
        require(enqueue(RUN, request), 'Unexpected existing request/result; no reuse allowed: %s' % request['id'])
    print('%s %s: queued %d exact requests' % (utc(), prefix, len(requests)), flush=True)
    return requests


def pair_stats(ranks, first, second):
    a, b, layers_a, layers_b = [], [], [], []
    for rank in ranks:
        rows = []
        for label in (first, second):
            name = 'a_%s_r%04d' % (label, rank)
            path = RESULTS / (name + '.json')
            record = json.loads(require_file(path).read_text())
            require(record.get('rank') == rank and record.get('id') == name,
                    'Paired result identity mismatch: %s' % path)
            require(math.isfinite(record['mse']) and record['mse'] > 0
                    and len(record['mse_layers']) == 16
                    and all(math.isfinite(x) and x >= 0 for x in record['mse_layers']),
                    'Invalid paired MSE: %s' % path)
            rows.append(record)
        a.append(rows[0]['mse']); b.append(rows[1]['mse'])
        layers_a.append(statistics.mean(rows[0]['mse_layers']))
        layers_b.append(statistics.mean(rows[1]['mse_layers']))
    require(len(a) >= 2, 'At least two complete paired ranks required')
    changes = [100 * (x / y - 1) for x, y in zip(a, b)]
    stats = dict(first=first, second=second, ranks=list(ranks), count=len(a),
                 mean=statistics.mean(a), baseline_mean=statistics.mean(b),
                 change_pct=100 * (statistics.mean(a) / statistics.mean(b) - 1),
                 se_pct=statistics.stdev(changes) / math.sqrt(len(a)),
                 better=sum(x < y for x, y in zip(a, b)),
                 all_layers_pct=100 * (statistics.mean(layers_a) / statistics.mean(layers_b) - 1))
    stats['summary'] = ('%d ranks: %.4e against %.4e: %+.3f%% (se %.3f, better on %d/%d; all layers %+.3f%%)' %
                        (len(a), stats['mean'], stats['baseline_mean'], stats['change_pct'], stats['se_pct'],
                         stats['better'], len(a), stats['all_layers_pct']))
    return stats


def main():
    global TAG, TRUTH_PLAN
    require(len(sys.argv) == 9,
            'usage: bash cert/refit/jmix24.sh TAG PARAMS HEAD FIRST LAST FIRST_BATCH EXTRA_JSON SIZING_FILE')
    TAG, params_arg, head_arg, first_arg, last_arg, batch_arg, extra_arg, sizing_arg = sys.argv[1:]
    require(re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', TAG), 'Tag must begin with a letter and contain letters/digits/underscore')
    require(Path.cwd().resolve() == Path(sys.path[0] or '.').resolve() and Path('otr_common.py').is_file(),
            'Run this driver from O/trainer')
    start_params, start_head = require_file(params_arg).resolve(), require_file(head_arg).resolve()
    head_values(start_head)
    first, last, first_batch = int(first_arg), int(last_arg), int(batch_arg)
    require(first == last == 1 and first_batch == 0, 'First-point driver requires FIRST=LAST=1 and FIRST_BATCH=0')
    sizing_path = require_file(sizing_arg).resolve()
    check_path = require_file(os.environ.get('JM_CHECK', '')).resolve()
    judge_path = require_file(os.environ.get('JM_JUDGE', 'cert/refit/gen_judge32_v4.txt')).resolve()
    sizing, check, judge = ranks_file(sizing_path), ranks_file(check_path), ranks_file(judge_path)
    judge50 = ranks_file('cert/runM_val50.txt')
    judge48 = ranks_file('cert/runM_val48_clean.txt')
    require(len(judge50) == 50 and len(judge48) == 48 and set(judge48) <= set(judge50), 'Invalid secondary judge sets')
    extras = json.loads(extra_arg or '[]')
    request_extra = json.loads(os.environ.get('JM_REQ') or '{}')
    hsplit = json.loads(os.environ.get('JM_HSPLIT') or '{}')
    require(isinstance(extras, list) and isinstance(request_extra, dict) and isinstance(hsplit, dict),
            'Expected JM_EXTRA list, JM_REQ object, and JM_HSPLIT object')
    require(not {'id', 'rank', 'params', 'mode', 'scale', 'hu', 'keep_final', 'keep_all'} & set(request_extra),
            'JM_REQ must not override request identity, parameters, head, mode, or saved predictions')
    common = dict(f4c=1, ss=0, hf=2, b6=1, ncar=[dict(module='n6y', dfit=2)] +
                  [dict(module='k31ly', age=age, k31=float(age == 2), leaf='nkl%d' % age)
                   for age in (1, 2, 3)] + extras)
    if hsplit:
        common['hsplit'] = hsplit
    common.update(request_extra)
    from otr_generated_manifest_v1 import TruthPlan
    TRUTH_PLAN = TruthPlan(os.environ['TRAINER_TRUTH_PLAN'], os.environ['TRAINER_TRUTH_PLAN_SHA256'], ranks_file(os.environ['TRAINER_GENERATED_ALLOW_FILE']))
    require(common.get('truth_plan') == TRUTH_PLAN.spec(), 'Explicit request truth plan required')
    require(not common.get('hbands') and not common.get('b7') and not common.get('hsplit')
            and common.get('hf') == 2 and common.get('b6') == 1,
            'First-point HU mask requires hf=2,b6=1,hbands/B7/HSPLIT off')
    reused_label = os.environ.get('JM_G1', '')
    previous = os.environ.get('JM_PREV', '')
    require(not reused_label, 'JM_G1 must be empty: all96 first-point gradients must run at this request configuration')
    require(not previous, 'JM_PREV must be empty: no historical comparison in this first-point report')
    gradient_path = require_file(os.environ.get('JM_GRAD_RANKS', 'cert/refit/astra_cut_first96_v1.txt')).resolve()
    selection_path = require_file(os.environ.get('JM_GRAD_SELECTION', 'cert/refit/astra_cut_first96_v1.selection.json')).resolve()
    prescribed = ranks_file(gradient_path)
    require(len(prescribed) == len(set(prescribed)) == 96, 'Exactly96 unique prescribed gradients required')
    drop = [int(value) for value in os.environ.get('JM_DROP', '').split(',') if value]
    require(not drop, 'JM_DROP must be empty: freeze one final variant before the round')
    exporter = require_file(os.environ.get('JM_EXPORT_TOOL', 'export_dials_s46_v2.py')).resolve()
    for folder in (RUN / 'queue', RUN / 'claimed', RESULTS):
        require(folder.is_dir(), 'Missing live trainer directory: %s' % folder)
    prefix = 'a_0m' + TAG
    prior = list(REFIT.glob('jm' + TAG + '*')) + list(Path('candidates').glob('jm' + TAG + '*'))
    for run in Path('.').glob('run_*'):
        for section in ('queue', 'claimed', 'results'):
            prior.extend((run / section).glob(prefix + '*'))
    require(not prior, 'Fresh tag required; existing artifact: %s' % (prior[0] if prior else TAG))
    launch = REFIT / ('jm%s_launch.json' % TAG)
    write_json(launch, dict(utc=utc(), tag=TAG, driver='cert/refit/jmix24.sh', argv=sys.argv[1:],
                           source_files={name: fingerprint(path) for name, path in {
                               'driver': 'cert/refit/jmix24.sh', 'solver': 'cert/refit/jmix_generated_v4.py',
                               'protocol': 'cert/refit/jmix_generated_protocol_v6.py', 'exporter': exporter,
                               'worker': 'worker_generated_v4.py', 'truth_adapter': 'otr_generated_manifest_v1.py',
                               'truth_result_check': 'otr_truth_records_v1.py', 'truth_plan': TRUTH_PLAN.path,
                           }.items()},
                           params=fingerprint(start_params), head=fingerprint(start_head), request=common,
                           sizing=str(sizing_path), check=str(check_path), generated_judge=str(judge_path),
                           gradient_ranks=fingerprint(gradient_path), gradient_selection=fingerprint(selection_path),
                           prescribed_gradient_count=96, first_point_only=True, approved_hu_rows=sorted(APPROVED_HU_ROWS),
                           truth_policy='sizing16 required at launch; check16 may be pending until exact acceptance waits',
                           reused_gradient=reused_label, historical_previous=previous, drop=drop,
                           environment={name: os.environ.get(name, '') for name in
                                        ('JM_DIALS', 'JM_DIAL_CAP', 'JM_ARR_SPLIT', 'TRAINER_GENERATED_ALLOW_FILE', 'JM_GRAD_RANKS', 'JM_GRAD_SELECTION')}))
    os.environ.update(JM_EXTRA=extra_arg or '[]', JM_SIZING=str(sizing_path), JM_KC_SPLIT='1')
    protocol = REFIT / ('jm%s_protocol.json' % TAG)
    command(['cert/refit/jmix_generated_protocol_v6.py', '--sizing', sizing_path, '--check', check_path,
             '--judge', judge_path, '--gradient-ranks', gradient_path, '--gradient-selection', selection_path,
             '--first-batch', first_batch, '--steps', last - first + 1, '--gradient-label', reused_label,
             '--params', start_params, '--head', start_head, '--out', protocol],
            REFIT / ('jm%s_protocol.out' % TAG), heavy=False)
    protocol_data = json.loads(require_file(protocol).read_text())
    require(protocol_data.get('protocol') == 'jmix_generated_protocol_v6', 'Wrong preflight manifest')
    require(protocol_data.get('prescribed_gradient_count') == 96 and len(protocol_data['planned_gradients']) == 1
            and protocol_data['planned_gradients'][0]['ranks'] == prescribed, 'Preflight changed exact96 prescription')
    require(protocol_data['files']['gradient_ranks'] == fingerprint(gradient_path)
            and protocol_data['files']['gradient_selection'] == fingerprint(selection_path), 'Gradient prescription identity changed')
    # Confirm requested historical context exists before committing any compute.
    if previous:
        pair_stats(judge48, previous, previous)
    params, head, accepted = start_params, start_head, first - 1
    base_label, check_label = '0m%s%d_base' % (TAG, first), '0m%s%d_chk' % (TAG, first)
    current_base = queue_group(base_label, params, head, 'evalkeep', sizing, common)
    current_check = queue_group(check_label, params, head, 'eval', check, common)
    for step in range(first, last + 1):
        directory = REFIT / ('jm%s_d%d' % (TAG, step))
        directory.mkdir(exist_ok=False)
        planned = protocol_data['planned_gradients'][step - first]
        gradient_ranks = planned['ranks']
        require(len(gradient_ranks) == len(set(gradient_ranks)) == 96 and gradient_ranks == prescribed,
                'Exact96 gradient plan changed before queueing')
        gradient_label = '0m%s%dg' % (TAG, step)
        gradients = queue_group(gradient_label, params, head, 'grad', gradient_ranks, common)
        require(len(gradients) == 96 and len({r['rank'] for r in gradients}) == 96, 'Gradient requests lost prescribed ranks')
        # Pending check truths must not hold up gradient/arm/sizing work.
        wait_exact(gradients + current_base)
        command(['cert/refit/jmix_generated_v4.py', 'arms', gradient_label, params, head, directory],
                directory / 'arms.out')
        arm_manifest = json.loads(require_file(directory / 'arms.json').read_text())
        arms = arm_manifest.get('live', [])
        require(arms and len(arms) == len(set(arms)) and all(re.fullmatch(r'[A-Za-z0-9_]+', arm) for arm in arms),
                'Invalid live-arm manifest')
        require(arm_manifest.get('networks') == 96, 'Arms must consume exactly96 prescribed gradient networks')
        arm_requests = []
        for arm in arms:
            arm_requests.extend(queue_group('0m%s%d_%s' % (TAG, step, arm), directory / ('arm_%s.pt' % arm),
                                            directory / ('arm_%s_hu.json' % arm), 'evalkeep', sizing, common))
        wait_exact(arm_requests)
        command(['cert/refit/jmix_generated_v4.py', 'solve', directory, '0m%s%d' % (TAG, step)],
                directory / 'solve.out')
        solved = json.loads(require_file(directory / 'solve.json').read_text())
        require(set(solved.get('weights', {})) == set(arms)
                and all(math.isfinite(value) for value in solved['weights'].values())
                and math.isfinite(solved['model_pct']) and math.isfinite(solved['loo_pct']), 'Invalid solve result')
        new_params, new_head = REFIT / ('jm%s_p%d.pt' % (TAG, step)), REFIT / ('jm%s_p%d_hu.json' % (TAG, step))
        require(not new_params.exists() and not new_head.exists(), 'Point artifacts already exist')
        command(['cert/refit/jmix_generated_v4.py', 'point', directory, new_params, new_head], directory / 'point.out')
        require_file(new_params); head_values(new_head)
        next_base_label, next_check_label = '0m%s%d_base' % (TAG, step + 1), '0m%s%d_chk' % (TAG, step + 1)
        next_check = queue_group(next_check_label, new_params, new_head, 'eval', check, common)
        next_base = queue_group(next_base_label, new_params, new_head, 'evalkeep', sizing, common)
        # Acceptance requires every baseline and candidate check result.
        wait_exact(current_check + next_base + next_check)
        fit = pair_stats(sizing, next_base_label, base_label)
        real = pair_stats(check, next_check_label, check_label)
        go = real['change_pct'] < -real['se_pct']
        write_json(directory / 'acceptance.json', dict(utc=utc(), accepted=go, criterion='change_pct < -se_pct',
                                                      sizing=fit, independent_check_selection=real, solve=solved))
        # Root authorized one point only. Archive a reviewable receipt and stop
        # before any next gradient, secondary judge, parity or export request.
        launch_data = json.loads(require_file(launch).read_text())
        for name, pin in launch_data['source_files'].items():
            require(fingerprint(pin['path']) == pin, 'Source changed during first point: ' + name)
        require(fingerprint(gradient_path) == launch_data['gradient_ranks']
                and fingerprint(selection_path) == launch_data['gradient_selection'], 'Prescription changed during first point')
        request_hashes = {name: hashlib.sha256(json.dumps(req, sort_keys=True, allow_nan=False).encode()).hexdigest()
                          for name, req in sorted(REQUESTS.items())}
        terminal = REFIT / ('jm%s_first_point.json' % TAG)
        write_json(terminal, dict(schema='astra_cut_first_point_v1', utc=utc(), tag=TAG, accepted=go,
                                 criterion='change_pct < -se_pct', first_point_only=True,
                                 next_compute_requires_lead_review=True, final_handover=False,
                                 gradient_count=96, approved_hu_rows=sorted(APPROVED_HU_ROWS),
                                 gradient_ranks=launch_data['gradient_ranks'],
                                 gradient_selection=launch_data['gradient_selection'], gradient_order=prescribed,
                                 start_params=fingerprint(start_params), start_head=fingerprint(start_head),
                                 trial_params=fingerprint(new_params), trial_head=fingerprint(new_head),
                                 standing_params=fingerprint(new_params if go else start_params),
                                 standing_head=fingerprint(new_head if go else start_head),
                                 request=common, request_sha256=hashlib.sha256(json.dumps(common, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                                 full_request_sha256_by_id=request_hashes, source_files=launch_data['source_files'],
                                 launch=fingerprint(launch), protocol=fingerprint(protocol),
                                 launch_truth_readiness=protocol_data['fresh_launch']['truth_readiness'],
                                 sizing=fit, independent_check_selection=real, solve=solved,
                                 acceptance=fingerprint(directory / 'acceptance.json'),
                                 generated_untouched_judge_reserved=judge,
                                 generated_judging_complete=False, native_qualification_complete=False,
                                 public_judging_complete=False, export_complete=False))
        log('JOINT REFIT %s FIRST POINT %s: exactly96 prescribed gradients; generated check %s; sizing %s. '
            'STOPPED for lead review before more compute. Receipt %s is NOT a final handover or qualification.' %
            (TAG, 'ACCEPTED' if go else 'NOT ACCEPTED', real['summary'], fit['summary'], terminal))
        return
    if accepted == first - 1:
        log('JOINT REFIT %s: NO STEP ACCEPTED; start point stands (%s); no final judging or export queued.' %
            (TAG, start_params))
        return
    start_label, final_label, parity_label = '0m%sSTARTJ' % TAG, '0m%sJ' % TAG, '0m%spar' % TAG
    terminal = queue_group(start_label, start_params, start_head, 'eval', judge50, common)
    terminal += queue_group(final_label, params, head, 'eval', judge50, common)
    terminal += queue_group(parity_label, params, head, 'evalall', [651, 378], common)
    drop_head, drop_label, drop_parity = None, None, None
    if drop:
        drop_head = REFIT / ('jm%s_final_drop_hu.json' % TAG)
        values = head_values(head)
        require(all(row < len(values) for row in drop), 'Dropped row absent from accepted head')
        for row in drop:
            values[row] = [0.0] * len(values[row])
        write_json(drop_head, values)
        drop_label, drop_parity = '0m%sJd' % TAG, '0m%spard' % TAG
        terminal += queue_group(drop_label, params, drop_head, 'eval', judge50, common)
        terminal += queue_group(drop_parity, params, drop_head, 'evalall', [651, 378], common)
    wait_exact(terminal)
    comparisons = {'same_config_start48': pair_stats(judge48, final_label, start_label),
                   'same_config_start50': pair_stats(judge50, final_label, start_label)}
    if previous:
        comparisons['historical_previous48_context_only'] = pair_stats(judge48, final_label, previous)
    for name, label in [('S42_context_only', '0hS41_u2'), ('S44_context_only', '0mCJq')]:
        if all((RESULTS / ('a_%s_r%04d.json' % (label, rank))).is_file() for rank in judge48):
            comparisons[name] = pair_stats(judge48, final_label, label)
    if drop:
        comparisons['drop_same_config_start48'] = pair_stats(judge48, drop_label, start_label)
        comparisons['full_vs_drop48'] = pair_stats(judge48, final_label, drop_label)
    candidate = Path('candidates') / ('jm' + TAG)
    require(not candidate.exists(), 'Candidate destination already exists')
    note = ('JOINT REFIT %s (jmix24.sh), %d accepted steps; public clean48 SECONDARY versus round start '
            'under identical simulator/request: %s. Untouched generated judging remains required.' %
            (TAG, accepted - first + 1, comparisons['same_config_start48']['summary']))
    command([exporter, params, candidate, note], REFIT / ('jm%s_export.out' % TAG))
    require_file(candidate / 'candidate.py'); require_file(candidate / 'physics_dials.json')
    export_record = json.loads((candidate / 'physics_dials.json').read_text())
    candidate_fingerprint = fingerprint(candidate / 'candidate.py')
    require(export_record.get('sha256') == candidate_fingerprint['sha256'], 'Export candidate hash mismatch')
    sidecar_meta = export_record.get('per_layer_sidecar')
    require(isinstance(sidecar_meta, dict), 'Exporter omitted required per-layer sidecar metadata')
    sidecar_path = require_file(candidate / 'per_layer_dials.json').resolve()
    require(Path(sidecar_meta.get('path', '')).resolve() == sidecar_path, 'Export sidecar path mismatch')
    sidecar_fingerprint = fingerprint(sidecar_path)
    require(sidecar_meta.get('sha256') == sidecar_fingerprint['sha256'], 'Export sidecar hash mismatch')
    sidecar_record = json.loads(sidecar_path.read_text())
    require(sidecar_record.get('schema') == sidecar_meta.get('schema') == 'trainer_pld_s46_v1',
            'Export sidecar schema mismatch')
    require(Path(sidecar_record.get('source', '')).resolve() == Path(params).resolve(),
            'Export sidecar source checkpoint mismatch')
    require(sidecar_meta.get('reconstruction_all16_exact') is True
            and sidecar_meta.get('arrays_roundtrip_exact') is True,
            'Exporter did not certify its serialized roundtrip')
    compile((candidate / 'candidate.py').read_text(), str(candidate / 'candidate.py'), 'exec')
    exported_head = candidate / 'head_weights_42x128.json'
    with exported_head.open('xb') as stream:
        stream.write(Path(head).read_bytes())
    require(head_values(exported_head) == head_values(head), 'Exported head mismatch')
    artifacts = {'candidate': candidate_fingerprint, 'physics_dials': fingerprint(candidate / 'physics_dials.json'),
                 'per_layer_dials': sidecar_fingerprint,
                 'head': fingerprint(exported_head), 'params': fingerprint(params), 'protocol': fingerprint(protocol)}
    if drop:
        exported_drop = candidate / ('head_weights_42x128_without_rows_%s.json' % ','.join(map(str, drop)))
        with exported_drop.open('xb') as stream:
            stream.write(drop_head.read_bytes())
        require(head_values(exported_drop) == head_values(drop_head), 'Exported dropped head mismatch')
        artifacts['head_without_rows'] = fingerprint(exported_drop)
    check_failures()
    require(fingerprint(sidecar_path) == sidecar_fingerprint, 'Sidecar changed before handover')
    receipt = REFIT / ('jm%s_handover.json' % TAG)
    write_json(receipt, dict(utc=utc(), tag=TAG, accepted_steps=accepted - first + 1,
                            last_accepted_step=accepted, artifacts=artifacts, request=common,
                            comparisons=comparisons, generated_untouched_judge_reserved=judge,
                            launch_truth_readiness=protocol_data['fresh_launch']['truth_readiness'],
                            generated_judging_complete=False, native_qualification_complete=False,
                            parity=[str(RESULTS / ('a_%s_r%04d.json' % (parity_label, rank))) for rank in (651, 378)],
                            drop_parity=([str(RESULTS / ('a_%s_r%04d.json' % (drop_parity, rank))) for rank in (651, 378)]
                                         if drop else [])))
    log('JOINT REFIT %s SECONDARY PUBLIC JUDGEMENT COMPLETE, frozen accepted point evaluated once: clean48 '
        'versus ROUND START under identical simulator/request %s. Export verified: %s; receipt %s; '
        'parity all16 on651/378 complete. Generated seeds in %s remain untouched and REQUIRED before release; '
        'historical comparisons in receipt are context only.' %
        (TAG, comparisons['same_config_start48']['summary'], candidate, receipt, judge_path))


try:
    main()
except Exception as error:
    traceback.print_exc()
    try:
        log('JOINT REFIT %s FAILED; no successful handover claim. %s: %s. All artifacts preserved.' %
            (TAG or '<unparsed>', type(error).__name__, error))
    except Exception:
        traceback.print_exc()
    sys.exit(1)
PYEOF
