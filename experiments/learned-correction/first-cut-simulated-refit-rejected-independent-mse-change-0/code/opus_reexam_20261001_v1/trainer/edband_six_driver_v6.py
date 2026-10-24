"""One EDBAND point: prescribed public24 -> six arms -> fresh16 fit -> check16.

SOURCE PREPARATION ONLY until a lead-reviewed authorization and genuine latest-
best full-card/native qualification exist. This does not launch a worker, retry,
resume, export, judge, or submit. Run a separately guarded, pinned worker v2 in a
new dedicated run directory. The allowlist is sizing+acceptance only. A worker
may wait for immutable A30 completion receipts while public gradients progress.

CLI: python -B edband_six_driver_v6.py CONFIG.json NEW_OUTPUT_DIRECTORY
Internal --inspect/--gradients commands deserialize only pinned checkpoints or
gradient artifacts, never run a network. Source imports are Torch/NumPy free.
All subprocesses run only while the worker queue is empty and every issued
request has a result. Existing artifacts are never replaced; a failed round
requires lead review and a new independent-check assignment, not a resume.
"""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
O = HERE.parent
RULE = 'change_pct < -se_pct'
PLAN = {'path': str(O / 'analyst/a30_edband_selection_and_final_20261011T0057Z/plan_v2.json'),
        'sha256': 'b35e8a1c7c83d4018ea44b8b01464f96069cab02f1583f2e0bfbfbc9ee758f4a'}
RANKS96 = {'path': str(HERE / 'cert/refit/astra_cut_first96_v1.txt'),
           'sha256': '666884abbc91276d50d9253b4311eeefbf2abbea862d67d81b4b12b2ed58bb02'}
SELECTION = {'path': str(HERE / 'cert/refit/astra_cut_first96_v1.selection.json'),
             'sha256': '250753620c02cb6fad2fed02185e274eede0d0e71b587c5774cdc93e33076de4'}
SIZING = list(range(7100232, 7100248))
CHECK = list(range(7100248, 7100264))
SEALED = list(range(7100200, 7100232))
SOURCES = {
    'edband_qualification_v6.py': '91b06e8cdc7701bfb0b587eb305f60fd16c2370b74144e30d2ae73e262829ba0',
    'cert/refit/jmix_protocol_v1.py': '8c82eb2261cf68996368f874889d596a44ac179ba5fb597b9576f52d484761d6',
    'edband_six_arms_v3.py': '272a1a070fedec8930494196d49336e957c73106075730979162304a312961f0',
    'edband_six_fit_v3.py': '86d632fad3b2d4b238c19847156fa478f610bad20d1e1317ce611dcfb32c17ec',
    'otr_edband_v1.py': '24a7debb6f8994982fb4052d01b9f28ca07729a8a179a1628c8bac9b6bb4ffc6',
    'otr_common.py': '5e87f0b5f768de0e9bdaec63ac49b4af31c10f52cb6d01e00931d11ba83ae8a2',
    'otr_generated_manifest_v1.py': '95987e2426b444648530c24323017a313dc6a053e5a4fcb085fe26c722dde5e8',
    'otr_truth_records_v1.py': '9abbd1d15ba6086fb783991b25b9b54acacdaf5fc747c855872ed537ba1173fa',
    'worker_edband_v2.py': 'f1dae5b23057a08871e9691fbb42e0cbe7b0565e1d970c6eea273ed66618366f',
    'cert/refit/jmix_generated_protocol_v5.py': '1211e3e178784f8f8353acc56ace585814f68c6eb58b2b64f022fb79ef8f98cc',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return subprocess.check_output(['date', '-u', '+%FT%TZ'], text=True).strip()


def pin(path):
    path = Path(path).resolve()
    require(path.name != 'key.txt', 'Forbidden credential path')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return {'path': str(path), 'sha256': h.hexdigest()}


def read_pin(item):
    require(set(item) == {'path', 'sha256'} and Path(item['path']).is_absolute(), 'Exact absolute pin required')
    require(pin(item['path']) == item, 'Pinned file changed: ' + item['path'])
    return Path(item['path']).read_bytes()


def read_json(item):
    return json.loads(read_pin(item))


def write(path, value):
    # Exclusive final publication. Temp files remain if interrupted.
    path = Path(path)
    temp = path.with_name('.' + path.name + '.tmp%d' % os.getpid())
    with temp.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    os.link(temp, path)


def rank_file(item):
    ranks = [int(value) for value in read_pin(item).decode().strip().split(',')]
    require(ranks and len(ranks) == len(set(ranks)), 'Empty/duplicate ranks')
    return ranks


def pinned_sources():
    inventory = {}
    for name, sha in SOURCES.items():
        item = {'path': str(HERE / name), 'sha256': sha}
        read_pin(item)
        inventory[name] = item
    # Imports here are light and verify the exact worker capture transformation.
    import edband_six_arms_v3 as A
    expected = A.expected_worker()
    for name, sha in expected['sources'].items():
        item = {'path': str(HERE / name), 'sha256': sha}
        read_pin(item); inventory[name] = item
    inventory[Path(__file__).name] = pin(__file__)
    return inventory, expected


def prescribed24():
    ranks = rank_file(RANKS96)
    selection = read_json(SELECTION)
    require(len(ranks) == 96 and selection['selected_order'] == ranks
            and selection['selection_uses_gradient_or_result_values'] is False,
            'Original eligible/public prescription changed')
    chosen = ranks[:24]
    require(set(chosen) <= set(selection['eligible_order']), 'First24 outside eligible pool')
    source = selection['source_files']
    for name, item in source.items():
        for entry in item if isinstance(item, list) else [item]:
            read_pin(entry)
    split = read_json(source['split'])
    require(set(chosen) <= set(split['train']) and all(0 <= r < 7_000_000 for r in chosen), 'Not public training')
    holds = []
    for entry in selection['hold_files']:
        item = {k: entry[k] for k in ('path', 'sha256')}
        values = rank_file(item)
        require(set(values) == set(entry['ranks']) and not set(chosen) & set(values), 'Historical hold overlap')
        holds.append(item)
    for values in selection['split_hold_roles'].values():
        require(not set(chosen) & set(values), 'Split hold overlap')
    return chosen, split['train'], holds


def plan_object():
    from otr_generated_manifest_v1 import TruthPlan
    plan = TruthPlan(PLAN['path'], PLAN['sha256'], SIZING + CHECK)
    require(list(plan.entries) == SIZING + CHECK + SEALED, 'A30 role/order assignment changed')
    for role, ranks in [('sizing', SIZING), ('acceptance', CHECK), ('final', SEALED)]:
        require(all(plan.entries[r]['role'] == role and plan.entries[r]['N'] == 15990784 for r in ranks),
                'A30 exact role/N changed')
    require(not plan.allowed & set(SEALED), 'Final seeds exposed')
    return plan


def validate_config(config):
    fields = {'schema', 'tag', 'checkpoint', 'head', 'request_template', 'qualification_receipt',
              'authorization', 'runtime', 'timeout_seconds'}
    require(set(config) == fields and config['schema'] == 'trainer_edge_six_driver_config_v1', 'Driver config schema')
    require(re.fullmatch('[A-Za-z0-9_]+', config['tag']) is not None, 'Unsafe tag')
    require(type(config['timeout_seconds']) is int and 300 <= config['timeout_seconds'] <= 14400, 'Bounded deadline required')
    sources, expected = pinned_sources()
    gradients, public, holds = prescribed24()
    template = read_json(config['request_template'])
    head = read_json(config['head'])
    read_pin(config['checkpoint'])
    require(len(head) == 42 and all(len(row) == 128 and all(math.isfinite(x) for x in row) for row in head), 'Head shape/finite')
    require(all(all(x == 0 for x in row) for i, row in enumerate(head) if i not in {6, 17, 20, 21, 28, 29}), 'Unqualified HU rows')
    import otr_edband_v1 as EB
    require(template['params'] == config['checkpoint']['path'] and template['hu'] == head
            and template.get('edge_band') == EB.request_spec(config['checkpoint']['path'], config['checkpoint']['sha256'], ['bias', 'output'])
            and template.get('truth_plan') == PLAN, 'Base template identity')
    profile_pin = {'path': str(O / 'engineer/a22_cutsim/S46_v11_PINNED_PROFILE.json'),
                   'sha256': '06fe3a1b601a7893f61725a8ddee17a590039ab21aaaf7477385b466890a2f7b'}
    profile = read_json(profile_pin)
    sources['cutsim/profile'] = profile_pin
    require(template.get('cutsim') == profile, 'Exact certified v11 profile required')
    module_dir = Path(profile['module']).parent
    require(pin(profile['module'])['sha256'] == profile['sha256'], 'Simulator entry changed')
    for name, sha in profile['sources'].items():
        item = {'path': str(module_dir / name), 'sha256': sha}
        read_pin(item); sources['cutsim/' + name] = item
    qualification = read_json(config['qualification_receipt'])
    require(qualification.get('schema') == 'trainer_edge_band_fullcard_qualification_v1'
            and qualification.get('zero_offset_all16_exact') is True
            and qualification.get('native_parity_pass') is True
            and qualification.get('worker_sha256') == expected['worker_sha256']
            and qualification.get('cutsim') == profile
            and qualification.get('checkpoint') == config['checkpoint']
            and qualification.get('head') == config['head']
            and qualification.get('request_template') == config['request_template'],
            'Genuine latest-point EDBAND qualification must cover the exact full request template')
    require(not template.get('dump15') and not template.get('featstats'), 'Diagnostic side effects are forbidden in this fitting driver')
    for name, item in qualification.get('runtime_sources', {}).items():
        read_pin(item)
        sources['qualified_runtime/' + name] = item
    runtime = config['runtime']
    require(set(runtime) == {'run_directory', 'worker_pid', 'guard_pid', 'worker_handle', 'guard_terminal', 'threads', 'allowlist'},
            'Runtime fields')
    run = Path(runtime['run_directory'])
    require(run.is_absolute() and run.is_relative_to(HERE) and run.name.startswith('run_edband_'), 'Dedicated trainer run directory required')
    require(type(runtime['threads']) is int and runtime['threads'] in (3, 4, 8)
            and qualification.get('threads') == runtime['threads'], 'Qualified thread count required')
    require(rank_file(runtime['allowlist']) == SIZING + CHECK, 'Worker exact selection-only allowlist')
    for name in ('worker_pid', 'guard_pid'):
        require(type(runtime[name]) is int and runtime[name] > 1, 'Runtime PID required')
    require(Path(runtime['guard_terminal']).is_absolute(), 'Absolute guard terminal required')
    handle = read_json(runtime['worker_handle'])
    require(handle.get('role') == 'worker' and handle.get('pid') == runtime['guard_pid']
            and handle.get('threads') == runtime['threads'], 'Guard/worker launch identity')
    env = handle.get('environment', {})
    expected_env = {'TRAINER_CARD': 'M', 'TRAINER_PL': '1', 'TRAINER_F4C': '1', 'TRAINER_RELIEF': '1',
        'TRAINER_STAR_ATTACH': '0', 'TRAINER_BIRTH_ATTACH': '0', 'TRAINER_THREADS': str(runtime['threads']),
        'TRAINER_TRUTH_PLAN': PLAN['path'], 'TRAINER_TRUTH_PLAN_SHA256': PLAN['sha256'],
        'TRAINER_GENERATED_ALLOW_FILE': runtime['allowlist']['path']}
    expected_env.update({name: str(runtime['threads']) for name in
                         ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')})
    require(all(env.get(k) == v for k, v in expected_env.items()), 'Worker environment differs from protocol')
    command = handle.get('command', [])
    require(command[-3:] == [str(HERE / 'worker_edband_v2.py'), str(run), '0']
            and '--rss-backstop' in command and handle.get('automatic_restart') is False
            and handle.get('automatic_scaling') is False, 'One explicitly guarded pinned worker required')
    authorization = read_json(config['authorization'])
    bindings = dict(tag=config['tag'], driver=pin(__file__), checkpoint=config['checkpoint'], head=config['head'],
        request_template=config['request_template'], qualification_receipt=config['qualification_receipt'], truth_plan=PLAN,
        gradient_order=gradients, sizing_order=SIZING, acceptance_order=CHECK, sealed_judge_order=SEALED,
        threads=runtime['threads'], worker_count=1, criterion=RULE, first_point_only=True)
    require(authorization.get('schema') == 'trainer_edge_six_round_authorization_v1'
            and authorization.get('decision') == 'ROOT_REVIEWED'
            and authorization.get('bindings') == bindings,
            'Explicit lead-reviewed round/starting-point/resource assignment required')
    import edband_qualification_v6 as Q
    evidence_check = Q.verify(qualification, config, expected, O / 'engineer')
    return dict(sources=sources, expected_worker=expected, gradients=gradients, public=public,
                holds=holds, template=template, head=head, qualification=qualification, bindings=bindings,
                qualification_evidence_check=evidence_check)


def inspect_checkpoint(config, checkpoint, request_pin, output, baseline):
    import torch
    torch.set_num_threads(1)
    import otr_edband_v1 as EB
    payload = EB.load_checkpoint(torch, checkpoint['path'], checkpoint['sha256'])
    request = read_json(request_pin)
    session = EB.Session(torch, request['edge_band'], request)
    identity = session._identity
    require(request['params'] == checkpoint['path'] and request['hu'] == read_json(config['head']), 'Inspected checkpoint/head')
    require(identity['payload_sha256'] == EB._fingerprint(torch, payload), 'Inspected request/payload mismatch')
    if baseline:
        qualification = read_json(config['qualification_receipt'])
        require(identity['base_sha256'] == qualification['base_content_sha256'], 'Qualification covers another base')
        require(all(bool((values == 0).all()) for values in payload['edge_band']['offsets'].values()), 'Latest-best baseline must have exact zero offsets')
        import edband_qualification_v6 as Q
        plain_pin = qualification['latest_best_checkpoint']
        read_pin(plain_pin)
        plain = torch.load(plain_pin['path'], map_location='cpu', weights_only=True)
        require(EB._fingerprint(torch, plain) == EB._fingerprint(torch, payload['base']),
                'Complete latest-best plain checkpoint differs from zero-envelope base')
        for case in qualification['native_evidence']['cases']:
            measured_request = read_json(case['torch_request'])
            measured_result = read_json(case['torch_result'])
            require(EB.Session(torch, measured_request['edge_band'], measured_request)._identity == measured_result['edge_band'],
                    'Actual enabled control identity differs from checkpoint inspection')
        probe, q, c = Q.probe_context(qualification, config)
        trial = EB.load_checkpoint(torch, probe['checkpoint']['path'], probe['checkpoint']['sha256'])
        require(EB._fingerprint(torch, trial['base']) == EB._fingerprint(torch, plain)
                and EB._fingerprint(torch, trial) == probe['payload_sha256'], 'Fixed probe changed the full plain base or payload binding')
        prescription = read_json(probe['prescription'])
        require(trial['source'].get('probe_prescription') == probe['prescription'], 'Probe checkpoint was not frozen with its prescription')
        for family in ('bias', 'output'):
            wanted = torch.tensor(prescription['offsets'][family], dtype=torch.float32)
            require(torch.equal(trial['edge_band']['offsets'][family], wanted)
                    and all(bool((row != 0).any()) for row in wanted), 'Probe offsets differ from the fixed nonzero prescription')
        for case in probe['native_evidence']['cases']:
            measured_request = read_json(case['torch_request'])
            measured_result = read_json(case['torch_result'])
            require(EB.Session(torch, measured_request['edge_band'], measured_request)._identity == measured_result['edge_band'],
                    'Actual nonzero probe identity differs from checkpoint inspection')

    write(output, dict(checkpoint=checkpoint, request=request_pin, identity=identity, inspection_only=True,
                       network_execution=False, truth_read=False, script=pin(__file__)))


def inspect_gradients(config, arm_config_path, output):
    import torch
    torch.set_num_threads(1)
    import edband_six_arms_v3 as A
    import otr_edband_v1 as EB
    from otr_truth_records_v1 import validate_record
    arms = json.loads(Path(arm_config_path).read_text())
    ranks, unused = A.validate_config(arms)
    require(ranks == prescribed24()[0], 'Gradient inspection rank prescription')
    payload = EB.load_checkpoint(torch, config['checkpoint']['path'], config['checkpoint']['sha256'])
    plan = plan_object()
    for rank, item in zip(ranks, arms['gradient_requests']):
        request = read_json(item['request'])
        read_pin(item['result'])
        record = torch.load(item['result']['path'], map_location='cpu', weights_only=False)
        A.verify_result(torch, arms, payload, request, record)
        validate_record(plan, record, rank)
        require(record['rank'] == rank and record.get('threads') == config['runtime']['threads'], 'Gradient rank/threads')
        require(len(record.get('mse_layers', [])) == 16 and all(math.isfinite(x) and x >= 0 for x in record['mse_layers']), 'Gradient layers')
        del record
    write(output, dict(schema='trainer_edge_six_gradient_verification_v1', count=24, ranks=ranks,
                       config=pin(arm_config_path), exact_request_worker_plan=True, script=pin(__file__)))


def pair_stats(candidate, baseline):
    require(len(candidate) == len(baseline) == 16, 'Sixteen complete independent pairs required')
    a, b = [r['mse'] for r in candidate], [r['mse'] for r in baseline]
    require(all(math.isfinite(x) and x > 0 for x in a + b), 'Invalid paired MSE')
    changes = [100 * (x / y - 1) for x, y in zip(a, b)]
    layers_a = [statistics.mean(r['mse_layers']) for r in candidate]
    layers_b = [statistics.mean(r['mse_layers']) for r in baseline]
    stats = dict(count=16, mean=statistics.mean(a), baseline_mean=statistics.mean(b),
                change_pct=100 * (statistics.mean(a) / statistics.mean(b) - 1),
                se_pct=statistics.stdev(changes) / math.sqrt(16), better=sum(x < y for x, y in zip(a, b)),
                all_layers_pct=100 * (statistics.mean(layers_a) / statistics.mean(layers_b) - 1))
    stats['summary'] = ('%d ranks: %.4e against %.4e: %+.3f%% (se %.3f, better on %d/%d; all layers %+.3f%%)' %
        (16, stats['mean'], stats['baseline_mean'], stats['change_pct'], stats['se_pct'], stats['better'], 16, stats['all_layers_pct']))
    return stats


class Round:
    def __init__(self, config_path, output):
        self.config_path, self.output = Path(config_path).resolve(), Path(output).resolve()
        self.config = json.loads(self.config_path.read_text())
        self.context = validate_config(self.config)
        self.plan = plan_object()
        self.run = Path(self.config['runtime']['run_directory'])
        self.requests, self.records, self.request_pins, self.result_pins = {}, {}, {}, {}
        self.deadline = time.monotonic() + self.config['timeout_seconds']
        self.started = False

    def health(self):
        require(time.monotonic() < self.deadline, 'Round deadline exceeded; no automatic retry')
        require(not (self.run / 'STOP').exists(), 'Worker STOP present')
        require(not Path(self.config['runtime']['guard_terminal']).exists(), 'Worker guard terminated')
        for name in ('worker_pid', 'guard_pid'):
            os.kill(self.config['runtime'][name], 0)
        errors = list((self.run / 'results').glob('*.error'))
        require(not errors, 'Worker failure: ' + str(errors[0] if errors else ''))
        require(all(path.stem in self.requests for path in (self.run / 'queue').glob('*.json')), 'Foreign queued request')

    def verify_pins(self):
        for item in self.context['sources'].values():
            read_pin(item)
        for key in ('checkpoint', 'head', 'request_template', 'qualification_receipt', 'authorization'):
            read_pin(self.config[key])
        read_pin(self.config_pin); read_pin(PLAN); read_pin(RANKS96); read_pin(SELECTION)

    def command(self, arguments, label):
        self.health(); self.verify_pins()
        require(not list((self.run / 'queue').glob('*.json')), 'Helper needs drained worker queue')
        require(all(self.result_path(req).is_file() for req in self.requests.values()), 'Helper cannot overlap unfinished neural request')
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
        with (self.output / (label + '.log')).open('x') as stream:
            process = subprocess.Popen([sys.executable, '-B', *map(str, arguments)], cwd=HERE, env=env,
                                       stdout=stream, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    self.health()
                    time.sleep(2)
                require(process.returncode == 0, 'Local command failed: ' + label)
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=20)
                raise
        self.verify_pins()

    def result_path(self, request):
        return self.run / 'results' / (request['id'] + ('.pt' if request['mode'] == 'grad' else '.json'))

    def group(self, label, template, ranks, mode):
        self.health(); self.verify_pins()
        require(len(ranks) == len(set(ranks)) and set(ranks) <= set(self.context['gradients'] if mode == 'grad' else SIZING + CHECK), 'Request rank/role rejected')
        from otr_common import enqueue
        requests = []
        for rank in ranks:
            request = dict(template, id='a_' + self.config['tag'] + '_' + label + '_r%04d' % rank,
                           rank=rank, mode=mode)
            for key in ('scale', 'keep_all', 'keep_final'):
                request.pop(key, None)
            request.update({'scale': 1e8} if mode == 'grad' else {'keep_final': 1})
            name = request['id']
            require(name not in self.requests and not self.result_path(request).exists(), 'Duplicate request/result')
            require(not list((self.run / 'claimed').glob(name + '.w*.json')), 'Unexpected prior claim')
            path = self.output / 'requests' / (name + '.json')
            write(path, request)
            self.requests[name] = request; self.request_pins[name] = pin(path)
            require(enqueue(self.run, request), 'Queue rejected unique request: ' + name)
            requests.append(request)
        print(utc(), label, 'queued', len(requests), flush=True)
        return requests

    def wait(self, requests, identity=None):
        require(len({r['id'] for r in requests}) == len(requests), 'Duplicate expected IDs')
        pending = dict((r['id'], r) for r in requests)
        last = 0.0
        while pending:
            self.health()
            for name, request in list(pending.items()):
                path = self.result_path(request)
                if not path.is_file():
                    continue
                require(path.stat().st_size > 0, 'Empty result')
                self.result_pins[name] = pin(path)
                if request['mode'] == 'eval':
                    import edband_six_fit_v3 as F
                    from otr_truth_records_v1 import validate_record
                    require(identity is not None, 'Missing frozen eval identity')
                    checkpoint = dict(path=request['params'], sha256=request['edge_band']['checkpoint_sha256'])
                    row = F.load_eval(dict(request=self.request_pins[name], result=self.result_pins[name]), checkpoint,
                                      request, request['rank'], identity['payload_sha256'], identity['base_sha256'])
                    validate_record(self.plan, row, request['rank'])
                    require(row.get('threads') == self.config['runtime']['threads'], 'Evaluation thread identity')
                    require(len(row.get('mse_layers', [])) == 16 and all(math.isfinite(x) and x >= 0 for x in row['mse_layers']), 'Evaluation layers')
                    self.records[name] = row
                del pending[name]
            if pending:
                if time.monotonic() - last >= 300:
                    print(utc(), 'waiting', len(pending), '/', len(requests), 'first', next(iter(pending)), flush=True)
                    last = time.monotonic()
                time.sleep(15)
        self.verify_pins()

    def pairs(self, requests):
        return [dict(request=self.request_pins[r['id']], result=self.result_pins[r['id']]) for r in requests]

    def terminal(self, status, **fields):
        self.verify_pins()
        write(self.output / 'DECISION.json', dict(schema='trainer_edge_six_one_point_decision_v1', utc=utc(),
            status=status, first_point_only=True, criterion=RULE, config=self.config_pin,
            bindings=self.context['bindings'], source_files=self.context['sources'],
            gradient_count=24, source_gradient_order=self.context['gradients'],
            independent_check_is_selection_evidence=True, final_judge_read=False,
            sealed_judge_ranks=SEALED, native_candidate_qualified=False, exported=False,
            next_compute_requires_lead_review=True, **fields))
        print(utc(), status, str(self.output / 'DECISION.json'), flush=True)

    def execute(self):
        self.health()
        require(not self.output.exists(), 'Output exists; this driver has no resume mode')
        for name in ('queue', 'claimed', 'results'):
            require((self.run / name).is_dir() and not any((self.run / name).iterdir()), 'Dedicated worker run must be empty')
        self.output.mkdir(); (self.output / 'requests').mkdir()
        self.config_pin = pin(self.config_path)
        # Exclusive run-level STARTED blocks a different output path restarting
        # the same round after any partial sizing/check exposure.
        write(self.run / 'EDBAND_ROUND_STARTED.json', dict(utc=utc(), config=self.config_pin, output=str(self.output)))
        self.started = True
        try:
            # Reuse the existing conservative queue/claim/result freshness audit;
            # it reads trainer metadata only, never a truth or gradient tensor.
            spec = importlib.util.spec_from_file_location('_edge_fresh', HERE / 'cert/refit/jmix_generated_protocol_v5.py')
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            uses, inspected = module.prior_trainer_uses(set(SIZING + CHECK))
            require(not uses, 'Fresh A30 selection ranks already used by trainer: ' + json.dumps(uses))
            write(self.output / 'PREFLIGHT.json', dict(utc=utc(), sources=self.context['sources'], config=self.config_pin,
                assignment=self.context['bindings'], qualification_evidence_check=self.context['qualification_evidence_check'],
                trainer_records_inspected=inspected, prior_uses=uses,
                truth_readiness={str(r): self.plan.ready(r) for r in SIZING + CHECK},
                pending_truths_allowed=True, final_truth_readiness_not_inspected=True))
            for name, ranks in [('grad24', self.context['gradients']), ('public_training', self.context['public']),
                                ('sizing16', SIZING), ('acceptance16', CHECK), ('sealed32', SEALED)]:
                with (self.output / (name + '.txt')).open('x') as stream:
                    stream.write(','.join(map(str, ranks)) + '\n')
            self.command([__file__, '--inspect', self.config_path, 'base', self.output / 'BASE_IDENTITY.json'], 'inspect_base')
            base_identity = json.loads((self.output / 'BASE_IDENTITY.json').read_text())['identity']
            template = self.context['template']
            grads = self.group('01_grad', template, self.context['gradients'], 'grad')
            base_size = self.group('02_base_size', template, SIZING, 'eval')
            self.wait(grads)
            self.wait(base_size, base_identity)
            arm_config = dict(schema='trainer_edge_six_arm_config_v1', tag=self.config['tag'],
                checkpoint=self.config['checkpoint'], head=self.config['head'], gradient_ranks=pin(self.output / 'grad24.txt'),
                held_rank_files=self.context['holds'] + [pin(self.output / (name + '.txt')) for name in ('sizing16', 'acceptance16', 'sealed32')],
                public_training_ranks=pin(self.output / 'public_training.txt'), gradient_requests=self.pairs(grads),
                request_template=self.config['request_template'], worker=pin(HERE / 'worker_edband_v2.py'),
                qualification_receipt=self.config['qualification_receipt'], t_threshold=2.5,
                target_change_fraction=0.005, max_rms_multiple=0.1)
            write(self.output / 'arms_config.json', arm_config)
            self.command([__file__, '--gradients', self.config_path, self.output / 'arms_config.json', self.output / 'GRADIENTS_VERIFIED.json'], 'verify_gradients')
            self.command([HERE / 'edband_six_arms_v3.py', self.output / 'arms_config.json', self.output / 'arms'], 'arms')
            arms = json.loads((self.output / 'arms/arms.json').read_text())
            require(arms['gradient_ranks'] == self.context['gradients'] and arms['count'] == 24 and len(arms['proposals']) <= 6, 'Arms lost exact24 or exceeded six')
            if not arms['proposals']:
                self.terminal('NO_ACTIVE_ARMS', accepted=False, arms=pin(self.output / 'arms/arms.json'), independent_check_exposed=False)
                return
            evaluations = {'base': self.pairs(base_size)}
            # Each arm is independent. Its exact payload identity is obtained by
            # the same small inspection child while no neural request is live.
            for name, arm in arms['proposals'].items():
                args = [__file__, '--inspect', self.config_path, str(arm['request']['path']), self.output / ('IDENTITY_' + name + '.json')]
                self.command(args, 'inspect_' + name)
                identity = json.loads((self.output / ('IDENTITY_' + name + '.json')).read_text())['identity']
                group = self.group('03_' + name, read_json(arm['request']), SIZING, 'eval')
                self.wait(group, identity)
                evaluations[name] = self.pairs(group)
            fit_config = dict(schema='trainer_edge_six_fit_config_v1', arms=pin(self.output / 'arms/arms.json'),
                sizing_ranks=pin(self.output / 'sizing16.txt'), acceptance_ranks=pin(self.output / 'acceptance16.txt'),
                judge_ranks=pin(self.output / 'sealed32.txt'), truth_plan=PLAN, evaluations=evaluations,
                ridge_fraction=0.02, cache_directory=str(self.run / 'generated_cache_v1'), tag=self.config['tag'])
            write(self.output / 'fit_config.json', fit_config)
            self.command([HERE / 'edband_six_fit_v3.py', self.output / 'fit_config.json', self.output / 'point'], 'fit')
            point = json.loads((self.output / 'point/point.json').read_text())
            self.command([__file__, '--inspect', self.config_path, point['request']['path'], self.output / 'POINT_IDENTITY.json'], 'inspect_point')
            point_identity = json.loads((self.output / 'POINT_IDENTITY.json').read_text())['identity']
            # This immutable seal precedes the FIRST independent-check request.
            # There is exactly one fit above and no branch returns to fitting.
            seal = dict(utc=utc(), point=pin(self.output / 'point/point.json'), checkpoint=point['point'], request=point['request'],
                identity=pin(self.output / 'POINT_IDENTITY.json'), head=point['head'], sizing_config=pin(self.output / 'fit_config.json'),
                acceptance_order=CHECK, candidate_variants=1, check_looks=1)
            write(self.output / 'POINT_FROZEN_BEFORE_CHECK.json', seal)
            candidate = read_json(point['request'])
            base_check = self.group('04_base_check', template, CHECK, 'eval')
            point_check = self.group('05_point_check', candidate, CHECK, 'eval')
            self.wait(base_check, base_identity); self.wait(point_check, point_identity)
            real = pair_stats([self.records[r['id']] for r in point_check], [self.records[r['id']] for r in base_check])
            size = dict(evidence='sizing model and leave-one-network-out forecasts; not finite candidate acceptance',
                        model_change_pct=point['model_change_pct'], leave_one_out_change_pct=point['leave_one_out_change_pct'])
            for item in (seal['point'], seal['request'], seal['identity'], seal['head'], seal['sizing_config']):
                read_pin(item)
            read_pin({k: seal['checkpoint'][k] for k in ('path', 'sha256')})
            for item in [*self.request_pins.values(), *self.result_pins.values()]:
                read_pin(item)
            write(self.output / 'RESULT_INDEX.json', {name: {'request': self.request_pins[name], 'result': self.result_pins[name]}
                                                    for name in self.requests})
            accepted = real['change_pct'] < -real['se_pct']
            self.terminal('ACCEPTED_SELECTION' if accepted else 'REJECTED_SELECTION', accepted=accepted,
                arms=pin(self.output / 'arms/arms.json'), fitted_point=seal['point'],
                frozen_before_check=pin(self.output / 'POINT_FROZEN_BEFORE_CHECK.json'),
                result_index=pin(self.output / 'RESULT_INDEX.json'), sizing=size, independent_check_selection=real,
                standing_checkpoint=seal['checkpoint'] if accepted else self.config['checkpoint'],
                standing_request=point['request'] if accepted else self.config['request_template'],
                standing_head=point['head'] if accepted else self.config['head'], independent_check_exposed=True)
        except BaseException:
            write(self.output / 'FAILED.json', dict(utc=utc(), traceback=traceback.format_exc(), config=self.config_pin,
                check_requests_issued=any(r['rank'] in CHECK for r in self.requests.values()),
                resume_permitted=False, source=pin(__file__)))
            raise
        finally:
            if self.started and not (self.run / 'STOP').exists():
                with (self.run / 'STOP').open('x') as stream:
                    stream.write(utc() + ' EDBAND one-point driver terminal; preserve all artifacts; no automatic restart.\n')


def main():
    if len(sys.argv) == 5 and sys.argv[1] == '--inspect':
        config = json.loads(Path(sys.argv[2]).read_text())
        request_pin = config['request_template'] if sys.argv[3] == 'base' else pin(sys.argv[3])
        request = read_json(request_pin)
        checkpoint = {'path': request['params'], 'sha256': request['edge_band']['checkpoint_sha256']}
        inspect_checkpoint(config, checkpoint, request_pin, sys.argv[4], sys.argv[3] == 'base')
    elif len(sys.argv) == 5 and sys.argv[1] == '--gradients':
        inspect_gradients(json.loads(Path(sys.argv[2]).read_text()), sys.argv[3], sys.argv[4])
    else:
        require(len(sys.argv) == 3, 'usage: edband_six_driver_v6.py CONFIG.json NEW_OUTPUT_DIRECTORY')
        require(Path.cwd().resolve() == HERE, 'Run from trainer directory')
        Round(sys.argv[1], sys.argv[2]).execute()


if __name__ == '__main__':
    main()
