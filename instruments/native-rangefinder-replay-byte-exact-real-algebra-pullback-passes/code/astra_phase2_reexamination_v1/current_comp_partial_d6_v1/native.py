"""Inert COMP/partial-D6 CHECK4 acquisition with retired lexical admission.

No estimator arithmetic lives here. Every borrowed observer, validator, paid
loop and outer-lifecycle edit is inverted to its frozen source before use.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import sys
import textwrap
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
COMP = NEW/'current_p1_node_composition_v1'
RD = NEW/'current_comp_receiver_directions_v1/native.py'
PD = NEW/'current_p1_partial_d6_v1/native.py'
JOINT = NEW/'current_comp_joint_p_node_v1/native.py'
TL = NEW/'current_d50_terminal_linear_v2'
OFFPY = P2/'data_gate/.venv/bin/python'
CENSUS = NEW/'current_d50_response10_census_v1/watch.py'
CENSUS_SHA = '7f868ba0250f142822b2aa36503d1908b21849c47e6e7406320f979e5d93bbcf'
RANKS = VALID = (37, 39, 42, 48)
FIXED = {'terminal': RANKS}
OUTPUT = HERE/'native_terminal_v1'
SCHEMA = NATIVE_SCHEMA = 'current_comp_partial_d6_native_v1'
BUDGET, RSS_LIMIT = 2**41, 8_000_000_000
EXTRA_BINDINGS, PRODUCT_FLOPS = 561056000, 7062260240
SHAPE = (1024, 832, 8192)
EXTRA_LAYOUTS = {'_P1_D6_832_PUSH_SPECS': 278566656, '_P1_D6_832_READ_SPECS': 282489344}
BLAS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
OWNERS = {
    RD: '0f107dcd4a6d27da4208a563f0b79c4dcfb966ab8423df3dad492c799edc4a9e',
    PD: 'fe5a9d8639b2c2cbd5d1d7886d6c82e953f13e20820161aafb885d5133592e1f',
    JOINT: '851813e9679cd58f1b23bbd15de36cdeb866df0e646e1e8cb9e9aa86e924b0bc',
    COMP/'native.py': 'fafc28732701aacfa70ada15d30ed083af76e91747bc01dc840ec8ba14d23d01',
    NEW/'current_d50_response10_candidate_v1/native.py': 'cccd8b50d18e6d093a48a3e3675aa20cbe2ef31cb45c2f167f153ddc842fe09c',
    NEW/'birth_direction_fit_v1/native_v2.py': '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b',
    TL/'native.py': '744f8c46bfdbe0198eb138c1fcacc5a0b90ca6db6eb08ad7d928847c34021abe',
    TL/'common.py': '52467763b2131ad325869ea8e1fffb214ca2a89ab7bd24b3150a3350f5ac3f2a',
    TL/'watch_fit.py': 'b2d3c1334bd642306eea4f0ee12d63fc9e16f8dfc557e36e0db00a8030a6eebd',
    CENSUS: CENSUS_SHA,
    HERE/'DECISION.md': 'b594093e92a190014b534ed6ebc7e0688326ba7d830dac404fefd76d0c43827a',
    HERE/'watch_build.py': 'c1231b73184a7d4a1a40cf50d9b65f7f999847bb57ed6028c68d6a7d5232c44d',
    HERE/'BUILD_WATCH_SPEC.md': 'c471cd2e7e4bf234b64d9fce6aad2528f573630b1cea86133abfdfdb520d4241',
    NEW/'current_comp_joint_p_covariance_v1/watch_build.py': '76eb2ae785c44fcd2d9a98aab97d0107f84766ebc62b5dcb34180f3d8af96095',
    NEW/'current_p1_native_objective_p2_monitor_v2/monitor.py': 'd4b3af53b716593670d8e0c7f0fc6f2133b57f1088d9966c06635d3a23f39873',
    NEW/'current_d50_native_objective_p1_v1/watch.py': '4e8177d6ff943cdee37e48ccca750bf9d552b32cec344aa2b37e6304cbd55fb3',
}
REFERENCES = {'terminal': (COMP/'native_terminal_v2',
    'a9071e5afa6dbf35e6a6871d6997a7dbc8a1105d2bf3e97bd172c7a1b0e0180f',
    '6c050c04fd565d840273f2a917f2a106182cfed1b69f59a09cd5392497706772')}
_BASE, _FUNCTIONS = None, None


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path, tick=None):
    path = Path(path).resolve()
    require(path.is_relative_to(P2) and 'truthpost' not in path.parts and not
        (path.suffix in ('.npy', '.npz') and 'truth' in path.name.lower()), 'No target-array access')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
            if tick:
                tick()
    return h.hexdigest()


def read(path, bound=None):
    path = Path(path)
    require(not path.is_symlink() and (bound is None or path.stat().st_size < bound), 'Bounded regular metadata')
    return json.loads(path.read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def bind(items, tick=None):
    require(isinstance(items, dict), 'Flat immutable hash map')
    for path, digest in items.items():
        require(Path(path).is_absolute() and re.fullmatch('[0-9a-f]{64}', digest or '')
            and sha(path, tick) == digest, 'Immutable bound dependency: '+str(path))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def load(path, name, digest):
    require(sha(path) == digest, 'Reviewed source owner')
    return module(path, name)


def base():
    global _BASE
    if _BASE is None:
        _BASE = load(COMP/'native.py', 'comp_d6_frozen_metadata_owner', OWNERS[COMP/'native.py'])
    return _BASE


def extract(path, name, nested=False):
    path = Path(path)
    require(sha(path) == OWNERS[path], 'Frozen extraction owner')
    text = path.read_text()
    tree = ast.parse(text)
    nodes = ast.walk(tree) if nested else tree.body
    found = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name]
    require(len(found) == 1, 'Unique source owner: '+name)
    n = found[0]
    return textwrap.dedent(''.join(text.splitlines(keepends=True)[n.lineno-1:n.end_lineno]))


def splice(source, changes):
    changed = source
    for before, after in changes:
        require(changed.count(before) == 1, 'Unique source adapter span: '+before[:80])
        changed = changed.replace(before, after)
    inverse = changed
    for before, after in reversed(changes):
        require(inverse.count(after) == 1, 'Unique inverse span: '+after[:80])
        inverse = inverse.replace(after, before)
    require(inverse == source, 'Byte-exact full source inverse')
    return changed


def environment():
    require(Path(sys.executable).resolve() == OFFPY.resolve() and sys.flags.optimize == 0
        and sys.dont_write_bytecode and os.getpriority(os.PRIO_PROCESS, 0) >= 10
        and all(os.environ.get(k) == '1' for k in BLAS)
        and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
        and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0', 'OFFPY/nice10/BLAS1/assertions/no MPS fallback')
    require(not any(k in sys.modules for k in ('numpy', 'torch', 'flopscope', 'threadpoolctl')),
        'All admission precedes numerical imports')


def admitted_inputs(n, refs, sources, inputs):
    """Only metadata and hashes; the future paid loop still owns every input copy."""
    require([refs[r]['call'] for r in RANKS] == [0, 1, 2, 3], 'Same cold37/warm39/42/48 reference frame')
    old = n.legacy()
    sources.update({str(p): h for p, h in old.PINS.items()})
    inputs[str(old.STORE/'manifest.json')] = n.STORE_SHA
    require(sha(old.STORE/'manifest.json') == n.STORE_SHA, 'Canonical official input manifest')
    rows = read(old.STORE/'manifest.json')['roles']['observer_fit']['row_manifests']
    for rank in RANKS:
        matches = [x for x in rows if x['role_rank'] == rank]
        require(len(matches) == 1, 'Unique canonical rank')
        item = matches[0]
        rp = old.STORE/item['path']
        require(sha(rp) == item['sha256'], 'Canonical row metadata')
        row, wp = read(rp), rp.parent/'weights.npy'
        require(row['files']['weights.npy']['sha256'] == refs[rank]['weight_file_sha256'], 'Exact COMP official weight binding')
        inputs.update({str(rp): item['sha256'], str(wp): refs[rank]['weight_file_sha256']})


def build_watch_admission(args, sources, inputs):
    folder, built = HERE/'build_watch_v1', HERE/'build_v1'
    require(sha(folder/'result.json') == args.build_watch_sha256, 'Explicit completed build-watch result')
    record, end, final = (read(folder/name, 4_000_000) for name in ('result.json', 'completion.json', 'finalization.json'))
    require(record['schema'] == 'current_comp_partial_d6_build_watch_v1' and record['role'] == 'terminal'
        and all(record.get(k) is True for k in ('passed', 'completed', 'sources_unchanged', 'whole_owned_tree_retired', 'cleanup_passed'))
        and all(end[k] is True and final[k] is True for k in ('passed', 'completed'))
        and end['result_sha256'] == final['result_sha256'] == args.build_watch_sha256
        and final['completion_sha256'] == sha(folder/'completion.json')
        and record['child_result_sha256'] == args.build_sha256
        and record['child_completion_sha256'] == sha(built/'completion.json')
        and record['child_finalization_sha256'] == sha(built/'finalization.json')
        and 0 <= record['elapsed_s'] <= final['elapsed_s'] < 60
        and 0 < record['maximum_combined_rss_bytes'] < 2_000_000_000
        and 0 < record['maximum_interval_s'] <= 1 and record['observation_samples'] > 0
        and record['numerical_imports'] is False and record['full_model_calls'] == 0,
        'Completed bounded source-build process and three external seals')
    required = {str(p): h for p, h in OWNERS.items() if p in (
        HERE/'watch_build.py', HERE/'BUILD_WATCH_SPEC.md',
        NEW/'current_comp_joint_p_covariance_v1/watch_build.py',
        NEW/'current_p1_native_objective_p2_monitor_v2/monitor.py',
        NEW/'current_d50_native_objective_p1_v1/watch.py')}
    required.update({str(HERE/'build.py'): args.builder_sha256, str(HERE/'BUILD_PROTOCOL.md'): args.build_protocol_sha256})
    require(record['source_bindings'] == required, 'Same exact reviewed source-build supervisor and builder')
    bind(required); sources.update(required)
    samples_path = folder/'samples.jsonl'
    require(samples_path.stat().st_size < 4_000_000 and not samples_path.is_symlink(), 'Bounded source-build sample history')
    samples = [json.loads(line) for line in samples_path.read_text().splitlines()]
    require(len(samples) == record['observation_samples'] and type(record['initial_swap_bytes']) in (int, float)
        and math.isfinite(record['initial_swap_bytes']) and record['initial_swap_bytes'] >= 0
        and all(0 <= s['swap_bytes'] <= record['initial_swap_bytes']
            and 0 < s['combined_rss_bytes'] < 2_000_000_000 and 0 <= s['elapsed_s'] < 60 for s in samples)
        and max(s['combined_rss_bytes'] for s in samples) == record['maximum_combined_rss_bytes'],
        'Zero added sampled build swap and complete recorded resource samples')
    for name in ('result.json', 'completion.json', 'finalization.json', 'samples.jsonl', 'started.json'):
        inputs[str(folder/name)] = sha(folder/name)


def observer_sources():
    original_domain = extract(TL/'native.py', 'Observer')
    begin = original_domain[original_domain.index('    def __init__'):original_domain.index('    @staticmethod')]
    unused = original_domain[original_domain.index('        elif code is self.gnn_code'):original_domain.index('\n    def export_bytes')]
    export = original_domain[original_domain.index('    def export_bytes'):original_domain.index('\n    def clear')]
    clear = original_domain[original_domain.index('    def clear'):]
    domain = splice(original_domain, [
        ('class Observer:', 'class DomainObserver:'),
        (begin, "    def __init__(self, module):\n        self.gaussian_code = module.gaussian_raw.__code__\n"
         "        self.select_code = module.select.__code__\n        self.variances, self.selectors = [], []\n\n"),
        (unused, '        # No head, parameter, or raw-mean capture in this endpoint.\n'),
        (export, "    def export_bytes(self):\n        c.require(len(self.variances) == 16 and len(self.selectors) == 47,\n"
         "                  'Complete unchanged domain/selector packet')\n"
         "        packet = dict(variances=tuple(self.variances),selectors=tuple(self.selectors))\n"
         "        self.clear()\n        return packet\n"),
        (clear, '    def clear(self):\n        self.variances.clear()\n        self.selectors.clear()\n')])
    original_comp = extract(COMP/'native.py', 'Observer')
    comp = splice(original_comp, [
        ('class Observer:', 'class CompObserver:'),
        ('        self.small = SMALL.Observer(self.module)\n', '        self.small = Domain(self.module)\n')])
    original_extra = extract(PD, 'Observer', nested=True)
    extra = splice(original_extra, [
        ('class Observer(r.Observer):', 'class Observer(CompObserver):'),
        ('        self.small = Domain(self.module)\n', '        # Domain-only capture is initialized by the COMP superclass.\n'),
        ('len(self.interested) == 9', 'len(self.interested) == 10'),
        ("'Exact nine observed code owners'", "'Exact ten COMP plus partial-D6 observed code owners'")])
    return (domain, comp, extra), dict(
        domain_original_sha256=hashlib.sha256(original_domain.encode()).hexdigest(),
        domain_derived_sha256=hashlib.sha256(domain.encode()).hexdigest(),
        comp_observer_original_sha256=hashlib.sha256(original_comp.encode()).hexdigest(),
        comp_observer_derived_sha256=hashlib.sha256(comp.encode()).hexdigest(),
        extra_observer_original_sha256=hashlib.sha256(original_extra.encode()).hexdigest(),
        extra_observer_derived_sha256=hashlib.sha256(extra.encode()).hexdigest(),
        all_three_observer_inverses_verified=True, ten_filtered_code_identities=True,
        RC_startup_observer_not_used=True, COMP_nested_cstar_and_NODE_A_accounting_retained=True,
        no_added_participant_array_operations=True)


def metadata_sources():
    original = {name: extract(PD, name) for name in
        ('validate_diagnostic', 'after_bill', 'finish_native', 'verify_acquisition')}
    old_prefix = """    require(startup['prefix']['P_empty_at_entry'] is cold and startup['prefix']['P_ready_at_binary_entry']
            and len(startup['coefficients']) == 1 and startup['coefficients'][0]['empty_at_entry'] is cold
            and startup['coefficients'][0]['flops'] == (120 if cold else 0)
            and startup['extra_empty_before'] is cold and startup['extra_ready_after'], 'P/cstar/both serial bindings initialized once')
"""
    new_prefix = """    require(all(startup['prefix'][k] is cold for k in ('P_empty_at_entry','cstar_empty_at_entry','A_empty_at_entry'))
            and all(startup['prefix'][k] for k in ('P_ready_at_binary_entry','cstar_ready_at_binary_entry','A_ready_at_binary_entry'))
            and startup['extra_empty_before'] is cold and startup['extra_ready_after'], 'COMP P/cstar/A and serial bindings initialized once')
    materials = startup['materializations']
    coeff = [x for x in materials if x['name'] == '_bd_coefficients']
    node = [x for x in materials if x['name'] == '_bdnh_parameters']
    require(len(materials) == 3 and len(coeff) == 2 and len(node) == 1
            and [x['inside_prefix'] for x in coeff] == [True,False]
            and [x['empty_at_entry'] for x in coeff] == [cold,False]
            and [x['literal_elements'] for x in coeff] == [60,60]
            and [x['flops'] for x in coeff] == [120 if cold else 0,0]
            and node[0]['inside_prefix'] and node[0]['empty_at_entry'] is cold
            and node[0]['literal_elements'] == 256 and node[0]['flops'] == (512 if cold else 0)
            and startup['NODE_A_materialization_flops'] == node[0]['flops'], 'Both cstar calls and sole paid NODE A, actual prefix ownership')
"""
    validate = splice(original['validate_diagnostic'], [
        (old_prefix, new_prefix),
        ("sum(x['flops'] for x in bindings+startup['coefficients'])", "sum(x['flops'] for x in bindings)+startup['outside_prefix_materialization_flops']"),
        ("    bindings = startup['bindings']; extra = [x for x in bindings if x['name'] in EXTRA_LAYOUTS]\n",
         "    require(startup['outside_prefix_materialization_flops'] == sum(x['flops'] for x in materials if not x['inside_prefix']),\n"
         "            'Only nonnested materialization costs supplement prefix')\n"
         "    bindings = startup['bindings']; extra = [x for x in bindings if x['name'] in EXTRA_LAYOUTS]\n")])
    after = splice(original['after_bill'], [("'Exact P1 official inputs'", "'Exact COMP official inputs'")])
    finish = splice(original['finish_native'], [
        ("len(result['diagnostic_records']) == 41", "len(result['diagnostic_records']) == 5"),
        ("'Complete40 primary calls and repeat37'", "'Complete four primaries and repeat37'"),
        ('records[:40]', 'records[:4]'),
        ('        all16_domains_qualified=True,all47_selectors_qualified=True,repeat_output_domain_byte_exact=True,',
         '        full_model_calls=5,retired_call_count=5,COMP_parameter_initialization_owned=True,\n'
         '        all16_domains_qualified=True,all47_selectors_qualified=True,repeat_output_domain_byte_exact=True,'),
        ('            nested_extra_binding_regions_not_double_counted=True),',
         '            nested_extra_binding_regions_not_double_counted=True,\n'
         '            nested_prefix_materializations_not_double_counted=True),')])
    verify = splice(original['verify_acquisition'], [
        ('record[\'candidate_sha256\'] == CANDIDATE_SHA', "record['candidate_sha256'] == context['candidate_sha256']"),
        ("record['build_sha256'] == BUILD_SHA and record['builder_sha256'] == BUILDER_SHA",
         "record['build_sha256'] == context['build_sha256'] and record['builder_sha256'] == context['builder_sha256']\n"
         "            and record['build_watch_sha256'] == context['build_watch_sha256']"),
        ("            and record['admission_watch_sha256'] == args.admission_watch_sha256\n",
         "            # This endpoint has a separately retired three-seal admission, not the old PD watch handoff.\n"),
        ("            and record['prerequisites'] == context['prerequisites'] and record['proof'] == context['proof'], 'Completed fixed native endpoint')",
         "            and record['full_model_calls'] == record['retired_call_count'] == 5\n"
         "            and record['COMP_parameter_initialization_owned'] and record['proof'] == context['proof'], 'Completed fixed native endpoint')"),
        ("len(record['diagnostic_records']) == 41", "len(record['diagnostic_records']) == 5"),
        ("'Exact41 own prediction/input/diagnostic records'", "'Exact five own prediction/input/diagnostic records'"),
        ("    refs = context['reference_by_rank']\n", "    refs = {int(k):v for k,v in context['reference_by_rank'].items()}\n"),
        ("len(paths) == 246, 'Only original205 artifacts plus41 small diagnostics'",
         "len(paths) == 30, 'Only original25 artifacts plus five small diagnostics'"),
        ("'partial_d6_lifecycle_verifier'", "'comp_partial_d6_lifecycle_verifier'")])
    changed = dict(validate_diagnostic=validate, after_bill=after, finish_native=finish, verify_acquisition=verify)
    proof = {k: dict(original_sha256=hashlib.sha256(original[k].encode()).hexdigest(),
        derived_sha256=hashlib.sha256(v.encode()).hexdigest(), inverse_verified=True) for k, v in changed.items()}
    return '\n'.join(changed.values()), proof


def callbacks():
    global _FUNCTIONS
    if _FUNCTIONS is None:
        source, _ = metadata_sources()
        env = dict(globals())
        exec(compile(source, str(__file__)+':source_inverse_metadata', 'exec'), env)
        _FUNCTIONS = env
    return _FUNCTIONS


def after_bill(np, observer, trial, result, folder, refs):
    return callbacks()['after_bill'](np, observer, trial, result, folder, refs)


def verify_acquisition(record, end, digest, args, context, tick=None):
    return callbacks()['verify_acquisition'](record, end, digest, args, context, tick)


def prepare(n):
    """Compile source text only; no estimator import, arrays, contexts or execution."""
    global _BASE
    _BASE = n
    if n.SMALL is None:
        n.load_diagnostics()
    owner = load(JOINT, 'comp_d6_original_paid_source_owner', OWNERS[JOINT])
    _, _, paid, paid_proof = owner.prepare(n)
    (domain, comp, extra), proof = observer_sources()
    env = dict(vars(n.SMALL))
    exec(compile(domain, str(__file__)+':domain_only', 'exec'), env)
    Domain = env['DomainObserver']
    env = dict(vars(n), Domain=Domain, EXTRA_LAYOUTS=EXTRA_LAYOUTS,
        EXTRA_BINDINGS=EXTRA_BINDINGS, PRODUCT_FLOPS=PRODUCT_FLOPS, SHAPE=SHAPE)
    exec(compile(comp+'\n'+extra, str(__file__)+':COMP_D6_observer', 'exec'), env)
    _, metadata_proof = metadata_sources()
    _, lifecycle_proof = lifecycle_source()
    proof.update(paid_loop=paid_proof, metadata_functions=metadata_proof,
        outer_lifecycle=lifecycle_proof, original_paid_arithmetic_unchanged=True,
        candidate_source_changes_owned_only_by_builder=True)
    return env['Observer'], callbacks()['finish_native'], paid, proof


def lifecycle_source():
    """Keep RD's retired-admission/main lifecycle; specialize only metadata/observer plumbing."""
    original_admit, original_run = extract(RD, 'admit'), extract(RD, 'run')
    no_zero = """        if args.role == 'terminal':
            prior_zero(args, inputs)
        else:
            require(not args.zero_result_sha256 and not args.zero_watch_sha256, 'No future zero dependency')
"""
    admitted = splice(original_admit, [
        ('    began = time.monotonic()\n', '    began = time.monotonic()\n    environment()\n'),
        ("HERE/'build_v2.py': args.builder_sha256, HERE/'BUILD_PROTOCOL_V2.md': args.build_protocol_sha256",
         "HERE/'build.py': args.builder_sha256, HERE/'BUILD_PROTOCOL.md': args.build_protocol_sha256"),
        ("module(HERE/'build_v2.py', 'receiver_native_builder_admission')",
         "module(HERE/'build.py', 'comp_partial_d6_builder_admission')"),
        ('builder.verify_build(args.role, args.build_sha256, args.candidate_sha256,',
         'builder.verify_build(args.build_sha256, args.candidate_sha256,'),
        ("        inputs = dict(context['input_bindings'])\n",
         "        inputs = dict(context['input_bindings'])\n        build_watch_admission(args, sources, inputs)\n"),
        (no_zero, '        admitted_inputs(n, refs, sources, inputs)\n'),
        ("candidate = HERE/f'build_{args.role}_v1/estimator.py'", "candidate = HERE/'build_v1/estimator.py'"),
        ('candidate_sha256=args.candidate_sha256, build_sha256=args.build_sha256,',
         'candidate_sha256=args.candidate_sha256, build_sha256=args.build_sha256,\n'
         '            builder_sha256=args.builder_sha256, build_protocol_sha256=args.build_protocol_sha256,\n'
         '            build_watch_sha256=args.build_watch_sha256,'),
        ('    elapsed = time.monotonic()-began; ok = bool(result[\'passed\'] and elapsed < 60)\n',
         "    final_result_sha, final_completion_sha = sha(folder/'result.json'), sha(folder/'completion.json')\n"
         "    elapsed = time.monotonic()-began; ok = bool(result['passed'] and elapsed < 60)\n"),
        ("write(folder/'finalization.json', dict(passed=ok, completed=ok, result_sha256=sha(folder/'result.json'),\n"
         "        completion_sha256=sha(folder/'completion.json'), elapsed_s=elapsed))",
         "write(folder/'finalization.json', dict(passed=ok, completed=ok, result_sha256=final_result_sha,\n"
         "        completion_sha256=final_completion_sha, elapsed_s=elapsed))")])
    old_after = textwrap.indent(extract(RD, 'after', nested=True), '    ')
    old_done = textwrap.indent(extract(RD, 'done', nested=True), '    ')
    running = splice(original_run, [
        ("    began = time.monotonic(); folder = HERE/f'native_{args.role}_v1'\n",
         "    began = time.monotonic(); environment(); folder = HERE/f'native_{args.role}_v1'\n"),
        ("context = read(ad/'context.json');", "context = read(ad/'context.json', 4_000_000);"),
        ("HERE/'model.py': args.model_sha256", "HERE/'build.py': args.builder_sha256"),
        ("'Same exact reviewed source/model/watch owners'", "'Same exact reviewed source/builder/watch owners'"),
        ("n = module(COMP/'native.py', 'receiver_actual_native_owner'); n.load_diagnostics()",
         'n = base(); n.load_diagnostics()'),
        (old_after, '    def after(np, observed, trial, result, out):\n'
         '        after_bill(np, observed, trial, result, out, refs)\n'),
        (old_done, '    def done(result):\n        finish(result)\n'),
        ("build_sha256=context['build_sha256'], versions=versions,",
         "build_sha256=context['build_sha256'], builder_sha256=context['builder_sha256'],\n"
         "        build_watch_sha256=context['build_watch_sha256'], versions=versions,"),
        ("record = read(folder/'result.json'); elapsed = time.monotonic()-began\n",
         "record = read(folder/'result.json')\n"
         "    final_result_sha, final_completion_sha = sha(folder/'result.json'), sha(folder/'completion.json')\n"
         "    elapsed = time.monotonic()-began\n"),
        ("write(folder/'finalization.json', dict(passed=passed, completed=passed, result_sha256=sha(folder/'result.json'),\n"
         "        completion_sha256=sha(folder/'completion.json'), elapsed_s=elapsed))",
         "write(folder/'finalization.json', dict(passed=passed, completed=passed, result_sha256=final_result_sha,\n"
         "        completion_sha256=final_completion_sha, elapsed_s=elapsed))")])
    source = admitted+'\n'+running
    proof = dict(parent_source_sha256=OWNERS[RD], admission_inverse_verified=True, run_inverse_verified=True,
        original_admit_sha256=hashlib.sha256(original_admit.encode()).hexdigest(),
        original_run_sha256=hashlib.sha256(original_run.encode()).hexdigest(),
        derived_sha256=hashlib.sha256(source.encode()).hexdigest(), no_model_or_zero_prerequisite=True)
    return source, proof


def invoke(name, args):
    source, _ = lifecycle_source()
    env = dict(globals())
    exec(compile(source, str(__file__)+':retired_admission_lifecycle', 'exec'), env)
    return env[name](args)


def admit(args):
    return invoke('admit', args)


def run(args):
    return invoke('run', args)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--admit', action='store_true')
    parser.add_argument('--role', choices=('terminal',))
    for name in ('source', 'builder', 'protocol', 'watcher', 'context', 'build-protocol', 'build', 'candidate', 'build-watch'):
        parser.add_argument('--'+name+'-sha256')
    args = parser.parse_args()
    if not args.run and not args.admit:
        parser.print_help()
        return None
    require(args.run != args.admit and args.role == 'terminal', 'One explicit terminal admission or worker stage')
    keys = ('source', 'builder', 'protocol', 'watcher')+(
        ('build-protocol', 'build', 'candidate', 'build-watch') if args.admit else ('context',))
    require(all(re.fullmatch('[0-9a-f]{64}', getattr(args, k.replace('-', '_')+'_sha256') or '') for k in keys),
        'Explicit reviewed source and stage artifact hashes')
    require((args.context_sha256 is None) is args.admit, 'Only worker consumes retired context')
    return args


if __name__ == '__main__':
    _args = arguments()
    if _args is not None:
        raise SystemExit(admit(_args) if _args.admit else run(_args))
