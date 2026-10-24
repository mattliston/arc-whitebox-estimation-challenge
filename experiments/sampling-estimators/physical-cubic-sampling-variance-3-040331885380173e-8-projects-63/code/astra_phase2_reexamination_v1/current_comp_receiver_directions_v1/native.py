"""Source-admitted receiver native null/terminal acquisition; no truth access."""
import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
COMP = NEW/'current_p1_node_composition_v1'
JOINT = NEW/'current_comp_joint_p_node_v1'
FIXED = {'zero': (35, 100), 'terminal': (37, 39, 42, 48)}
SCHEMA = 'current_comp_receiver_directions_native_v1'
OWNERS = {
    JOINT/'native.py': '851813e9679cd58f1b23bbd15de36cdeb866df0e646e1e8cb9e9aa86e924b0bc',
    JOINT/'watch_native.py': 'd35ba7cbe406ae5e6200f8778b7a582bc6ea17336be8086b5e813e5a25dbe65e',
    COMP/'native.py': 'fafc28732701aacfa70ada15d30ed083af76e91747bc01dc840ec8ba14d23d01',
    NEW/'current_d50_response10_candidate_v1/native.py': 'cccd8b50d18e6d093a48a3e3675aa20cbe2ef31cb45c2f167f153ddc842fe09c',
    NEW/'birth_direction_fit_v1/native_v2.py': '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b',
    NEW/'current_d50_terminal_linear_v2/native.py': '744f8c46bfdbe0198eb138c1fcacc5a0b90ca6db6eb08ad7d928847c34021abe',
    NEW/'current_d50_terminal_linear_v2/common.py': '52467763b2131ad325869ea8e1fffb214ca2a89ab7bd24b3150a3350f5ac3f2a',
}
REFERENCES = {
    'zero': (COMP/'panel_v2/terminal',
        'de766126360ce24263f548c27a2cb51f60e52107386c4aeb7cf15f9ea20db625',
        '0973d38be5b37c2ff99d5b6243119a22feb5fe591c851ed4aac0aae7346a9073'),
    'terminal': (COMP/'native_terminal_v2',
        'a9071e5afa6dbf35e6a6871d6997a7dbc8a1105d2bf3e97bd172c7a1b0e0180f',
        '6c050c04fd565d840273f2a917f2a106182cfed1b69f59a09cd5392497706772'),
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    path = Path(path).resolve()
    require(path.is_relative_to(P2) and 'truthpost' not in str(path), 'No target-file access')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def prepare(n):
    """Only observational R ownership is added; the original paid loop is retained."""
    require(sha(JOINT/'native.py') == OWNERS[JOINT/'native.py'], 'Frozen exact paid-loop adapter')
    owner = module(JOINT/'native.py', 'receiver_native_joint_source_owner')
    _, _, paid, paid_proof = owner.prepare(n)
    original = owner.extract(COMP/'native.py', 'Observer')
    changes = (
        ("'_bd_coefficients','_bdnh_parameters')", "'_bd_coefficients','_bdnh_parameters','_bdr_parameters')"),
        ('        self.begin()\n', '        self.begin()\n'
         '        self.interested_code_ids = frozenset(id(x) for x in self.codes.values()) | frozenset(\n'
         "            id(getattr(self.small,k)) for k in ('predict_code','gaussian_code','select_code','gnn_code'))\n"
         "        require(len(self.interested_code_ids) == 9, 'Eight original plus one R getter code identity')\n"),
        ('        self.small(frame,event,answer)\n',
         "        if event not in ('call','return') or id(frame.f_code) not in self.interested_code_ids:\n"
         '            return\n        self.small(frame,event,answer)\n'),
        ('A_empty_at_entry=self.module._BDNH_PARAMETERS is None)',
         'A_empty_at_entry=self.module._BDNH_PARAMETERS is None,\n'
         '                R_empty_at_entry=self.module._BDR_PARAMETERS is None)'),
        ('A_ready_at_binary_entry=self.module._BDNH_PARAMETERS is not None)',
         'A_ready_at_binary_entry=self.module._BDNH_PARAMETERS is not None,\n'
         '                R_ready_at_binary_entry=self.module._BDR_PARAMETERS is not None)'),
        ("('_bdnh_parameters','_BDNH_PARAMETERS','_BDNH_A_LITERAL')):",
         "('_bdnh_parameters','_BDNH_PARAMETERS','_BDNH_A_LITERAL'),\n"
         "                                        ('_bdr_parameters','_BDR_PARAMETERS','_BDR_LITERAL')):"),
        ("'P_ready_at_binary_entry','cstar_ready_at_binary_entry','A_ready_at_binary_entry'))",
         "'P_ready_at_binary_entry','cstar_ready_at_binary_entry','A_ready_at_binary_entry','R_ready_at_binary_entry'))"),
        ("        outside = sum(x['flops'] for x in self.materializations if not x['inside_prefix'])\n",
         "        receiver = [x for x in self.materializations if x['name'] == '_bdr_parameters']\n"
         "        require(len(receiver) == 1 and receiver[0]['inside_prefix'] and receiver[0]['literal_elements'] == 256,\n"
         "                'Sole actual256-value receiver getter inside original prefix')\n"
         "        outside = sum(x['flops'] for x in self.materializations if not x['inside_prefix'])\n"),
        ("prefix_scope='P, first cstar, NODE A, and original zero-state allocation; nested materializations not added twice',",
         "prefix_scope='P, first cstar, NODE A, receiver R, original zero-state; nested getters not added twice',\n"
         "            receiver_R_materialization_flops=receiver[0]['flops'],"),
    )
    observer = owner.splice(original, changes)
    env = dict(vars(n), FIXED=FIXED)
    exec(compile(observer, str(__file__)+':receiver_ownership_observer', 'exec'), env)
    finish_source = owner.extract(COMP/'native.py', 'finish_native')
    finish_changed = owner.splice(finish_source, [
        ("zero_all16_P1_byte_exact=role == 'zero')", "zero_all16_COMP_byte_exact=role == 'zero', receiver_R_initialization_owned=True)")])
    exec(compile(finish_changed, str(__file__)+':original_finish_plus_R', 'exec'), env)
    proof = dict(paid_loop=paid_proof, receiver_observer_inverse_verified=True,
        observer_original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        observer_modified_sha256=hashlib.sha256(observer.encode()).hexdigest(),
        nine_filtered_code_identities=True, original_paid_arithmetic_unchanged=True,
        one_additional_getter_observed=True, getter_inside_existing_prefix_not_double_counted=True)
    return env['Observer'], env['finish_native'], paid, proof


def prior_zero(args, inputs):
    require(args.zero_result_sha256 and args.zero_watch_sha256, 'Terminal requires its newly qualified native zero')
    packets = []
    for folder, digest in ((HERE/'native_zero_v1', args.zero_result_sha256),
                           (HERE/'native_zero_watch_v1', args.zero_watch_sha256)):
        rp, ep, fp = (folder/x for x in ('result.json', 'completion.json', 'finalization.json'))
        require(sha(rp) == digest, 'Reviewed zero result')
        r, e, f = read(rp), read(ep), read(fp)
        require(r['passed'] and e['passed'] and f['passed'] and e['result_sha256'] == digest
            and f['completion_sha256'] == sha(ep), 'Complete zero three-seal chain')
        for group in ('source_bindings', 'input_bindings', 'artifacts'):
            for path, value in r.get(group, {}).items():
                require(sha(path) == value, 'Frozen native zero dependency'); inputs[path] = value
        for path in (rp, ep, fp):
            inputs[str(path)] = sha(path)
        packets.append((r, sha(ep), sha(fp)))
    (z, ze, zf), (w, _, _) = packets
    require(z['schema'] == SCHEMA and z['role'] == 'zero' and z['own_complete_bills']
        and z['all16_domains_qualified'] and z['all47_selectors_qualified']
        and z['zero_all16_COMP_byte_exact'] and z['zero_all47_COMP_byte_exact']
        and z['receiver_R_initialization_owned'] and z['repeat_output_domain_byte_exact']
        and z['full_model_calls'] == z['retired_call_count'] == 3
        and z['source_bindings'][str(HERE/'native.py')] == args.source_sha256
        and z['source_bindings'][str(HERE/'watch_native.py')] == args.watcher_sha256
        and w['retired_calls'] == 3 and w['whole_owned_tree_retired'] and w['cleanup_passed']
        and w['child_result_sha256'] == args.zero_result_sha256
        and w['child_completion_sha256'] == ze and w['child_finalization_sha256'] == zf
        and w['maximum_combined_rss_bytes'] < 8_000_000_000 and w['maximum_interval_s'] <= 1,
        'Same-source complete zero/reference/resource qualification')


def admit(args):
    began = time.monotonic()
    def guard(*_):
        require(time.monotonic()-began < 60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2_000_000_000,
                '60s/2GB source-only admission')
    signal.signal(signal.SIGALRM, guard); signal.setitimer(signal.ITIMER_REAL, .5, .5)
    folder = HERE/f'native_{args.role}_admission_v1'; folder.mkdir(exist_ok=False)
    result = dict(passed=False, completed=False, role=args.role, admission_pid=os.getpid(), numerical_imports=False, truth_accessed=False)
    try:
        require(os.getpriority(os.PRIO_PROCESS, 0) >= 10 and sys.flags.optimize == 0, 'Nice10, checks enabled')
        owners = {**OWNERS, Path(__file__).resolve(): args.source_sha256, HERE/'NATIVE_PROTOCOL.md': args.protocol_sha256,
            HERE/'watch_native.py': args.watcher_sha256, HERE/'NATIVE_WATCH_SPEC.md': sha(HERE/'NATIVE_WATCH_SPEC.md'),
            HERE/'build_v2.py': args.builder_sha256, HERE/'BUILD_PROTOCOL_V2.md': args.build_protocol_sha256}
        require(all(sha(p) == h for p, h in owners.items()), 'Reviewed native/build sources')
        builder = module(HERE/'build_v2.py', 'receiver_native_builder_admission')
        context = builder.verify_build(args.role, args.build_sha256, args.candidate_sha256,
                                       args.builder_sha256, args.build_protocol_sha256)
        n = module(COMP/'native.py', 'receiver_original_native_admission')
        _, _, _, proof = prepare(n)
        sources = {str(p.resolve()): h for p, h in owners.items()}; sources.update(context['source_bindings'])
        inputs = dict(context['input_bindings'])
        ref_folder, digest, end_digest = REFERENCES[args.role]
        require(sha(ref_folder/'result.json') == digest and sha(ref_folder/'completion.json') == end_digest, 'Exact COMP reference seals')
        baseline, end = read(ref_folder/'result.json'), read(ref_folder/'completion.json')
        require(baseline['passed'] and baseline['completed'] and baseline['own_complete_bills']
            and baseline['all16_domains_qualified'] and baseline['all47_selectors_qualified']
            and baseline['candidate_sha256'] == '2e13d40a21e62691155918719f8ec088f581d495ae17800c5503c3b56504918a'
            and end['passed'] and end['result_sha256'] == digest, 'Exact qualified current COMP comparator')
        inputs[str(ref_folder/'result.json')] = digest; inputs[str(ref_folder/'completion.json')] = end_digest
        refs = {}
        for trial, prepared, dm in zip(baseline['trials'], baseline['input_preparation'], baseline['diagnostic_records']):
            if trial['rank'] in FIXED[args.role] and trial['rank'] not in refs:
                require(trial['rank'] == prepared['rank'] and trial['call'] == prepared['call'], 'Input/prediction ownership')
                require(sha(trial['output']) == trial['output_sha256'] == baseline['artifacts'][trial['output']]
                    and sha(dm['path']) == dm['sha256'] == baseline['artifacts'][dm['path']], 'Frozen baseline output/domain bytes')
                refs[trial['rank']] = dict(trial, weight_file_sha256=prepared['weight_file_sha256'], diagnostic=dm)
                inputs[trial['output']] = trial['output_sha256']; inputs[dm['path']] = dm['sha256']
        require(set(refs) == set(FIXED[args.role]), 'Every fixed reference row')
        if args.role == 'terminal':
            prior_zero(args, inputs)
        else:
            require(not args.zero_result_sha256 and not args.zero_watch_sha256, 'No future zero dependency')
        candidate = HERE/f'build_{args.role}_v1/estimator.py'
        require(sha(candidate) == args.candidate_sha256, 'Exact candidate bytes')
        sources[str(candidate)] = args.candidate_sha256
        document = dict(role=args.role, ranks=list(FIXED[args.role]), candidate=str(candidate),
            candidate_sha256=args.candidate_sha256, build_sha256=args.build_sha256,
            source_bindings=sources, input_bindings=inputs, reference_by_rank=refs, proof=proof,
            admission_pid=os.getpid(), source_admission_complete=True, truth_accessed=False, numerical_imports=False)
        require(len(json.dumps(document).encode()) < 4_000_000, 'Bounded metadata context')
        require(not any(k in sys.modules for k in ('numpy', 'torch', 'flopscope', 'threadpoolctl')), 'Stdlib-only admission')
        require(all(sha(p) == h for p, h in {**sources, **inputs}.items()), 'Unchanged admitted bytes'); guard()
        write(folder/'context.json', document)
        result.update(passed=True, completed=True, context_sha256=sha(folder/'context.json'), sources_unchanged=True)
    except BaseException:
        result['error'] = traceback.format_exc()
    result.update(elapsed_s=time.monotonic()-began, peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    if result['elapsed_s'] >= 60 or result['peak_rss_bytes'] >= 2_000_000_000:
        result.update(passed=False, completed=False)
    write(folder/'result.json', result)
    write(folder/'completion.json', dict(passed=result['passed'], completed=result['completed'], result_sha256=sha(folder/'result.json')))
    elapsed = time.monotonic()-began; ok = bool(result['passed'] and elapsed < 60)
    write(folder/'finalization.json', dict(passed=ok, completed=ok, result_sha256=sha(folder/'result.json'),
        completion_sha256=sha(folder/'completion.json'), elapsed_s=elapsed))
    signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps(result, indent=2)); return 0 if ok else 1


def run(args):
    began = time.monotonic(); folder = HERE/f'native_{args.role}_v1'
    ad = HERE/f'native_{args.role}_admission_v1'
    require(sha(ad/'context.json') == args.context_sha256, 'Reviewed retired admission')
    context = read(ad/'context.json'); ar, ae, af = (read(ad/x) for x in ('result.json', 'completion.json', 'finalization.json'))
    require(ar['passed'] and ar['completed'] and ae['passed'] and ae['result_sha256'] == sha(ad/'result.json')
        and ar['elapsed_s'] < 60 and ar['peak_rss_bytes'] < 2_000_000_000 and af['passed'] and af['completed']
        and af['elapsed_s'] < 60 and af['result_sha256'] == sha(ad/'result.json')
        and af['completion_sha256'] == sha(ad/'completion.json') and ar['context_sha256'] == args.context_sha256
        and context['role'] == args.role and context['source_admission_complete'], 'Complete retired admission seals')
    try:
        os.kill(context['admission_pid'], 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError('Full source-admission process must exit before numerical worker')
    require(all(context['source_bindings'].get(str(p)) == h for p, h in {
        Path(__file__).resolve(): args.source_sha256, HERE/'model.py': args.model_sha256,
        HERE/'NATIVE_PROTOCOL.md': args.protocol_sha256, HERE/'watch_native.py': args.watcher_sha256}.items()),
        'Same exact reviewed source/model/watch owners')
    require(all(sha(p) == h for p, h in {**context['source_bindings'], **context['input_bindings']}.items()), 'Unchanged admitted inventory')
    require(not any(k in sys.modules for k in ('numpy', 'torch', 'flopscope', 'threadpoolctl')), 'Admission before numerical imports')
    versions = {k: importlib.metadata.version(k) for k in ('numpy', 'flopscope', 'whestbench', 'threadpoolctl')}
    require(versions == dict(numpy='2.4.6', flopscope='0.12.0', whestbench='0.16.0', threadpoolctl='3.6.0'), 'Exact official native versions')
    n = module(COMP/'native.py', 'receiver_actual_native_owner'); n.load_diagnostics()
    observer, finish, paid, proof = prepare(n)
    require(proof == context['proof'], 'Identical admitted native source inverse')
    old = n.legacy(); candidate = Path(context['candidate'])
    require(not folder.exists(), 'Exclusive fixed native roster')
    pins = {**context['source_bindings'], **context['input_bindings']}
    pins.update({str(ad/name): sha(ad/name) for name in ('context.json', 'result.json', 'completion.json', 'finalization.json')})
    refs = {int(k): value for k, value in context['reference_by_rank'].items()}
    def after(np, observed, trial, result, out):
        n.after_bill(np, observed, trial, result, out, refs)
        d = read(result['diagnostic_records'][-1]['path']); startup = d['startup']; cold = trial['call'] == 0
        require(startup['prefix']['R_empty_at_entry'] is cold and startup['prefix']['R_ready_at_binary_entry']
            and startup['receiver_R_materialization_flops'] == (512 if cold else 0), 'Actual once-only paid R materialization')
        if args.role == 'zero':
            reference = refs[trial['rank']]['diagnostic']; require(sha(reference['path']) == reference['sha256'], 'Exact zero reference domains')
            require(d['domains'] == read(reference['path'])['domains'], 'All16/47 exact native COMP null domains/selectors')
    def done(result):
        finish(result)
        calls = len(FIXED[args.role])+1
        result.update(full_model_calls=calls, retired_call_count=calls, zero_all47_COMP_byte_exact=args.role == 'zero')
    header = dict(schema=SCHEMA, stage='native', role=args.role, completed=False, sources_unchanged=False,
        build_sha256=context['build_sha256'], versions=versions, truth_accessed=False, no_fit=True, score_measured=False,
        source_inverse_verified=True, paid_arithmetic_and_lifecycle_unchanged=True, own_complete_bills=False,
        full_admission_in_retired_process=True, admission_context_sha256=args.context_sha256, proof=proof,
        response_floor=.01, active_floor=.01, birth_floor=.01)
    namespace = dict(vars(old), __file__=str(Path(__file__).resolve()), PARENT=NEW, CANDIDATE=candidate, NATIVE_PINS=pins,
        FIXED_RANKS=FIXED[args.role], Observer=observer, observed_predict=n.observed_predict,
        after_bill=after, finish_native=done, NATIVE_HEADER=header)
    os.chdir(NEW); saved = sys.argv
    try:
        sys.argv = [str(__file__), '--source', str(candidate), '--split', 'validation', '--out', str(folder)]
        exec(compile(paid, str(__file__)+':unchanged_paid_native', 'exec'), namespace)
        code = namespace['main']()
    finally:
        sys.argv = saved
    record = read(folder/'result.json'); elapsed = time.monotonic()-began
    passed = bool(code == 0 and record['passed'] and record['completed'] and elapsed < 600)
    write(folder/'finalization.json', dict(passed=passed, completed=passed, result_sha256=sha(folder/'result.json'),
        completion_sha256=sha(folder/'completion.json'), elapsed_s=elapsed))
    return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true'); parser.add_argument('--admit', action='store_true')
    parser.add_argument('--role', choices=tuple(FIXED))
    for name in ('source', 'model', 'protocol', 'watcher', 'builder', 'build-protocol', 'build', 'candidate', 'context', 'zero-result', 'zero-watch'):
        parser.add_argument('--'+name+'-sha256')
    args = parser.parse_args()
    if args.run or args.admit:
        require(args.role in FIXED and args.run != args.admit, 'One explicit source-admission or native role')
        raise SystemExit(admit(args) if args.admit else run(args))
    parser.print_help()
