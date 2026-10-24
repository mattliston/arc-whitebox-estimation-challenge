"""One-shot35/100/repeat35 supervisor; no score or automatic next stage.

Exact-inverse adapter to the qualified census watchdog. Its resource,
completed-bill/released-context deadline, signal and owned-group cleanup
logic are unchanged. Inert without --run; no numerical modules imported.
"""
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
BASE = NEW/'current_d50_response10_census_v1/watch.py'
BASE_SHA = '7f868ba0250f142822b2aa36503d1908b21849c47e6e7406320f979e5d93bbcf'
PROTOCOL_SHA = 'c931293ede873bf193dec3229b7a02583a9ec96a2e2d95996550b9de20861d7f'
SCHEMA = 'current_d50_response10_candidate_watch_v1'
CHILD_SCHEMA = 'current_d50_response10_candidate_native_v1'
RANKS, BUDGET, RSS_LIMIT = (35,100,35), 2**41, 8_000_000_000


def require(value,message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    path = Path(path).resolve()
    require('truthpost' not in path.parts and not
            (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()), 'Never open target arrays')
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bind(mapping):
    require(isinstance(mapping,dict),'Flat immutable bindings')
    for path,digest in mapping.items():
        require(isinstance(path,str) and Path(path).is_absolute() and
                re.fullmatch('[0-9a-f]{64}',digest or '') and sha(path) == digest,
                'Bound file changed: '+str(path))


def admit(args,worker,helper,helper_sha):
    sources = {str(Path(__file__).resolve()):args.source_sha256,str(worker):args.worker_sha256,
        str(HERE/'build.py'):args.builder_sha256,str(HERE/'PROTOCOL.md'):PROTOCOL_SHA,
        str(HERE/'build_v1/estimator.py'):args.candidate_sha256,str(BASE):BASE_SHA,str(helper):helper_sha}
    inputs = {str(HERE/'build_v1/result.json'):args.build_sha256,
              str(HERE/'build_v1/completion.json'):sha(HERE/'build_v1/completion.json')}
    bind(sources); bind(inputs)
    record,end = read(HERE/'build_v1/result.json'),read(HERE/'build_v1/completion.json')
    require(record['schema'] == 'current_d50_response10_candidate_build_v1' and
            all(record.get(k) is True for k in ('passed','completed','hashes_unchanged','sources_unchanged')) and
            end['passed'] is True and end['completed'] is True and end['result_sha256'] == args.build_sha256,
            'Complete reviewed literal-only build')
    for group in ('source_bindings','input_bindings'):
        bind(record[group])
        inputs.update(record[group])
    return sources,inputs,record


def finite_bill(trial):
    require(type(trial['flops']) is int and 0 < trial['flops'] <= BUDGET and
            all(math.isfinite(trial[k]) and trial[k] >= 0 for k in
                ('wall_s','context_wall_s','flopscope_backend_s','flopscope_overhead_s','native_residual_s')) and
            trial['wall_s'] < 120 and trial['context_wall_s'] < 120 and
            0 < trial['peak_rss_bytes'] < RSS_LIMIT, 'Original complete-call cost/resource gates')


def validate_child(record,end,digest,args,folder,control,lifecycle):
    require(record['schema'] == CHILD_SCHEMA and all(record.get(k) is True for k in
        ('passed','completed','hashes_unchanged','sources_unchanged','own_complete_bills',
         'all16_domains_qualified','all47_selectors_qualified','repeat_output_domain_byte_exact',
         'initialization_ownership_passed','source_inverse_verified','diagnostics_inverse_verified',
         'paid_arithmetic_and_lifecycle_unchanged')) and record['truth_accessed'] is False and
        end['passed'] is True and end['completed'] is True and end['result_sha256'] == digest,
        'Completed target-free candidate acquisition')
    require(record['candidate_sha256'] == args.candidate_sha256 and record['build_sha256'] == args.build_sha256 and
            record['builder_sha256'] == args.builder_sha256 and record['ranks'] == [35,100] and
            len(record['trials']) == len(record['input_preparation']) == len(record['diagnostic_records']) == 3,
            'Sole fixed35/100/repeat35 candidate')
    for path,expected in ((HERE/'native.py',args.worker_sha256),(HERE/'build.py',args.builder_sha256),
                          (HERE/'PROTOCOL.md',PROTOCOL_SHA),(HERE/'build_v1/estimator.py',args.candidate_sha256)):
        require(record['source_bindings'][str(path)] == expected,'Reviewed worker/build/protocol/candidate owner')
    for group in ('source_bindings','input_bindings','artifacts'):
        if group in record:
            bind(record[group])
    require(len(record['artifacts']) == 18,'Original five artifacts plus one small diagnostic per call')
    domains,startups = [],[]
    for call,trial in enumerate(record['trials']):
        require(trial['call'] == call and trial['rank'] == RANKS[call] and
                trial['includes_initialization'] is (call == 0),'Exact primary/repeat call ownership')
        finite_bill(trial)
        bp,lp = folder/f'call_{call:02d}_bill.json',folder/f'call_{call:02d}_lifecycle.json'
        require(read(bp) == trial and record['artifacts'][str(bp)] == sha(bp) and
                record['artifacts'][str(lp)] == sha(lp),'Actual complete bill/lifecycle files')
        lifecycle(trial,bp,lp)
        item = record['diagnostic_records'][call]
        dp = folder/f'call_{call:02d}_diagnostics.json'
        require(item == dict(path=str(dp),sha256=sha(dp)) and record['artifacts'][str(dp)] == sha(dp),
                'Own small domain/startup receipt')
        diagnostic = read(dp)
        require(diagnostic['call'] == call and diagnostic['rank'] == RANKS[call] and
                diagnostic['truth_accessed'] is False and diagnostic['offline_after_saved_bill_and_context_release'] is True,
                'Target-free post-release observation')
        domain,startup = diagnostic['domains'],diagnostic['startup']
        keys = {f'{l}:{slot}' for l in range(16) for slot in range(2 if l == 15 else 3)}
        require(domain['variance_count'] == 16 and domain['selector_count'] == 47 and
                set(domain['variance_domain']) == {str(i) for i in range(16)} and set(domain['selection_cells']) == keys,
                'All16/all47 observed cells')
        require(all(v['finite'] and math.isfinite(v['minimum']) and v['minimum'] >= 1e-10 and
                    v['below_floor_count'] == 0 for v in domain['variance_domain'].values()), 'Unfloored finite variances')
        require(all(v['floor'] == (.1 if key.split(':')[1] == '1' else .01) and
                    type(v['count']) is int and 0 <= v['count'] <= 1024 and v['count'] % 32 == 0
                    for key,v in domain['selection_cells'].items()), 'Response.1/other.01 selector floors')
        require(domain['added_participant_array_operations'] == startup['added_participant_array_operations'] == 0 and
                startup['public_counters_only'] is True,'Passive observation only')
        domains.append(domain); startups.append(startup)
    require(domains[0] == domains[2] and record['trials'][0]['output_bytes_sha256'] == record['trials'][2]['output_bytes_sha256'],
            'Exact repeated domain/order/output bytes')
    measured = record['trials'][0]['flops']-record['trials'][2]['flops']
    observed = startups[0]['observed_startup_region_flops']-startups[2]['observed_startup_region_flops']
    account = record['initialization_accounting']
    require(measured > 0 and measured == observed == record['initialization_flops'] ==
            account['cold_minus_repeat_flops'] == account['observed_startup_difference_flops'] and
            account['startup_subtracted_from_primary_bills'] is False and account['old_initialization_value_assumed'] is False,
            'Measured source-owned startup only; complete primary bills unchanged')


def derived_main(base):
    text = BASE.read_text()
    node = next(x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef) and x.name == 'main')
    original = ''.join(text.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    start = original.index('    sources = {str(Path(__file__).resolve()):args.source_sha256')
    stop = original.index('    require(not HERE.is_symlink()',start)
    edits = [
        ("for name in ('source','worker','protocol'):", "for name in ('source','worker','protocol','builder','build','candidate'):"),
        ("for k in ('source','worker','protocol'))", "for k in ('source','worker','protocol','builder','build','candidate'))"),
        ("worker,child_out,out = HERE/'capture.py',HERE/'native',HERE/'native_watch'",
         "worker,child_out,out = HERE/'native.py',HERE/'native_v1',HERE/'native_watch_v1'"),
        (original[start:stop], '    sources,inputs,control = admit(args,worker,HELPER,HELPER_SHA)\n'),
        ("'--source-sha256',args.worker_sha256,'--protocol-sha256',PROTOCOL_SHA]",
         "'--source-sha256',args.worker_sha256,'--protocol-sha256',PROTOCOL_SHA,\n"
         "        '--builder-sha256',args.builder_sha256,'--build-sha256',args.build_sha256,\n"
         "        '--candidate-sha256',args.candidate_sha256]"),
        ('candidate_sha256=BANK_SHA,control_result_sha256=CONTROL_SHA,truth_arrays_opened=False,',
         'candidate_sha256=args.candidate_sha256,build_sha256=args.build_sha256,truth_arrays_opened=False,'),
        ("len(found) <= 2,'Ordered call00/01 only'", "len(found) <= 3,'Ordered call00/01/02 only'"),
        ("bill['rank'] == 35 and", "bill['rank'] == RANKS[call] and"),
        ('dict(call=call,rank=35,path=str(bp)', 'dict(call=call,rank=RANKS[call],path=str(bp)'),
        ("result['completed_bill_count'] == 2,'Both completed lifecycles observed in order'",
         "result['completed_bill_count'] == 3,'All three completed lifecycles observed in order'"),
        ("        started_utc=datetime.now(timezone.utc).isoformat())",
         "        watch_source_inverse_verified=True,watch_parent_sha256=BASE_SHA,\n"
         "        started_utc=datetime.now(timezone.utc).isoformat())")]
    changed = original
    for before,after in edits:
        require(changed.count(before) == 1,'Unique watchdog splice: '+before[:80])
        changed = changed.replace(before,after)
    inverse = changed
    for before,after in reversed(edits):
        require(inverse.count(after) == 1,'Unique watchdog inverse')
        inverse = inverse.replace(after,before)
    require(inverse == original,'All other supervision/deadline/cleanup statements unchanged')
    return changed


def main():
    require(sha(BASE) == BASE_SHA,'Frozen census supervisor')
    spec = importlib.util.spec_from_file_location('response10_watch_parent',BASE)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    source = derived_main(base)
    namespace = dict(vars(base),__file__=str(Path(__file__).resolve()),__doc__=__doc__,HERE=HERE,
        SCHEMA=SCHEMA,CHILD_SCHEMA=CHILD_SCHEMA,PROTOCOL_SHA=PROTOCOL_SHA,BASE_SHA=BASE_SHA,RANKS=RANKS,
        admit=admit,validate_child=lambda record,end,digest,args,folder,control:
            validate_child(record,end,digest,args,folder,control,base.lifecycle))
    exec(compile(source,str(__file__)+':preserved_watch','exec'),namespace)
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
