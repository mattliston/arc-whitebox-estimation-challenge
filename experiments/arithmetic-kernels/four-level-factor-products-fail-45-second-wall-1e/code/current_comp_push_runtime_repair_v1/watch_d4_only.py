"""Fixed D4 pair only; preserve the failed c2 cold-call experiment verbatim."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER = HERE/'watch_rpc37.py'
OWNER_SHA = '4c636a5598be174bab6d9b44cb8d3ab5c2731d963276f6fd13d8bf7606540fef'
OLD_PROTOCOL = HERE/'RPC37_PROTOCOL.md'
OLD_PROTOCOL_SHA = '8b5fcd42830570faa17dbdc172c918b48c7ee164ff114e7fe4df372aca1161a0'
NEGATIVE = ('2b4960474222f76129fdda728512a82b807afac0d847e3c53f16965b3e2f4930',
    '0c0a5b5223aea0661a2a7f3beb0dab5930329969cf57c5b92d158f2bdf209984',
    '35b70b981fd74df4a44d2e2a2e2213621ee72809ccbde2911f648ac54858e2d2')


def negative_control(result, tick):
    a = _api; require,sha,read = (a[k] for k in ('require','sha','read'))
    folder = HERE/'rpc37_v1'
    paths = [folder/n for n in ('result.json','completion.json','finalization.json')]
    ds = tuple(sha(p,tick) for p in paths); require(ds == NEGATIVE,'Exact preserved negative seals')
    r,e,f = map(read,paths)
    require(all(x.get('passed') is False and x.get('completed') is False for x in (r,e,f))
        and e['result_sha256'] == f['result_sha256'] == ds[0] and f['completion_sha256'] == ds[1]
        and r['retired_calls'] == 0 and r['records'] == [[],[]]
        and all(r[k] is True for k in ('cleanup_passed','whole_owned_tree_retired','observational_groups_retired'))
        and r['maximum_swap_growth_bytes'] == 0 and 0 < r['maximum_interval_s'] <= 1
        and r['maximum_combined_rss_bytes'] < 8_000_000_000 and 0 < r['elapsed_s'] <= f['elapsed_s'] < 300
        and 'Context wall60/residual.4 bounds' in r['error']
        and r['candidate_sha256'] == '27f6765896063873658e0dda67091d11644a85a59924325637e5606c1c2dedb5',
        'Old baseline timeout and retired owners, no old PASS')
    log = folder/'base.log'
    require(sha(log,tick) == r['artifacts'][str(log)],'Actual saved failed-baseline client log')
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    require(len(rows) == 1,'Only one closed baseline context reported')
    row = rows[0]
    require(row['index'] == 0 and row['rank'] == 37 and row['output_shape'] == [16,1024]
        and row['output_dtype'] == 'float32' and row['output_sha256'] == a['BASE_OUTPUT']
        and row['flops'] == a['BASE_F'][0] and row['util'] == row['flops']/2**41
        and row['timing'] == dict(wall_time_s=62.916841667,flopscope_backend_time_s=46.211133248,
            flopscope_overhead_time_s=16.365047812,residual_wall_time_s=0.3406606069999967),
        'Exact same-head cold37 response and own complete bill, still fails60s gate')
    raw_path = HERE.parent/'current_comp_push_d4_runtime_base37_v1_0_r37.f32'
    raw,_ = a['values'](raw_path)
    require(hashlib.sha256(raw).hexdigest() == a['BASE_OUTPUT'],'Actual original failed-call all16 output')
    for p,d in zip(paths,ds):
        result['input_bindings'][str(p)] = d
    result['input_bindings'][str(log)] = sha(log,tick)
    result['input_bindings'][str(raw_path)] = a['BASE_OUTPUT']
    result['records'][0] = [row]
    result.update(negative_baseline_seals=list(ds),negative_baseline_still_failed=True,
        baseline_pair_qualified=False,fresh_baseline_calls=0,source_adapter_proof=SOURCE_PROOF,
        historical_cold_output=str(raw_path))


def derive_source():
    for p,d in ((OWNER,OWNER_SHA),(OLD_PROTOCOL,OLD_PROTOCOL_SHA)):
        if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest() != d:
            raise RuntimeError('Exact preserved v1 owner')
    original = OWNER.read_text()
    edits = (
        ("OUT = HERE/'rpc37_v1'","OUT = HERE/'rpc37_d4_only_v1'"),
        ("TAGS = ('current_comp_push_d4_runtime_base37_v1','current_comp_push_d4_runtime_new37_v1')",
         "TAGS = ('unused_negative_baseline','current_comp_push_d4_only37_v1')"),
        ("HERE/'watch_rpc37.py':args.source_sha256","HERE/'watch_d4_only.py':args.source_sha256"),
        ("HERE/'RPC37_PROTOCOL.md':args.protocol_sha256","HERE/'D4_ONLY_PROTOCOL.md':args.protocol_sha256"),
        ("paths = [NEW/f'{tag}{suffix}' for tag in TAGS for suffix in ('.json','_0_r37.f32','_1_r37.f32')]",
         "paths = [NEW/f'{TAGS[1]}{suffix}' for suffix in ('.json','_0_r37.f32','_1_r37.f32')]"),
        ("schema='current_comp_push_d4_rpc37_v1'","schema='current_comp_push_d4_only_rpc37_v1'"),
        ('        for path,digest in FIXTURE.items():',
         '        negative_control(result,tick)\n        for path,digest in FIXTURE.items():'),
        ('for role,source in enumerate((BASE,candidate)):', 'for role,source in ((1,candidate),):'),
        ("payloads = [[values(NEW/f'{tag}_{i}_r37.f32') for i in range(2)] for tag in TAGS]",
         "payloads = [[values(Path(result['historical_cold_output']))],\n"
         "            [values(NEW/f'{TAGS[1]}_{i}_r37.f32') for i in range(2)]]"),
        ('        for role in range(2):','        for role in (1,):'),
        ("tuple(x['flops'] for x in base) == BASE_F", "tuple(x['flops'] for x in base) == (BASE_F[0],)"),
        ("init = [rows[0]['flops']-rows[1]['flops'] for rows in result['records']]",
         "init = [None,new[0]['flops']-new[1]['flops']]"),
        ('init == [1944970298,1383914298]', 'init == [None,1383914298]'),
        ("all(abs(n['flops']/b['flops']-1) <= .05 for b,n in zip(base,new,strict=True))",
         "all(abs(n['flops']/b-1) <= .05 for b,n in zip(BASE_F,new,strict=True))"),
        ('ratio = sum(walls[1])/sum(walls[0]);','ratio = walls[1][0]/walls[0][0];'),
        ('paired_wall_ratio=ratio,','cold_wall_ratio=ratio,'),
        ("'paired_wall_ratio','maximum_new_wall_s'","'cold_wall_ratio','maximum_new_wall_s'"),
    )
    changed,spans = original,[]
    for before,after in edits:
        if changed.count(before) != 1:
            raise RuntimeError('Exact D4-only metadata span: '+before[:80])
        pos=changed.index(before);spans.append((pos,before,after));changed=changed[:pos]+after+changed[pos+len(before):]
    inverse=changed
    for pos,before,after in reversed(spans):
        if inverse[pos:pos+len(after)] != after:
            raise RuntimeError('Exact positional D4-only inverse')
        inverse=inverse[:pos]+before+inverse[pos+len(after):]
    if inverse != original:
        raise RuntimeError('Entire previous monitor restored')
    return changed,dict(owner_sha256=OWNER_SHA,whole_source_inverse_verified=True,splices=len(spans),
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),original_resource_methods_unchanged=True,
        original_client_and_server_unchanged=True,fresh_calls=2,negative_baseline_never_promoted=True)


_SOURCE,SOURCE_PROOF = derive_source()
_api = dict(__name__='_push_d4_only_effective',__file__=str(Path(__file__).resolve()),negative_control=negative_control)
exec(compile(_SOURCE,str(Path(__file__).resolve())+':full-monitor-inverse','exec'),_api)
_api['PINS'].update({OWNER:OWNER_SHA,OLD_PROTOCOL:OLD_PROTOCOL_SHA})
main = _api['main']


if __name__ == '__main__':
    raise SystemExit(main())
