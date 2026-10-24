"""Inert exact-source adapter of qualified scalar-v3 process supervision.

No stage, Python child or numerical import unless main receives --run.
The original resource loop/cleanup/publication tail is retained with explicit
fixed qualification contract,300s/8GB scope and actual retired-context progress.
"""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent/'current_comp_shared_old384_v1'
PARENT = SHARED/'watch_scalar_semantics_v3.py'
PARENT_SHA = '42f643f2d484f39faa52060d36f7983b13c59b083f764df36899245f4a26f297'
PARENT_SPEC = SHARED/'SCALAR_SEMANTICS_WATCH_V3_SPEC.md'
PARENT_SPEC_SHA = '691da41946775190b1652f40f047dcaffc267c6a823328d390a59b4832171c12'


def _parent():
    for path,digest in ((PARENT,PARENT_SHA),(PARENT_SPEC,PARENT_SPEC_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Frozen qualified scalar-v3 resource owner')
    spec = importlib.util.spec_from_file_location('pair256_original_scalar_v3_watch',PARENT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replace(text, old, new, changes):
    if text.count(old) != 1:
        raise RuntimeError('Unique pair256 watcher splice: '+old[:90])
    at = text.index(old)
    changes.append((at,old,new))
    return text[:at]+new+text[at+len(old):]


def _nested_span(text, name):
    nodes = [n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise RuntimeError('One exact inherited function: '+name)
    node = nodes[0]
    lines = text.splitlines(keepends=True)
    # These inherited functions are ASCII and end on their final statement.
    return ''.join(lines[node.lineno-1:node.end_lineno])


def derive_source():
    base = _parent()
    original,parent_proof = base.derive_source()
    source,changes = original,[]
    edits = (
        ('CAP, RSS = 60.,2_000_000_000','CAP, RSS = 300.,8_000_000_000'),
        ("for key in ('source','worker','protocol','spec'):",
         "for key in ('source','worker','client','protocol','spec','decision'):"),
        ('began = previous = time.monotonic()',
         'began = previous = last_progress = QUALIFICATION_BEGIN'),
        ("worker = HERE/'scalar_semantics_probe_v3.py'", "worker = HERE/'tiny_worker.py'"),
        ("HERE/'SCALAR_SEMANTICS_PROTOCOL_V3.md'", "HERE/'TINY_PROTOCOL.md'"),
        ("HERE/'SCALAR_SEMANTICS_WATCH_V3_SPEC.md'", "HERE/'TINY_WATCH_SPEC.md'"),
        ("out,root = HERE/'scalar_semantics_watch_v3',HERE/'scalar_semantics_v3'",
         "out,root = HERE/'tiny_qualification_watch_v1',HERE/'tiny_qualification_v1'"),
        ("schema='current_comp_shared_old384_scalar_semantics_watch_v3'",
         "schema='current_comp_shared_pair256_tiny_watch_v1'"),
        ("    require(all(re.fullmatch('[0-9a-f]{64}',d or '') and sha(p) == d for p,d in own.items()),",
         "    own.update({HERE/'tiny_client.py':args.client_sha256,\n"
         "        HERE/'TINY_SOURCE_PREPARATION.md':args.decision_sha256,\n"
         "        PARENT:PARENT_SHA,PARENT_SPEC:PARENT_SPEC_SHA})\n"
         "    require(all(re.fullmatch('[0-9a-f]{64}',d or '') and sha(p) == d for p,d in own.items()),"),
        ("    children,logs = [],[]\n", "    children,logs = [],[]\n"
         "    readers,pending = {},{}\n"
         "    result.update(retired_contexts=0,context_markers=[],raw_value_measured=False,\n"
         "        automatic_next_stage=False,source_adapter_proof=SOURCE_ADAPTER_PROOF)\n"),
        ("        require(time.monotonic()-began < CAP,'Whole diagnostic60s, including final publication')",
         "        require(time.monotonic()-began < CAP and time.monotonic()-last_progress < 120.,\n"
         "            'Whole tiny300s and actual120s retirement progress through publication')"),
        ('        nonlocal previous,initial_swap\n', '        nonlocal previous,initial_swap,last_progress\n'),
        ('        absolute = min(previous+1.,began+CAP)',
         '        absolute = min(previous+1.,began+CAP,last_progress+120.)'),
        ("        require(growth <= 0 and combined < RSS,'Full owned tree2GB/no added swap')",
         "        require(growth <= 0 and combined < RSS,'Full owned tree8GB/no added swap')"),
        ("        deadline()\n        result['maximum_combined_rss_bytes']",
         "        for mode,reader in readers.items():\n"
         "            pending[mode] += reader.read()\n"
         "            require(len(pending[mode]) < 4*1024**2,'Bounded actual context marker log')\n"
         "            while '\\n' in pending[mode]:\n"
         "                line,pending[mode] = pending[mode].split('\\n',1)\n"
         "                try:\n"
         "                    item = json.loads(line)\n"
         "                except (ValueError,TypeError):\n"
         "                    continue\n"
         "                if not isinstance(item,dict) or item.get('event') != 'retired':\n"
         "                    continue\n"
         "                deadline()\n"
         "                number = result['retired_contexts']\n"
         "                expected_mode,expected_call = ('local',number) if number < 2 else ('remote',number-2)\n"
         "                require(number < 4 and mode == item['mode'] == expected_mode\n"
         "                    and item['call'] == expected_call,'Actual fixed four-context roster')\n"
         "                path = root/mode/f'call_{expected_call:02d}.json'\n"
         "                digest,record = sha(path),read(path)\n"
         "                require(item['path'] == str(path) and item['complete_sha256'] == digest\n"
         "                    and record['call'] == expected_call and record['mode'] == mode\n"
         "                    and record['schema'] == 'current_comp_shared_pair256_tiny_v1'\n"
         "                    and all(record.get(k) is True for k in ('passed','completed',\n"
         "                        'context_retired','all_array_weakrefs_retired','own_complete_bill'))\n"
         "                    and record['complete_flops'] == item['complete_flops'],\n"
         "                    'Progress only after complete billed and released real receipt')\n"
         "                result['context_markers'].append(dict(mode=mode,call=expected_call,\n"
         "                    path=str(path),sha256=digest))\n"
         "                result['retired_contexts'] += 1\n"
         "                last_progress = time.monotonic()\n"
         "        deadline()\n        result['maximum_combined_rss_bytes']"),
        ("        children.append(child)\n", "        children.append(child)\n"
         "        if name in ('local','remote'):\n"
         "            readers[name] = (out/(name+'.log')).open('r')\n"
         "            pending[name] = ''\n"),
        ("        argv_tail = ['-B','-u',str(worker),'--run','--source-sha256',args.worker_sha256,\n"
         "            '--protocol-sha256',args.protocol_sha256]",
         "        argv_tail = ['-B','-u',str(worker),'--run','--source-sha256',args.worker_sha256,\n"
         "            '--protocol-sha256',args.protocol_sha256,'--client-sha256',args.client_sha256,\n"
         "            '--watcher-sha256',args.source_sha256,'--watch-spec-sha256',args.spec_sha256,\n"
         "            '--decision-sha256',args.decision_sha256]"),
        ("        env = dict(os.environ,PYTHONUNBUFFERED='1')",
         "        env = dict(os.environ,PYTHONUNBUFFERED='1',FLOPSCOPE_MAX_ARRAY_BYTES='4294967296')"),
        ("prefix='astra_scalar_semantics_'", "prefix='astra_pair256_tiny_'"),
        ("        server = spawn([str(OPENPY),'-B','-u','-m','flopscope_server','--url',url,'--timeout','60'],\n"
         "            'server',remote_env)",
         "        server = spawn([str(OPENPY),*argv_tail,'--mode','server','--server-url',url],\n"
         "            'server',remote_env)"),
        ("        client = spawn([str(CLIENTPY),*argv_tail,'--mode','remote','--server-url',url],'remote',remote_env)",
         "        client_tail = list(argv_tail)\n"
         "        client_tail[2] = str(HERE/'tiny_client.py')\n"
         "        client = spawn([str(CLIENTPY),*client_tail,'--mode','remote','--server-url',url],'remote',remote_env)"),
        ("        result['comparison'] = api.compare(local,remote); tick()",
         "        attestation_path = root/'server_attestation.json'\n"
         "        server_attestation = read(attestation_path)\n"
         "        require(server_attestation['schema'] == 'current_comp_shared_pair256_tiny_v1'\n"
         "            and server_attestation['passed'] and server_attestation['mode'] == 'server'\n"
         "            and server_attestation['pid'] == server.pid and server_attestation['url'] == url\n"
         "            and server_attestation['no_fixture_arrays_created']\n"
         "            and server_attestation['backend_attestation']['actual_origins_qualified']\n"
         "            and server_attestation['backend_attestation']['production_OpenBLAS_qualified']\n"
         "            and server_attestation['admitted_sources'] == local['admitted_sources'],\n"
         "            'Actual same-production OpenBLAS server, not only an interpreter label')\n"
         "        for path,digest in server_attestation['source_bindings'].items():\n"
         "            require(sha(Path(path),tick) == digest,'Fresh actual server imported source')\n"
         "            require(path not in result['source_bindings'] or result['source_bindings'][path] == digest,\n"
         "                'Consistent server source union')\n"
         "            result['source_bindings'][path] = digest\n"
         "        result['artifacts'][str(attestation_path)] = sha(attestation_path,tick)\n"
         "        require(result['retired_contexts'] == 4,'All four actual contexts fully retired')\n"
         "        result['comparison'] = api.compare(local,remote,tick); tick()"),
        ("            scalar_hypothesis_supported=result['comparison']['all_predicted_expression_tariffs_observed']\n"
         "                and result['comparison']['slice_control_all_intermediate_bytes_exact'],",
         "            qualification_passed=result['comparison']['passed'],\n"
         "            four_contexts_completed=True,both_interfaces_repeated=True,\n"
         "            actual_native_and_server_OpenBLAS_qualified=True,"),
        ('        publication_deadline = min(previous+1.,began+CAP)',
         '        publication_deadline = min(previous+1.,began+CAP,last_progress+120.)\n'
         '        for reader in readers.values():\n            reader.close()'),
        ("'Final write inside original1s/60s tail'", "'Final write inside actual1s/300s/progress tail'"),
        ("('passed','completed','scalar_hypothesis_supported',", "('passed','completed','qualification_passed',"),
    )
    for old,new in edits:
        source = _replace(source,old,new,changes)
    old_verify = _nested_span(source,'verify')
    new_verify = """    def verify(mode):
        paths = [root/mode/n for n in ('result.json','completion.json','finalization.json')]
        ds = [sha(p,tick) for p in paths]
        r,e,f = map(read,paths)
        require(all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
            and e['result_sha256'] == f['result_sha256'] == ds[0] and f['completion_sha256'] == ds[1]
            and r['mode'] == mode and r['schema'] == 'current_comp_shared_pair256_tiny_v1'
            and r['own_complete_bill'] and r['two_contexts_completed'] and r['source_hashes_unchanged']
            and r['retired_contexts'] == 2 and r['repeat_byte_exact']
            and r['all_array_weakrefs_retired'] and r['all_independent_references_passed']
            and r['metric_allowance_passed'] and r['expected_rejections_passed']
            and r['original_physical_coordinates_verified'] and r['model_calls'] == 0
            and r['truth_accessed'] is False and r['candidate_changed'] is False
            and r['full_estimator_fidelity_established'] is False and r['raw_value_measured'] is False
            and r['automatic_next_stage'] is False and 0 < r['peak_rss_bytes'] < 1_000_000_000
            and 0 <= r['elapsed_s'] < 270 and 0 <= f['elapsed_s'] < 270,
            'Actual newly positive paired-circuit mode, not a full-card certificate')
        require(r['backend_attestation']['actual_origins_qualified']
            and (r['backend_attestation']['production_OpenBLAS_qualified'] if mode == 'local'
                 else r['backend_attestation']['numpy_free']), 'Actual backend/client origins')
        api.call_records(r,tick)
        for path,digest in r['source_bindings'].items():
            require(sha(Path(path),tick) == digest,'Every actual numerical owner remains frozen')
            require(path not in result['source_bindings'] or result['source_bindings'][path] == digest,
                    'No conflicting owner')
            result['source_bindings'][path] = digest
        for path,digest in r['artifacts'].items():
            require(sha(Path(path),tick) == digest,'Actual fully billed context receipt')
            result['artifacts'][path] = digest
        result['artifacts'].update({str(p):d for p,d in zip(paths,ds)})
        return r
"""
    source = _replace(source,old_verify,new_verify,changes)
    inverse = source
    for at,old,new in reversed(changes):
        if inverse[at:at+len(new)] != new:
            raise RuntimeError('Exact positional whole-watch inverse')
        inverse = inverse[:at]+old+inverse[at+len(new):]
    if inverse != original:
        raise RuntimeError('Complete qualified scalar-v3 supervisor inverse')
    proof = dict(parent_source_sha256=PARENT_SHA,parent_spec_sha256=PARENT_SPEC_SHA,
                 parent=parent_proof,whole_source_inverse_verified=True,
                 derived_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                 preserved_actual_observation_and_cleanup_loop=True,
                 fresh_fixed_context_contract=True,whole_cap_s=300,tree_cap_bytes=8_000_000_000,
                 actual_retired_context_progress_s=120,bounded_publication_tail_s=1,
                 numerical_imports=False,stage_executed=False)
    return source,proof


def main():
    if '--run' not in sys.argv:
        print('Inert source-only pair256 supervisor; explicit --run and all reviewed SHA flags are required.')
        return 0
    began = time.monotonic()
    base = _parent()
    source,proof = derive_source()
    namespace = dict(base._api,__file__=str(Path(__file__).resolve()),__name__='pair256_tiny_watch_effective',
                     PARENT=PARENT,PARENT_SHA=PARENT_SHA,PARENT_SPEC=PARENT_SPEC,PARENT_SPEC_SHA=PARENT_SPEC_SHA,
                     QUALIFICATION_BEGIN=began,SOURCE_ADAPTER_PROOF=proof)
    exec(compile(source,str(Path(__file__).resolve())+':whole-source-inverse','exec'),namespace)
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
