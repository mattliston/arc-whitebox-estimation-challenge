"""DRAFT source-inverse reuse: one RPC algebra context,120s/8GB/no added swap.

The proven completed-observation, process-group cleanup and sealed1s tail are
retained. No numerical library or estimator is imported by this supervisor.
"""
import hashlib
import importlib.util
from pathlib import Path
import re
import shlex
import sys
import time

HERE = Path(__file__).resolve().parent
ADAPTER_OWNER = HERE.parent/'current_comp_shared_pair256_v1/watch_tiny.py'
ADAPTER_OWNER_SHA = '736304302b274697b19e031c645203f41bd25d0a6ee041e20b8ce9f82126f950'
ADAPTER_SPEC = ADAPTER_OWNER.with_name('TINY_WATCH_SPEC.md')
ADAPTER_SPEC_SHA = '50752c1a67057eb78f75c314fb52b5161a532192b825499e1915e486c77f0c97'


def _parent():
    for path,digest in ((ADAPTER_OWNER,ADAPTER_OWNER_SHA),(ADAPTER_SPEC,ADAPTER_SPEC_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Exact qualified tiny resource/transport owner')
    spec = importlib.util.spec_from_file_location('_lazy_qualified_tiny_watch',ADAPTER_OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def foreign_servers(rows,allowed):
    """Same reviewed underscore-server addition; never enlarge the allowlist."""
    found = []
    for row in rows:
        if row['pid'] in allowed:
            continue
        try:
            argv = shlex.split(row['command'])
        except ValueError:
            argv = row['command'].split()
        if argv and re.fullmatch(r'(?:python|pypy)[0-9.]*',Path(argv[0]).name.lower()) and any(
            x=='flopscope_server' or x.startswith('flopscope_server.') for x in argv[1:]):
            found.append(row)
    return found


def derive_source():
    base = _parent()
    original,parent_proof = base.derive_source()
    source,changes = original,[]
    def change(before,after,count=1):
        nonlocal source
        if source.count(before)!=count:
            raise RuntimeError('Exact lazy supervisor splice count: '+before[:100])
        for unused in range(count):
            at = source.index(before)
            changes.append((at,before,after))
            source = source[:at]+after+source[at+len(before):]
    change('CAP, RSS = 300.,8_000_000_000','CAP, RSS = 120.,8_000_000_000')
    change("HERE/'TINY_SOURCE_PREPARATION.md'","HERE/'PROTOCOL.md'")
    change("'current_comp_shared_pair256_tiny_watch_v1'","'current_comp_shared_lazy_rotation_tiny_watch_v1'")
    change("'current_comp_shared_pair256_tiny_v1'","'current_comp_shared_lazy_rotation_tiny_v1'",3)
    change("    out.mkdir()\n","    out.mkdir()\n    root.mkdir()\n")
    change("        PARENT:PARENT_SHA,PARENT_SPEC:PARENT_SPEC_SHA})",
           "        PARENT:PARENT_SHA,PARENT_SPEC:PARENT_SPEC_SHA,\n"
           "        ADAPTER_OWNER:ADAPTER_OWNER_SHA,ADAPTER_SPEC:ADAPTER_SPEC_SHA})")
    change("'Whole tiny300s and actual120s retirement progress through publication'",
           "'Whole tiny120s and actual120s retirement progress through publication'")
    change("'Final write inside actual1s/300s/progress tail'",
           "'Final write inside actual1s/120s/progress tail'")
    change("        require(not classifier(sample['rows'],allowed),'No foreign numerical worker')",
           "        require(not classifier(sample['rows'],allowed),'No foreign numerical worker')\n"
           "        require(not foreign_servers(sample['rows'],allowed),'No foreign underscore-spelled server')")
    change("                expected_mode,expected_call = ('local',number) if number < 2 else ('remote',number-2)\n"
           "                require(number < 4 and mode == item['mode'] == expected_mode\n"
           "                    and item['call'] == expected_call,'Actual fixed four-context roster')",
           "                expected_mode,expected_call = 'remote',0\n"
           "                require(number == 0 and mode == item['mode'] == expected_mode\n"
           "                    and item['call'] == expected_call,'ONE actual RPC algebra context')")
    change("        local_child = spawn([str(OPENPY),*argv_tail,'--mode','local'],'local',env)\n"
           "        finish(local_child); local = verify('local')",
           "        # No local/model context: the sole test is the real RPC client below.")
    change("prefix='astra_pair256_tiny_'","prefix='astra_lazy_rotation_tiny_'")
    change("server_attestation['admitted_sources'] == local['admitted_sources']",
           "server_attestation['admitted_sources'] == remote['admitted_sources']")
    change("        require(result['retired_contexts'] == 4,'All four actual contexts fully retired')\n"
           "        result['comparison'] = api.compare(local,remote,tick); tick()",
           "        require(result['retired_contexts'] == 1,'The sole real RPC context fully retired')\n"
           "        result['comparison'] = api.compare(remote,tick); tick()")
    change("            four_contexts_completed=True,both_interfaces_repeated=True,\n"
           "            actual_native_and_server_OpenBLAS_qualified=True,",
           "            one_context_completed=True,repeat_claimed=False,dense_branch_only=True,\n"
           "            actual_server_OpenBLAS_qualified=True,runtime_gain_measured=False,")
    before = base._nested_span(source,'verify')
    after = """    def verify(mode):
        require(mode == 'remote','Only one newly measured client mode')
        paths = [root/mode/n for n in ('result.json','completion.json','finalization.json')]
        ds = [sha(p,tick) for p in paths]
        r,e,f = map(read,paths)
        require(all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
            and e['result_sha256'] == f['result_sha256'] == ds[0] and f['completion_sha256'] == ds[1]
            and r['mode'] == mode and r['schema'] == 'current_comp_shared_lazy_rotation_tiny_v1'
            and r['own_complete_bill'] and r['one_context_completed'] and r['source_hashes_unchanged']
            and r['retired_contexts'] == 1 and r['repeat_claimed'] is False
            and r['all_array_weakrefs_retired'] and r['all_independent_references_passed']
            and r['dense_component_tariffs_exact'] and r['expected_rejections_passed']
            and r['dense_branch_only'] and r['model_calls'] == 0
            and r['truth_accessed'] is False and r['candidate_changed'] is False
            and r['full_estimator_fidelity_established'] is False and r['raw_value_measured'] is False
            and r['automatic_next_stage'] is False and 0 < r['peak_rss_bytes'] < 8_000_000_000
            and 0 <= r['elapsed_s'] < 120 and 0 <= f['elapsed_s'] < 120,
            'Actual positive one-context algebra certificate, never packed/full-card qualification')
        require(r['backend_attestation']['actual_origins_qualified']
            and r['backend_attestation']['numpy_free'],'Actual NumPy-free client origins')
        api.call_records(r,tick)
        for path,digest in r['source_bindings'].items():
            require(sha(Path(path),tick) == digest,'Every actual source remains frozen')
            require(path not in result['source_bindings'] or result['source_bindings'][path] == digest,
                    'No conflicting source owner')
            result['source_bindings'][path] = digest
        for path,digest in r['artifacts'].items():
            require(sha(Path(path),tick) == digest,'Actual completely billed retired receipt')
            result['artifacts'][path] = digest
        result['artifacts'].update({str(p):d for p,d in zip(paths,ds)})
        result['worker_seals'] = dict(result_sha256=ds[0],completion_sha256=ds[1],finalization_sha256=ds[2])
        return r
"""
    change(before,after)
    inverse = source
    for at,before,after in reversed(changes):
        if inverse[at:at+len(after)]!=after:
            raise RuntimeError('Exact complete lazy-watch positional inverse')
        inverse = inverse[:at]+before+inverse[at+len(after):]
    if inverse!=original:
        raise RuntimeError('Entire qualified tiny supervisor restored')
    proof = dict(owner_sha256=ADAPTER_OWNER_SHA,owner_spec_sha256=ADAPTER_SPEC_SHA,
        parent=parent_proof,whole_source_inverse_verified=True,
        derived_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        one_RPC_context_only=True,whole_cap_s=120,tree_cap_bytes=8_000_000_000,
        original_completed_observation_cleanup_and_sealed_tail_preserved=True,
        no_foreign_allowlist_expansion=True,numerical_imports=False,stage_executed=False)
    return source,proof


def main():
    if '--run' not in sys.argv:
        print('Inert lazy algebra watcher: explicit reviewed six SHA flags and --run are required.')
        return 0
    began = time.monotonic()
    base = _parent()
    source,proof = derive_source()
    namespace = dict(base._parent()._api,__file__=str(Path(__file__).resolve()),
        __name__='lazy_rotation_tiny_watch_effective',PARENT=base.PARENT,PARENT_SHA=base.PARENT_SHA,
        PARENT_SPEC=base.PARENT_SPEC,PARENT_SPEC_SHA=base.PARENT_SPEC_SHA,
        ADAPTER_OWNER=ADAPTER_OWNER,ADAPTER_OWNER_SHA=ADAPTER_OWNER_SHA,
        ADAPTER_SPEC=ADAPTER_SPEC,ADAPTER_SPEC_SHA=ADAPTER_SPEC_SHA,
        foreign_servers=foreign_servers,QUALIFICATION_BEGIN=began,SOURCE_ADAPTER_PROOF=proof)
    exec(compile(source,str(Path(__file__).resolve())+':whole-watch-inverse','exec'),namespace)
    return namespace['main']()


if __name__=='__main__':
    raise SystemExit(main())
