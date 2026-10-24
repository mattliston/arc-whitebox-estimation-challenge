"""DRAFT V2: unchanged120s/8GB tiny supervisor and negative-ledger sealing."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
OWNER = HERE/'watch_tiny.py'
OWNER_SHA = '67adbce35428959e007bf4372b8e6afc887313acca8a60a3d44fc7e39dc8523b'
OLD_SPEC = HERE/'TINY_WATCH_SPEC.md'
OLD_SPEC_SHA = '61f82640a43f5b725b86c7e89bfbc8f41925cd9f5e10d8afb51d7fc9222c9515'


def _owner():
    for path,digest in ((OWNER,OWNER_SHA),(OLD_SPEC,OLD_SPEC_SHA)):
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Exact frozen V1 supervisor and resource specification')
    spec = importlib.util.spec_from_file_location('_lazy_v1_watch_source',OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FAILURE_COLLECTION = '''        if not result['passed']:
            # Failure receipts are evidence only, never a qualified context.
            failure_path = root/'remote/call_00_failure.json'
            try:
                if failure_path.exists():
                    require(time.monotonic() < publication_deadline,'Bounded negative-ledger collection')
                    deadline()
                    require(not failure_path.is_symlink() and failure_path.stat().st_size < 64*1024,
                            'Only a bounded scalar failed-call ledger')
                    paths = [root/'remote'/n for n in ('result.json','completion.json','finalization.json')]
                    ds = [sha(p) for p in paths]
                    r,e,f = map(read,paths)
                    ledger,ld = read(failure_path),sha(failure_path)
                    require(all(x.get('passed') is False and x.get('completed') is False for x in (r,e,f))
                        and e['result_sha256']==f['result_sha256']==ds[0] and f['completion_sha256']==ds[1]
                        and r['schema']=='current_comp_shared_lazy_rotation_tiny_v2' and r['mode']=='remote'
                        and r['failure_ledger_saved'] is True and r['artifacts'][str(failure_path)]==ld
                        and ledger['schema']=='current_comp_shared_lazy_rotation_tiny_v2_failure_ledger'
                        and ledger['passed'] is False and ledger['completed'] is False
                        and ledger['mode']=='remote' and ledger['call']==0
                        and ledger['qualifying_retired_contexts']==0,
                        'Actual linked negative worker bill, never a positive retirement marker')
                    result['artifacts'].update({str(p):d for p,d in zip(paths,ds)})
                    result['artifacts'][str(failure_path)] = ld
                    result['failed_worker_seals'] = dict(result_sha256=ds[0],completion_sha256=ds[1],finalization_sha256=ds[2])
                    result['failure_ledger_observed'] = dict(path=str(failure_path),sha256=ld,
                        closed_bill_available=ledger['closed_bill_available'],
                        complete_flops=ledger['complete_flops'])
                    require(time.monotonic() < publication_deadline,'Negative ledger remains inside sealed tail')
                    deadline()
            except BaseException as ledger_error:
                result['failure_ledger_collection_error'] = str(ledger_error)[:1000]
'''


def derive_source():
    base = _owner()
    original,parent_proof = base.derive_source()
    source,changes = original,[]
    def change(before,after,count=1):
        nonlocal source
        if source.count(before)!=count:
            raise RuntimeError('Exact V2 supervisor splice count: '+before[:100])
        for unused in range(count):
            at = source.index(before)
            changes.append((at,before,after))
            source = source[:at]+after+source[at+len(before):]
    change("HERE/'tiny_worker.py'","HERE/'tiny_worker_v2.py'")
    change("HERE/'tiny_client.py'","HERE/'tiny_client_v2.py'",2)
    change("HERE/'TINY_PROTOCOL.md'","HERE/'TINY_V2_PROTOCOL.md'")
    change("HERE/'TINY_WATCH_SPEC.md'","HERE/'TINY_V2_WATCH_SPEC.md'")
    change("out,root = HERE/'tiny_qualification_watch_v1',HERE/'tiny_qualification_v1'",
           "out,root = HERE/'tiny_qualification_watch_v2',HERE/'tiny_qualification_v2'")
    change("'current_comp_shared_lazy_rotation_tiny_watch_v1'","'current_comp_shared_lazy_rotation_tiny_watch_v2'")
    change("'current_comp_shared_lazy_rotation_tiny_v1'","'current_comp_shared_lazy_rotation_tiny_v2'",3)
    change("ADAPTER_OWNER:ADAPTER_OWNER_SHA,ADAPTER_SPEC:ADAPTER_SPEC_SHA})",
           "ADAPTER_OWNER:ADAPTER_OWNER_SHA,ADAPTER_SPEC:ADAPTER_SPEC_SHA,\n"
           "        HERE/'watch_tiny.py':'"+OWNER_SHA+"',\n"
           "        HERE/'TINY_WATCH_SPEC.md':'"+OLD_SPEC_SHA+"'})")
    before = "        result['no_post_publication_observation_claimed'] = True\n"
    change(before,FAILURE_COLLECTION+before)
    inverse = source
    for at,before,after in reversed(changes):
        if inverse[at:at+len(after)]!=after:
            raise RuntimeError('Exact V2 complete supervisor inverse span')
        inverse = inverse[:at]+before+inverse[at+len(after):]
    if inverse!=original:
        raise RuntimeError('Entire frozen V1 effective supervisor restored')
    return source,dict(parent=parent_proof,owner_sha256=OWNER_SHA,owner_spec_sha256=OLD_SPEC_SHA,
        whole_V1_effective_watch_inverse=True,derived_sha256=hashlib.sha256(source.encode()).hexdigest(),
        original_sampling_deadlines_caps_cleanup_and_sealed_tail_unchanged=True,
        negative_ledger_never_qualifies=True,one_RPC_context_only=True,numerical_execution=False)


def main():
    began = time.monotonic()
    base = _owner()
    original = OWNER.read_text()
    nodes = [n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name=='main']
    if len(nodes)!=1:
        raise RuntimeError('Exact unchanged V1 watch entry')
    node = nodes[0]
    entry = ''.join(original.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    before,after = '    began = time.monotonic()\n','    began = V2_QUALIFICATION_BEGIN\n'
    if entry.count(before)!=1 or after in entry:
        raise RuntimeError('Exact original watch entry clock')
    changed = entry.replace(before,after)
    if changed.replace(after,before)!=entry:
        raise RuntimeError('Complete watch entry inverse, earlier whole-stage clock only')
    namespace = dict(vars(base),__file__=str(Path(__file__).resolve()),derive_source=derive_source,
                     V2_QUALIFICATION_BEGIN=began)
    exec(compile(changed,str(Path(__file__).resolve())+':V1-entry-inverse','exec'),namespace)
    return namespace['main']()


if __name__=='__main__':
    raise SystemExit(main())
