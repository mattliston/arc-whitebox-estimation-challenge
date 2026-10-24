"""DRAFT V2: exact V1 arithmetic, corrected reduction tariff and failure ledger.

Source admission is stdlib-only. No fixture, context or stage runs at import.
The frozen fragment, parent, original worker and failed receipts are untouched.
"""
import ast
import gc
import hashlib
import json
from pathlib import Path
import traceback

HERE = Path(__file__).resolve().parent
V1_WORKER = HERE/'tiny_worker.py'
V1_WORKER_SHA = '434bc0f3b696433ca68562a75f017e60ba984ac5a6272a833a066363e106e7a4'
V1_SOURCES = {
    V1_WORKER: V1_WORKER_SHA,
    HERE/'tiny_client.py': '2e00d4fd9cb52cbb4fe8839d8c3165fce6205f58e65048c40a262ec0605415b7',
    HERE/'watch_tiny.py': '67adbce35428959e007bf4372b8e6afc887313acca8a60a3d44fc7e39dc8523b',
    HERE/'TINY_PROTOCOL.md': '16e021f57fc680b06a4d89f65aa55badaceb7468d265f0bf4d15c23cd7a73097',
    HERE/'TINY_WATCH_SPEC.md': '61f82640a43f5b725b86c7e89bfbc8f41925cd9f5e10d8afb51d7fc9222c9515',
}


def _read_owner():
    for path,digest in V1_SOURCES.items():
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Immutable V1 lazy unit source: '+str(path))
    return V1_WORKER.read_text()


def _span(source, name):
    nodes = [n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name]
    if len(nodes)!=1:
        raise RuntimeError('One exact V1 function: '+name)
    node = nodes[0]
    return ''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])


def _change(source, before, after, changes):
    if source.count(before)!=1:
        raise RuntimeError('One exact V2 metadata/ledger splice: '+before[:100])
    at = source.index(before)
    changes.append((at,before,after))
    return source[:at]+after+source[at+len(before):]


def _inverse(source, original, changes):
    for at,before,after in reversed(changes):
        if source[at:at+len(after)]!=after:
            raise RuntimeError('Exact V2 positional inverse')
        source = source[:at]+before+source[at+len(after):]
    if source!=original:
        raise RuntimeError('Entire frozen V1 owner restored')


def _error_and_release(error):
    """Keep bounded strings only; traceback frames must not retain fixture arrays."""
    detail = dict(error_type=type(error).__name__,error=str(error)[:1000])
    pending, seen = [error], set()
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        pending.extend(x for x in (item.__cause__,item.__context__) if x is not None)
        traceback.clear_frames(item.__traceback__)
        item.__traceback__ = None
    return detail


class TinyCallFailure(RuntimeError):
    def __init__(self, ledger):
        super().__init__(ledger['error_type']+': '+ledger['error'])
        self.failure_ledger = ledger


ARITHMETIC = '''def arithmetic(flops,fnp,ns,mode,index,guard):
    require(mode=='remote' and index==0, 'ONE actual RPC unit context')
    audit = dict(sections=[],rows=[],rejected=[],refs=[],active_section=None)
    ctx,summary,failure = None,None,None
    close_returned = False
    owners_retired_inside_context = None
    try:
        with flops.BudgetContext(flop_budget=CALL_BUDGET,quiet=True,namespace='lazy-rotation-tiny') as ctx:
            try:
                rows,sections,rejected,refs = _exercise(fnp,ns,ctx,audit)
                gc.collect()
                require(all(ref() is None for ref in refs), 'All tracked numerical owners released before close')
                guard()
                owners_retired_inside_context = True
            except BaseException as error:
                failure = _error_and_release(error)
                gc.collect()
                owners_retired_inside_context = all(ref() is None for ref in audit['refs'])
        close_returned = True
        summary = ctx.summary_dict()
        if failure is not None:
            raise RuntimeError(failure['error_type']+': '+failure['error'])
        complete = int(summary['flops_used'])
        require(complete==sum(s['flops'] for s in sections) and 0<complete<CALL_BUDGET,
                'Complete input/reference/guard/operator tariff, no hidden subtraction')
        tracked = len(refs)
        record = dict(schema=SCHEMA,mode=mode,call=index,passed=True,completed=True,
            context_retired=True,all_array_weakrefs_retired=True,tracked_array_weakrefs=tracked,
            own_complete_bill=True,complete_flops=complete,bill=summary,rows=rows,
            sections=sections,rejected_cases=rejected,dense_branch_only=True,
            remote_sections_are_actual_budget_deltas=True,model_calls=0,target_accessed=False)
        verify_call(record,mode,index)
        del refs
        gc.collect()
        return record
    except BaseException as error:
        terminal = _error_and_release(error)
        gc.collect()
        # Never ask for a live summary. A normal __exit__ return is our evidence
        # that the public summary API can use its acknowledged closed cache.
        if close_returned and summary is None:
            try:
                summary = ctx.summary_dict()
            except BaseException as summary_error:
                terminal['summary_error'] = _error_and_release(summary_error)
        known = sum(s['flops'] for s in audit['sections'])
        complete = int(summary['flops_used']) if summary is not None else None
        ledger = dict(schema=SCHEMA+'_failure_ledger',passed=False,completed=False,
            mode=mode,call=index,qualifying_retired_contexts=0,
            context_entered=ctx is not None,context_close_returned=close_returned,
            closed_bill_available=summary is not None,bill=summary,complete_flops=complete,
            last_acknowledged_flops=int(ctx.flops_used) if ctx is not None else None,
            reported_section_flops=known,
            unattributed_flops=None if complete is None else complete-known,
            sections=audit['sections'],active_section=audit['active_section'],
            tracked_array_weakrefs=len(audit['refs']),
            tracked_arrays_retired_inside_context=owners_retired_inside_context,
            tracked_arrays_remaining_after_unwind=sum(ref() is not None for ref in audit['refs']),
            terminal_error=terminal,model_calls=0,truth_accessed=False,
            numerical_retry_authorized=False,**(failure or terminal))
        raise TinyCallFailure(ledger) from None
'''


def derive_source():
    original = _read_owner()
    v1 = dict(__file__=str(V1_WORKER),__name__='_lazy_v1_frozen_metadata')
    exec(compile(original,str(V1_WORKER),'exec'),v1)
    old_main, parent_main_proof = v1['derive_main']()
    main, main_changes = old_main, []
    for before,after in (
        ("HERE/'tiny_qualification_v1'/args.mode", "HERE/'tiny_qualification_v2'/args.mode"),
        ("HERE/'tiny_qualification_v1/server_attestation.json'", "HERE/'tiny_qualification_v2/server_attestation.json'"),
        ("        result.update(passed=False,completed=False,error_type=type(error).__name__,error=str(error)[:1000])\n",
         "        result.update(passed=False,completed=False,error_type=type(error).__name__,error=str(error)[:1000])\n"
         "        if isinstance(error,TinyCallFailure):\n"
         "            result['_failure_ledger'] = error.failure_ledger\n"),
        ("        if not result['passed']:\n            signal.setitimer(signal.ITIMER_REAL,2.)\n",
         "        if not result['passed']:\n            signal.setitimer(signal.ITIMER_REAL,2.)\n"
         "            ledger = result.pop('_failure_ledger',None)\n"
         "            if ledger is not None:\n"
         "                try:\n"
         "                    require(len((json.dumps(ledger,indent=2,sort_keys=True,allow_nan=False)+'\\n').encode()) < 64*1024, 'Bounded failure-only closed ledger')\n"
         "                    path = out/'call_00_failure.json'\n"
         "                    result['artifacts'][str(path)] = write(path,ledger)\n"
         "                    result['failure_ledger_saved'] = True\n"
         "                    result['failed_context_closed_bill_available'] = ledger['closed_bill_available']\n"
         "                except BaseException as ledger_error:\n"
         "                    result['failure_ledger_saved'] = False\n"
         "                    result['failure_ledger_error'] = str(ledger_error)[:1000]\n"),
    ):
        main = _change(main,before,after,main_changes)
    _inverse(main,old_main,main_changes)
    main_proof = dict(parent=parent_main_proof,whole_V1_effective_main_inverse=True,
        failure_ledger_only_no_success_marker=True,derived_sha256=hashlib.sha256(main.encode()).hexdigest())
    source,changes = original,[]
    edits = (
        ("SCHEMA = 'current_comp_shared_lazy_rotation_tiny_v1'", "SCHEMA = 'current_comp_shared_lazy_rotation_tiny_v2'"),
        ("HERE/'tiny_worker.py':args.source_sha256", "HERE/'tiny_worker_v2.py':args.source_sha256"),
        ("HERE/'tiny_client.py':args.client_sha256, HERE/'TINY_PROTOCOL.md':args.protocol_sha256",
         "HERE/'tiny_client_v2.py':args.client_sha256, HERE/'TINY_V2_PROTOCOL.md':args.protocol_sha256"),
        ("HERE/'watch_tiny.py':args.watcher_sha256, HERE/'TINY_WATCH_SPEC.md':args.watch_spec_sha256",
         "HERE/'watch_tiny_v2.py':args.watcher_sha256, HERE/'TINY_V2_WATCH_SPEC.md':args.watch_spec_sha256"),
        ("HERE/'tiny_qualification_v1/remote/call_00.json'", "HERE/'tiny_qualification_v2/remote/call_00.json'"),
        ("return h(r*old)+h(r*joined)+h(r*r)+h(r*old)+h(r*r)+20",
         "return h(r*old)+h(r*joined)+h(r*r)+h(r*old)+h(r*r)+21"),
        ("finite_price(RANK*RANK)+finite_price(RANK*PREFIX)+20)",
         "finite_price(RANK*RANK)+finite_price(RANK*PREFIX)+21)"),
        ("fnp.full((RANK,RANK),2**35,dtype=fnp.float32),",
         "fnp.full((RANK,RANK),2**35,dtype=fnp.float32).astype(fnp.float32),"),
        ("            fnp.full((RANK,PREFIX),2**36,dtype=fnp.float32)))",
         "            fnp.full((RANK,PREFIX),2**36,dtype=fnp.float32)),\n"
         "            2*RANK*RANK+RANK*PREFIX)"),
        ("def _exercise(fnp, ns, ctx):", "def _exercise(fnp, ns, ctx, audit):"),
        ("    sections, rows, rejected, refs = [], [], [], []",
         "    sections, rows, rejected, refs = (audit[k] for k in ('sections','rows','rejected','refs'))"),
        ("        before = int(ctx.flops_used)\n"
         "        value = own(function())\n"
         "        used = int(ctx.flops_used)-before\n"
         "        require(used>=0 and (expected is None or used==expected), 'Exact actual section tariff: '+name)\n"
         "        sections.append(dict(name=name,flops=used,expected_flops=expected))\n"
         "        return value",
         "        before = int(ctx.flops_used)\n"
         "        audit['active_section'] = name\n"
         "        try:\n"
         "            value = own(function())\n"
         "        except BaseException as error:\n"
         "            sections.append(dict(name=name,flops=int(ctx.flops_used)-before,\n"
         "                expected_flops=expected,function_completed=False,error_type=type(error).__name__))\n"
         "            raise\n"
         "        used = int(ctx.flops_used)-before\n"
         "        sections.append(dict(name=name,flops=used,expected_flops=expected,function_completed=True,\n"
         "            tariff_matched=used>=0 and (expected is None or used==expected)))\n"
         "        require(used>=0 and (expected is None or used==expected), 'Exact actual section tariff: '+name)\n"
         "        audit['active_section'] = None\n"
         "        return value"),
        (_span(original,'arithmetic'),ARITHMETIC),
        (_span(original,'derive_main'),"def derive_main():\n    return V2_MAIN_SOURCE, V2_MAIN_PROOF\n"),
    )
    for before,after in edits:
        source = _change(source,before,after,changes)
    _inverse(source,original,changes)
    proof = dict(owner_sha256=V1_WORKER_SHA,whole_V1_worker_inverse=True,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),main=main_proof,
        numerical_fragment_unchanged=True,fixture_values_guards_tolerances_unchanged=True,
        safe_fixture_values_unchanged=True,safe_square_fill_metadata_normalized_by_paid_copy=True,
        additional_safe_fixture_copy_flops=147456,
        final_singleton_all_flops=1,virtual_envelope_scalar_flops=21,
        failed_call_bill_is_not_a_qualified_context=True,numerical_execution=False)
    return source,proof,main,main_proof


_source,SOURCE_ADAPTER_PROOF,V2_MAIN_SOURCE,V2_MAIN_PROOF = derive_source()
_api = dict(__file__=str(Path(__file__).resolve()),__name__='_lazy_tiny_v2_inert',
    V2_MAIN_SOURCE=V2_MAIN_SOURCE,V2_MAIN_PROOF=V2_MAIN_PROOF,
    TinyCallFailure=TinyCallFailure,_error_and_release=_error_and_release)
exec(compile(_source,str(Path(__file__).resolve())+':whole-V1-inverse','exec'),_api)
_api['PINS'].update(V1_SOURCES)
_api['PINS'].update({
    _api['_BASE']['OSITE']/'flopscope/_flops.py':
        'bbdad83348b7ce7b63c9d8da66512c40c69e055c653d24687e62132bd34220c2',
    _api['_BASE']['OSITE']/'flopscope/_accumulation/_reduction.py':
        'a60b2ef74d568c1a1db5ad0cc06edcdea438efa0fdec7830b17bf439bdf65536',
})
for _name in ('main','admit','owner_map','helper_source','primitive_namespace','derive_main',
              'PINS','numerical_modules','sha','read','write','verify_call','call_records',
              'compare','join_price','consumer_price','CALL_BUDGET','SCHEMA','RANK','ROWS',
              'RESPONSE','LIVE','PREFIX','JOIN'):
    globals()[_name] = _api[_name]
del _source


if __name__=='__main__':
    raise SystemExit(main())
