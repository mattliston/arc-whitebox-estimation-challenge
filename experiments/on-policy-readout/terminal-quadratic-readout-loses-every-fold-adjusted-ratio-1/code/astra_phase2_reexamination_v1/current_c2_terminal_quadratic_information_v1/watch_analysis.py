"""Inert external unit/information supervisor; no numerical or target decoder."""
import hashlib
import math
from pathlib import Path
import re
import shlex
import types

HERE = Path(__file__).resolve().parent
OWNER = HERE.parent/'current_comp_shared_old384_v1/watch_score_probe.py'
OWNER_SHA = 'ce125eeb2bf774f9647b925018ba3ea1bf4e7b0a4e18277d6fab2d1c09d7d6a5'
LIMITS = {
    'unit':dict(cap=120.,rss=2_000_000_000,arms=('unit',),context_wall=60.,
        context_flops=(12_000_000_000,),total_flops=12_000_000_000,
        folder='analysis_unit_v1',watch='analysis_unit_watch_v1',
        schema='current_c2_terminal_quadratic_analysis_unit_v1'),
    'information':dict(cap=600.,rss=8_000_000_000,arms=('fold0','fold1','fold2','fold3','fold4','score'),
        context_wall=120.,context_flops=(350_000_000_000,)*5+(10_000_000,),total_flops=2_000_000_000_000,
        folder='information_v1',watch='information_watch_v1',schema='current_c2_terminal_quadratic_information_v1'),
}


def require(ok, message):
    if not ok: raise RuntimeError(message)


def foreign_servers(rows, allowed):
    """Retain the actual underscore-server classifier repair, without imports."""
    found = []
    for row in rows:
        if row['pid'] in allowed: continue
        try: argv = shlex.split(row['command'])
        except ValueError: argv = row['command'].split()
        if argv and re.fullmatch(r'(?:python|pypy)[0-9.]*',Path(argv[0]).name.lower()) and any(
            word == 'flopscope_server' or word.startswith('flopscope_server.') for word in argv[1:]):
            found.append(row)
    return found


def worker_args(args):
    return types.SimpleNamespace(mode=args.mode,source_sha256=args.worker_sha256,
        watcher_sha256=args.source_sha256,protocol_sha256=args.protocol_sha256,
        watch_spec_sha256=args.spec_sha256,math_sha256=args.math_sha256,
        capture_manifest=args.capture_manifest,capture_manifest_sha256=args.capture_manifest_sha256)


def child_command(args):
    command = [str(HERE.parent.parent/'data_gate/.venv/bin/python'),'-B','-u',str(HERE/'analysis.py'),
        '--run','--mode',args.mode,'--source-sha256',args.worker_sha256,
        '--watcher-sha256',args.source_sha256,'--protocol-sha256',args.protocol_sha256,
        '--watch-spec-sha256',args.spec_sha256,'--math-sha256',args.math_sha256]
    if args.mode == 'information':
        command += ['--capture-manifest',str(Path(args.capture_manifest).resolve()),
            '--capture-manifest-sha256',args.capture_manifest_sha256]
    return command


def retired_record(item, index, folder, mode, sha, read):
    """Only actual sealed released numerical contexts advance progress."""
    limits = LIMITS[mode]
    require(index < len(limits['arms']) and item.get('call') == index
        and item.get('arm') == limits['arms'][index], 'Exact unit/fivefold+score retired roster')
    path = folder/f'call_{index:02d}_complete.json'; digest = sha(path); complete = read(path)
    require(digest == item['complete_sha256'] and complete['passed'] is True and complete['completed'] is True
        and complete['call'] == index and complete['arm'] == limits['arms'][index]
        and complete['context_retired'] is True, 'Real completed context, never a heartbeat')
    for suffix,key in (('bill','bill_sha256'),('lifecycle','lifecycle_sha256')):
        require(sha(folder/f'call_{index:02d}_{suffix}.json') == complete[key], 'Sealed actual bill/lifecycle')
    if 'diagnostic_sha256' in complete:
        require(sha(folder/f'call_{index:02d}_diagnostics.json') == complete['diagnostic_sha256'], 'Declared diagnostic seal')
    outputs = complete['output_bindings']
    require(isinstance(outputs,dict) and bool(outputs), 'Output bytes sealed before retirement event')
    for p,d in outputs.items():
        p = Path(p)
        require(p.resolve().is_relative_to(folder.resolve()) and p.stat().st_size <= 16*1024**2
            and sha(p) == d, 'Bounded same-context sealed output, not a target path')
    bill = read(folder/f'call_{index:02d}_bill.json'); lifecycle = read(folder/f'call_{index:02d}_lifecycle.json')
    require(isinstance(bill['flops'],(int,float)) and math.isfinite(bill['flops'])
        and 0 < bill['flops'] <= limits['context_flops'][index]
        and isinstance(bill['wall_s'],(int,float)) and math.isfinite(bill['wall_s'])
        and 0 <= bill['wall_s'] < limits['context_wall'], 'Actual complete per-context F/wall limits')
    require(lifecycle['context_released'] is True and lifecycle['closed_context_bill_unchanged_by_reset'] is True,
        'Actual release/reset evidence')
    return dict(path=str(path),sha256=digest,call=index,arm=limits['arms'][index],
        bill_sha256=complete['bill_sha256'],lifecycle_sha256=complete['lifecycle_sha256'],
        output_bindings=outputs,flops=bill['flops'],wall_s=bill['wall_s'])


def derive_source():
    original = OWNER.read_text()
    require(hashlib.sha256(original.encode()).hexdigest() == OWNER_SHA, 'Actual completed saved-output monitor owner')
    start = original.index("        scorer = load(HERE/'score_probe.py'")
    end = original.index('        env = dict(os.environ',start)
    old_admission = original[start:end]
    start = original.index("        flags = ('integrity_passed'")
    end = original.index('        union = {}',start)
    old_validation = original[start:end]
    old_input = """    inputs = {**h.MOCK_PINS,HERE/'probe_v1/result.json':args.probe_sha256,
        HERE/'probe_watch_v1/result.json':args.probe_watch_sha256,HERE/'probe.py':args.probe_source_sha256,
        HERE/'watch_probe.py':args.probe_watch_source_sha256,HERE/'PROBE_PROTOCOL.md':args.probe_protocol_sha256,
        HERE/'PROBE_WATCH_SPEC.md':args.probe_watch_spec_sha256}"""
    old_own = """    own = {HERE/'watch_score_probe.py':args.source_sha256,HERE/'score_probe.py':args.worker_sha256,
        HERE/'SCORE_PROBE_PROTOCOL.md':args.protocol_sha256,TRANSPORT:TRANSPORT_SHA,**h.MON_PINS}"""
    new_validation = """        require(r['mode'] == args.mode and r['retired_call_count'] == len(limits['arms'])
            and result['retired_calls'] == len(limits['arms'])
            and r['full_model_calls'] == r['parameter_updates'] == 0
            and r['own_complete_bills'] is True and r['all_numerical_array_owners_retired'] is True
            and r['all_parameter_and_estimator_owners_retired'] is True
            and r['hashes_unchanged'] is True and r['sources_unchanged'] is True
            and r['submission_authorized'] is False and r['automatic_next_stage'] is False,
            'One bounded unit/information worker; no model or deployment authority')
        require(r['research_flops'] == sum(x['flops'] for x in result['markers'])
            and 0 < r['research_flops'] <= limits['total_flops'], 'All context costs included, never a component-only tariff')
        require(scorer.verify_result(r,tick) is True, 'Independent worker metadata contract; no numerical re-execution')
        require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules), 'Final verification remains stdlib-only')
        if args.mode == 'unit':
            require(r['truth_accessed'] is False, 'Unit is wholly target-free')
        else:
            require(r['truth_accessed'] is True and type(r['information_promising']) is bool
                and r['capture_manifest'] == str(Path(args.capture_manifest).resolve())
                and r['capture_manifest_sha256'] == args.capture_manifest_sha256,
                'Actual admitted fixed captures; scientific FAIL remains operationally complete')
        for group in ('source_bindings','input_bindings'):
            require(all(r[group].get(path) == digest for path,digest in context[group].items()),
                'Final worker preserves the exact pre-numerical admission union')
"""
    edits = [
        ("KEYS = ('source','worker','protocol','probe-source','probe-watch-source',\n        'probe-protocol','probe-watch-spec','probe','probe-watch')",
         "KEYS = ('source','worker','protocol','spec','math')",1),
        ("    parser.add_argument('--run',action='store_true'); parser.add_argument('--numerical-slot-clear',action='store_true')",
         "    parser.add_argument('--run',action='store_true'); parser.add_argument('--numerical-slot-clear',action='store_true')\n    parser.add_argument('--mode',choices=('unit','information'))\n    parser.add_argument('--capture-manifest')\n    parser.add_argument('--capture-manifest-sha256')",1),
        ('    began = previous = time.monotonic()',
         "    require(args.mode in LIMITS, 'Explicit fixed unit/information mode')\n    limits = LIMITS[args.mode]; CAP,RSS = limits['cap'],limits['rss']\n    require((args.mode == 'unit' and args.capture_manifest is None and args.capture_manifest_sha256 is None) or\n        (args.mode == 'information' and args.capture_manifest and re.fullmatch('[0-9a-f]{64}',args.capture_manifest_sha256 or '')),\n        'Actual capture manifest path and SHA required only for information')\n    if args.mode == 'information':\n        manifest = Path(args.capture_manifest)\n        require(manifest.is_file() and not manifest.is_symlink() and manifest.resolve().is_relative_to(HERE),\n            'Actual regular H15 descendant manifest; no hypothetical version path')\n    began = previous = last_progress = time.monotonic()",1),
        ("and all(os.environ.get(k) == '1' for k in BLAS), 'Exclusive original OFFPY nice10/sixBLAS1 scorer'",
         "and all(os.environ.get(k) == '1' for k in BLAS) and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'\n        and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0', 'Exclusive original OFFPY nice10/sixBLAS1/no fallback'",1),
        (old_own,"""    own = {HERE/'watch_analysis.py':args.source_sha256,HERE/'analysis.py':args.worker_sha256,
        HERE/'ANALYSIS_PROTOCOL.md':args.protocol_sha256,HERE/'ANALYSIS_WATCH_SPEC.md':args.spec_sha256,
        HERE/'analysis_math.py':args.math_sha256,OWNER:OWNER_SHA,TRANSPORT:TRANSPORT_SHA,**h.MON_PINS}""",1),
        (old_input,"""    inputs = dict(h.MOCK_PINS)
    if args.mode == 'information':
        inputs[Path(args.capture_manifest).resolve()] = args.capture_manifest_sha256""",1),
        ("    folder,out = HERE/'score_probe_v1',HERE/'score_probe_watch_v1'", "    folder,out = HERE/limits['folder'],HERE/limits['watch']",1),
        ("schema='current_comp_shared_old384_score_probe_watch_v1'", "schema=limits['schema'][:-3]+'_watch_v1',mode=args.mode",1),
        ('whole_cap_s=CAP,combined_rss_limit_bytes=RSS,maximum_combined_rss_bytes=0,',
         "whole_cap_s=CAP,progress_cap_s=120.,combined_rss_limit_bytes=RSS,maximum_combined_rss_bytes=0,\n        retired_calls=0,markers=[],roster=list(limits['arms']),capture_manifest_sha256=args.capture_manifest_sha256,\n        capture_manifest=None if args.mode == 'unit' else str(Path(args.capture_manifest).resolve()),",1),
        ('    child=log=stream=None; swap0=None; samples=0', "    child=log=reader=stream=None; swap0=None; samples=0; pending=''",1),
        ("        require(time.monotonic()-began < CAP,'Whole120s through source admission and final publication')",
         "        now = time.monotonic()\n        require(now-began < CAP and now-last_progress < 120., 'Whole mode cap and120s actual retirement progress')",1),
        ('        nonlocal previous,swap0,samples', '        nonlocal previous,swap0,samples,last_progress,pending',1),
        ('        absolute = min(previous+1.,began+CAP)', '        absolute = min(previous+1.,began+CAP,last_progress+120.)',1),
        ("        require(not classifier(snapshot['rows'],scope.pop('allowed')),'No foreign numerical worker')",
         "        allowed = scope.pop('allowed')\n        require(not classifier(snapshot['rows'],allowed) and not foreign_servers(snapshot['rows'],allowed),\n            'No foreign numerical worker or underscore-spelled flopscope server')",1),
        ("'Zero added swap and combined owned-tree1GB'", "'Zero added swap and complete mode-specific tree limit'",1),
        ("        require(snapshot['commands_reaped'] and not mon.PENDING,'Every observation group retired')\n        deadline()",
         """        require(snapshot['commands_reaped'] and not mon.PENDING,'Every observation group retired')
        if reader is not None:
            pending += reader.read()
            require(len(pending) < 4*1024**2, 'Bounded retired-marker stream')
            while '\\n' in pending:
                line,pending = pending.split('\\n',1)
                try: item = json.loads(line)
                except (ValueError,TypeError): continue
                if isinstance(item,dict) and item.get('event') == 'retired':
                    deadline()
                    marker = retired_record(item,result['retired_calls'],folder,args.mode,sha,h.read)
                    result['markers'].append(marker); result['retired_calls'] += 1
                    last_progress = time.monotonic()
        deadline()""",1),
        ('transport=snapshot[\'command_records\'],uncached_single_process_table=True)',
         "transport=snapshot['command_records'],retired_calls=result['retired_calls'],uncached_single_process_table=True)",1),
        (old_admission,"""        scorer = load(HERE/'analysis.py',args.worker_sha256,'current_c2_information_metadata_only')
        context = scorer.admission(worker_args(args),tick)
        require(isinstance(context,dict) and context['mode'] == args.mode, 'Exact stdlib mode admission')
        for group in ('source_bindings','input_bindings'):
            require(isinstance(context[group],dict), 'Flat direct pre-numerical admission maps')
            for path,digest in context[group].items():
                require(path not in result[group] or result[group][path] == digest, 'Consistent admission union')
                require(sha(path,tick) == digest, 'Actual source/input bytes before any worker')
                result[group][path] = digest
        require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules), 'Admission remains stdlib-only')
        result['watch_source_proof'] = WATCH_SOURCE_PROOF
        argv = child_command(args)
""",1),
        ("        log = (out/'worker.log').open('x')", "        log = (out/'worker.log').open('x'); reader = (out/'worker.log').open('r')",1),
        ("observe(); require(child.wait(timeout=1) == 0,'Scorer completed with exit0')",
         "observe(); require(child.wait(timeout=1) == 0 and result['retired_calls'] == len(limits['arms']),\n            'Worker exit0 and all actual declared context retirements')",1),
        ("r['schema'] == scorer.SCHEMA", "r['schema'] == limits['schema']",1),
        (old_validation,new_validation,1),
        ("require(isinstance(r[group],dict) and r[group],'Full actual child non-target inventory')",
         "require(isinstance(r[group],dict) and (bool(r[group]) or group == 'input_bindings'),'Full actual child non-target inventory')",1),
        ("if p in (HERE/'score_probe.py',HERE/'watch_score_probe.py',HERE/'SCORE_PROBE_PROTOCOL.md')",
         "if p in (HERE/'analysis.py',HERE/'watch_analysis.py',HERE/'ANALYSIS_PROTOCOL.md',HERE/'ANALYSIS_WATCH_SPEC.md',HERE/'analysis_math.py')",1),
        ("'Exact three current scorer/watch/protocol source owners'", "'Exact five current worker/watch/protocol/spec/math owners'",1),
        ("        for path,digest in {**own,**inputs}.items():\n            require(sha(path,tick) == digest,'Own sources and explicit acquisition inputs unchanged')",
         "        for group in ('source_bindings','input_bindings'):\n            for path,digest in result[group].items():\n                require(sha(path,tick) == digest,'Complete watch admission sources/inputs unchanged')",1),
        ("information_promising=r['information_promising'],\n            scientific_advancement=r['information_promising'],verified_child_nontarget_files=len(union),",
         "information_promising=bool(r.get('information_promising',False)),\n            scientific_advancement=bool(r.get('information_promising',False)),verified_child_nontarget_files=len(union),\n            retired_call_count=len(limits['arms']),research_flops=r['research_flops'],own_complete_bills=True,\n            all_numerical_array_owners_retired=True,all_parameter_and_estimator_owners_retired=True,",1),
        ('        for handle in (log,stream):', '        for handle in (log,reader,stream):',1),
        ("return bool(result['passed'] and time.monotonic()-began < CAP and time.monotonic()-previous <= 1.)",
         "now = time.monotonic()\n            return bool(result['passed'] and now-began < CAP and now-last_progress < 120. and now-previous <= 1.)",1),
    ]
    changed,spans = original,[]
    for before,after,count in edits:
        require(changed.count(before) == count, 'Exact prospective analysis-watch splice: '+before[:90])
        cursor = 0
        for _ in range(count):
            at = changed.index(before,cursor); spans.append((at,before,after))
            changed = changed[:at]+after+changed[at+len(before):]; cursor = at+len(after)
    inverse = changed
    for at,before,after in reversed(spans):
        require(inverse[at:at+len(after)] == after, 'Recorded-position inverse, no substring ambiguity')
        inverse = inverse[:at]+before+inverse[at+len(after):]
    require(inverse == original, 'Complete qualified source/collect/cleanup/sealed-tail owner inverse')
    return changed,dict(owner_sha256=OWNER_SHA,whole_source_inverse_verified=True,splices=len(spans),
        derived_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        original_completed_collect_scope_cleanup_retained=True,strict_1s_samples_and_publication_tail=True,
        real_retired_context_progress_only=True,no_array_or_truth_decoder=True)


def main():
    source,proof = derive_source()
    env = dict(__name__='_current_c2_information_watch_body',__file__=str(Path(__file__).resolve()),
        OWNER=OWNER,OWNER_SHA=OWNER_SHA,LIMITS=LIMITS,WATCH_SOURCE_PROOF=proof,foreign_servers=foreign_servers,
        worker_args=worker_args,child_command=child_command,retired_record=retired_record)
    exec(compile(source,'<qualified-unit-information-analysis-watch>','exec'),env)
    return env['main']()


if __name__ == '__main__':
    raise SystemExit(main())
