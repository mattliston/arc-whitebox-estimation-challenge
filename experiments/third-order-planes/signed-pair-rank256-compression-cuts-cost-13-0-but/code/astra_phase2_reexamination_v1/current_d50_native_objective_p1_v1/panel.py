"""Fixed P1 independent14: original native lifecycle and unchanged NC scorer.

No fitting or candidate choice. P1's passive all-layer observer is retained.
Changed P is allowed to change every later output and its own compute bill.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
TL = NEW/'current_d50_terminal_linear_v2'
BD = NEW/'birth_direction_fit_v1'
ROOT = HERE/'panel_v1'
PROTOCOL_SHA = '3985ac8107375b88c5452b455033ea2f13c3d166a3a5f3ed5de645962e1e5079'
P1_PROTOCOL_SHA = '2141e2d4f025522a7cbea55fbd5ff10a66c313feacb1b4bd49a13f75a4b9af35'
NATIVE_P1_SHA = 'fad59d76d82baf40c823e0468b7304a909b719f8bc4a8c1f5c4e7118a4f9d9e8'
NATIVE_SHA = '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
BUILD_SHA = '877990bc84cb85431ff98f83e1e7012517c8b7914a1e0b39b29bf7d0b776b0c1'
SCORE_SHA = '4f60a8def29f1a9326d68d5c25bbe79b3f26289af8cf46de81609fb91cf055e5'
SCORE_WATCH_SHA = 'c9d015f7ed6b1da7db58bcf237582874042dd2d1e3315e72e7c55b5f1b9afa34'
SCORE_SOURCE_SHA = 'b7a2a60bf3607d6d68bb5c7c00cd0953aba0508f171628f6e06efae6f4466f27'
CHARTS_SHA = '443c1e7ddcc4e160be248e2a47d5f06b342013ca52037d001e63fbba44a19f6a'
SCORE_WATCH_SOURCE_SHA = 'adfcad6c846f9fac3bb5317d4a4ef9dd7f9c8728c929aa64c5f2060358859f04'
WATCH_SHA = '05ca75985c285a1c220690eca3d8884cfb76536cb3708defe7f3d80bdf5e3963'
TL_PANEL_SHA = '0a004bb746a0cdee4345a1b7e5621c2048ef8cfe8e6b46dd2036adb2b581da19'
BD_PANEL_SHA = 'e5f16497f4c8c825f9a203dac3e8c0c3ceac218731605c7c818ae5e8a237268e'
CANDIDATE = HERE/'build_v1/estimator.py'
CANDIDATE_SHA = '0deb237350e753220b575796c0278f86884b8461bfeec4ab53eae9c15522c435'
CONTROL = BD/'panel_v2/terminal'
CONTROL_SHA = '6ad5687a0baf3265e2c663b2d0ea453262f57fa6a84474fd2fbac9ad1211aade'
CONTROL_SCORE = BD/'panel_v2/score'
CONTROL_SCORE_SHA = '767cf92415a0391f7b7596c9681adede144bd10b3e5d36f354ec2dce7ce52c1b'
INITIALIZATION = 1383913786
SCHEMA = 'current_d50_native_objective_p1_independent14_v1'
FLAGS = ('all16_domains_qualified','all47_selectors_qualified','own_complete_bills',
         'all_outputs_and_bills_sealed_before_scoring','repeat_byte_and_bill_exact')
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load(path, name, expected):
    require(sha(path) == expected, 'Frozen source '+str(path))
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def function_source(path, name):
    source = Path(path).read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name == name)
    return ''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])


def replace_exact(source, edits):
    changed = source
    for before,after in edits:
        require(changed.count(before) == 1, 'Unique declared source splice')
        changed = changed.replace(before,after)
    inverse = changed
    for before,after in reversed(edits):
        require(inverse.count(after) == 1, 'Unique declared inverse splice')
        inverse = inverse.replace(after,before)
    require(inverse == source, 'Every remaining source statement preserved')
    return changed


def prerequisites(args):
    require(args.protocol_sha256 == PROTOCOL_SHA and args.terminal_score_sha256 == SCORE_SHA
            and sha(__file__) == args.source_sha256, 'Fixed panel source/protocol/advancing result')
    require(not any(k in sys.modules for k in ('numpy','torch','flopscope')), 'Metadata-only admission')
    require(sys.flags.optimize == 0 and all(os.environ.get(k) == '1' for k in BLAS)
            and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
            and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0', 'Fixed launch frame')
    n = load(HERE/'native_v2.py','p1_panel_native_owner',NATIVE_P1_SHA)
    tl = n.legacy()
    pins = {}; pin = tl.pinner(pins)
    fit = n.admission(BUILD_SHA,NATIVE_P1_SHA,pin)
    require(fit['candidate'] == CANDIDATE and fit['candidate_sha256'] == CANDIDATE_SHA,
            'Exact single trained and literal-only built P1')
    for path,digest in ((Path(__file__),args.source_sha256),(HERE/'PANEL_PROTOCOL.md',PROTOCOL_SHA),
        (HERE/'PROTOCOL.md',P1_PROTOCOL_SHA),(HERE/'judge_v2.py',SCORE_SOURCE_SHA),
        (HERE/'native_watch_v2.py',SCORE_WATCH_SOURCE_SHA),(TL/'panel.py',TL_PANEL_SHA),
        (BD/'panel_v2.py',BD_PANEL_SHA),(HERE/'judge.py',CHARTS_SHA),(HERE/'watch_panel.py',WATCH_SHA)):
        pin(path,digest)
    score = tl.sealed(HERE/'score2_v1',SCORE_SHA,pin)
    require(score['schema'] == 'current_d50_native_objective_p1_score_v2'
            and all(score.get(k) is True for k in ('integrity_passed','advances_to_independent14',
                'objective_improved','wins_gate','all_predictions_verified_before_truth'))
            and len(score['records']) == 40 and score['adjusted_wins'] >= 24
            and score['mean_adjusted_ratio'] < 1 and score['candidate_sha256'] == CANDIDATE_SHA
            and score['requested_seals']['build_sha256'] == BUILD_SHA
            and score['requested_seals']['source_sha256'] == SCORE_SOURCE_SHA
            and score['requested_seals']['native_source_sha256'] == NATIVE_P1_SHA,
            'Completed actual-objective native40, not a selected surrogate')
    tl.bind(score['source_bindings'],pin); tl.bind(score['artifacts'],pin)
    watch = tl.sealed(HERE/'score2_watch_v1',SCORE_WATCH_SHA,pin)
    require(watch['stage'] == 'score' and watch['child_result_sha256'] == SCORE_SHA
            and watch['child_completion_sha256'] == pin(HERE/'score2_v1/completion.json')
            and watch['child_execution_completed'] and watch['whole_owned_tree_retired']
            and watch['source_bindings'][str(HERE/'native_watch_v2.py')] == SCORE_WATCH_SOURCE_SHA,
            'Completed independent score resource watch')
    tl.bind(watch['source_bindings'],pin); tl.bind(watch['artifacts'],pin)
    panel = load(BD/'panel_v2.py','p1_panel_preserved_fixtures',BD_PANEL_SHA)
    h,p,old,v2 = panel.modules()
    old.runtime(h)
    prior_context = panel.prerequisites(h,p,old,v2,panel.ADVANCE_SHA)
    control,seals,_ = panel.verify_acquisition(h,p,old,v2,panel.ADVANCE_SHA,'terminal',prior_context)
    original_pins,fixtures,rows,_,_,_ = prior_context
    for mapping in (original_pins,fixtures,seals):
        tl.bind(mapping,pin)
    pin(CONTROL/'result.json',CONTROL_SHA)
    outputs = panel.verifier(h,p,v2,fixtures,rows)(control,CONTROL,p.RANKS,pin)
    require(control['candidate_sha256'] == tl.BANK_SHA and control['initialization_flops'] == INITIALIZATION
            and len(control['trials']) == 15, 'Archived actual current-d50 independent14')
    old_score = tl.sealed(CONTROL_SCORE,CONTROL_SCORE_SHA,pin)
    require(old_score['schema'] == panel.SCHEMA and old_score['mode'] == 'score'
            and old_score['qualification_passed'] and old_score['candidate_sha256'] == tl.BANK_SHA
            and old_score['acquisition_sha256']['terminal'] == CONTROL_SHA
            and [r['rank'] for r in old_score['per_role']['candidate']] == list(p.RANKS),
            'Original exact same-frame scored comparator, no target IO')
    require(all(sha(path) == digest for path,digest in pins.items()), 'Admission end seals')
    return dict(fit,n=n,tl=tl,h=h,p=p,old=old,v2=v2,panel=panel,pins=pins,fixtures=fixtures,
        rows=rows,control=control,control_outputs=outputs,control_score=old_score,fit=fit,
        role='terminal',ranks=p.RANKS)


def native_source(context):
    source,base_digest = context['n'].derived_main()
    start = source.index("    manifest=json.loads((STORE/'manifest.json').read_text())")
    stop = source.index('    args.out.mkdir()\n',start)
    changed = replace_exact(source,[
        (source[start:stop],
         "    rows={r['rank']:dict(weights=r['weights_path'],mlp_seed=r['mlp_seed']) for r in PANEL_ROWS}\n"
         "    inputs.update({v['path']:v['sha256'] for v in PANEL_FIXTURES.values()})\n"),
        ("args.out=HERE/f'native2_{args.role}_v1'","args.out=ROOT/'terminal'"),
        ("    write(args.out/'started.json',result)\n",
         "    result.update(PANEL_HEADER)\n"
         "    write(args.out/'started.json',result)\n"
         "    result['started_sha256']=sha(args.out/'started.json')\n")])
    return changed,base_digest


def acquire(args,context):
    n,tl = context['n'],context['tl']
    os.chdir(NEW)
    require(not ROOT.is_symlink(), 'Nonsymlink exclusive panel root')
    ROOT.mkdir(exist_ok=True)
    source,base_digest = native_source(context)
    old = n.old()
    previous = sys.modules.get('common')
    sys.modules['common'] = old.module(TL/'common.py','p1_panel_passive_common')
    try:
        capture = old.module(TL/'native.py','p1_panel_passive_observer')
    finally:
        if previous is None:
            sys.modules.pop('common',None)
        else:
            sys.modules['common'] = previous
    base = old.module(BD/'native_v2.py','p1_panel_preserved_native_loop')
    def inputs(candidate,native_args):
        require(candidate == CANDIDATE and native_args.source_sha256 == args.source_sha256
                and native_args.protocol_sha256 == PROTOCOL_SHA and native_args.build_sha256 == BUILD_SHA,
                'No alternate candidate/native invocation')
        return dict(context['pins']),context
    def check(result,trial,values,packet,helper,np,ctx):
        n.check_call(result,trial,values,packet,helper,np,ctx)
        require(values.tobytes()[:4096] == context['control_outputs'][trial['call']][:4096],
                'Exact fixed layer0 only; all later P outputs may change')
    def finish(result,ctx):
        old.finish_native(result,ctx)
        require(result['threadpools'] == context['control']['threadpools'], 'Same reported backend frame')
        result.update(sources_unchanged=True,repeat_byte_and_bill_exact=True,layer0_byte_exact=True)
    namespace = dict(vars(base),__file__=str(Path(__file__).resolve()),HERE=HERE,PARENT=NEW,ROOT=ROOT,
        native_inputs=inputs,observed_predict=tl.observed_predict,check_call=check,finish_native=finish,
        CAPTURE_HELPER=capture,MAIN_SHA=base_digest,SCHEMA=SCHEMA,BANK_SOURCE=tl.BANK,
        NATIVE_SHA=NATIVE_SHA,CONTROL_SHA=CONTROL_SHA,VERSIONS=tl.VERSIONS,
        PANEL_ROWS=context['rows'],PANEL_FIXTURES=context['fixtures'],
        PANEL_HEADER=dict(mode='acquire',fixture_bindings=context['fixtures'],
            terminal_score_sha256=SCORE_SHA,terminal_score_watch_sha256=SCORE_WATCH_SHA,
            build_result_sha256=BUILD_SHA,control_result_sha256=CONTROL_SHA,no_refit=True,
            sources_unchanged=False,pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),
            blas_env=dict.fromkeys(BLAS,'1'),executable=sys.executable,no_warmup=True,
            truth_files_accessed=False,native_adapter_sha256=hashlib.sha256(source.encode()).hexdigest(),
            all_outputs_and_bills_sealed_before_scoring=False))
    exec(compile(source,str(__file__)+':preserved_native','exec'),namespace)
    saved = sys.argv
    try:
        sys.argv = [str(__file__),'--run','--role','terminal','--source-sha256',args.source_sha256,
            '--protocol-sha256',PROTOCOL_SHA,'--build-sha256',BUILD_SHA]
        return namespace['main']()
    finally:
        sys.argv = saved


def verifier(context):
    require(sha(TL/'panel.py') == TL_PANEL_SHA, 'Frozen canonical verifier adaptation')
    source = function_source(TL/'panel.py','verifier')
    namespace = dict(globals())
    exec(compile(source,str(__file__)+':preserved_panel_verifier','exec'),namespace)
    return namespace['verifier'](context)


def verify_acquisition(args,context):
    tl,p = context['tl'],context['p']
    pin = tl.pinner(context['pins'])
    prospective = dict(context['pins'])
    folder = ROOT/'terminal'
    result = tl.sealed(folder,args.acquisition_sha256,pin)
    source,base_digest = native_source(context)
    require(result['schema'] == SCHEMA and result['mode'] == 'acquire' and result['role'] == 'terminal'
            and all(result.get(k) is True for k in (*FLAGS,'sources_unchanged','no_refit',
                'paid_arithmetic_and_lifecycle_unchanged','passive_byte_observation','layer0_byte_exact'))
            and not result['truth_accessed'] and not result['truth_files_accessed']
            and result['candidate_sha256'] == CANDIDATE_SHA and result['terminal_score_sha256'] == SCORE_SHA
            and result['build_result_sha256'] == BUILD_SHA and result['control_result_sha256'] == CONTROL_SHA
            and result['native_original_sha256'] == NATIVE_SHA and result['native_main_sha256'] == base_digest
            and result['native_adapter_sha256'] == hashlib.sha256(source.encode()).hexdigest()
            and all(result['source_bindings'].get(path) == digest for path,digest in prospective.items()),
            'Complete immutable new P1 acquisition with its own bills')
    outputs = verifier(context)(result,folder,p.RANKS,pin)
    require(result['versions'] == tl.VERSIONS and result['threadpools'] == context['control']['threadpools']
            and result['initialization_flops'] == INITIALIZATION, 'Exact runtime/init ownership')
    charts = load(HERE/'judge.py','p1_panel_original_charts',CHARTS_SHA)
    charts.charts(result,15)
    for call,(trial,prior) in enumerate(zip(result['trials'],context['control']['trials'])):
        require(trial['rank'] == prior['rank'] and trial['includes_initialization'] == prior['includes_initialization']
                and outputs[call][:4096] == context['control_outputs'][call][:4096],
                'Canonical input/cold history and exact layer0; independent own costs')
    pin(folder/'started.json',result['started_sha256'])
    started = read(folder/'started.json')
    require(started['source_bindings'] == result['source_bindings']
            and started['fixture_bindings'] == context['fixtures'] and started['candidate_sha256'] == CANDIDATE_SHA
            and started['no_warmup'] and started['nice'] >= 10, 'Exclusive native start ownership')
    watch_digest = pin(HERE/'watch_panel.py',WATCH_SHA)
    watch = tl.sealed(ROOT/'terminal_watch',args.acquisition_watch_sha256,pin)
    require(watch['schema'] == 'current_d50_native_objective_p1_independent14_watch_v1'
            and watch['stage'] == 'native' and watch['sources_unchanged']
            and watch['completed_bill_count'] == 15 and watch['child_result_sha256'] == args.acquisition_sha256
            and watch['child_completion_sha256'] == pin(folder/'completion.json')
            and watch['prerequisites']['terminal_score_sha256'] == SCORE_SHA
            and watch['source_bindings'][str(HERE/'watch_panel.py')] == watch_digest
            and watch['source_bindings'][str(Path(__file__).resolve())] == args.source_sha256
            and watch['child_execution_completed'] and watch['whole_owned_tree_retired'],
            'Independent complete 15-call resource watch before truths')
    tl.bind(watch['source_bindings'],pin); tl.bind(watch['artifacts'],pin)
    return result,outputs


def score(args,context,folder,result,guard):
    require(sha(TL/'panel.py') == TL_PANEL_SHA, 'Original independent4M score convention')
    source = function_source(TL/'panel.py','score')
    changed = replace_exact(source,[(
        '        earlier15_rows_byte_exact=True,own_complete_F_and_opcounts_exact=True,\n',
        '        own_complete_bills=True,all16_domains_qualified=True,all47_selectors_qualified=True,\n')])
    namespace = dict(globals())
    exec(compile(changed,str(__file__)+':preserved_NC_score','exec'),namespace)
    return namespace['score'](args,context,folder,result,guard)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--acquire',action='store_true'); mode.add_argument('--score',action='store_true')
    for key in ('source','protocol','terminal-score','acquisition','acquisition-watch'):
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.acquire and not args.score:
        parser.print_help(); return 0
    require(all(isinstance(getattr(args,key),str) and len(getattr(args,key)) == 64
                and set(getattr(args,key)) <= set('0123456789abcdef')
                for key in ('source_sha256','protocol_sha256','terminal_score_sha256')), 'Explicit reviewed seals')
    require((args.score and all(isinstance(x,str) and len(x) == 64 and set(x) <= set('0123456789abcdef')
                for x in (args.acquisition_sha256,args.acquisition_watch_sha256)))
            or (args.acquire and args.acquisition_sha256 is None and args.acquisition_watch_sha256 is None),
            'Only score binds prior acquisition; no alternate fit or output')
    def expired(signum,frame):
        raise TimeoutError('Fixed independent14 admission/score deadline')
    signal.signal(signal.SIGALRM,expired); signal.alarm(120)
    context = prerequisites(args)
    if args.acquire:
        signal.alarm(0)
        return acquire(args,context)
    ROOT.mkdir(exist_ok=True)
    folder = ROOT/'score'; folder.mkdir(exist_ok=False)
    h = context['h']; began = time.monotonic()
    result = dict(schema=SCHEMA,mode='score',passed=False,completed=False,qualification_passed=False,
        integrity_passed=False,hashes_unchanged=False,sources_unchanged=False,requested_seals=vars(args),
        terminal_score_sha256=SCORE_SHA,truth_files_accessed=False,no_refit=True,production_qualified=False,
        official_score=False,milestone_achieved=False,retain_local_component=False,
        started_utc=datetime.now(timezone.utc).isoformat())
    h.write_json(folder/'started.json',result)
    def guard():
        require(time.monotonic()-began < 120 and h.peak_rss_bytes() < 2_000_000_000, 'Score120s/2GB')
    try:
        score(args,context,folder,result,guard)
    except BaseException:
        result.update(passed=False,completed=False,qualification_passed=False,integrity_passed=False,
            hashes_unchanged=False,sources_unchanged=False,retain_local_component=False,error=traceback.format_exc())
    finally:
        signal.alarm(0)
        result.update(completed_utc=h.utc(),wall_s=time.monotonic()-began,peak_rss_bytes=h.peak_rss_bytes())
        digest = h.write_json(folder/'result.json',result)
        h.write_json(folder/'completion.json',dict(passed=result['passed'],completed=result['completed'],
            integrity_passed=result['integrity_passed'],result_sha256=digest,
            retain_local_component=result['retain_local_component']))
    print(json.dumps({k:result.get(k) for k in ('passed','completed','retain_local_component',
        'adjusted_candidate_over_parent','error')},allow_nan=False),flush=True)
    return 0 if result['completed'] and result['integrity_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
