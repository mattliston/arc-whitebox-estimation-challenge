"""Inert fixed terminal-P65 independent14 acquisition and separate score.

One new native role, no fit/profiling/extra participant arithmetic. The archived
d50 role retains its own complete bills. Operational PASS and the strict two-
group scientific retention decision are separate. No target IO in acquisition.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
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
NEW, P2 = HERE.parent,HERE.parent.parent
BD = NEW/'birth_direction_fit_v1'
ROOT = HERE/'panel_v1'
NATIVE = BD/'native_v2.py'
NATIVE_SHA = '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
PANEL_SHA = 'e5f16497f4c8c825f9a203dac3e8c0c3ceac218731605c7c818ae5e8a237268e'
VALIDATOR_SHA = '69c8eeee8078feb2b12662f128ea873e5d67435182d4c30bd551a1369c84f69c'
PROTOCOL_SHA = '7e2c4d86de5d600b19134fd3127d0878d05a527cab811ec36b2ebca4b3c69f2f'
TL_PROTOCOL_SHA = 'febed4bdccc7c031cb846e6d97f695536ca98d3bef10d07de48f1418a91b6a58'
SCORE_SHA = '9e327c7e7f8df5471679eb7532a4531b5a763ed6b051d32408e0231b6ceda030'
SCORE_WATCH_SHA = '06a97b4215360af3446b1b166c78403ebf87d66ca0019ad6cd8eacaad6683e61'
TL_JUDGE_SHA = 'd755605c7a1773cf3fe740ee7a494435fc67efeb7377fc0f442b9e00f9277c9e'
TL_WATCH_SHA = 'b8b8d7563487c66e1424fbbc6dd34811bc8836cde1abbb5bca475f29227f7e44'
FIT_SHA = '65fe4bf467d088277773db68fb7e558bc2d4ea77a07c12abd430a4e2be90b443'
CANDIDATE = HERE/'fit_v1/estimator.py'
CANDIDATE_SHA = '03d78ee8182749e6c5bed07618420bd9b134b242236a079dddba14837c823641'
CONTROL = BD/'panel_v2/terminal'
CONTROL_SHA = '6ad5687a0baf3265e2c663b2d0ea453262f57fa6a84474fd2fbac9ad1211aade'
CONTROL_SCORE = BD/'panel_v2/score'
CONTROL_SCORE_SHA = '767cf92415a0391f7b7596c9681adede144bd10b3e5d36f354ec2dce7ce52c1b'
WATCH_SHA = '82919d983fbca86ffd418393f027a95d662ef1f18518613c831c1febccdeaaca'
INITIALIZATION = 1383913786
SCHEMA = 'current_d50_terminal_linear_independent14_v1'
FLAGS = ('earlier15_rows_byte_exact','own_complete_F_and_opcounts_exact',
         'all_outputs_and_bills_sealed_before_scoring','repeat_byte_and_bill_exact')
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')


def require(value,message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1 << 20),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load(path,name,expected):
    require(sha(path) == expected,'Frozen source '+str(path))
    spec = importlib.util.spec_from_file_location(name,path)
    require(spec is not None and spec.loader is not None,'Module specification')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prerequisites(args):
    """All source/receipt/weight/output authentication, never numerical truths."""
    require(args.protocol_sha256 == PROTOCOL_SHA and args.terminal_score_sha256 == SCORE_SHA
            and sha(__file__) == args.source_sha256,'Reviewed sole panel/source/advancing score')
    require(not any(k in sys.modules for k in ('numpy','torch','flopscope')),'Admission before numerical imports')
    require(sys.flags.optimize == 0 and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
            and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
            and all(os.environ.get(k) == '1' for k in BLAS),'Assertions/BLAS1/no fallback')
    panel = load(BD/'panel_v2.py','tl65_preserved_panel_v2',PANEL_SHA)
    h,p,old,v2 = panel.modules()
    old.runtime(h)
    tl = load(HERE/'native_validation.py','tl65_native_metadata_admission',VALIDATOR_SHA)
    pins = {}; pin = tl.pinner(pins)
    for path,digest in ((Path(__file__),args.source_sha256),(HERE/'PANEL_PROTOCOL.md',PROTOCOL_SHA),
            (HERE/'PROTOCOL.md',TL_PROTOCOL_SHA),(BD/'panel_v2.py',PANEL_SHA),
            (HERE/'native_validation.py',VALIDATOR_SHA),(NATIVE,NATIVE_SHA),
            (HERE/'judge_terminal.py',TL_JUDGE_SHA),(HERE/'watch_validation.py',TL_WATCH_SHA)):
        pin(path,digest)
    score = tl.sealed(HERE/'terminal_score_v1',SCORE_SHA,pin)
    require(score['schema'] == 'current_d50_terminal_linear_native40_score_v1'
            and all(score.get(k) is True for k in ('integrity_passed','advances_to_independent14',
                'objective_improved','wins_gate','all_predictions_verified_before_truth',
                'all_earlier15_outputs_byte_exact','all_own_complete_bills_verified'))
            and len(score['records']) == 40 and score['adjusted_wins'] >= 24
            and score['mean_adjusted_ratio'] < 1 and score['candidate_sha256'] == CANDIDATE_SHA
            and score['requested_seals']['fit_sha256'] == FIT_SHA
            and score['requested_seals']['source_sha256'] == TL_JUDGE_SHA
            and score['requested_seals']['native_source_sha256'] == VALIDATOR_SHA,
            'Completed fixed native40 admission, no refit/selected checkpoint')
    tl.bind(score['source_bindings'],pin); tl.bind(score['artifacts'],pin)
    watch = tl.sealed(HERE/'terminal_score_watch_v1',SCORE_WATCH_SHA,pin)
    require(watch['stage'] == 'score' and watch['child_result_sha256'] == SCORE_SHA
            and watch['child_completion_sha256'] == pin(HERE/'terminal_score_v1/completion.json')
            and watch['child_execution_completed'] and watch['whole_owned_tree_retired']
            and watch['source_bindings'][str(HERE/'watch_validation.py')] == TL_WATCH_SHA,
            'Completed independent terminal40 score resource watch')
    tl.bind(watch['source_bindings'],pin); tl.bind(watch['artifacts'],pin)
    fit = tl.authenticate_fit(FIT_SHA,pin)
    require(fit['candidate'] == CANDIDATE and fit['candidate_sha256'] == CANDIDATE_SHA,
            'Exact sole65-parameter candidate, other P/cstar/source frozen')
    pin(CONTROL/'result.json',CONTROL_SHA)
    context = panel.prerequisites(h,p,old,v2,panel.ADVANCE_SHA)
    control,seals,_ = panel.verify_acquisition(h,p,old,v2,panel.ADVANCE_SHA,'terminal',context)
    original_pins,fixtures,rows,_,_,_ = context
    for mapping in (original_pins,fixtures,seals):
        tl.bind(mapping,pin)
    blobs = panel.verifier(h,p,v2,fixtures,rows)(control,CONTROL,p.RANKS,pin)
    require(control['candidate_sha256'] == tl.BANK_SHA and control['initialization_flops'] == INITIALIZATION
            and len(control['trials']) == 15,'Exact qualified d50 independent14 plus repeat')
    old_score = tl.sealed(CONTROL_SCORE,CONTROL_SCORE_SHA,pin)
    require(old_score['schema'] == panel.SCHEMA and old_score['mode'] == 'score'
            and old_score['qualification_passed'] and old_score['candidate_sha256'] == tl.BANK_SHA
            and old_score['acquisition_sha256']['terminal'] == CONTROL_SHA
            and [r['rank'] for r in old_score['per_role']['candidate']] == list(p.RANKS),
            'Original same-frame d50 score ownership, no target bytes opened')
    require(all(sha(path) == digest for path,digest in pins.items()),'Admission end seals')
    return dict(h=h,p=p,old=old,v2=v2,panel=panel,tl=tl,pins=pins,fixtures=fixtures,rows=rows,
                control=control,control_outputs=blobs,control_score=old_score,fit=fit)


def native_source(context):
    """Preserve the reviewed panel/native numeric loop; change admission/checks only."""
    panel,h = context['panel'],context['h']
    original = panel.native_source(h)
    start = original.index('    if candidate==BANK:\n')
    end = original.index('    ranks=tuple(PANEL_RANKS)\n',start)
    edits = [(original[start:end],
              '    assert candidate==PANEL_CANDIDATE\n'
              '    inputs.update(PANEL_PINS)\n'),
        ("                assert trial['layer0_byte_exact'] and trial['archive_output_control_passed']\n",
         "                assert trial['layer0_byte_exact'] and trial['archive_output_control_passed']\n"
         "                assert trial['earlier15_rows_byte_exact']\n"
         "                assert trial['flops']==trial['parent_flops'] and trial['operation_count']==trial['parent_operation_count']\n"),
        ("        result['error']=traceback.format_exc()\n",
         "        result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,error=traceback.format_exc())\n")]
    changed = panel.replace_exact(original,edits,require)
    return changed,hashlib.sha256(original.encode()).hexdigest()


def acquire(args,context):
    h,p = context['h'],context['p']
    source,parent_adapter_sha = native_source(context)
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    ROOT.mkdir(exist_ok=True)
    require(ROOT.is_dir() and not ROOT.is_symlink(),'Fixed nonsymlink panel root')
    folder = ROOT/'terminal'
    require(not folder.exists(),'Exclusive one-role acquisition; no resume')
    def checks(np,values,call):
        prior = context['control']['trials'][call]
        before = context['control_outputs'][call]
        actual = values.tobytes()
        earlier = actual[:15*1024*4] == before[:15*1024*4]
        return dict(layer0_byte_exact=actual[:4096] == before[:4096],
            archive_output_byte_exact=actual == before,archive_output_control_passed=earlier,
            earlier15_rows_byte_exact=earlier,served_materialized_byte_exact=True,
            parent_flops=prior['flops'],parent_operation_count=prior['operation_count'],
            archived_output_bytes_sha256=prior['output_bytes_sha256'])
    def finish(result,directory):
        require(len(result['trials']) == len(result['input_preparation']) == 15
                and len(result['artifacts']) == 75,'Exactly14+repeat and five artifacts per call')
        for call,(new,old) in enumerate(zip(result['trials'],context['control']['trials'])):
            require(new['call'] == old['call'] == call and new['rank'] == old['rank']
                    and new['includes_initialization'] == old['includes_initialization']
                    and new['flops'] == old['flops'] and new['operation_count'] == old['operation_count']
                    and new['earlier15_rows_byte_exact'],'All actual corresponding cold/warm bills and prefix')
        init = result['trials'][0]['flops']-result['trials'][-1]['flops']
        require(init == INITIALIZATION and result['trials'][0]['output_bytes_sha256'] ==
                result['trials'][-1]['output_bytes_sha256'],'Actual initialization and exact repeat')
        require(isinstance(result['threadpools'],list) and result['threadpools'] == context['control']['threadpools'],
                'Same reported backend inventory; empty means unreported')
        require(h.peak_rss_bytes() < 8_000_000_000,'Final native RSS gate')
        for mapping in (result['source_bindings'],result['artifacts']):
            require(all(sha(path) == digest for path,digest in mapping.items()),'Final native source/artifact seals')
        seal = dict(utc=h.utc(),source_bindings=result['source_bindings'],fixture_bindings=context['fixtures'],
            candidate_sha256=CANDIDATE_SHA,role='terminal',trials=result['trials'],
            input_preparation=result['input_preparation'],artifacts=result['artifacts'],
            initialization_flops=init,initialization_once=True,repeat_excluded=True,truth_files_accessed=False,
            control_result_sha256=CONTROL_SHA)
        seal_sha = h.write_json(directory/'predictions_and_bills_sealed.json',seal)
        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,
            initialization_flops=init,outputs_seal_sha256=seal_sha,**dict.fromkeys(FLAGS,True))
    namespace = dict(__name__='tl65_panel_native_adapter',__file__=str(NATIVE),
        PANEL_RANKS=p.RANKS,PANEL_ROWS=context['rows'],PANEL_FIXTURES=context['fixtures'],
        PANEL_PINS=context['pins'],PANEL_CANDIDATE=CANDIDATE,
        PANEL_HEADER=dict(schema=SCHEMA,mode='acquire',role='terminal',completed=False,sources_unchanged=False,
            fixture_bindings=context['fixtures'],terminal_score_sha256=SCORE_SHA,
            terminal_score_watch_sha256=SCORE_WATCH_SHA,fit_result_sha256=FIT_SHA,
            control_result_sha256=CONTROL_SHA,candidate_source_inverse=context['fit']['proof'],
            versions=context['old'].VERSIONS,pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),
            blas_env=dict.fromkeys(BLAS,'1'),executable=sys.executable,no_warmup=True,instrumented=False,
            native_original_sha256=NATIVE_SHA,native_adapter_sha256=source_sha,
            preserved_panel_adapter_sha256=parent_adapter_sha,
            native_inverse_except_declared_harness_edits=True,paid_arithmetic_and_lifecycle_unchanged=True,
            truth_files_accessed=False,control_bills_are_archived_actual=True,no_refit=True,
            original_full_domain_readiness_reused=True,new_domain_arrays_captured=False),
        signal=signal,panel_output_checks=checks,panel_finish=finish)
    exec(compile(source,str(NATIVE),'exec'),namespace)
    namespace['PINS'][NATIVE] = NATIVE_SHA
    saved = sys.argv
    try:
        sys.argv = [str(__file__),'--source',str(CANDIDATE),'--split','validation','--out',str(folder)]
        return namespace['main']()
    finally:
        signal.alarm(0); sys.argv = saved


def verifier(context):
    """Exact inherited panel fixture splice plus sole candidate-eligibility change."""
    panel,v2 = context['panel'],context['v2']
    text = panel.QUALIFIER.read_text()
    node = next(x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef) and x.name == 'verify_native')
    original = ''.join(text.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    start = original.index("    rows = read(STORE / 'manifest.json')['roles']['observer_fit']['row_manifests']\n")
    stop = original.index("    trials, preps = acq['trials'], acq['input_preparation']\n",start)
    fixture = ("    require(acq['fixture_bindings'] == PANEL_FIXTURES, 'Canonical independent14 fixtures')\n"
        "    weight_hashes = {}\n"
        "    for row in PANEL_ROWS:\n"
        "        rank = row['rank']\n"
        "        wp = path(row['weights_path'])\n"
        "        weight_hashes[rank] = PANEL_FIXTURES['weights_'+str(rank)]['sha256']\n"
        "        require(bindings[str(wp)] == pin(wp,weight_hashes[rank]), 'Canonical panel weights')\n"
        "    for item in PANEL_FIXTURES.values():\n"
        "        pin(path(item['path']),item['sha256'])\n")
    before = ("    expected_sha = BANK_SHA if source == BANK else {HERE / 'build_zero_v1/estimator.py': ZERO_SHA,\n"
              "                       HERE / 'build_terminal_v1/estimator.py': TERMINAL_SHA}.get(source)")
    after = '    expected_sha = ELIGIBLE_SOURCES.get(source)'
    changed = panel.replace_exact(original,[(original[start:stop],fixture),(before,after)],require)
    env = dict(vars(v2),PANEL_FIXTURES=context['fixtures'],PANEL_ROWS=context['rows'],
               ELIGIBLE_SOURCES={CANDIDATE:CANDIDATE_SHA,context['tl'].BANK:context['tl'].BANK_SHA})
    exec(compile(changed,str(panel.QUALIFIER)+':tl65_panel_verifier','exec'),env)
    return env['verify_native']


def verify_acquisition(args,context):
    tl,h,p = context['tl'],context['h'],context['p']
    pins = context['pins']; pin = tl.pinner(pins)
    prospective_pins = dict(pins)
    folder = ROOT/'terminal'
    result = tl.sealed(folder,args.acquisition_sha256,pin)
    source,parent_sha = native_source(context)
    require(result['schema'] == SCHEMA and result['mode'] == 'acquire' and result['role'] == 'terminal'
            and all(result.get(k) is True for k in (*FLAGS,'sources_unchanged','no_refit',
                'native_inverse_except_declared_harness_edits','paid_arithmetic_and_lifecycle_unchanged'))
            and not result['truth_files_accessed'] and not result['instrumented']
            and result['candidate_sha256'] == CANDIDATE_SHA and result['terminal_score_sha256'] == SCORE_SHA
            and result['terminal_score_watch_sha256'] == SCORE_WATCH_SHA
            and result['fit_result_sha256'] == FIT_SHA and result['control_result_sha256'] == CONTROL_SHA
            and result['candidate_source_inverse'] == context['fit']['proof']
            and result['native_original_sha256'] == NATIVE_SHA
            and result['native_adapter_sha256'] == hashlib.sha256(source.encode()).hexdigest()
            and result['preserved_panel_adapter_sha256'] == parent_sha
            and result['source_bindings'][str(Path(__file__).resolve())] == args.source_sha256
            and result['source_bindings'][str(HERE/'PANEL_PROTOCOL.md')] == PROTOCOL_SHA,
            'One complete source-preserved new terminal acquisition')
    require(all(result['source_bindings'].get(path) == digest for path,digest in prospective_pins.items()),
            'Every prospective source/prerequisite/fixture pin retained by acquisition')
    outputs = verifier(context)(result,folder,p.RANKS,pin)
    require(result['versions'] == context['control']['versions'] and
            isinstance(result['threadpools'],list) and result['threadpools'] == context['control']['threadpools']
            and all(x['num_threads'] == 1 for x in result['threadpools'] if x.get('user_api') == 'blas'),
            'Same actual library/reported thread frame; empty is not backend evidence')
    for call,(trial,prior) in enumerate(zip(result['trials'],context['control']['trials'])):
        require(trial['flops'] == prior['flops'] == trial['parent_flops']
                and trial['operation_count'] == prior['operation_count'] == trial['parent_operation_count']
                and trial['includes_initialization'] == prior['includes_initialization']
                and trial['earlier15_rows_byte_exact']
                and outputs[call][:15*1024*4] == context['control_outputs'][call][:15*1024*4],
                'All actual full bills and earlier15 output bytes before any panel truth')
    require(result['initialization_flops'] == INITIALIZATION,'Actual unchanged startup ownership')
    pin(folder/'started.json',result['started_sha256'])
    started = read(folder/'started.json')
    require(started['pid'] == result['pid'] and started['nice'] >= 10 and started['no_warmup']
            and not started['instrumented'] and started['source_bindings'] == result['source_bindings']
            and started['fixture_bindings'] == context['fixtures'] and started['candidate_sha256'] == CANDIDATE_SHA
            and started['blas_env'] == dict.fromkeys(BLAS,'1'),'Sealed runtime start')
    pin(folder/'predictions_and_bills_sealed.json',result['outputs_seal_sha256'])
    seal = read(folder/'predictions_and_bills_sealed.json')
    require(all(seal[k] == result[k] for k in ('source_bindings','fixture_bindings','candidate_sha256',
            'role','trials','input_preparation','artifacts','initialization_flops'))
            and not seal['truth_files_accessed'] and seal['initialization_once'] and seal['repeat_excluded']
            and seal['control_result_sha256'] == CONTROL_SHA,'All output/bill seal ownership')
    require(len(WATCH_SHA) == 64,'Watcher must be source-reviewed and frozen before use')
    pin(HERE/'watch_panel.py',WATCH_SHA)
    watch = tl.sealed(ROOT/'terminal_watch',args.acquisition_watch_sha256,pin)
    require(watch['schema'] == 'current_d50_terminal_linear_independent14_watch_v1'
            and watch['stage'] == 'native' and watch['sources_unchanged']
            and watch['completed_bill_count'] == 15
            and watch['prerequisites']['terminal_score_sha256'] == SCORE_SHA
            and watch['child_result_sha256'] == args.acquisition_sha256
            and watch['child_completion_sha256'] == pin(folder/'completion.json')
            and watch['source_bindings'][str(HERE/'watch_panel.py')] == WATCH_SHA
            and watch['source_bindings'][str(Path(__file__).resolve())] == args.source_sha256
            and watch['child_execution_completed'] and watch['whole_owned_tree_retired'],
            'Completed independent new native resource watch before truths')
    tl.bind(watch['source_bindings'],pin); tl.bind(watch['artifacts'],pin)
    return result,outputs


def score(args,context,folder,result,guard):
    h,p,old = context['h'],context['p'],context['old']
    acquisition,new_bytes = verify_acquisition(args,context)
    pins = context['pins']
    require(all(sha(path) == digest for path,digest in pins.items()),'Every output/source/bill seal before numeric imports')
    import numpy as np
    require(np.__version__ == '2.4.6' and not any(k in sys.modules for k in ('torch','flopscope')),
            'Offline NumPy-only original scoring frame')
    outputs = {role:[np.frombuffer(x,dtype=np.float32).reshape(16,1024) for x in blobs]
               for role,blobs in (('current_d50',context['control_outputs']),('terminal',new_bytes))}
    for arrays in outputs.values():
        require(len(arrays) == 15 and all(np.isfinite(x).all() and (x >= 0).all() for x in arrays)
                and arrays[0].tobytes() == arrays[-1].tobytes(),'All served rows finite/nonnegative and repeats exact')
    guard()
    seal = dict(utc=h.utc(),source_bindings=dict(pins),fixture_bindings=context['fixtures'],
        candidate_sha256=CANDIDATE_SHA,parent_sha256=context['tl'].BANK_SHA,ranks=list(p.RANKS),
        terminal_score_sha256=SCORE_SHA,acquisition_sha256=args.acquisition_sha256,
        acquisition_watch_sha256=args.acquisition_watch_sha256,control_result_sha256=CONTROL_SHA,
        native_trials=acquisition['trials'],control_trials=context['control']['trials'],
        truth_files_accessed=False,initialization_once=True,repeat_excluded=True)
    before = h.write_json(folder/'both_roles_verified_before_truth.json',seal)
    result.update(source_bindings=pins,fixture_bindings=context['fixtures'],candidate_sha256=CANDIDATE_SHA,
        parent_sha256=context['tl'].BANK_SHA,both_roles_verified_before_truth_sha256=before,
        all_predictions_verified_before_truth=True,truth_bindings={},per_role={'current_d50':[],'terminal':[]})
    index = h.load_json(old.TRUTH_INDEX)
    require(index['qualification_passed'] and index['hashes_unchanged']
            and set(index['truth_bindings']) == {str(r) for r in p.RANKS},'Original fixed independent4M index')
    result['truth_files_accessed'] = True
    h.write_json(folder/'truth_access_started.json',dict(utc=h.utc(),prior_seal_sha256=before))
    for rank in p.RANKS:
        path = p.G19/(f'mps_truth_rank{rank:04d}_indep4M.npz' if rank in p.PUBLIC else f'fresh_{rank}_truth4M.npz')
        expected = index['truth_bindings'][str(rank)]
        require(Path(expected['path']).resolve() == path and sha(path) == expected['sha256'],'Original4M truth binding')
        result['truth_bindings'][str(rank)] = h.binding(path)
    for call,rank in enumerate(p.RANKS):
        with np.load(result['truth_bindings'][str(rank)]['path'],allow_pickle=False) as data:
            mean,se = data['mc_post'][15].astype(np.float64),data['mc_se'][15].astype(np.float64)
        require(mean.shape == se.shape == (1024,) and np.isfinite(mean).all()
                and np.isfinite(se).all() and (se >= 0).all(),'Finite original independent means/SE')
        noise = float(np.mean(se*se))
        for role,record in (('current_d50',context['control']),('terminal',acquisition)):
            trial = record['trials'][call]
            raw = float(np.mean((outputs[role][call][15].astype(np.float64)-mean)**2))
            nc = raw-noise
            factor = max(.1,trial['flops']/h.BUDGET)
            require(math.isfinite(raw) and raw >= 0 and math.isfinite(nc),'Finite unmodified score')
            row = dict(rank=rank,flops=trial['flops'],output_bytes_sha256=trial['output_bytes_sha256'],
                raw_observed=raw,raw_noise_corrected=nc,mc_se_squared_mean=noise,multiplier=factor,
                adjusted_observed=raw*factor,adjusted_noise_corrected=nc*factor)
            if role == 'current_d50':
                archived = context['control_score']['per_role']['candidate'][call]
                require(all(row[k] == archived[k] for k in row),'Exact d50 observed/NC raw and own-bill score replay')
            result['per_role'][role].append(row)
        print(json.dumps({role:rows[-1] for role,rows in result['per_role'].items()},allow_nan=False),flush=True)
        guard()
    fields = ('raw_observed','raw_noise_corrected','mc_se_squared_mean','flops','adjusted_observed','adjusted_noise_corrected')
    summaries = {}
    for role,rows in result['per_role'].items():
        summaries[role] = {}
        for group,ranks in (('public8',p.PUBLIC),('virgin6',p.VIRGIN),('combined14',p.RANKS)):
            subset = [x for x in rows if x['rank'] in ranks]
            summaries[role][group] = dict(n=len(subset),**{
                'mean_'+key:sum(x[key] for x in subset)/len(subset) for key in fields})
    require(summaries['current_d50'] == context['control_score']['summaries']['candidate'],
            'Exact original d50 aggregate score, same summation convention')
    ratios,gains = {},{}
    for group in ('public8','virgin6','combined14'):
        a = summaries['current_d50'][group]['mean_adjusted_noise_corrected']
        b = summaries['terminal'][group]['mean_adjusted_noise_corrected']
        require(a > 0,'Positive original group comparator')
        ratios[group],gains[group] = b/a,b < a
    require(all(sha(path) == digest for path,digest in pins.items()),'End source/output/bill seals')
    h.check_bindings(result['truth_bindings'])
    require(sha(folder/'both_roles_verified_before_truth.json') == before,'Pretruth seal unchanged')
    guard()
    result.update(passed=True,completed=True,qualification_passed=True,integrity_passed=True,
        hashes_unchanged=True,sources_unchanged=True,summaries=summaries,
        adjusted_candidate_over_parent=ratios,group_strict_gain_gate=gains,
        retain_local_component=gains['public8'] and gains['virgin6'],baseline_recomputed_exact=True,
        earlier15_rows_byte_exact=True,own_complete_F_and_opcounts_exact=True,
        acquisition_sha256=args.acquisition_sha256,acquisition_watch_sha256=args.acquisition_watch_sha256,
        control_result_sha256=CONTROL_SHA,control_score_sha256=CONTROL_SCORE_SHA,
        initialization_flops=INITIALIZATION,repeat_excluded=True,
        artifacts={str(folder/name):sha(folder/name) for name in
                   ('both_roles_verified_before_truth.json','truth_access_started.json')},
        aggregation='Mean per-net unclipped (final observed MSE-mean(mc_se15**2))*max(.1,own complete F/2**41).',
        frame='Reused independent4M public8/virgin6. Public35/100/200 were TRAIN and150 VALID; not unseen model holdouts.')


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
                and set(getattr(args,key)) <= set('0123456789abcdef') for key in
                ('source_sha256','protocol_sha256','terminal_score_sha256')),'All reviewed source/prerequisite SHAs')
    require((args.score and all(isinstance(v,str) and len(v) == 64 and set(v) <= set('0123456789abcdef')
                for v in (args.acquisition_sha256,args.acquisition_watch_sha256)))
            or (args.acquire and args.acquisition_sha256 is None and args.acquisition_watch_sha256 is None),
            'Score binds acquisition and watch; acquire has no output-selection arguments')
    def expired(_signal,_frame):
        raise TimeoutError('Fixed independent14 child watchdog')
    signal.signal(signal.SIGALRM,expired); signal.alarm(120)
    context = prerequisites(args)
    if args.acquire:
        return acquire(args,context)
    ROOT.mkdir(exist_ok=True)
    folder = ROOT/'score'; folder.mkdir(exist_ok=False)
    h = context['h']; began = time.monotonic()
    result = dict(schema=SCHEMA,mode='score',passed=False,completed=False,qualification_passed=False,
        integrity_passed=False,hashes_unchanged=False,sources_unchanged=False,
        requested_seals=vars(args),terminal_score_sha256=SCORE_SHA,truth_files_accessed=False,
        no_refit=True,production_qualified=False,official_score=False,milestone_achieved=False,
        retain_local_component=False,started_utc=datetime.now(timezone.utc).isoformat())
    h.write_json(folder/'started.json',result)
    def guard():
        require(time.monotonic()-began < 120 and h.peak_rss_bytes() < 2_000_000_000,'Offline score120s/2GB')
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
