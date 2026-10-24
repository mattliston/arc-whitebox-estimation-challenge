"""DRAFT: conditional unchanged below4 package/submission; inert without a phase.

No inference or score work in this parent. Reviewed R2 observations cover every
owned CLI process. Package and submit are separate exclusive stages. Existing
key enters only the submit child's environment after an immutable attempt marker.
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
import re
import selectors
import signal
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
ENV = P2/'data_gate/.venv'
SITE = ENV/'lib/python3.11/site-packages/whestbench'
CLI = ENV/'bin/whest'
SOURCE = HERE/'build_v1/estimator.py'
CANDIDATE_SHA = '4d98fef2c11c571efe1eca945018bb9fe9ece1454d16b4abde2eb52d71e755c3'
LEGACY = NEW/'birth_direction_fit_v1/release_confirmation_v1/package_submit.py'
LEGACY_SHA = '48edb1eb0bf69d82a3523c38de3129ebc7c9b751e70e7090f74af0a9dfbbac27'
TRANSPORT = NEW/'current_comp_cstar_proposal_v1/watch.py'
TRANSPORT_SHA = 'd3258f391ca501698e6c5f14a9dce21f28ae4e63d05731800ce6b2da8c94f23e'
OUT = HERE/'release_v1'
STAGE = OUT/'package_stage'
ARCHIVE = OUT/'shared_old384_4d98fef2_whest016.tar.gz'
KEY_FILE = Path('/Users/matt/arc-white-box-estimation/key.txt')
CHALLENGE = 'arc-white-box-estimation-challenge-2026'
DESCRIPTION = ('Phase 2 shared old-cumulant basis: fixed current heads, '
    'complete-cost validation and independent public8/virgin6 panels, '
    'five NumPy-free production and native-fidelity checks')
VERSIONS = {'whestbench':'0.16.0','flopscope':'0.12.0','numpy':'2.4.6'}
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')
PINS = {
    CLI:'3dac97e9083bda232206eb95fc77e4e33f8e7bc14fecdf2f1a441759d98b6b63',
    SITE/'cli.py':'f6b975fd39a1dfda5e71efada4f7f4da0b5fde0b0130816484a660f91dd2b58d',
    SITE/'packaging.py':'64f38dfe46951e1f49e40b8872e137a9338b7b33b115d8edd6058ddd3431b0a6',
    SITE/'validation.py':'239ceb596742baf534743a64ea9bf539011677eb7d0cc13f31b22593660a236b',
    SITE/'aicrowd_config.py':'fcf5175963f0a905b05a7daa4419c964376f4c11ed761b35595dd53bc253a188',
    SITE/'aicrowd_client.py':'441cf36acb61266a960164de80fc66b21c5015773a898c76d1807b9c46bae6a6',
    SOURCE:CANDIDATE_SHA,LEGACY:LEGACY_SHA,TRANSPORT:TRANSPORT_SHA,
}
# Actual future result hashes are mandatory launch inputs, not fabricated pins.
STAGES = {
    'native40':('native40_v1',3000,8_000_000_000),
    'native40_watch':('native40_watch_v1',3000,8_000_000_000),
    'score40':('score40_v1',120,1_000_000_000),
    'score40_watch':('score40_watch_v1',120,1_000_000_000),
    'native_panel':('native_panel_v1',1200,8_000_000_000),
    'native_panel_watch':('native_panel_watch_v1',1200,8_000_000_000),
    'score_panel':('score_panel_v1',120,2_000_000_000),
    'score_panel_watch':('score_panel_watch_v1',120,2_000_000_000),
    'fidelity':('fidelity_v1',120,1_000_000_000),
    'fidelity_watch':('fidelity_watch_v1',120,1_000_000_000),
}
RUNTIME_SHA = 'c8d325e35229396bf67f63fd65e92f09f75ed8305ef1e169388d975507318af0'
BUILD_SHA = '54a4d8d9751dca4c654820a48e19c2e94290cdf8b004e2befa86686f4558380a'
BUILD_WATCH_SHA = '1145ed7295d8d4c738074db4da5718531470c7719a46765758b1f803277f1d16'
SEAL_NAMES = ('result.json','completion.json','finalization.json')
SEAL_KEYS = ('result_sha256','completion_sha256','finalization_sha256')
TICK = None


class CheckFailure(RuntimeError):
    """Only non-secret, fixed diagnostic codes are allowed."""


def require(ok, code):
    if not ok:
        raise CheckFailure(code)


def utc():
    return datetime.now(timezone.utc).isoformat()


def valid_sha(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(P2)
        and 'truthpost' not in path.parts and path.name != 'key.txt'
        and not (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()),
        'regular_nontarget_inscope_file')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
            if TICK is not None:
                TICK()
    return h.hexdigest()


def load(path):
    path = Path(path)
    require(path.suffix == '.json' and not path.is_symlink()
        and 0 < path.stat().st_size < 16*1024**2,'bounded_regular_metadata')
    return json.loads(path.read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


def bind(bindings, path, expected):
    path = str(Path(path).resolve())
    require(valid_sha(expected) and (path not in bindings or bindings[path] == expected),
        'consistent_reviewed_input_pin')
    require(sha(path) == expected,'changed_qualified_input')
    bindings[path] = expected


def unchanged(bindings):
    for path,digest in bindings.items():
        require(sha(path) == digest,'sealed_input_changed')


def legacy_helpers():
    """Reuse exact successful archive and secret-safe response functions only."""
    require(sha(LEGACY) == LEGACY_SHA,'reviewed_legacy_helpers')
    source = LEGACY.read_text(); lines = source.splitlines(keepends=True)
    functions = {}
    for node in ast.parse(source).body:
        if isinstance(node,ast.FunctionDef) and node.name in (
                'inspect_archive','approved_response','safe_failure'):
            functions[node.name] = ''.join(lines[node.lineno-1:node.end_lineno])
    require(set(functions) == {'inspect_archive','approved_response','safe_failure'},
        'exact_three_pure_legacy_helpers')
    scope = dict(globals())
    for name,text in functions.items():
        exec(compile(text,str(LEGACY)+':unchanged-'+name,'exec'),scope)
    return scope,{k:hashlib.sha256(v.encode()).hexdigest() for k,v in functions.items()}


def chain(key, folder, digest, cap, rss, bindings, watched=False):
    paths = [HERE/folder/name for name in SEAL_NAMES]
    hashes = [sha(p) for p in paths]
    require(valid_sha(digest) and hashes[0] == digest,'actual_positive_result_pin_'+key)
    r,e,f = map(load,paths)
    require(all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
        and e['result_sha256'] == f['result_sha256'] == hashes[0]
        and f['completion_sha256'] == hashes[1]
        and r['hashes_unchanged'] and r['sources_unchanged']
        and 0 <= r['elapsed_s'] <= f['elapsed_s'] < cap,'positive_original_three_seal_scope_'+key)
    if watched:
        require(all(r.get(k) is True for k in ('whole_owned_tree_retired','cleanup_passed',
            'observational_groups_retired','actual_completed_samples','foreign_numerical_workers_checked'))
            and r['numerical_imports'] is False and r['maximum_swap_growth_bytes'] == 0
            and 0 < r['maximum_interval_s'] <= 1
            and 0 < r['maximum_combined_rss_bytes'] < rss,'full_original_watched_resource_scope_'+key)
    else:
        require(0 < r['peak_rss_bytes'] < rss,'original_worker_memory_'+key)
    for group in ('source_bindings','input_bindings','artifacts'):
        require(isinstance(r[group],dict) and r[group],'complete_direct_nontarget_inventory_'+key)
        for path,value in r[group].items():
            bind(bindings,path,value)
    for path,value in zip(paths,hashes):
        bind(bindings,path,value)
    return r,dict(zip(SEAL_KEYS,hashes))


def evidence(args, bindings):
    records,seals = {},{}
    for key,(folder,cap,rss) in STAGES.items():
        records[key],seals[key] = chain(key,folder,getattr(args,key+'_sha256'),
            cap,rss,bindings,watched=key.endswith('_watch'))
    for child in ('native40','score40','native_panel','score_panel','fidelity'):
        require(all(records[child+'_watch']['child_'+k] == v for k,v in seals[child].items()),
            'matching_child_watcher_seals_'+child)
    n,s,p,q,f = (records[k] for k in ('native40','score40','native_panel','score_panel','fidelity'))
    require(n['schema'] == 'current_comp_shared_old384_native40_v1'
        and p['schema'] == 'current_comp_shared_old384_native_panel_v1'
        and n['candidate_sha256'] == p['candidate_sha256'] == CANDIDATE_SHA
        and n['full_model_calls'] == n['retired_call_count'] == 41
        and p['full_model_calls'] == p['retired_call_count'] == 15
        and n['initialization_flops'] == p['initialization_flops'] == 1944970298
        and n['own_complete_bills'] and p['own_complete_bills']
        and n['all16_domains_qualified'] and p['all16_domains_qualified']
        and n['all47_selectors_qualified'] and p['all47_selectors_qualified'],
        'same_source_complete_native40_and_independent14')
    require(s['schema'] == 'current_comp_shared_old384_score40_v1'
        and s['lineage']['probe'] == seals['native40']
        and s['lineage']['probe_watch'] == seals['native40_watch']
        and s['scientific_gate_passed'] and s['information_promising']
        and s['comparison']['adjusted_ratio'] <= .97 and len(s['records']) == 40
        and s['own_complete_bills'] and s['full_model_calls'] == s['parameter_updates'] == 0,
        'prospective_full40_objective_gate')
    development = {k:seals[k] for k in ('native40','native40_watch','score40','score40_watch')}
    require(p['native40_prerequisite_seals'] == development
        and q['lineage']['native'] == seals['native_panel']
        and q['lineage']['native_watch'] == seals['native_panel_watch']
        and q['lineage']['development'] == development,'matching_actual_native_and_scoring_frames')
    require(q['schema'] == 'current_comp_shared_old384_score_panel_v1'
        and q['candidate_sha256'] == CANDIDATE_SHA
        and q['bank_scientific_gate_passed'] and q['both_panels_three_percent_gain']
        and q['combined_noise_corrected_adjusted_below_4e9']
        and all(q['group_three_percent_gain_gate'][g] is True for g in ('public8','virgin6'))
        and all(q['adjusted_candidate_over_parent'][g] <= .97 for g in ('public8','virgin6'))
        and q['summaries']['shared']['combined14']['mean_adjusted_noise_corrected'] < 4e-9,
        'both_independent_groups_and_strict_below4_milestone')
    require(f['schema'] == 'current_comp_shared_old384_fidelity_v1'
        and f['candidate_sha256'] == CANDIDATE_SHA
        and all(f.get(k) is True for k in ('all_five_output_vector_tolerances_passed',
            'all_four_unique_raw_mse_ratios_passed','exact_native_observed_score_replay',
            'exact_fixture_compatibility','native_and_runtime_repeat_exact','own_complete_bills',
            'runtime_timing_qualified','production_fidelity_qualified'))
        and f['initialization_flops'] == 1944970298,
        'complete_same_candidate_native_and_real_client_fidelity')
    runtime,rs = chain('runtime','runtime_v1',RUNTIME_SHA,600,8_000_000_000,bindings,True)
    require(runtime['candidate_sha256'] == CANDIDATE_SHA
        and runtime['ranks'] == [35,100,500,3016500,35]
        and runtime['full_model_calls'] == 5 and runtime['all5_production_timing_passed']
        and runtime['repeat_byte_exact'] and runtime['own_complete_bills']
        and runtime['initialization_flops'] == 1944970298,'original_fixed_five_production_timing')
    require(p['production_runtime_seals'] == rs and f['lineage'] == dict(
        panel_score=seals['score_panel'],panel_score_watch=seals['score_panel_watch'],
        native=seals['native_panel'],native_watch=seals['native_panel_watch'],runtime=rs),
        'exact_fidelity_native_score_and_runtime_three_seal_joins')
    b,bs = chain('build','build_v1',BUILD_SHA,60,2_000_000_000,bindings)
    bw,bws = chain('build_watch','build_watch_v1',BUILD_WATCH_SHA,60,2_000_000_000,bindings,True)
    require(b['candidate_sha256'] == bw['candidate_sha256'] == CANDIDATE_SHA
        and b['artifacts'] == {str(SOURCE):CANDIDATE_SHA}
        and bw['build_seals'] == bs and b['proof']['materialization_only'],
        'exact_qualified_source_materialization')
    seals.update(runtime=rs,build=bs,build_watch=bws)
    return records,seals


class OwnedWatch:
    """No numerical package: original R2 complete sample/tree/cleanup primitives."""
    def __init__(self, result, phase, began):
        self.began = self.previous = began
        self.result = result; self.phase = phase
        require(sha(TRANSPORT) == TRANSPORT_SHA,'reviewed_monitor_transport')
        spec = importlib.util.spec_from_file_location('shared_release_transport',TRANSPORT)
        self.h = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.h)
        self.mon,self.classifier = self.h.transport()
        self.child = None; self.commands = []; self.initial_swap = None
        self.stream = (OUT/(phase+'_samples.jsonl')).open('x')
        self.child_deadline = self.began+600
        self.result.update(maximum_combined_rss_bytes=0,maximum_interval_s=0.,
            maximum_swap_growth_bytes=0.,observation_samples=0,whole_cap_s=600.,
            combined_rss_limit_bytes=8_000_000_000)

    def deadline(self):
        require(time.monotonic() < min(self.began+600,self.child_deadline),
            'whole_or_cli_deadline_exhausted')

    def observe(self):
        absolute = min(self.previous+1,self.began+600,self.child_deadline)
        sample = self.mon.collect(absolute)
        scoped = self.mon.scope(sample['rows'],os.getpid(),None if self.child is None else self.child.pid)
        require(not self.classifier(sample['rows'],scoped.pop('allowed')),'foreign_numerical_worker')
        if self.initial_swap is None:
            self.initial_swap = sample['swap']['used_bytes']
        growth = sample['swap']['used_bytes']-self.initial_swap
        require(growth <= 0 and scoped['combined_rss_bytes'] < 8_000_000_000,'owned_tree_memory_or_swap')
        self.result['maximum_combined_rss_bytes'] = max(self.result['maximum_combined_rss_bytes'],scoped['combined_rss_bytes'])
        self.result['maximum_swap_growth_bytes'] = max(self.result['maximum_swap_growth_bytes'],growth)
        self.stream.write(json.dumps(dict(elapsed_s=time.monotonic()-self.began,**scoped,
            swap_bytes=sample['swap']['used_bytes'],transport=sample['command_records'],
            observational_groups_retired=sample['commands_reaped']),sort_keys=True)+'\n')
        self.stream.flush()
        self.previous,interval = self.mon.complete(self.previous,absolute)
        self.result['maximum_interval_s'] = max(self.result['maximum_interval_s'],interval)
        self.result['observation_samples'] += 1

    def tick(self):
        self.deadline()
        if time.monotonic()-self.previous >= .2:
            self.observe()

    def cli(self, arguments, name, env, cap, response_filter, secret=None):
        self.deadline(); self.observe()
        began = time.monotonic(); self.child_deadline = min(self.began+600,began+cap)
        self.child = subprocess.Popen([str(CLI),*arguments],cwd=STAGE,env=env,
            stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        self.commands.append(self.child)
        streams = selectors.DefaultSelector(); captured = {'stdout':bytearray(),'stderr':bytearray()}
        for name_,pipe in (('stdout',self.child.stdout),('stderr',self.child.stderr)):
            os.set_blocking(pipe.fileno(),False); streams.register(pipe,selectors.EVENT_READ,name_)
        try:
            while streams.get_map() or self.child.poll() is None:
                self.tick()
                for key,_ in streams.select(.05):
                    data = os.read(key.fileobj.fileno(),65536)
                    if data:
                        captured[key.data].extend(data)
                        require(len(captured['stdout'])+len(captured['stderr']) < 2_000_000,'bounded_private_cli_capture')
                    else:
                        streams.unregister(key.fileobj)
                self.tick()
            self.observe(); code = self.child.wait(timeout=.1)
            self.mon.cleanup([self.child],time.monotonic()+.5)
            self.child = None
            try:
                payload = json.loads(captured['stdout'])
            except (ValueError,TypeError,UnicodeError):
                payload = None
            projected = response_filter(payload,secret)
            record = dict(command=name,returncode=code,elapsed_s=time.monotonic()-began,
                stdout_bytes=len(captured['stdout']),stderr_bytes=len(captured['stderr']),response=projected)
            self.result['commands'].append(record)
            require(code == 0 and isinstance(payload,dict) and payload.get('ok') is True,'whest_command_failed')
            self.observe()
            self.deadline()
            self.child_deadline = self.began+600
            return projected
        finally:
            streams.close()
            for command in self.commands:
                for pipe in (command.stdout,command.stderr):
                    if pipe is not None and not pipe.closed:
                        pipe.close()
            captured.clear()

    def retire(self):
        self.mon.cleanup(self.commands,time.monotonic()+.5)
        self.mon.retire_pending(time.monotonic()+.5)
        self.child = None
        self.result.update(whole_owned_tree_retired=True,observational_groups_retired=True,cleanup_passed=True)


def main():
    global TICK
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_mutually_exclusive_group()
    phases.add_argument('--package',action='store_true'); phases.add_argument('--submit',action='store_true')
    parser.add_argument('--numerical-slot-clear',action='store_true')
    parser.add_argument('--source-sha256'); parser.add_argument('--protocol-sha256')
    parser.add_argument('--package-sha256')
    for key in STAGES:
        parser.add_argument('--'+key.replace('_','-')+'-sha256')
    args = parser.parse_args()
    if not (args.package or args.submit):
        parser.print_help(); return 0
    began = time.monotonic()
    require(Path(sys.prefix).resolve() == ENV.resolve() and sys.flags.optimize == 0
        and sys.dont_write_bytecode and os.getpriority(os.PRIO_PROCESS,0) >= 10
        and all(os.environ.get(k) == '1' for k in BLAS) and args.numerical_slot_clear,
        'official_assertions_bytecodeoff_nice10_serial_BLAS1')
    require({k:importlib.metadata.version(k) for k in VERSIONS} == VERSIONS,'pinned_official_versions')
    require(valid_sha(args.source_sha256) and sha(__file__) == args.source_sha256
        and valid_sha(args.protocol_sha256) and sha(HERE/'RELEASE_DESIGN_20260912.md') == args.protocol_sha256,
        'reviewed_release_sources')
    require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules),'stdlib_release_parent')
    phase = 'package' if args.package else 'submit'
    if args.package:
        require(not OUT.exists(),'exclusive_release_directory')
        OUT.mkdir(); STAGE.mkdir()
    else:
        require(OUT.is_dir() and not OUT.is_symlink() and STAGE.is_dir(),'existing_exclusive_package_stage')
    result_path = OUT/(phase+'_result.json')
    require(not result_path.exists() and not (OUT/(phase+'_started.json')).exists()
        and (args.package or not (OUT/'submit_attempt.json').exists()),'never_repeat_release_stage')
    result = dict(schema='current_comp_shared_old384_release_v1',phase=phase,passed=False,completed=False,
        candidate_sha256=CANDIDATE_SHA,source_bindings={},input_bindings={},artifacts={},commands=[],
        uploaded=False,official_score=False,full_phase2_model_calls=0,parameter_updates=0,
        numerical_imports=False,truth_arrays_accessed=False,started_utc=utc(),no_automatic_retry=True)
    watch = None; helpers = None; key = env = None
    def interrupted(*unused):
        raise CheckFailure('release_interrupted')
    handlers = {s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        watch = OwnedWatch(result,phase,began); TICK = watch.tick; watch.observe()
        helpers,proof = legacy_helpers(); result['legacy_pure_helper_proof'] = proof
        for path,digest in {**PINS,Path(__file__):args.source_sha256,
                HERE/'RELEASE_DESIGN_20260912.md':args.protocol_sha256,**watch.h.MON_PINS}.items():
            bind(result['source_bindings'],path,digest)
        for path,digest in watch.h.MOCK_PINS.items():
            bind(result['input_bindings'],path,digest)
        records,seals = evidence(args,result['input_bindings'])
        result.update(evidence_seals=seals,all_scientific_and_release_evidence_qualified=True,
            local_independent14_NC_adjusted=records['score_panel']['summaries']['shared']['combined14']['mean_adjusted_noise_corrected'],
            local_native40_adjusted=records['score40']['summary']['shared']['adjusted'])
        write(OUT/(phase+'_started.json'),result)
        env = dict(os.environ); env.pop('AICROWD_API_KEY',None)
        env.pop('PYTHONOPTIMIZE',None)
        env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTORCH_ENABLE_MPS_FALLBACK='0')
        if args.package:
            with (STAGE/'estimator.py').open('xb') as stream:
                stream.write(SOURCE.read_bytes())
            require(sha(STAGE/'estimator.py') == CANDIDATE_SHA,'exact_staged_candidate')
            watch.cli(['validate','--estimator',str(STAGE/'estimator.py'),'--json'],
                'validate',env,120,helpers['approved_response'])
            watch.cli(['package','--estimator',str(STAGE/'estimator.py'),'--output',str(ARCHIVE),'--yes','--json'],
                'package',env,120,helpers['approved_response'])
            watch.cli(['validate-package',str(ARCHIVE),'--json'],'validate-package',env,120,helpers['approved_response'])
            result.update(helpers['inspect_archive']())
            bind(result['artifacts'],STAGE/'estimator.py',CANDIDATE_SHA)
            bind(result['artifacts'],ARCHIVE,result['archive_sha256'])
            result.update(generic_api_smoke_calls=1,generic_smoke_is_phase2_score_evidence=False)
        else:
            require(valid_sha(args.package_sha256),'explicit_successful_package_result')
            package_path = OUT/'package_result.json'
            bind(result['input_bindings'],package_path,args.package_sha256)
            prior = load(package_path)
            cp,fp = OUT/'package_completion.json',OUT/'package_finalization.json'
            ch,fh = sha(cp),sha(fp)
            complete,final = load(cp),load(fp)
            require(all(v.get('passed') is True and v.get('completed') is True for v in (complete,final))
                and complete['result_sha256'] == final['result_sha256'] == args.package_sha256
                and final['completion_sha256'] == ch and final['elapsed_s'] < 600
                and 0 <= final['terminal_publication_interval_s'] <= 1,
                'actual_complete_positive_package_three_seals')
            bind(result['input_bindings'],cp,ch); bind(result['input_bindings'],fp,fh)
            require(prior['passed'] and prior['completed'] and prior['hashes_unchanged']
                and prior['sources_unchanged'] and prior['candidate_sha256'] == CANDIDATE_SHA
                and prior['evidence_seals'] == seals and prior['uploaded'] is False
                and all(prior.get(k) is True for k in ('whole_owned_tree_retired','cleanup_passed',
                    'observational_groups_retired','actual_completed_samples','foreign_numerical_workers_checked'))
                and prior['maximum_swap_growth_bytes'] == 0 and 0 < prior['maximum_interval_s'] <= 1
                and 0 < prior['maximum_combined_rss_bytes'] < 8_000_000_000,
                'same_qualified_package')
            for group in ('source_bindings','input_bindings','artifacts'):
                for path,digest in prior[group].items():
                    bind(result['input_bindings'],path,digest)
            archive = helpers['inspect_archive']()
            require(all(prior[k] == v for k,v in archive.items()),'unchanged_exact_validated_archive')
            marker = dict(candidate_sha256=CANDIDATE_SHA,archive_sha256=archive['archive_sha256'],
                package_result_sha256=args.package_sha256,evidence_seals=seals,
                challenge=CHALLENGE,description=DESCRIPTION,started_utc=utc(),
                authorized_scope='User preauthorized fully validated strict below4 Phase2 milestone bank',
                maximum_wrapper_cli_invocations=1,no_automatic_wrapper_retry=True,
                cli_internal_HTTP_retries='Pinned Whest0.16 bounded5 attempts/45s per logical request')
            write(OUT/'submit_attempt.json',marker)
            bind(result['artifacts'],OUT/'submit_attempt.json',sha(OUT/'submit_attempt.json'))
            key = KEY_FILE.read_text().strip()
            require(bool(key) and '\n' not in key and '\r' not in key,'existing_key_unavailable')
            env['AICROWD_API_KEY'] = key
            result['external_call_attempted'] = True
            response = watch.cli(['submit',str(ARCHIVE),'--challenge',CHALLENGE,'--description',DESCRIPTION,'--json'],
                'submit',env,180,helpers['approved_response'],key)
            require(type(response.get('submission_id')) is int and response['submission_id'] > 0,
                'submission_identity_unknown_do_not_retry')
            result.update(submission_id=response['submission_id'],uploaded=True,
                tracking_url=f'https://www.aicrowd.com/challenges/{CHALLENGE}/submissions/{response["submission_id"]}')
        watch.retire()
        for group in ('source_bindings','input_bindings','artifacts'):
            unchanged(result[group])
        require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules),'parent_still_stdlib_only')
        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,
            actual_completed_samples=True,foreign_numerical_workers_checked=True)
        watch.observe()
    except BaseException as exc:
        result['error'] = dict(type=type(exc).__name__,code=str(exc) if isinstance(exc,CheckFailure) else 'stage_exception')
        if result.get('external_call_attempted'):
            result['submission_status_may_be_unknown'] = True
        result.update(passed=False,completed=False)
    finally:
        if env is not None:
            env.pop('AICROWD_API_KEY',None)
        key = None
        if watch is not None:
            try:
                watch.retire()
                if result['passed']:
                    watch.deadline()
            except BaseException:
                result.update(passed=False,completed=False,cleanup_error='owned_cleanup_or_deadline_failed')
            watch.stream.close()
            result.update(elapsed_s=time.monotonic()-watch.began,initial_swap_bytes=watch.initial_swap,
                last_actual_completed_sample_elapsed_s=watch.previous-watch.began)
        TICK = None
        for name in (phase+'_started.json',phase+'_samples.jsonl'):
            path = OUT/name
            if path.exists():
                result['artifacts'][str(path)] = sha(path)
        result['ended_utc'] = utc()
        write(result_path,result); rh = sha(result_path)
        ok = bool(result['passed'] and watch is not None and time.monotonic()-watch.began < 600
            and time.monotonic()-watch.previous <= 1)
        cp = OUT/(phase+'_completion.json')
        write(cp,dict(passed=ok,completed=ok,result_sha256=rh))
        ch = sha(cp)
        ok = bool(ok and time.monotonic()-watch.began < 600 and time.monotonic()-watch.previous <= 1)
        fp = OUT/(phase+'_finalization.json')
        write(fp,dict(passed=ok,completed=ok,result_sha256=rh,completion_sha256=ch,
            elapsed_s=None if watch is None else time.monotonic()-watch.began,
            terminal_publication_interval_s=None if watch is None else time.monotonic()-watch.previous))
        sha(fp)
        ok = bool(ok and time.monotonic()-watch.began < 600 and time.monotonic()-watch.previous <= 1)
        for sig,handler in handlers.items():
            signal.signal(sig,handler)
    print(json.dumps(dict(phase=phase,passed=ok,receipt=str(result_path),receipt_sha256=rh,
        uploaded=result['uploaded'],submission_id=result.get('submission_id'),official_score=False,
        no_automatic_retry=True)),flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
