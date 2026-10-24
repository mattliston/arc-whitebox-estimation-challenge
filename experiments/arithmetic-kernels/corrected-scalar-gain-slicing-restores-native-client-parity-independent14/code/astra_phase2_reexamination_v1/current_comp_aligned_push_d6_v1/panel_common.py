"""Target-free APUSH panel metadata; exact CD v2 lifecycle and direct receipts."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
CD, COMP = NEW/'current_comp_partial_d6_v1', NEW/'current_p1_node_composition_v1'
APUSH_NATIVE_SHA = '751bc155506b2825a8c0d27553e8ced8d7db66d68434247bb158c22fad3c3ad6'
ENDPOINT_OWNER_SHA = 'c6ed25eaa1a6ff2131640c5ec665fccb1886446531475bff4d2d83c8076aa652'
COMP_HELPER_SHA = '63832d341b4e4e78b5dda456f0dcfeba42a06114c2142d529320c6dcddea1313'
HASH_SHA = 'db708ef69a539f918c1ad2b0ad68ffb3a9d293a05a2bb2ddae92624e1e12b653'
CD_OWNERS = {
    'panel_common.py':'7d9c8b2d97ac5bbb85c077da694472a52fbede2946ec2643df44e1018a6185b7',
    'panel_common_v2.py':'77d238cbaa3f046ecadcaa7520d5b7edf4075843b3bc329696ab4457dd90571c',
    'panel_admit.py':'ac38f0c27d61289c53b1278309ef99b5e9d126fb7c03f06188354fdad015b83f',
    'panel_admit_v2.py':'9f93131823df7c7be4ccd2ebd6433f8c100f380c4e788d5c7d3792f585eebdc7',
    'panel_native.py':'9f6dd0de56e3af8fc2c8c3be400963148b9fde1a414b7f888a603bb988a501b3',
    'panel_native_v2.py':'1f3388271c3bfcb58dbc66b4b8a1c07091f3ce5b0e4b7f781d7de1cca94ce3f3',
    'panel_watch.py':'56793bf236fb43ef363abbff7cf8d323a828715947a625ec01b4ecd1c469335f',
    'panel_watch_v2.py':'b9a37cc507ccfdf36ca8375d2cbfb7fb13b1b3452507501538fcfe0f9d7f7b92',
    'PANEL_PROTOCOL_V2.md':'cc4bd6dfb06d4903ad2c54fd1a222132b5b004eac95ded498ac2765e7df69233',
    'PANEL_WATCH_SPEC_V2.md':'7df0525fccd8988a9e349d0e30e6df904eca6de5901da51e3b23fde2fee366e1',
    'panel_hash_v2.py':HASH_SHA,
}


def _load(path,name,digest):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
        raise RuntimeError('Frozen metadata owner changed: '+str(path))
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


d = _load(HERE/'native.py','apush_panel_native_metadata',APUSH_NATIVE_SHA)
e = _load(COMP/'endpoint_v2.py','apush_panel_endpoint_metadata',ENDPOINT_OWNER_SHA)
require, read, write, bind, load, splice = d.require,d.read,d.write,d.bind,d.load,d.splice
environment, function_source = d.environment,e.function_source
OFFPY, BLAS = d.OFFPY,d.BLAS
_hash_owner = _load(CD/'panel_hash_v2.py','apush_panel_hash_metadata',HASH_SHA)
METADATA_HASHES = _hash_owner.MetadataHashes(P2)
sha = METADATA_HASHES.sha
_adapt = splice
d.sha = e.sha = sha
ROOT = HERE/'panel_v1'
ADMISSION, OUTPUT, WATCH_OUT = ROOT/'admission',ROOT/'terminal',ROOT/'terminal_watch'
SCORE_OUT, SCORE_WATCH_OUT = ROOT/'score',ROOT/'score_watch'
PUBLIC = (35,100,150,200,250,300,400,500)
VIRGIN = (3016500,3016501,3016502,3016503,3016504,3016505)
RANKS = VALID = PUBLIC+VIRGIN
SEQUENCE = RANKS+(35,)
FIXED = {'terminal':RANKS}
PREFIX = 'current_comp_aligned_push_d6_independent14_'
CONTEXT_SCHEMA, ADMIT_SCHEMA = PREFIX+'context_v1',PREFIX+'admission_v1'
NATIVE_SCHEMA, WATCH_SCHEMA = PREFIX+'native_v1',PREFIX+'watch_v1'
RETIRED_SCHEMA = PREFIX+'admission_watch_v1'
CONTEXT_LIMIT, BUDGET, RSS_LIMIT = 2*1024**2,2**41,8_000_000_000
PROTOCOL_SHA = '9ff7c4591af0d7931dda4dba084d43db4a898ad5d2ef82b40a2255e28b236c47'
CANDIDATE, CANDIDATE_SHA = HERE/'build_v1/estimator.py',d.CANDIDATE_SHA
BUILD_SHA, BUILDER_SHA, RUNTIME_SHA, INIT = d.BUILD_SHA,d.BUILDER_SHA,d.RUNTIME_SHA,d.INIT
EXTRA_LAYOUTS, EXTRA_BINDINGS = d.EXTRA_LAYOUTS,d.EXTRA_BINDINGS
CENSUS, CENSUS_SHA = d.CENSUS,d.CENSUS_SHA
validate_dispatches = d.validate_dispatches
OWNER_FILES = dict(watcher='panel_watch.py',worker='panel_native.py',admitter='panel_admit.py',
    helper='panel_common.py',protocol='PANEL_PROTOCOL.md',watch_spec='PANEL_WATCH_SPEC.md')
SCORE40_SOURCES = {
    'native40.py':'cae10c7f225c9ea34e60e439508582e93a933cac4ee3b9b4c0abb822436c2ee9',
    'NATIVE40_PROTOCOL.md':'e261df146c48ce7b68e93f7b5800a842e0125a36e953635e33905d576d81743d',
    'watch_native40.py':'429b9e3de389030e3a6d3326cc76a3abdef26432776d5b88c6056e194b5d277b',
    'NATIVE40_WATCH_SPEC.md':'4a74b80e92691b414093b7a4133b4ccfebb351ae75118d40986b6a3e857d3f72',
    'score40.py':'84dc74b9c89aec51c8dede5044ae5a98512dfd72b1ccc200077b46623d683561',
    'SCORE40_PROTOCOL.md':'15a7e922979091cd864f5cac1af06ffb177fa80e3b5a37329dc96f43284590f5',
    'watch_score40.py':'1a47582123c3f5365e9dbac9f41ee3d3805f824c3380fc6288ffc888baad1c64',
    'SCORE40_WATCH_SPEC.md':'3eab3e1c90110086faaef5076f8605f207262aee462baaa27d275e26527c106e',
}
APUSH_CHAINS = {
    'score40_v1':('b9667c17e474f99e394481169f9081d5e3320ca74255e2e3e411b7eaf9995279',
        '0d81736f3f104eb8efeb8c64b1f1ff481e573b85d04ac59469e4c6dbef478c81',
        'a4c9386b43bb5fc73a3a8ddaf0b18a175cefd46c584ebe4a8691cfbeff18ad67'),
    'score40_watch_v1':('577c84e81fe81e0f06a199463ffe0ccd68f215f50166a9cd84b2321ef7d9cf14',
        '7d6eea910c417aeb6b588c3708c5ef6f2a7ee75918be1d5b1c50fc0b665ac19f',
        '875aa37234f3e45bb2ec365e1d688424a7357dd96b59318b99249534627f0655'),
    'native40_terminal_v1':('23b88f12449442cad00eb352e2aca1d90d277120b7ed348f800a9bcff64684a7',
        '9cba07e8d9e7531a7b8bd0c9842e6f44c2d258f089bdd5c3253242a6a5a3ba42',
        'dee0a6eb80aba2b5bb56b71752dfd83e23f64f7e8b9a8985cbd9f27dce5b307e'),
    'native40_terminal_admission_v1':('0de1d40c7c2db22c87c8105f0de6e88a6403bda767f030dae316701bb0698273',
        '77df28d1618cf1672eef93e1923f4401c40516508e410d1b01ff386bb64902cd',
        'a5a63ab8582c740ef1fb0a353a77b542553d6e1b984bca3e352238e80b42e547'),
    'native40_watch_v1':('8665b4705c07d14f7f315ee1e1229154177fc86e3345f12d7029fe31ed8b526b',
        '669bf57b4e3efa6cafac6b48d8592e4a02e3907ee6f818c0fe5d101bfe3c479a',
        'd68ac6dcd781d28490a4271e8480a98e240056b7ce1e9bbb1f7bd62d5f1a91d9'),
}
NATIVE40_CONTEXT_SHA = '85dbaa586ce173ea05892a0b2c9e2dc34f108883424ef6ce415179768dfeeeb4'
SCORE_SHA, SCORE_WATCH_SHA = APUSH_CHAINS['score40_v1'][0],APUSH_CHAINS['score40_watch_v1'][0]
CD_CHAINS = {
    'admission':('2ffa9e352d267d8f67147db4faa81941c13424981003cd1b7c85c636be939898',
        '49f449322530ceb5837a14ca687937486c8fc499a3a3bc35d803d95357da18fe',
        'b9ca38d6e89f6faaa8c14711f7f981a3559ffab6f7a0f9517d91c3890fbf271b'),
    'terminal':('3aaf29635c3822067f37a9cc934e1f8ae2692beef48be4538c6caa43189cc7f1',
        'cdd8284fb933c99e1934c3b58df2933a0ad5e1d003cdc5f889a32e1ab2fc5015',
        'f3739a6648a8845fdc3c7fd0efb6725c8daf349779f9ad702f74c624743b5e05'),
    'terminal_watch':('376323c876acc78594f7f23caa67c2a0388ea32374e540c75a8bec8273ec5f94',
        'b1f7ecaa4b08aac4f73cad2a32ce1fad5e0701bd1c23948be4710f6dd68ca4a8',
        '12eb30c865691bf01659778a4d63b1bcc522da348bcb2309bfade3a91a5d9e7f'),
    'score':('aab95242f83cc9a0a1937a2e6b195830e1ef8d1bc8dc4366101d732426998481',
        '55097b43846618ef9e1f81df27086064a57b1149c32b93f446a630d6bfca70bd',
        '9c0370e2143e76d41b749ca3928a6d8b3f69b825fa25a1eeb16460373ba8d5cd'),
    'score_watch':('653ace788f0474335cd2c20d04b385069e521f2f625e584c2a2a73f19c787a2c',
        'ddaeaf3d278a8916021653ef3bf5aee18193d82f82af19a046d852ac4b2beead',
        '57d6ac57ab8e05dd2a98f967567dbe3607b9ca6e802ea6cc038ec5b3dfe9ddbe'),
}
CD_CONTEXT_SHA = '3ec7d569e15287497e9ec14fce87df4dad2ec1725cb6d7b6c6d50a08f5660a47'
CD_RETIRED_SHA = '654b7ee68db4d2908337e42646e6f82ed8fa07a791083fa88dc9915b885525dc'
CONTROL = CD/'panel_v2/terminal'
CONTROL_SCORE, CONTROL_WATCH = CD/'panel_v2/score',CD/'panel_v2/terminal_watch'
CONTROL_SHA, CONTROL_END_SHA = CD_CHAINS['terminal'][:2]
CONTROL_WATCH_SHA, CONTROL_SCORE_SHA = CD_CHAINS['terminal_watch'][0],CD_CHAINS['score'][0]
CONTROL_SCORE_WATCH_SHA = CD_CHAINS['score_watch'][0]
_COMP_HELPER = _CALLBACKS = None
_RESOLVED = {}
SOURCE_PROOFS = {}


def set_hash_guard(guard):
    require(guard is None or callable(guard),'Stage-local metadata deadline only')
    METADATA_HASHES.guard = guard


def final_bind(mappings,tick=None):
    return METADATA_HASHES.final_bind(mappings,tick)


def assignment(source,name):
    matches = [n.value for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id == name for t in n.targets)]
    require(len(matches) == 1,'Unique source-owned metadata assignment '+name)
    return matches[0]


def resolved_owner(kind):
    """Resolve only the pinned v2 source expression, never its module/stage."""
    require(kind in ('common','admit','native'),'Only selected qualified source owners')
    if kind not in _RESOLVED:
        first,second = 'panel_'+kind+'.py','panel_'+kind+'_v2.py'
        require(sha(CD/first) == CD_OWNERS[first] and sha(CD/second) == CD_OWNERS[second],
            'Both frozen CD source versions')
        original,wrapper = (CD/first).read_text(),(CD/second).read_text()
        env = dict(_original=original,_adapt=splice,c=SimpleNamespace(_adapt=splice))
        if kind == 'common': env['_PROTOCOL_SHA'] = ast.literal_eval(assignment(wrapper,'_PROTOCOL_SHA'))
        if kind == 'native': env['_fresh_finish'] = ast.literal_eval(assignment(wrapper,'_fresh_finish'))
        derived = eval(compile(ast.Expression(assignment(wrapper,'_source')),str(CD/second)+':source-only-expression','eval'),env)
        ast.parse(derived)
        SOURCE_PROOFS['CD_'+kind] = dict(owner_sha256=CD_OWNERS[second],parent_sha256=CD_OWNERS[first],
            original_sha256=hashlib.sha256(original.encode()).hexdigest(),
            derived_sha256=hashlib.sha256(derived.encode()).hexdigest(),whole_source_inverse_verified=True,
            no_original_module_or_stage_executed=True)
        _RESOLVED[kind] = derived
    return _RESOLVED[kind]


def function_source_text(source,name):
    return d.function_source(source,name)


def install(kind,names,namespace,changes=None):
    source = resolved_owner(kind)
    for name in names:
        original = function_source_text(source,name)
        derived = splice(original,(changes or {}).get(name,[]))
        ast.parse(derived)
        SOURCE_PROOFS[kind+'.'+name] = dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),
            derived_sha256=hashlib.sha256(derived.encode()).hexdigest(),source_inverse_verified=True)
        exec(compile(derived,str(CD/('panel_'+kind+'_v2.py'))+':exact-selected-function','exec'),namespace)


install('common',('comp_helper',),globals())
pc = comp_helper()
# Private metadata hash slots only: no global filesystem/hashlib or model patch.
pc.pin_to.__globals__['sha'] = sha
pc.sha = sha
e._OLD = d.base()
for module in (d.owner(),d.base()): module.sha = sha
COMP_OWNERS = eval(compile(ast.Expression(assignment((CD/'panel_common.py').read_text(),'COMP_OWNERS')),
    str(CD/'panel_common.py')+':source-constants','eval'),dict(COMP_HELPER_SHA=COMP_HELPER_SHA))
P1, TL, BD = pc.P1,pc.TL,pc.BD
WATCH_OWNER_SHA, NATIVE_OWNER_SHA = pc.WATCH_OWNER_SHA,pc.NATIVE_OWNER_SHA
MONITOR_DIR, MONITOR_SHA = pc.MONITOR_DIR,pc.MONITOR_SHA
MONITOR_TEST_SHA, MONITOR_PROTOCOL_SHA = pc.MONITOR_TEST_SHA,pc.MONITOR_PROTOCOL_SHA
MONITOR_TEST_RESULT_SHA, MONITOR_TEST_END_SHA = pc.MONITOR_TEST_RESULT_SHA,pc.MONITOR_TEST_END_SHA
INVENTORY_SHA = pc.INVENTORY_SHA
pin_to, inventory, sealed = pc.pin_to,pc.inventory,pc.sealed
transport_admission, panel_modules = pc.transport_admission,pc.panel_modules
_common_v2 = (CD/'panel_common_v2.py').read_text()
exec(compile(function_source_text(_common_v2,'require_fresh_union'),str(CD/'panel_common_v2.py')+':fresh-union-contract','exec'),
    dict(globals(),_require=require),_fresh_namespace := {})
require_fresh_union = _fresh_namespace['require_fresh_union']


def owners(args,tick=None):
    # This original function only hashes source owners; no ancestor admission.
    original_args = SimpleNamespace(**{key+'_sha256':digest for key,(_,digest) in COMP_OWNERS.items()})
    values = pc.owners(original_args,tick)
    for mapping in (d.OWNERS,d.owner().OWNERS):
        values.update({str(path):digest for path,digest in mapping.items()})
    values.update({str(CD/name):digest for name,digest in CD_OWNERS.items()})
    values.update({str(HERE/name):digest for name,digest in SCORE40_SOURCES.items()})
    values.update({str(HERE/'native.py'):APUSH_NATIVE_SHA,str(COMP/'endpoint_v2.py'):ENDPOINT_OWNER_SHA})
    values.update({str(HERE/name):getattr(args,key+'_sha256') for key,name in OWNER_FILES.items()})
    bind(values,tick); return values


def prerequisites():
    return dict(native40_score_sha256=SCORE_SHA,native40_score_watch_sha256=SCORE_WATCH_SHA,
        native40_sha256=APUSH_CHAINS['native40_terminal_v1'][0],native40_watch_sha256=APUSH_CHAINS['native40_watch_v1'][0],
        control_sha256=CONTROL_SHA,control_watch_sha256=CONTROL_WATCH_SHA,
        control_score_sha256=CONTROL_SCORE_SHA,control_score_watch_sha256=CONTROL_SCORE_WATCH_SHA,
        candidate_sha256=CANDIDATE_SHA,build_sha256=BUILD_SHA,runtime_sha256=RUNTIME_SHA)


def direct_chain(folder,digests,pins,tick=None,limit=1200):
    result,end,final = d.seal3(folder,digests,pins,tick=tick)
    require(result.get('hashes_unchanged') is True and result.get('sources_unchanged') is True
        and 0 <= final['elapsed_s'] < limit,'Direct positive three-seal boundary and complete clock')
    inventory(result,pins,tick)
    return result,end,final


def seal_map(digests):
    return dict(zip(('result_sha256','completion_sha256','finalization_sha256'),digests))


def watch_boundary(record,child,limit,rss,*,cd=False):
    require(all(record.get(k) is True for k in ('passed','completed','hashes_unchanged','sources_unchanged',
        'whole_owned_tree_retired','child_execution_completed','observational_groups_retired',
        'actual_completed_samples','foreign_numerical_workers_checked'))
        and tuple(record['child_'+k] for k in ('result_sha256','completion_sha256','finalization_sha256')) == tuple(child)
        and 0 <= record['elapsed_s'] < limit and record['maximum_swap_growth_bytes'] == 0,
        'Actual completed independent supervisor, child three seals, whole retirement and swap')
    interval = record['maximum_sample_interval_s' if cd else 'maximum_interval_s']
    memory = record['maximum_sampled_rss_bytes' if cd else 'maximum_combined_rss_bytes']
    intervals = record['completed_sample_intervals_s']
    require(0 < memory < rss and intervals and 0 < interval <= 1 and max(intervals) == interval
        and all(math.isfinite(x) and 0 <= x <= 1 for x in intervals),'Actual completed observation intervals/resources')


install('common',('arguments','command','seal3','metadata_sources','callbacks','after_bill',
    'finish_function','paid_source','read_context','retired_admission','verify_native','verify_watch'),globals(),{
    'arguments': [('Unchanged COMP/partial-D6 independent14 ','Unchanged APUSH independent14 ')],
    'paid_source': [("native_original_main_sha256=proof['paid_loop']['original_paid_main_sha256']",
        "native_original_main_sha256=proof['parent_CD']['paid_loop']['original_paid_main_sha256']")],
    'read_context': [
        ("value['build_watch_sha256'] == BUILD_WATCH_SHA","value['runtime_sha256'] == RUNTIME_SHA"),
        ("value['COMP_panel_comparator_qualified']","value['CD_panel_comparator_qualified']"),
        ('actual COMP panel','actual CD panel')],
})


def cd_callbacks(context):
    """Original narrow CD validator on its own already completed15 packets only."""
    owner = d.owner()
    env = dict(globals(),d=owner)
    # Reuse the exact same roster adaptation, applied to the original CD source.
    exec(compile(function_source_text(resolved_owner('common'),'metadata_sources'),
        str(CD/'panel_common_v2.py')+':CD-comparator-metadata','exec'),env)
    source,proof = env['metadata_sources']()
    require(proof == context['proof']['panel_metadata'],'Exact completed CD panel metadata proof')
    env = dict(vars(owner),sha=sha,VALID=RANKS,OUTPUT=CONTROL,
        NATIVE_SCHEMA='current_comp_partial_d6_independent14_native_v2',
        CANDIDATE_SHA=d.CD_SHA,BUILD_SHA=context['build_sha256'],BUILDER_SHA=context['builder_sha256'])
    exec(compile(source,str(CD/'panel_common_v2.py')+':CD-comparator-only','exec'),env)
    return env


def source_proof():
    return dict(functions=dict(SOURCE_PROOFS),private_metadata_hash_slots_only=True,
        exact_generic_APUSH_observer=True,no_fixed_CD_shape_tariff_for_APUSH=True,
        no_ancestor_admission_or_score_execution=True,original_failed_watch_preserved=True,
        no_estimator_or_array_operator_patch=True)
