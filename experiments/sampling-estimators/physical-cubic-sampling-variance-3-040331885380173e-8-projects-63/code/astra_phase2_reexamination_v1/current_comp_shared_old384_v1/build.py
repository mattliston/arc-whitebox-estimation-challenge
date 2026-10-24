"""Inert source materializer for the unchanged, already probed shared card."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent,HERE.parent.parent
OFFPY = P2/'data_gate/.venv/bin/python'
OUT = HERE/'build_v1'
SCHEMA = 'current_comp_shared_old384_build_v1'
CANDIDATE_SHA = '4d98fef2c11c571efe1eca945018bb9fe9ece1454d16b4abde2eb52d71e755c3'
COMPOSE = HERE/'compose.py'
COMPOSE_SHA = 'fdf5bd87eb6411d008043f5e989a670979cde20965366dca212146c075ee1d4e'
OWNER = NEW/'current_comp_padded_push_d6_v1/build.py'
OWNER_SHA = '63ae8a312de114ab03e02e91916e63f997c81475104a0d8351da7dc652aad6f7'
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')
CHAINS = {
    'probe_v1':('016e81e7bd0750f0e99671ac4def7deb02f591a2a292d07185eee25b7e2c591f',
        'd0fffc724a84d330b7db629cb59cb5a321fd36e91f787b533688746c855aba02',
        '0b0e98f17218eb23260f95a8fba4dddfca2b706ae4aa328cc437641c29c29d51'),
    'probe_watch_v1':('735a6efcd8206dd77a9eaa3a940d79cc321232a2993cb25643dbf079b9be7b42',
        '95754bb3d25dfb2088f8812d7ce2ebe418b0510a241f774de6132d93f6630d9a',
        '56fe2953128820874809e4a11f2798a3ac13be669d8cbdd1558c81f7498c7368'),
    'score_probe_v1':('d00066a38fc9c6012282c80a3a3d7f9a97f420b137f67e1daae5473589b88dd0',
        'c5f5fe56d376558154ee0aa16cc47789ec8054768f5570a2409e1848a32a69ce',
        '10c6ccc33527499885662a1c8c2638a4839d7995047c5174a38fade253adf79d'),
    'score_probe_watch_v1':('ea4a6847a04d82b159d3dbcea51d801e80b84628f1d11eebc22be44322de29f5',
        '7958912d3cb50a1de35840d2387df545b590dcd1b1280ed459cef24fff8fc477',
        '701b1bdc06927cab4242bd05f51680c36c4db5eb069cb5bd9eb99f241b5b00c1'),
}
SEALS = ('result.json','completion.json','finalization.json')


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def no_numerics():
    require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules), 'No numerical imports')


def sha(path,tick=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(P2)
        and path.suffix not in ('.npy','.npz','.f32') and 'truthpost' not in path.parts,
        'Only regular in-scope source/metadata; no numerical-array reads')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            h.update(block)
            if tick is not None:
                tick()
    return h.hexdigest()


def read(path):
    path = Path(path)
    require(path.suffix == '.json' and not path.is_symlink() and path.stat().st_size < 8_000_000,
        'Bounded source/receipt metadata')
    return json.loads(path.read_text())


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


def composer(tick=None):
    no_numerics()
    require(sha(COMPOSE,tick) == COMPOSE_SHA,'Exact frozen, inert source composer')
    spec = importlib.util.spec_from_file_location('shared_original_source_composer',COMPOSE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    no_numerics()
    return module


def admission(args,tick):
    require(all(os.environ.get(k) == '1' for k in BLAS),'All six original thread variables1')
    sources = {str(HERE/'build.py'):args.source_sha256,str(HERE/'BUILD_PROTOCOL.md'):args.protocol_sha256,
        str(COMPOSE):COMPOSE_SHA,str(OWNER):OWNER_SHA}
    inputs = {}
    records = {}
    for name,expected in CHAINS.items():
        paths = [HERE/name/s for s in SEALS]
        actual = tuple(sha(p,tick) for p in paths)
        require(actual == expected,'Actual completed positive source-selection boundary '+name)
        r,e,f = map(read,paths)
        require(all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
            and e['result_sha256'] == f['result_sha256'] == actual[0]
            and f['completion_sha256'] == actual[1] and r['hashes_unchanged'] and r['sources_unchanged'],
            'Actual three-seal links, no earlier false receipt relabeling')
        inputs.update({str(p):h for p,h in zip(paths,actual)})
        records[name] = r
    for child,watch in (('probe_v1','probe_watch_v1'),('score_probe_v1','score_probe_watch_v1')):
        r = records[watch]
        require(tuple(r['child_'+k] for k in ('result_sha256','completion_sha256','finalization_sha256'))
            == CHAINS[child] and r['whole_owned_tree_retired'] and r['cleanup_passed']
            and r['observational_groups_retired'] and r['actual_completed_samples']
            and r['maximum_swap_growth_bytes'] == 0 and 0 < r['maximum_interval_s'] <= 1,
            'Actual direct positive watch and retirement authority')
    p,s = records['probe_v1'],records['score_probe_v1']
    require(p['candidate_sha256'] == CANDIDATE_SHA and p['ranks'] == [37,37]
        and p['full_model_calls'] == p['retired_call_count'] == 2 and p['initialization_flops'] == 1944970298
        and all(p[k] is True for k in ('all16_domains_qualified','all47_selectors_qualified',
            'repeat_byte_exact','own_complete_bills')) and p['truth_accessed'] is False,
        'The unchanged candidate has the actual two-call native source qualification')
    require(s['information_promising'] and s['scientific_gate_passed']
        and s['comparison']['adjusted_ratio'] <= .95 and s['exact_APUSH_score_replay']
        and s['lineage']['probe']['result_sha256'] == CHAINS['probe_v1'][0],
        'Original prospective probe gate, not CHECK4 or production authority')
    # These pinned complete receipts are direct boundaries. Do not open their
    # numerical artifacts/targets or reconstruct earlier native/training stages.
    for p,h in composer(tick).authenticate(tick).items():
        (sources if Path(p).suffix in ('.py','.md') else inputs)[p] = h
    for p,h in {**sources,**inputs}.items():
        require(sha(p,tick) == h,'Complete direct source/metadata admission')
    no_numerics()
    return sources,inputs


def derive(tick):
    text,proof = composer(tick).derive_source(tick)
    require(hashlib.sha256(text.encode()).hexdigest() == proof['composed_source_sha256'] == CANDIDATE_SHA
        and proof['full_source_inverse_verified'] and proof['physical_function_inverse_verified']
        and proof['parent_arena_and_dispatchers_unchanged'] and proof['no_import_time_helper_array_work']
        and proof['weight_rule'] == 'actual_FP32_lambda_squared' and proof['rank'] == 384
        and proof['first_join_layer'] == 5 and proof['last_join_layer'] == 14
        and proof['fitted'] is False and proof['extra_incumbent_plane_repair'] is False,
        'Exactly the already probed source, not a new production estimator')
    return text,dict(proof,candidate_sha256=CANDIDATE_SHA,materialization_only=True,
        production_fidelity_qualified=False,production_timing_qualified=False)


def build_source():
    """Reuse the completed PAD source-only publication/guard function exactly."""
    require(sha(OWNER) == OWNER_SHA,'Exact source-only build lifecycle owner')
    original = OWNER.read_text()
    nodes = [n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name == 'build']
    require(len(nodes) == 1,'One original guarded source materializer')
    n = nodes[0]
    source = ''.join(original.splitlines(keepends=True)[n.lineno-1:n.end_lineno])
    return source,dict(owner_sha256=OWNER_SHA,guarded_build_body_byte_exact=True,
        source_sha256=hashlib.sha256(source.encode()).hexdigest())


def verify_build(result_sha,source_sha,protocol_sha,tick=None):
    """Target-free direct build revalidation; no composer/estimator execution."""
    no_numerics()
    paths = [OUT/s for s in SEALS]
    hashes = [sha(p,tick) for p in paths]
    require(hashes[0] == result_sha,'Actual future build result pin')
    r,e,f = map(read,paths)
    require(all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
        and e['result_sha256'] == f['result_sha256'] == hashes[0] and f['completion_sha256'] == hashes[1]
        and r['schema'] == SCHEMA and r['candidate_sha256'] == CANDIDATE_SHA
        and r['hashes_unchanged'] and r['sources_unchanged'] and r['parameter_updates'] == 0
        and all(r[k] is False for k in ('estimator_executed','estimator_imported','truth_accessed',
            'truth_arrays_accessed','score_measured','numerical_imports'))
        and 0 <= r['elapsed_s'] < 60 and 0 <= f['elapsed_s'] < 60
        and 0 < r['peak_rss_bytes'] < 2_000_000_000,'Actual source-only complete build')
    require(r['source_bindings'][str(HERE/'build.py')] == source_sha
        and r['source_bindings'][str(HERE/'BUILD_PROTOCOL.md')] == protocol_sha
        and r['artifacts'] == {str(OUT/'estimator.py'):CANDIDATE_SHA}
        and r['proof']['composed_source_sha256'] == CANDIDATE_SHA
        and r['proof']['materialization_only'] and r['proof']['full_source_inverse_verified'],
        'Exact reviewed literal source artifact and inverse')
    pins = {}
    for key in ('source_bindings','input_bindings','artifacts'):
        require(isinstance(r[key],dict) and r[key],'Complete actual non-target build inventory')
        for p,h in r[key].items():
            require(p not in pins or pins[p] == h,'Consistent immutable build inventories')
            require(sha(p,tick) == h,'Fresh actual build source/input/output')
            pins[p] = h
    pins.update({str(p):h for p,h in zip(paths,hashes)})
    return r,dict(zip(('result_sha256','completion_sha256','finalization_sha256'),hashes)),pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--source-sha256'); parser.add_argument('--protocol-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    require(all(re.fullmatch('[0-9a-f]{64}',v or '') for v in (args.source_sha256,args.protocol_sha256)),
        'Explicit reviewed builder and protocol hashes')
    no_numerics()
    source,_ = build_source()
    scope = dict(globals())
    exec(compile(source,str(OWNER)+':exact-build-lifecycle','exec'),scope)
    return scope['build'](args)


if __name__ == '__main__':
    raise SystemExit(main())
