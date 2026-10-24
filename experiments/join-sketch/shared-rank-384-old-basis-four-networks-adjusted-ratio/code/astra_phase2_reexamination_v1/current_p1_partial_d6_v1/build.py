"""Inert stdlib-only fixed-shape serial-D6 source composition on P1."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
P1 = NEW/'current_d50_native_objective_p1_v1'
BD = NEW/'birth_direction_fit_v1'
KERNEL = NEW/'current_d50_serial_d6_kernel_v1'
PROBE = NEW/'current_p1_d6_push832_probe_v1'
CENSUS = NEW/'current_d50_d6_cost_census_v1'
TL = NEW/'current_d50_terminal_linear_v2'
OUT = HERE/'build_v1'
PARENT = P1/'build_v1/estimator.py'
PARENT_SHA = '0deb237350e753220b575796c0278f86884b8461bfeec4ab53eae9c15522c435'
BANK_SHA = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'
KERNEL_SHA = 'a8d1cb1a972b37afd9eaaa67acf820cba3196dabea10ba1058bb3409c0b20ec7'
SHAPE = (1024,832,8192)
EXTRA_BINDINGS = 561056000
SAVING_PER_HIT = 1355278064
SOURCE_PINS = {
    PARENT:PARENT_SHA,
    BD/'build_terminal_v1/estimator.py':BANK_SHA,
    KERNEL/'kernel.py':KERNEL_SHA,
    NEW/'packed_arena_generic.py':'4e4a3dea533c222f34ba3a08538245414e149dbc0f918fd097267b0f869fb27e',
    KERNEL/'PROTOCOL.md':'888674b8ebfb4fe3f0e1caa071d9528d69c45c880967ffbbb827167536212477',
    KERNEL/'RESULT.md':'d87c5d29e8eb37a8192f19902c9e92298458435da5d0816e396febdfa0814125',
    PROBE/'PROTOCOL.md':'50dca9fe2ffab1bb51464cb98cbf1d643458abeedf750f9feb452484d1b120ca',
    PROBE/'client.py':'8ea425a33de0fb1aca6f3ae0e5c38f2d8991b68b5f6d27eae69969715ed8328f',
    PROBE/'run.py':'209962dc9aa51387bae6d1802662d02072df09d7f437c943344a6f768e9f2491',
    CENSUS/'PROTOCOL.md':'25e054db244f1edb2bef96afbf4c0b7af274dd5991a4f94352727be7606e3075',
    CENSUS/'run.py':'7e278ed04becda90602773ab607538b0a2330b1e597766a52a391cc6794704cc',
    P1/'native_v2.py':'fad59d76d82baf40c823e0468b7304a909b719f8bc4a8c1f5c4e7118a4f9d9e8',
    TL/'native.py':'744f8c46bfdbe0198eb138c1fcacc5a0b90ca6db6eb08ad7d928847c34021abe',
}
INPUT_PINS = {
    P1/'build_v1/result.json':'877990bc84cb85431ff98f83e1e7012517c8b7914a1e0b39b29bf7d0b776b0c1',
    P1/'build_v1/completion.json':'cdcb0a934224f242cad404dec3bae7eba60a712d8bd430ace0c7ab7111faea5c',
    P1/'score2_v1/result.json':'4f60a8def29f1a9326d68d5c25bbe79b3f26289af8cf46de81609fb91cf055e5',
    P1/'score2_v1/completion.json':'c72831702f18a70a4bc40a8d88a655879be3e816e3edd3351cce9ea75e8cd587',
    P1/'panel_v1/score/result.json':'81ebbb54207e4f0060aaa1a91320c17de89b6fdd53e7d5326c17993829e6e526',
    P1/'panel_v1/score/completion.json':'ee2df444d7e3984993e6367f707b3ffd3a8408a182f85ca9dc7701341167f309',
    P1/'native2_terminal_v1/result.json':'7a1f37d80a7ed32850b10047408d85eed050220b4a6271dde90bdd21c07062e3',
    P1/'native2_terminal_v1/completion.json':'b73a957bf78e45da0db2741cea0b33a2cd98317726edaaa0954eaece054e5957',
    KERNEL/'run_v1/result.json':'86ce7e8b1ea3d9dcc81c42bad6cb78828b520f38aed5f4918acf6052a6cf8660',
    KERNEL/'run_v1/completion.json':'caafbeb845cd7277e9c408e4ef824759a8bc521831dd7150c356882e03cf169e',
    PROBE/'run_v1/result.json':'bf79b160f8952db1f433093c245c5497d2d25a623db6888cafa13f0ae48aaeb2',
    PROBE/'run_v1/completion.json':'6fafd4fa5d8e5e1ce7f6f4ca8886919404ff35eb1417f5c8d7c9238cda1b6b5f',
    CENSUS/'native/result.json':'d3d080115b8b27ad5af26196d045c3324db2c5c44496357b2ab8f4ec658666c2',
    CENSUS/'native/completion.json':'a1898f0b068ec74cd16250e71b20bb10bcdbc9bd6e53bf759ec277cd9f253496',
    CENSUS/'native/call_00_tiles.json':'42bf244f40c1e9964b9a722253f06d951240c1e06c483d709a146119ff257a79',
    CENSUS/'price.json':'995bd57f77d526cee191ec407473946c1d3116ed7390754ba635485458c76c72',
    TL/'qualify_v1/native_0035/result.json':'292b6e5f42ae5a57ce4a1414c4e471cb59a4756441b8ed2053901f31aeaafd24',
    TL/'qualify_v1/native_0035/completion.json':'def887e8eb3e1b261060882faf01400b0ded2c581535f75365c775d5bc3bb0bb',
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    path = Path(path).resolve()
    require(path.is_relative_to(NEW) and path.suffix in ('.py','.json','.md'),
            'Source and JSON receipts only; no weights/predictions/targets')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            digest.update(block)
    return digest.hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def read(path):
    require(Path(path).suffix == '.json', 'Only receipt metadata is decoded')
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


def bind(mapping):
    for path,digest in mapping.items():
        require(isinstance(digest,str) and len(digest) == 64 and
                set(digest) <= set('0123456789abcdef') and sha(path) == digest,
                'Exact reviewed source/receipt: '+str(path))


def receipt(folder, completed=True):
    record,end = read(folder/'result.json'),read(folder/'completion.json')
    require(record['passed'] is True and end['passed'] is True and
            end['result_sha256'] == INPUT_PINS[folder/'result.json'], 'Completed receipt seal')
    if completed:
        require(record['completed'] is True and end['completed'] is True, 'Explicit modern completion')
    return record


def admission(source_sha, protocol_sha):
    require(sys.flags.optimize == 0 and sys.dont_write_bytecode and not any(x in sys.modules for x in
            ('numpy','torch','flopscope')), 'Assertions on, bytecode off; no numerical import before source admission')
    sources = {str(p):v for p,v in SOURCE_PINS.items()}
    sources.update({str(Path(__file__).resolve()):source_sha,str(HERE/'PROTOCOL.md'):protocol_sha})
    inputs = {str(p):v for p,v in INPUT_PINS.items()}
    bind(sources); bind(inputs)
    built = receipt(P1/'build_v1')
    require(built['candidate_sha256'] == PARENT_SHA and built['inverse_source_byte_exact']
            and built['cstar_byte_exact'] and built['null_source_byte_exact'], 'Original frozen P1 build')
    score,panel = receipt(P1/'score2_v1'),receipt(P1/'panel_v1/score')
    require(score['candidate_sha256'] == panel['candidate_sha256'] == PARENT_SHA and
            score['integrity_passed'] and score['advances_to_independent14'] and
            panel['qualification_passed'] and panel['retain_local_component'] and
            all(panel['group_strict_gain_gate'][g] for g in ('public8','virgin6')), 'Retained P1 evidence')
    native = receipt(P1/'native2_terminal_v1')
    require(native['candidate_sha256'] == PARENT_SHA and native['truth_accessed'] is False and
            native['own_complete_bills'] and native['all16_domains_qualified'] and
            native['all47_selectors_qualified'], 'Actual frozen P1 geometry and complete bills')
    old = receipt(KERNEL/'run_v1',completed=False)
    require(old['completed'] and old['single_product_qualification_passed'] is False and
            old['timing_gate_passed'] is False, 'Old maximum-read timing failure remains FAILED')
    probe = receipt(PROBE/'run_v1')
    require(probe['shape'] == list(SHAPE) and probe['numerical_and_billing_gate_passed'] and
            probe['timing_gate_passed'] and probe['partial_shape_test_admitted'] and
            probe['additional_serial_setup_flops'] == EXTRA_BINDINGS and
            probe['whole_owned_tree_retired'] and probe['maximum_swap_growth_bytes'] == 0 and
            probe['maximum_sampled_rss_bytes'] < 8_000_000_000 and
            probe['maximum_sample_interval_s'] <= 1 and probe['incremental_six_call_wall_s'] <= 8 and
            probe['truth_accessed'] is False and probe['whole_estimator_qualified'] is False,
            'Exact fixed-push qualification, not whole-estimator qualification')
    census = receipt(CENSUS/'native',completed=False)
    price = read(CENSUS/'price.json')
    require(census['candidate_sha256'] == BANK_SHA and census['truth_accessed'] is False and
            census['hashes_unchanged'] and price['passed'] and price['completed'] and
            price['native_result_sha256'] == INPUT_PINS[CENSUS/'native/result.json'], 'Preserved actual d50 census')
    for record in (native,probe,census):
        for path,digest in sources.items():
            if path in record.get('source_bindings',{}):
                require(record['source_bindings'][path] == digest, 'Receipt source-owner consistency')
    return sources,inputs


def hits(cells):
    keys = {f'{l}:{s}' for l in range(16) for s in range(2 if l == 15 else 3)}
    require(set(cells) == keys and all(type(x['count']) is int and 0 <= x['count'] <= 1024 and
            x['count'] % 32 == 0 and x['floor'] == .01 for x in cells.values()), 'Exact all47 count chart')
    rows = []
    for layer in range(1,16):
        active = cells[f'{layer-1}:0']['count']
        inherited = 2*sum(cells[f'{j}:2']['count'] for j in range(layer-1))
        count = inherited//8192 if active == 832 else 0
        if count:
            rows.append(dict(layer=layer,active_before=active,inherited_columns=inherited,hits=count))
    return rows


def frozen_estimate():
    record = read(P1/'native2_terminal_v1/result.json')
    trials,charts = record['trials'],record['domain_diagnostics']
    require(len(trials) == len(charts) == 41 and len(record['ranks']) == 40 and
            len(set(record['ranks'])) == 40 and trials[-1]['rank'] == trials[0]['rank'] == 37,
            'Fixed40 and repeat37')
    rows = []
    for i,(trial,chart) in enumerate(zip(trials,charts)):
        require(trial['call'] == chart['call'] == i and trial['rank'] == chart['rank'] and
                trial['includes_initialization'] is (i == 0) and type(trial['flops']) is int,
                'Same saved call/cold ownership')
        layers = hits(chart['selection_cells']); count = sum(x['hits'] for x in layers)
        rows.append(dict(call=i,rank=trial['rank'],layers=layers,hits=count,old_complete_flops=trial['flops'],
            conservative_counterfactual_flops=trial['flops']-count*SAVING_PER_HIT+EXTRA_BINDINGS))
    require(rows[0]['layers'] == rows[-1]['layers'] and
            [x['rank'] for x in rows[:40]] == record['ranks'], 'Repeat and exact primary roster')
    primary = rows[:40]; count = sum(x['hits'] for x in primary)
    old = sum(x['old_complete_flops'] for x in primary)
    conservative = sum(x['conservative_counterfactual_flops'] for x in primary)
    require(count == 172 and max(x['hits'] for x in primary) == 9 and
            [x['rank'] for x in primary if x['hits'] == 0] == [130,204], 'Source-derived saved P1 census')
    d50 = receipt(TL/'qualify_v1/native_0035')
    require(d50['passed'] and d50['completed'] and d50['candidate_sha256'] == BANK_SHA and
            d50['rank'] == 35, 'Independent original d50 selector capture')
    actual = [x for x in read(CENSUS/'native/call_00_tiles.json') if x['shape'] == list(SHAPE)]
    require(len(actual) == sum(x['hits'] for x in hits(d50['diagnostics']['selection_cells'])) == 6 and
            all(x['depth'] == 4 and x['dtype'] == 'float32' and x['flops'] == 8417538304 for x in actual),
            'Independent selector-to-actual-tile crosscheck')
    probe = read(PROBE/'run_v1/result.json')
    return dict(frozen_trajectory_only=True,candidate_score=False,candidate_bill=False,truth_accessed=False,
        formula='sum_l 1[a[l-1]==832]*floor((2*sum_{j<l-1}birth_count[j])/8192)',
        source_owner='P1 _push_bank_shared_newborn, final _predict_binary and final _mm; no state replay',
        primary_rows=primary,repeat_excluded=True,repeat_hits=rows[-1]['hits'],total_hits=count,mean_hits=count/40,
        mean_old_complete_flops=old/40,mean_conservative_counterfactual_flops=conservative/40,
        conservative_fraction_saving=(old-conservative)/old,
        conservative_both_binding_charge_per_primary=EXTRA_BINDINGS,
        once_per_process_fraction_saving=(count*SAVING_PER_HIT-EXTRA_BINDINGS)/old,
        max9_timing_heuristic_s=probe['delta_setup_s']+9*probe['delta_warm_s'],whole_card_time_certificate=False)


def nodes(source):
    return {n.name:n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}


def literal(source, name):
    found = [n for n in ast.parse(source).body if isinstance(n,ast.Assign) and
        len(n.targets) == 1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id == name]
    require(len(found) == 1, 'One unchanged literal: '+name)
    return ast.get_source_segment(source,found[0])


def replace_definition(source, name, replacement):
    node = nodes(source)[name]
    lines = source.splitlines(keepends=True)
    start = sum(map(len,lines[:node.lineno-1]))+node.col_offset
    end = sum(map(len,lines[:node.end_lineno-1]))+node.end_col_offset
    before = source[start:end]
    require(before == ast.get_source_segment(source,node), 'Exact AST-selected source span')
    changed = source[:start]+replacement+source[end:]
    return changed,before


def construct():
    require(sha(PARENT) == PARENT_SHA and sha(KERNEL/'kernel.py') == KERNEL_SHA, 'Frozen source before construction')
    parent = PARENT.read_text(); bank = (BD/'build_terminal_v1/estimator.py').read_text()
    kernel_text = (KERNEL/'kernel.py').read_text(); layout_text = (NEW/'packed_arena_generic.py').read_text()
    spec = importlib.util.spec_from_file_location('partial832_source_only_kernel',KERNEL/'kernel.py')
    kernel = importlib.util.module_from_spec(spec); spec.loader.exec_module(kernel)
    audit = kernel.audit_sources(bank,layout_text,kernel_text)
    for name in kernel.HELPER_NAMES:
        require(ast.get_source_segment(parent,nodes(parent)[name]) ==
                ast.get_source_segment(bank,nodes(bank)[name]), 'Exact P1/d50 child helper: '+name)
    require('_P1_D6_832_' not in parent and '_p1_d6_832_' not in parent, 'Unused private names')
    metadata = [kernel.metadata(kernel.PUSH_CAPACITY),kernel.metadata(kernel.READ_CAPACITY)]
    require(sum(x['binding_flops'] for x in metadata) == EXTRA_BINDINGS and
            [x['scratch_bytes'] for x in metadata] == [472026112,639320064] and
            kernel.price(*SHAPE,4) == 8417538304 and kernel.price(*SHAPE,6) == 7062260240,
            'Frozen max-capacity layouts and measured tariffs')
    binding = ast.get_source_segment(kernel_text,nodes(kernel_text)['_Binding'])
    product = ast.get_source_segment(kernel_text,nodes(kernel_text)['product'])
    renamed_binding = binding.replace('class _Binding:', 'class _P1_D6_832_Binding:',1)
    renamed_product = product.replace('def product(', 'def _p1_d6_832_product(',1)
    require(renamed_binding.replace('class _P1_D6_832_Binding:','class _Binding:',1) == binding and
            renamed_product.replace('def _p1_d6_832_product(','def product(',1) == product,
            'Frozen serial class/product inverse names only')
    bundle = '\n\n# BEGIN FIXED P1 PUSH832 SERIAL-D6: source-only generated metadata\n'
    for name,info in zip(('PUSH','READ'),metadata):
        bundle += f'_P1_D6_832_{name}_SPECS = {info["specs"]!r}\n'
        bundle += f'_P1_D6_832_{name}_INFO = {info!r}\n'
    bundle += '_P1_D6_832_BINDINGS = None\n\n'+renamed_binding+'\n\n'+renamed_product+'\n\n'
    bundle += '''def _p1_d6_832_bind(arena):
    global _P1_D6_832_BINDINGS
    assert _P1_D6_832_BINDINGS is None
    assert tuple(arena.shape) == (292225024,) and str(arena.dtype) == 'float32'
    namespace = {'fnp':fnp, '_packed_product_cached':_packed_product_cached}
    push = _cp3_bind_arena_views(arena,_P1_D6_832_PUSH_SPECS)
    read = _cp3_bind_arena_views(arena,_P1_D6_832_READ_SPECS)
    _P1_D6_832_BINDINGS = (
        _P1_D6_832_Binding(fnp,arena,(1024,1024,8192),namespace,push,_P1_D6_832_PUSH_INFO),
        _P1_D6_832_Binding(fnp,arena,(1024,8192,1024),namespace,read,_P1_D6_832_READ_INFO))
# END FIXED P1 PUSH832 SERIAL-D6
'''
    predict = ast.get_source_segment(parent,nodes(parent)['_predict_binary'])
    anchor = '        _SCRATCH = [arena, _cp3_bind_arena_views(arena, _PACKED_PUSH), _cp3_bind_arena_views(arena, _PACKED_READ), None, None, None, None]\n'
    require(predict.count(anchor) == 1, 'One original first-prediction scratch binding')
    new_predict = predict.replace(anchor,anchor+'        _p1_d6_832_bind(_SCRATCH[0])\n')
    changed,_ = replace_definition(parent,'_predict_binary',new_predict)
    dispatch = '''def _packed_product_4(a,b,out,arena_views):
    if (tuple(a.shape) == (1024,832) and tuple(b.shape) == (832,8192)
            and tuple(out.shape) == (1024,8192)
            and str(a.dtype) == str(b.dtype) == str(out.dtype) == 'float32'
            and _SCRATCH is not None and arena_views is _SCRATCH[1]):
        assert _P1_D6_832_BINDINGS is not None
        return _p1_d6_832_product(a,b,out,_P1_D6_832_BINDINGS[0])
    return _packed_product_cached(a,b,out,arena_views,4)'''
    changed,old_dispatch = replace_definition(changed,'_packed_product_4',dispatch)
    require(old_dispatch == 'def _packed_product_4(a,b,out,arena_views):\n    return _packed_product_cached(a,b,out,arena_views,4)',
            'Exact original final D4 dispatcher')
    terminal = changed+bundle
    inverse,_ = replace_definition(terminal[:-len(bundle)],'_packed_product_4',old_dispatch)
    inverse,_ = replace_definition(inverse,'_predict_binary',predict)
    require(inverse == parent and text_sha(inverse) == PARENT_SHA, 'Whole-source two-edit plus appended-bundle inverse')
    before_nodes,after_nodes = nodes(parent),nodes(terminal)
    for name in set(before_nodes)-{'_predict_binary','_packed_product_4'}:
        require(ast.get_source_segment(parent,before_nodes[name]) == ast.get_source_segment(terminal,after_nodes[name]),
                'Unrelated definition remains byte-exact: '+name)
    for name in ('_GNN_LITERAL','_BD_COEFFICIENT_LITERAL'):
        require(literal(parent,name) == literal(terminal,name), 'Unchanged full literal source: '+name)
    imports = lambda s:[ast.dump(n,include_attributes=False) for n in ast.parse(s).body
                       if isinstance(n,(ast.Import,ast.ImportFrom))]
    require(imports(parent) == imports(terminal), 'No runtime imports added')
    require(not any(x in sys.modules for x in ('numpy','torch','flopscope')), 'Source-only kernel metadata generation')
    proof = dict(parent_inverse_sha256=text_sha(inverse),whole_source_inverse_verified=True,
        unchanged_P1_GNN_literal_sha256=text_sha(literal(parent,'_GNN_LITERAL')),
        unchanged_cstar_literal_sha256=text_sha(literal(parent,'_BD_COEFFICIENT_LITERAL')),
        kernel_source_audit=audit,kernel_class_product_name_only_inverse=True,
        cached_child_helpers_byte_exact=True,child_bypasses_dispatcher=True,imports_byte_exact=True,
        runtime_layout_generation=False,second_backing_arena=False,original_arena_bytes=1168900096,
        additional_both_binding_flops=EXTRA_BINDINGS,additional_reshape_calls=68,
        additional_layouts=[dict(literal_name=f'_P1_D6_832_{name}_SPECS',capacity=info['capacity'],
            binding_flops=info['binding_flops'],reshape_calls=34,required_elements=info['required_elements'])
            for name,info in zip(('PUSH','READ'),metadata)],
        dispatch_shape=SHAPE,dispatch_depth_before=4,dispatch_depth_after=6,
        warm_flops_before=8417538304,warm_flops_after=7062260240,bundle_sha256=text_sha(bundle))
    return parent,terminal,proof


def verify_build(build_sha, candidate_sha, builder_sha, protocol_sha):
    sources,inputs = admission(builder_sha,protocol_sha)
    require(sha(OUT/'result.json') == build_sha, 'Reviewed build result')
    record,end = read(OUT/'result.json'),read(OUT/'completion.json')
    require(record['schema'] == 'current_p1_partial_d6_build_v1' and all(record.get(k) is True for k in
            ('passed','completed','hashes_unchanged','sources_unchanged')) and end['passed'] and end['completed'] and
            end['result_sha256'] == build_sha and record['source_bindings'] == sources and
            record['input_bindings'] == inputs, 'Complete immutable source-build seal')
    parent,terminal,proof = construct()
    expected = {'parent':(PARENT,parent),'null':(OUT/'null_estimator.py',parent),'terminal':(OUT/'estimator.py',terminal)}
    require(set(record['candidates']) == set(expected), 'Only exact parent/null/terminal candidates')
    artifacts = {}
    for role,(path,text) in expected.items():
        item = record['candidates'][role]
        require(item == dict(path=str(path),sha256=text_sha(text)) and path.read_bytes() == text.encode(),
                'Exact complete candidate source: '+role)
        if role != 'parent':
            artifacts[str(path)] = item['sha256']
    require(record['artifacts'] == artifacts and record['candidate_sha256'] == candidate_sha == text_sha(terminal) and
            record['proof'] == json.loads(json.dumps(proof)) and record['frozen_trajectory_estimate'] == frozen_estimate(),
            'Exact inverse proof, metadata-only estimate and sole terminal identity')
    bind(sources); bind(inputs); bind(artifacts)
    inputs.update({str(OUT/'result.json'):build_sha,str(OUT/'completion.json'):sha(OUT/'completion.json')})
    return dict(build=record,candidates=record['candidates'],source_bindings=sources,input_bindings=inputs,
                proof=record['proof'],additional_layouts=record['proof']['additional_layouts'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',action='store_true')
    parser.add_argument('--source-sha256')
    parser.add_argument('--protocol-sha256')
    args = parser.parse_args()
    if not args.build:
        parser.print_help(); return 0
    require(not OUT.exists() and not OUT.is_symlink(), 'Exclusive source build; no overwrite/resume')
    start = time.monotonic(); OUT.mkdir()
    result = dict(schema='current_p1_partial_d6_build_v1',passed=False,completed=False,
        hashes_unchanged=False,sources_unchanged=False,estimator_imported=False,numerical_execution=False,
        truth_accessed=False,refit_performed=False,candidate_score_measured=False,production_qualified=False,
        old_max_read_timing_gate_still_failed=True,automatic_next_stage=False,
        started_utc=datetime.now(timezone.utc).isoformat())
    def guard(*unused):
        require(time.monotonic()-start < 60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2_000_000_000,
                'Source-only build60s/2GB guard')
    previous = signal.signal(signal.SIGALRM,guard); signal.setitimer(signal.ITIMER_REAL,.2,.2)
    try:
        sources,inputs = admission(args.source_sha256,args.protocol_sha256)
        result.update(source_bindings=sources,input_bindings=inputs)
        parent,terminal,proof = construct()
        result.update(proof=proof,frozen_trajectory_estimate=frozen_estimate(),candidates={},artifacts={})
        for role,path,text in (('parent',PARENT,parent),('null',OUT/'null_estimator.py',parent),
                               ('terminal',OUT/'estimator.py',terminal)):
            if role != 'parent':
                with path.open('x') as stream:
                    stream.write(text)
                result['artifacts'][str(path)] = sha(path)
            result['candidates'][role] = dict(path=str(path),sha256=sha(path))
        result.update(candidate_sha256=result['candidates']['terminal']['sha256'],null_source_byte_exact=True)
        bind(sources); bind(inputs); bind(result['artifacts']); guard()
        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True)
    except BaseException:
        result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,error=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,previous)
        result.update(wall_s=time.monotonic()-start,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write(OUT/'result.json',result)
        write(OUT/'completion.json',dict(passed=result['passed'],completed=result['completed'],
                                        result_sha256=sha(OUT/'result.json')))
    print(json.dumps({k:result.get(k) for k in ('passed','completed','candidate_sha256','wall_s','error')},allow_nan=False),flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
