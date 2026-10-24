"""Inert stdlib-only three-span composition of retained partial-D6 on COMP."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from math import prod
import os
from pathlib import Path
import re
import resource
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
COMP = NEW/'current_p1_node_composition_v1'
OLD = NEW/'current_p1_partial_d6_v1'
P1 = NEW/'current_d50_native_objective_p1_v1'
KERNEL = NEW/'current_d50_serial_d6_kernel_v1'
PROBE = NEW/'current_p1_d6_push832_probe_v1'
OUT = HERE/'build_v1'
SCHEMA = 'current_comp_partial_d6_build_v1'
PARENT = COMP/'build_v1/terminal_estimator.py'
DONOR = OLD/'build_v2/estimator.py'
OLD_PARENT = P1/'build_v1/estimator.py'
PARENT_SHA = '2e13d40a21e62691155918719f8ec088f581d495ae17800c5503c3b56504918a'
DONOR_SHA = 'eaeaac6ebe75aa272b712b4dfed85f220e876a4236227e77e47933e0d363f1d0'
OLD_PARENT_SHA = '0deb237350e753220b575796c0278f86884b8461bfeec4ab53eae9c15522c435'
BUNDLE_SHA = '3a0d21619e54bc99335403aa2d3ddf3364105d1899a0232836d63a4181c39fe3'
BEGIN = '\n\n# BEGIN FIXED P1 PUSH832 SERIAL-D6: source-only generated metadata\n'
END = '# END FIXED P1 PUSH832 SERIAL-D6\n'
BINDING = '        _p1_d6_832_bind(_SCRATCH[0])\n'
ANCHOR = '        _SCRATCH = [arena, _cp3_bind_arena_views(arena, _PACKED_PUSH), _cp3_bind_arena_views(arena, _PACKED_READ), None, None, None, None]\n'
OLD_DISPATCH = 'def _packed_product_4(a,b,out,arena_views):\n    return _packed_product_cached(a,b,out,arena_views,4)'
NEW_DISPATCH = '''def _packed_product_4(a,b,out,arena_views):
    if (tuple(a.shape) == (1024,832) and tuple(b.shape) == (832,8192)
            and tuple(out.shape) == (1024,8192)
            and str(a.dtype) == str(b.dtype) == str(out.dtype) == 'float32'
            and _SCRATCH is not None and arena_views is _SCRATCH[1]):
        assert _P1_D6_832_BINDINGS is not None
        return _p1_d6_832_product(a,b,out,_P1_D6_832_BINDINGS[0])
    return _packed_product_cached(a,b,out,arena_views,4)'''
EXTRA_FLOPS = 561056000
SAVED_PER_HIT = 1355278064
SOURCE_PINS = {
    HERE/'DECISION.md':'b594093e92a190014b534ed6ebc7e0688326ba7d830dac404fefd76d0c43827a',
    PARENT:PARENT_SHA, DONOR:DONOR_SHA, OLD_PARENT:OLD_PARENT_SHA,
    COMP/'build.py':'f3a3c95b3a7e99a40d8951800fb9d6ec0c0252f0bd4193c3cc2941e6712e61bf',
    COMP/'PROTOCOL.md':'5aa1fefd2e7b60ce77814f39efc4dcea3bcd2cc7ed372d4841da4eb8f973696c',
    OLD/'build.py':'c7afaa3a4fe347b0aeac2c0181ebb7ea76512d2910b26cc813c0375744be3138',
    OLD/'build_v2.py':'850de8855cfd50a0ff98e5bce072d401aafd0e3d87282c79b17256dfa1397f08',
    OLD/'PROTOCOL.md':'b4fc1edd8cabec7b59d11ede87de42ce3f3c142aa0b142582bcccac8bdcdaff9',
    OLD/'SOURCE_BUILD_V2.md':'a8bb785fdedc3678a2ec44478be8d338cbee057795646dbd1a99ce6d70137cc8',
    KERNEL/'kernel.py':'a8d1cb1a972b37afd9eaaa67acf820cba3196dabea10ba1058bb3409c0b20ec7',
    NEW/'packed_arena_generic.py':'4e4a3dea533c222f34ba3a08538245414e149dbc0f918fd097267b0f869fb27e',
    PROBE/'PROTOCOL.md':'50dca9fe2ffab1bb51464cb98cbf1d643458abeedf750f9feb452484d1b120ca',
}
INPUT_PINS = {
    COMP/'build_v1/result.json':'78f7adbb09ce5749a442c0b4d171110627a0881a478fece4e19f4f5467ec26d6',
    COMP/'build_v1/completion.json':'d005b2245db255169bc5807349279a52271d451c57b9fa23c1c760e9b1a93306',
    COMP/'native_terminal_v2/result.json':'a9071e5afa6dbf35e6a6871d6997a7dbc8a1105d2bf3e97bd172c7a1b0e0180f',
    COMP/'native_terminal_v2/completion.json':'6c050c04fd565d840273f2a917f2a106182cfed1b69f59a09cd5392497706772',
    COMP/'score_v3/result.json':'17bd92bb65abb5d113cf3cd57df175b865f03295a7e0d96eadfeb133b6658390',
    COMP/'score_v3/completion.json':'23d4e593af714034e7208a3a5947bef47386cf87a47dd5872c568a22c65415b4',
    COMP/'panel_v2/terminal/result.json':'de766126360ce24263f548c27a2cb51f60e52107386c4aeb7cf15f9ea20db625',
    COMP/'panel_v2/terminal/completion.json':'0973d38be5b37c2ff99d5b6243119a22feb5fe591c851ed4aac0aae7346a9073',
    COMP/'panel_v2/score/result.json':'379f7fa393f81dfe9af0cd30cb9d8ad5a3a08e71f34242c58bc30f540569023c',
    COMP/'panel_v2/score/completion.json':'c539b369bd5e8ca15ee6213c6acb9e4e208e3fcaeda52fa3d146dbc1d0d3733f',
    OLD/'build_v2/result.json':'43773c4204d869732b40ca345115e272782fe1d17b64da343bcaf98137b0fab4',
    OLD/'build_v2/completion.json':'6f259844d30eeebd4114d2751d6a0adc8f0888662b86c335da47053d6c728664',
    OLD/'native_v1/result.json':'37bdc1accf4850ccae6f7302f49697763f2e0d58a34a5bf3173b598b504a0610',
    OLD/'native_v1/completion.json':'a3515049e322bcb802a418525b3112afbb5ec5d43ad9fb480a8c66e27ec717b3',
    OLD/'native_watch_v1/result.json':'6c01f70261373fe6f7e3f99377506c40d734bbff8120f2b701b3c46d6183f47b',
    OLD/'native_watch_v1/completion.json':'69c1dcde1729f7b8ddb100822306d84fc01250f9b868dc3fb007daf0fd36917c',
    OLD/'score_v1/result.json':'56dfdf2ff19765488ddfa43ea803a689178ec16b1b57db4ba838b52a397ce3cb',
    OLD/'score_v1/completion.json':'b88764257aeb3cbc81488b2b0395f14c80db154fdfd1841ef6755e8f389c41d6',
    OLD/'build_v1/result.json':'4da862eb19744eb0de1d895d492a297832171d84132a6cde6631ea5ddc4f728f',
    OLD/'build_v1/completion.json':'e67c1d1154f1c19d33ac8ba3531a86d187c5e29c36b8259a4b707dba9d099b40',
    KERNEL/'run_v1/result.json':'86ce7e8b1ea3d9dcc81c42bad6cb78828b520f38aed5f4918acf6052a6cf8660',
    KERNEL/'run_v1/completion.json':'caafbeb845cd7277e9c408e4ef824759a8bc521831dd7150c356882e03cf169e',
    PROBE/'run_v1/result.json':'bf79b160f8952db1f433093c245c5497d2d25a623db6888cafa13f0ae48aaeb2',
    PROBE/'run_v1/completion.json':'6fafd4fa5d8e5e1ce7f6f4ca8886919404ff35eb1417f5c8d7c9238cda1b6b5f',
}
NUMERICAL_MODULES = ('numpy','torch','flopscope','whestbench','threadpoolctl')
THREAD_ENV = ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def no_numerics():
    require(sys.flags.optimize == 0 and sys.dont_write_bytecode and
            not any(name.split('.')[0] in NUMERICAL_MODULES for name in sys.modules),
            'Assertions/bytecode guard; no estimator or numerical-library import')


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def allowed(path):
    path = Path(path).resolve()
    require(path.is_relative_to(NEW) and path.suffix in ('.py','.md','.json') and
            'truthpost' not in str(path).lower(), 'Only scoped source/receipt files')
    return path


def sha(path):
    digest = hashlib.sha256()
    with allowed(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1 << 20),b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    path = allowed(path)
    require(path.suffix == '.json', 'Only receipt JSON may be decoded')
    return json.loads(path.read_bytes())


def bind(mapping):
    for path, value in mapping.items():
        require(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) and
                sha(path) == value, 'Immutable reviewed source/receipt: '+str(path))


def chain(folder, schema=None, explicit_completion=True):
    record, end = read(folder/'result.json'), read(folder/'completion.json')
    require(record.get('passed') is True and end.get('passed') is True and
            end.get('result_sha256') == INPUT_PINS[folder/'result.json'], 'Completed prior seal')
    if explicit_completion:
        require(record.get('completed') is True and end.get('completed') is True,
                'Explicit prior completion')
    if schema is not None:
        require(record.get('schema') == schema, 'Exact prior schema')
    return record


def fixed_admission():
    no_numerics(); bind(SOURCE_PINS); bind(INPUT_PINS)
    parent_build = chain(COMP/'build_v1','current_p1_node_composition_build_v1')
    require(parent_build['candidates']['terminal'] == dict(path=str(PARENT),sha256=PARENT_SHA) and
            all(parent_build.get(k) is True for k in ('P1_payload_byte_exact','NODE_A_byte_exact',
                'cstar_byte_exact','whole_source_literal_inverse','hashes_unchanged')),
            'Qualified exact COMP P/A/cstar source')
    donor_build = chain(OLD/'build_v2','current_p1_partial_d6_build_v2')
    require(donor_build['candidate_sha256'] == DONOR_SHA and
            donor_build['candidates']['parent'] == dict(path=str(OLD_PARENT),sha256=OLD_PARENT_SHA) and
            donor_build['candidates']['terminal'] == dict(path=str(DONOR),sha256=DONOR_SHA) and
            all(donor_build.get(k) is True for k in ('hashes_unchanged','sources_unchanged','null_source_byte_exact')) and
            donor_build['proof']['whole_source_inverse_verified'] and
            donor_build['proof']['bundle_sha256'] == BUNDLE_SHA, 'Retained source-build lineage')
    for folder,candidate,count in ((COMP/'native_terminal_v2',PARENT_SHA,41),
            (COMP/'panel_v2/terminal',PARENT_SHA,15),(OLD/'native_v1',DONOR_SHA,41)):
        record = chain(folder)
        require(record['candidate_sha256'] == candidate and len(record['trials']) == count and
                all(record.get(k) is True for k in ('hashes_unchanged','sources_unchanged',
                    'own_complete_bills','all16_domains_qualified','all47_selectors_qualified')),
                'Qualified prior complete native acquisition')
    score = chain(COMP/'score_v3','current_p1_node_composition_score_v3')
    panel = chain(COMP/'panel_v2/score','current_p1_node_composition_independent14_score_v2')
    retained = chain(OLD/'score_v1','current_p1_partial_d6_score_v1')
    require(score['candidate_sha256'] == panel['candidate_sha256'] == PARENT_SHA and
            score['integrity_passed'] and score['advances_to_independent14'] and
            panel['integrity_passed'] and panel['qualification_passed'] and panel['retain_local_component'] and
            all(panel['group_strict_gain_gate'][g] for g in ('public8','virgin6')) and
            panel['terminal_score_sha256'] == INPUT_PINS[COMP/'score_v3/result.json'], 'Retained COMP evidence')
    require(retained['candidate_sha256'] == DONOR_SHA and retained['integrity_passed'] and
            retained['objective_improved'] and retained['advances_to_independent14'] and
            retained['mean_adjusted_ratio'] < 1 and retained['adjusted_wins'] >= 24 and
            retained['build_sha256'] == INPUT_PINS[OLD/'build_v2/result.json'] and
            retained['native_sha256'] == INPUT_PINS[OLD/'native_v1/result.json'] and
            retained['native_watch_sha256'] == INPUT_PINS[OLD/'native_watch_v1/result.json'] and
            retained['production_qualified'] is False, 'Retained partial-D6 objective, not production')
    watch = chain(OLD/'native_watch_v1')
    require(watch['whole_owned_tree_retired'] and watch['maximum_swap_growth_bytes'] == 0 and
            watch['child_result_sha256'] == INPUT_PINS[OLD/'native_v1/result.json'], 'Retained native retirement')
    probe = chain(PROBE/'run_v1')
    require(probe['shape'] == [1024,832,8192] and probe['partial_shape_test_admitted'] and
            probe['numerical_and_billing_gate_passed'] and probe['timing_gate_passed'] and
            probe['additional_serial_setup_flops'] == EXTRA_FLOPS and
            probe['whole_estimator_qualified'] is False, 'Fixed occurring push evidence only')
    old = chain(KERNEL/'run_v1',explicit_completion=False)
    require(old['completed'] and old['single_product_qualification_passed'] is False and
            old['timing_gate_passed'] is False, 'Old maximum-read timing failure stays failed')
    failed,end = read(OLD/'build_v1/result.json'),read(OLD/'build_v1/completion.json')
    require(failed['passed'] is False and failed['completed'] is False and
            failed['numerical_execution'] is False and end['passed'] is False and
            end['result_sha256'] == INPUT_PINS[OLD/'build_v1/result.json'], 'Preserved old source-only timeout')
    no_numerics()
    return donor_build


class Source:
    """One AST and one UTF-8/Python-newline index; never evaluate the module."""
    def __init__(self, data):
        self.data = data
        self.text = data.decode('utf-8')
        self.starts = (0,)+tuple(m.end() for m in re.finditer(rb'\r\n|\r|\n',data))
        self.tree = ast.parse(self.text)

    def span(self,node):
        first,last,left,right = node.lineno,node.end_lineno,node.col_offset,node.end_col_offset
        require(all(type(x) is int for x in (first,last,left,right)) and
                1 <= first <= last <= len(self.starts) and left >= 0 and right >= 0,
                'Valid complete AST byte locations')
        start,end = self.starts[first-1]+left,self.starts[last-1]+right
        require(0 <= start <= end <= len(self.data), 'AST byte span bounds')
        return start,end

    def segment(self,node):
        start,end = self.span(node)
        return self.data[start:end]

    def final(self,name):
        matches = [n for n in self.tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name == name]
        require(matches,'Required effective definition: '+name)
        return matches[-1]

    def assignment(self,name):
        matches = [n for n in self.tree.body if isinstance(n,ast.Assign) and len(n.targets) == 1 and
                   isinstance(n.targets[0],ast.Name) and n.targets[0].id == name]
        require(len(matches) == 1, 'Unique immutable assignment: '+name)
        return matches[0]


def splice(data, edits):
    last = len(data)
    for start,end,before,after in sorted(edits,reverse=True):
        require(0 <= start <= end <= last and data[start:end] == before, 'Disjoint exact source splice')
        data = data[:start]+after+data[end:]
        last = start
    return data


def definition_edit(source,name,after):
    node = source.final(name); start,end = source.span(node)
    return start,end,source.segment(node),after


def unchanged(source, terminal):
    def definitions(s):
        counts={}; result={}
        for node in s.tree.body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
                index=counts.get(node.name,0); counts[node.name]=index+1
                result[(node.name,index)]=s.segment(node)
        return result
    before,after = definitions(source),definitions(terminal)
    changed_keys = {(name,max(index for owner,index in before if owner == name))
                    for name in ('_predict_binary','_packed_product_4')}
    for key,value in before.items():
        if key in changed_keys:
            continue
        require(after.get(key) == value, 'Unrelated definition bytes: '+str(key))
    imports = lambda s:[s.segment(n) for n in s.tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    require(imports(source) == imports(terminal), 'Every import source byte unchanged')
    literals = {}
    for name in ('_GNN_LITERAL','_BD_COEFFICIENT_LITERAL','_BDNH_A_LITERAL'):
        value = source.segment(source.assignment(name))
        require(terminal.segment(terminal.assignment(name)) == value, 'Every learned literal byte: '+name)
        literals[name] = digest_bytes(value)
    return literals,len(before)


def construct():
    prior = fixed_admission()
    p1, donor, comp = (Source(path.read_bytes()) for path in (OLD_PARENT,DONOR,PARENT))
    require(donor.text.count(BEGIN) == donor.text.count(END) == 1 and donor.text.endswith(END),
            'Exactly one complete appended donor block at EOF')
    bundle = donor.data[donor.data.index(BEGIN.encode()):]
    require(digest_bytes(bundle) == BUNDLE_SHA and b'_P1_D6_832_' not in comp.data and
            b'_p1_d6_832_' not in comp.data, 'Exact retained bundle and unused COMP namespace')
    donor_prefix = Source(donor.data[:-len(bundle)])
    old_predict = p1.segment(p1.final('_predict_binary'))
    donor_predict = donor_prefix.segment(donor_prefix.final('_predict_binary'))
    require(old_predict.count(ANCHOR.encode()) == 1 and BINDING.encode() not in old_predict and
            donor_predict == old_predict.replace(ANCHOR.encode(),(ANCHOR+BINDING).encode(),1),
            'Only original guarded first-prediction binding statement differs')
    binding_nodes = [n for n in ast.walk(donor_prefix.final('_predict_binary')) if isinstance(n,ast.Expr) and
        isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id == '_p1_d6_832_bind']
    require(len(binding_nodes) == 1 and donor_prefix.segment(binding_nodes[0]) == BINDING.strip().encode(),
            'One AST-owned binding call')
    dispatch = donor_prefix.segment(donor_prefix.final('_packed_product_4'))
    require(dispatch == NEW_DISPATCH.encode() and p1.segment(p1.final('_packed_product_4')) == OLD_DISPATCH.encode(),
            'Exact retained dispatcher and original D4 fallback')
    old_inverse = splice(donor_prefix.data,[definition_edit(donor_prefix,'_predict_binary',old_predict),
        definition_edit(donor_prefix,'_packed_product_4',OLD_DISPATCH.encode())])
    require(old_inverse == p1.data and digest_bytes(old_inverse) == OLD_PARENT_SHA,
            'Every donor byte inverts to the original P1 source')
    comp_predict = comp.segment(comp.final('_predict_binary'))
    require(comp_predict.count(ANCHOR.encode()) == 1 and BINDING.encode() not in comp_predict and
            comp.segment(comp.final('_packed_product_4')) == OLD_DISPATCH.encode(), 'Same exact COMP insertion owners')
    changed_predict = comp_predict.replace(ANCHOR.encode(),(ANCHOR+BINDING).encode(),1)
    candidate_prefix = splice(comp.data,[definition_edit(comp,'_predict_binary',changed_predict),
        definition_edit(comp,'_packed_product_4',dispatch)])
    terminal = Source(candidate_prefix+bundle)
    inverse = splice(candidate_prefix,[definition_edit(terminal,'_predict_binary',comp_predict),
        definition_edit(terminal,'_packed_product_4',OLD_DISPATCH.encode())])
    require(inverse == comp.data and digest_bytes(inverse) == PARENT_SHA,
            'Every new candidate byte inverts to exact COMP')
    literals,definition_count = unchanged(comp,terminal)
    for name in ('_cp3_bind_arena_views','_packed_frame','_packed_product_cached'):
        require(p1.segment(p1.final(name)) == comp.segment(comp.final(name)) == terminal.segment(terminal.final(name)),
                'Inherited cached child helper byte equality: '+name)
    require(terminal.segment(terminal.final('_P1_D6_832_Binding')) == donor.segment(donor.final('_P1_D6_832_Binding')) and
            terminal.segment(terminal.final('_p1_d6_832_product')) == donor.segment(donor.final('_p1_d6_832_product')) and
            b'child(aa[j],bb[j],products[j],binding.child_views,5)' in bundle and
            b'for j in range(7):' in bundle, 'Exact original seven D5 children and no wrapper recursion')
    layouts=[]
    for name in ('PUSH','READ'):
        literal_name=f'_P1_D6_832_{name}_SPECS'
        specs=ast.literal_eval(terminal.assignment(literal_name).value)
        info=ast.literal_eval(terminal.assignment(f'_P1_D6_832_{name}_INFO').value)
        require(len(specs) == 34 and all(type(start) is int and type(stop) is int and
                0 <= start < stop <= 292225024 and stop-start == prod(shape) for start,stop,shape in specs),
                'Retained static view extents; no second backing owner')
        item=dict(literal_name=literal_name,capacity=list(info['capacity']),binding_flops=sum(b-a for a,b,s in specs),
            reshape_calls=34,required_elements=info['required_elements'])
        require(item == prior['proof']['additional_layouts'][len(layouts)] and
                item['binding_flops'] == info['binding_flops'] and
                max(stop for start,stop,shape in specs) == item['required_elements'], 'Exact source/receipt layout tariff')
        layouts.append(item)
    require(sum(x['binding_flops'] for x in layouts) == EXTRA_FLOPS, 'Both cold bindings fully charged')
    compile(terminal.tree,str(OUT/'estimator.py'),'exec')  # syntax only, never execute/import
    proof=dict(whole_source_inverse_verified=True,old_partial_inverse_to_P1_verified=True,
        new_partial_inverse_to_COMP_verified=True,parent_inverse_sha256=PARENT_SHA,
        donor_inverse_sha256=OLD_PARENT_SHA,donor_sha256=DONOR_SHA,bundle_sha256=BUNDLE_SHA,
        exact_three_span_composition=True,imports_byte_exact=True,all_unrelated_source_bytes_unchanged=True,
        unrelated_definition_count=definition_count-2,learned_literal_sha256=literals,
        P_A_cstar_literal_bytes_unchanged=True,cached_child_helpers_byte_exact=True,
        child_bypasses_dispatcher=True,kernel_class_product_byte_exact=True,
        runtime_layout_generation=False,second_backing_arena=False,original_arena_bytes=1168900096,
        additional_both_binding_flops=EXTRA_FLOPS,additional_reshape_calls=68,additional_layouts=layouts,
        dispatch_shape=[1024,832,8192],dispatch_depth_before=4,dispatch_depth_after=6,
        warm_flops_before=8417538304,warm_flops_after=7062260240,saved_flops_per_hit=SAVED_PER_HIT,
        syntax_compiled_not_executed=True,new_candidate_score_measured=False,new_geometry_measured=False)
    no_numerics(); bind(SOURCE_PINS); bind(INPUT_PINS)
    return comp.text,terminal.text,proof


def admission(builder_sha,protocol_sha):
    sources={str(p):v for p,v in SOURCE_PINS.items()}
    sources.update({str(Path(__file__).resolve()):builder_sha,str(HERE/'BUILD_PROTOCOL.md'):protocol_sha})
    inputs={str(p):v for p,v in INPUT_PINS.items()}
    no_numerics(); bind(sources); bind(inputs)
    return sources,inputs


def peak_rss():
    require(sys.platform in ('darwin','linux'),'Known process-RSS unit convention')
    value=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == 'darwin' else value*1024


def limit(start):
    require(time.monotonic()-start < 60 and peak_rss() < 2_000_000_000,'Source-only build <60s/<2GB')
    no_numerics()


def publish(path,value):
    require(path.parent == OUT and OUT.is_dir() and not OUT.is_symlink(), 'Exclusive build output only')
    data=value if isinstance(value,bytes) else (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    temporary=path.with_name(path.name+'.pending')
    with temporary.open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary,path)  # atomic, fails rather than overwrite any existing target
    temporary.unlink()      # only this function's newly created hard-link name


def verify_build(build_sha,candidate_sha,builder_sha,protocol_sha):
    start=time.monotonic()
    sources,inputs=admission(builder_sha,protocol_sha)
    require(sha(OUT/'result.json') == build_sha,'Reviewed source-build result')
    record,end,final=(read(OUT/name) for name in ('result.json','completion.json','finalization.json'))
    require(record.get('schema') == SCHEMA and record.get('role') == 'terminal' and
            all(record.get(k) is True for k in ('passed','completed','hashes_unchanged','sources_unchanged')) and
            all(end.get(k) is True and final.get(k) is True for k in ('passed','completed')) and
            end['result_sha256'] == final['result_sha256'] == build_sha and
            final['completion_sha256'] == sha(OUT/'completion.json') and
            0 <= record['elapsed_s'] <= final['elapsed_s'] == final['elapsed_through_completion_io_s'] < 60 and
            0 <= record['peak_rss_bytes'] <= final['peak_rss_bytes'] < 2_000_000_000 and
            record['source_bindings'] == sources and record['input_bindings'] == inputs,
            'Exact three seals and source authority')
    require(record['parameter_updates'] == 0 and all(record.get(k) is False for k in
            ('estimator_executed','estimator_imported','numerical_execution','truth_arrays_accessed',
             'refit_performed','candidate_score_measured','production_qualified','automatic_next_stage')),
            'Source-only build; no inference, fit or score')
    parent,terminal,proof=construct()
    expected={'parent':(PARENT,parent),'null':(OUT/'null_estimator.py',parent),'terminal':(OUT/'estimator.py',terminal)}
    require(set(record['candidates']) == set(expected),'Only parent/null/sole-terminal sources')
    artifacts={}
    for role,(path,value) in expected.items():
        item=dict(path=str(path),sha256=digest_bytes(value.encode()))
        require(record['candidates'][role] == item and path.read_bytes() == value.encode(), 'Exact emitted source: '+role)
        if role != 'parent': artifacts[str(path)]=item['sha256']
    require(record['artifacts'] == artifacts and record['proof'] == proof and
            record['extra_layouts'] == proof['additional_layouts'] and
            record['candidate_sha256'] == candidate_sha == digest_bytes(terminal.encode()),'Sole candidate and reproducible proof')
    bind(sources); bind(inputs); bind(artifacts); limit(start)
    inputs.update({str(OUT/name):sha(OUT/name) for name in ('result.json','completion.json','finalization.json')})
    return dict(build=record,candidates=record['candidates'],proof=proof,source_bindings=sources,input_bindings=inputs,
                extra_layouts=proof['additional_layouts'],additional_layouts=proof['additional_layouts'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',action='store_true')
    parser.add_argument('--source-sha256'); parser.add_argument('--protocol-sha256')
    args=parser.parse_args()
    if not args.build:
        parser.print_help(); return 0
    start=time.monotonic()
    no_numerics()
    require(os.getpriority(os.PRIO_PROCESS,0) >= 10 and all(os.environ.get(k) == '1' for k in THREAD_ENV),
            'One nice>=10 source worker; all five BLAS environment guards')
    require(not OUT.exists() and not OUT.is_symlink(),'Exclusive source build; no overwrite/resume')
    OUT.mkdir()
    result=dict(schema=SCHEMA,role='terminal',passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,
        parameter_updates=0,estimator_executed=False,estimator_imported=False,numerical_execution=False,
        truth_arrays_accessed=False,truth_accessed=False,refit_performed=False,candidate_score_measured=False,
        production_qualified=False,automatic_next_stage=False,old_max_read_timing_gate_still_failed=True,
        started_utc=datetime.now(timezone.utc).isoformat())
    previous=signal.signal(signal.SIGALRM,lambda *unused:limit(start))
    signal.setitimer(signal.ITIMER_REAL,.2,.2)
    try:
        sources,inputs=admission(args.source_sha256,args.protocol_sha256)
        parent,terminal,proof=construct()
        result.update(source_bindings=sources,input_bindings=inputs,proof=proof,
                      extra_layouts=proof['additional_layouts'],candidates={},artifacts={})
        for role,path,value in (('parent',PARENT,parent),('null',OUT/'null_estimator.py',parent),
                                ('terminal',OUT/'estimator.py',terminal)):
            if role != 'parent': publish(path,value.encode())
            item=dict(path=str(path),sha256=sha(path)); result['candidates'][role]=item
            require(item['sha256'] == digest_bytes(value.encode()),'Exact source publication')
            if role != 'parent': result['artifacts'][str(path)]=item['sha256']
        bind(sources); bind(inputs); bind(result['artifacts']); limit(start)
        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,
            candidate_sha256=result['candidates']['terminal']['sha256'],null_source_byte_exact=True)
    except BaseException:
        result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,error=traceback.format_exc())
    try:
        result.update(elapsed_s=time.monotonic()-start,peak_rss_bytes=peak_rss(),finished_utc=datetime.now(timezone.utc).isoformat())
        if result['passed']: limit(start)
        publish(OUT/'result.json',result)
        result_sha=sha(OUT/'result.json')
        publish(OUT/'completion.json',dict(passed=result['passed'],completed=result['completed'],result_sha256=result_sha))
        completion_sha=sha(OUT/'completion.json')
        if result['passed']: limit(start)
        final_elapsed=time.monotonic()-start
        final=dict(passed=result['passed'],completed=result['completed'],result_sha256=result_sha,
            completion_sha256=completion_sha,elapsed_s=final_elapsed,
            elapsed_through_completion_io_s=final_elapsed,peak_rss_bytes=peak_rss())
        publish(OUT/'finalization.json',final)
        if result['passed']: limit(start)
        print(json.dumps({k:result.get(k) for k in ('passed','completed','candidate_sha256','elapsed_s','error')},allow_nan=False),flush=True)
        if result['passed']: limit(start)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,previous)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
