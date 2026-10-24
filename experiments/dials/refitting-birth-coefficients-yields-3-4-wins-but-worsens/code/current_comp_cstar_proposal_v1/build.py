"""Sole sealed cstar literal build; inert and target-free, never a fit."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import struct
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
COMP = NEW/'current_p1_node_composition_v1'
JP = NEW/'current_comp_joint_p_node_v1'
RD = NEW/'current_comp_receiver_directions_v1'
PARENT = COMP/'build_v1/terminal_estimator.py'
PARENT_SHA = '2e13d40a21e62691155918719f8ec088f581d495ae17800c5503c3b56504918a'
OUT = HERE/'build_v1'
SCHEMA = 'current_comp_cstar_proposal_build_v1'
PROPOSAL_SCHEMA = 'current_comp_cstar_proposal_v1'
OFFPY = P2/'data_gate/.venv'
VALID = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
    139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,204,219,224,227,233,234)
TRAIN = tuple(r for r in range(35,235) if r not in VALID)
PROPOSAL_PINS = {
    HERE/'model.py': 'f5c26f04750fa147ef897484ddc3a152ad744e6170b7542e8046af3193a79656',
    HERE/'PROTOCOL.md': '0a1db21c3a9f6f0a268ae8c47c1f4ad1c6b16ff6a0438aaeb2278dfaa6e97359',
    HERE/'common.py': 'ba8671ce39908efcbb4fe8ac5349149bfaa98a8b1eaebe4ad30ca1d88967d490',
    HERE/'worker.py': '6cc6fb3f01fdef72677de0a06ab50133f89deb062da1938f917df4477ae4002f',
    HERE/'watch.py': 'd3258f391ca501698e6c5f14a9dce21f28ae4e63d05731800ce6b2da8c94f23e',
    HERE/'WATCH_SPEC.md': '2dd7bf7bb03ed0ea38fdb1a96c5c4be528d9269197783043b48e56157c1fd265',
}
HELPER_PINS = {
    JP/'build.py': '11e7f7588e692587c11480f6a09bd7cfb9492d02851381755821b7dbf01a9ed4',
    RD/'build.py': 'c844c0526e1d98d7138a8958c9a9501e9a794274222294cbc6e30dbf4073aa39',
    COMP/'native.py': 'fafc28732701aacfa70ada15d30ed083af76e91747bc01dc840ec8ba14d23d01',
}
SHA_KEYS = ('source','protocol','watcher','watch-spec','qualification-result',
            'qualification-watch','training-result','training-watch')
BLAS = ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
_HELPERS = None


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path, tick=None):
    path = Path(path).resolve()
    require(path.is_relative_to(P2) and 'truthpost' not in path.parts
        and not (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()), 'No truth-array access through builder bindings')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
            if tick is not None:
                tick()
    return h.hexdigest()


def read(path):
    path = Path(path)
    require(not path.is_symlink() and path.stat().st_size < 16*1024**2, 'Bounded regular receipt metadata')
    return json.loads(path.read_text())


def publish(path, data):
    path = Path(path)
    require(path.parent == OUT and not OUT.is_symlink(), 'Only exclusive build-owned output')
    pending = path.with_name(path.name+'.publishing')
    with pending.open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.link(pending, path); pending.unlink()


def write(path, value):
    publish(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode())


def helpers():
    """Exact small source slices, not imports of historical experiment modules."""
    global _HELPERS
    if _HELPERS is not None:
        return _HELPERS
    sources = {**HELPER_PINS, HERE/'watch.py': PROPOSAL_PINS[HERE/'watch.py']}
    require(all(sha(p) == h for p,h in sources.items()), 'Exact helper source owners')
    def extract(path, name):
        text = path.read_text(); lines = text.splitlines(keepends=True)
        nodes = [x for x in ast.parse(text).body if isinstance(x, ast.FunctionDef) and x.name == name]
        require(len(nodes) == 1, 'Unique helper body: '+name)
        n = nodes[0]
        return ''.join(lines[n.lineno-1:n.end_lineno])
    originals = [extract(JP/'build.py', k) for k in ('value_span','replace_values','array_meta')]
    originals += [extract(RD/'build.py','literal_metadata'), extract(COMP/'native.py','npy_bytes'),
                  extract(HERE/'watch.py','check_roster')]
    changes = ((3, 'isinstance(item, list)', 'isinstance(item, (list, tuple))'),
        (4, 'def npy_bytes(path,expected_raw):', 'def npy_payload(path,expected_raw,shape):'),
        (4, 'shape=(16,1024)', 'shape=tuple(shape)'), (4, '16*1024*4', 'math.prod(shape)*4'))
    derived = list(originals)
    for i, old, new in changes:
        require(derived[i].count(old) == 1, 'Unique metadata helper adaptation')
        derived[i] = derived[i].replace(old,new,1)
    inverse = list(derived)
    for i, old, new in reversed(changes):
        require(inverse[i].count(new) == 1, 'Unique helper inverse')
        inverse[i] = inverse[i].replace(new,old,1)
    require(inverse == originals, 'Complete metadata helper source inverse')
    env = dict(globals())
    exec(compile('\n\n'.join(derived), str(__file__)+':exact-helper-subset', 'exec'), env)
    proof = dict(owners={str(p):h for p,h in sources.items()},
        source_inverse_byte_exact=True, value_span_and_replace_unchanged=True,
        FP32_metadata_tuple_support=True, NPY_header_shape_parameter_only=True,
        original_subset_sha256=hashlib.sha256('\n\n'.join(originals).encode()).hexdigest(),
        derived_subset_sha256=hashlib.sha256('\n\n'.join(derived).encode()).hexdigest())
    _HELPERS = ({k:env[k] for k in ('value_span','replace_values','array_meta','literal_metadata','npy_payload','check_roster')}, proof)
    return _HELPERS


def chain(folder, digest, pin, limit):
    names = ('result.json','completion.json','finalization.json')
    values = []
    for name in names:
        path = folder/name
        values.append(pin(path, digest if name == 'result.json' else sha(path)))
    result, end, final = (read(folder/name) for name in names)
    require(all(x.get('passed') is True and x.get('completed') is True for x in (result,end,final))
        and result['hashes_unchanged'] and result['sources_unchanged']
        and end['result_sha256'] == final['result_sha256'] == values[0]
        and final['completion_sha256'] == values[1] and math.isfinite(final['elapsed_s'])
        and 0 <= final['elapsed_s'] < limit, 'All three complete actual seals: '+str(folder))
    return result, values


def array_fields(item):
    return {k:item[k] for k in ('dtype','shape','raw_sha256')}


def admission(args, record, tick=lambda: None):
    h, helper_proof = helpers(); verified = {}
    def pin(path, digest):
        path = Path(path).resolve(); key = str(path)
        require(re.fullmatch('[0-9a-f]{64}', digest or ''), 'Explicit complete digest')
        if key not in verified:
            verified[key] = sha(path,tick)
        require(verified[key] == digest, 'Unchanged admitted source/non-target input: '+key)
        group = 'source_bindings' if path.suffix in ('.py','.md') else 'input_bindings'
        require(key not in record[group] or record[group][key] == digest, 'No conflicting owner binding')
        record[group][key] = digest; tick(); return digest
    def inventory(value):
        for group in ('source_bindings','input_bindings','weight_bindings','artifacts'):
            for path,digest in value.get(group,{}).items():
                pin(path,digest)
        # target_bindings remain sealed metadata only; never hash those arrays.
    own = {HERE/'build.py':args.source_sha256, HERE/'BUILD_PROTOCOL.md':args.protocol_sha256,
           HERE/'watch_build.py':args.watcher_sha256, HERE/'BUILD_WATCH_SPEC.md':args.watch_spec_sha256}
    for path,digest in {**PROPOSAL_PINS,**HELPER_PINS,PARENT:PARENT_SHA,**own}.items():
        pin(path,digest)
    def pair(stage, worker_digest, watch_digest):
        folder = HERE/('qualification_v1' if stage == 'qualify' else 'training_v1')
        wfolder = HERE/('qualification_watch_v1' if stage == 'qualify' else 'training_watch_v1')
        count, limit = (5,300) if stage == 'qualify' else (200,1800)
        value, vp = chain(folder,worker_digest,pin,limit)
        watch, wp = chain(wfolder,watch_digest,pin,limit)
        require(value['schema'] == PROPOSAL_SCHEMA and value['stage'] == stage
            and value['full_model_calls'] == value['retired_call_count'] == value['graphs_completed'] == count
            and len(value['graph_records']) == count and value['tiny_model_calls'] == 0
            and all(value.get(k) is True for k in ('all_graphs_retired','all_persistent_owners_retired',
                'all16_domains_qualified','all47_selectors_qualified','frozen_P_A_c0','CS_failure_preserved','no_native_gradient_claim'))
            and value['native_gradient_qualified'] is False and value['validation_targets_accessed'] is False
            and value['native_score_measured'] is False and value['checkpoint_selection'] is False
            and value['frozen_parent_parameter_updates'] == 0
            and value['parent_candidate_sha256'] == PARENT_SHA
            and value['allocated_parameters'] == value['active_parameters'] == 56
            and value['trainable_keys'] == ['delta'], 'Complete exact current-COMP delta-only scope')
        require(value['environment']['versions'] == dict(numpy='1.26.4',torch='2.7.0')
            and value['mps_allocator']['configured'] and value['maximum_sampled_mps_driver_bytes'] < 16_000_000_000,
            'Actual qualified proposal runtime and driver guard')
        require(watch['schema'] == 'current_comp_cstar_proposal_watch_v1' and watch['stage'] == stage
            and watch['retired_calls'] == watch['completed_graph_count'] == count
            and [watch['child_result_sha256'],watch['child_completion_sha256'],watch['child_finalization_sha256']] == vp
            and all(watch.get(k) is True for k in ('child_execution_completed','whole_owned_tree_retired',
                'cleanup_passed','observational_groups_retired','actual_completed_samples','foreign_numerical_workers_checked'))
            and watch['whole_cap_s'] == limit and watch['elapsed_s'] < limit
            and 0 < watch['maximum_combined_rss_bytes'] < 12_000_000_000
            and watch['maximum_swap_growth_bytes'] == 0 and 0 < watch['maximum_interval_s'] <= 1
            and watch['completed_sample_intervals_s']
            and max(watch['completed_sample_intervals_s']) == watch['maximum_interval_s']
            and all(math.isfinite(x) and 0 <= x <= 1 for x in watch['completed_sample_intervals_s']),
            'Actual independent proposal-resource, interval and whole-group retirement seals')
        for item in (value,watch):
            require(all(item['source_bindings'].get(str(p)) == d for p,d in PROPOSAL_PINS.items()), 'Same exact six frozen proposal owners')
            inventory(item)
        return value,watch,vp,wp
    q,qw,qp,qwp = pair('qualify',args.qualification_result_sha256,args.qualification_watch_sha256)
    require(q['execution_integrity_passed'] and q['gradient_calls'] == q['zero_lr_Adam_calls'] == 3
        and q['reference_forwards'] == 2 and q['parameter_updates'] == q['optimizer_updates'] == 0
        and not q['training_targets_accessed'] and all(q.get(k) is True for k in (
            'same_backend_null_byte_exact','first_repeat35_prediction_byte_exact',
            'first_repeat35_gradient_byte_exact','zero_lr_delta_byte_exact')), 'Fresh five-call execution gate, never CS derivative eligibility')
    train,watch,tp,twp = pair('train',args.training_result_sha256,args.training_watch_sha256)
    prerequisites = dict(qualification_result_sha256=qp[0],qualification_completion_sha256=qp[1],qualification_finalization_sha256=qp[2],
        qualification_watch_result_sha256=qwp[0],qualification_watch_completion_sha256=qwp[1],qualification_watch_finalization_sha256=qwp[2])
    require(train['prerequisites'] == prerequisites and train['proposal_complete']
        and train['epochs'] == 1 and train['training_graphs'] == train['gradient_calls'] == 160
        and train['validation_graphs'] == 40 and train['updates'] == train['optimizer_updates'] == train['parameter_updates'] == 40
        and train['training_targets_accessed'] and train['terminal_export_reloaded_without_inverse']
        and train['training_optimizer_retired_before_validation'] and train['seed'] == 9309099
        and train['old_d50_weights_are_proposal_constants'] and not train['native_cost_derivative_claimed'], 'Sole fixed completed160/40 terminal proposal')
    require(train['optimizer'] == dict(name='Adam',coordinates='relative_headroom_delta',lr_start=.025,lr_terminal=.0025,
        betas=[.9,.999],eps=1e-8,weight_decay=0,foreach=False,gradient_norm_cap=5.)
        and train['train_ranks'] == list(TRAIN) and train['validation_ranks'] == list(VALID), 'Exact fixed optimizer and data frame')
    def artifact(value, item, name):
        folder = HERE/('qualification_v1' if value['stage'] == 'qualify' else 'training_v1')
        path = folder/name
        require(item['path'] == str(path) and value['artifacts'].get(str(path)) == item['sha256'], 'Actual owned sealed artifact: '+name)
        pin(path,item['sha256']); return path
    for value,watch_value in ((q,qw),(train,watch)):
        roster_path = artifact(value,value['model_call_roster'],'model_call_roster.json')
        expected = h['check_roster'](roster_path,value['stage'])
        require(watch_value['model_call_roster_sha256'] == sha(roster_path)
            and len(watch_value['graph_markers']) == len(expected), 'Independent complete ordered roster')
        for index,(item,spec,observed) in enumerate(zip(value['graph_records'],expected,watch_value['graph_markers'])):
            path = artifact(value,item,f'graph_{index:04d}_complete.json'); graph = read(path)
            require(all(graph[k] == v for k,v in spec.items())
                and all(graph.get(k) is True for k in ('passed','completed','graph_retired','all16_domains_qualified',
                    'all47_selectors_qualified','observed_checkpoint_replay_byte_exact','frozen_P_A_c0'))
                and observed['call'] == index and observed['path'] == str(path) and observed['sha256'] == item['sha256']
                and observed['rank'] == spec['rank'] and observed['kind'] == spec['kind'], 'Every actual released graph and externally observed marker')
            pp = artifact(value,graph['prediction'],f'graph_{index:04d}_served.npy')
            require(array_fields(graph['prediction'])['shape'] == [16,1024] and graph['prediction']['dtype'] == 'float32', 'Full original served frame')
            h['npy_payload'](pp,graph['prediction']['raw_sha256'],(16,1024))
            if value is train and index >= 160:
                require(train['validation_outputs'][str(spec['rank'])] == graph['prediction'], 'All40 terminal output ownership')
            tick()
    parent = PARENT.read_bytes()
    p0 = h['value_span'](parent,'_GNN_LITERAL')[3]
    a0 = h['literal_metadata'](h['value_span'](parent,'_BDNH_A_LITERAL')[3])[0]
    cm,craw = h['literal_metadata'](h['value_span'](parent,'_BD_COEFFICIENT_LITERAL')[3])
    pmeta = {k:h['literal_metadata'](v)[0] for k,v in p0.items()}
    for value in (q,train):
        require(value['initial_P']['arrays'] == pmeta and array_fields(value['initial_A']) == a0
            and array_fields(value['initial_cstar']) == cm and cm['shape'] == [15,4]
            and array_fields(value['initial_delta']) == dict(dtype='float32',shape=[14,4],raw_sha256=hashlib.sha256(bytes(224)).hexdigest()), 'Exact original P/A/c0 and positive-zero56 initialization')
    seal_path = HERE/'training_v1/terminal_seal.json'
    pin(seal_path,train['terminal_seal_sha256']); require(train['artifacts'][str(seal_path)] == train['terminal_seal_sha256'],'Terminal seal is an actual training artifact')
    terminal = read(seal_path)
    require(terminal['terminal_epoch'] == 1 and terminal['updates'] == 40 and terminal['training_graphs'] == 160
        and terminal['terminal_delta'] == train['terminal_delta'] and terminal['terminal_coefficients'] == train['terminal_coefficients']
        and terminal['initial_P'] == train['initial_P'] and terminal['initial_A'] == train['initial_A']
        and terminal['initial_cstar'] == train['initial_cstar'] and terminal['source_bindings'] == train['source_bindings']
        and terminal['qualification'] == prerequisites and terminal['parent_candidate_sha256'] == PARENT_SHA
        and not terminal['validation_predictions_started'] and not terminal['validation_targets_accessed']
        and terminal['frozen_parent_parameter_updates'] == 0 and not terminal['checkpoint_selection'], 'Only the predetermined exported terminal, sealed before validation')
    validation_path = HERE/'training_v1/validation_predictions_sealed.json'
    pin(validation_path,train['validation_predictions_sealed_sha256'])
    validation = read(validation_path)
    require(train['artifacts'][str(validation_path)] == train['validation_predictions_sealed_sha256']
        and validation['terminal_coefficients'] == train['terminal_coefficients']
        and validation['terminal_seal_sha256'] == train['terminal_seal_sha256']
        and validation['validation_outputs'] == train['validation_outputs']
        and set(validation['validation_outputs']) == {str(r) for r in VALID} and validation['all_40_complete']
        and not validation['validation_targets_accessed'] and not validation['native_score_measured'], 'All40 target-blind actual-terminal prediction seal')
    require(len(train['update_records']) == 40, 'All40 finite fixed updates retained')
    for i,item in enumerate(train['update_records']):
        update = read(artifact(train,item,f'update_{i+1:02d}.json'))
        require(update['update'] == i+1 and update['learning_rate'] == .0025+.5*(.025-.0025)*(1+math.cos(math.pi*i/39))
            and update['ranks'] == train['training_order'][4*i:4*i+4] and math.isfinite(update['pre_clip_gradient_norm'])
            and update['coefficient_abs_max'] < 1, 'Prescribed update/order/domain, not an adaptive schedule')
    for name,shape,key in (('terminal_delta.npy',(14,4),'terminal_delta'),('terminal_coefficients.npy',(15,4),'terminal_coefficients')):
        item = train[key]; path = artifact(train,item,name)
        require(item['dtype'] == 'float32' and item['shape'] == list(shape), 'Exact exported terminal shape/precision')
        h['npy_payload'](path,item['raw_sha256'],shape)
    require(terminal['model_source_proof'] == train['model_source_proof'] == q['model_source_proof']
        and all(train['model_source_proof'].get(k) is True for k in ('original_comp_step_unchanged',
            'inherited_step_inverse_byte_exact','original_node_function_unchanged','original_birth_action_and_stored_plane_unchanged',
            'actual_delta_and_nonleaf_cstar_identities','frozen_P_A_c0','terminal_uses_sealed_coefficients_without_inverse','CS_failure_preserved')),
        'Identical qualified proposal source/operand and frozen-parent proof')
    record.update(helper_source_proof=helper_proof, prerequisites=dict(qualification=qp,qualification_watch=qwp,training=tp,training_watch=twp),
        terminal_coefficients=train['terminal_coefficients'],terminal_delta=train['terminal_delta'],
        terminal_seal=dict(path=str(seal_path),sha256=train['terminal_seal_sha256']),
        validation_seal=dict(path=str(validation_path),sha256=train['validation_predictions_sealed_sha256']),
        initial_P=pmeta,initial_A=a0,initial_cstar=cm,training_result_sha256=tp[0],training_watch_sha256=twp[0])
    return train,pin


def source_candidate(literal):
    h,_ = helpers(); parent = PARENT.read_bytes()
    require(sha(PARENT) == PARENT_SHA, 'Exact retained COMP source')
    meta,raw = h['literal_metadata'](literal)
    require(meta['shape'] == [15,4] and all(abs(v) < 1 for row in literal for v in row), 'Finite60 cstar with original strict coefficient domain')
    original = h['value_span'](parent,'_BD_COEFFICIENT_LITERAL')
    _,oldraw = h['literal_metadata'](original[3])
    require(raw[:16] == oldraw[:16], 'Original Gaussian row0 bytes unchanged')
    candidate = h['replace_values'](parent,{'_BD_COEFFICIENT_LITERAL':repr(literal).encode()})
    require(h['replace_values'](candidate,{'_BD_COEFFICIENT_LITERAL':original[2]}) == parent, 'Only one AST-owned VALUE span changed; whole-source byte inverse')
    restored,restored_raw = h['literal_metadata'](h['value_span'](candidate,'_BD_COEFFICIENT_LITERAL')[3])
    require(restored == meta and restored_raw == raw and all(h['value_span'](candidate,k)[2] == h['value_span'](parent,k)[2]
        for k in ('_GNN_LITERAL','_BDNH_A_LITERAL')), 'Actual FP32 literal roundtrip and untouched full P/A source')
    proof = dict(literal_inverse_byte_exact=True,literal_roundtrip_byte_exact=True,full_shape_FP32_finite=True,
        all_non_cstar_source_byte_exact=True,P_A_source_byte_exact=True,frozen_nonP_and_inactive66_byte_exact=True,
        gaussian_row0_byte_exact=True,original_runtime_getter_and_physics_unchanged=True,
        changed_value_names=['_BD_COEFFICIENT_LITERAL'],new_inference_operations_at_fixed_geometry=0,
        actual_candidate_masks_and_complete_cost_still_required=True)
    return candidate,meta,proof


def verify_build(build_sha,candidate_sha,builder_sha,protocol_sha,tick=None):
    """Stdlib full revalidation; no estimator, Torch, NumPy or target access."""
    tick = tick or (lambda: None)
    require(sha(__file__,tick) == builder_sha and sha(HERE/'BUILD_PROTOCOL.md',tick) == protocol_sha,'Reviewed builder/protocol owners')
    pins = {}
    def pin(path,digest):
        require(sha(path,tick) == digest,'Sealed build binding'); pins[str(Path(path).resolve())] = digest; return digest
    result,chain_pins = chain(OUT,build_sha,pin,120)
    require(result['schema'] == SCHEMA and result['stage'] == 'build' and result['role'] == 'terminal'
        and not result['estimator_executed'] and not result['truth_arrays_accessed'] and not result['refit_performed']
        and not result['native_gradient_qualified'] and result['CS_failure_preserved']
        and result['source_bindings'][str(HERE/'build.py')] == builder_sha
        and result['source_bindings'][str(HERE/'BUILD_PROTOCOL.md')] == protocol_sha,'Completed exact sole terminal build')
    args = argparse.Namespace(**result['launch_pins'])
    require(args.source_sha256 == builder_sha and args.protocol_sha256 == protocol_sha,'Same reviewed build launch authority')
    context = dict(source_bindings={},input_bindings={})
    train,_ = admission(args,context,tick)
    require(context == {k:result[k] for k in context},'Revalidated complete prerequisite/source/export metadata')
    h,_ = helpers(); literal = h['value_span']((OUT/'estimator.py').read_bytes(),'_BD_COEFFICIENT_LITERAL')[3]
    candidate,meta,proof = source_candidate(literal)
    expected = dict(path=str(OUT/'estimator.py'),sha256=candidate_sha)
    require(result['candidate'] == expected and result['parent'] == dict(path=str(PARENT),sha256=PARENT_SHA)
        and result['proof'] == proof and result['coefficient_literal'] == meta == array_fields(train['terminal_coefficients'])
        and hashlib.sha256(candidate).hexdigest() == candidate_sha and (OUT/'estimator.py').read_bytes() == candidate
        and result['artifacts'] == {str(OUT/'estimator.py'):candidate_sha}, 'Exact source bytes and actual sealed coefficient binding')
    pin(OUT/'estimator.py',candidate_sha)
    return dict(build=result,candidate=expected,parent=result['parent'],candidates=dict(parent=result['parent'],terminal=expected),
        proof=proof,source_bindings=dict(result['source_bindings']),input_bindings={**result['input_bindings'],**pins})


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--build',action='store_true')
    for key in SHA_KEYS:
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.build:
        parser.print_help(); return 0
    began = time.monotonic()
    require(all(re.fullmatch('[0-9a-f]{64}',getattr(args,k.replace('-','_')+'_sha256') or '') for k in SHA_KEYS),'Every future authority is an explicit reviewed SHA')
    OUT.mkdir(exist_ok=False)
    record = dict(schema=SCHEMA,stage='build',role='terminal',passed=False,completed=False,source_bindings={},input_bindings={},artifacts={},
        estimator_executed=False,truth_arrays_accessed=False,refit_performed=False,native_gradient_qualified=False,native_score_measured=False,
        CS_failure_preserved=True,no_native_gradient_claim=True,parameter_updates=0,hashes_unchanged=False,sources_unchanged=False,
        build_watch_required_before_native=True,launch_pins={k.replace('-','_')+'_sha256':getattr(args,k.replace('-','_')+'_sha256') for k in SHA_KEYS})
    def guard(*_):
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        require(time.monotonic()-began < 120 and rss < 2_000_000_000,'Whole120s/2GB source build including final publication')
        return rss
    old = signal.signal(signal.SIGALRM,guard); signal.setitimer(signal.ITIMER_REAL,.25,.25)
    ok = False
    try:
        require(Path(sys.prefix).resolve() == OFFPY.resolve() and 'anaconda3' not in sys.executable
            and sys.dont_write_bytecode and sys.flags.optimize == 0 and os.getpriority(os.PRIO_PROCESS,0) >= 10
            and all(os.environ.get(k) == '1' for k in BLAS) and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'OFFPY/nice10/BLAS1/bytecodeoff source build')
        require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules),'Complete metadata admission before numerical import')
        train,_ = admission(args,record,guard)
        require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules),'Historical helpers were source slices only')
        import numpy as np
        require(np.__version__ == '2.4.6' and not {'torch','flopscope','threadpoolctl'} & set(sys.modules),'NumPy-only official-runtime literal conversion')
        actual = np.load(train['terminal_coefficients']['path'],allow_pickle=False)
        h,_ = helpers()
        require(actual.dtype == np.float32 and actual.shape == (15,4) and np.isfinite(actual).all()
            and (np.abs(actual) < 1).all() and h['array_meta'](actual) == array_fields(train['terminal_coefficients']), 'Sole sealed actual FP32 c, no reconstruction from delta')
        candidate,meta,proof = source_candidate(actual.tolist())
        require(np.asarray(h['value_span'](candidate,'_BD_COEFFICIENT_LITERAL')[3],dtype=np.float32).tobytes() == actual.tobytes(), 'Actual NPY-to-literal FP32 byte roundtrip')
        for group in ('source_bindings','input_bindings'):
            for path,digest in record[group].items():
                require(sha(path,guard) == digest,'All admitted bytes unchanged before candidate publication')
        guard(); publish(OUT/'estimator.py',candidate)
        digest = sha(OUT/'estimator.py',guard)
        record.update(candidate=dict(path=str(OUT/'estimator.py'),sha256=digest),parent=dict(path=str(PARENT),sha256=PARENT_SHA),
            coefficient_literal=meta,proof=proof,artifacts={str(OUT/'estimator.py'):digest},numpy_version=np.__version__,
            passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,peak_rss_bytes=guard())
    except BaseException:
        record.update(passed=False,completed=False,error=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,old)
        record['elapsed_s'] = time.monotonic()-began
        if record['elapsed_s'] >= 120:
            record.update(passed=False,completed=False)
        write(OUT/'result.json',record); digest = sha(OUT/'result.json')
        ok = bool(record['passed'] and time.monotonic()-began < 120)
        write(OUT/'completion.json',dict(passed=ok,completed=ok,result_sha256=digest))
        end = sha(OUT/'completion.json'); elapsed = time.monotonic()-began; ok = bool(ok and elapsed < 120)
        write(OUT/'finalization.json',dict(passed=ok,completed=ok,result_sha256=digest,completion_sha256=end,elapsed_s=elapsed))
        sha(OUT/'finalization.json'); ok = bool(ok and time.monotonic()-began < 120)
    print(json.dumps(dict(passed=ok,elapsed_s=time.monotonic()-began,candidate=record.get('candidate'),error=record.get('error'))),flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
