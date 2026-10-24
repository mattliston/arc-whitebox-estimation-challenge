"""Inert source admission and ONE paid, target-free32x2 RPC noise diagnostic.

Only the previously qualified closed-context/environment/publication lifecycle
is reused. The new kernel is not an inherited fixture, estimator or head.
"""
import argparse
import ast
import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
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
import weakref

HERE=Path(__file__).resolve().parent
NEW,P2=HERE.parent,HERE.parent.parent
RB1=NEW/'current_comp_root_batch_gram_v1'
OWNER=RB1/'worker.py'
OWNER_SHA='ac92c6f6a6de5aeca89d3e994509caffd8db392c105165edb41ffdd9efb78b38'
SCHEMA='current_comp_cubic_innovation_noise_v1'
CALL_BUDGET=14_000_000_000
OPENBLAS=NEW/'openblas_runtime_env_v1'
CLIENT=P2/'data_gate/.client-venv'
OSITE=OPENBLAS/'lib/python3.11/site-packages'
CSITE=CLIENT/'lib/python3.11/site-packages'
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
         'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
JCELL=NEW/'current_comp_joint_cell_attribution_v1'
C2=NEW/'current_comp_shared_gain_slice_v1/build_v1/estimator.py'
C2_SHA='c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
ROW=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3/rows/observer_fit/0035'
WEIGHTS_SHA='14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2'
MANIFEST_SHA='c68f3cddfb18ffa16e5d2ae77ac1f4801c9c6554704c4886d637216fb85923cd'
PACKET_SHA='1f29ab16f3454a00c180170c7993cdc048c1cb36b15f2c72f8bfb2ff6a259225'
CAPTURE_SEALS=(
 '7867fb6b645d042d88fb359d7f831708d2e7feb37e26d845309805aec4796804',
 '0bdb594c411d07ffe9c1fd5c21f394c51d8d558d147aa18752e4067b75021c54',
 '5e5589cbb3fa91d8ef8c3d6dae6587e68c4f174283c0e0e3ec049a44f15cbb78')
CAPTURE_WATCH_SEALS=(
 '790b0d8ad7e88df9aa258d16de35ec6f5638c3f202f3a0cae51f46812abb9a34',
 '4fd68816a97ae3fb3ee905923fc55eb2fb258eaa531e6359ae927c893551f04e',
 'aa5e395b811694fdbbedee8141cf62238aef6e017fb1c077b4a4cb12d6c703ed')
PRODUCER_SOURCES={
 JCELL/'adapter_v2.py':'6f30900ea7a0e3babf2b339c327475d5d64d16fa6ebfdc2ff74c44167e8b3510',
 JCELL/'capture_v2.py':'975f0a311c261e3bbe85f409b80872a4fe7ade8762dc08e38c3038508b36f8fc',
 JCELL/'watch_capture_v2.py':'111c67bdb425c53902046229ee2373f20ca02ab4ce03e48729c1ec935b99fdd4',
 JCELL/'CAPTURE_PROTOCOL_V2.md':'a84dc711246cbc218cc6712f9cc7e91e8e6a226a145b8f9a75ebc429376f111c',
 JCELL/'CAPTURE_WATCH_SPEC_V2.md':'1d0cc069a4018c2a8c7386fb6c9325fb0546b8b52b73ffaf658784c7ec5edbd6',
 C2:C2_SHA}
ARRAYS={
 'alive':(('pre8','alive'),'bool',132875536,1024,'6c3332f5f7f6931f7e444dc06887ba1c6e88ed20e00ea3b6274869bb88d414ad'),
 'gate':(('pair8','aux','mapping','gate'),'float64',175040536,8192,'95522e6eb61cd0e3cfb107631632c2d5b3aa1dc0aef2cd147c4311176e0b82bb'),
 'alpha':(('pair9','aux','mapping','alpha'),'float64',342417720,8192,'70ae5d1bd067212a49a011e9215a7a02faf7dd1cf1e42a88176a96d46943b549'),
 'pdf':(('pair9','aux','mapping','pdf'),'float64',342425912,8192,'d3dba626fb56da7c8e88165d70ffc8f73c8ab08c11b8934a0d023d313970254d'),
 'sigma':(('pair9','aux','mapping','sigma'),'float64',342409528,8192,'08846afbaf2b27080f98bfba36cd1866ec09644b883a527e76f508c0d34635fe')}
_KERNEL=None
_ADMISSION=None


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def _load(path,digest,name):
    require(path.is_file() and not path.is_symlink()
        and hashlib.sha256(path.read_bytes()).hexdigest()==digest,'Exact inert owner: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_R=_load(OWNER,OWNER_SHA,'noise_qualified_RPC_lifecycle_owner')
_BASE,_ENV=_R._BASE,_R._ENV
sha,read,write,numerical_modules=(_ENV[k] for k in ('sha','read','write','numerical_modules'))
backend_attestation=_ENV['backend_attestation']
TinyCallFailure,_error_and_release=_R.TinyCallFailure,_R._error_and_release
PINS={**_R.PINS,**PRODUCER_SOURCES,OWNER:OWNER_SHA,
 OSITE/'flopscope/_registry.py':'4e091020d5cb8167271fa6ed76ac43983664e3a3cdd9e67ae76968f3884f6623',
 OSITE/'flopscope/_pointwise.py':'7d376515446f1b174bef375c48116a552098bd514f5552450876f56822beb0c7',
 OSITE/'flopscope/_array_ops.py':'e45e927f18c54352503a4951d3d925a5f31f92ecea245e32360825d71a05da6c',
 RB1/'client.py':'b1563e8eb542ac5e8a4dc958fb5515223749bbbbd49be48f5fe58c4cced7b82d',
 RB1/'watch.py':'49efaa4392c26bade85b3fc050ef85832da5dadf1caefc3066c2e340b1e94bf9',
 RB1/'PROTOCOL.md':'704db621eaa27697367318371c043fc641b9bd12b6c87d425c13b1fb19ce3d0f',
 RB1/'WATCH_SPEC.md':'de62c9d98b2dfa03924c4281b91668acbdce7637cadff3c97e34a432026c1bac'}


def _merge(target,items,tick):
    for path,digest in items.items():
        path=str(Path(path))
        require(re.fullmatch('[0-9a-f]{64}',digest or '')
            and (path not in target or target[path]==digest),'Consistent actual file authority')
        if path not in target:
            require(sha(Path(path),tick)==digest,'Fresh exact source/input/artifact: '+path)
            target[path]=digest


def _three(folder,pins,schema,inputs,tick):
    paths=[folder/n for n in ('result.json','completion.json','finalization.json')]
    _merge(inputs,{str(p):d for p,d in zip(paths,pins)},tick)
    r,e,f=map(read,paths)
    require(r['schema']==schema and all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
        and e['result_sha256']==f['result_sha256']==pins[0] and f['completion_sha256']==pins[1],
        'Exact actual positive producer three-seal chain')
    return r


def _npy_header():
    path=ROW/'weights.npy'
    with path.open('rb') as stream:
        require(stream.read(8)==b'\x93NUMPY\x01\x00','Canonical NPYv1 weight representation')
        length=struct.unpack('<H',stream.read(2))[0]
        require(0<length<4096,'Bounded NPY header')
        header=ast.literal_eval(stream.read(length).decode('latin1'))
        offset=stream.tell()
    require(header==dict(descr='<f4',fortran_order=False,shape=(16,1024,1024))
        and path.stat().st_size==offset+16*1024*1024*4,'Exact canonical physical weights, no array decoding')
    return offset


def admission(args,tick=None):
    """Only SHA/JSON/NPY-header metadata. No producer execution or label access."""
    global _KERNEL,_ADMISSION
    require(not numerical_modules(),'Metadata admission before every numerical import')
    own={**PINS,HERE/'worker.py':args.source_sha256,HERE/'client.py':args.client_sha256,
        HERE/'kernel.py':args.kernel_sha256,HERE/'PROTOCOL.md':args.protocol_sha256,
        HERE/'watch.py':args.watcher_sha256,HERE/'WATCH_SPEC.md':args.watch_spec_sha256}
    sources,inputs={},{}
    _merge(sources,own,tick)
    _KERNEL=_load(HERE/'kernel.py',args.kernel_sha256,'fixed_cubic_noise_math')
    r=_three(JCELL/'capture_v1',CAPTURE_SEALS,'current_comp_joint_cell_capture_v1',inputs,tick)
    w=_three(JCELL/'capture_watch_v1',CAPTURE_WATCH_SEALS,'current_comp_joint_cell_capture_watch_v1',inputs,tick)
    require(r['candidate_sha256']==C2_SHA and r['ranks']==[35] and r['retired_call_count']==1
        and all(r.get(k) is True for k in ('all16_domains_qualified','all47_selectors_qualified',
            'all_capture_arrays_released','all_original_operations_exact',
            'all_parameter_and_estimator_owners_retired','baseline_output_byte_exact',
            'own_complete_bills','hashes_unchanged','sources_unchanged'))
        and r['truth_accessed'] is False and r['score_measured'] is False,
        'Actual unchanged-c2 target-blind rank35 capture, not an oracle supply')
    require(all(w.get(k) is True for k in ('hashes_unchanged','sources_unchanged','cleanup_passed',
        'whole_owned_tree_retired','observational_groups_retired'))
        and [w['child_'+k+'_sha256'] for k in ('result','completion','finalization')]==list(CAPTURE_SEALS)
        and 0<w['maximum_combined_rss_bytes']<8_000_000_000
        and 0<w['maximum_interval_s']<=1 and w['maximum_swap_growth_bytes']==0,
        'Actual positive independently observed whole capture')
    proof=r['capture_source_proof']
    require(proof['parent_sha256']==C2_SHA and proof['source_inverse'] is True
        and proof['original_physical_arithmetic_unchanged'] is True
        and proof['derived_sha256']=='d6f84caaf91fa198be71760fc5477f99c18217ff977a0341889c586941fea43a',
        'Sealed exact passive-capture source inverse')
    for path,digest in PRODUCER_SOURCES.items():
        require(r['source_bindings'][str(path)]==w['source_bindings'][str(path)]==digest,
            'Actual producer owns each fixed source')
    # Positive fixed receipts are the trust boundary. Rehash their final union;
    # never import/reexecute any ancestral estimator or numerical verifier.
    for record in (r,w):
        _merge(sources,record['source_bindings'],tick)
        _merge(inputs,record['input_bindings'],tick)
        _merge(inputs,record['artifacts'],tick)
    manifest_path=JCELL/'capture_v1/restart_packet.json'
    packet_path=JCELL/'capture_v1/restart_packet.bin'
    require(r['artifacts'][str(manifest_path)]==MANIFEST_SHA
        and r['artifacts'][str(packet_path)]==PACKET_SHA
        and r['input_bindings'][str(ROW/'weights.npy')]==WEIGHTS_SHA,
        'Capture packet and actual same rank35 physical fixture joined')
    manifest=read(manifest_path)
    require(manifest['schema']=='current_comp_joint_cell_packet_v1'
        and manifest['candidate_sha256']==C2_SHA
        and (manifest['rank'],manifest['layer'],manifest['consumer'])==(35,8,9)
        and manifest['packet_file']==str(packet_path) and manifest['packet_sha256']==PACKET_SHA
        and manifest['bytes']==packet_path.stat().st_size==351195456
        and manifest['saved_after_context_release'] is True
        and manifest['all_explicit_arrays_released'] is True and manifest['truth_accessed'] is False,
        'Actual retired packet and exact current layer frame')
    for name,(route,dtype,offset,size,digest) in ARRAYS.items():
        item=manifest['packets']
        for key in route:
            item=item[key]
        item=item['array']
        require(item['dtype']==dtype and item['shape']==[1024] and item['byteorder']=='little'
            and item['offset']==offset and item['bytes']==size and item['sha256']==digest
            and item['original_type_name']=='FlopscopeArray'
            and item['original_type_module']=='flopscope._ndarray'
            and item['original_c_contiguous'] and item['original_f_contiguous']
            and item['original_strides']==[size//1024], 'Exact captured vector metadata: '+name)
    row=read(ROW/'row_manifest.json')
    require(row['role_rank']==35 and row['mlp_seed']==2758000291903877946
        and row['files']['weights.npy']['sha256']==WEIGHTS_SHA,'Canonical rank35 fixture identity')
    _ADMISSION=dict(source_bindings=sources,input_bindings=inputs,
        capture_seals=list(CAPTURE_SEALS),capture_watch_seals=list(CAPTURE_WATCH_SEALS),
        weights_payload_offset=_npy_header(),candidate_sha256=C2_SHA,
        packet_manifest_sha256=MANIFEST_SHA,packet_sha256=PACKET_SHA,
        cost_certificate=_KERNEL.cost_certificate(),truth_accessed=False,numerical_imports=False)
    require(not numerical_modules(),'All admission/source certificate remains inert')
    return dict(_ADMISSION)


def admit(args,guard=None):
    return admission(args,guard)['source_bindings']


def helper_source():
    require(_KERNEL is not None and _ADMISSION is not None,'Complete fixed source/producer admission first')
    raw=(HERE/'kernel.py').read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    require(digest==_ADMISSION['source_bindings'][str(HERE/'kernel.py')],'Same admitted new arithmetic')
    return raw.decode(),dict(kernel_sha256=digest,candidate_sha256=C2_SHA,
        capture_seals=list(CAPTURE_SEALS),capture_watch_seals=list(CAPTURE_WATCH_SEALS),
        packet_sha256=PACKET_SHA,packet_manifest_sha256=MANIFEST_SHA,
        cost_certificate=_KERNEL.cost_certificate(),new_arithmetic_not_inherited_fixture=True,
        no_full_estimator_or_heads=True,truth_accessed=False,numerical_execution=False)


def load_serialized():
    require(_ADMISSION is not None,'Actual admitted fixed bytes only')
    with (ROW/'weights.npy').open('rb') as stream:
        stream.seek(_ADMISSION['weights_payload_offset'])
        data=stream.read(10*1024*1024*4)
    require(len(data)==10*1024*1024*4,'Complete W0..W9 stored FP32 byte slab')
    result={'weights':data}
    with (JCELL/'capture_v1/restart_packet.bin').open('rb') as stream:
        for name,(_,dtype,offset,size,digest) in ARRAYS.items():
            stream.seek(offset);raw=stream.read(size)
            require(len(raw)==size and hashlib.sha256(raw).hexdigest()==digest,'Exact selected packet bytes')
            result[name]=raw
    return result


def primitive_namespace(fnp):
    return dict(fnp=fnp),helper_source()[1]


def _exercise(fnp,ns,ctx,audit):
    sections,rows,rejected,refs=(audit[k] for k in ('sections','rows','rejected','refs'))
    def own(value):
        if hasattr(value,'shape') and hasattr(value,'dtype'):
            refs.append(weakref.ref(value))
        elif isinstance(value,dict):
            for item in value.values(): own(item)
        elif isinstance(value,(tuple,list)):
            for item in value: own(item)
        return value
    def section(name,function,expected=None):
        require(name==_KERNEL.SECTIONS[len(sections)],'Fixed complete section order')
        before=int(ctx.flops_used);began=time.perf_counter();audit['active_section']=name
        try:
            value=own(function())
        except BaseException as error:
            sections.append(dict(name=name,flops=int(ctx.flops_used)-before,
                expected_flops=None,function_completed=False,error_type=type(error).__name__))
            raise
        used=int(ctx.flops_used)-before
        maximum=_KERNEL.SECTION_CAPS[len(sections)]
        sections.append(dict(name=name,flops=used,expected_flops=None,maximum_flops=maximum,
            function_completed=True,tariff_matched=0<=used<=maximum,
            helper_wall_s=time.perf_counter()-began))
        require(0<=used<=maximum,'Actual complete section within predeclared certificate')
        audit['active_section']=None
        return value
    rows.extend(_KERNEL.run(fnp,load_serialized,section,own,ctx))
    return rows,sections,rejected,refs


def verify_call(record,mode,index,tick=None):
    require(record['schema']==SCHEMA and mode==record['mode']=='remote' and index==record['call']==0
        and all(record.get(k) is True for k in ('passed','completed','context_retired',
            'all_array_weakrefs_retired','own_complete_bill','fixed_noise_only',
            'remote_sections_are_actual_budget_deltas'))
        and record['model_calls']==0 and record['target_accessed'] is False,
        'ONE actual closed target-free RPC context')
    sections=record['sections']
    require([s['name'] for s in sections]==list(_KERNEL.SECTIONS)
        and all(s.get('function_completed') and s.get('tariff_matched')
            and type(s['flops']) is int and 0<=s['flops']<=cap
            and s['maximum_flops']==cap and s['expected_flops'] is None
            for s,cap in zip(sections,_KERNEL.SECTION_CAPS))
        and sum(s['flops'] for s in sections)==record['complete_flops']==record['bill']['flops_used']
        and 0<record['complete_flops']<CALL_BUDGET,'All ingress/math/guards/exports in the actual whole bill')
    require(len(record['rows'])==1 and record['rejected_cases']==[],'Only the fixed32x2 law, no extra arm')
    row=record['rows'][0];p=row['primary']
    require(row['name']=='fixed32_two_tangent_noise' and row['outer']==32
        and row['independent_tangent_replicas']==2 and row['seed']==_KERNEL.SEED
        and row['physical_prefix_forward_passes']==3 and row['physical_tangent_prefix_passes']==4
        and row['matmul_counts']==_KERNEL.GEMM_COUNTS and row['matmul_flops']==89*_KERNEL.DENSE_ONE
        and row['current_gate_includes_alive'] and row['hard_pathwise_gates_include_alive']
        and row['raw_mean_or_final_score_measured'] is False and row['outer_variance_measured'] is False,
        'Fixed current-gate conditional tangent-noise scope')
    require(p['threshold']==1e-9 and all(math.isfinite(p[k]) and p[k]>=0 for k in
        ('paired_average_difference_mse','rowwise_tangent_variance_V32'))
        and p['paired_average_passed']==(p['paired_average_difference_mse']<=1e-9)
        and p['rowwise_variance_passed']==(p['rowwise_tangent_variance_V32']<=1e-9)
        and p['host_noise_precision_passed']==(p['paired_average_passed'] and p['rowwise_variance_passed']),
        'BOTH prospective HOST precision gates, never a selected favorable metric')
    for name,item in row['exports'].items():
        shape=[32,1024] if name=='scaled_rowwise_difference' else [1024]
        require(item['shape']==shape and item['dtype']=='float64'
            and item['bytes']==8*math.prod(shape) and re.fullmatch('[0-9a-f]{64}',item['raw_sha256']),
            'Detached paid output representation checksum')
    require(set(row['exports'])=={'replica0_host_average','replica1_host_average',
        'paired_average_difference','scaled_rowwise_difference'},'Fixed exported checksums only')
    bill=record['bill']
    require(all(math.isfinite(bill[k]) and bill[k]>=0 for k in
        ('wall_time_s','residual_wall_time_s','flopscope_backend_time_s','flopscope_overhead_time_s'))
        and bill['wall_time_s']<60 and bill['residual_wall_time_s']<.4,
        'Actual full original RPC wall60/residual.4 gate')


def call_records(result,tick=None):
    require(result['mode']=='remote' and len(result['calls'])==1,'One actual retired diagnostic')
    path=HERE/'noise_v1/remote/call_00.json';item=result['calls'][0]
    require(item['path']==str(path) and sha(path,tick)==item['sha256']==result['artifacts'][str(path)],
        'Exact exclusive closed-call receipt')
    record=read(path);verify_call(record,'remote',0,tick)
    return [record]


def compare(remote,tick=None):
    require(all(remote.get(k) is True for k in ('passed','completed','own_complete_bill',
        'one_context_completed','source_hashes_unchanged','input_hashes_unchanged'))
        and remote['retired_contexts']==1 and remote['input_bindings']==_ADMISSION['input_bindings'],
        'Fresh positive actual context and full fixed producer inventory')
    record=call_records(remote,tick)[0]
    require(remote['calls_complete_flops']==record['complete_flops'],
        'Reported total is this actual complete closed bill')
    row=record['rows'][0]
    return dict(passed=True,one_context_completed=True,primary=row['primary'],
        host_noise_precision_passed=row['primary']['host_noise_precision_passed'],
        complete_flops=remote['calls_complete_flops'],fixed_noise_only=True,
        model_calls=0,truth_accessed=False,outer_variance_measured=False,
        signed_innovation_measured=False,raw_value_measured=False,
        full_estimator_fidelity_established=False,automatic_next_stage=False)


def _change(source,before,after,changes):
    require(source.count(before)==1,'One exact lifecycle-only splice: '+before[:100])
    at=source.index(before);changes.append((at,before,after))
    return source[:at]+after+source[at+len(before):]


def _inverse(source,original,changes):
    for at,before,after in reversed(changes):
        require(source[at:at+len(after)]==after,'Exact lifecycle positional inverse')
        source=source[:at]+before+source[at+len(after):]
    require(source==original,'Entire original effective lifecycle restored')


def derive_main():
    original,parent=_R.derive_main();source=original;changes=[]
    edits=(
      ("HERE/'unit_v1'/args.mode","HERE/'noise_v1'/args.mode"),
      ("HERE/'unit_v1/server_attestation.json'","HERE/'noise_v1/server_attestation.json'"),
      ('    admitted = admit(args,guard)\n',
       "    context = admission(args,guard)\n    admitted = context['source_bindings']\n"),
      ('        admitted_sources=admitted,input_bindings={},artifacts={},calls=[],primitive_proof=proof,',
       "        admitted_sources=admitted,input_bindings=dict(context['input_bindings']),artifacts={},calls=[],primitive_proof=proof,"),
      ("            source_bindings=sources,admitted_sources=admitted,pid=os.getpid(),url=args.server_url,",
       "            source_bindings=sources,admitted_sources=admitted,input_bindings=context['input_bindings'],pid=os.getpid(),url=args.server_url,"),
      ("        result.update(passed=True,completed=True,own_complete_bill=True,source_hashes_unchanged=True,",
       "        for path,digest in result['input_bindings'].items():\n"
       "            require(sha(Path(path),guard)==digest,'Fresh final complete input/producer union')\n"
       "        result['input_hashes_unchanged']=True\n"
       "        result['calls_complete_flops']=record['complete_flops']\n"
       "        result.update(passed=True,completed=True,own_complete_bill=True,source_hashes_unchanged=True,"),
      ('            all_independent_references_passed=True,helper_tariffs_exact=True,\n'
       '            byte_exact=True,helper_repeats_exact=True,fixed_D5_Gram_only=True,repeat_claimed=True,',
       '            actual_noise_circuit_qualified=True,complete_sections_reconciled=True,\n'
       '            fixed_noise_only=True,repeat_claimed=False,'),
    )
    for before,after in edits: source=_change(source,before,after,changes)
    _inverse(source,original,changes)
    return source,dict(parent=parent,owner_sha256=OWNER_SHA,whole_effective_main_inverse=True,
        derived_sha256=hashlib.sha256(source.encode()).hexdigest(),new_arithmetic_is_separately_owned=True,
        context_bill_environment_and_retirement_unchanged=True,one_RPC_context_only=True)


def arithmetic(flops,fnp,ns,mode,index,guard):
    original=_BASE.ARITHMETIC;source=original;changes=[]
    for before,after in (("namespace='lazy-rotation-tiny'","namespace='cubic-innovation-noise'"),
        ('dense_branch_only=True','fixed_noise_only=True')):
        source=_change(source,before,after,changes)
    _inverse(source,original,changes)
    namespace=dict(globals());exec(compile(source,str(HERE/'worker.py')+':closed-budget-inverse','exec'),namespace)
    return namespace['arithmetic'](flops,fnp,ns,mode,index,guard)


def main(forced_mode='local'):
    source,proof=derive_main();env=dict(_ENV)
    env.update({k:v for k,v in globals().items() if k not in ('main','_R','_BASE','_ENV')})
    env['SOURCE_ADAPTER_PROOF']=proof
    exec(compile(source,str(HERE/'worker.py')+':whole-main-inverse','exec'),env)
    return env['main'](forced_mode=forced_mode)


if __name__=='__main__':
    raise SystemExit(main())
