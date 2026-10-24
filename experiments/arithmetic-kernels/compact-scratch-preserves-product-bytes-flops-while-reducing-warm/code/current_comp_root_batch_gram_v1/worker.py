"""DRAFT inert, one paid NumPy-free RPC Gram unit. Never an estimator run.

The frozen lazy-v2 source supplies only its stdlib environment/attestation,
single-context entry, closed-budget failure ledger and linked publication.
Its old numerical fixture/function is never called.
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
import sys
import time
import traceback
import weakref

HERE=Path(__file__).resolve().parent
NEW,P2=HERE.parent,HERE.parent.parent
LAZY=NEW/'current_comp_shared_lazy_rotation_v1'
OWNER=LAZY/'tiny_worker_v2.py'
OWNER_SHA='7bac9bac7a5f331ba700e6ca2f4709db0ad94e7f41285c716a1e0254ae371fa0'
OWNER_SPEC=LAZY/'TINY_V2_PROTOCOL.md'
OWNER_SPEC_SHA='46d8f661531b6cb7b77c91793b4f0c95f921dea85c8d49bf945e9434c701d78f'
SCHEMA='current_comp_root_batch_gram_unit_v1'
CALL_BUDGET=10_000_000_000
ROSTER=('serial_first','batch_first','serial_warm','batch_warm')
OPENBLAS=NEW/'openblas_runtime_env_v1'
CLIENT=P2/'data_gate/.client-venv'
OSITE=OPENBLAS/'lib/python3.11/site-packages'
CSITE=CLIENT/'lib/python3.11/site-packages'
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
         'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
_KERNEL=None


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def _load(path,digest,name):
    require(not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest()==digest,
            'Exact inert primitive owner: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# These are inert definitions/metadata. No inherited _exercise/main is invoked.
_BASE=_load(OWNER,OWNER_SHA,'root_batch_qualified_closed_budget_provider')
_ENV=_BASE._api['_BASE']
sha,read,write,numerical_modules=(_ENV[k] for k in ('sha','read','write','numerical_modules'))
backend_attestation=_ENV['backend_attestation']
TinyCallFailure,_error_and_release=_BASE.TinyCallFailure,_BASE._error_and_release
PINS={p:d for p,d in _BASE.PINS.items() if p.is_relative_to(OSITE) or p.is_relative_to(CSITE)}
PINS.update(_BASE.V1_SOURCES)
PINS.update({OWNER:OWNER_SHA,OWNER_SPEC:OWNER_SPEC_SHA,
    _BASE._api['OWNER']:_BASE._api['OWNER_SHA'],
    NEW/'current_comp_shared_pair256_v1/tiny_client.py':
        'a8876ee4dffab18e5c468c90a91ef7c96289264ecf5014d95b2feea7cdf6ca10',
    OSITE/'flopscope/numpy/random/__init__.py':
        '224414ad4e756e42f587259e1567088d2535c712c77ddb342f64e80c79d95029',
    OSITE/'flopscope/numpy/random/_counted_classes.py':
        'b7c8b29a339c0cdf2e65c660ca733c96db90313bc62649a635d4b57341918f2d',
    OSITE/'flopscope/numpy/random/_cost_formulas.py':
        'd14d86a2ca0700c0899318a9c7cd3f08e91ac80948682225d383d71e2d628f8f',
    CSITE/'flopscope/random/__init__.py':
        'fe1a04659f8a80cc033987e951b4b95d6b1b6946db67347a4f2cc0ec851fb13b',
    CSITE/'flopscope/_registry_data.py':
        '312fd35ec7ad526d423a4e7f056ec6560ff4f64d02416d2d88351c723acf180f'})


def owner_map(args):
    global _KERNEL
    _KERNEL=_load(HERE/'kernel.py',args.kernel_sha256,'root_batch_fixed_kernel_owner')
    return {**PINS,**_KERNEL.PINS,HERE/'worker.py':args.source_sha256,
        HERE/'client.py':args.client_sha256,HERE/'kernel.py':args.kernel_sha256,
        HERE/'PROTOCOL.md':args.protocol_sha256,HERE/'watch.py':args.watcher_sha256,
        HERE/'WATCH_SPEC.md':args.watch_spec_sha256}


def admit(args,guard=None):
    require(not numerical_modules(),'No numerical imports before exact source admission')
    owners=owner_map(args)
    for path,digest in owners.items():
        require(re.fullmatch('[0-9a-f]{64}',digest or '') and sha(path,guard)==digest,
                'Reviewed whole-unit owner: '+str(path))
    _KERNEL.source_bundle(guard)
    require(not numerical_modules(),'Kernel source proof is inert')
    return {str(p):d for p,d in owners.items()}


def helper_source():
    require(_KERNEL is not None,'Actual kernel SHA admitted first')
    source,_,_,proof=_KERNEL.source_bundle()
    return source,proof


def primitive_namespace(fnp):
    require(_KERNEL is not None,'Admitted primitive only')
    return _KERNEL.namespace(fnp)


def _exercise(fnp,ns,ctx,audit):
    sections,rows,rejected,refs=(audit[k] for k in ('sections','rows','rejected','refs'))
    def own(value):
        if hasattr(value,'shape') and hasattr(value,'dtype'):
            refs.append(weakref.ref(value))
        elif isinstance(value,dict):
            for item in value.values():
                own(item)
        elif isinstance(value,(tuple,list)):
            for item in value:
                own(item)
        return value
    def section(name,function,expected=None):
        before=int(ctx.flops_used)
        audit['active_section']=name
        began=time.perf_counter()
        try:
            value=function()
        except BaseException as error:
            sections.append(dict(name=name,flops=int(ctx.flops_used)-before,
                expected_flops=expected,function_completed=False,error_type=type(error).__name__))
            raise
        elapsed=time.perf_counter()-began
        own(value)
        used=int(ctx.flops_used)-before
        sections.append(dict(name=name,flops=used,expected_flops=expected,
            function_completed=True,tariff_matched=used>=0 and (expected is None or used==expected),
            helper_wall_s=elapsed))
        require(used>=0 and (expected is None or used==expected),'Actual complete section tariff: '+name)
        audit['active_section']=None
        return value
    def finite(value):
        require(bool(fnp.all(fnp.isfinite(value))),'All actual array values finite')
    def serialized(value):
        # Paid explicit representation copy; fetch/hash/byte equality do not
        # calculate numerical statistics. The FNP finite scan is paid here.
        require(tuple(value.shape)==(1024,1024) and str(value.dtype)=='float32',
                'Only the fixed typed Gram operand/output')
        snapshot=value.astype(fnp.float32)
        finite(snapshot)
        raw,shape,dtype=snapshot._fetch_data()
        require(tuple(shape)==(1024,1024) and str(dtype)=='float32' and len(raw)==4194304,
                'Exact serialized stored FP32 payload')
        return raw
    def reference_check(actual,reference):
        widened=actual.astype(fnp.float64)
        finite(widened)
        delta=widened-reference
        norm=fnp.sqrt(fnp.sum(reference*reference))
        denominator=fnp.maximum(norm,fnp.full((1,),1e-30,dtype=fnp.float64))[0]
        relative=fnp.sqrt(fnp.sum(delta*delta))/denominator
        maximum=fnp.max(fnp.abs(delta))
        return dict(reference_norm=float(norm),relative_frobenius=float(relative),maxabs=float(maximum))

    try:
        section('all_D5_bindings',lambda:_KERNEL.setup(fnp,ns),689569792)
        def fixture():
            # Same single PCG64 stream and normal call as check_synthetic.
            scale=fnp.sqrt(fnp.full((1,),0.001953125,dtype=fnp.float64))[0]
            rng=fnp.random.default_rng(938248)
            right=rng.normal(0.0,float(scale),(1024,1024)).astype(fnp.float32)
            diagonal=fnp.ones(1024,dtype=fnp.float32)
            left=(right*diagonal[:,None]).T.copy()
            del rng,scale,diagonal
            return left,right
        left,right=section('FNP_rng_and_fixed_operands',fixture)
        raw_left=section('serialize_left',lambda:serialized(left))
        raw_right=section('serialize_right',lambda:serialized(right))
        outputs=[]
        cache_history=[]
        for index,name in enumerate(ROSTER):
            before_cache=[len(ns['_PACKED_FRAME_CACHE']),len(ns['_ROOT_FRAME_CACHE'])]
            require(before_cache==([0,0] if index==0 else [1,0] if index==1 else [1,1]),
                    'Actual prescribed first-touch/warm cache state')
            function=(lambda:ns['_resident_gram512'](left,right)) if name.startswith('serial') else (
                lambda:_KERNEL.batched_gram(fnp,ns,left,right))
            value=section(name,function,_KERNEL.WARM_FLOPS)
            after_cache=[len(ns['_PACKED_FRAME_CACHE']),len(ns['_ROOT_FRAME_CACHE'])]
            require(after_cache==([1,0] if index==0 else [1,1]),'Exactly one frame per helper')
            cache_history.append(dict(before=before_cache,after=after_cache))
            outputs.append(value)
        # Reference is constructed from the exact serialized FP32 operands,
        # widened once; no new RNG draw and no unmetered NumPy calculation.
        def reference_operands():
            a=fnp.frombuffer(raw_left,dtype=fnp.float32).reshape((1024,1024)).astype(fnp.float64)
            b=fnp.frombuffer(raw_right,dtype=fnp.float32).reshape((1024,1024)).astype(fnp.float64)
            finite(a);finite(b)
            return a,b
        a64,b64=section('serialized_operand_FP64_reference_copies',reference_operands)
        reference=section('one_dense_FP64_reference',lambda:a64@b64,4292870144)
        section('reference_finite',lambda:finite(reference))
        raw_outputs=[]
        for index,(name,value) in enumerate(zip(ROSTER,outputs)):
            raw=section(name+'_snapshot',lambda:serialized(value))
            diagnostics=section(name+'_reference_check',lambda:reference_check(value,reference))
            raw_outputs.append(raw)
            rows.append(dict(name=name,dtype='float32',shape=[1024,1024],finite=True,
                raw_sha256=hashlib.sha256(raw).hexdigest(),serialized_bytes=len(raw),
                helper_flops=_KERNEL.WARM_FLOPS,
                helper_wall_s=next(s['helper_wall_s'] for s in sections if s['name']==name),
                frame_cache=cache_history[index],
                input_left_sha256=hashlib.sha256(raw_left).hexdigest(),
                input_right_sha256=hashlib.sha256(raw_right).hexdigest(),**diagnostics))
        require(raw_outputs[0]==raw_outputs[1]==raw_outputs[2]==raw_outputs[3],
                'Actual serial/batch byte equality and within-helper repeat equality')
        require(all(r['maxabs']<=5e-5 and r['relative_frobenius']<=2e-5
                    and r['reference_norm']>0 for r in rows),'Original stored-operand Gram reference gates')
        own(ns['_PACKED_FRAME_CACHE']);own(ns['_ROOT_FRAME_CACHE'])
        del outputs,left,right,a64,b64,reference,value,function,raw_outputs,raw,raw_left,raw_right
    finally:
        # No frame, arena or output proxy is retained in the helper namespace.
        _KERNEL.clear(ns)
    return rows,sections,rejected,refs


def verify_call(record,mode,index,tick=None):
    require(mode=='remote' and index==0 and record['schema']==SCHEMA
        and record['mode']==mode and record['call']==index
        and all(record.get(k) is True for k in ('passed','completed','context_retired',
            'all_array_weakrefs_retired','own_complete_bill','fixed_D5_Gram_only',
            'remote_sections_are_actual_budget_deltas'))
        and record['model_calls']==0 and record['target_accessed'] is False,
        'One genuinely retired paid RPC context, not a model')
    sections=record['sections']
    exact_sections=['all_D5_bindings','FNP_rng_and_fixed_operands','serialize_left','serialize_right',
        *ROSTER,'serialized_operand_FP64_reference_copies','one_dense_FP64_reference','reference_finite',
        *(name+suffix for name in ROSTER for suffix in ('_snapshot','_reference_check'))]
    require([s['name'] for s in sections]==exact_sections
        and len({s['name'] for s in sections})==len(sections)
        and all(s.get('function_completed') and s.get('tariff_matched')
            and type(s['flops']) is int and s['flops']>=0
            and (s['expected_flops'] is None or s['flops']==s['expected_flops']) for s in sections)
        and sum(s['flops'] for s in sections)==record['complete_flops']==record['bill']['flops_used']
        and 0<record['complete_flops']<=CALL_BUDGET,'All actual costs, no warm credit against total')
    expected={'all_D5_bindings':689569792,'one_dense_FP64_reference':4292870144,
              **{name:955654144 for name in ROSTER}}
    for name,flops in expected.items():
        found=[s for s in sections if s['name']==name]
        require(len(found)==1 and found[0]['flops']==flops,'Fixed setup/core/reference full tariff')
    rows=record['rows']
    require([r['name'] for r in rows]==list(ROSTER) and not record['rejected_cases'],
            'Exactly four fixed helper calls; no adaptive repeat or extra test')
    require(len({r['raw_sha256'] for r in rows})==1
        and len({r['input_left_sha256'] for r in rows})==1
        and len({r['input_right_sha256'] for r in rows})==1,'Serialized byte exact comparisons')
    for index,row in enumerate(rows):
        require(row['finite'] and row['dtype']=='float32' and row['shape']==[1024,1024]
            and row['serialized_bytes']==4194304 and row['helper_flops']==955654144
            and re.fullmatch('[0-9a-f]{64}',row['raw_sha256'])
            and all(math.isfinite(row[k]) and row[k]>=0 for k in
                    ('maxabs','relative_frobenius','reference_norm','helper_wall_s'))
            and row['maxabs']<=5e-5 and row['relative_frobenius']<=2e-5
            and row['reference_norm']>0 and row['helper_wall_s']>0,
            'Actual finite stored-operand reference and direct helper measurement')
        require(row['helper_wall_s']==next(s['helper_wall_s'] for s in sections if s['name']==row['name']),
                'Timing is the fixed helper block, not a whole bill difference')
        require(row['frame_cache']==dict(before=[0,0] if index==0 else [1,0] if index==1 else [1,1],
                    after=[1,0] if index==0 else [1,1]),'Exact first/warm cache metadata')
    bill=record['bill']
    require(all(math.isfinite(bill[k]) and bill[k]>=0 for k in
        ('wall_time_s','residual_wall_time_s','flopscope_backend_time_s','flopscope_overhead_time_s'))
        and bill['wall_time_s']<60. and bill['residual_wall_time_s']<.4,
        'Unchanged actual RPC context60s/residual<.4 gate')


def call_records(result,tick=None):
    require(result['mode']=='remote' and len(result['calls'])==1,'One actual context')
    path=HERE/'unit_v1/remote/call_00.json'
    item=result['calls'][0]
    require(item['path']==str(path) and sha(path,tick)==item['sha256']==result['artifacts'][str(path)],
            'Exclusive real retired context receipt')
    record=read(path)
    verify_call(record,'remote',0,tick)
    return [record]


def compare(remote,tick=None):
    require(all(remote.get(k) is True for k in ('passed','completed','own_complete_bill',
        'one_context_completed','source_hashes_unchanged','helper_tariffs_exact','byte_exact',
        'helper_repeats_exact','fixed_D5_Gram_only')) and remote['retired_contexts']==1,
        'Fresh complete one-context worker authority')
    record=call_records(remote,tick)[0]
    rows=record['rows']
    ratio=rows[3]['helper_wall_s']/rows[2]['helper_wall_s']
    require(math.isfinite(ratio) and ratio>0,'Actual fixed warm ratio')
    return dict(passed=True,one_context_completed=True,all_independent_references_passed=True,
        helper_tariffs_exact=True,byte_exact=True,helper_repeats_exact=True,
        complete_flops=record['complete_flops'],warm_helper_flops=955654144,
        warm_wall_ratio=ratio,component_gate_passed=ratio<=.90,
        timing_condition='batch_warm/serial_warm <= 0.90',runtime_gain_measured=True,
        full_estimator_fidelity_established=False,full_45s_gate_established=False,
        raw_value_measured=False,model_calls=0,truth_accessed=False,automatic_next_stage=False)


def derive_main():
    original,parent_proof=_BASE.derive_main()
    source,changes=original,[]
    for before,after in (
        ("('source','client','protocol','watcher','watch-spec','decision')",
         "('source','client','protocol','watcher','watch-spec','kernel')"),
        ("HERE/'tiny_qualification_v2'/args.mode","HERE/'unit_v1'/args.mode"),
        ("HERE/'tiny_qualification_v2/server_attestation.json'","HERE/'unit_v1/server_attestation.json'"),
        ("            all_independent_references_passed=True,dense_component_tariffs_exact=True,\n"
         "            expected_rejections_passed=True,dense_branch_only=True,repeat_claimed=False,",
         "            all_independent_references_passed=True,helper_tariffs_exact=True,\n"
         "            byte_exact=True,helper_repeats_exact=True,fixed_D5_Gram_only=True,repeat_claimed=True,"),
    ):
        require(source.count(before)==1,'Exact inherited closed-context main splice')
        at=source.index(before);changes.append((at,before,after))
        source=source[:at]+after+source[at+len(before):]
    inverse=source
    for at,before,after in reversed(changes):
        require(inverse[at:at+len(after)]==after,'Whole main positional inverse')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    require(inverse==original,'Complete qualified single-RPC lifecycle restored')
    return source,dict(parent=parent_proof,owner_sha256=OWNER_SHA,
        whole_effective_main_inverse=True,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        original_environment_closed_budget_failure_ledger_retained=True,no_old_numerical_fixture_called=True)


def arithmetic(flops,fnp,ns,mode,index,guard):
    original=_BASE.ARITHMETIC
    changes=(('namespace=\'lazy-rotation-tiny\'','namespace=\'root-batch-gram-unit\''),
             ('dense_branch_only=True','fixed_D5_Gram_only=True'))
    source=original
    for before,after in changes:
        require(source.count(before)==1,'Exact original acknowledged closed-budget arithmetic lifecycle')
        source=source.replace(before,after)
    inverse=source
    for before,after in reversed(changes):
        require(inverse.count(after)==1,'Exact lifecycle inverse')
        inverse=inverse.replace(after,before)
    require(inverse==original,'Original complete failure ledger and context retirement')
    env=dict(globals())
    exec(compile(source,str(Path(__file__).resolve())+':closed-budget-lifecycle','exec'),env)
    return env['arithmetic'](flops,fnp,ns,mode,index,guard)


def main(forced_mode='local'):
    source,proof=derive_main()
    env=dict(_ENV)
    env.update({k:v for k,v in globals().items() if k not in ('main','_BASE','_ENV')})
    env['SOURCE_ADAPTER_PROOF']=proof
    exec(compile(source,str(Path(__file__).resolve())+':whole-main-inverse','exec'),env)
    return env['main'](forced_mode=forced_mode)


if __name__=='__main__':
    raise SystemExit(main())
