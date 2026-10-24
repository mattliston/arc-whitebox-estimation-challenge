"""DRAFT exact qualified single-IPC unit supervisor,120s/8GB/no added swap.

Only prospective unit paths/owners/metadata change. The complete observation,
retired-context progress, process-group cleanup and final <=1s tail remain.
No numerical import, fixture decoding, candidate or endpoint beyond local IPC.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
LAZY=HERE.parent/'current_comp_shared_lazy_rotation_v1'
ROOT_OWNER=LAZY/'watch_tiny_v2.py'
ROOT_OWNER_SHA='771c39521502ea520fae2ca50ffe59e32fe9c520b78ba2b087fc5525f93b5f79'
ROOT_SPEC=LAZY/'TINY_V2_WATCH_SPEC.md'
ROOT_SPEC_SHA='0488a49fb7d6c3f79f0f879d0824401723c69161d7cea1ec5f1dd3477d32bd28'


def _owner():
    for path,digest in ((ROOT_OWNER,ROOT_OWNER_SHA),(ROOT_SPEC,ROOT_SPEC_SHA)):
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Exact qualified single-IPC resource owner')
    spec=importlib.util.spec_from_file_location('_root_batch_lazy_v2_watch',ROOT_OWNER)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def derive_source():
    base=_owner()
    original,parent_proof=base.derive_source()
    source,changes=original,[]
    def change(before,after,count=1):
        nonlocal source
        if source.count(before)!=count:
            raise RuntimeError('Exact root-batch supervisor splice count: '+before[:100])
        for unused in range(count):
            at=source.index(before)
            changes.append((at,before,after))
            source=source[:at]+after+source[at+len(before):]
    change("('source','worker','client','protocol','spec','decision')",
           "('source','worker','client','protocol','spec','kernel')")
    change("HERE/'tiny_worker_v2.py'","HERE/'worker.py'")
    change("HERE/'tiny_client_v2.py'","HERE/'client.py'",2)
    change("HERE/'TINY_V2_PROTOCOL.md'","HERE/'PROTOCOL.md'")
    change("HERE/'TINY_V2_WATCH_SPEC.md'","HERE/'WATCH_SPEC.md'")
    change("HERE/'PROTOCOL.md':args.decision_sha256","HERE/'kernel.py':args.kernel_sha256")
    change("'--decision-sha256',args.decision_sha256","'--kernel-sha256',args.kernel_sha256")
    change("out,root = HERE/'tiny_qualification_watch_v2',HERE/'tiny_qualification_v2'",
           "out,root = HERE/'unit_watch_v1',HERE/'unit_v1'")
    change("'current_comp_shared_lazy_rotation_tiny_watch_v2'","'current_comp_root_batch_gram_watch_v1'")
    change("'current_comp_shared_lazy_rotation_tiny_v2'","'current_comp_root_batch_gram_unit_v1'",4)
    change("'current_comp_shared_lazy_rotation_tiny_v2_failure_ledger'",
           "'current_comp_root_batch_gram_unit_v1_failure_ledger'")
    change("        HERE/'watch_tiny.py':'67adbce35428959e007bf4372b8e6afc887313acca8a60a3d44fc7e39dc8523b',\n"
           "        HERE/'TINY_WATCH_SPEC.md':'61f82640a43f5b725b86c7e89bfbc8f41925cd9f5e10d8afb51d7fc9222c9515'})",
           "        ROOT_OWNER:ROOT_OWNER_SHA,ROOT_SPEC:ROOT_SPEC_SHA,\n"
           "        LAZY/'watch_tiny.py':'67adbce35428959e007bf4372b8e6afc887313acca8a60a3d44fc7e39dc8523b',\n"
           "        LAZY/'TINY_WATCH_SPEC.md':'61f82640a43f5b725b86c7e89bfbc8f41925cd9f5e10d8afb51d7fc9222c9515'})")
    change("prefix='astra_lazy_rotation_tiny_'","prefix='astra_root_batch_gram_'")
    change("r['repeat_claimed'] is False","r['repeat_claimed'] is True")
    change("r['dense_component_tariffs_exact'] and r['expected_rejections_passed']",
           "r['helper_tariffs_exact'] and r['byte_exact'] and r['helper_repeats_exact']")
    change("r['dense_branch_only']","r['fixed_D5_Gram_only']")
    change("        api.call_records(r,tick)",
           "        require(r['primitive_proof']==api.helper_source()[1],\n"
           "                'Actual unchanged original and batch scalar source proof')\n"
           "        tick()\n"
           "        api.call_records(r,tick)")
    change("            one_context_completed=True,repeat_claimed=False,dense_branch_only=True,\n"
           "            actual_server_OpenBLAS_qualified=True,runtime_gain_measured=False,",
           "            one_context_completed=True,repeat_claimed=True,fixed_D5_Gram_only=True,\n"
           "            actual_server_OpenBLAS_qualified=True,runtime_gain_measured=True,\n"
           "            component_gate_passed=result['comparison']['component_gate_passed'],\n"
           "            full_45s_gate_established=False,")
    # Before launching, use the exact worker admission as well as fixed imports.
    # It checks all six direct owners and the byte-inverse kernel proof without
    # constructing any array or invoking any old unit fixture.
    change("        require(not api.numerical_modules(),'Supervisor remains numerical-import free')",
           "        ns=type('UnitAuthority',(),dict(source_sha256=args.worker_sha256,\n"
           "            client_sha256=args.client_sha256,kernel_sha256=args.kernel_sha256,\n"
           "            protocol_sha256=args.protocol_sha256,watcher_sha256=args.source_sha256,\n"
           "            watch_spec_sha256=args.spec_sha256))()\n"
           "        admitted=api.admit(ns,tick)\n"
           "        for path,digest in admitted.items():\n"
           "            require(path not in result['source_bindings'] or result['source_bindings'][path]==digest,\n"
           "                    'Consistent exact six-owner kernel authority')\n"
           "            result['source_bindings'][path]=digest\n"
           "        require(not api.numerical_modules(),'Supervisor remains numerical-import free')")
    inverse=source
    for at,before,after in reversed(changes):
        if inverse[at:at+len(after)]!=after:
            raise RuntimeError('Exact complete root-batch watch positional inverse')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    if inverse!=original:
        raise RuntimeError('Entire qualified lazy-v2 resource loop restored')
    return source,dict(parent=parent_proof,owner_sha256=ROOT_OWNER_SHA,owner_spec_sha256=ROOT_SPEC_SHA,
        whole_effective_watch_inverse=True,derived_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        original_sampling_progress_cleanup_and_sealed_tail_unchanged=True,
        one_RPC_context_only=True,component_gate_does_not_control_operational_pass=True,
        numerical_execution=False)


def main():
    if '--run' not in sys.argv:
        print('Inert root-batch unit; --run, --numerical-slot-clear and six reviewed SHA values required.')
        return 0
    began=time.monotonic()
    v2=_owner()
    lazy=v2._owner()
    pair=lazy._parent()
    source,proof=derive_source()
    env=dict(pair._parent()._api,__file__=str(Path(__file__).resolve()),
        __name__='_root_batch_watch_effective',PARENT=pair.PARENT,PARENT_SHA=pair.PARENT_SHA,
        PARENT_SPEC=pair.PARENT_SPEC,PARENT_SPEC_SHA=pair.PARENT_SPEC_SHA,
        ADAPTER_OWNER=lazy.ADAPTER_OWNER,ADAPTER_OWNER_SHA=lazy.ADAPTER_OWNER_SHA,
        ADAPTER_SPEC=lazy.ADAPTER_SPEC,ADAPTER_SPEC_SHA=lazy.ADAPTER_SPEC_SHA,
        ROOT_OWNER=ROOT_OWNER,ROOT_OWNER_SHA=ROOT_OWNER_SHA,ROOT_SPEC=ROOT_SPEC,ROOT_SPEC_SHA=ROOT_SPEC_SHA,
        LAZY=LAZY,foreign_servers=lazy.foreign_servers,
        QUALIFICATION_BEGIN=began,SOURCE_ADAPTER_PROOF=proof)
    exec(compile(source,str(Path(__file__).resolve())+':whole-watch-inverse','exec'),env)
    return env['main']()


if __name__=='__main__':
    raise SystemExit(main())
