"""Inert exact qualified single-IPC supervisor for the fixed noise diagnostic.

The whole120s/8GB/zero-swap/1s observation,60s RPC/residual.4, actual retirement,
foreign-server rejection, complete tree cleanup and bounded sealed tail remain.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
RB1=HERE.parent/'current_comp_root_batch_gram_v1'
OWNER=RB1/'watch.py'
OWNER_SHA='49efaa4392c26bade85b3fc050ef85832da5dadf1caefc3066c2e340b1e94bf9'
OWNER_SPEC=RB1/'WATCH_SPEC.md'
OWNER_SPEC_SHA='de62c9d98b2dfa03924c4281b91668acbdce7637cadff3c97e34a432026c1bac'


def _owner():
    for path,digest in ((OWNER,OWNER_SHA),(OWNER_SPEC,OWNER_SPEC_SHA)):
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Exact qualified whole-RPC supervisor owner')
    spec=importlib.util.spec_from_file_location('_noise_frozen_RPC_watch',OWNER)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def derive_source():
    base=_owner()
    original,parent=base.derive_source();source=original;changes=[]
    def change(before,after,count=1):
        nonlocal source
        if source.count(before)!=count:
            raise RuntimeError('Exact noise supervisor splice count: '+before[:110])
        for unused in range(count):
            at=source.index(before);changes.append((at,before,after))
            source=source[:at]+after+source[at+len(before):]
    change("out,root = HERE/'unit_watch_v1',HERE/'unit_v1'",
        "out,root = HERE/'noise_watch_v1',HERE/'noise_v1'")
    change("'current_comp_root_batch_gram_watch_v1'","'current_comp_cubic_innovation_noise_watch_v1'")
    change("'current_comp_root_batch_gram_unit_v1'","'current_comp_cubic_innovation_noise_v1'",4)
    change("'current_comp_root_batch_gram_unit_v1_failure_ledger'",
        "'current_comp_cubic_innovation_noise_v1_failure_ledger'")
    change("prefix='astra_root_batch_gram_'","prefix='astra_cubic_noise_'")
    change("r['repeat_claimed'] is True","r['repeat_claimed'] is False")
    change("r['all_independent_references_passed']","r['actual_noise_circuit_qualified']")
    change("r['helper_tariffs_exact'] and r['byte_exact'] and r['helper_repeats_exact']",
        "r['complete_sections_reconciled'] and r['input_hashes_unchanged']")
    change("r['fixed_D5_Gram_only']","r['fixed_noise_only']")
    change("        admitted=api.admit(ns,tick)",
        "        context=api.admission(ns,tick)\n"
        "        admitted=context['source_bindings']\n"
        "        result['input_bindings'].update(context['input_bindings'])")
    change("        for path,digest in r['artifacts'].items():",
        "        require(r['input_bindings']==context['input_bindings'],\n"
        "                'Same fixed positive capture/input union in actual client')\n"
        "        for path,digest in r['input_bindings'].items():\n"
        "            require(sha(Path(path),tick)==digest,'Fresh full producer/input union')\n"
        "            result['input_bindings'][path]=digest\n"
        "        for path,digest in r['artifacts'].items():")
    change("            one_context_completed=True,repeat_claimed=True,fixed_D5_Gram_only=True,\n"
        "            actual_server_OpenBLAS_qualified=True,runtime_gain_measured=True,\n"
        "            component_gate_passed=result['comparison']['component_gate_passed'],\n"
        "            full_45s_gate_established=False,",
        "            one_context_completed=True,repeat_claimed=False,fixed_noise_only=True,\n"
        "            actual_server_OpenBLAS_qualified=True,runtime_gain_measured=False,\n"
        "            host_noise_precision_passed=result['comparison']['host_noise_precision_passed'],\n"
        "            outer_variance_measured=False,signed_innovation_measured=False,")
    inverse=source
    for at,before,after in reversed(changes):
        if inverse[at:at+len(after)]!=after:
            raise RuntimeError('Exact complete noise-watch positional inverse')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    if inverse!=original:
        raise RuntimeError('Entire original qualified whole-RPC loop restored')
    return source,dict(parent=parent,owner_sha256=OWNER_SHA,owner_spec_sha256=OWNER_SPEC_SHA,
        whole_effective_watch_inverse=True,derived_sha256=hashlib.sha256(source.encode()).hexdigest(),
        original_sampling_progress_cleanup_and_sealed_tail_unchanged=True,
        actual_positive_capture_admitted_before_spawn=True,
        precision_gate_does_not_control_operational_pass=True,numerical_execution=False)


def main():
    if '--run' not in sys.argv:
        print('Inert fixed32x2 noise diagnostic; explicit reviewed watch launch required.')
        return 0
    began=time.monotonic()
    base=_owner();v2=base._owner();lazy=v2._owner();pair=lazy._parent()
    source,proof=derive_source()
    env=dict(pair._parent()._api,__file__=str(Path(__file__).resolve()),
        __name__='_noise_watch_effective',PARENT=pair.PARENT,PARENT_SHA=pair.PARENT_SHA,
        PARENT_SPEC=pair.PARENT_SPEC,PARENT_SPEC_SHA=pair.PARENT_SPEC_SHA,
        ADAPTER_OWNER=lazy.ADAPTER_OWNER,ADAPTER_OWNER_SHA=lazy.ADAPTER_OWNER_SHA,
        ADAPTER_SPEC=lazy.ADAPTER_SPEC,ADAPTER_SPEC_SHA=lazy.ADAPTER_SPEC_SHA,
        ROOT_OWNER=base.ROOT_OWNER,ROOT_OWNER_SHA=base.ROOT_OWNER_SHA,
        ROOT_SPEC=base.ROOT_SPEC,ROOT_SPEC_SHA=base.ROOT_SPEC_SHA,LAZY=base.LAZY,
        foreign_servers=lazy.foreign_servers,QUALIFICATION_BEGIN=began,SOURCE_ADAPTER_PROOF=proof)
    exec(compile(source,str(Path(__file__).resolve())+':whole-watch-inverse','exec'),env)
    return env['main']()


if __name__=='__main__':
    raise SystemExit(main())
