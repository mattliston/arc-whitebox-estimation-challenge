"""Fixed compact-scratch unit, reusing the complete qualified RPC lifecycle.

Only exact helper selection, binding tariff and prospective timing question
change. No numerical imports, arrays, old fixture or stage runs at import.
"""
import ast
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
V1=HERE.parent/'current_comp_root_batch_gram_v1'
V1_SOURCES={
    V1/'kernel.py':'8a32612c3df91e5c24d1baa4f9c2f33352049f07ac38a8c2be6699f1b4a1e73c',
    V1/'worker.py':'ac92c6f6a6de5aeca89d3e994509caffd8db392c105165edb41ffdd9efb78b38',
    V1/'client.py':'b1563e8eb542ac5e8a4dc958fb5515223749bbbbd49be48f5fe58c4cced7b82d',
    V1/'watch.py':'49efaa4392c26bade85b3fc050ef85832da5dadf1caefc3066c2e340b1e94bf9',
    V1/'PROTOCOL.md':'704db621eaa27697367318371c043fc641b9bd12b6c87d425c13b1fb19ce3d0f',
    V1/'WATCH_SPEC.md':'de62c9d98b2dfa03924c4281b91668acbdce7637cadff3c97e34a432026c1bac',
    HERE/'DESIGN.md':'fba209fe46e4ab0421a396f2befbe9cf0b54c1514ab42d488aad85de42977e8b'}

COMPARISON="""def compare(remote,tick=None):
    require(all(remote.get(k) is True for k in ('passed','completed','own_complete_bill',
        'one_context_completed','source_hashes_unchanged','helper_tariffs_exact','byte_exact',
        'helper_repeats_exact','fixed_D5_Gram_only')) and remote['retired_contexts']==1,
        'Fresh complete one-context worker authority')
    record=call_records(remote,tick)[0]
    rows=record['rows']
    warm=rows[3]['helper_wall_s']/rows[2]['helper_wall_s']
    first=rows[1]['helper_wall_s']/rows[0]['helper_wall_s']
    require(all(math.isfinite(r) and r>0 for r in (warm,first)),'Two actual fixed component ratios')
    return dict(passed=True,one_context_completed=True,all_independent_references_passed=True,
        helper_tariffs_exact=True,byte_exact=True,helper_repeats_exact=True,
        complete_flops=record['complete_flops'],warm_helper_flops=955654144,
        warm_wall_ratio=warm,first_wall_ratio=first,
        warm_gate_passed=warm<=.85,first_gate_passed=first<=.90,
        component_gate_passed=warm<=.85 and first<=.90,
        timing_condition='compact_warm/serial_warm <= 0.85 AND compact_first/serial_first <= 0.90',
        runtime_gain_measured=True,full_estimator_fidelity_established=False,
        full_45s_gate_established=False,raw_value_measured=False,model_calls=0,
        truth_accessed=False,automatic_next_stage=False)
"""


def derive_source():
    for path,digest in V1_SOURCES.items():
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Immutable unit/design source owner: '+str(path))
    original=(V1/'worker.py').read_text()
    old_compare=[n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name=='compare']
    if len(old_compare)!=1:
        raise RuntimeError('One original operational/scientific comparator')
    before_compare=ast.get_source_segment(original,old_compare[0])
    edits=(
        ("SCHEMA='current_comp_root_batch_gram_unit_v1'","SCHEMA='current_comp_compact_scratch_unit_v1'",1),
        ("ROSTER=('serial_first','batch_first','serial_warm','batch_warm')",
         "ROSTER=('serial_first','compact_first','serial_warm','compact_warm')",1),
        ("689569792","245120512",2),
        ("_KERNEL.batched_gram(fnp,ns,left,right)","_KERNEL.compact_gram(fnp,ns,left,right)",1),
        ("Actual serial/batch byte equality and within-helper repeat equality",
         "Actual capacity/compact byte equality and within-helper repeat equality",1),
        (before_compare,COMPARISON.rstrip(),1))
    source,positions=original,[]
    for before,after,count in edits:
        if source.count(before)!=count:
            raise RuntimeError('Exact compact-storage worker splice count: '+before[:80])
        for unused in range(count):
            at=source.index(before);positions.append((at,before,after))
            source=source[:at]+after+source[at+len(before):]
    inverse=source
    for at,before,after in reversed(positions):
        if inverse[at:at+len(after)]!=after:
            raise RuntimeError('Exact positional compact worker inverse')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    if inverse!=original:
        raise RuntimeError('Whole frozen RPC lifecycle/fixture/reference restored')
    return source,dict(owner_sha256=V1_SOURCES[V1/'worker.py'],whole_V1_source_inverse=True,
        exact_fixture_reference_lifecycle_retained=True,no_root_batching=True,
        combined_binding_flops=245120512,prospective_warm_ratio_limit=.85,
        prospective_first_ratio_limit=.90,derived_sha256=hashlib.sha256(source.encode()).hexdigest())


_source,COMPACT_SOURCE_PROOF=derive_source()
_api=dict(__file__=str(Path(__file__).resolve()),__name__='_compact_scratch_worker_inert')
exec(compile(_source,str(Path(__file__).resolve())+':whole-qualified-worker-inverse','exec'),_api)
_api['PINS'].update(V1_SOURCES)
for _name in ('main','admit','owner_map','helper_source','primitive_namespace','derive_main','PINS',
              'numerical_modules','sha','read','write','verify_call','call_records','compare',
              'SCHEMA','CALL_BUDGET','ROSTER'):
    globals()[_name]=_api[_name]
del _source


if __name__=='__main__':
    raise SystemExit(main())

