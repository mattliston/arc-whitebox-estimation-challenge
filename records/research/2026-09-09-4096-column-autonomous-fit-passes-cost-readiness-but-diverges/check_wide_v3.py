"""Additive physical-identity comparator after the preserved target-free v2 stop.

No estimator or numerical formula changes. Versioned output directories and a
completed literal identity certificate prevent silent reuse/adjudication of v2.
Inert without --stage. See IDENTITY_COMPARISON_V3.md for the cutoff-tie caveat.
"""
import ast
from pathlib import Path
from common import HERE, completed, module, sha

V2 = HERE/'check_wide_v2.py'
V2_SHA = 'a2abd2dc2bdb6310f06a5669e370cec02866b2176484aa857e5674c0f2929f53'


def identity_comparison(diagnostic, native_diagnostic, cap=4096):
    """Track each backend's local slots separately, compare physical identities."""
    # The only production call omits cap. The literal width8 certificate uses4.
    assert isinstance(cap, int) and cap > 0
    assert len(native_diagnostic['selections']) == 47
    nlist, tlist, offset, retirement_offset = [], [], 0, 0
    records, mismatches = [], []
    for layer in range(16):
        item = dict(layer=layer, birth_local_permutation=[], retained_local_permutation=[])
        for slot in range(2 if layer == 15 else 3):
            observed = diagnostic['selections'][str((layer, slot))]
            reference = native_diagnostic['selections'][offset]
            offset += 1
            assert len(set(observed)) == len(observed) and len(set(reference)) == len(reference)
            if sorted(observed) != sorted(reference):
                mismatches.append(dict(layer=layer, type=('alive', 'response', 'birth')[slot],
                    missing=sorted(set(reference)-set(observed)), extra=sorted(set(observed)-set(reference))))
            if slot == 2:
                nmap = {unit: index for index, unit in enumerate(reference)}
                item['birth_local_permutation'] = [[index, nmap.get(unit)]
                    for index, unit in enumerate(observed) if nmap.get(unit) != index]
                nlist += [(layer, unit) for unit in reference]
                tlist += [(layer, unit) for unit in observed]
        if layer < 15:
            tr = diagnostic['retirement'][str(layer)]
            assert tr['pairs_before'] == len(tlist)
            if len(nlist) > cap:
                nr = native_diagnostic['retirement'][retirement_offset]
                retirement_offset += 1
                assert nr['pairs_before'] == len(nlist)
                nids = nr['chosen']
                assert nids == sorted(set(nids)) and len(nids) == cap
                assert all(0 <= i < len(nlist) for i in nids)
                nlist = [nlist[i] for i in nids]
            tids = tr['chosen']
            assert tids == sorted(set(tids)) and len(tids) == min(cap, len(tlist))
            assert all(0 <= i < len(tlist) for i in tids)
            assert tr['retired'] == (len(tlist) > cap)
            tlist = [tlist[i] for i in tids]
            assert len(set(nlist)) == len(nlist) and len(set(tlist)) == len(tlist)
            missing, extra = sorted(set(nlist)-set(tlist)), sorted(set(tlist)-set(nlist))
            if missing or extra:
                mismatches.append(dict(layer=layer, type='retained_pair_identity', missing=missing, extra=extra))
            nmap = {identity: index for index, identity in enumerate(nlist)}
            item['retained_local_permutation'] = [[index, nmap.get(identity)]
                for index, identity in enumerate(tlist) if nmap.get(identity) != index]
            item['retained_identity_count'] = len(tlist)
            item['exact_retained_identity_set'] = not missing and not extra
        records.append(item)
    assert retirement_offset == len(native_diagnostic['retirement'])
    return dict(passed=not mismatches, physical_identity='(birth_layer, birth_unit)',
        layers=records, mismatches=mismatches, local_slot_permutations_are_diagnostic_only=True,
        physical_cutoff_tie_differences_are_not_waived=True)


def identity_admission(digest, pins):
    certificate = completed(HERE/'identity_v3', digest)
    assert certificate['schema'] == 'cap4096_identity_certificate_v3'
    assert certificate['checks']['cutoff_tie_changes_physical_identities']
    assert certificate['checks']['original_stable_tie_completion_preserved']
    for name in ('model.py', 'identity_certificate_v3.py', 'IDENTITY_COMPARISON_V3.md'):
        assert certificate['source_bindings'][str(HERE/name)] == sha(HERE/name)
    pins[str(V2)] = V2_SHA
    for path in (HERE/'identity_certificate_v3.py', HERE/'IDENTITY_COMPARISON_V3.md',
                 HERE/'identity_v3/result.json', HERE/'identity_v3/completion.json'):
        pins[str(path)] = sha(path)


def derived_source():
    assert sha(V2) == V2_SHA
    predecessor = module(V2, 'cap4096_preserved_wide_v2')
    original = predecessor.derived_source()
    fn = next(n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef) and n.name == 'same_cells')
    old_function = ''.join(original.splitlines(keepends=True)[fn.lineno-1:fn.end_lineno])
    new_function = ('def same_cells(diagnostic, native_diagnostic):\n'
                    '    return identity_comparison(diagnostic, native_diagnostic)\n')
    edits = [
        (old_function, new_function),
        ("    parser.add_argument('--native-sha256')", "    parser.add_argument('--native-sha256')\n    parser.add_argument('--identity-sha256')"),
        ("    native = native_admission(args.native_sha256)",
         "    identity_admission(args.identity_sha256,pins)\n    native = native_admission(args.native_sha256)"),
        ("        same_cells(forward_diag, native_diag)",
         "        entry['identity_comparison']=same_cells(forward_diag, native_diag)\n"
         "        identity_path=out/f'{rank}_identity.json'\n"
         "        write(identity_path,entry['identity_comparison'])\n"
         "        result['artifacts'][str(identity_path)]=sha(identity_path)\n"
         "        assert entry['identity_comparison']['passed'], 'Native/Torch physical pair identities differ'"),
        ("completed(HERE/'cpu_v2', args.cpu_sha256)", "completed(HERE/'cpu_v3', args.cpu_sha256)"),
        ("completed(HERE/'mps_v2', args.mps_sha256)", "completed(HERE/'mps_v3', args.mps_sha256)"),
        ("'model.py', 'check_wide_v2.py', 'PROTOCOL.md'", "'model.py', 'check_wide_v3.py', 'PROTOCOL.md'"),
        ("HERE/'cpu_v2/result.json'", "HERE/'cpu_v3/result.json'"),
        ("HERE/'mps_v2/result.json'", "HERE/'mps_v3/result.json'"),
        ("out = HERE/('parity_v2' if args.stage == 'judge' else args.stage+'_v2')",
         "out = HERE/('parity_v3' if args.stage == 'judge' else args.stage+'_v3')"),
        ("schema='current_paircap4096_e2e_wide_v2'", "schema='current_paircap4096_e2e_wide_v3'"),
    ]
    source = original
    for old, new in edits:
        assert source.count(old) == 1, old[:100]
        source = source.replace(old, new)
    inverse = source
    for old, new in reversed(edits):
        assert inverse.count(new) == 1, new[:100]
        inverse = inverse.replace(new, old)
    assert inverse == original
    return source, predecessor


def main():
    source, predecessor = derived_source()
    ast.parse(source)
    env = dict(__name__='cap4096_wide_v3_identity', __file__=str(Path(__file__).resolve()),
        PREDECESSOR=predecessor.PREDECESSOR, PREDECESSOR_SHA=predecessor.PREDECESSOR_SHA,
        identity_comparison=identity_comparison, identity_admission=identity_admission)
    exec(compile(source, str(__file__)+':preserved_wide', 'exec'), env)
    return env['main']()


if __name__ == '__main__':
    raise SystemExit(main())
