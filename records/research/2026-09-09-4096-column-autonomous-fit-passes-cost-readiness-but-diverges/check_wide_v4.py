"""Additive explicit-host-transfer model binding; all wide scientific gates retained."""
import ast
from pathlib import Path
from common import HERE, completed, module, sha

V3 = HERE/'check_wide_v3.py'
V3_SHA = '3956f82bf4f62b55b9e142af4fded6792ba042b727fd78e7868b044926f5c3e8'


def transfer_admission(digest, identity_digest, pins):
    receipt = completed(HERE/'transfer_v4', digest)
    assert receipt['schema'] == 'cap4096_transfer_certificate_v4'
    assert receipt['identity_result_sha256'] == identity_digest
    assert receipt['checks']['detached_ranking_and_differentiable_diagonal_exercised']
    for name in ('model.py', 'model_v2.py', 'transfer_certificate_v4.py', 'TRANSFER_REPAIR_V4.md'):
        assert receipt['source_bindings'][str(HERE/name)] == sha(HERE/name)
        pins[str(HERE/name)] = sha(HERE/name)
    pins[str(V3)] = V3_SHA
    for path in (HERE/'transfer_v4/result.json', HERE/'transfer_v4/completion.json'):
        pins[str(path)] = sha(path)


def derived_source():
    assert sha(V3) == V3_SHA
    previous = module(V3, 'cap4096_preserved_wide_v3')
    original, previous_v2 = previous.derived_source()
    edits = [
        ("    parser.add_argument('--identity-sha256')", "    parser.add_argument('--identity-sha256')\n    parser.add_argument('--transfer-sha256')"),
        ("    identity_admission(args.identity_sha256,pins)",
         "    identity_admission(args.identity_sha256,pins)\n"
         "    transfer_admission(args.transfer_sha256,args.identity_sha256,pins)"),
        ("            import model\n", "            import model_v2 as model\n"),
        ("completed(HERE/'cpu_v3', args.cpu_sha256)", "completed(HERE/'cpu_v4', args.cpu_sha256)"),
        ("completed(HERE/'mps_v3', args.mps_sha256)", "completed(HERE/'mps_v4', args.mps_sha256)"),
        ("'model.py', 'check_wide_v3.py', 'PROTOCOL.md'", "'model.py', 'model_v2.py', 'check_wide_v4.py', 'PROTOCOL.md'"),
        ("HERE/'cpu_v3/result.json'", "HERE/'cpu_v4/result.json'"),
        ("HERE/'mps_v3/result.json'", "HERE/'mps_v4/result.json'"),
        ("out = HERE/('parity_v3' if args.stage == 'judge' else args.stage+'_v3')",
         "out = HERE/('parity_v4' if args.stage == 'judge' else args.stage+'_v4')"),
        ("schema='current_paircap4096_e2e_wide_v3'", "schema='current_paircap4096_e2e_wide_v4'"),
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
    return source, previous, previous_v2


def main():
    source, previous, previous_v2 = derived_source()
    ast.parse(source)
    env = dict(__name__='cap4096_wide_v4_host_transfer', __file__=str(Path(__file__).resolve()),
        PREDECESSOR=previous_v2.PREDECESSOR, PREDECESSOR_SHA=previous_v2.PREDECESSOR_SHA,
        identity_comparison=previous.identity_comparison,
        identity_admission=previous.identity_admission, transfer_admission=transfer_admission)
    exec(compile(source, str(__file__)+':preserved_wide', 'exec'), env)
    return env['main']()


if __name__ == '__main__':
    raise SystemExit(main())
