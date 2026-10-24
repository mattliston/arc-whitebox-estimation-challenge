"""Additive source-inventory repair; the cap model and numerical gates are unchanged.

The preserved v1 failed before constructing a card because a synthetic module
reported __file__='_ops.py'. Bind only actual existing Python files under NEW;
retain skipped nonexistent filenames as diagnostics. New output directories
prevent mixing incomplete v1 acquisitions with this repaired harness.
Inert without --stage; this module executes no numerical work at import.
"""
from pathlib import Path
from common import HERE, sha

PREDECESSOR = HERE/'check_wide.py'
PREDECESSOR_SHA = '8e01dd4b5547662e2f71babebac2c8f0fec66a96f748e2f9524d09a8a95ef429'


def derived_source():
    if sha(PREDECESSOR) != PREDECESSOR_SHA:
        raise RuntimeError('Frozen wide-check predecessor changed')
    original = PREDECESSOR.read_text()
    edits = [
        ("        if path and Path(path).suffix == '.py' and Path(path).resolve().is_relative_to(NEW):\n"
         "            result['source_bindings'][str(Path(path).resolve())] = sha(path)",
         "        if path and Path(path).suffix == '.py':\n"
         "            resolved = Path(path).resolve()\n"
         "            if resolved.is_file() and resolved.is_relative_to(NEW):\n"
         "                result['source_bindings'][str(resolved)] = sha(resolved)\n"
         "            elif not resolved.is_file():\n"
         "                result.setdefault('skipped_nonexistent_module_filenames', []).append(str(path))"),
        ("completed(HERE/'cpu', args.cpu_sha256)", "completed(HERE/'cpu_v2', args.cpu_sha256)"),
        ("completed(HERE/'mps', args.mps_sha256)", "completed(HERE/'mps_v2', args.mps_sha256)"),
        ("'model.py', 'check_wide.py', 'PROTOCOL.md'", "'model.py', 'check_wide_v2.py', 'PROTOCOL.md'"),
        ("HERE/'cpu/result.json'", "HERE/'cpu_v2/result.json'"),
        ("HERE/'mps/result.json'", "HERE/'mps_v2/result.json'"),
        ("out = HERE/('parity' if args.stage == 'judge' else args.stage)",
         "out = HERE/('parity_v2' if args.stage == 'judge' else args.stage+'_v2')"),
        ("schema='current_paircap4096_e2e_wide_v1'", "schema='current_paircap4096_e2e_wide_v2'"),
        ("    native = native_admission(args.native_sha256)",
         "    pins[str(PREDECESSOR)] = PREDECESSOR_SHA\n"
         "    native = native_admission(args.native_sha256)"),
    ]
    source = original
    for old, new in edits:
        if source.count(old) != 1:
            raise RuntimeError('Unexpected predecessor edit count: '+old[:100])
        source = source.replace(old, new)
    inverse = source
    for old, new in reversed(edits):
        if inverse.count(new) != 1:
            raise RuntimeError('Ambiguous source-repair inverse')
        inverse = inverse.replace(new, old)
    if inverse != original:
        raise RuntimeError('Source repair escaped the declared edits')
    return source


def main():
    import ast
    source = derived_source()
    ast.parse(source)
    env = dict(__name__='cap4096_wide_v2_repaired', __file__=str(Path(__file__).resolve()),
               PREDECESSOR=PREDECESSOR, PREDECESSOR_SHA=PREDECESSOR_SHA)
    exec(compile(source, str(__file__)+':preserved_wide', 'exec'), env)
    return env['main']()


if __name__ == '__main__':
    raise SystemExit(main())
