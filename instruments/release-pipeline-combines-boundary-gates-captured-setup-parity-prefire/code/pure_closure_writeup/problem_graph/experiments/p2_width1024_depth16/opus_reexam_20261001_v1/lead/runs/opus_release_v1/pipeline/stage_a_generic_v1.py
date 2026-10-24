"""TAG opus-release STAGE A (generic v1: proven = fired 2b108 #336342, 100/100; no candidate-specific review block): contract and packaging check of a candidate. Builds the archive in a scratch directory and STOPS
THERE: nothing is uploaded and no credential is read (the upload step, `whest submit`, is the lead's and is not in this file).

Checks: (1) `whest validate` (estimator contract on a tiny network); (2) `whest package` of a copy named estimator.py;
(3) `whest validate-package`; (4) the archive holds exactly estimator.py + manifest.json and estimator.py is byte-identical to the
candidate; (5) static source checks: no numpy import (the platform client is numpy-free), and every einsum signature already
occurs in the platform-proven bank. Why: flopscope infers a symmetric output from a REPEATED operand, computes it pairwise in
float32 and validates the symmetry to 1e-6; a network on which that fails raises SymmetryError, and one raise is one failed
network = a ruined score (crash law of 2026-09-23). So (a) an einsum call that passes the same expression twice and is not in
the proven source in exactly that form fails the stage outright (`einsum_no_new_repeated_operand`); (b) any other new signature
is listed with its operands and line and fails the stage until someone has read it and passes it in --reviewed;
(6) the source compiles under Python 3.10 (the platform runs 3.10.21, this machine's tools 3.11) and names nothing that only
exists from 3.11 on, when a 3.10 interpreter is found under ~/.local/share/uv/python (otherwise noted as not checked).
usage: stage_a_package.py <candidate.py> <tag> [--proven PATH] [--reviewed "sig1;sig2;..."]
       -> O/release/runs/<tag>/{payload/estimator.py,submission.tar.gz,stage_a.json}"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

sys.path.insert(0, '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release')
from common import BANK, ROOT, WHEST, run_dir, sha, verdict, write_json


def whest(args, out, name):
    result = subprocess.run([str(WHEST), *map(str, args)], cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                            capture_output=True, text=True, timeout=300)
    write_json(out/name, dict(command=[str(WHEST), *map(str, args)], returncode=result.returncode, stdout=result.stdout,
                              stderr=result.stderr))
    return result


def static(path):
    """(numpy import lines, einsum signature counts, einsum calls as (signature, operand texts, line)) of a source file."""
    tree = ast.parse(Path(path).read_text())
    numpy_imports, einsums, calls = [], {}, []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name.split('.')[0] == 'numpy' for a in node.names):
            numpy_imports.append(node.lineno)
        if isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[0] == 'numpy':
            numpy_imports.append(node.lineno)
        if isinstance(node, ast.Call) and getattr(node.func, 'attr', getattr(node.func, 'id', None)) == 'einsum':
            first = node.args[0] if node.args else None
            key = first.value if isinstance(first, ast.Constant) else '<non-literal>'
            einsums[key] = einsums.get(key, 0) + 1
            calls.append((key, tuple(ast.unparse(x) for x in node.args[1:]), node.lineno))
    return numpy_imports, einsums, calls


PY310_CHECK = """
import ast, sys
src = open(sys.argv[1]).read()
compile(src, sys.argv[1], 'exec')
late = {'cbrt', 'exp2', 'add_note', 'StrEnum', 'TaskGroup', 'ExceptionGroup', 'BaseExceptionGroup', 'tomllib', 'LiteralString',
        'NotRequired', 'Required', 'assert_never', 'assert_type', 'reveal_type', 'UTC'}
hits = sorted({(n.attr if isinstance(n, ast.Attribute) else n.id, n.lineno) for n in ast.walk(ast.parse(src))
               if (isinstance(n, ast.Attribute) and n.attr in late) or (isinstance(n, ast.Name) and n.id in late)})
print(sys.version.split()[0], hits)
sys.exit(1 if hits else 0)
"""


def python310(candidate):
    """(ok or None when no interpreter was found, note): the candidate compiled under a local Python 3.10."""
    found = sorted(Path.home().glob('.local/share/uv/python/cpython-3.10.*/bin/python3.10'))
    if not found:
        return None, 'Python 3.10 compile not checked (no 3.10 interpreter found)'
    result = subprocess.run([str(found[-1]), '-B', '-c', PY310_CHECK, str(candidate)], capture_output=True, text=True, timeout=120)
    reply = (result.stdout.strip() or result.stderr.strip().splitlines()[-1] if (result.stdout.strip() or result.stderr.strip()) else '?')
    return result.returncode == 0, f"under Python 3.10: {'compiles, no 3.11-only name' if result.returncode == 0 else 'FAILS'} ({reply})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('candidate', type=Path)
    ap.add_argument('tag')
    ap.add_argument('--proven', type=Path, default=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py'), help='last platform-proven source (einsum signatures are compared to it)')
    ap.add_argument('--reviewed', default='', help='new einsum signatures someone has read and found free of repeated operands '
                    'and symmetric-tensor inputs, separated by ";"')
    a = ap.parse_args()
    candidate = a.candidate.resolve()
    out = run_dir(a.tag)
    digest = sha(candidate)
    gates, notes = {}, []

    contract = whest(['validate', '--estimator', candidate, '--class', 'Estimator', '--json'], out, 'validate.json')
    try:
        reply = json.loads(contract.stdout[contract.stdout.index('{'):])
        gates['contract'] = contract.returncode == 0 and reply.get('ok') is True and reply.get('class_name') == 'Estimator'
        notes.append(f"contract output shape {reply.get('output_shape')}")
    except ValueError:
        gates['contract'] = False

    payload = out/'payload'
    payload.mkdir()
    estimator = payload/'estimator.py'
    estimator.write_bytes(candidate.read_bytes())
    archive = out/'submission.tar.gz'
    packed = whest(['package', '--estimator', estimator, '--class', 'Estimator', '--output', archive, '--yes', '--json'],
                   out, 'package.json')
    gates['package'] = packed.returncode == 0 and archive.exists()
    checked = whest(['validate-package', archive, '--json'], out, 'validate_package.json') if archive.exists() else None
    gates['validate_package'] = bool(checked) and checked.returncode == 0

    gates['archive_contents'] = False
    if archive.exists():
        with tarfile.open(archive, 'r:gz') as tar:
            members = {m.name: m for m in tar.getmembers()}
            if sorted(members) == ['estimator.py', 'manifest.json'] and all(m.isfile() for m in members.values()):
                body = tar.extractfile(members['estimator.py']).read()
                manifest = json.loads(tar.extractfile(members['manifest.json']).read())
                gates['archive_contents'] = (hashlib.sha256(body).hexdigest() == digest
                                             and manifest.get('entrypoint') == {'module': 'estimator', 'class': 'Estimator'})
        notes.append(f"archive {archive.stat().st_size} bytes sha {sha(archive)[:12]}")

    numpy_imports, einsums, calls = static(candidate)
    _, proven, proven_calls = static(a.proven.resolve())
    reviewed = {x for x in a.reviewed.split(';') if x}
    new_einsums = sorted(k for k in einsums if k not in proven)
    unread = [k for k in new_einsums if k not in reviewed]
    known = {(k, ops) for k, ops, _ in proven_calls}
    repeated = [(k, ops, line) for k, ops, line in calls if len(set(ops)) < len(ops) and (k, ops) not in known]
    reviewed_repeated = []
    gates['no_numpy_import'] = not numpy_imports
    gates['einsum_signatures_proven'] = not unread
    gates['einsum_no_new_repeated_operand'] = not repeated
    if numpy_imports:
        notes.append(f"numpy imported at lines {numpy_imports}")
    where = '; '.join(f"{k!r} at line {line} on ({', '.join(ops)})" for k, ops, line in calls if k in new_einsums)
    notes.append(f"einsum signatures {len(einsums)}, new vs proven source: "
                 + (f"{len(new_einsums)} [{where}], read and passed in --reviewed: {sorted(reviewed & set(new_einsums)) or 'none'}"
                    if new_einsums else 'none'))
    if repeated:
        notes.append('NEW REPEATED-OPERAND einsum (inferred-symmetric output, can raise SymmetryError on the platform): '
                     + '; '.join(f"{k!r} at line {line} on ({', '.join(ops)})" for k, ops, line in repeated))

    ok310, note310 = python310(candidate)
    if ok310 is not None:
        gates['python310'] = ok310
    notes.append(note310)

    passed = all(gates.values())
    text = (f"source {candidate.stat().st_size} bytes sha {digest[:12]}; " + '; '.join(notes)
            + '; failed: ' + (', '.join(k for k, v in gates.items() if not v) or 'none') + '; NOT uploaded')
    result = dict(stage='A', candidate=str(candidate), candidate_sha256=digest, archive=str(archive), gates=gates, passed=passed,
                  einsum_signatures=einsums, new_einsum_signatures=new_einsums, reviewed_einsum_signatures=sorted(reviewed),
                  new_repeated_operand_einsums=[list(x) for x in repeated], uploaded=False)
    result['root_reviewed_repeated_operand_einsums'] = [list(x) for x in reviewed_repeated]
    result['line'] = verdict('A package', passed, text)
    write_json(out/'stage_a.json', result)
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
