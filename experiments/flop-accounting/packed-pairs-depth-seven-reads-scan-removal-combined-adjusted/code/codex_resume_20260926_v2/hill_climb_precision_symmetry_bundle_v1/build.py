"""Compose a fixed source version, using stdlib source/AST operations only."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys

D = Path(__file__).resolve().parent
PARENT = D.parent/'hill_climb_exact_bundle_v1/candidate.py'
PARENT_SHA = '3651f482c3b4369228aab6741d9f40c3011b3ee155db78a51d247bfb6f215ba7'


def pin(path):
    data = path.read_bytes()
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def main():
    assert pin(PARENT)['sha256'] == PARENT_SHA
    for name in ('candidate.py', 'build.json', 'edits.json'):
        assert not (D/name).exists(), name
    inputs = [PARENT] + [D/name for name in ('precision.py', 'precision_notes.md', 'symmetry.py', 'pricing.json', 'precision_inventory.json', 'PROTOCOL.md', 'symmetry_probe.py', 'make_assessor.py', 'build.py')]
    assert all(path.is_file() for path in inputs)
    spec = importlib.util.spec_from_file_location('fixed_precision_transform', D/'precision.py')
    transform = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = transform
    spec.loader.exec_module(transform)
    parent = PARENT.read_text()
    precision, edits = transform.transform(parent)
    assert transform.inverse(precision, edits) == parent
    for edit in edits:
        assert set(('label', 'before', 'after')) <= set(edit)
    symmetry = (D/'symmetry.py').read_text()
    suffix = '\n\n' + symmetry
    candidate = precision + suffix
    ast.parse(candidate)
    assert candidate.endswith(suffix)
    assert transform.inverse(candidate[:-len(suffix)], edits) == parent
    with (D/'candidate.py').open('x') as stream:
        stream.write(candidate)
    with (D/'edits.json').open('x') as stream:
        json.dump(edits, stream, indent=2)
        stream.write('\n')
    files = inputs + [D/'edits.json', D/'candidate.py']
    result = dict(parent=pin(PARENT), candidate=pin(D/'candidate.py'), files=[pin(path) for path in files],
                  precision_edits=len(edits), precision_source_sha256=hashlib.sha256(precision.encode()).hexdigest(),
                  symmetry_suffix_bytes=len(suffix.encode()), inverse_exact=True,
                  source_only=True, candidate_imported=False, model_executed=False, serving_qualified=False, submitted=False)
    with (D/'build.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(candidate=result['candidate'], inverse_exact=True, precision_edits=len(edits))))


if __name__ == '__main__':
    main()
