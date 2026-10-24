"""Small standard-library provenance helpers; no numerical imports or I/O at import."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
P2 = NEW.parent
OLD = NEW / 'current_paircap4096_v1'
BD = NEW / 'birth_direction_fit_v1'
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
PARENT = NEW / 'unused_materializations_v1/estimator.py'
PARENT_SHA = '409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f'
PINS = {
    PARENT: PARENT_SHA,
    OLD/'core.py': 'b4932aed4b3070b39031bc93a688c0c9486821415d299ae083f4b4081b93389e',
    OLD/'build.py': '60da4a0c7fd973035cddbeba5442ea1707115aa05b1c36b4c364cc6c40e8815a',
    OLD/'PROTOCOL.md': '79d5cf5b98aaf3ed412d19602ea4c62f0cea0115483e66aef9a7e95331ec85e8',
    BD/'model.py': 'cea1ea473db821cea6b4b9d34aeabf06f43ded8c5d77ee4037dd3f3d269df613',
    BD/'native_v2.py': '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b',
    NEW/'centered_bank_autodiff_v1.py': 'c70e7bc5de06e59d1c95a41658f406a4849820c28ecd8e5aa4edd48ea67ca3e2',
    NEW/'centered_bank_autodiff_v3.py': '45c88f1e73d694f0781c4f0a01053511addbe465a6746ef11ac38378b093983c',
    NEW/'full_carrier_autodiff_v1.py': 'd6480f443e807688dca380d19d34e7cfe5c1f7e625104aae87201cfba7d199f7',
    STORE/'manifest.json': '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def bindings(extra=()):
    result = {str(path): sha(path) for path in PINS}
    for path, expected in PINS.items():
        if result[str(path)] != expected:
            raise RuntimeError('Frozen source mismatch: '+str(path))
    for path in extra:
        result[str(Path(path).resolve())] = sha(path)
    return result


def unchanged(record):
    if not all(sha(path) == digest for path, digest in record.items()):
        raise RuntimeError('An authenticated input changed')


def completed(folder, digest):
    folder = Path(folder)
    if sha(folder/'result.json') != digest:
        raise RuntimeError('Result pin mismatch')
    result = json.loads((folder/'result.json').read_text())
    seal = json.loads((folder/'completion.json').read_text())
    if not result['passed'] or not seal['passed'] or seal['result_sha256'] != digest:
        raise RuntimeError('A completed PASS prerequisite is required')
    for path, expected in result.get('artifacts', {}).items():
        if sha(path) != expected:
            raise RuntimeError('Artifact mismatch: '+path)
    unchanged(result.get('source_bindings', result.get('sources', {})))
    return result


def weight_bindings(ranks=(35, 100)):
    items = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
    result, rows = {}, {}
    for rank in ranks:
        matching = [item for item in items if item['role_rank'] == rank]
        if len(matching) != 1:
            raise RuntimeError('Canonical row is not unique')
        item = matching[0]
        path = STORE/item['path']
        if sha(path) != item['sha256']:
            raise RuntimeError('Canonical row manifest mismatch')
        row = json.loads(path.read_text())
        weights = path.parent/'weights.npy'
        if sha(weights) != row['files']['weights.npy']['sha256']:
            raise RuntimeError('Canonical weight mismatch')
        result[str(path)] = sha(path)
        result[str(weights)] = sha(weights)
        rows[rank] = dict(weights=str(weights), mlp_seed=int(row['mlp_seed']))
    return result, rows
