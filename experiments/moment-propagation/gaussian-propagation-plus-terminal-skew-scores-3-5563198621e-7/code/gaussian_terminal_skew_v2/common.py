"""Inert versioned harness adapter; all v1 sources/results remain untouched."""
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'gaussian_terminal_skew_v1'
EXPECTED_COMMON = '2d82f6ab66de7cadaa2b9d91fbbfb5b5a29b39042fe21b654de5de4461ae3092'
assert hashlib.sha256((PREVIOUS / 'common.py').read_bytes()).hexdigest() == EXPECTED_COMMON
_spec = importlib.util.spec_from_file_location('gaussian_v1_common_frozen', PREVIOUS / 'common.py')
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
_base.HERE = HERE  # Only this adapter's module namespace; no old file/state edits.
for _name in ('NEW', 'P2', 'OFFPY', 'SITE', 'STORE', 'BUDGET', 'require', 'sha',
              'safe', 'write', 'utc', 'rss', 'verify', 'runtime', 'load_estimator',
              'alarm', 'metrics', 'guard', 'reset_closed', 'finish'):
    globals()[_name] = getattr(_base, _name)
PINS = dict(_base.PINS)
PINS.update({
    PREVIOUS / 'common.py': EXPECTED_COMMON,
    PREVIOUS / 'estimator.py': '4af4cefd575acb7e00803c1f9505f15440e724788bdcc5e0bae0da0f77ba2824',
    PREVIOUS / 'check_small.py': '2572c80a09c2255b70281991cc5ff9684566ecde8b016f4457298355b6aaeb52',
    PREVIOUS / 'run_native.py': '475952ece8a27ac77d14c0d403bcac769b1a2b5f3e7e4ba39a71127b2e5fd120',
    PREVIOUS / 'PROTOCOL.md': 'ff7a6aad4d05749f4eb17f3194ba2d5c050d431789ff6836bdf4fc039063ed96',
    PREVIOUS / 'NUMERICAL_ANNEX.md': 'e5fe6a31e284f33dfa07493a5329bd153bb36e90051c1610c088e375d7bd62e6',
    PREVIOUS / 'ROOT_DECISION.md': '5173d2ea431355e602c2e0a4636480da76a1d0bcf303f9a6f4d2dca0479de450',
    PREVIOUS / 'native_gaussian/result.json': '42990ce7aba07f23ec92cecc23709452ae97dc6f54c28bb5329a25e2d4ee51bb',
    HERE.parent / 'certify_reflected_relu_marginal_v1.py': 'f77b5ac71b08a172ec5cf481d4a0a8ea9387f9039cacf276aa14886e9acfa4a1',
})


def sources():
    result = {}
    for path, expected in PINS.items():
        require(sha(path) == expected, 'Predecessor/source pin: ' + str(path))
        result[str(path)] = expected
    for name in ('PROTOCOL.md', 'stable_gaussian.py', 'estimator.py', 'common.py',
                 'scalar_reference.py', 'check_scalar.py', 'check_small.py', 'run_native.py'):
        path = HERE / name
        result[str(path)] = sha(path)
    original = (PREVIOUS / 'estimator.py').read_text()
    start, stop = original.index('def gaussian_raw(mu, variance):'), original.index('def gaussian_covariance(')
    expected = original[:start] + original[stop:]
    expected = expected.replace('import flopscope as flops\n', 'from stable_gaussian import gaussian_raw\n')
    expected = expected.replace('Gaussian formulas follow parent 409f903e gaussian_raw/derivatives; the new\n'
                                'birth and terminal-only closure are specified in the companion PROTOCOL.md.',
                                'Gaussian formulas are evaluated by the stable numerical successor helper.\n'
                                'All nonscalar definitions are byte-preserved from gaussian_terminal_skew_v1.')
    require((HERE / 'estimator.py').read_text() == expected, 'Only scalar helper/import/docstring replacement')
    return result


def template(name):
    path = PREVIOUS / name
    require(sha(path) == PINS[path], 'Frozen harness template')
    spec = importlib.util.spec_from_file_location('gaussian_v1_' + name.replace('.', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
