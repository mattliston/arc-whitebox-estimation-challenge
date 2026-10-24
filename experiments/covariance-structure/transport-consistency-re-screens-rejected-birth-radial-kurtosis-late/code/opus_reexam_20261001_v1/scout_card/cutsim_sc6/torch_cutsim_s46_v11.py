"""Composite research simulator for S46 native cuts, unqualified until parity.

Worker interface: install(cd, opts=None), then install_m(cd, original_M_install).
Install after PL.install, before M.install. opts: enabled (default true),
ru (default 256), rowcuts (overrides for rowcuts DEFAULTS).
{enabled: false} gives a strictly untouched original card, including M.install.
No torch import, network execution, or submission operations occur on import.

Remaining residual scope: native covariance dead cut, learned-edge row/column cut
and tanh stand-in, native float32/tree arithmetic. Native-law response fit and
repaired beta columns are separately switchable diagnostics. These must be quantified by
paired prediction residuals before treating this as a refit training card.
"""
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_CHILDREN = {}


def _load(filename):
    if filename not in _CHILDREN:
        path = HERE / filename
        spec = importlib.util.spec_from_file_location('_cutsim_' + path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CHILDREN[filename] = module
    return _CHILDREN[filename]


def sources():
    """The entry-file hash alone is insufficient: record every child hash."""
    names = (Path(__file__).name, 'torch_push_native_v2.py', 'torch_rowcuts_v15.py', 'pbase_cutsim_v1.py', 'n6y_cutsim_v2.py')
    return {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in names}


def install(cd, opts=None):
    opts = dict(opts or {})
    assert set(opts) <= {'enabled', 'ru', 'rowcuts', 'law_response_fit', 'law_beta', 'law_restore_right'}, sorted(opts)
    enabled = bool(opts.get('enabled', True))
    cd.cutsim_s46_enabled = enabled
    cd.cutsim_law_response_fit = bool(opts.get('law_response_fit', True))
    cd.cutsim_law_beta = bool(opts.get('law_beta', True))
    cd.cutsim_law_restore_right = bool(opts.get('law_restore_right', False))
    cd.cutsim_s46_sources = sources()
    if not enabled:
        return cd
    _load('torch_push_native_v2.py').install(cd, cd.t, ru=opts.get('ru', 256))
    _load('torch_rowcuts_v15.py').install(cd, cd.t, opts=opts.get('rowcuts'))
    return cd


def install_m(cd, original_M_install):
    if not getattr(cd, 'cutsim_s46_enabled', False):
        return original_M_install(cd)
    return _load('torch_rowcuts_v15.py').install_m(cd, original_M_install)
