"""runtime_weightfit_v1 (scout-card, 2026-10-12): = trainer/otr_cut_runtime_v2.py that also accepts any O/scout_card/cutsim_sc<k>/ simulator."""
"""Pinned offline cut-simulator hook, between PL.install and M.install.

Every request explicitly identifies the immutable simulator and its options.
Plain requests retain the current torch arithmetic. The simulator must expose
install(cd, opts=None); optionally install_m(cd, original_M_install) replaces the
last installation call for native-specific learned-feature projections.
"""
import hashlib
import importlib.util
from pathlib import Path

from otr_common import O


def enable(engine):
    assert engine.M is not None
    assert not hasattr(engine, '_otr_cut_runtime_v1'), 'Hook already installed'
    original = engine.M.install
    modules = {}
    state = {'spec': None, 'module': None}

    def configure(spec):
        if not spec:
            state.update(spec=None, module=None)
            return
        assert isinstance(spec, dict) and set(spec) <= {'module', 'sha256', 'opts', 'sources'}
        path = Path(spec['module']).resolve()
        # This isolated worker accepts only its own immutable sibling cutsim.
        assert path == Path(__file__).resolve().parent / 'cutsim' / 'torch_cutsim_s46_v11.py', path
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert sha == spec['sha256'], 'Cut-simulator source changed: %s' % path
        identity = (str(path), sha)
        if identity not in modules:
            name = 'otr_cut_simulator_' + sha[:16]
            module_spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            assert callable(module.install)
            modules[identity] = module
        module = modules[identity]
        if hasattr(module, 'sources'):
            assert spec.get('sources') == module.sources(), 'Cut-simulator dependency source changed'
        state.update(spec=dict(spec), module=module)

    def install(cd):
        module = state['module']
        if module is None:
            return original(cd)
        module.install(cd, opts=state['spec'].get('opts'))
        if hasattr(module, 'install_m'):
            return module.install_m(cd, original)
        return original(cd)

    engine.M.install = install
    engine._otr_cut_runtime_v1 = state
    return configure
