"""Private copy of scout_card/otr_cut_runtime_sc4; only own simulator root differs."""
import hashlib
import importlib.util
from pathlib import Path
from astra_runtime_safety_v1 import check_pressure

ASTRA_ROOT = Path(__file__).resolve().parents[1]


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
        path.relative_to(ASTRA_ROOT)
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
            result = original(cd)
        else:
            module.install(cd, opts=state['spec'].get('opts'))
            result = module.install_m(cd, original) if hasattr(module, 'install_m') else original(cd)
        # Card.run invokes self.step for each layer 0..15. Wrap only after all
        # carrier/source composition has completed; no numerical expression changes.
        step = cd.step
        def pressure_step(layer, state):
            check_pressure(layer)
            return step(layer, state)
        cd.step = pressure_step
        return result

    engine.M.install = install
    engine._otr_cut_runtime_v1 = state
    return configure
