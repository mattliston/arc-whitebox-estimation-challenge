"""Versioned optional EDBAND worker v2 with explicit truth-plan support; source-only integration, not qualified.

Same CLI/environment as worker_generated_v4.py. Imports are light until main.
The original worker/engine remain untouched; exact source pins and count-checked
in-memory edits add envelope loading and pre-card-deletion provenance capture.
No request field means the original numerical parameter/M/cutsim path, without
an EDGE hook. An envelope sent without edge_band fails the legacy loader.
"""
import ast
import hashlib
import inspect
from pathlib import Path
import sys

import otr_edband_v1 as EB

HERE = Path(__file__).resolve().parent
PINS = {
    'worker_generated_v4.py': '308cc7b08d2291e8e817ac722ae3e7184dc9e95bdc3f13990659a5b958e244a1',
    'otr_engine.py': 'ea8a3bc6a8d178427470ef74389f2a5ecf94ee1a717e39710a87fcd184a7ec17',
    'otr_edband_v1.py': '24a7debb6f8994982fb4052d01b9f28ca07729a8a179a1628c8bac9b6bb4ffc6',
}


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('EDBAND integration source changed: ' + before[:100])
    return source.replace(before, after)


def verify_sources():
    for name, digest in PINS.items():
        if EB.sha256(HERE / name) != digest:
            raise ValueError('EDBAND pinned source changed: ' + name)


def engine_source(source):
    marker = '    del prediction, loss, cd, original_step, step, weights, truth, target\n'
    return replace_once(source, marker,
        '    _EDBAND_CAPTURE_V1(record, mode, scale)  # Card and offset leaves still alive.\n' + marker)


def bind_engine(engine):
    """Called only after the cut-runtime adapter is installed."""
    verify_sources()
    if Path(engine.__file__).resolve() != HERE / 'otr_engine.py':
        raise ValueError('Unexpected engine module')
    source = inspect.getsource(engine.run_network)
    original_text = (HERE / 'otr_engine.py').read_text()
    nodes = [node for node in ast.parse(original_text).body
             if isinstance(node, ast.FunctionDef) and node.name == 'run_network']
    if len(nodes) != 1:
        raise ValueError('Pinned engine run_network missing')
    original_source = ''.join(original_text.splitlines(keepends=True)[nodes[0].lineno-1:nodes[0].end_lineno])
    if source != original_source:
        raise ValueError('EDBAND requires original, unwrapped run_network')
    runtime = EB.Runtime(engine.torch).bind(engine)
    patched = engine_source(source)
    identity = dict(schema='trainer_edge_band_worker_v1', worker_sha256=EB.sha256(__file__),
        sources=dict(PINS), patched_run_sha256=hashlib.sha256(patched.encode()).hexdigest())

    def capture(record, mode, scale):
        # The ordinary engine adds scale after deleting the card. Make the same
        # scalar available here so Session can validate and snapshot gradients.
        if mode == 'grad':
            record['scale'] = scale
        runtime.record(record)
        if 'edge_band_worker' in record:
            raise ValueError('EDBAND worker provenance already present')
        record['edge_band_worker'] = dict(identity)

    engine.__dict__['_EDBAND_CAPTURE_V1'] = capture
    exec(compile(patched, '<EDBAND pre-delete capture v1>', 'exec'), engine.__dict__)
    return runtime


def transformed_worker():
    verify_sources()
    source = (HERE / 'worker_generated_v4.py').read_text()
    source = replace_once(source, 'configure_cutsim = enable_cutsim(E)\n',
        'configure_cutsim = enable_cutsim(E)\n'
        'edband_runtime = _EDBAND_BIND_ENGINE_V1(E)\n')
    source = replace_once(source, 'def serve(claim):\n', 'def _serve_edband_inner(claim):\n')
    source = replace_once(source, "        configure_cutsim(request.get('cutsim'))\n",
        "        configure_cutsim(request.get('cutsim'))\n"
        "        edband_base = edband_runtime.configure(request.get('edge_band'), request)\n")
    before = """        if request['params'] != loaded:
            loaded = None
            E.set_parameters(*load_params(torch, request['params']))
            loaded_pl = load_pl(torch, request['params'])
            loaded = request['params']
"""
    after = """        if edband_base is not None:
            # Never cache an enabled envelope as a legacy-loaded checkpoint.
            # A later omitted edge_band field must hit (and fail) the old loader.
            loaded, loaded_pl = None, {}
            E.set_parameters(edband_base['arrays'], edband_base['dials'])
            loaded_pl = edband_base.get('pl', {})
        elif request['params'] != loaded:
            loaded = None
            E.set_parameters(*load_params(torch, request['params']))
            loaded_pl = load_pl(torch, request['params'])
            loaded = request['params']
"""
    source = replace_once(source, before, after)
    source = replace_once(source, '    except Exception:\n',
        '    except Exception:\n'
        '        loaded, loaded_pl = None, {}\n'
        '        edband_runtime.session, edband_runtime.request = None, None\n'
        '        configure_cutsim(None)\n')
    marker = "say(event='start', run=str(RUN), threads=E.THREADS, torch=torch.__version__, card=E.CARD, per_layer=E.PL is not None)\n"
    wrapper = """def serve(claim):
    # Reset even for skipped/already-completed claims and malformed requests.
    # A failed cut spec cannot retain the preceding request's simulator state.
    edband_runtime.session, edband_runtime.request = None, None
    configure_cutsim(None)
    try:
        return _serve_edband_inner(claim)
    finally:
        edband_runtime.session, edband_runtime.request = None, None
        configure_cutsim(None)


"""
    source = replace_once(source, marker, wrapper + marker)
    compile(source, '<EDBAND worker v1>', 'exec')
    return source


def main():
    if len(sys.argv) != 3:
        raise SystemExit('usage: python -B worker_edband_v1.py <run directory> <worker index>')
    source = transformed_worker()
    namespace = dict(__name__='__main__', __file__=str(HERE / 'worker_generated_v4.py'),
                     _EDBAND_BIND_ENGINE_V1=bind_engine)
    exec(compile(source, '<EDBAND worker v1>', 'exec'), namespace)


if __name__ == '__main__':
    main()
