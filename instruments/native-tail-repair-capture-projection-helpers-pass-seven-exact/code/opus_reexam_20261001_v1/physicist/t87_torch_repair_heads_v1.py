"""Prospective separate tail-repair heads on the frozen S46 v11 simulator.

Source-only preparation: not numerically qualified. No torch import or network
execution on import. Install via the pinned delegate in engineer/a22_cutsim.
The current v11, M, rowcuts and trainer worker remain untouched.
"""
import hashlib
import inspect
import json
from pathlib import Path
import types

HERE = Path(__file__).resolve().parent
O = HERE.parent
CUTS = O / 'engineer/a22_cutsim'
FEATURE_SPEC = HERE / 't87_repair_head_feature_spec_v1.json'
FEATURE_SHA = '29fd46fa9ad6243249bb4b4606f20e3cd78f6b282781061fda500a8b6a7af821'
PINS = {
    'torch_cutsim_s46_v11.py': 'bfe53193470738c6c594d338ca4ccb9cf8294f357831e032d62a7dd4006abb96',
    'torch_push_native_v2.py': 'c3b128b9e1a17b1685e071fd05cd3152f55ad0ef38d4e8e51a18553560e880b7',
    'torch_rowcuts_v15.py': 'bb824a6c4772ab5a394e6ee5da7c09602c54f5fe6d72d9b2470b21a7ee05a288',
    'pbase_cutsim_v1.py': '5af2eb004986052cf2aff23fb126554825f489d0c0f4ba9a097fcbc7f32e2087',
    'n6y_cutsim_v2.py': '8ccc97beefd7a43379b4344afab28031b03907d5fd2fb140bbf28286a02870c8',
}
M_SHA = 'bba0dea098ad93eee1bf207ace391529d716b7e180e085d2ebb3a25fb8a787f1'
LAW_FLAGS = dict(enabled=True, law_response_fit=True, law_beta=True, law_restore_right=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    result = {str(Path(__file__).resolve()): sha(__file__)}
    for name, expected in PINS.items():
        path = CUTS / name
        assert sha(path) == expected, ('changed parent', path)
        result[str(path)] = expected
    assert sha(FEATURE_SPEC) == FEATURE_SHA, 'changed feature specification'
    result[str(FEATURE_SPEC)] = FEATURE_SHA
    mpath = O / 'trainer/otr_m.py'
    assert sha(mpath) == M_SHA, 'changed M source'
    result[str(mpath)] = M_SHA
    return result


def standardize(t, repair, rows, variance):
    """Exact feature specification: full moments, actual-roster-only output."""
    assert repair.shape == variance.shape == (1024,)
    assert rows.ndim == 1 and rows.dtype == t.int64
    vc = variance.detach().to(t.float64).clamp_min(1e-10)
    raw = repair.detach().to(t.float64) / (vc * vc.sqrt())
    mean = raw.sum() / 1024
    centered = raw - mean
    spread = ((centered * centered).sum() / 1023).sqrt()
    z = t.zeros(1024, dtype=t.float32, device=raw.device)
    if bool(t.isfinite(spread)) and float(spread) > 0:
        selected = (centered[rows] / spread).clamp(-8, 8).to(t.float32)
        z = z.index_copy(0, rows, selected)
    return z


def _old(cd, layer, repair, blocks):
    if not 7 <= layer <= 14:
        return
    t = cd.t
    rows = blocks[0][0] if blocks else t.empty(0, dtype=t.int64, device=repair.device)
    assert all(t.equal(b[0], rows) for b in blocks), 'old blocks use different repair rosters'
    cd.repair_head_parts = dict(layer=int(layer), old=repair.detach(), old_rows=rows.detach(),
                               young=t.zeros_like(repair), young_rows=t.empty(0, dtype=t.int64, device=repair.device))


def _young(cd, layer, tail, part, scale):
    if not 7 <= layer <= 14:
        return
    state = cd.repair_head_parts
    assert state is not None and state['layer'] == layer, 'young repair without old capture'
    # Match the actual part*yrep independently; never combined-minus-old.
    state['young'] = cd.t.zeros_like(state['old']).index_add(0, tail, part.detach() * scale)
    state['young_rows'] = tail.detach()


def _prepare(cd, t, layer, ctx):
    state = cd.repair_head_parts
    assert state is not None and state['layer'] == layer, 'repair capture layer mismatch'
    variance = t.diagonal(ctx['Cpre']).detach()
    with t.no_grad():
        z = t.stack((standardize(t, state['old'], state['old_rows'], variance),
                     standardize(t, state['young'], state['young_rows'], variance)), dim=1)
    assert z.shape == (1024, 2) and bool(t.isfinite(z).all())
    cd.repair_head_features = (int(layer), z)


def _apply(cd, t, layer, P, z, hu, side, record):
    assert hu.shape == (42, 128) and z.shape == (1024, 42)
    # Preserve incumbent projection shape and reduction order. Both new HU rows
    # participate only in the separate projection, including at zero weights.
    incumbent = t.cat((hu[:31], t.zeros_like(hu[31:33]), hu[33:]), dim=0)
    bias = P['enc0_b'][None, :] + z @ incumbent
    if 7 <= layer <= 14:
        key = ('repair_hf', layer)
        if side['again']:
            extra = record[key]
        else:
            saved = cd.repair_head_features
            assert saved is not None and saved[0] == layer, 'head features missing or from another layer'
            extra = saved[1]
            if t.is_grad_enabled():
                record[key] = extra
        bias = bias + extra @ hu[31:33]
    return dict(P, enc0_b=bias)


def patch_m(source):
    edits = [
        ("                    P = dict(P, enc0_b=P['enc0_b'][None, :] + z @ HEADU)\n",
         "                    P = _repair_head_apply(cd, t, l, P, z, HEADU, side, record)\n"),
        ('            pin = mine_in(l, pin, ctx)\n',
         "            pin = mine_in(l, pin, ctx)\n"
         "            if HEADF and not side['again'] and 7 <= l <= 14:\n"
         "                _repair_head_prepare(cd, t, l, ctx)\n"),
    ]
    for before, after in edits:
        assert source.count(before) == 1, ('M source site changed', before)
        source = source.replace(before, after)
    return source


def row_source():
    body = (CUTS / 'torch_rowcuts_v15.py').read_text()
    assert hashlib.sha256(body.encode()).hexdigest() == PINS['torch_rowcuts_v15.py']
    edits = [
        ('        cd.cutsim_law_pp = None\n',
         '        cd.cutsim_law_pp = None\n        cd.repair_head_parts = None\n        cd.repair_head_features = None\n'),
        ("        if 7 <= layer <= 14 and o['ynt'] < 90 and pp.shape[1] > old:\n",
         "        _repair_head_old(cd, layer, repair, blocks)\n"
         "        if 7 <= layer <= 14 and o['ynt'] < 90 and pp.shape[1] > old:\n"),
        ("                repair = repair.index_add(0, tail, part * o['yrep'])\n",
         "                _repair_head_young(cd, layer, tail, part, o['yrep'])\n"
         "                repair = repair.index_add(0, tail, part * o['yrep'])\n"),
        ('    source = textwrap.dedent(inspect.getsource(original_m_install))\n',
         '    source = textwrap.dedent(inspect.getsource(original_m_install))\n    source = _repair_head_patch_m(source)\n'),
        ('    namespace = dict(original_m_install.__globals__)\n',
         '    namespace = dict(original_m_install.__globals__)\n'
         '    namespace.update(_repair_head_apply=_repair_head_apply, _repair_head_prepare=_repair_head_prepare)\n'),
    ]
    for before, after in edits:
        assert body.count(before) == 1, ('row source site changed', before)
        body = body.replace(before, after)
    return body


def _module(name, body=None):
    module = types.ModuleType('_t87_private_' + Path(name).stem)
    module.__file__ = str(CUTS / name)
    module.__dict__.update(_repair_head_old=_old, _repair_head_young=_young,
                          _repair_head_patch_m=patch_m, _repair_head_apply=_apply,
                          _repair_head_prepare=_prepare)
    exec(compile(body if body is not None else (CUTS / name).read_text(), module.__file__, 'exec'), module.__dict__)
    return module


def install(cd, opts=None):
    opts = dict(opts or {})
    assert set(opts) == {'enabled', 'law_response_fit', 'law_beta', 'law_restore_right', 'repair_head'}
    assert {k: opts[k] for k in LAW_FLAGS} == LAW_FLAGS, 'exact passing v11 cuts required'
    repair = opts['repair_head']
    assert isinstance(repair, dict) and set(repair) == {'enabled', 'feature_spec'}
    assert type(repair['enabled']) is bool
    assert repair['feature_spec'] == dict(path=str(FEATURE_SPEC), sha256=FEATURE_SHA)
    cd.repair_head_sources = sources()
    cd.repair_head_enabled = repair['enabled']
    if not cd.repair_head_enabled:
        # Strict bypass: exactly the unmodified passing simulator installer.
        parent = _module('torch_cutsim_s46_v11.py')
        cd._repair_head_parent = parent
        return parent.install(cd, dict(LAW_FLAGS))
    cd.cutsim_s46_enabled = True
    cd.cutsim_law_response_fit = cd.cutsim_law_beta = cd.cutsim_law_restore_right = True
    cd.cutsim_s46_sources = dict(cd.repair_head_sources)
    cd.repair_head_parts = cd.repair_head_features = None
    _module('torch_push_native_v2.py').install(cd, cd.t, ru=256)
    rows = _module('torch_rowcuts_v15.py', row_source())
    rows.install(cd, cd.t)
    cd._repair_head_rows = rows
    return cd


def install_m(cd, original_M_install):
    assert sources() == cd.repair_head_sources, 'feature source changed during install'
    if not cd.repair_head_enabled:
        return cd._repair_head_parent.install_m(cd, original_M_install)
    env = original_M_install.__globals__
    assert env['HEADF'] and env['HEADU'] is not None, 'repair heads require normal head features'
    assert not env['HSPLIT'] and not env['HBANDS'] and not env['B7'], 'HU31/32 alias conflict'
    assert tuple(env['HEADU'].shape) == (42, 128)
    return cd._repair_head_rows.install_m(cd, original_M_install)
