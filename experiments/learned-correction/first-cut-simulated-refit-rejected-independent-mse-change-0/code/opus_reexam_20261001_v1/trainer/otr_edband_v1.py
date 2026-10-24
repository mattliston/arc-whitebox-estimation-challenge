"""Source-only trainer EDGE bands; no torch import or network work on import.

Bands 8-9 and 10-12 have two independent FP32 offsets per family; band
13-14 is the shared reference. Shared e.bias/e.output remain in the original
28-array / 11-edge schema. This adds 64 identifiable degrees of freedom.

Install Runtime.bind(engine) AFTER the cut-runtime wrapper, then configure on
EVERY request. An enabled request must load its base from the new envelope;
legacy load_params deliberately cannot read it. See EDBAND_HANDOFF_v1.md.
"""
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

from otr_common import DIAL_KEYS

EDGES = ((8, 9), (10, 12), (13, 14))
FAMILIES = ('bias', 'output')
CHECKPOINT_SCHEMA = 'trainer_edge_band_checkpoint_v1'
OFFSET_SCHEMA = 'trainer_edge_band_offsets_v1'
REQUEST_SCHEMA = 'trainer_edge_band_request_v1'
DESIGN = 'shared_reference_13_14_two_offsets'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha(value, name):
    _require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), name)
    return value


def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _clone(torch, value):
    """Detached CPU storage copies, preserving extra base fields recursively."""
    if isinstance(value, torch.Tensor):
        _require(value.device.type == 'cpu', 'GPU tensors are unsupported')
        _require(value.layout == torch.strided, 'non-strided tensor unsupported')
        _require(bool(torch.isfinite(value).all()), 'nonfinite checkpoint tensor')
        return value.detach().clone()
    if isinstance(value, dict):
        _require(all(type(k) is str for k in value), 'checkpoint keys must be strings')
        return {k: _clone(torch, v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone(torch, v) for v in value)
    _require(value is None or type(value) in (str, bool, int, float), 'unsupported checkpoint value')
    _require(type(value) is not float or math.isfinite(value), 'nonfinite checkpoint scalar')
    return value


def _fingerprint(torch, value):
    """Hash shapes/dtypes/bytes and all nested metadata, with unambiguous framing."""
    h = hashlib.sha256()
    def frame(data):
        h.update(len(data).to_bytes(8, 'big')); h.update(data)
    def visit(v):
        if isinstance(v, torch.Tensor):
            frame(b'tensor'); frame(str(v.dtype).encode()); frame(repr(tuple(v.shape)).encode())
            frame(v.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            frame(b'dict'); frame(str(len(v)).encode())
            for key in sorted(v):
                frame(key.encode()); visit(v[key])
        elif isinstance(v, (list, tuple)):
            frame(type(v).__name__.encode()); frame(str(len(v)).encode())
            for item in v: visit(item)
        else:
            frame(type(v).__name__.encode()); frame(json.dumps(v, allow_nan=False).encode())
    visit(value)
    return h.hexdigest()


def _base(torch, base):
    _require(isinstance(base, dict) and {'arrays', 'dials'} <= set(base), 'base arrays/dials required')
    arrays = base['arrays']
    _require(isinstance(arrays, dict) and len(arrays) == 28, 'base must retain 28 arrays')
    _require(sum(k.startswith('e.') for k in arrays) == 11, 'base must retain 11 edge arrays')
    for key, value in arrays.items():
        _require(isinstance(value, torch.Tensor) and value.dtype == torch.float32
                 and value.device.type == 'cpu' and bool(torch.isfinite(value).all()), 'invalid base array ' + key)
    for key in FAMILIES:
        _require(tuple(arrays['e.' + key].shape) == (16,), 'invalid shared edge shape ' + key)
    _require(isinstance(base['dials'], dict) and tuple(base['dials']) == DIAL_KEYS, 'dial schema/order changed')
    _require(all(type(x) in (int, float) and math.isfinite(x) for x in base['dials'].values()), 'invalid dials')
    if 'pl' in base:
        _require(isinstance(base['pl'], dict), 'pl must be a dictionary')
        for key, values in base['pl'].items():
            _require(isinstance(values, (list, tuple)) and len(values) == 16
                     and all(type(x) in (int, float) and math.isfinite(x) for x in values), 'invalid pl ' + key)
    _clone(torch, base)  # Validate preserved extra fields too.
    return base


def validate(torch, payload):
    _require(isinstance(payload, dict) and set(payload) == {'schema', 'base', 'edge_band', 'source'}, 'checkpoint envelope fields')
    _require(payload['schema'] == CHECKPOINT_SCHEMA, 'checkpoint schema')
    _base(torch, payload['base'])
    state = payload['edge_band']
    _require(isinstance(state, dict) and set(state) == {'schema', 'design', 'edges', 'offsets'}, 'offset fields')
    _require(state['schema'] == OFFSET_SCHEMA and state['design'] == DESIGN, 'offset schema/design')
    _require(state['edges'] == [list(x) for x in EDGES], 'band definition changed')
    _require(set(state['offsets']) == set(FAMILIES), 'offset families')
    for family, value in state['offsets'].items():
        _require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.dtype == torch.float32
                 and tuple(value.shape) == (2, 16) and bool(torch.isfinite(value).all()), 'invalid offsets ' + family)
    _require(isinstance(payload['source'], dict), 'source must be JSON dictionary')
    _json_copy(payload['source'])
    return payload


def make_checkpoint(torch, base, offsets=None, source=None):
    """Construct an independent payload; base extras (including pl) are retained."""
    values = {f: torch.zeros((2, 16), dtype=torch.float32) for f in FAMILIES} if offsets is None else offsets
    result = dict(schema=CHECKPOINT_SCHEMA, base=_clone(torch, base),
                  edge_band=dict(schema=OFFSET_SCHEMA, design=DESIGN, edges=[list(x) for x in EDGES],
                                 offsets=_clone(torch, values)), source=_json_copy(source or {}))
    return validate(torch, result)


def _publish(path, write, binary=False):
    """Publish by exclusive hard link: a race cannot overwrite an existing file."""
    path = Path(path)
    _require(path.parent.is_dir(), 'output directory must already exist')
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb' if binary else 'w') as stream:
            write(stream); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)  # Private temporary only; never removes a published artifact.


def save_checkpoint(torch, path, payload):
    saved = _clone(torch, validate(torch, payload))
    _publish(path, lambda stream: torch.save(saved, stream), binary=True)
    return dict(path=str(Path(path).resolve()), sha256=sha256(path), content_sha256=_fingerprint(torch, saved))


def load_checkpoint(torch, path, expected_sha256):
    _sha(expected_sha256, 'checkpoint SHA required')
    _require(sha256(path) == expected_sha256, 'checkpoint file hash changed')
    # Read one immutable byte snapshot so validation and deserialization share bytes.
    import io
    raw = Path(path).read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == expected_sha256, 'checkpoint changed during read')
    value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    return _clone(torch, validate(torch, value))


def request_spec(path, checkpoint_sha256, train_families=FAMILIES):
    _sha(checkpoint_sha256, 'checkpoint SHA required')
    families = list(train_families)
    _require(len(families) == len(set(families)) and set(families) <= set(FAMILIES), 'training families')
    return dict(schema=REQUEST_SCHEMA, module_sha256=sha256(__file__), checkpoint=str(Path(path).resolve()),
                checkpoint_sha256=checkpoint_sha256, train_families=families)


class Session:
    """One request's leaves and card wrappers. Construct anew for every request."""
    def __init__(self, torch, spec, request):
        self.t = torch
        _require(set(spec) == {'schema', 'module_sha256', 'checkpoint', 'checkpoint_sha256', 'train_families'}, 'request spec fields')
        _require(spec['schema'] == REQUEST_SCHEMA and spec['module_sha256'] == sha256(__file__), 'request module/schema mismatch')
        _require(isinstance(spec['checkpoint'], str) and Path(spec['checkpoint']).is_absolute(), 'absolute checkpoint path required')
        families = spec['train_families']
        _require(isinstance(families, list) and len(families) == len(set(families)) and set(families) <= set(FAMILIES), 'training families')
        _require(request.get('mode') in ('grad', 'eval') and isinstance(request.get('id'), str)
                 and type(request.get('rank')) is int, 'request identity required')
        if request['mode'] == 'grad':
            _require(type(request.get('scale')) in (float, int) and math.isfinite(request['scale']) and request['scale'] > 0, 'positive gradient scale required')
        _require(isinstance(request.get('params'), str) and Path(request['params']).resolve() == Path(spec['checkpoint']).resolve(),
                 'request params must identify the EDBAND envelope')
        self.spec = _json_copy(spec)
        self.request = _json_copy(request)
        self.payload = load_checkpoint(torch, spec['checkpoint'], spec['checkpoint_sha256'])
        self.offsets = {f: v.clone().requires_grad_(request['mode'] == 'grad' and f in families)
                        for f, v in self.payload['edge_band']['offsets'].items()}
        self._installed = 0
        self._base_verified = False
        self._identity = dict(schema='trainer_edge_band_provenance_v1', enabled=True,
                              design=DESIGN, edges=[list(x) for x in EDGES], request=_json_copy(request),
                              request_sha256=hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                              spec=self.spec, base_sha256=_fingerprint(torch, self.payload['base']),
                              payload_sha256=_fingerprint(torch, self.payload))
        # Ordinary addition has identity derivatives. Preserve shared signed-zero
        # bits at exactly zero offsets; do not detach/suppress zero-leaf gradients.
        class AddOffset(torch.autograd.Function):
            @staticmethod
            def forward(ctx, base, offset):
                return torch.where(offset == 0, base, base + offset)
            @staticmethod
            def backward(ctx, gradient):
                return gradient, gradient
        self._add = AddOffset.apply

    def base(self):
        return _clone(self.t, self.payload['base'])

    def verify_base(self, engine):
        """Fail closed if a worker forgot any base arrays, dials or PL shifts."""
        base = self.payload['base']
        _require(_fingerprint(self.t, engine.T) == _fingerprint(self.t, base['arrays']), 'live engine arrays differ from checkpoint')
        current_dials = {k: float(v.detach()) for k, v in engine.DL.items()}
        _require(current_dials == base['dials'], 'live engine dials differ from checkpoint')
        shifts = {k: list(v) for k, v in base.get('pl', {}).items()}
        extra = self.request.get('pl') or {}
        _require(isinstance(extra, dict), 'request pl dictionary required')
        for family, values in extra.items():
            _require(isinstance(values, list) and len(values) == 16
                     and all(type(x) in (int, float) and math.isfinite(x) for x in values), 'invalid request pl ' + family)
            shifts[family] = [a + b for a, b in zip(shifts.get(family, [0.] * 16), values)]
        if engine.PL is None:
            _require(not shifts, 'PL is required by checkpoint/request')
        else:
            _require(set(shifts) <= set(engine.PL.EPS), 'unknown checkpoint/request PL family')
            for family, leaf in engine.PL.EPS.items():
                expected = self.t.tensor(shifts.get(family, [0.] * 16), dtype=self.t.float64)
                _require(_fingerprint(self.t, leaf) == _fingerprint(self.t, expected), 'live PL differs: ' + family)
        self._base_verified = True
        return self

    def install(self, cd):
        _require(self._base_verified, 'verify full engine base before installing EDBAND')
        _require(self._installed == 0, 'Session cannot be reused across cards/requests')
        _require(not hasattr(cd, '_otr_edband_v1'), 'EDBAND already installed on card')
        env = cd.env
        original = env['_message_edge_source']
        shared = env['_EDGE_PARAMETERS']
        _require(callable(original) and isinstance(shared, dict), 'edge card interface')
        for family in FAMILIES:
            _require(_fingerprint(self.t, shared[family]) == _fingerprint(self.t, self.payload['base']['arrays']['e.' + family]),
                     'card/checkpoint shared edge mismatch ' + family)
        def edge(layer, *args, **kwargs):
            _require(type(layer) is int and 0 <= layer <= 15, 'explicit integer layer required')
            if not 8 <= layer <= 12:
                return original(layer, *args, **kwargs)
            index = 0 if layer <= 9 else 1
            previous = env['_EDGE_PARAMETERS']
            _require(previous is shared, 'concurrent or nested edge dictionary mutation')
            selected = dict(shared)
            for family in FAMILIES:
                selected[family] = self._add(shared[family], self.offsets[family][index])
            env['_EDGE_PARAMETERS'] = selected
            try:
                return original(layer, *args, **kwargs)
            finally:
                env['_EDGE_PARAMETERS'] = previous
        env['_message_edge_source'] = edge
        cd._otr_edband_v1 = self
        self._installed += 1
        return cd

    def record(self, record):
        _require(self._installed == 1, 'exactly one card installation per request required')
        for name in ('rank', 'mode'):
            _require(record.get(name) == self.request[name], 'result request mismatch ' + name)
        for name in ('id', 'params'):
            if name in record:
                _require(record[name] == self.request[name], 'result request mismatch ' + name)
        _require('edge_band' not in record and 'edge_band_grads' not in record, 'result already decorated')
        _require(_fingerprint(self.t, self.offsets) == _fingerprint(self.t, self.payload['edge_band']['offsets']), 'offset values changed during request')
        record['edge_band'] = _json_copy(self._identity)
        if self.request['mode'] == 'grad':
            _require(record.get('scale') == self.request['scale'], 'gradient scale mismatch')
            grads = {}
            for family, leaf in self.offsets.items():
                if leaf.requires_grad:
                    _require(leaf.grad is not None and bool(self.t.isfinite(leaf.grad).all()), 'missing/nonfinite offset gradient ' + family)
                    grads[family] = leaf.grad.detach().clone()
            record['edge_band_grads'] = dict(schema='trainer_edge_band_gradients_v1',
                                            identity=_json_copy(self._identity), scale=self.request['scale'], values=grads)
        return record


class Runtime:
    """Optional install adapter; no worker is modified by importing this module."""
    def __init__(self, torch):
        self.t, self.session, self.request = torch, None, None

    def configure(self, spec, request):
        # Clear before validating: a failed or disabled request cannot reuse leaves.
        self.session, self.request = None, None
        candidate = None if spec is None else Session(self.t, spec, request)
        self.request = _json_copy(request)
        self.session = candidate
        return None if self.session is None else self.session.base()

    def bind(self, engine):
        _require(engine.M is not None and not hasattr(engine, '_otr_edband_runtime_v1'), 'M install adapter unavailable/already bound')
        original = engine.M.install
        def install(cd):
            _require(self.request is not None, 'configure EDBAND on every request')
            if self.session is not None:
                self.session.verify_base(engine)
            result = original(cd)  # Includes M and any already-bound cut simulator.
            if self.session is not None:
                self.session.install(cd)
            return result
        engine.M.install = install
        engine._otr_edband_runtime_v1 = self
        return self

    def record(self, record):
        _require(self.request is not None, 'no configured request')
        for key in ('rank', 'mode'):
            _require(record.get(key) == self.request[key], 'result request mismatch ' + key)
        if self.session is None:
            _require('edge_band' not in record and 'edge_band_grads' not in record, 'result already decorated')
            record['edge_band'] = dict(schema='trainer_edge_band_provenance_v1', enabled=False,
                                       module_sha256=sha256(__file__), request=_json_copy(self.request))
            return record
        return self.session.record(record)


def export_native(torch, payload, *, parent_native_sha256, source):
    """Full FP32-rounded constants; mandatory binding is the actual builder parent."""
    validate(torch, payload)
    _sha(parent_native_sha256, 'explicit direct-parent native SHA required')
    required = {'checkpoint_path', 'checkpoint_sha256', 'request_id', 'request_sha256', 'head_sha256'}
    _require(isinstance(source, dict) and required <= set(source), 'checkpoint/request/head export identities required')
    _sha(source['checkpoint_sha256'], 'export checkpoint SHA required')
    _sha(source['request_sha256'], 'export request SHA required')
    _require(isinstance(source['request_id'], str) and bool(source['request_id']), 'export request id required')
    _require(isinstance(source['checkpoint_path'], str) and Path(source['checkpoint_path']).is_absolute(), 'absolute export checkpoint path required')
    if source['head_sha256'] is not None:
        _sha(source['head_sha256'], 'head SHA must be explicit, or null for no head')
    saved = load_checkpoint(torch, source['checkpoint_path'], source['checkpoint_sha256'])
    _require(_fingerprint(torch, saved) == _fingerprint(torch, payload), 'export payload is not the identified saved checkpoint')
    out = dict(schema='edge_band_v1', edges=[list(x) for x in EDGES], parent_native_sha256=parent_native_sha256,
               source=dict(checkpoint_source=_json_copy(payload['source']), export=_json_copy(source),
                           payload_sha256=_fingerprint(torch, payload), module_sha256=sha256(__file__), design=DESIGN))
    for family in FAMILIES:
        base = payload['base']['arrays']['e.' + family]
        offsets = payload['edge_band']['offsets'][family]
        rows = [torch.where(delta == 0, base, base + delta) for delta in offsets] + [base]
        _require(all(bool(torch.isfinite(row).all()) for row in rows), 'nonfinite exported sums')
        out[family] = [row.detach().tolist() for row in rows]
    return out


def save_native(path, values):
    _publish(path, lambda stream: json.dump(values, stream, allow_nan=False, sort_keys=True, indent=2))
    return dict(path=str(Path(path).resolve()), sha256=sha256(path))
