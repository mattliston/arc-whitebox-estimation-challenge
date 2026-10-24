"""DRAFT inert source composer for one sealed, final-only v29 observer.

No numerical package, parameter archive or estimator is loaded on import.
The physical MIT source/feature owner remain immutable. compose() is a source
operation, not eligibility to build, qualify, score or submit a candidate.
"""
import ast
import hashlib
import importlib.util
import math
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
PROJECTED = HERE/'projection_adapter.py'
PROJECTED_SHA = '3bcefbe9503e8457cc8be2d9dcf3d23effd5b0217bd44dbe9c3a3fd955d29cf0'
FEATURE = HERE/'adapter.py'
FEATURE_SHA = 'cab844cd7981d31c7e44f4797ad60490729bdfadb8ce46655cdd5966ae9f36b7'
MODEL = HERE/'model.py'
MODEL_SHA = '59564d394f37758bb85805099b2f32ff9a6576825339b561c689e688f9c087ef'
PHYSICAL_SHA = 'c50070d4f1dfd47cf7148dbee4c0ca7119c42f114ff9df05bc5fe4806d8d4f24'
SHAPES = dict(M=(64,48),b_M=(48,),E0=(126,128),b0=(128,),E2=(128,128),
    b2=(128,),E4=(128,64),b4=(64,),q=(64,),b=(1,),centers=(16,30),scales=(16,30),r=(1,))
KEYS = tuple(SHAPES)
MODEL_KEYS = KEYS[:10]
HEAD_CEILING = 4_827_338_240
FEATURE_CEILING = 1_140_850_688
TOTAL_CEILING = 5_968_188_928
LITERAL_ANCHOR = '_VO_LITERAL = None'
_OWNERS = {}


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def owner(path,digest):
    key = str(path)
    if key not in _OWNERS:
        require(not path.is_symlink() and sha_bytes(path.read_bytes())==digest,'Immutable inert owner '+key)
        spec=importlib.util.spec_from_file_location('v29_fused_'+path.stem,path)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        _OWNERS[key]=module
    return _OWNERS[key]


def rewrite(original,edits):
    """Recorded per-occurrence inverse, including overlapping owner names."""
    changed,history=original,[]
    for old,new,count in edits:
        require(old and changed.count(old)==count,'Unique fused source splice '+old[:100])
        positions=[]; cursor=0
        while True:
            at=changed.find(old,cursor)
            if at<0:
                break
            positions.append(at); cursor=at+len(old)
        chunks=[]; cursor=shift=0
        for at in positions:
            chunks.extend((changed[cursor:at],new)); history.append((at+shift,old,new))
            cursor=at+len(old); shift+=len(new)-len(old)
        chunks.append(changed[cursor:]); changed=''.join(chunks)
    inverse=changed
    for at,old,new in reversed(history):
        require(inverse[at:at+len(new)]==new,'Exact recorded fused inverse position')
        inverse=inverse[:at]+old+inverse[at+len(new):]
    require(inverse==original,'Whole source inverse')
    return changed


def selected(source,name):
    nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name]
    require(len(nodes)==1,'Exact source-owned function '+name)
    n=nodes[0]
    return ''.join(source.splitlines(keepends=True)[n.lineno-1:n.end_lineno])


def source_parts():
    """Extract the unchanged feature arithmetic and exact displayed neural step."""
    feature=owner(FEATURE,FEATURE_SHA)
    require(not MODEL.is_symlink() and sha_bytes(MODEL.read_bytes())==MODEL_SHA,'Frozen neural arithmetic owner')
    fs=selected(FEATURE.read_text(),'feature_packet')
    feature_part=rewrite(fs,[('def feature_packet(','def _vo_features(',1),
        ('    require(','    _vo_require(',fs.count('    require(')),
        ('len(FEATURE_NAMES)','len(_VO_FEATURE_NAMES)',1)])
    ms=selected(MODEL.read_text(),'step')
    start=ms.index('    shifted = x-center\n')
    body=ms[start:]
    neural_part=rewrite(body,[
        ('torch.clamp(standardized, -8., 8.)','fnp.clip(standardized, constants[2:3], constants[3:4])',1),
        ('torch.clamp(e1, min=0.)','fnp.maximum(e1, constants[0:1])',1),
        ('torch.clamp(e2, min=0.)','fnp.maximum(e2, constants[0:1])',1),
        ('torch.clamp(e3, min=0.)','fnp.maximum(e3, constants[0:1])',1),
        ('torch.clamp(messages, -8., 8.)','fnp.clip(messages, constants[2:3], constants[3:4])',1),
        ('torch.log1p','fnp.log1p',3),('torch.cat','fnp.concatenate',2),('dim=1','axis=1',2),
        ('torch.exp','fnp.exp',2),('den0 = 1+exp0','den0 = constants[1:2]+exp0',1),
        ('den2 = 1+exp2','den2 = constants[1:2]+exp2',1),('    _finite(','    _vo_finite(',1),
    ])
    readout='''def _vo_readout(state, sigma, host, parameters, constants):
    head = state @ parameters['q'] + parameters['b']
    physical_scale = sigma*parameters['r']
    delta = physical_scale*head
    raw_terminal = host+delta
    _vo_finite((head, physical_scale, delta, raw_terminal), 'terminal readout')
    return delta, raw_terminal
'''
    # The readout's only deployment shape change is scalar b/r -> one-element
    # arrays; its exact grouping is checked against the frozen model source.
    forward=selected(MODEL.read_text(),'forward')
    expected="""    head = state @ parameters['q'] + parameters['b']
    physical_scale = sigma[15]*r
    delta = physical_scale*head
    raw_terminal = host[15]+delta
"""
    require(forward.count(expected)==1,'Exact frozen grouped terminal equation')
    return feature_part,neural_part,readout,dict(
        feature_owner_sha256=FEATURE_SHA,feature_source_inverse=True,
        original_feature_sha256=sha_bytes(fs.encode()),derived_feature_sha256=sha_bytes(feature_part.encode()),
        model_owner_sha256=MODEL_SHA,neural_arithmetic_source_inverse=True,
        original_neural_body_sha256=sha_bytes(body.encode()),derived_neural_body_sha256=sha_bytes(neural_part.encode()),
        unchanged_grouped_terminal_equation=True,scalar_b_r_deployment_shape=[1],
        fixed_feature_names=list(feature.FEATURE_NAMES))


SUPPORT = '''
def _vo_require(ok, message):
    if not ok:
        raise RuntimeError(message)


def _vo_finite(values, where):
    checks = []
    for value in values:
        _vo_require(value.dtype == fnp.float32, 'FP32 observer buffer: '+where)
        checks.append(bool(fnp.all(fnp.isfinite(value))))
    _vo_require(all(checks), 'Finite observer buffers: '+where)


def _vo_begin(self, mlp):
    _vo_require(mlp.width == 1024 and len(mlp.weights) == 16, 'Fixed qualified phase2 shape')
    _vo_require(getattr(self, '_vo_state', None) is None, 'Previous observer state retired')
    parameters = getattr(self, '_vo_parameters', None)
    if parameters is None:
        _vo_require(isinstance(_VO_LITERAL, dict) and tuple(_VO_LITERAL) == _VO_KEYS, 'Only sealed literal payload')
        parameters = {key:fnp.asarray(_VO_LITERAL[key], dtype=fnp.float32) for key in _VO_KEYS}
        self._vo_parameters = parameters
        self._vo_constants = fnp.asarray([0.0, 1.0, -8.0, 8.0], dtype=fnp.float32)
    _vo_require(tuple(parameters) == _VO_KEYS, 'Exactly13 immutable deployed payload arrays')
    for key in _VO_KEYS:
        _vo_require(parameters[key].shape == _VO_SHAPES[key] and parameters[key].dtype == fnp.float32,
                    'Exact deployed FP32 shape: '+key)
    constants = self._vo_constants
    _vo_require(constants.shape == (4,) and constants.dtype == fnp.float32, 'Typed constants array')
    _vo_finite(tuple(parameters.values())+(constants,), 'fixed payload and constants')
    _vo_require(bool(fnp.all(parameters['scales'] > constants[0:1]))
                and bool(fnp.all(parameters['r'] > constants[0:1])), 'Positive fixed scales/r')
    self._vo_state = fnp.zeros((1024, 64), dtype=fnp.float32)
    self._vo_terminal = None
    self._vo_next_layer = 0


def _vo_release(self):
    # Metadata-only cleanup; persistent literal arrays are not graph/state.
    for name in ('_vo_state', '_vo_terminal', '_vo_next_layer'):
        if hasattr(self, name):
            delattr(self, name)


def _vo_step(layer, x, state, W, parameters, center, scale, constants):
    _vo_require(1 <= layer <= 15 and x.shape == (1024,30) and state.shape == (1024,64)
                and W.shape == (1024,1024) and center.shape == scale.shape == (30,), 'Fixed observer step shapes')
    _vo_require(bool(fnp.all(scale > constants[0:1])), 'Positive frozen layer scale')
'''

EMITTER = '''
def _vo_emit(self, layer, last, mu, var, sigma, alpha, phi, gate, k3, k4, host, covariance_off, skew_plane, weights):
    _vo_require(layer == self._vo_next_layer and last == (layer == 15), 'Exactly16 original-order paid hooks')
    packet = _vo_features(fnp, layer, last, mu, var, sigma, alpha, phi, gate,
                          k3, k4, host, covariance_off, skew_plane, weights)
    p = self._vo_parameters
    if layer > 0:
        self._vo_state = _vo_step(layer, packet['features'], self._vo_state, weights, p,
                                  p['centers'][layer], p['scales'][layer], self._vo_constants)
    if last:
        delta, self._vo_terminal = _vo_readout(self._vo_state, packet['sigma'], packet['host'], p, self._vo_constants)
        del delta
    self._vo_next_layer += 1
    # A separately pinned qualification observer may export these observer-
    # owned/fresh arrays. It performs no FNP operator, and never receives a
    # physical pooled buffer or W. Production has no trace attachment.
    binding = getattr(self, '_vo_trace', None)
    if binding is not None:
        sink, context = binding
        before = (int(context.flops_used), len(context.op_log))
        sink(dict(packet, state=self._vo_state))
        _vo_require((int(context.flops_used), len(context.op_log)) == before, 'Trace sink remains zero-F')


def _vo_cached(features, sigma, host, weights, parameters, constants):
    # Qualification-only use of the EXACT deployed step/readout functions.
    # No physical estimator, labels, optimizer or alternate neural formula.
    _vo_require(features.shape == (16,1024,30) and sigma.shape == host.shape == (16,1024)
                and weights.shape == (16,1024,1024), 'Exact sealed cached trajectory')
    state = fnp.zeros((1024,64), dtype=fnp.float32)
    for layer in range(1,16):
        state = _vo_step(layer, features[layer], state, weights[layer], parameters,
                         parameters['centers'][layer], parameters['scales'][layer], constants)
    delta, raw_terminal = _vo_readout(state, sigma[15], host[15], parameters, constants)
    terminal = fnp.maximum(raw_terminal, constants[0:1])
    prefix = fnp.maximum(host[:15], constants[0:1])
    served = fnp.concatenate((prefix, terminal[None,:]), axis=0)
    _vo_finite((delta, raw_terminal, terminal, prefix, served), 'cached served output')
    return served
'''


def derive_template():
    """Return an unpopulated source template. Never synthesize a parameter value."""
    projected=owner(PROJECTED,PROJECTED_SHA); feature=owner(FEATURE,FEATURE_SHA)
    original,prior=projected.derive_source()
    require(sha_bytes(original.encode())==PHYSICAL_SHA,'Exact positive projected physical source')
    fp,step,readout,parts=source_parts()
    hook='''
            # Separately owned paid observer; all original physics remains live.
            _vo_emit(self, li, last, mu, var, sigma, alpha, phi, Phi,
                     D3, g4row, pk1v, C_off, D21, w32)
'''
    changed=rewrite(original,[
        (feature.INSERT.decode(),hook,1),
        ('        try:\n            return self._predict_core(mlp, budget)\n        finally:\n',
         '        try:\n            _vo_begin(self, mlp)\n            return self._predict_core(mlp, budget)\n        finally:\n            _vo_release(self)\n',1),
        ('                rows.append(pk1v if delta is None else pk1v + delta)',
         "                _vo_require(delta is None and self._vo_next_layer == 16, 'No old correction or missing observer layer')\n"
         '                rows.append(self._vo_terminal)',1),
        (projected.NEW_RETURN,
         "        _vo_output = fnp.maximum(fnp.stack(rows, axis=0), 0.0)\n"
         "        _vo_finite((_vo_output,), 'complete served output')\n"
         '        return _vo_output',1),
    ])
    additions='\n\n# New final-only observer; physical MIT attribution retained.\n'+LITERAL_ANCHOR+'\n'
    additions+='_VO_KEYS = '+repr(KEYS)+'\n_VO_SHAPES = '+repr(SHAPES)+'\n'
    additions+='_VO_FEATURE_NAMES = '+repr(feature.FEATURE_NAMES)+'\n\n'
    additions+=fp+'\n\n'+SUPPORT+step+'\n\n'+readout+'\n'+EMITTER
    require('_vo_' not in original and '_VO_' not in original,'No public namespace collisions')
    source=changed+additions
    require(source.count(LITERAL_ANCHOR)==1 and source[len(changed):]==additions,'Unique literal and appended helpers')
    return source,dict(schema='current_v29_temporal_fused_source_v1',parent=prior,parts=parts,
        physical_sha256=PHYSICAL_SHA,physical_edits_whole_source_inverse=True,
        source_template_sha256=sha_bytes(source.encode()),additions_sha256=sha_bytes(additions.encode()),
        source_bindings={str(PROJECTED):PROJECTED_SHA,str(FEATURE):FEATURE_SHA,str(MODEL):MODEL_SHA,
            **projected.authenticate()},fixed_allocated_parameters=44209,frozen_normalizer_literals=961,
        fixed_deployed_payload_values=45170,program_constants_values=4,scalar_b_r_deployment_shape=[1],
        head_ceiling_flops=HEAD_CEILING,feature_ceiling_flops=FEATURE_CEILING,total_ceiling_flops=TOTAL_CEILING,
        physical_feedback=False,parameter_choice=False,zero_F_capture_sink_unchanged=True,
        production_has_no_trace_attachment=True,estimator_executed=False,numerical_imports=False)


def literal_metadata(literal):
    require(isinstance(literal,dict) and tuple(literal)==KEYS,'Exact ordered sole-payload literal keys')
    report={}
    for key,shape in SHAPES.items():
        def walk(value,dims):
            if dims:
                require(isinstance(value,(list,tuple)) and len(value)==dims[0],'Exact nested literal shape '+key)
                return b''.join(walk(v,dims[1:]) for v in value)
            require(type(value) is float and math.isfinite(value),'Finite actual FP32 literal '+key)
            raw=struct.pack('<f',value)
            require(struct.unpack('<f',raw)[0]==value,'Literal is exactly one widened FP32 value '+key)
            return raw
        raw=walk(literal[key],shape)
        report[key]=dict(dtype='float32',shape=list(shape),raw_sha256=sha_bytes(raw))
    require(sum(math.prod(v) for v in SHAPES.values())==45170,'Exact deployed payload count')
    return report


def compose(literal):
    metadata=literal_metadata(literal); template,proof=derive_template()
    encoded=repr(literal)
    require(ast.literal_eval(encoded)==literal,'Stdlib literal exact value roundtrip')
    source=template.replace(LITERAL_ANCHOR,'_VO_LITERAL = '+encoded,1)
    require(source.replace('_VO_LITERAL = '+encoded,LITERAL_ANCHOR,1)==template,'Whole literal substitution inverse')
    return source,dict(proof,terminal_literal=metadata,candidate_sha256=sha_bytes(source.encode()),
        literal_source_sha256=sha_bytes(encoded.encode()),literal_roundtrip=True)


if __name__=='__main__':
    raise SystemExit('Inert source composer; only a separately admitted sealed-payload builder may create an estimator.')
