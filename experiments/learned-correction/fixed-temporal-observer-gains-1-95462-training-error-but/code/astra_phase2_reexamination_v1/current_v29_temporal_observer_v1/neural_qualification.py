"""Inert, target-free qualification of the fixed projected temporal observer.

No physical prediction, fit, label, CHECK feature or source-stage launch.
The CPU64 reference below is independent of model.step/model.forward.
"""
import argparse
import ast
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
import types
import weakref

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
AP = '/Users/matt/anaconda3/bin/python'
SITE = Path('/Users/matt/anaconda3/lib/python3.11/site-packages')
MODEL_SHA = '13a5e562561c324709a31ccfab4e53b1f7ac10079020459df098bd453802a138'
MODEL_OWNER_SHA = '59564d394f37758bb85805099b2f32ff9a6576825339b561c689e688f9c087ef'
SCHEMA = 'current_v29_temporal_observer_neural_qualification_v1'
CAP, RSS, DRIVER = 180.,12_000_000_000,16_000_000_000
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')
PINS = {
    HERE/'model.py':MODEL_OWNER_SHA,
    HERE/'projected_model.py':MODEL_SHA,
    HERE/'QUALIFICATION_DESIGN.md':'dd036422f9907e4c0134d2e444f3d969fb68bf15599b7f86a05f5c811ef90b89',
    HERE/'OBSERVER_PROJECTION_COMPATIBILITY.md':'261427beabb59dd814863ef3c73fcc34d42e818e2717ec05ab98b1a716099449',
    HERE/'TRAIN_DESIGN.md':'4c9124877e509570e4d6fcfb2d487dc33154267610a0295c775c6f1e1f60332f',
    NEW/'current_comp_cstar_proposal_v1/watch.py':'d3258f391ca501698e6c5f14a9dce21f28ae4e63d05731800ce6b2da8c94f23e',
}
EXTERNAL_PINS = {
    SITE/'torch/optim/adam.py':'78a0aa4a6b182fd586e40f964409cf3a9e2b935a1f8dcb62f31d1a4ca4c2e59a',
    SITE/'torch/optim/adamw.py':'3112d8799ef9ab608b855a5708087d5576971463923c41d4a6a227006871f551',
    SITE/'torch/__init__.py':'9801e75447c7f585545f989f8a21940b60e0c4cc888effb2f08e739664b4b904',
    SITE/'numpy/__init__.py':'22cd1535fa14d74ef6f457cca149ffdc80875f460be313b8f895273f78bc402e',
}
CAPTURE_FLAGS = ('capture-worker','capture-watch-source','capture-protocol','capture-watch-spec',
    'capture','capture-watch','projection-qualification','projection-qualification-watch')
FLAGS = ('source','watcher','protocol','watch-spec','model')+CAPTURE_FLAGS
TRAIN = (35,36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def valid(value):
    return isinstance(value,str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def sha(path, tick=None):
    p = Path(path)
    require(not p.is_symlink(), 'No symlink source/input')
    p = p.resolve()
    require((p.is_relative_to(P2) or p in EXTERNAL_PINS) and 'truthpost' not in p.parts
        and not (p.suffix in ('.npz','.npy') and 'truth' in p.name.lower()), 'No target access; exact source allowlist')
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            h.update(block)
            if tick is not None:
                tick()
    return h.hexdigest()


def read(path):
    p = Path(path)
    require(not p.is_symlink() and p.stat().st_size < 16*1024**2, 'Bounded regular JSON')
    return json.loads(p.read_text())


def write(path, value):
    def safe(x):
        if isinstance(x,float) and not math.isfinite(x):
            return dict(nonfinite_float=repr(x))
        if isinstance(x,dict):
            return {k:safe(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)):
            return [safe(v) for v in x]
        return x
    with Path(path).open('x') as stream:
        json.dump(safe(value),stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')


def load(path, name, digest):
    require(sha(path) == digest, 'Pinned inert source '+str(path))
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def add(mapping, path, digest, tick=None):
    path = str(Path(path).resolve())
    require(valid(digest) and mapping.get(path,digest) == digest and sha(path,tick) == digest, 'Immutable pin '+path)
    mapping[path] = digest


def roster():
    rows = ((None,'tiny_null','cpu'),(None,'tiny_nonzero_backward','cpu'),
        (None,'tiny_null','mps'),(None,'tiny_nonzero_backward','mps'),
        (None,'tiny_step7_backward','cpu'),(None,'tiny_step7_backward','mps'),
        (35,'cached_null','mps'),(58,'cached_null','mps'),
        (35,'cached_nonzero_backward','mps'),(58,'cached_nonzero_backward','mps'),
        (35,'cached_nonzero_backward','mps'))
    return [dict(graph_index=i,rank=r,kind=k,device=d,update=None) for i,(r,k,d) in enumerate(rows)]


def owners(args):
    require(all(valid(getattr(args,k.replace('-','_')+'_sha256',None)) for k in FLAGS)
        and args.model_sha256 == MODEL_SHA, 'Every actual source/prerequisite SHA, fixed projected observer')
    return {HERE/'neural_qualification.py':args.source_sha256,
        HERE/'watch_neural_qualification.py':args.watcher_sha256,
        HERE/'NEURAL_QUALIFICATION_PROTOCOL.md':args.protocol_sha256,
        HERE/'NEURAL_QUALIFICATION_WATCH_SPEC.md':args.watch_spec_sha256,**PINS,**EXTERNAL_PINS,
        HERE/'projection_capture_worker.py':args.capture_worker_sha256,
        HERE/'projection_capture_watch.py':args.capture_watch_source_sha256,
        HERE/'PROJECTION_CAPTURE_PROTOCOL.md':args.capture_protocol_sha256,
        HERE/'PROJECTION_CAPTURE_WATCH_SPEC.md':args.capture_watch_spec_sha256}


def environment():
    require(sys.executable == AP and sys.dont_write_bytecode and sys.flags.optimize == 0
        and os.getpriority(os.PRIO_PROCESS,0) >= 10 and all(os.environ.get(k) == '1' for k in BLAS)
        and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
        and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
        and os.environ.get('PYTORCH_MPS_LOW_WATERMARK_RATIO') == '0.1', 'Exact AP/nice10/six BLAS1/MPS frame')
    require(importlib.metadata.version('numpy') == '1.26.4'
        and importlib.metadata.version('torch').split('+')[0] == '2.7.0', 'Pinned numerical versions')
    require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules), 'All admission before numerical imports')


def admit(args, sources, inputs, weights, tick=None):
    """Complete direct projected TRAIN metadata admission, never array decoding."""
    for path,digest in owners(args).items():
        add(sources,path,digest,tick)
    adapter = load(HERE/'projected_model.py','qualified_projected_observer',MODEL_SHA)
    model,proof = adapter.load_model()
    for path,digest in model.authenticate().items():
        add(sources,path,digest,tick)
    require(model.ALLOCATED_PARAMETERS == 44209 and model.ZERO_INPUT_PARAMETERS == 384
        and model.DEPLOYED_LITERALS == 45170 and len(model.PARAMETER_KEYS) == 10,
        'Unchanged actual allocated/dead/deployed counts')
    model_source,_ = adapter.derive_source()
    tree = ast.parse(model_source)
    for fn in [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('step','forward')]:
        require(not any(isinstance(n,ast.Attribute) and n.attr in ('detach','detach_','copy_')
            for n in ast.walk(fn)), 'Live temporal recurrence, no hidden state detach or replacement')
    capture = load(HERE/'projection_capture_worker.py','observer_projected_capture_metadata',args.capture_worker_sha256)
    ca = types.SimpleNamespace(stage='train',source_sha256=args.capture_worker_sha256,
        watch_sha256=args.capture_watch_source_sha256,protocol_sha256=args.capture_protocol_sha256,
        watch_spec_sha256=args.capture_watch_spec_sha256,adapter_sha256=capture.ADAPTER_SHA,
        adapter_protocol_sha256=capture.ADAPTER_PROTOCOL_SHA,
        qualification_sha256=args.projection_qualification_sha256,
        qualification_watch_sha256=args.projection_qualification_watch_sha256,
        train_sha256=None,train_watch_sha256=None)
    inherited = capture.prerequisites(ca,sources,inputs,tick)
    helpers = capture._helpers()
    train,ts = helpers['chain'](HERE/'train_projection_capture_v1',args.capture_sha256,inputs,2460.,tick)
    watched,ws = helpers['chain'](HERE/'train_projection_capture_watch_v1',args.capture_watch_sha256,inputs,2460.,tick)
    require(train['prerequisite_seals'] == inherited['seals'], 'Actual projected qualification admitted by captured TRAIN')
    capture.verify_expanded(train,'train',ca,inherited['qualification'],tick)
    helpers['watcher_contract'](watched,train,ts,TRAIN,'train')
    helpers['inventory'](train,sources,inputs,tick); helpers['inventory'](watched,sources,inputs,tick)
    require(watched['projected_capture_verified'] and train['ranks'] == list(TRAIN)
        and train['downstream_training_authorized'] is False, 'Capture evidence, not a training permission')
    for rank in (35,58):
        item = train['fixture_bindings'][str(rank)]
        add(weights,item['path'],item['sha256'],tick)
    require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules), 'All actual seals/sources before any array')
    return dict(model=model,model_proof=proof,capture=train,
        prerequisite_seals=dict(**inherited['seals'],capture=ts,capture_watch=ws),
        capture_worker_pid=train['worker_pid'],capture_candidate_sha256=train['candidate_sha256'])


def verify_result(record, args, admitted, tick=None):
    require(record['schema'] == SCHEMA and record['stage'] == 'qualify'
        and all(record.get(k) is True for k in ('passed','completed','hashes_unchanged','sources_unchanged',
            'execution_integrity_passed','all_tiny_precision_gates_passed','all_cached_execution_gates_passed',
            'all_ten_parameter_blocks_qualified','dead384_data_and_moments_zero','all_graphs_retired',
            'all_parameter_and_optimizer_owners_retired','all_inputs_normalizers_parameters_unchanged',
            'projected_null_all16_byte_exact','cached35_repeat_byte_exact','independent_CPU64_reference',
            'no_native_gradient_claim','original_failed_physical_qualifications_preserved'))
        and record['graphs_completed'] == record['retired_call_count'] == record['neural_forward_evaluations'] == 11
        and record['tiny_full_rollouts'] == 4 and record['tiny_single_steps'] == 2 and record['cached_rollouts'] == 5
        and record['backward_calls'] == 7 and record['zero_lr_adamw_steps'] == 3
        and record['qualification_optimizer_fixture_steps'] == 1
        and record['physical_model_calls'] == record['model_parameter_updates'] == record['parameter_updates'] == 0
        and record['native_gradient_qualified'] is False and record['training_targets_accessed'] is False
        and record['validation_targets_accessed'] is False and record['truth_accessed'] is False
        and record['score_measured'] is False and record['downstream_training_authorized'] is False
        and record['prerequisite_seals'] == admitted['prerequisite_seals']
        and record['model_source_proof'] == admitted['model_proof'], 'Complete exact target-free qualification scope')
    require(record['roster'] == roster() and len(record['graph_records']) == 11, 'Exact fixed eleven evaluations')
    precision = record['tiny_full_precision']
    require(set(precision['blocks']) == {'gradient_'+k for k in admitted['model'].PARAMETER_KEYS}
        and all(math.isfinite(v['reference_norm']) and v['reference_norm']>1e-12
            and math.isfinite(v['relative_L2_error']) and 0<=v['relative_L2_error']<=.05
            for v in precision['blocks'].values())
        and len(precision['states']) == 16
        and all(v['layer']==i and 0<=v['max_absolute_error']<=v['bound']
            for i,v in enumerate(precision['states']))
        and all(0<=v<=1e-5 for v in precision['terminal_max_absolute_errors'].values())
        and record['tiny_step_precision']['reference_norm']>0
        and 0<=record['tiny_step_precision']['relative_L2_error']<=.05,
        'Actual reported independent-reference per-block/state/terminal/temporal gates')
    for entry,packet in zip(roster(),record['graph_records']):
        require(all(packet[k] == v for k,v in entry.items()) and packet['graph_retired'] is True,
                'Actual complete per-evaluation retirement')
        path = HERE/'neural_qualification_v1'/f"graph_{entry['graph_index']:04d}_complete.json"
        require(read(path) == packet and sha(path,tick) == record['artifacts'][str(path)], 'Actual linked graph marker')
    for path,digest in owners(args).items():
        require(record['source_bindings'].get(str(path)) == digest, 'Same reviewed source and data authority')


def rawsha(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def asarray(tensor):
    return tensor.detach().cpu().contiguous().numpy().copy()


def finite(torch, values, label):
    require(all(bool(torch.isfinite(v).all().item()) for v in values), 'Finite independent reference '+label)


def reference_step(torch, layer, x, H, W, p, center, scale):
    """Independent CPU64 displayed circuit, no call into model arithmetic."""
    shifted = x-center; standardized = shifted/scale; X = torch.clamp(standardized,-8.,8.)
    gate = X[:,5:6]
    S = H @ p['M'] + p['b_M']; S1,S2,S3 = S[:,:16],S[:,16:32],S[:,32:48]
    W2 = W*W; W3 = W2*W; W4 = W2*W2; W6 = W3*W3
    m1 = W.T @ S1; m2 = W2.T @ S2; m3 = W3.T @ S3
    sq1,sq2,sq3 = S1*S1,S2*S2,S3*S3
    e1 = W2.T @ sq1; e2 = W4.T @ sq2; e3 = W6.T @ sq3
    pos1,pos2,pos3 = torch.clamp(e1,min=0.),torch.clamp(e2,min=0.),torch.clamp(e3,min=0.)
    z1,z2,z3 = torch.log1p(pos1),torch.log1p(pos2),torch.log1p(pos3)
    gm1,gm2,gm3 = gate*m1,gate*m2,gate*m3
    messages = torch.cat((gm1,gm2,gm3,z1,z2,z3),dim=1)
    clipped = torch.clamp(messages,-8.,8.); V = torch.cat((X,clipped),dim=1)
    a0 = V @ p['E0'] + p['b0']; neg0 = -a0; exp0 = torch.exp(neg0); den0 = 1+exp0; U1 = a0/den0
    a2 = U1 @ p['E2'] + p['b2']; neg2 = -a2; exp2 = torch.exp(neg2); den2 = 1+exp2; U2 = a2/den2
    answer = U2 @ p['E4'] + p['b4']
    finite(torch,(x,H,W,center,scale,shifted,standardized,X,gate,S,W2,W3,W4,W6,m1,m2,m3,
        sq1,sq2,sq3,e1,e2,e3,pos1,pos2,pos3,z1,z2,z3,gm1,gm2,gm3,messages,clipped,V,
        a0,neg0,exp0,den0,U1,a2,neg2,exp2,den2,U2,answer),str(layer))
    return answer


def reference_forward(torch, data, p):
    H = torch.zeros((8,64),dtype=torch.float64,device='cpu'); states = [H]
    for layer in range(1,16):
        H = reference_step(torch,layer,data['features'][layer],H,data['weights'][layer],p,
                           data['centers'][layer],data['scales'][layer])
        states.append(H)
    head = H @ p['q'] + p['b']; physical_scale = data['sigma'][15]*data['r']
    delta = physical_scale*head; raw = data['host'][15]+delta
    terminal = torch.clamp(raw,min=0.); served = torch.cat((torch.clamp(data['host'][:15],min=0.),terminal[None,:]),dim=0)
    finite(torch,(head,physical_scale,delta,raw,terminal,served),'terminal')
    return dict(served=served,delta=delta,state=H,states=tuple(states))


def initial_values(np, model):
    rng = np.random.Generator(np.random.PCG64DXSM(93091229)); result = {}
    for key,fan in (('M',64),('E0',126),('E2',128),('E4',128)):
        result[key] = rng.uniform(-1/math.sqrt(fan),1/math.sqrt(fan),size=model.SHAPES[key]).astype(np.float32)
    for key in model.PARAMETER_KEYS:
        if key not in result:
            result[key] = np.zeros(model.SHAPES[key],dtype=np.float32)
    return result


def nonzero_head(np, values):
    return dict(values,q=(.02*np.sin(.37*(np.arange(64,dtype=np.float64)+1))).astype(np.float32))


def normalizers(np, feature_arrays):
    pool = np.stack(feature_arrays,axis=0).astype(np.float64)
    centers = pool.mean(axis=(0,2)).astype(np.float32)
    scales = (pool.std(axis=(0,2))+1e-6).astype(np.float32)
    require(np.isfinite(centers).all() and np.isfinite(scales).all() and (scales>0).all(), 'Finite TRAIN-only normalization')
    for X in feature_arrays:
        normalized = np.clip((X-centers[:,None,:])/scales[:,None,:],-8,8)
        require(np.all(normalized[:,:,:3] == 0), '384 zero-input E0 rows remain structurally dead')
    return centers,scales


def tiny_data(np):
    k = np.arange(20,dtype=np.float64)[:,None,None,None]
    l = np.arange(16,dtype=np.float64)[None,:,None,None]
    i = np.arange(8,dtype=np.float64)[None,None,:,None]
    j = np.arange(30,dtype=np.float64)[None,None,None,:]
    X = (.01*(j+1)+.2*np.sin(.13*(k+1)+.17*(l+1)+.19*(i+1)+.23*(j+1))).astype(np.float32)
    X[:,:,:,0] = (2*np.arange(16,dtype=np.float64)/15-1).astype(np.float32)[None,:,None]
    X[:,:,:,1] = (np.arange(16,dtype=np.float64)/15).astype(np.float32)[None,:,None]
    X[:,:,:,2] = np.float32(1)
    centers,scales = normalizers(np,list(X))
    z = np.clip((X[0]-centers[:,None,:])/scales[:,None,:],-8,8)
    require(np.all(np.std(z[:,:,3:].astype(np.float64),axis=(0,1))>0)
        and float(np.std(z[:,:,5].astype(np.float64)))>0, 'Nonconstant real standardized synthetic channels/gate')
    layer = np.arange(16,dtype=np.float64)[:,None,None]
    row = np.arange(8,dtype=np.float64)[None,:,None]
    col = np.arange(8,dtype=np.float64)[None,None,:]
    W = (math.sqrt(2/8)*np.sin(.11*(layer+1)+.29*(row+1)+.31*(col+1))).astype(np.float32)
    sigma = (1+.05*np.sin(.17*(layer[:,:,0]+1)+.23*(row[:,:,0]+1))).astype(np.float32)
    host = (2+.1*np.cos(.19*(layer[:,:,0]+1)+.29*(row[:,:,0]+1))).astype(np.float32)
    return dict(features=X[0].copy(),sigma=sigma,host=host,weights=W,centers=centers,scales=scales,r=np.array(1,dtype=np.float32))


def cached_data(np, capture, rank, centers, scales):
    call = TRAIN.index(rank); record = capture['capture_records'][call]
    with np.load(record['path'],allow_pickle=False) as saved:
        arrays = {k:saved[k].copy() for k in saved.files}
    require(set(arrays) == set(record['arrays']) and len(arrays) == 64, 'Exactly64 original raw capture arrays')
    for key,value in arrays.items():
        meta = record['arrays'][key]
        require(str(value.dtype) == meta['dtype'] and list(value.shape) == meta['shape']
            and rawsha(value) == meta['raw_sha256'] and np.isfinite(value).all(), 'Authenticated actual raw capture '+key)
    weights = np.load(capture['fixture_bindings'][str(rank)]['path'],allow_pickle=False)
    require(weights.dtype == np.float32 and weights.shape == (16,1024,1024) and np.isfinite(weights).all()
        and rawsha(weights) == capture['fixture_bindings'][str(rank)]['raw_sha256'], 'Same canonical fixed physical weights')
    return dict(features=np.stack([arrays[f'layer_{l:02d}_features'] for l in range(16)]),
        sigma=np.stack([arrays[f'layer_{l:02d}_sigma'] for l in range(16)]),
        host=np.stack([arrays[f'layer_{l:02d}_host'] for l in range(16)]),weights=weights,
        centers=centers,scales=scales,r=np.array(1,dtype=np.float32))


def train_normalizers(np, capture):
    features = []
    for record in capture['capture_records']:
        with np.load(record['path'],allow_pickle=False) as source:
            rows = []
            for l in range(16):
                key = f'layer_{l:02d}_features'; value = source[key].copy(); meta = record['arrays'][key]
                require(value.dtype == np.float32 and value.shape == (1024,30) and np.isfinite(value).all()
                    and rawsha(value) == meta['raw_sha256'], 'Only authenticated20 TRAIN features for normalization')
                rows.append(value)
            features.append(np.stack(rows))
    require(len(features) == 20, 'Exactly20 TRAIN normalizer population')
    return normalizers(np,features)


def tensor_parameters(torch, values, device, dtype, trainable):
    return {k:torch.tensor(v,dtype=dtype,device=device,requires_grad=trainable) for k,v in values.items()}


def gradients(np, parameters, *, nontrivial=False):
    answer,norms = {},{}
    for key,value in parameters.items():
        require(value.grad is not None and value.grad.grad_fn is None and not value.grad.requires_grad, 'Owned detached gradient '+key)
        g = asarray(value.grad); require(np.isfinite(g).all(), 'Finite gradient '+key)
        if key == 'E0':
            require(np.all(g[:3] == 0), 'Exact384 zero data gradients')
        connected = g[3:] if key == 'E0' else g
        norm = float(np.linalg.norm(connected.astype(np.float64)))
        require(not nontrivial or norm>1e-12, 'Nonzero connected gradient block '+key)
        answer[key],norms[key] = g,norm
    return answer,norms


def optimizer_state(np, torch, optimizer, parameters, step):
    require(len(optimizer.state) == 10 and len(optimizer.param_groups) == 1
        and optimizer.param_groups[0]['lr'] == 0. and optimizer.param_groups[0]['foreach'] is False
        and optimizer.param_groups[0]['weight_decay'] == 2e-5, 'Exact persistent zero-LR AdamW ownership')
    report = {}
    for key,p in parameters.items():
        state = optimizer.state[p]
        require(set(state) == {'step','exp_avg','exp_avg_sq'} and int(state['step'].item()) == step,
                'All10 actual optimizer state counts')
        for value in state.values():
            require(bool(torch.isfinite(value).all().item()), 'Finite actual optimizer state')
        if key == 'E0':
            require(bool((state['exp_avg'][:3] == 0).all().item())
                and bool((state['exp_avg_sq'][:3] == 0).all().item()), 'Dead384 moments stay exact zero')
        report[key] = dict(step=step,exp_avg_sha256=rawsha(asarray(state['exp_avg'])),
            exp_avg_sq_sha256=rawsha(asarray(state['exp_avg_sq'])))
    return report


def evaluate(np, torch, model, spec, source, values, *, persistent=None, optimizer=None, optimizer_step=None):
    """One graph only; returns detached data plus weakrefs, never a tensor owner."""
    device = spec['device']; dtype = torch.float64 if device == 'cpu' else torch.float32
    backward = 'backward' in spec['kind']; single = 'step7' in spec['kind']
    parameters = persistent if persistent is not None else tensor_parameters(torch,values,device,dtype,backward and not single)
    parameter_ids = {id(v) for v in parameters.values()}
    before_parameters = {k:rawsha(asarray(v)) for k,v in parameters.items()}
    data = {k:torch.tensor(v,dtype=dtype,device=device) for k,v in source.items()}
    before_inputs = {k:rawsha(asarray(v)) for k,v in data.items()}
    refs = [weakref.ref(v) for v in data.values()]
    if persistent is None:
        refs += [weakref.ref(v) for v in parameters.values()]
    saved = []
    def pack(tensor):
        if id(tensor) not in parameter_ids:
            saved.append(weakref.ref(tensor))
        return tensor
    def unpack(tensor):
        return tensor
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    n = source['features'].shape[1]
    cotangent = torch.tensor((.25+.75*np.sin(.41*(np.arange(n,dtype=np.float64)+1))).astype(np.float32),dtype=dtype,device=device)
    packet,arrays = dict(spec),{}
    driver_samples = []
    def driver_sample():
        if device == 'mps':
            torch.mps.synchronize()
            used = int(torch.mps.driver_allocated_memory())
            require(used<DRIVER, 'Actual driver16GB before graph/cache retirement')
            driver_samples.append(used)
    driver_sample()
    with torch.set_grad_enabled(backward), torch.autograd.graph.saved_tensors_hooks(pack,unpack):
        if single:
            ii = np.arange(8,dtype=np.float64)[:,None]; pp = np.arange(64,dtype=np.float64)[None,:]
            H = torch.tensor((.1*np.sin(.17*(ii+1)+.23*(pp+1))).astype(np.float32),dtype=dtype,device=device,requires_grad=True)
            refs.append(weakref.ref(H))
            state = (reference_step(torch,7,data['features'][7],H,data['weights'][7],parameters,data['centers'][7],data['scales'][7])
                if device == 'cpu' else model.step(7,data['features'][7],H,data['weights'][7],parameters,data['centers'][7],data['scales'][7]))
            driver_sample()
            loss = ((state @ parameters['q'])*cotangent).mean()
            require(bool(torch.isfinite(loss).item()), 'Finite fixed single-step readout')
            loss.backward()
            driver_sample()
            hgrad = asarray(H.grad); require(np.isfinite(hgrad).all() and float(np.linalg.norm(hgrad.astype(np.float64)))>0,
                'Live nonzero finite H-input gradient')
            arrays.update(state=asarray(state),H_gradient=hgrad)
            refs.extend((weakref.ref(state),weakref.ref(loss),weakref.ref(H.grad)))
            packet['H_gradient_nonzero'] = True
        else:
            output = (reference_forward(torch,data,parameters) if device == 'cpu' else
                model.forward(data['features'],data['sigma'],data['host'],data['weights'],parameters,
                    data['centers'],data['scales'],data['r'],retain_tiny_states=n == 8))
            driver_sample()
            arrays.update(served=asarray(output['served']),delta=asarray(output['delta']),state=asarray(output['state']))
            if n == 8:
                arrays['states'] = np.stack([asarray(v) for v in output['states']])
            projected = np.maximum(source['host'].astype(np.float64 if device == 'cpu' else np.float32),0.)
            require(arrays['served'][:15].tobytes() == projected[:15].tobytes(), 'Unchanged projected prefix bytes')
            if not backward:
                require(arrays['served'].tobytes() == projected.tobytes() and np.all(arrays['delta'] == 0), 'Exact all16 projected null')
                packet['null_all16_byte_exact'] = True
            if backward:
                if n == 8:
                    margin = float(np.min(np.abs(source['host'][15].astype(np.float64)+arrays['delta'].astype(np.float64))))
                    require(margin >= 1e-4, 'Fixed tiny readout away from clamp ties')
                    packet['tiny_unclamped_margin'] = margin
                loss = (output['served'][15]*cotangent).mean()
                require(bool(torch.isfinite(loss).item()), 'Finite fixed terminal readout')
                loss.backward()
                driver_sample()
                grad,packet['gradient_norms'] = gradients(np,parameters,nontrivial=n == 8 and device == 'cpu')
                arrays.update({'gradient_'+k:v for k,v in grad.items()})
                refs.append(weakref.ref(loss)); refs.extend(weakref.ref(p.grad) for p in parameters.values())
                packet['loss'] = float(loss.detach().cpu())
                if optimizer is not None:
                    optimizer.step()
                    driver_sample()
                    packet['optimizer_states'] = optimizer_state(np,torch,optimizer,parameters,optimizer_step)
                    packet['zero_lr_adamw_step'] = optimizer_step
            refs.extend(weakref.ref(v) for v in (output['served'],output['delta'],output['state'],*output['states']))
    if device == 'mps':
        torch.mps.synchronize()
    require(before_parameters == {k:rawsha(asarray(v)) for k,v in parameters.items()}, 'All parameter bytes unchanged')
    require(before_inputs == {k:rawsha(asarray(v)) for k,v in data.items()}
        and all(not v.requires_grad and v.grad is None for v in data.values()), 'Actual inputs/normalizers frozen')
    packet.update(parameter_hashes=before_parameters,input_hashes=before_inputs,inputs_normalizers_parameters_unchanged=True,
        backward=backward,reference64=device == 'cpu',saved_tensor_references=len(saved),finite=True,
        live_mps_driver_samples=driver_samples,maximum_live_mps_driver_bytes=max(driver_samples,default=0))
    for p in parameters.values():
        p.grad = None
    return packet,arrays,refs+saved


def precision_gates(np, reference, actual, *, single=False):
    if single:
        g = reference['H_gradient'].astype(np.float64); a = actual['H_gradient'].astype(np.float64)
        norm = float(np.linalg.norm(g)); error = float(np.linalg.norm(a-g))/max(norm,1e-12)
        require(norm>0 and error<=.05, 'Original5% live H-gradient precision gate')
        return dict(reference_norm=norm,relative_L2_error=error)
    errors = {}
    for key in [k for k in reference if k.startswith('gradient_')]:
        g,a = reference[key].astype(np.float64),actual[key].astype(np.float64)
        if key == 'gradient_E0':
            require(np.all(g[:3] == 0) and np.all(a[:3] == 0), '384 exact zero data-gradient rows')
            g,a = g[3:],a[3:]
        norm = float(np.linalg.norm(g)); err = float(np.linalg.norm(a-g))/max(norm,1e-12)
        require(norm>1e-12 and err<=.05, 'Original per-block5% precision gate '+key)
        errors[key] = dict(reference_norm=norm,relative_L2_error=err)
    require(len(errors) == 10, 'No combined norm conceals a bad small block')
    state_errors = []
    for l,(cpu,mps) in enumerate(zip(reference['states'],actual['states'])):
        err = float(np.max(np.abs(mps.astype(np.float64)-cpu)))
        bound = 1e-5*max(1.,float(np.max(np.abs(cpu))))
        require(err<=bound, 'Original each-layer H forward precision gate')
        state_errors.append(dict(layer=l,max_absolute_error=err,bound=bound))
    terminal = {}
    for key in ('delta','served'):
        cpu = reference[key] if key == 'delta' else reference[key][15]
        mps = actual[key] if key == 'delta' else actual[key][15]
        err = float(np.max(np.abs(mps.astype(np.float64)-cpu)))
        require(err<=1e-5, 'Original terminal vector1e-5 gate '+key); terminal[key] = err
    return dict(blocks=errors,states=state_errors,terminal_max_absolute_errors=terminal)


def disposable_decay(np, torch, values):
    parameter = torch.tensor(values['E0'],dtype=torch.float32,device='mps',requires_grad=True)
    before = asarray(parameter); parameter.grad = torch.zeros_like(parameter)
    expected = parameter.detach().clone(); expected.mul_(1-1e-3*2e-5)
    optimizer = torch.optim.AdamW([parameter],lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=2e-5,foreach=False)
    optimizer.step(); torch.mps.synchronize()
    state = optimizer.state[parameter]
    require(asarray(parameter).tobytes() == asarray(expected).tobytes()
        and int(state['step'].item()) == 1 and bool((state['exp_avg'] == 0).all().item())
        and bool((state['exp_avg_sq'] == 0).all().item()), 'Exact disposable ordinary FP32 decay and zero moments')
    report = dict(passed=True,completed=True,learning_rate=.001,weight_decay=2e-5,
        steps=1,zero_gradient=True,exact_installed_FP32_multiplication=True,
        visible_parameter_change=before.tobytes() != asarray(parameter).tobytes(),
        before_sha256=rawsha(before),after_sha256=rawsha(asarray(parameter)),model_parameter_updates=0)
    refs = [weakref.ref(parameter),weakref.ref(parameter.grad),weakref.ref(expected),weakref.ref(optimizer)]
    refs.extend(weakref.ref(v) for v in state.values())
    return report,refs


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run',action='store_true')
    for key in FLAGS:
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    environment(); began = time.monotonic(); folder = HERE/'neural_qualification_v1'
    require(not folder.exists() and not HERE.is_symlink(), 'Exclusive never-resumed neural qualification')
    folder.mkdir()
    result = dict(schema=SCHEMA,stage='qualify',passed=False,completed=False,source_bindings={},input_bindings={},
        weight_bindings={},artifacts={},graph_records=[],roster=roster(),graphs_completed=0,retired_call_count=0,
        neural_forward_evaluations=0,backward_calls=0,zero_lr_adamw_steps=0,qualification_optimizer_fixture_steps=0,
        physical_model_calls=0,model_parameter_updates=0,parameter_updates=0,native_gradient_qualified=False,
        no_native_gradient_claim=True,training_targets_accessed=False,validation_targets_accessed=False,
        truth_accessed=False,score_measured=False,downstream_training_authorized=False)
    torch = optimizer = persistent = null_parameters = None
    def guard():
        require(time.monotonic()-began<CAP and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<RSS, 'Whole180s/host12GB')
        if torch is not None and torch.backends.mps.is_available():
            require(torch.mps.driver_allocated_memory()<DRIVER, 'Actual MPS driver16GB')
    def json_artifact(name,value):
        path=folder/name; write(path,value); result['artifacts'][str(path)]=sha(path); guard(); return path
    def save(name,arrays):
        path=folder/name
        with path.open('xb') as stream:
            np.savez(stream,**arrays)
        result['artifacts'][str(path)]=sha(path); guard()
        return {k:dict(dtype=str(v.dtype),shape=list(v.shape),raw_sha256=rawsha(v)) for k,v in arrays.items()}
    old_handlers = {s:signal.signal(s,lambda *unused: (_ for _ in ()).throw(RuntimeError('Neural qualification interrupted')))
                    for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        admitted=admit(args,result['source_bindings'],result['input_bindings'],result['weight_bindings'],guard)
        result.update(prerequisite_seals=admitted['prerequisite_seals'],model_source_proof=admitted['model_proof'],
            capture_candidate_sha256=admitted['capture_candidate_sha256'],capture_worker_pid=admitted['capture_worker_pid'])
        json_artifact('model_call_roster.json',dict(schema='current_v29_observer_neural_qualification_roster_v1',
            stage='qualify',initialization_seed=93091229,entries=roster()))
        require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules), 'Full admission before numerics')
        import numpy as np
        torch = admitted['model'].runtime(); model=admitted['model']
        require(Path(np.__file__).resolve()==SITE/'numpy/__init__.py' and Path(torch.__file__).resolve()==SITE/'torch/__init__.py', 'Actual AP module origins')
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        require(torch.get_num_threads()==torch.get_num_interop_threads()==1 and torch.backends.mps.is_available(), 'Sole MPS/Torch1 runtime')
        values=initial_values(np,model); fixture=nonzero_head(np,values)
        checked=model.initialize('cpu',trainable=False)
        require(all(asarray(checked[k]).tobytes()==v.tobytes() for k,v in values.items()), 'Exact fresh prescribed initialization stream')
        del checked; gc.collect()
        save('initial_parameters.npz',values); save('nonzero_head_fixture.npz',{'q':fixture['q']})
        tiny=tiny_data(np); save('tiny_inputs.npz',tiny)
        centers,scales=train_normalizers(np,admitted['capture'])
        save('train_feature_normalizers.npz',dict(centers=centers,scales=scales,r=np.array(1,dtype=np.float32)))
        cpu_full=cpu_step=repeat=None
        for spec in roster():
            index=spec['graph_index']; guard()
            if index==6:
                decay,refs=disposable_decay(np,torch,values); gc.collect(); torch.mps.empty_cache(); torch.mps.synchronize()
                require(all(r() is None for r in refs), 'Disposable optimizer fixture fully retired')
                json_artifact('disposable_decay.json',decay); result['qualification_optimizer_fixture_steps']=1
                null_parameters=tensor_parameters(torch,values,'mps',torch.float32,False)
            if index==8:
                refs=[weakref.ref(v) for v in null_parameters.values()]; null_parameters=None; gc.collect()
                require(all(r() is None for r in refs), 'Null-parameter owners retired before nonzero fixture')
                persistent=tensor_parameters(torch,fixture,'mps',torch.float32,True)
                optimizer=torch.optim.AdamW(list(persistent.values()),lr=0.,betas=(.9,.999),eps=1e-8,weight_decay=2e-5,foreach=False)
            source=tiny if index<6 else cached_data(np,admitted['capture'],spec['rank'],centers,scales)
            parameters=null_parameters if index in (6,7) else persistent if index>=8 else None
            packet,arrays,refs=evaluate(np,torch,model,spec,source,
                values if 'null' in spec['kind'] else fixture,persistent=parameters,
                optimizer=optimizer if index>=8 else None,optimizer_step=index-7 if index>=8 else None)
            # All tensors are local to evaluate except the explicitly declared
            # persistent parameters/optimizer. Only detached NP arrays survive.
            del source,parameters; gc.collect()
            if spec['device']=='mps':
                torch.mps.synchronize(); torch.mps.empty_cache(); torch.mps.synchronize()
            require(all(r() is None for r in refs), 'Every graph/input/power/state reference actually retired')
            if index==1:
                cpu_full=arrays
            elif index==3:
                result['tiny_full_precision']=precision_gates(np,cpu_full,arrays)
            elif index==4:
                cpu_step=arrays
            elif index==5:
                result['tiny_step_precision']=precision_gates(np,cpu_step,arrays,single=True)
            elif index==8:
                repeat={k:v.copy() for k,v in arrays.items() if k=='served' or k.startswith('gradient_')}
            elif index==10:
                require(all(arrays[k].tobytes()==v.tobytes() for k,v in repeat.items()), 'Exact cached35 output and every gradient repeat')
                result['cached35_repeat_byte_exact']=True
            if index in (6,7):
                original=admitted['capture']['trials'][TRAIN.index(spec['rank'])]
                require(rawsha(arrays['served'])==original['output_bytes_sha256'], 'Actual projected captured all16 null bytes')
            packet['saved_arrays']=save(f'graph_{index:04d}_arrays.npz',arrays)
            packet.update(passed=True,completed=True,graph_retired=True,host_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                mps_driver_bytes=int(torch.mps.driver_allocated_memory()) if spec['device']=='mps' else 0,elapsed_s=time.monotonic()-began)
            guard(); path=json_artifact(f'graph_{index:04d}_complete.json',packet)
            result['graph_records'].append(packet); result['graphs_completed']+=1; result['retired_call_count']+=1; result['neural_forward_evaluations']+=1
            result['backward_calls']+=int(packet['backward']); result['zero_lr_adamw_steps']+=int(index>=8)
            print(json.dumps(dict(event='retired',call=index,rank=spec['rank'],kind=spec['kind'],device=spec['device'],
                complete_sha256=sha(path),host_peak_rss_bytes=packet['host_peak_rss_bytes'],mps_driver_bytes=packet['mps_driver_bytes'])),flush=True)
        refs=[weakref.ref(optimizer)]+[weakref.ref(v) for v in persistent.values()]
        refs.extend(weakref.ref(v) for state in optimizer.state.values() for v in state.values())
        optimizer=persistent=None; gc.collect(); torch.mps.empty_cache(); torch.mps.synchronize()
        require(all(r() is None for r in refs), 'All persistent parameter and optimizer owners retired')
        require(result['backward_calls']==7 and result['zero_lr_adamw_steps']==3 and result['retired_call_count']==11, 'Exact complete fixed qualification')
        for group in ('source_bindings','input_bindings','weight_bindings','artifacts'):
            for path,digest in result[group].items():
                require(sha(path,guard)==digest, 'All final source/input/artifact bytes unchanged')
        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,execution_integrity_passed=True,
            tiny_full_rollouts=4,tiny_single_steps=2,cached_rollouts=5,all_tiny_precision_gates_passed=True,
            all_cached_execution_gates_passed=True,all_ten_parameter_blocks_qualified=True,dead384_data_and_moments_zero=True,
            all_graphs_retired=True,all_parameter_and_optimizer_owners_retired=True,all_inputs_normalizers_parameters_unchanged=True,
            projected_null_all16_byte_exact=True,independent_CPU64_reference=True,original_failed_physical_qualifications_preserved=True)
        guard()
    except BaseException:
        result.update(passed=False,completed=False,error=traceback.format_exc())
    finally:
        optimizer=persistent=null_parameters=None; gc.collect()
        if torch is not None and torch.backends.mps.is_available():
            torch.mps.synchronize(); torch.mps.empty_cache()
        for sig,handler in old_handlers.items():
            signal.signal(sig,handler)
    result.update(elapsed_s=time.monotonic()-began,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    if result['elapsed_s']>=CAP or result['peak_rss_bytes']>=RSS:
        result.update(passed=False,completed=False)
    write(folder/'result.json',result); rh=sha(folder/'result.json')
    ok=bool(result['passed'] and time.monotonic()-began<CAP)
    write(folder/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh)); ch=sha(folder/'completion.json')
    elapsed=time.monotonic()-began; ok=bool(ok and elapsed<CAP)
    write(folder/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,completion_sha256=ch,elapsed_s=elapsed))
    sha(folder/'finalization.json'); ok=bool(ok and time.monotonic()-began<CAP)
    print(json.dumps(dict(passed=ok,completed=ok,retired_call_count=result['retired_call_count'],error=result.get('error'))),flush=True)
    return 0 if ok else 1


if __name__=='__main__':
    raise SystemExit(main())
