"""Inert exact-native, target-free entry8 -> mean9 local adjoint qualifier.

This reuses only authenticated source/packet/lifecycle infrastructure. It never
calls an old information experiment, supply reader, fit, or submission. Every
call restores the same admitted entry8 and H7; subsequent states are recomputed
by that call. This is a LOCAL derivative qualification, not a full-P rollout.
The unchanged call must match every archived checkpoint. Changed-P calls never
substitute an archived downstream state. Native arithmetic and all current
heads remain, apart from the caller's explicitly priced parameter perturbation.
"""
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import types
import weakref

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
TRACE = NEW/'current_comp_join_trace_information_v1'
CELL = NEW/'current_comp_joint_cell_attribution_v1'
C2 = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
PINS = {
    TRACE/'probe.py':'17dc7f67e126f78d1bc390ed3c74f8347cd8077bc262406d032677be89c879b3',
    TRACE/'watch_probe.py':'bfb12b6801aa0ccef1e064276c602303e9f6d228a4d9b6914c4e05c28716fa75',
    TRACE/'PROTOCOL.md':'a46b8ceb091d99241fb9f38140c390dc2e8097f98f2ae17229b4b6092f5c8cea',
    TRACE/'WATCH_SPEC.md':'98483acbac017f1f97418ccc3cf9ed6030956a423cf28a09d43d7ce13d61652a',
    CELL/'consumer_v2.py':'c57d149127a6e23ba5403fc2c260f765327b63e2a36bb84ff06915e65b512a82',
}
PACKET_CAP = 512*1024**2
NATIVE_CAP = 250_000_000_000


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            h.update(block)
    return h.hexdigest()


def load(path, name, expected):
    require(sha(path) == expected, 'Pinned inert source owner')
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(reviewed_sha, tick=None):
    require(sha(__file__) == reviewed_sha and not
        {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules),
        'Reviewed source and stdlib admission before numerical imports')
    j = load(TRACE/'probe.py','native_adjoint_packet_admission',PINS[TRACE/'probe.py'])
    args = types.SimpleNamespace(**{key:PINS[TRACE/name] for key,name in (
        ('source_sha256','probe.py'),('watcher_sha256','watch_probe.py'),
        ('protocol_sha256','PROTOCOL.md'),('watch_spec_sha256','WATCH_SPEC.md'))})
    base = j.admission(args,tick)
    unused_c,a,h,backend,legacy = j.helpers()
    c = load(CELL/'consumer_v2.py','native_adjoint_source_api',PINS[CELL/'consumer_v2.py'])
    owner = h.load(h.OBSERVER,'native_adjoint_observation_api',h.OBSERVER_SHA)
    source, proof = c.derive(a,tick=tick)
    source, parameter_proof = a.replace_function(source,'_predict_learned_compact',[
        ('    P=_GNN_PARAMS\n','    P=_GNN_PARAMS\n    _nadj_parameters(P)\n')])
    certificate = c.cost_certificate(base['manifest'],a)
    # The existing bound includes all original copies/diagnostics. Retaining
    # their bytes adds no numerical work. H7's extra copy+finite/export and
    # all64M input bytes, literal perturbation/checks and return validation
    # fit this additional100MF; this is not borrowed deployment cost.
    upper = certificate['complete_partial_upper_flops']+100_000_000
    require(upper < NATIVE_CAP, 'Complete native source bound before execution')
    sources = dict(base['source_bindings'])
    j.merge(sources,PINS); j.merge(sources,owner.OWNERS)
    sources[str(Path(__file__).resolve())] = reviewed_sha
    controls_path = CELL/'capture_v1/call_00_diagnostics.json'
    require(str(controls_path) in base['input_bindings'], 'Original controls already sealed')
    j.bind(sources,tick)
    return dict(source_bindings=sources,input_bindings=base['input_bindings'],
        source=source,source_proof=dict(restart=proof,parameter_callback=parameter_proof,
        candidate_sha256=C2,only_local_entry_fixed=True,downstream_states_recomputed=True),
        ledger_source=base['source']['ledger'],manifest=base['manifest'],
        fixture=base['fixture'],capture_seals=base['capture_seals'],
        controls=j.read(controls_path),cost_certificate=dict(base=certificate,
        additional_input_and_parameter_upper=100_000_000,native_upper=upper,
        hard_cap=NATIVE_CAP,actual_complete_native_bill_required=True)),(j,c,a,h,backend,legacy,owner)


def sink_class(c,a):
    Parent = c.replay_sink_class(a)

    class BoundarySink(Parent):
        def __init__(self,fnp,manifest,arm):
            super().__init__(fnp,manifest,arm)
            self.head7 = None

        def array(self,value,path,keep=True):
            # Same one paid copy/finite/export as the original observer; just
            # retain the bytes needed by the native-state coordinate adapter.
            persist = self.arm == 'N' and (path.startswith('entry8.') or
                path in ('head7.state','head8.state','pre8.live_indices','pre8.response_indices',
                         'pre9.live_indices','pre9.response_indices') or
                (path.startswith('end8.') and path.split('.')[1] in a.ENTRY_KEYS))
            row = a.Sink.array(self,value,path,persist)
            if self.arm == 'N':
                expected = self.reference
                for token in path.split('.'):
                    if 'mapping' in expected: expected = expected['mapping']
                    if 'sequence' in expected: expected = expected['sequence']
                    expected = expected[int(token)] if isinstance(expected,list) else expected[token]
                expected = expected['array']
                require(all(row[k] == expected[k] for k in (*c.physical_metadata(value),'sha256')),
                    'Exact current native byte/layout checkpoint: '+path)
                self.checks[path] = True
            return row

        def save_head7(self,value):
            require(self.ctx is not None and not self.packets, 'Paid fresh H7 boundary copy')
            before = int(self.ctx.flops_used)
            self.head7 = {'state':{'array':self.array(value,'head7.state')}}
            self.flops += int(self.ctx.flops_used)-before

        def bundle(self):
            require(self.arm == 'N' and self.ctx is None and self.head7 is not None,
                'Only retired unchanged call provides adjoint primal boundaries')
            blob = b''.join(self.blocks)
            require(len(blob) == self.bytes < PACKET_CAP, 'Bounded detached boundary packet')
            metadata = dict(schema='current_native_trajectory_packet_v1',rank=35,layer=8,consumer=9,
                candidate_sha256=C2,bytes=len(blob),packet_sha256=digest(blob),
                packets=dict(self.packets,head7=self.head7),truth_accessed=False,
                archived_downstream_states_never_used_as_inputs=True,
                fresh_downstream_states=True,local_entry_fixed=True)
            return dict(metadata=metadata,blob=blob)
    return BoundarySink


def phase(context,apis,np,flops,fnp,ctx,refs,arm,perturb,MLP,seed_fn,materialise):
    """Execute inside caller's fresh paid context; return only bytes/metadata.

    perturb(P,fnp,arm) must be the pinned prospective parameter direction.
    N is an exact identity; + and - own fresh original caches before mutation.
    No global numerical object, target or old downstream state is injected.
    """
    require(arm in ('N','plus','minus') and 'torch' not in sys.modules,
        'Exactly one declared native arm, no Torch runtime')
    j,c,a,h,backend,legacy,owner = apis
    raw = Path(context['manifest']['packet_file']).read_bytes()
    require(digest(raw) == context['manifest']['packet_sha256'], 'Immutable target-blind entry')
    weights = np.load(context['fixture'],mmap_mode='r',allow_pickle=False)
    require(weights.shape == (16,1024,1024) and weights.dtype == np.float32, 'Actual physical fixture35')
    layers = [fnp.array(w,dtype=fnp.float32,copy=True) for w in weights]
    require(all(bool(fnp.all(fnp.isfinite(w))) and w.tobytes() == weights[i].tobytes()
        for i,w in enumerate(layers)), 'Paid complete independent physical weights')
    refs.extend(weakref.ref(w) for w in layers)
    row = j.read(Path(context['fixture']).with_name('row_manifest.json'))
    mlp = MLP(width=1024,depth=16,weights=layers,seed=int(seed_fn(row['mlp_seed'])),
        name='native-adjoint-local35-'+arm)
    subject = types.ModuleType('native_adjoint_private_'+arm)
    subject.__file__ = '<native-adjoint-current-c2-local-'+arm+'>'
    refs.append(weakref.ref(subject))
    observer = sink = served = materialized = None
    callback_calls = []; parameter_record = None
    def parameter_callback(P):
        nonlocal parameter_record
        require(not callback_calls, 'Exactly one current original P allocation per call')
        parameter_record = perturb(P,fnp,arm)
        callback_calls.append(arm)
    try:
        subject._nadj_parameters = parameter_callback
        exec(compile(context['source'],subject.__file__,'exec'),vars(subject))
        entry = context['manifest']['packets']['entry8']
        subject._jca_entry = {k:c.restore(v,raw,np,fnp,refs,'entry8.'+k) for k,v in entry.items()}
        subject._jca_H7 = c.restore(context['manifest']['packets']['head7']['state'],
            raw,np,fnp,refs,'head7.state')
        sink = sink_class(c,a)(fnp,context['manifest'],arm)
        refs.append(weakref.ref(sink)); sink.ctx = ctx
        sink.save_head7(subject._jca_H7)
        subject._jca_emit = sink
        subject._jca_intervene = lambda mu,C,K,k4,k22:(mu,C,K,k4,k22,None)
        Observer,diagnostics,op = c.partial_observers(owner,a)
        observer = Observer(subject); refs.append(weakref.ref(observer))
        observer.begin(); observer.ctx = ctx
        estimator = subject.Estimator(); refs.append(weakref.ref(estimator))
        served = owner.base().observed_predict(estimator,mlp,observer)
        require(isinstance(served,np.ndarray) and served.shape == (2,1024)
            and served.dtype == np.float32, 'Actual native returned means8/9')
        materialized = materialise(served,depth=2,width=1024)
        require(bool(fnp.all(fnp.isfinite(served))) and bool(fnp.all(served >= 0))
            and served.tobytes() == materialized.tobytes(), 'Paid finite/nonnegative served bytes')
        values = materialized.tobytes()
        observer.ctx = None; sink.ctx = None
        packet,startup,dispatches = observer.finish()
        domain = diagnostics(np,packet); checkpoint = sink.finish()
        require(domain['variance_count'] == 2 and domain['selector_count'] == 5,
            'Only actual local two variances/five selectors, never all16/47')
        require(all(startup['prefix'][k] is True for k in ('P_empty_at_entry','A_empty_at_entry',
            'cstar_empty_at_entry','P_ready_at_binary_entry','A_ready_at_binary_entry','cstar_ready_at_binary_entry')),
            'Fresh original P/NODE/cstar owners in every call')
        if arm == 'N':
            old = context['controls']['domains']
            require(domain['variance_domain'] == {str(i):old['variance_domain'][str(i)] for i in (8,9)}
                and domain['selection_cells'] == {k:old['selection_cells'][k]
                    for k in ('8:0','8:1','8:2','9:0','9:1')}
                and checkpoint['null_byte_exact'], 'All original native local physical/domain bytes')
            original = np.load(CELL/'capture_v1/call_00_rank35.npy',allow_pickle=False)
            require(original[8:10].tobytes() == values, 'Exact archived served8/9, not approximate parity')
            strip = lambda xs:[{k:v for k,v in x.items() if k != 'operation_count'} for x in xs]
            require(strip(dispatches) == strip([x for x in context['controls']['dispatches']
                if x['layer'] in (8,9)]), 'Exact native dispatch shapes/costs; extra observer operations are paid')
            del original
        require(callback_calls == [arm], 'Actual pinned parameter callback executed once')
        bundle = sink.bundle() if arm == 'N' else None
        return dict(served=values,served_sha256=digest(values),bundle=bundle,diagnostic=dict(
            arm=arm,domains=domain,startup=startup,dispatches=dispatches,observer_proof=op,
            checkpoints=checkpoint,parameter_record=parameter_record,
            full_model_calls=0,partial_model_calls=1,truth_accessed=False,
            score_measured=False,local_entry_fixed=True,fresh_downstream_trajectory=True))
    finally:
        if observer is not None: observer.ctx = None
        if sink is not None: sink.ctx = None
        c.parameter_refs(subject,refs)
        for name in ('_jca_entry','_jca_H7'):
            value = getattr(subject,name,None)
            def remember(item):
                if isinstance(item,dict):
                    for x in item.values(): remember(x)
                elif isinstance(item,(list,tuple)):
                    for x in item: remember(x)
                elif not isinstance(item,(types.ModuleType,type)) and hasattr(item,'tobytes'):
                    try: refs.append(weakref.ref(item))
                    except TypeError: pass
            remember(value)
        subject.__dict__.clear()
