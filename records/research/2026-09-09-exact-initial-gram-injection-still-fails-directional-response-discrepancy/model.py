"""Inert CPU adapter: only weights-only Cpre/Bpre at layer0 become native G0."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
CO = NEW/'current_bd_p_coadaptation_v1'
IG = NEW/'current_d50_initial_gram_witness_v1'
CO_SHA = '556f3b97613110e638f48127e40112ab203623fa9f6b076b0eab7f07d24658a0'
FIXED = {
    CO/'model.py': CO_SHA,
    IG/'acquire.py': '4d81833c904f43b8e7531bf377c998193cf1cfbfd3e88c215c4a65ba097b140c',
    IG/'watch.py': '736125d779a8eb9e1a530f7a3047276be585f8be09f47a4c48be0039e127a9d1',
    IG/'PROTOCOL.md': '2be63ef4fd10d1d30aa14b496ec238a1f06e0391c151937e484daa765c3d5ee4',
    IG/'acquisition_v1/result.json': '1f12c9a5992c29866a53005a75931fb2849298749a43365915df8e5986667b73',
    IG/'watch_v1/result.json': 'cf6488bc1bd38fff7dd52de2d843e63674feac7beb1797f34cb5b5708f5d3f3a',
}
_OWNER = _CARD = None


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def owner():
    global _OWNER
    require(sha(CO/'model.py') == CO_SHA, 'Frozen CPU owner changed')
    if _OWNER is None:
        spec = importlib.util.spec_from_file_location('native_g0_frozen_cpu_owner',CO/'model.py')
        _OWNER = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = _OWNER
        spec.loader.exec_module(_OWNER)
    return _OWNER


def authenticate():
    result = owner().authenticate()
    for path,digest in FIXED.items():
        require(sha(path)==digest,'Frozen initial-Gram input changed: '+str(path))
        result[str(path)] = digest
    records = {}
    for stage in ('acquisition_v1','watch_v1'):
        rp,cp = IG/stage/'result.json',IG/stage/'completion.json'
        record,seal = json.loads(rp.read_text()),json.loads(cp.read_text())
        require(record['passed'] and record['completed'] and seal['passed'] and seal['completed']
                and seal['result_sha256']==sha(rp),'Initial-Gram completed PASS required')
        require(record.get('hashes_unchanged',record.get('sources_unchanged',False)),
                'Initial-Gram source integrity')
        result[str(cp)] = sha(cp)
        for name in ('source_bindings','input_bindings','artifacts'):
            for path,digest in record.get(name,{}).items():
                require(sha(path)==digest,'Initial-Gram dependency changed: '+str(path))
                require(path not in result or result[path]==digest,'Conflicting binding')
                result[path] = digest
        records[stage] = record
    r,w = records['acquisition_v1'],records['watch_v1']
    require(not w['terminated'] and w['child_exit_code']==0
            and w['child_result_sha256']==FIXED[IG/'acquisition_v1/result.json']
            and w['elapsed_s']<900 and w['maximum_sampled_rss_bytes']<4_000_000_000
            and w['maximum_swap_growth_bytes']<=2*1024**3,'Initial-Gram external watch')
    require(r['all_native_layer0_bytes_exact'] and r['repeat_Gram_mean_byte_exact']
            and r['ranks']==[35,100,35] and r['truth_accessed'] is False
            and r['full_card_calls']==0 and r['P_accessed'] is False,'Weights-only native witness')
    for path in (Path(__file__).resolve(),HERE/'PROTOCOL.md'):
        result[str(path)] = sha(path)
    return result


def card_class():
    global _CARD
    if _CARD is not None:
        return _CARD
    authenticate()
    frozen = owner()
    parent,base = frozen._card_class(),frozen._base_module()
    torch,np = base.torch,base.np
    observed = base.replace_once(base.source,'    vv = torch.diagonal(Cpre)\n',
        '    vv = torch.diagonal(Cpre)\n    self._record_variance(l, vv)\n')
    edits = (
        ('    Cpre = ((w.T@C.float())@w).to(dt)',
         '    Cpre = self.native_G0 if self.enabled and l == 0 else ((w.T@C.float())@w).to(dt)'),
        ('    Bpre = ((w.T@metric.float())@w).to(dt)',
         '    Bpre = self.native_G0 if self.enabled and l == 0 else ((w.T@metric.float())@w).to(dt)'),
    )
    source = observed
    for old,new in edits: source=base.replace_once(source,old,new)
    inverse = source
    for old,new in reversed(edits): inverse=base.replace_once(inverse,new,old)
    require(inverse==observed,'Exact original observed-step inverse')
    code,count = base.rewrite_clamps(source,str(__file__)+':step')
    require(count==2,'Original clamp roster changed')
    env = dict(base.env)
    exec(code,env)

    class NativeG0Card(parent):
        def __init__(self,rank,P=None,enabled=True):
            require(rank in (35,100),'Only fixed witness ranks admitted')
            pins = authenticate()
            rec = json.loads((IG/'acquisition_v1/result.json').read_text())
            row = next(x for x in rec['trials'] if x['rank']==rank)
            meta = row['G0']
            g0 = np.load(meta['path'],allow_pickle=False)
            require(g0.dtype==np.float64 and g0.shape==(1024,1024)
                    and hashlib.sha256(g0.tobytes()).hexdigest()==meta['bytes_sha256'],
                    'Exact native G0 array')
            require(np.isfinite(g0).all() and np.diag(g0).min()>=1e-10,'Native G0 domain')
            super().__init__(rank,P=P,device='cpu',audit_selections=True)
            weight = str(Path(base.weight_path(rank)).resolve())
            require(weight in rec['input_bindings'] and pins[weight]==rec['input_bindings'][weight]
                    and sha(weight)==rec['input_bindings'][weight],'Exact same canonical weights')
            self.native_G0 = torch.tensor(g0,dtype=torch.float64,device='cpu')
            require(not self.native_G0.requires_grad,'Weights-only G0 must not have a P path')
            self.enabled = bool(enabled)
            self.g0_file_sha256,self.g0_raw_sha256 = meta['sha256'],meta['bytes_sha256']
            self.repaired_source_sha256 = hashlib.sha256(source.encode()).hexdigest()

        def step(self,layer,state,control,P):
            require(P is self.P0 and not bool((control.detach()!=0).any()),
                    'No payload override or exogenous control')
            self._layer,self._selection_slot = layer,0
            return env['step'](self,layer,state,control,P)

        def diagnostics(self):
            result=super().diagnostics()
            result.update(native_G0_enabled=self.enabled,native_G0_file_sha256=self.g0_file_sha256,
                native_G0_raw_sha256=self.g0_raw_sha256,source_inverse_byte_exact=True,
                repaired_source_sha256=self.repaired_source_sha256,
                native_G0_requires_grad=self.native_G0.requires_grad,
                only_Cpre_Bpre_at_layer0_replaced=True)
            return result

    _CARD = NativeG0Card
    return _CARD


def card(rank,P=None,enabled=True):
    return card_class()(rank,P=P,enabled=enabled)
