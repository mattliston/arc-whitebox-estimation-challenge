"""Target-free full-state capture for a frozen terminal-response fit.
Fetches are diagnostic charged operations; no score/runtime claim is valid.
No retained native handles and no estimator code mutation.
"""
import gzip,hashlib,json,subprocess,sys
from pathlib import Path
def utc():return subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with p.open('x') as out:json.dump(x,out,allow_nan=False,separators=(',',':'));out.write('\n')
class Observer:
    def __init__(self,mod,outdir,recipe):
        self.mod=mod;self.outdir=Path(outdir)/'gnn_full_state_v1';self.outdir.mkdir(exist_ok=False)
        self.recipe=dict(recipe);self.rank=None;self.phase=None;self.seen=set();self.files=[];self.finished=False
        self.parent=sys.getprofile();self.code=mod._gop2_gnn_step.__code__
        assert mod._GOP2_ON and mod._FAST_RESCUE_FRAC=={}
        assert sha(mod.__file__)=='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
        self.metadata=dict(utc=utc(),source_sha256=sha(mod.__file__),hook_sha256=sha(__file__),
            rows_per_layer=1024,layers=list(range(1,16)),target_free=True,reference_submission=337422,
            row_rule='all original unit rows, original order',gnn_literal=mod._GNN_LITERAL,
            limitation='Diagnostic fetches change tariff/timing; source prediction bytes must match exact777a.')
        write(self.outdir/'METADATA_v1.json',self.metadata);sys.setprofile(self.profile)
    def begin(self,rank,phase):
        assert not self.finished and int(rank) in (35,150,200,250)
        self.rank=int(rank);self.phase=str(phase)
    def fetch(self,value):return dict(shape=[int(x) for x in value.shape],data=value.tolist())
    def profile(self,frame,event,arg):
        if self.parent is not None:self.parent(frame,event,arg)
        if self.finished or self.rank is None or event!='return' or frame.f_code is not self.code:return
        s=frame.f_locals;l=int(s['l']);assert 1<=l<=15
        key=(self.rank,self.phase,l);assert key not in self.seen
        assert tuple(s['state'].shape)==(1024,64) and tuple(s['h'].shape)==(1024,128)
        assert tuple(s['k']['H'].shape)==(1024,78)
        gate=s['k']['H'][:,5]
        record=dict(rank=self.rank,phase=self.phase,layer=l,state=self.fetch(s['state']),
            hidden2=self.fetch(s['h']),sigma=self.fetch(s['sigma']),delta=self.fetch(s['delta']),
            gate_feature=self.fetch(gate))
        del gate
        path=self.outdir/('rank%04d_%s_L%02d.json.gz'%key)
        with gzip.open(path,'xt',encoding='utf-8',compresslevel=3) as out:
            json.dump(record,out,allow_nan=False,separators=(',',':'));out.write('\n')
        self.files.append(dict(path=path.name,sha256=sha(path),bytes=path.stat().st_size,
            rank=self.rank,phase=self.phase,layer=l));self.seen.add(key)
    def finish(self):
        assert not self.finished;self.finished=True;sys.setprofile(self.parent)
        groups=sorted({(r,p) for r,p,_ in self.seen})
        for r,p in groups:assert {l for rr,pp,l in self.seen if rr==r and pp==p}==set(range(1,16))
        assert self.files and sum(x['bytes'] for x in self.files)<1_000_000_000
        receipt=dict(utc=utc(),complete=True,groups=groups,files=self.files,
            source_sha256=self.metadata['source_sha256'],target_free=True,native_handles_retained=False)
        write(self.outdir/'CAPTURE_COMPLETE_v1.json',receipt);return receipt
def install(mod,outdir,recipe):return Observer(mod,outdir,recipe)

