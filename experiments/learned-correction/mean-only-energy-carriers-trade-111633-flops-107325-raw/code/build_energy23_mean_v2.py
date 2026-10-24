"""One fixed current777a energy approximation; signed messages/E1 unchanged."""
import argparse, ast, hashlib, json, subprocess
from pathlib import Path

SCI='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
REL='e41095e464be4f5973cae13252a716bc1d85a56bd3ba9d0837e28509c3d08f05'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,s):
    with Path(p).open('x') as h:h.write(s)
def utc():return subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()

HELPER='''
def _vmfr_e23_energy(T, X):
    # E1 uses the unchanged rank2 contraction. E2/E3 retain exact full means.
    keep = _E2Hold()
    out = fnp.empty_like(X)
    T1, X1 = keep(T[0:1]), keep(X[0:1])
    T23, X23 = keep(T[1:3]), keep(X[1:3])
    out1, out23 = keep(out[0:1]), keep(out[1:3])
    fnp.copyto(out1, keep(_gop2_contract(T1, X1)))
    mu = keep(fnp.mean(X23, axis=1))
    mass = keep(fnp.sum(T23, axis=2))
    fnp.multiply(keep(mass[:, :, None]), keep(mu[:, None, :]), out=out23)
    return out
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input-manifest',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    spec=next(x for x in json.loads(a.input_manifest.read_text())['candidates'] if x['name']=='msg8_teacher_loaderfix')
    parents={r:Path(spec[r]['path']) for r in ('science','release')}
    assert sha(parents['science'])==SCI and sha(parents['release'])==REL
    sources={r:p.read_text() for r,p in parents.items()}
    needed={'_gop2_gnn_step','_gop2_contract','_gop2_messages','_gop2_message_product','_w3r_product'}
    functions={}
    for role, text in sources.items():
        lines=text.splitlines()
        functions[role]={n.name:'\n'.join(lines[n.lineno-1:n.end_lineno]) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name in needed}
    for name in ('_gop2_gnn_step','_gop2_contract','_gop2_messages','_gop2_message_product','_w3r_product'):
        assert functions['science'][name]==functions['release'][name]
    step=functions['science']['_gop2_gnn_step'];assert step.count('E = _gop2_contract(')==1
    changed=step.replace('E = _gop2_contract(', 'E = _vmfr_e23_energy(')
    a.out.mkdir(parents=True,exist_ok=False)
    tail='\n# Fixed E2/E3 full-mean energy approximation; no fitted or selected layers.\n'
    tail+='assert _VMFR_CM == 8 and _O20_FLATEXTRA_ON is False\n'
    tail+='assert _GNN_PARAMS is None and not _E2_GNN_STATE and not _E2_MSG\n'
    tail+='_VMFR_ENERGY23_MEAN = True\n'+HELPER+'\n'+changed+'\n_FAST_RESCUE_FRAC = {}\n'
    name='msg8_energy23_mean';append=a.out/('tail_'+name+'_v1.py');fixture=a.out/('composed_fixture_'+name+'_v1.py')
    oldtail=Path(spec['tail']);assert sha(oldtail)==spec['tail_sha256']
    write(append,tail);write(fixture,oldtail.read_text()+tail)
    c=dict(name=name,kind='fixed_E2_E3_fullmean_only_energy',message_width=8,unchanged_signed_messages=True,
        unchanged_E1_rank2=True,full_mean_preserved=True,changed_layers=list(range(1,16)),
        fit=False,validation_selection=False,reference_submission=337422,
        append_tail=str(append),append_tail_sha256=sha(append),tail=str(fixture),tail_sha256=sha(fixture),fixture_not_deployment=True)
    for role,p in parents.items():
        target=a.out/('candidate_'+name+('_sci_v1.py' if role=='science' else '_rel_v1.py'))
        source=sources[role]+tail;compile(source,str(target),'exec');write(target,source)
        c[role]=dict(path=str(target),sha256=sha(target),parent_path=str(p),parent_sha256=sha(p))
    report=dict(utc=utc(),source_sha256=sha(__file__),candidates=[c],reference_submission=337422,
        status='FIXED_SOURCE_ONLY_NO_NATIVE_SCORE',historical_delivery_failure_preserved=True)
    write(a.out/'CANDIDATES_v1.json',json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(science=c['science'],release=c['release'],manifest=str(a.out/'CANDIDATES_v1.json')),indent=2))
if __name__=='__main__':main()
