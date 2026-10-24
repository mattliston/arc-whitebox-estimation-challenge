"""Freeze one pair-complete energy-ranked compactJSK candidate on actual777a."""
import ast,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;V=HERE.parents[1]/'astra_oct20_vm_frontier_v1'
PINS={'science':'777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510','release':'e41095e464be4f5973cae13252a716bc1d85a56bd3ba9d0837e28509c3d08f05'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
CHANGES=[('_jn_c = fnp.sum(_jn_JJ, axis=0) * weight','_jn_c = fnp.sum(_jn_JJ, axis=0)'),
         ("_jn_cols1 = fnp.argsort(-(_jn_c.astype(fnp.float64)), kind='stable')[:_JSK_KC[layer]]","_jn_cols1 = _o21_pair_jsk_columns(J, _jn_c, weight, tuple(int(c[1]) for c in state['cohorts'][joins_here[0]:joins_here[1] + 1]), _JSK_KC[layer])")]
def main():
    assert not(HERE/'CANDIDATES_v1.json').exists()
    parents={r:V/'msg8_loaderfix_v1'/('candidate_msg8_teacher_loaderfix_'+('sci'if r=='science'else'rel')+'_v1.py')for r in PINS};texts={r:p.read_text()for r,p in parents.items()};assert all(sha(parents[r])==PINS[r]for r in PINS)
    s=texts['science'];node=[n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)and n.name=='_astra_bcc_push_impl'][-1];old=ast.get_source_segment(s,node);new=old
    for before,after in CHANGES:assert new.count(before)==1;new=new.replace(before,after)
    check=new
    for before,after in reversed(CHANGES):check=check.replace(after,before)
    assert check==old
    tail='\n\n# Pair-complete tensor-energy compactJSK selection; '+utc()+'.\n'
    tail+='assert _ASTRA_BCC_ON and _JOIN_NYSTROM\nassert _VMFR_CM == 8 and _O20_EDGE_HIDDEN == 14\nassert _GNN_PARAMS is None and _O20_FLATEXTRA_ON is False and _FAST_RESCUE_FRAC == {}\n'
    tail+=(HERE/'selector_v1.py').read_text()+'\n\n'+new+'\n\n_FAST_RESCUE_FRAC = {}\n';tp=HERE/'tail_msg8_pair_energy_jsk_v1.py';tp.open('x').write(tail)
    rec=dict(name='msg8_pair_energy_jsk',component='_astra_bcc_push_impl',append_tail=str(tp),append_tail_sha256=sha(tp),fit=False,truth_used=False)
    for role in PINS:
        out=HERE/('candidate_msg8_pair_energy_jsk_'+('sci'if role=='science'else'rel')+'_v1.py');out.open('x').write(texts[role]+tail);compile(out.read_text(),str(out),'exec');rec[role]=dict(path=str(out),sha256=sha(out),bytes=out.stat().st_size,parent_path=str(parents[role]),parent_sha256=PINS[role])
    receipt=dict(utc=utc(),status='PAIR_COMPLETE_ENERGY_SELECTOR_SOURCE_FROZEN_NO_NATIVE',reference_submission=337422,parent_source_sha256=PINS['science'],candidates=[rec],builder_sha256=sha(__file__),selector_sha256=sha(HERE/'selector_v1.py'),changes=CHANGES,preserved='Same rank384, kc1536/1024/1024, λ² weighted Gram, old core, row landmarks, QR, reconstruction and trace normalization. Only joined young-column ranking and pair completeness change.',motivation='CompactJSK already benefits from selecting fewer columns; signed-pair cancellation is missing from the current per-leg λ²||p||² score.',accuracy_gain_unknown=True,rounded_f32_norms=True,expected_bill_unmeasured=True,no_governor=True)
    (HERE/'CANDIDATES_v1.json').open('x').write(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':receipt['status'],'candidates':rec},indent=2))
if __name__=='__main__':main()
