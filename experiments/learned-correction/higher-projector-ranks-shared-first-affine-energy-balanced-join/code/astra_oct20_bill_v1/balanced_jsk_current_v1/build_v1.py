"""One fixed balanced-operand JSK variant on exact VALID337422; no model run."""
import ast,copy,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
V=HERE.parents[1]/'astra_oct20_vm_frontier_v1'
PARENT=V/'msg8_loaderfix_v1'
PINS={'science':'777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510','release':'e41095e464be4f5973cae13252a716bc1d85a56bd3ba9d0837e28509c3d08f05'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
OLD="Y = _shared_old384_mm(J1, weight[_jn_cols1][:, None] * J1[_jn_rows, :].T)"
NEW="Y = _o21_jbl_young_gram(J1, weight[_jn_cols1], _jn_rows)"
HELPER='''def _o21_jbl_young_gram(J1, selected_weight, rows):
    hold = _E2Hold()
    scale = hold(fnp.sqrt(selected_weight))
    A = hold(J1 * hold(scale[None, :]))
    selected = hold(A[rows, :])
    return _shared_old384_mm(A, hold(selected.T))
'''
def main():
    assert not (HERE/'CANDIDATES_v1.json').exists()
    parents={r:PARENT/('candidate_msg8_teacher_loaderfix_'+('sci' if r=='science' else 'rel')+'_v1.py')for r in PINS}
    texts={r:p.read_text()for r,p in parents.items()}
    assert all(sha(parents[r])==PINS[r]for r in PINS)
    tree=ast.parse(texts['science']);n=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_astra_bcc_push_impl'][-1]
    source=ast.get_source_segment(texts['science'],n)
    assert source.count(OLD)==1
    changed=source.replace(OLD,NEW)
    assert ast.dump(ast.parse(changed.replace(NEW,OLD)))==ast.dump(ast.parse(source))
    tail='\n\n# Fixed balanced JSK operands; '+utc()+'. Numerical-order hypothesis, no fitted targets.\n'
    tail+='assert _ASTRA_BCC_ON and _JOIN_NYSTROM\nassert _VMFR_CM == 8 and _O20_EDGE_HIDDEN == 14\nassert _GNN_PARAMS is None and _O20_FLATEXTRA_ON is False and _FAST_RESCUE_FRAC == {}\n'
    tail+='_O21_BALANCED_JSK = True\n\n'+HELPER+'\n'+changed+'\n\n_FAST_RESCUE_FRAC = {}\n'
    tp=HERE/'tail_msg8_balanced_jsk_v1.py';tp.open('x').write(tail)
    rec=dict(name='msg8_balanced_jsk',component='_astra_bcc_push_impl',append_tail=str(tp),append_tail_sha256=sha(tp),fit=False,truth_used=False)
    for role in PINS:
        out=HERE/('candidate_msg8_balanced_jsk_'+('sci'if role=='science'else'rel')+'_v1.py')
        out.open('x').write(texts[role]+tail);compile(out.read_text(),str(out),'exec')
        rec[role]=dict(path=str(out),sha256=sha(out),bytes=out.stat().st_size,parent_path=str(parents[role]),parent_sha256=PINS[role])
    receipt=dict(utc=utc(),status='FIXED_BALANCED_JSK_SOURCE_FROZEN_NO_NATIVE',reference_submission=337422,parent_source_sha256=PINS['science'],candidates=[rec],builder_sha256=sha(__file__),source_delta='One live young weighted Gram call; all selectors, ranks, columns, old core, QR, factors, trace normalization and routes unchanged.',expected_accuracy_gain=None,real_identity='J diag(w) J_rows.T = (J sqrt(w)) (J_rows sqrt(w)).T; w=lambda**2 >=0',rounding_changes_intentional=True,no_governor=True)
    (HERE/'CANDIDATES_v1.json').open('x').write(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
