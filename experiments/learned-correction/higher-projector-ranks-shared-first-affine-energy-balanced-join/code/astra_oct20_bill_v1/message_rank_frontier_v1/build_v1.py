"""Two fixed all-layer rank singles on exact777a; no fitting or model calls."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
R=HERE.parents[1]
V=R/'astra_oct20_vm_frontier_v1'
PARENT=V/'msg8_loaderfix_v1'
PINS={'science':'777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510',
      'release':'e41095e464be4f5973cae13252a716bc1d85a56bd3ba9d0837e28509c3d08f05'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()


def main():
    manifest=HERE/'CANDIDATES_v1.json'
    if manifest.exists(): raise FileExistsError(manifest)
    parents={role:PARENT/('candidate_msg8_teacher_loaderfix_'+('sci' if role=='science' else 'rel')+'_v1.py')
             for role in PINS}
    texts={role:p.read_text() for role,p in parents.items()}
    assert all(sha(parents[role])==pin for role,pin in PINS.items())
    tree=ast.parse(texts['science'])
    candidates=[]
    for tag,function,oldrank,newrank in [('w3rank6','_w3r_product',4,6),('gop2rank3','_gop2_contract',2,3)]:
        original=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function][-1]
        node=copy.deepcopy(original)
        changed=0
        for n in ast.walk(node):
            if (isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='vectors'
                    and isinstance(n.slice,ast.Tuple)):
                last=n.slice.elts[-1]
                if isinstance(last,ast.Slice):
                    assert isinstance(last.lower,ast.UnaryOp) and isinstance(last.lower.op,ast.USub)
                    assert last.lower.operand.value==oldrank and last.upper is None and last.step is None
                    last.lower.operand.value=newrank;changed+=1
        assert changed==1
        back=copy.deepcopy(node)
        for n in ast.walk(back):
            if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='vectors':
                n.slice.elts[-1].lower.operand.value=oldrank
        assert ast.dump(back)==ast.dump(original)
        suffix='\n\n# Fixed accuracy-for-compute single, '+utc()+'. No fitted weights change.\n'
        suffix+="assert _VMFR_CM == 8 and _O20_EDGE_HIDDEN == 14\n"
        suffix+="assert _GNN_PARAMS is None and _W3R_METRIC is None and not _E2_MSG\n"
        suffix+="assert _O20_FLATEXTRA_ON is False and _FAST_RESCUE_FRAC == {}\n"
        suffix+='_O21_MESSAGE_RANK_VARIANT = '+repr(tag)+'\n\n'
        suffix+=ast.unparse(node)+'\n\n_FAST_RESCUE_FRAC = {}\n'
        tail=HERE/('tail_msg8_'+tag+'_v1.py');tail.open('x').write(suffix)
        rec=dict(name='msg8_'+tag,component=function,old_rank=oldrank,new_rank=newrank,
                 append_tail=str(tail),append_tail_sha256=sha(tail),fit=False,truth_used=False)
        for role in PINS:
            out=HERE/('candidate_msg8_'+tag+'_'+('sci' if role=='science' else 'rel')+'_v1.py')
            out.open('x').write(texts[role]+suffix)
            compile(out.read_text(),str(out),'exec')
            rec[role]=dict(path=str(out),sha256=sha(out),bytes=out.stat().st_size,
                           parent_path=str(parents[role]),parent_sha256=PINS[role])
        candidates.append(rec)
    receipt=dict(utc=utc(),status='TWO_FIXED_RANK_SINGLES_SOURCE_FROZEN_NO_NATIVE',
                 reference_submission=337422,parent_source_sha256=PINS['science'],
                 candidates=candidates,builder_sha256=sha(__file__),
                 no_combined_point=True,no_sweep=True,no_carrier_correction=True,
                 source_delta='Only one vector slice rank in one live function per candidate; exact parent prefix and weights.')
    manifest.open('x').write(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
