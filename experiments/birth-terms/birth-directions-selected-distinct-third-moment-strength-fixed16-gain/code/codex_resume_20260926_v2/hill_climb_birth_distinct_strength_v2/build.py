"""Source-only build: one birth callsite plus one appended selector."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'hill_climb_star_projection32_v1/candidate.py'
SHA='f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef'
OLD='selected, births = select(_e2_gate_min, gate_floor)'
NEW='selected, births = _birth_distinct_select(_e2_gate_min, gate_floor, birth, g, alive, Cpre)'


def pin(path):
    b=path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))


def main():
    candidate=HERE/'candidate.py';receipt=HERE/'build.json'
    assert not candidate.exists() and not receipt.exists()
    parent=PARENT.read_bytes();text=parent.decode();fragment=(HERE/'fragment.py').read_bytes()
    assert hashlib.sha256(parent).hexdigest()==SHA and len(parent)==930166
    assert text.count(OLD)==1 and NEW not in text
    modified=text.replace(OLD,NEW)
    assert modified.replace(NEW,OLD).encode()==parent
    extra=ast.parse(fragment)
    assert [node.name for node in extra.body if isinstance(node,ast.FunctionDef)]==['_birth_distinct_select']
    assert not any(isinstance(node,(ast.Import,ast.ImportFrom)) for node in ast.walk(extra))
    combined=modified.encode()+b'\n\n'+fragment
    ast.parse(combined)
    inverse=combined[:-len(fragment)-2].decode().replace(NEW,OLD)
    assert inverse.encode()==parent and ast.dump(ast.parse(inverse))==ast.dump(ast.parse(parent))
    protocol=json.loads((HERE/'protocol.json').read_text())
    expected=dict(path=str(candidate),sha256=hashlib.sha256(combined).hexdigest(),bytes=len(combined))
    assert protocol['parent']==pin(PARENT) and protocol['fragment']==pin(HERE/'fragment.py') and protocol['candidate']==expected
    with candidate.open('xb') as f:f.write(combined)
    result=dict(passed=True,parent=pin(PARENT),fragment=pin(HERE/'fragment.py'),candidate=pin(candidate),
        builder=pin(Path(__file__).resolve()),protocol=pin(HERE/'protocol.json'),
        exact_inverse=True,ast_inverse=True,callsite_replacements=1,old_call=OLD,new_call=NEW,
        other_selection_calls_unchanged=True,numerical_execution=False)
    with receipt.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
