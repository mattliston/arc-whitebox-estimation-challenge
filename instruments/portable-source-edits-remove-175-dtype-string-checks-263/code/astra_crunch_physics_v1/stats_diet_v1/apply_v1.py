"""Apply only audited scalar-counter spans to a compatible append/dtype fork.
Usage: python -B apply_v1.py INPUT.py NEW_OUTPUT.py; no candidate import.
"""
from pathlib import Path
import ast,collections,hashlib,json,sys
D=Path(__file__).resolve().parent;proof=json.loads((D/'SOURCE_AUDIT_v1.json').read_text())
source=Path(sys.argv[1]);out=Path(sys.argv[2]);old=Path(proof['parent_path']).read_text();text=source.read_text()
names={e['name'] for e in proof['edits']}
def signatures(s):
    t=ast.parse(s);ps={c:n for n in ast.walk(t) for c in ast.iter_child_nodes(n)}
    result=collections.Counter()
    for n in ast.walk(t):
        if isinstance(n,ast.Name) and n.id in names:
            p=ps[n];q=ps.get(p,p)
            # Do not include entire containing function/module for initializers.
            if isinstance(q,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef)):q=p
            result[(n.id,ast.dump(q,include_attributes=False))]+=1
        if isinstance(n,ast.Constant) and isinstance(n.value,str):assert not any(name in n.value for name in names)
    return result
assert signatures(text)==signatures(old),'A selected counter gained/changed a reference: re-audit required'
lines=text.splitlines(keepends=True);original_lines=list(lines)
for e in sorted(proof['edits'],key=lambda e:(e['line'],e['start']),reverse=True):
    i=e['line']-1
    assert lines[i].count(e['text'])==1,(e['line'],e['text'],'line shifted or source changed')
    lines[i]=lines[i].replace(e['text'],e['replacement'],1)
result=''.join(lines);assert len(result)==len(text) and result.count('\n')==text.count('\n')
compile(result,str(out),'exec');out.open('x').write(result)
receipt=dict(parent_path=str(source),parent_sha256=hashlib.sha256(text.encode()).hexdigest(),
 output_path=str(out),output_sha256=hashlib.sha256(result.encode()).hexdigest(),edits=len(proof['edits']),
 no_new_counter_references=True,line_preserving=True,no_candidate_import=True)
out.with_suffix('.stats_audit.json').open('x').write(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
