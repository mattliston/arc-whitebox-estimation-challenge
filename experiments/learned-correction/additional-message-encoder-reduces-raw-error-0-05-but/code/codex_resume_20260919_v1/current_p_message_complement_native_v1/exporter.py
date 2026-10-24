"""Replace the one packed complement-encoder AST value span, preserving all other bytes."""
import ast
from common import EKEYS

def edge_spans(source):
    nodes=[n.value for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='_GNN_LITERAL']
    assert len(nodes)==1 and isinstance(nodes[0],ast.Dict)
    lines=source.splitlines(keepends=True);spans={}
    for k,v in zip(nodes[0].keys,nodes[0].values):
        name=ast.literal_eval(k)
        start=sum(map(len,lines[:v.lineno-1]))+v.col_offset
        end=sum(map(len,lines[:v.end_lineno-1]))+v.end_col_offset
        text=source[start:end];assert text==ast.get_source_segment(source,v)
        assert name not in spans;spans[name]=(start,end,text)
    assert set(EKEYS)<=set(spans)
    return spans

def replace_values(source,values):
    assert set(values)==set(EKEYS)
    spans=edge_spans(source);result=source
    for start,end,value in sorted((spans[k][0],spans[k][1],values[k]) for k in EKEYS)[::-1]:
        ast.literal_eval(value)
        result=result[:start]+value+result[end:]
    ast.parse(result)
    original={k:spans[k][2] for k in EKEYS}
    new_spans=edge_spans(result);inverse=result
    for start,end,value in sorted((new_spans[k][0],new_spans[k][1],original[k]) for k in EKEYS)[::-1]:
        inverse=inverse[:start]+value+inverse[end:]
    assert inverse==source
    for k in set(spans)-set(EKEYS):assert new_spans[k][2]==spans[k][2]
    return result

def literal(source):
    return {k:ast.literal_eval(v[2]) for k,v in edge_spans(source).items()}
