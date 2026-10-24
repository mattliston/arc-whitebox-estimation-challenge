"""Copy exactly the already served donor final head; all other bytes fixed."""
import ast,base64,hashlib,struct,zlib
from common import *
seal_check()
def assignment(s,name):
    nodes=[n for n in ast.parse(s).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
    assert len(nodes)==1;return nodes[0]
def decode(encoded):
    out={}
    for key,(shape,blob) in encoded.items():
        raw=zlib.decompress(base64.b64decode(blob,validate=True));count=1
        for d in shape:count*=d
        assert len(raw)==count*4
        values=struct.unpack('<'+str(count)+'f',raw);assert all(abs(x)<float('inf') for x in values)
        out[key]=(tuple(shape),raw)
    return out
parent=PARENT.read_text();donor=DONOR.read_text()
pn=assignment(parent,'_GNN_LITERAL');dn=assignment(donor,'_GNN_LITERAL')
p=decode(ast.literal_eval(pn.value));d=decode(ast.literal_eval(dn.value))
assert p.keys()==d.keys();changed=[]
for k in p:
    assert p[k][0]==d[k][0]
    if k not in ('head','head_b'):assert p[k]==d[k],k
    else:
        last=64 if k=='head' else 1;assert p[k][0]==((16,64) if k=='head' else (16,))
        assert p[k][1][:-last*4]==d[k][1][:-last*4]
        changed += [(k,i) for i in range(len(p[k][1])//4) if p[k][1][4*i:4*i+4]!=d[k][1][4*i:4*i+4]]
assert len(changed)==65
assert decode(ast.literal_eval(assignment(parent,'_FINAL_LITERAL').value))==decode(ast.literal_eval(assignment(donor,'_FINAL_LITERAL').value))
pfields={k.value:v for k,v in zip(pn.value.keys,pn.value.values)}
dfields={k.value:v for k,v in zip(dn.value.keys,dn.value.values)}
lines=parent.encode().splitlines(keepends=True);patches=[]
for key in ('head','head_b'):
    n=pfields[key];lo=sum(map(len,lines[:n.lineno-1]))+n.col_offset;hi=sum(map(len,lines[:n.end_lineno-1]))+n.end_col_offset
    old=parent.encode()[lo:hi].decode();new=ast.get_source_segment(donor,dfields[key])
    assert ast.literal_eval(old)==ast.literal_eval(n)
    patches.append(dict(key=key,start=lo,stop=hi,old=old,new=new))
candidate=parent.encode()
for q in reversed(patches):candidate=candidate[:q['start']]+q['new'].encode()+candidate[q['stop']:]
assert decode(ast.literal_eval(assignment(candidate.decode(),'_GNN_LITERAL').value))==d
restored=candidate;shift=0;inverse=[]
for q in patches:
    lo=q['start']+shift;hi=lo+len(q['new'].encode());inverse.append((lo,hi,q['old'].encode()))
    shift+=len(q['new'].encode())-(q['stop']-q['start'])
for lo,hi,old in reversed(inverse):restored=restored[:lo]+old+restored[hi:]
assert restored==PARENT.read_bytes();ast.parse(candidate)
with (HERE/'candidate.py').open('xb') as f:f.write(candidate)
write(HERE/'build.json',dict(passed=True,parent=pin(PARENT),donor=pin(DONOR),candidate=pin(HERE/'candidate.py'),
    inverse_whole_source_exact=True,changed_parameters=changed,changed_parameter_count=65,patches=patches,
    all_other_GNN_and_FINAL_bytes_exact=True,first15_head_bytes_exact=True,new_fits=0,new_estimator_calls=0,targets_decoded=False))
print(dict(built=True,candidate=pin(HERE/'candidate.py'),changed_parameters=65),flush=True)
