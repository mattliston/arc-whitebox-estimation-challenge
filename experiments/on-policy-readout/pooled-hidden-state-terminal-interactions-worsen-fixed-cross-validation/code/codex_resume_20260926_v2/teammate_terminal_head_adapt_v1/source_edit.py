"""Change exactly the existing active terminal64 head and one bias."""
import ast
import base64
import math
import struct
import sys
import zlib


def payload_nodes(source):
    nodes=[n for n in ast.parse(source).body if isinstance(n,ast.Assign) and
           any(isinstance(t,ast.Name) and t.id=='_GNN_LITERAL' for t in n.targets)]
    assert len(nodes)==1 and isinstance(nodes[0].value,ast.Dict)
    literal=nodes[0].value
    return nodes[0],{k.value:v for k,v in zip(literal.keys,literal.values)}


def decoded_payload(source):
    node,_=payload_nodes(source)
    payload=ast.literal_eval(node.value)
    return {key:(tuple(shape),zlib.decompress(base64.b64decode(blob,validate=True)))
            for key,(shape,blob) in payload.items()}


def terminal_head(source):
    assert sys.byteorder=='little'
    values=decoded_payload(source)
    assert values['head'][0]==(16,64) and len(values['head'][1])==16*64*4
    assert values['head_b'][0]==(16,) and len(values['head_b'][1])==16*4
    return list(struct.unpack('<64f',values['head'][1][15*64*4:]))+[
           struct.unpack('<f',values['head_b'][1][15*4:])[0]]


def edit(source,head):
    assert len(head)==65 and all(math.isfinite(float(value)) for value in head)
    data=source.encode();lines=data.splitlines(keepends=True)
    literal_node,values=payload_nodes(source)
    old_payload=decoded_payload(source)
    old_head=terminal_head(source)
    replacements=[]
    for key,prefix,new_bytes in [('head',15*64*4,struct.pack('<64f',*map(float,head[:64]))),
                                 ('head_b',15*4,struct.pack('<f',float(head[64])))]:
        node=values[key]
        assert isinstance(node,ast.Tuple) and len(node.elts)==2
        blob_node=node.elts[1]
        original=old_payload[key][1]
        rebuilt=original[:prefix]+new_bytes
        assert len(rebuilt)==len(original) and rebuilt[:prefix]==original[:prefix]
        new=repr(base64.b64encode(zlib.compress(rebuilt,9)).decode('ascii'))
        replacements.append((blob_node,new))
    patches=[]
    for node,new in replacements:
        lo=sum(map(len,lines[:node.lineno-1]))+node.col_offset
        hi=sum(map(len,lines[:node.end_lineno-1]))+node.end_col_offset
        patches.append(dict(start=lo,stop=hi,old=data[lo:hi].decode(),new=new))
    candidate=data
    for patch in sorted(patches,key=lambda p:p['start'],reverse=True):
        candidate=candidate[:patch['start']]+patch['new'].encode()+candidate[patch['stop']:]
    shift=0;inverse=[]
    for patch in sorted(patches,key=lambda p:p['start']):
        lo=patch['start']+shift;hi=lo+len(patch['new'].encode())
        inverse.append((lo,hi,patch['old'].encode()))
        shift+=len(patch['new'].encode())-(patch['stop']-patch['start'])
    disabled=candidate
    for lo,hi,original in reversed(inverse):disabled=disabled[:lo]+original+disabled[hi:]
    assert disabled==data
    oldtree=ast.parse(data);newtree=ast.parse(candidate)
    newnode=next(n for n in newtree.body if isinstance(n,ast.Assign) and
                 any(isinstance(t,ast.Name) and t.id=='_GNN_LITERAL' for t in n.targets))
    # Replace the entire changed literal with the original AST: all executable
    # bodies, imports, other learned tensors, setup and arithmetic stay identical.
    newnode.value=literal_node.value
    assert ast.dump(newtree,include_attributes=False)==ast.dump(oldtree,include_attributes=False)
    new_payload=decoded_payload(candidate)
    for key,(shape,raw) in old_payload.items():
        assert new_payload[key][0]==shape
        if key=='head':assert new_payload[key][1][:15*64*4]==raw[:15*64*4]
        elif key=='head_b':assert new_payload[key][1][:15*4]==raw[:15*4]
        else:assert new_payload[key][1]==raw
    assert terminal_head(candidate)==list(struct.unpack('<65f',struct.pack('<65f',*map(float,head))))
    compile(candidate,'terminal_head_candidate','exec')
    return candidate,disabled,patches,old_head
