from pathlib import Path
import ast,hashlib,json
D=Path(__file__).resolve().parent;s=(D/'smirnov_arena_fused_v4.py').read_text();lines=s.splitlines(True);tree=ast.parse(s);changes=[];count=0
for fn in ast.walk(tree):
 if not isinstance(fn,ast.FunctionDef)or fn.name not in ('linear_slots','bb_batch'):continue
 first=fn.body[0];indent=' '*(first.col_offset)
 changes.append((first.lineno-1,first.lineno-1,indent+'hold=[]\n'))
 for node in ast.walk(fn):
  if not isinstance(node,ast.Expr)or not isinstance(node.value,ast.Call)or not any(k.arg=='out'for k in node.value.keywords):continue
  call=node.value;assert node.lineno==node.end_lineno
  changes.append((node.lineno-1,node.col_offset,node.end_col_offset,'hold.append('+ast.get_source_segment(s,call)+')'));count+=1
# Source spans preserve semicolon statements and parent operation ordering.
for change in sorted(changes,key=lambda z:(z[0],z[1]),reverse=True):
 if len(change)==3:lines.insert(change[0],change[2])
 else:i,a,b,repl=change;lines[i]=lines[i][:a]+repl+lines[i][b:]
out=''.join(lines);ast.parse(out);p=D/'smirnov_arena_fused_v5.py';p.write_text(out)
proof=dict(parent_sha256=hashlib.sha256(s.encode()).hexdigest(),candidate_sha256=hashlib.sha256(out.encode()).hexdigest(),wrapped_source_calls=count,scope='Only local lists retain fresh return proxies of out= operations until linear circuit or BB batch returns. Same array allocations, operation order and arithmetic.')
(D/'ALIAS_HOLD_SOURCE_v5.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
