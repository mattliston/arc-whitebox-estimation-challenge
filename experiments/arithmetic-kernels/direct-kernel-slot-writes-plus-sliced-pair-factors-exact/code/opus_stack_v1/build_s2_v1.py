"""S2 = S1 (Full384+compactB ca337b18) + GK4 0.6 + b709 table167 (transported union) + b709 PREBIND(9) + R7 + [compactB-trim slot] + RQX v1 + R1.
R1 is placed AFTER RQX: RQX v1 redefines _rs_load_a/_rs_load_b (quad-view cache only); R1's loads are a superset (unstack views + persistent
quad-view cache), so R1 must be the live definition. Writes once; asserts every hunk against its source."""
from pathlib import Path
import hashlib,ast,json,difflib
D=Path(__file__).resolve().parent;R=D.parent
sha=lambda b:hashlib.sha256(b).hexdigest()
S1=R/'opus_f384_v1/stack/candidate_f384_compactB_S1_v1.py';B=R/'astra_smirnov_unseen_shapes_v1/candidate_smirnov_unseen_table_R7_HELD_v2.py'
P=R/'astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py';F=R/'astra_full384_smirnov_v1/candidate_full384_smirnov_HELD_v2.py'
X=R/'opus_req_diet_v1/rqx_patch_v1.py.txt';T1=R/'opus_tree_v1/fragment_tree_r1_v1.py.txt'
src={p:p.read_bytes() for p in (S1,B,P,F,X,T1)}
assert sha(src[S1]).startswith('ca337b18') and sha(src[B]).startswith('b7090c35') and sha(src[P]).startswith('2b10843a') and sha(src[F]).startswith('e33563bc')
assert sha(src[X]).startswith('a2a142e4') and sha(src[T1]).startswith('402eed38')
s1=src[S1].decode().splitlines(keepends=True);bl=src[B].decode().splitlines(keepends=True);pl=src[P].decode().splitlines(keepends=True)
def lit(lines,name):
    idx=[i for i,l in enumerate(lines) if l.startswith(name+' = ')];assert len(idx)==1,(name,idx);return idx[0],ast.literal_eval(lines[idx[0]].split(' = ',1)[1])
# GK4 literal (the one in fragment_gskv_gk11)
gi=[i for i,l in enumerate(s1) if l=='_GK4 = 1.0\n'];assert len(gi)==1;assert 'fragment_gskv_gk11' in ''.join(s1[gi[0]-4:gi[0]])
out=list(s1);out[gi[0]]='_GK4 = 0.6\n'
# Smirnov tables
i_pre,_=lit(out,'_ASTRA_SM40_PREBIND');b_pre_i,b_pre=lit(bl,'_ASTRA_SM40_PREBIND');out[i_pre]=bl[b_pre_i]
i_pl,f_pl=lit(out,'_ASTRA_SM40_PLANS');_,b_pl=lit(bl,'_ASTRA_SM40_PLANS');_,p_pl=lit(pl,'_ASTRA_SM40_PLANS')
tr=lambda k:k[:3]+(384,) if k[3]==320 else k
assert set(f_pl)=={tr(k) for k in p_pl} and all(f_pl[tr(k)]==v for k,v in p_pl.items())
union=dict(f_pl);added=[]
for k,v in b_pl.items():
    if k in p_pl:assert v==p_pl[k];continue
    t=tr(k);assert t not in union,('collision',k);union[t]=v;added.append((k,t))
assert len(union)==167 and len(added)==62
out[i_pl]='_ASTRA_SM40_PLANS = '+repr(union)+'\n'
assert ast.literal_eval(out[i_pl].split(' = ',1)[1])==union
_,fam=lit(out,'_ASTRA_SM40_FAMILIES');_,ffam=lit(F.read_text().splitlines(keepends=True),'_ASTRA_SM40_FAMILIES');assert fam==ffam
# R7 block = b709 tail beyond 2b108
assert bl[:18487]==pl[:18487] and len(pl)==18999 and bl[18491:18999]==pl[18491:18999];r7=''.join(bl[18999:]);assert '_ASTRA_R7FIRST_PARENT_BIND' in r7
slot='\n# SLOT (release teammate): compactB setup-trim fragment is appended here, before RQX v1.\n'
text=''.join(out)+r7+slot+src[X].decode()+src[T1].decode()
o=D/'candidate_S2_f384_cb_gk4_t167r7_rqx_r1_v1.py'
with o.open('x') as f:f.write(text)
compile(text,str(o),'exec')
# diff proof vs S1: only GK4, PREBIND, PLANS lines changed + appended tail
ol=text.splitlines(keepends=True);n=len(s1)
assert ol[:n]==out and [i for i in range(n) if out[i]!=s1[i]]==sorted([gi[0],i_pre,i_pl])
proof=dict(S2=str(o),S2_sha256=sha(text.encode()),lines=len(ol),
  components=dict(S1=dict(sha256=sha(src[S1]),content='2b108 + Full384 e33563bc hunks + compactB 9288d12b hunks'),
    GK4=dict(line=gi[0]+1,change='_GK4 = 1.0 -> 0.6 (fragment_gskv_gk11)'),
    b709_PREBIND=dict(line=i_pre+1,entries=len(b_pre),source='b7090c35 verbatim (R7 funding: 5 low-frequency prefetches removed)'),
    PLANS_union=dict(line=i_pl+1,keys=len(union),full384_keys=105,b709_new=62,b709_new_rank320_transported=sum(1 for k,t in added if k!=t),collisions=0),
    FAMILIES=dict(source='Full384 (L4_all inner 81)'),
    R7=dict(appended=True,sha256=sha(r7.encode()),source='b709 tail lines 19000-19016 verbatim'),
    compactB_trim_slot=dict(marker=slot.strip()),
    RQX_v1=dict(appended=True,sha256=sha(src[X])),
    R1=dict(appended=True,sha256=sha(src[T1]),note='placed after RQX: live _rs_load_a/_rs_load_b are R1 (superset of RQX loads); RQX xsg/gop2/predict-clear remain live')))
(D/'S2_BUILD_PROOF_v1.json').open('x').write(json.dumps(proof,indent=2)+'\n')
print(json.dumps({k:v for k,v in proof.items() if k!='components'}),json.dumps(proof['components']['PLANS_union']))
