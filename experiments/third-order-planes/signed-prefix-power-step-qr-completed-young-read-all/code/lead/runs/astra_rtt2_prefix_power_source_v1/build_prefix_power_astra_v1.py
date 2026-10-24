"""Prospective fixed one-step signed young-prefix refinement; private files only."""
from pathlib import Path
import ast,hashlib,json
D=Path(__file__).resolve().parent;OLD=D.parent/'astra_read_tail_rank2_native_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
parent=D.parent/'astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py'
original=OLD/'fragment_rtt2_HELD_v1.py'
assert sha(parent)=='d401532639bdc3d1b294518828f4af03a873727e510e0d4ec91399b1fc169d5a'
assert sha(original)=='107057d12269776983a1a8bb261899b09d94c26fb0b150a3e71f174fc0cf7a28'
base=original.read_text()
helper='''
# ONE prospective subspace step from the exact signed young prefix.
# OFF delegates the original RTT2 basis verbatim. Rank loss retains that basis.
_RTP_ON = True
def _rtp_refine(prefix, Q, inner, old_right, hold):
    rank=int(Q.shape[1]);assert rank in (1,2)
    V=hold(prefix.T@Q)
    if inner is not None:
        assert old_right is not None and tuple(inner.shape)==(int(prefix.shape[0]),int(old_right.shape[1]))
        V=hold(V-hold(old_right@hold(inner.T@Q)))
    else:assert old_right is None
    U=hold(prefix@V)
    if inner is not None:U=hold(U-hold(inner@hold(old_right.T@V)))
    second=hold(U[:,1]) if rank==2 else hold(fnp.zeros(int(U.shape[0]),dtype=fnp.float32))
    fresh=hold(_rtt_basis(hold(U[:,0]),second,hold))
    qr_rank=int(fresh.shape[1]);fallback=qr_rank<rank
    return (Q if fallback else fresh),qr_rank,fallback

'''
marker='def _cp_slr_inner_sq(old_left, lam_old, state):'
assert base.count(marker)==1
record="record=dict(layer=layer,m=m,k=k,c15=c15,c2=c2,old_rank=rank,changed=c15<c2,basis_rank=None,old_inner_calls=0,forced_d7=False)"
newrecord=record[:-1]+',refine_rank_in=None,refine_rank_out=None,refine_qr_rank=None,refine_fallback=None,refine_old_rank=None)'
assert base.count(record)==1
anchor="        record['basis_rank']=int(Q.shape[1])\n"
insert='''        record['basis_rank']=int(Q.shape[1])
        if _RTP_ON:
            record['refine_rank_in']=int(Q.shape[1])
            record['refine_old_rank']=rank
            if int(Q.shape[1]):
                if old:
                    assert context['inner_calls']==1 and context['inner']is not None
                    prefix_old=hold(_xsg_rows(5119,state['pre_basis'],hold(live_indices[:c15])))
                    assert tuple(prefix_old.shape)==(c15,rank)
                    Q,qr_rank,fallback=_rtp_refine(prefix,Q,context['inner'],prefix_old,hold)
                else:Q,qr_rank,fallback=_rtp_refine(prefix,Q,None,None,hold)
            else:qr_rank,fallback=0,False
            record['refine_rank_out']=int(Q.shape[1])
            record['refine_qr_rank']=qr_rank;record['refine_fallback']=bool(fallback)
            assert int(Q.shape[1])==record['refine_rank_in']
'''
assert base.count(anchor)==1
source=base.replace(marker,helper+marker).replace(record,newrecord).replace(anchor,insert)
assert source.replace(helper,'').replace(newrecord,record).replace(insert,anchor)==base
ast.parse(source)
def write(path,data):
 with path.open('xb')as f:f.write(data if isinstance(data,bytes)else data.encode())
fragment=D/'fragment_prefix_power_HELD_v1.py';write(fragment,source)
for mode in('ON','OFF'):
 text=source if mode=='ON' else source.replace('_RTP_ON = True','_RTP_ON = False')
 write(D/f'candidate_prefix_power_{mode}_HELD_v1.py',parent.read_bytes()+b'\n\n'+text.encode())
out=dict(status='HELD_SOURCE_ONLY',source_parent=str(parent),source_parent_sha256=sha(parent),original_RTT2_fragment=str(original),original_RTT2_fragment_sha256=sha(original),reverse_fragment_exact=True,all_original_functions_unchanged_except_read_metadata_and_refinement=True,old_prefix_not_materialized=True,one_fixed_subspace_step=True,rank_loss_keeps_original_basis=True,pins={str(p):sha(p)for p in(fragment,D/'candidate_prefix_power_ON_HELD_v1.py',D/'candidate_prefix_power_OFF_HELD_v1.py',Path(__file__).resolve())})
write(D/'SOURCE_MANIFEST_astra_v1.json',json.dumps(out,indent=2)+'\n');print(json.dumps(out))
