"""Independent integer cost, memory interval and full-census reconciliation."""
import gzip,hashlib,json,resource,time
from common import *
def main():
    seal_check();t=time.monotonic();j=read(HERE/'audit_result.json')
    old=read(D/'all_positive_leaf_catalog_v1/reprice_result.json')['families'][0]['records']
    records={r['name']:r for r in read(A/'smirnov_executable_transpose_certificate_v1.json')['records']}
    checked=set();placements=0;eligible=0;totals=[]
    for net,parent in zip(j['networks'],old):
        assert net['rank']==parent['rank'] and net['complete_cold_flops']==parent['complete_cold_flops']
        parents={c['index']:c for c in parent['details']};assert len(net['details'])==295==len(parents)
        saved=[]
        for row in net['details']:
            c=parents[row['index']]
            assert row['shape']==c['shape'] and row['caller']==c['caller'] and row['old_flops']==c['actual_call_flops']
            dims=row['shape'];axes=[i for i,d in enumerate(dims) if d>1024]
            name={0:'signed_executable_2',1:'signed_executable_1',2:'signed_executable_0'}[axes[0]] if len(axes)==1 and min(dims)>=64 else None
            assert row['program']==name
            if name is None:assert row['saved']==0 and not row['selected'];saved.append(0);continue
            eligible+=1;p=row['price'];rec=records[name];a,b,d=rec['shape']
            units=[2*a*a*a,2*b*b*b,2*d*d*d]
            norm=[((x-1)//u+1)*u for x,u in zip(dims,units)]
            assert norm==p['normalized_shape'] and p['shape']==dims and p['program']==name
            linear=0
            for depth in [1,2,3]:
                sizes=[norm[0]//a**depth,norm[1]//b**depth,norm[2]//d**depth]
                linear+=40**(depth-1)*(rec['arithmetic'][0]*sizes[0]*sizes[1]+rec['arithmetic'][1]*sizes[1]*sizes[2]+rec['arithmetic'][2]*sizes[0]*sizes[2])
            x,y,z=[norm[i]//units[i] for i in range(3)]
            leaf=40**3*(7*x*z*(2*y-1)+4*x*y+4*y*z+7*x*z)
            arithmetic=linear+leaf;ops=p['operation_flops']
            assert arithmetic==p['arithmetic_flops']==sum(v for k,v in ops.items() if k not in ('reshape','copyto','zeros'))
            copies=(dims[0]*dims[1] if dims[:2]!=norm[:2] else 0)+(dims[1]*dims[2] if dims[1:]!=norm[1:] else 0)
            assert ops.get('copyto',0)==copies and ops.get('zeros',0)==0
            assert p['flops']==sum(ops.values()) and p['counted_calls']==sum(p['operation_calls'].values())
            assert p['cold_flops']==p['flops']-p['first_fit_binding_flops']+p['placed_binding_flops']
            assert p['binding_delta']==p['placed_binding_flops']-p['first_fit_binding_flops']
            assert p['workspace_bytes']==p['arena_bytes']+p['padding_bytes']+p['padded_output_bytes']
            padding=4*((norm[0]*norm[1] if dims[:2]!=norm[:2] else 0)+(norm[1]*norm[2] if dims[1:]!=norm[1:] else 0))
            assert p['padding_bytes']==padding and p['padded_output_bytes']==4*norm[0]*norm[2]
            assert p['fits']==(p['workspace_bytes']<=1168900096)
            key=tuple(dims)
            if key not in checked:
                verify(p['placement_file']);v=json.loads(gzip.decompress(Path(p['placement_file']['path']).read_bytes()))
                assert hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()==p['placement_sha256']
                assert len(v)==p['placement_count'];placements+=len(v)
                assert len({r['id'] for r in v})==len(v)
                assert p['arena_bytes']==4*max(r['offset']+r['size'] for r in v)
                orig={(r['first_fit_offset'],tuple(r['shape'])):r['size'] for r in v}
                placed={(r['offset'],tuple(r['shape'])):r['size'] for r in v}
                assert sum(orig.values())==p['first_fit_binding_flops'] and sum(placed.values())==p['placed_binding_flops']
                events=[]
                for r in v:
                    size=1
                    for n in r['shape']:size*=n
                    assert size==r['size'] and r['offset']>=0 and 0<r['start_time']<r['end_time']
                    events.extend([(r['start_time'],1,r),(r['end_time'],-1,r)])
                active={};live=peak=0
                for _,sign,r in sorted(events,key=lambda e:(e[0],e[1])):
                    if sign<0:assert active.pop(r['id'])==r;live-=r['size']
                    else:
                        for q in active.values():assert r['offset']+r['size']<=q['offset'] or q['offset']+q['size']<=r['offset']
                        active[r['id']]=r;live+=r['size'];peak=max(peak,live)
                assert not active and live==0 and 4*peak==p['peak_live_arena_bytes']
                checked.add(key)
            s=max(0,row['old_flops']-p['cold_flops']) if p['fits'] else 0
            assert s==row['saved'] and row['selected']==(s>0);saved.append(s)
        s=sum(saved);assert s==net['saved'] and net['new_total']==net['complete_cold_flops']-s
        assert net['saving_fraction']==s/net['complete_cold_flops']
        assert net['passes']==(100*s>=3*net['complete_cold_flops']) and net['selected_count']==sum(x>0 for x in saved)
        totals.append({k:net[k] for k in ('rank','saved','saving_fraction','selected_count','passes')})
    assert j['opportunity_gate_passed']==all(n['passes'] for n in j['networks'])
    if not j['opportunity_gate_passed']:assert not j['native_fixtures'] and not list(HERE.glob('*_plan.json'))
    else:
        for f in j['native_fixtures']:
            candidates=[dict(c,rank=n['rank']) for n in j['networks'] for c in n['details'] if c['selected'] and c['program']==f['program']]
            r=sorted(candidates,key=lambda r:(-r['saved'],r['rank'],r['index']))[0]
            assert all(r[k]==f[k] for k in ('shape','saved','rank','index'));verify(f['plan'])
    assert len(checked)==j['unique_shapes']
    result=dict(completed=True,calls=590,eligible_calls=eligible,unique_shapes=len(checked),lifetime_intervals=placements,networks=totals,opportunity_gate_passed=j['opportunity_gate_passed'],audit_result=pin(HERE/'audit_result.json'),new_model_calls=0,wall_s=time.monotonic()-t,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    write(HERE/'replay_result.json',result);print(result,flush=True)
if __name__=='__main__':main()
