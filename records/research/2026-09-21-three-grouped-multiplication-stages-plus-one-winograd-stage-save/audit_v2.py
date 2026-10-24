import gzip,hashlib,json,resource,time
from fractions import Fraction
from common import *
from loader_v2 import environment
from schedule import NormalizedProduct,orientation,normalized_shape,arithmetic_formula

LIMIT=1168900096
def exact_circuit(c,inputs):
    values=[]
    for node in c['nodes']:
        op=node['op'];a=node['args']
        if op=='input':v=inputs[int(node['literal'])]
        elif op=='zero':v=Fraction(0)
        elif op=='add':v=values[a[0]]+values[a[1]]
        elif op=='sub':v=values[a[0]]-values[a[1]]
        elif op=='scale':v=values[a[0]]*Fraction(node['literal'])
        elif op=='neg':v=-values[a[0]]
        else:raise ValueError(op)
        values.append(v)
    return [values[o['ref']]*Fraction(o['scale']) for o in c['outputs']]
def basis_certificate(record):
    m,k,n=record['shape'];count=0
    for a in range(m*k):
        av=[Fraction(int(i==a)) for i in range(m*k)]
        la=exact_circuit(record['circuits']['L'],av)
        for b in range(k*n):
            bv=[Fraction(int(i==b)) for i in range(k*n)]
            rb=exact_circuit(record['circuits']['R'],bv)
            out=exact_circuit(record['circuits']['P'],[x*y for x,y in zip(la,rb)])
            expected=[Fraction(int(a//k==i and b%n==j and a%k==b//n)) for i in range(m) for j in range(n)]
            assert out==expected,(record['name'],a,b);count+=1
    return count
def old_replay(record):
    e=environment();meter=e['meter'];kernel=e['GlobalBatch'](record)
    a,b,out=[e['Symbol'](s,meter) for s in [(1024,832),(832,8192),(1024,8192)]]
    rows=[]
    for repeat in ['cold','warm']:
        kernel.arena.reset_trace();meter.reset();kernel(a,b,out)
        high,placement=e['place_largest_first'](kernel.arena.intervals)
        original={(r['first_fit_offset'],tuple(r['shape'])):r['size'] for r in kernel.arena.intervals}
        placed={(r['offset'],tuple(r['shape'])):r['size'] for r in placement}
        cost=meter.record()['flops']+(sum(placed.values())-sum(original.values()) if repeat=='cold' else 0)
        expected=7258335196 if repeat=='cold' else 6611249012
        assert cost==expected and 4*high==1115801600 and kernel.arena.padding_bytes==46283536
        rows.append(dict(repetition=repeat,flops=cost,arena_bytes=4*high,padding_bytes=kernel.arena.padding_bytes))
    return rows
def price(shape,record,keep_plan=False):
    e=environment();meter=e['meter'];arena=e['TraceArena']();meter.reset()
    kernel=NormalizedProduct(record,arena,meter,e['Block'],e['prepare'],e['linear'],e['streamed_leaf'])
    m,k,n=shape;a=e['Symbol']((m,k),meter);b=e['Symbol']((k,n),meter)
    result=kernel(a,b);assert result.shape==(m,n)
    high,placement=e['place_largest_first'](arena.intervals)
    original={(r['first_fit_offset'],tuple(r['shape'])):r['size'] for r in arena.intervals}
    placed={(r['offset'],tuple(r['shape'])):r['size'] for r in placement}
    data=meter.record();binding_delta=sum(placed.values())-sum(original.values())
    arithmetic=sum(v for key,v in data['operation_flops'].items() if key not in ('reshape','copyto','zeros'))
    assert arithmetic==arithmetic_formula(shape,record),(shape,arithmetic,arithmetic_formula(shape,record))
    assert set(data['operation_flops'])<=set(['reshape','copyto','zeros','add','subtract','multiply','matmul'])
    expected_copy=(m*k if (m,k)!=normalized_shape(shape,record['shape'])[:2] else 0)
    norm=normalized_shape(shape,record['shape'])
    expected_copy+=(k*n if (k,n)!=norm[1:] else 0)
    assert data['operation_flops'].get('copyto',0)==expected_copy
    data.update(shape=shape,normalized_shape=norm,program=record['name'],arithmetic_flops=arithmetic,
                cold_flops=data['flops']+binding_delta,binding_delta=binding_delta,
                first_fit_binding_flops=sum(original.values()),placed_binding_flops=sum(placed.values()),
                arena_bytes=4*high,padding_bytes=kernel.padding_bytes,padded_output_bytes=kernel.output_bytes,
                workspace_bytes=4*high+kernel.padding_bytes+kernel.output_bytes,
                peak_live_arena_bytes=4*arena.peak_live,placement_count=len(placement),
                placement_sha256=hashlib.sha256(json.dumps(placement,sort_keys=True).encode()).hexdigest())
    data['fits']=data['workspace_bytes']<=LIMIT
    if keep_plan:data['placement']=placement
    return data
def main():
    seal_check();started=time.monotonic()
    programs={r['name']:r for r in read(A/'smirnov_executable_transpose_certificate_v1.json')['records']}
    certificates={n:basis_certificate(programs[n]) for n in ('signed_executable_0','signed_executable_1','signed_executable_2')}
    inherited=old_replay(programs['signed_executable_0'])
    parents=read(D/'all_positive_leaf_catalog_v1/reprice_result.json')['families'][0]['records']
    cache={};nets=[]
    for net in parents:
        details=[]
        for call in sorted(net['details'],key=lambda r:r['index']):
            shape=tuple(call['shape']);name=orientation(shape)
            row=dict(index=call['index'],shape=shape,caller=call['caller'],old_flops=call['actual_call_flops'],program=name)
            if name:
                key=(shape,name)
                if key not in cache:
                    q=price(shape,programs[name],True)
                    placement=q.pop('placement')
                    path=HERE/('placement_'+'_'.join(map(str,shape))+'.json.gz')
                    with path.open('xb') as f:f.write(gzip.compress(json.dumps(placement,sort_keys=True).encode(),mtime=0))
                    q['placement_file']=pin(path)
                    cache[key]=q;print(dict(priced=shape,program=name,cold_flops=q['cold_flops'],workspace_bytes=q['workspace_bytes']),flush=True)
                q=cache[key];saved=max(0,row['old_flops']-q['cold_flops']) if q['fits'] else 0
                row.update(price=q,selected=saved>0,saved=saved)
            else:row.update(selected=False,saved=0)
            details.append(row)
        saving=sum(r['saved'] for r in details)
        nets.append(dict(rank=net['rank'],complete_cold_flops=net['complete_cold_flops'],calls=len(details),saved=saving,saving_fraction=saving/net['complete_cold_flops'],new_total=net['complete_cold_flops']-saving,passes=saving>=.03*net['complete_cold_flops'],selected_count=sum(r['selected'] for r in details),details=details))
    passed=all(n['passes'] for n in nets)
    selected=[]
    if passed:
        for name in sorted(programs):
            rows=[dict(r,rank=n['rank']) for n in nets for r in n['details'] if r['selected'] and r['program']==name]
            if rows:
                winner=sorted(rows,key=lambda r:(-r['saved'],r['rank'],r['index']))[0]
                detailed=price(tuple(winner['shape']),programs[name],True)
                write(HERE/(name+'_plan.json'),detailed)
                selected.append(dict(program=name,rank=winner['rank'],index=winner['index'],shape=winner['shape'],saved=winner['saved'],plan=pin(HERE/(name+'_plan.json'))))
    result=dict(completed=True,opportunity_gate_passed=passed,threshold_fraction=.03,native_fixtures=selected,certificates=certificates,old_replay=inherited,unique_shapes=len(cache),networks=nets,new_model_calls=0,weight_arrays_read=0,truth_arrays_read=0,candidate=False,wall_s=time.monotonic()-started,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),source_seal=pin(HERE/'sources_sealed.json'),qualification='Concrete cold symbolic schedules and workspace; no native or autonomous estimator qualification.')
    write(HERE/'audit_result.json',result)
    print({k:v for k,v in result.items() if k not in ('networks','source_seal')},flush=True)
    for n in nets:print({k:v for k,v in n.items() if k!='details'},flush=True)
if __name__=='__main__':main()
