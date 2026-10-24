"""Fixed shared-arena integrity checks; no network data or scores."""
from pathlib import Path
import gc
import hashlib
import json
import sys
import types
import time
import numpy as np
import flopscope as flops

D = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
digest = lambda a: hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def views(x):
    if isinstance(x,(tuple,list)):
        for v in x: yield from views(v)
    elif isinstance(x,dict):
        for v in x.values(): yield from views(v)
    elif hasattr(x,'shape') and hasattr(x,'dtype'): yield x


def main():
    seal=json.loads((D/'source_seal.json').read_text())
    for name,value in seal['files'].items(): assert sha(D/name)==value,name
    p=D.parent/'hill_climb_hk15_primitive_v1/programs.json'
    assert sha(p)=='d9d58df654b1b73a501a3e24cb31ebc6914ebfc706b2ef24f9e8455a4d729cb1'
    records=json.loads(p.read_text())['records']
    module=types.ModuleType('borrowed_fixture_kernel')
    module.__file__=str(D/'kernel_blocked.py')
    sys.modules[module.__name__]=module
    exec(compile((D/'kernel_blocked.py').read_text(),module.__file__,'exec'),module.__dict__)
    fnp=module.fnp
    specs=[(r,tuple(q*16-int(padded) for q in r['shape'])) for r in records for padded in (False,True)]
    size=max(module.layout_for_batch(r,3,shape,5)['complete_workspace_elements'] for r,shape in specs)
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        arena=fnp.empty(size,dtype=fnp.float32)
    assert ctx.summary_dict()['flops_used']==0
    fixtures=[]
    rng=np.random.default_rng(2026092816)
    for rec,shape in specs:
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            product=module.BorrowedProduct(rec,3,shape,arena)
        assert product.batch_size==5 and product.batch_count==3
        bind=int(ctx.summary_dict()['flops_used'])
        assert bind==product.bind_flops
        m,k,n=shape
        a=rng.integers(-2,3,size=(m,k)).astype(np.float32)
        b=rng.integers(-2,3,size=(k,n)).astype(np.float32)
        input_hashes=(digest(a),digest(b))
        truth=a.astype(np.float64)@b.astype(np.float64)
        with flops.BudgetContext(flop_budget=2**41,quiet=True):
            aa,bb=fnp.asarray(a),fnp.asarray(b)
            out=fnp.empty((m,n),dtype=fnp.float32)
        for x in views([product.left,product.right,product.products,product.scratch,product.work,product.pad_c]):
            assert np.shares_memory(np.asarray(x),np.asarray(arena))
        for side in product.sides:
            for x in views([side.T,side.chunks,side.pair,side.basis,side.ops]):
                assert np.shares_memory(np.asarray(x),np.asarray(arena))
        for x in (aa,bb,out): assert not np.shares_memory(np.asarray(x),np.asarray(arena))
        fixtures.append((product,aa,bb,out,truth,input_hashes,bind))
    rows=[]
    # Every descriptor survives two passes through every other overlapping descriptor.
    # Explicit nonzero contamination before each call also tests padding halo ownership.
    for repetition in range(2):
        for product,aa,bb,out,truth,input_hashes,bind in fixtures:
            with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
                contamination=fnp.asarray(np.full(size,17+repetition,dtype=np.float32))
                fnp.copyto(arena,contamination)
            contamination_bill=int(ctx.summary_dict()['flops_used'])
            with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
                returned=product(aa,bb,out)
            bill=int(ctx.summary_dict()['flops_used'])
            assert returned is out
            assert bill==module.warm_flops(product.record,3,product.shape)
            assert np.array_equal(np.asarray(out,dtype=np.float64),truth)
            assert (digest(aa),digest(bb))==input_hashes
            rows.append(dict(name=product.record['name'],shape=product.shape,repetition=repetition,
                             bind_flops=bind,warm_flops=bill,contamination_flops=contamination_bill,
                             output_sha256=digest(out),exact=True))
    for j in range(12): assert rows[j]['output_sha256']==rows[j+12]['output_sha256']
    result=dict(passed=True,rows=rows,arena_elements=size,descriptors=12,
                no_network_data=True,source_seal_sha256=sha(D/'source_seal.json'))
    with (D/'shared_fixtures.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(passed=True,calls=len(rows),arena_elements=size)),flush=True)
    production_equivalence(module,records)


def production_equivalence(module,records):
    reference_path=D.parent/'hill_climb_hk15_integration_v1/kernel_borrowed.py'
    assert sha(reference_path)=='06676cb661b818a4587b3f796d94dbbcaee24b322a9c7cf32949fe487c5e06c3'
    reference=types.ModuleType('reference_unblocked_v1')
    reference.__file__=str(reference_path)
    sys.modules[reference.__name__]=reference
    exec(compile(reference_path.read_text(),str(reference_path),'exec'),reference.__dict__)
    old_plan_path=D.parent/'hill_climb_hk15_primitive_v1/representative.json'
    assert sha(old_plan_path)=='72f75809240350ead5e82a2d9625e10172c6d18af81ccdd7634af06c79504957'
    old_plan=json.loads(old_plan_path.read_text())
    fixed=json.loads((D/'fixed_plan.json').read_text())
    blocked={tuple(r['key']) for r in fixed['descriptors'] if r['batch_size']<15}
    chosen=[]
    for depth in (6,5):
        choices=[r for r in old_plan['ordered_choices']
                 if r['alternative']['depth']==depth and tuple(r['key']) in blocked]
        assert choices
        chosen.append(choices[0])
    by_name={r['name']:r for r in records}
    rng=np.random.default_rng(2026092817)
    rows=[]
    for choice in chosen:
        rows.append(compare_case(module,reference,by_name[choice['alternative']['name']],choice,rng))
        gc.collect()
    result=dict(passed=True,rows=rows,selection='largest original complete workspace among blocked cases at each depth6 then5',
                source_seal_sha256=sha(D/'source_seal.json'),reference_kernel_sha256=sha(reference_path),
                no_network_data=True)
    with (D/'equivalence.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(production_byte_parity=True,cases=len(rows))),flush=True)


def compare_case(module,reference,rec,choice,rng):
    shape=tuple(choice['key'][1:]);depth=choice['alternative']['depth']
    fnp=module.fnp
    old_layout=reference.borrowed_layout(rec,depth,shape)
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        old_arena=fnp.empty(old_layout['required_arena_elements'],dtype=fnp.float32)
        new_arena=fnp.empty(module.ORIGINAL_ARENA_ELEMENTS,dtype=fnp.float32)
    products={};bind={}
    for name,cls,arena in (('v1',reference.BorrowedProduct,old_arena),('v2',module.BorrowedProduct,new_arena)):
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            products[name]=cls(rec,depth,shape,arena)
        bind[name]=int(ctx.summary_dict()['flops_used'])
        assert bind[name]==products[name].bind_flops
    assert products['v2'].batch_size==5
    m,k,n=shape
    a=rng.standard_normal((m,k)).astype(np.float32)
    b=rng.standard_normal((k,n)).astype(np.float32)
    input_hashes=(digest(a),digest(b))
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        aa,bb=fnp.asarray(a),fnp.asarray(b)
        out=fnp.empty((m,n),dtype=fnp.float32)
    for x in (aa,bb,out):
        assert not np.shares_memory(np.asarray(x),np.asarray(old_arena))
        assert not np.shares_memory(np.asarray(x),np.asarray(new_arena))
    hashes={};runs=[]
    for repetition in range(2):
        for arm in (('v1','v2') if repetition==0 else ('v2','v1')):
            with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
                c0=time.process_time();t0=time.monotonic()
                returned=products[arm](aa,bb,out)
                cpu=time.process_time()-c0;wall=time.monotonic()-t0
            assert returned is out and np.isfinite(np.asarray(out)).all()
            bill=int(ctx.summary_dict()['flops_used'])
            assert bill==module.warm_flops(rec,depth,shape)==reference.warm_flops(rec,depth,shape)
            value=digest(out)
            if hashes:assert value==next(iter(hashes.values()))
            hashes[arm]=value
            assert (digest(aa),digest(bb))==input_hashes
            runs.append(dict(arm=arm,repetition=repetition,flops=bill,cpu_s=cpu,wall_s=wall,sha256=value))
    row=dict(key=choice['key'],program=rec['name'],depth=depth,batch_size=5,
             bytes_exact=True,bind_flops=bind,runs=runs)
    print(json.dumps(dict(event='production_equivalence',key=choice['key'],passed=True,runs=runs)),flush=True)
    return row


if __name__=='__main__': main()
