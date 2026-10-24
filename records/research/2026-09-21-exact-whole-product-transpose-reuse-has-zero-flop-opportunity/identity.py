"""Research-only streaming fingerprints and explicit whole-product classes."""
import hashlib

BLOCK_BYTES=1024*1024

def fingerprint(value):
    import numpy as np
    a=np.asarray(value)
    assert a.ndim==2 and a.dtype==np.float32
    h=hashlib.sha256();zero=True;payload=0;largest=0
    if a.shape[1]:
        rows=max(1,BLOCK_BYTES//(int(a.shape[1])*a.dtype.itemsize))
        for start in range(0,a.shape[0],rows):
            block=np.ascontiguousarray(a[start:start+rows])
            assert np.isfinite(block).all(),'Nonfinite research operand'
            zero=zero and bool(np.all(block==0))
            data=block.tobytes(order='C');assert len(data)<=BLOCK_BYTES
            h.update(data);payload+=len(data);largest=max(largest,len(data))
    assert payload==a.size*a.dtype.itemsize
    return dict(shape=list(a.shape),dtype=a.dtype.str,sha256=h.hexdigest(),bytes=payload,all_zero=zero,finite=True,max_block_bytes=largest)

def signature(a):return (tuple(a['shape']),a['dtype'],a['sha256'])

def key(row,transpose=False):
    direct=(signature(row['a']),signature(row['b']))
    reverse=(signature(row['bt']),signature(row['at']))
    flipped=transpose and reverse<direct
    return (reverse if flipped else direct),bool(flipped)

def group_records(rows,transpose=False):
    groups={};duplicates=[];removed=set();zero=[]
    for i,row in enumerate(rows):
        assert row['index']==i
        k,flip=key(row,transpose);out=row['ct'] if flip else row['c']
        if row['a']['all_zero'] or row['b']['all_zero']:
            zero.append(i);removed.add(i)
        if k in groups:
            first=groups[k][0]
            duplicates.append(dict(index=i,first=first['index'],orientation_flipped=flip,
                output_fingerprint_equal=signature(out)==first['oriented_output'],flops=row['flops']))
            removed.add(i)
        groups.setdefault(k,[]).append(dict(index=i,flipped=flip,oriented_output=signature(out)))
    duplicate_indices={r['index'] for r in duplicates}
    # Retain a copied first result until its last use, with free lookup/storage.
    lifetimes=[];events={}
    for members in groups.values():
        if len(members)<2:continue
        first=members[0]['index'];last=members[-1]['index']
        if first in zero:continue
        size=rows[first]['c']['bytes'];events[first]=events.get(first,0)+size;events[last]=events.get(last,0)-size
        lifetimes.append(dict(first=first,last=last,bytes=size,uses=len(members)))
    live=0;peak=0
    for i in range(len(rows)):
        live+=events.get(i,0);peak=max(peak,live)
    assert live==0
    return dict(transpose_equivalence_included=transpose,unique_classes=len(groups),duplicates=duplicates,zero_indices=zero,
        duplicate_indices=sorted(duplicate_indices),removed_indices=sorted(removed),
        duplicate_removable_flops=sum(rows[i]['flops'] for i in duplicate_indices),
        zero_removable_flops=sum(rows[i]['flops'] for i in zero),
        optimistic_union_removed_flops=sum(rows[i]['flops'] for i in removed),
        hypothetical_copy_lifetimes=lifetimes,hypothetical_retained_peak_bytes=peak,
        free_lookup_copy_transpose_storage_assumed=True,not_a_candidate_bill=True)
