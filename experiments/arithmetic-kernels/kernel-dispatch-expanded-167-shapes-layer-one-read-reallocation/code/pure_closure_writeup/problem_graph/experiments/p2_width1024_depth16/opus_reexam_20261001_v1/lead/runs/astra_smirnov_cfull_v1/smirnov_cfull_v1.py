"""Inactive exact FNP-only rank40 outer circuit, current batched AB children.

No source import, model, setup or predict is run by this module. Caller supplies
the privately imported candidate module and a local certified circuit record.
"""
from collections import Counter
from fractions import Fraction
import math

def prepare(raw):
    nodes=raw['nodes'];outputs=[o['ref'] for o in raw['outputs']]
    assert all(Fraction(o['scale'])==1 for o in raw['outputs'])
    live=set()
    def visit(i):
        if i not in live:
            live.add(i)
            for a in nodes[i]['args']:visit(a)
    for i in outputs:visit(i)
    uses=Counter(a for i in live for a in nodes[i]['args'])
    destinations={}
    for i,ref in enumerate(outputs):destinations.setdefault(ref,[]).append(i)
    ops=[]
    for i in sorted(live):
        n=nodes[i];lit=n['literal']
        if n['op']=='scale':lit=float(Fraction(lit))
        ops.append((i,n['op'],tuple(n['args']),lit))
    return ops,uses,destinations

def linear(fnp,circuit,inputs,outputs,shape,stats):
    operations,uses,destinations=circuit;remaining=uses.copy();values={};owners=set()
    for ref,op,args,lit in operations:
        dest=destinations.get(ref,())
        if op=='input':value=inputs[lit]
        else:
            if dest:value=outputs[dest[0]]
            else:
                value=fnp.empty(shape,dtype=fnp.float32);owners.add(ref)
                stats['temp_live']+=math.prod(shape)*4
                stats['temp_peak']=max(stats['temp_peak'],stats['temp_live'])
            operands=[values[a]for a in args]
            if op=='add':fnp.add(operands[0],operands[1],out=value)
            elif op=='sub':fnp.subtract(operands[0],operands[1],out=value)
            elif op in ('scale','neg'):fnp.multiply(operands[0],-1 if op=='neg'else lit,out=value)
            else:raise ValueError(op)
        values[ref]=value
        for j,slot in enumerate(dest):
            if op=='input'or j:fnp.copyto(outputs[slot],value)
        for a in args:
            remaining[a]-=1
            if remaining[a]==0:
                if a in owners:owners.remove(a);stats['temp_live']-=math.prod(shape)*4
                values.pop(a)
        if remaining[ref]==0:
            if ref in owners:owners.remove(ref);stats['temp_live']-=math.prod(shape)*4
            values.pop(ref)
    assert not owners and not values and stats['temp_live']==0

"""Exact-order Smirnov linear DAG with call-local lifetime-slotted temporaries.

Inactive helper. No candidate mutation, setup, model, or runtime launch.
Compile uses source constants only. Output arrays always own their storage
outside the temporary pool, so the pool can retire immediately after a call.
"""


def prepare_slots(circuit):
    operations, uses, destinations = circuit
    remaining = uses.copy()
    locations = {}
    free = []
    count = 0
    schedule = []
    peak = 0
    live = set()
    for ref, op, args, lit in operations:
        dest = destinations.get(ref, ())
        if op == 'input':
            location = ('input', lit)
        elif dest:
            location = ('output', dest[0])
        else:
            if free:
                slot = free.pop()
            else:
                slot = count
                count += 1
            location = ('temp', slot)
            assert slot not in live
            live.add(slot)
            peak = max(peak, len(live))
        operands = tuple(locations[a] for a in args)
        schedule.append((op, operands, lit, location, tuple(dest)))
        locations[ref] = location
        for a in args:
            remaining[a] -= 1
            if remaining[a] == 0:
                where = locations.pop(a)
                if where[0] == 'temp':
                    live.remove(where[1])
                    free.append(where[1])
        if remaining[ref] == 0:
            where = locations.pop(ref)
            if where[0] == 'temp':
                live.remove(where[1])
                free.append(where[1])
    assert not locations and not live and len(free) == count
    return tuple(schedule), count, peak


def linear_slots(fnp, plan, inputs, outputs, shape, stats):
    hold=[]
    schedule, count, peak = plan
    pool = fnp.empty((count,) + tuple(shape), dtype=fnp.float32)
    temporaries = fnp.unstack(pool, axis=0)
    arrays = dict(input=inputs, output=outputs, temp=temporaries)
    for op, args, lit, location, destinations in schedule:
        value = arrays[location[0]][location[1]]
        operands = [arrays[k][i] for k, i in args]
        if op == 'add':
            hold.append(fnp.add(operands[0], operands[1], out=value))
        elif op == 'sub':
            hold.append(fnp.subtract(operands[0], operands[1], out=value))
        elif op in ('scale', 'neg'):
            hold.append(fnp.multiply(operands[0], -1 if op == 'neg' else lit, out=value))
        elif op != 'input':
            raise ValueError(op)
        for j, slot in enumerate(destinations):
            if op == 'input' or j:
                fnp.copyto(outputs[slot], value)
    stats['temp_live'] = 0
    stats['temp_peak'] = max(stats['temp_peak'], count * shape[0] * shape[1] * 4)


"""Inactive Smirnov structural views over the incumbent packed arena.

Each family is active alone. The three side intervals are disjoint within a
family; different families deliberately overlap the same arena. The caller
must retain no numerical leaf values between products and invalidate these
views when its arena owner changes. No allocation, setup, or model runs here.
"""

FAMILIES = (
    ('L4_all', 4, 40, 13, 80, 16),
    ('L5_narrow20', 5, 20, 9, 62, 9),
    ('L5_wide20', 5, 20, 11, 54, 10),
    ('L6_small', 6, 16, 4, 24, 4),
    ('L6_other', 6, 10, 5, 29, 5),
)


def family_for(L, rows, inner, cols):
    u = 1 << L
    assert not (rows % u or inner % u or cols % u)
    r, k, c = rows // u, inner // u, cols // u
    return next(f for f in FAMILIES if f[1] == L and r <= f[3] and k <= f[4] and c <= f[5])


def bind_family(fnp, module, arena, family, extra_c=None, shared_c=None):
    name, L, P, r, k, c = family
    sides = []
    offset = 0
    for kind, (rr, cc) in enumerate(((r, k), (k, c), (r, c))):
        if kind == 2 and shared_c is not None:
            sides.append(shared_c)
            continue
        count = 7 ** L * P * rr * cc
        owner = extra_c if kind == 2 and extra_c is not None else arena
        start = 0 if owner is extra_c else offset
        assert start + count <= int(owner.shape[0])
        piece = owner[start:start + count]
        T = fnp.reshape(piece, (7,) * L + (P, rr, cc))
        if owner is arena: offset += count
        data = slice(0, 7, 2) if kind == 0 else slice(0, 4)
        levels = [None, None]
        for j in range(2, L + 1):
            nr = L - j
            level = T[(data,) * nr] if nr else T
            levels.append(fnp.moveaxis(level, nr, 0) if nr else level)
        nr = L - 1
        R = T[(data,) * nr]
        V = fnp.reshape(R, (2, 2) * nr + (7, P, rr, cc))
        swap = [x for q in range(nr) for x in (2*q+1, 2*q)] if kind == 2 else list(range(2*nr))
        base = swap + list(range(2*nr, 2*nr+4))
        G = fnp.transpose(V, [base[x] for x in module._rs_nat_perm(L, 1)])
        sides.append(dict(T=T, levels=levels, G=G,
                          basis=module._rs_basis_views(kind, L, T),
                          L=L, P=P, rows=rr, cols=cc))
    return dict(name=name, owner=arena, family=family, sides=tuple(sides), elements=offset)


def slice_side(fnp, frame, P, rows, cols):
    L = frame['L']
    u = 1 << L
    assert not (rows % u or cols % u)
    r, c = rows // u, cols // u
    assert 0 < P <= frame['P'] and 0 < r <= frame['rows'] and 0 < c <= frame['cols']
    tail = (Ellipsis, slice(0, P), slice(0, r), slice(0, c))
    T = frame['T'][tail]
    chunks = [None, None]
    levels = [None, None]
    for j in range(2, L + 1):
        level = T if j == L else frame['levels'][j][tail]
        levels.append(level)
        chunks.append(fnp.unstack(level, axis=0))
    index = [slice(None)] * (2*L+2)
    index[1] = slice(0, P)
    index[L+1] = slice(0, r)
    index[2*L+1] = slice(0, c)
    G = frame['G'][tuple(index)]
    chunks[1] = fnp.unstack(G, axis=0)
    basis = [(lo[tail], hi[tail]) for lo, hi in frame['basis']]
    return T, chunks, G, basis, levels


def bind_universal(module, arena):
    module._sm40_owner=None
    module._sm40_cache={}
    module._sm40_frames={}
    module._sm40_outer={}
    module._sm40_io_cache={}
    module._sm40_outer_cache={}
    module._sm40_extra_leaf=None
    module._sm40_extra_leaf_bytes=0
    special=[f for f in FAMILIES if f[2]==20]
    assert special and len(set(f[1] for f in special))==1
    largest=max(special,key=lambda f:f[3]*f[5])
    assert all(f[3]<=largest[3] and f[5]<=largest[5] for f in special)
    extra_count=7**largest[1]*20*largest[3]*largest[5]
    extra=module.fnp.empty((extra_count,),dtype=module.fnp.float32)
    wide=bind_family(module.fnp,module,arena,largest,extra_c=extra)
    frames={largest[0]:wide}
    for f in FAMILIES:
        if f[0]==largest[0]:continue
        frames[f[0]]=bind_family(module.fnp,module,arena,f,shared_c=wide['sides'][2]) if f[2]==20 else bind_family(module.fnp,module,arena,f)
    module._sm40_extra_leaf=extra
    module._sm40_extra_leaf_bytes=extra_count*4
    module._sm40_frames=frames
    mr=max(f[3]*(1<<f[1])for f in FAMILIES)
    mk=max(f[4]*(1<<f[1])for f in FAMILIES)
    mc=max(f[5]*(1<<f[1])for f in FAMILIES)
    module._sm40_outer=dict(A=module.fnp.empty((40,mr,mk),dtype=module.fnp.float32),B=module.fnp.empty((40,mk,mc),dtype=module.fnp.float32),C=module.fnp.empty((40,mr,mc),dtype=module.fnp.float32))
    module._sm40_outer_bytes=40*(mr*mk+mk*mc+mr*mc)*4
    # Linear circuits occupy this arena only before and after child products.
    group_rows=max(mr,mc);group_cols=max(mk,mr,mc)
    group_count=125*group_rows*group_cols
    assert group_count<=int(arena.shape[0])
    base=module.fnp.reshape(arena[:group_count],(125,group_rows,group_cols))
    module._sm40_group_base=(base,module.fnp.swapaxes(base,1,2))
    module._sm40_group_cache={}
    bind_padding(module)
    bind_c_full(module,arena)
    module._sm40_owner=arena


def prepare_grouped(circuit):
    operations, uses, destinations = circuit
    depth={}; groups={}; inputs={}; outputs={}
    for ref,op,args,lit in operations:
        if op=='input':
            depth[ref]=0; inputs[ref]=lit
        else:
            depth[ref]=1+max(depth[a] for a in args)
            key=(depth[ref],op,lit if op in ('scale','neg') else None)
            groups.setdefault(key,[]).append((ref,args))
        for dest in destinations.get(ref,()):outputs[dest]=ref
    return inputs, tuple((key,tuple(nodes)) for key,nodes in sorted(groups.items(),key=lambda kv:kv[0][0])), tuple(outputs[i] for i in range(len(outputs))), uses


def linear_grouped(fnp,plan,inputs,shape,stats,out=None,prepared=None):
    input_refs,groups,output_refs,uses=plan
    remaining=uses.copy(); keep=set(output_refs)
    values={ref:inputs[i] for ref,i in input_refs.items()}
    groups_live=[]; scalar_bytes=math.prod(shape)*4
    held=[];node_at=0
    for group_index,((depth,op,lit),nodes) in enumerate(groups):
        left=[values[args[0]] for ref,args in nodes]
        right=[values[args[1]] for ref,args in nodes] if op in ('add','sub') else None
        if len(nodes)==1:
            lhs=left[0]; rhs=right[0] if right else (-1 if op=='neg' else lit)
        else:
            lhs=left; rhs=right if right else (-1 if op=='neg' else lit)
        operation={'add':fnp.add,'sub':fnp.subtract,'scale':fnp.multiply,'neg':fnp.multiply}[op]
        if prepared is None:
            result=operation(lhs,rhs)
            parts=(result,) if len(nodes)==1 else fnp.unstack(result,axis=0)
        else:
            batches,node_views=prepared
            destination=node_views[node_at] if len(nodes)==1 else batches[group_index]
            result=operation(lhs,rhs,out=destination);held.append(result)
            parts=node_views[node_at:node_at+len(nodes)]
        node_at+=len(nodes)
        for (ref,args),part in zip(nodes,parts):values[ref]=part
        # Input packing is internal to the standard ufunc for a sequence. The
        # conservative ledger includes two fully materialized operand batches.
        groups_live.append((tuple(ref for ref,args in nodes),len(nodes)*scalar_bytes))
        live_refs=set(values)
        current=0 if prepared is not None else sum(size for refs,size in groups_live if any(ref in live_refs for ref in refs))
        stats['temp_peak']=max(stats['temp_peak'],current+2*len(nodes)*scalar_bytes)
        for ref,args in nodes:
            for a in args:
                remaining[a]-=1
                if remaining[a]==0 and a not in keep:values.pop(a,None)
        del left,right,lhs,rhs,result,parts,part
    answer=fnp.stack(tuple(values[ref] for ref in output_refs),axis=0,**({}if out is None else {'out':out}))
    stats['temp_live']=0
    return answer


"""Inactive exact AA/BB pad views in P20 C scratch, retired before BB uses C."""
def bind_padding(module):
    fnp=module.fnp;owner=module._sm40_extra_leaf
    ar,ak,an=3,6,3
    _,mr,mk=module._sm40_outer['A'].shape
    _,_,mc=module._sm40_outer['B'].shape
    count_a=ar*mr*ak*mk;count_b=ak*mk*an*mc
    assert count_a+count_b<=owner.shape[0]
    A=fnp.reshape(owner[:count_a],(ar*mr,ak*mk))
    B=fnp.reshape(owner[count_a:count_a+count_b],(ak*mk,an*mc))
    module._sm40_padding=(A,B)
    module._sm40_padding_grid={}
    module._sm40_padding_cache={}


def padding_views(module,m,k,n,rank,cm,ck,cn,ar,ak,an):
    fnp=module.fnp;gridkey=(cm,ck,cn,ar,ak,an)
    if gridkey not in module._sm40_padding_grid:
        fulla,fullb=module._sm40_padding
        aa=fulla[:ar*cm,:ak*ck];bb=fullb[:ak*ck,:an*cn]
        av=tuple(cell for row in fnp.split(aa,ar,axis=0)for cell in fnp.split(row,ak,axis=1))
        bv=tuple(cell for row in fnp.split(bb,ak,axis=0)for cell in fnp.split(row,an,axis=1))
        module._sm40_padding_grid[gridkey]=(aa,bb,av,bv)
    aa,bb,av,bv=module._sm40_padding_grid[gridkey]
    key=(m,k,n,rank,cm,ck,cn,ar,ak,an)
    if key not in module._sm40_padding_cache:
        young_a=aa[:m,:k];young_b=bb[:k,:n]
        old_a=aa[:m,k:k+rank]if rank else None
        old_b=bb[k:k+rank,:n]if rank else None
        zero=[]
        if m<ar*cm:zero.append(aa[m:,:])
        if k+rank<ak*ck:zero.append(aa[:m,k+rank:])
        if k+rank<ak*ck:zero.append(bb[k+rank:,:])
        if n<an*cn:zero.append(bb[:k+rank,n:])
        module._sm40_padding_cache[key]=(young_a,young_b,old_a,old_b,tuple(zero))
    return aa,bb,av,bv,module._sm40_padding_cache[key]

def bind_c_full(module, arena):
    # Only coordinate-independent C unwind operations use family bounds.
    # Actual matmul and the final output operations retain cropped geometry.
    fnp=module.fnp
    module._sm40_cfull={}
    module._sm40_ccrop_cache={}
    full={}
    for family in FAMILIES:
        name,L,P,r,k,c=family;u=1<<L
        frame=module._sm40_frames[name]['sides'][2]
        T,chunks,G,basis,levels=slice_side(fnp,frame,P,r*u,c*u)
        ops=[]
        for j in range(L,1,-1):
            Z=chunks[j];V=levels[j]
            ops.extend(((fnp.add,Z[0],V[5:1:-3],V[5:None:-5]),
                        (fnp.add,Z[5],Z[4],Z[5]),
                        (fnp.add,Z[5],V[3:0:-2],V[2:4]),
                        (fnp.add,Z[4],Z[6],Z[1])))
        full[name]=(basis,ops)
    module._sm40_cfull=full

class Kernel:
    def __init__(self,module,record):
        self.M=module;self.fnp=module.fnp;self.record=record
        self.circuits={k:prepare_grouped(prepare(record['circuits'][k]))for k in ('L','R','P')}
        self.cache={}
        self.stats={}
    def group_views(self,kind,shape):
        key=(self.record['name'],kind,tuple(shape))
        cache=self.M._sm40_group_cache
        if key not in cache:
            groups=self.circuits[kind][1]
            sizes=[len(nodes) for _,nodes in groups]
            total=sum(sizes);assert total<=125
            base=self.M._sm40_group_base[1 if kind=='R' else 0]
            rows,cols=map(int,shape)
            assert rows<=int(base.shape[1]) and cols<=int(base.shape[2])
            allnodes=base[:total,:rows,:cols]
            cuts=[];at=0
            for size in sizes[:-1]:at+=size;cuts.append(at)
            batches=tuple(self.fnp.split(allnodes,cuts,axis=0))
            nodes=tuple(self.fnp.unstack(allnodes,axis=0))
            cache[key]=(batches,nodes)
        return cache[key]
    def c_side(self,L,P,rows,cols):
        fnp=self.fnp;M=self.M;u=1<<L;r,c=rows//u,cols//u
        frame=M._sm40_frames[self.family[0]]['sides'][2]
        assert not(rows%u or cols%u)
        assert 0<P<=frame['P'] and 0<r<=frame['rows'] and 0<c<=frame['cols']
        # Shared physical L5 C views may use distinct family unwind bounds.
        key=(id(frame),L,P,rows,cols)
        cache=M._sm40_ccrop_cache
        if key not in cache:
            T=frame['T'][..., :P, :r, :c]
            index=[slice(None)]*(2*L+2)
            index[1]=slice(0,P);index[L+1]=slice(0,r);index[2*L+1]=slice(0,c)
            G=frame['G'][tuple(index)]
            chunks=[None,fnp.unstack(G,axis=0)]
            cache[key]=(T,chunks,G)
        T,chunks,G=cache[key]
        basis,ops=M._sm40_cfull[self.family[0]]
        return T,chunks,G,basis,ops
    def side(self,kind,L,P,rows,cols):
        if kind==2:return self.c_side(L,P,rows,cols)
        key=(('shared_C20' if kind==2 and self.family[2]==20 else self.family[0]),kind,L,P,rows,cols)
        cache=self.M._sm40_cache
        if key in cache:return cache[key]
        fnp=self.fnp;M=self.M;u=1<<L;r,c=rows//u,cols//u
        frame=M._sm40_frames[self.family[0]]['sides'][kind]
        T,chunks,G,basis,levels=slice_side(fnp,frame,P,rows,cols)
        ops=[]
        if kind==0:
            for j in range(1,L+1):
                X=chunks[j]
                V=G if j==1 else levels[j]
                ops.extend(((fnp.subtract,X[4],X[0],X[5]),(fnp.subtract,V[2:7:4],X[5],V[3:0:-2])))
        elif kind==1:
            for j in range(1,L+1):
                Y=chunks[j];V=G if j==1 else levels[j]
                at=lambda sl:V[sl]
                ops.extend(((fnp.add,Y[1],Y[0],Y[5]),(fnp.subtract,at(slice(5,2,-2)),at(slice(2,6,3)),at(slice(6,3,-2)))))
        else:
            for j in range(L,1,-1):
                Z=chunks[j];V=levels[j]
                ops.extend(((fnp.add,Z[0],V[5:1:-3],V[5:None:-5]),(fnp.add,Z[5],Z[4],Z[5]),(fnp.add,Z[5],V[3:0:-2],V[2:4]),(fnp.add,Z[4],Z[6],Z[1])))
        value=(T,chunks,G,basis,ops)
        cache[key]=value;return value
    def bb_batch(self,A,B,C,L,key):
        hold=[]; fnp=self.fnp;P,m,k=map(int,A.shape);n=int(B.shape[-1]);u=1<<L
        cache=self.M._sm40_io_cache
        if key not in cache:
            TA,XA,GA,basA,opsA=self.side(0,L,P,m,k)
            TB,XB,GB,basB,opsB=self.side(1,L,P,k,n)
            TC,XC,GC,basC,opsC=self.side(2,L,P,m,n)
            Ar=fnp.reshape(A,(P,)+(2,)*L+(m//u,)+(2,)*L+(k//u,))
            Br=fnp.reshape(B,(P,)+(2,)*L+(k//u,)+(2,)*L+(n//u,))
            corner=lambda V,i,j:V[(slice(None),i)+(slice(None),)*L+(j,)]
            dA=fnp.moveaxis(GA[0:3:2],0,L+1);firstA=Ar[:,0]
            a21=corner(Ar,1,0);a22=corner(Ar,1,1)
            left=(slice(None),)*(L+2)+(0,)
            dB=fnp.moveaxis(GB[0:3:2],0,1);firstB=Br[left]
            b12=corner(Br,0,1);b22=corner(Br,1,1)
            Cr=fnp.reshape(C,(P,)+(2,)*L+(m//u,)+(2,)*L+(n//u,))
            corners=tuple(corner(Cr,i,j)for i,j in ((0,0),(0,1),(1,0),(1,1)))
            cache[key]=(TA,TB,TC,XA[1],XB[1],XC[1],basA,basB,basC,opsA,opsB,opsC,dA,firstA,a21,a22,dB,firstB,b12,b22,corners)
        TA,TB,TC,X,Y,Z,basA,basB,basC,opsA,opsB,opsC,dA,firstA,a21,a22,dB,firstB,b12,b22,corners=cache[key]
        fnp.copyto(dA,firstA);fnp.copyto(X[6],a22);hold.append(fnp.add(a21,a22,out=X[4]))
        fnp.copyto(dB,firstB);fnp.copyto(Y[3],b22);hold.append(fnp.subtract(b22,b12,out=Y[1]))
        for lo,hi in basA:hold.append(fnp.add(lo,hi,out=lo))
        for lo,hi in basB:hold.append(fnp.subtract(hi,lo,out=lo))
        for func,x,y,out in opsA:hold.append(func(x,y,out=out))
        for func,x,y,out in opsB:hold.append(func(x,y,out=out))
        hold.append(fnp.matmul(TA,TB,out=TC))
        for func,x,y,out in opsC:hold.append(func(x,y,out=out))
        for lo,hi in basC:hold.append(fnp.subtract(hi,lo,out=lo))
        tl,tr,bl,br=corners
        hold.append(fnp.add(Z[0],Z[5],out=Z[5]));hold.append(fnp.add(Z[0],Z[2],out=tl));hold.append(fnp.add(Z[5],Z[4],out=Z[5]))
        hold.append(fnp.add(Z[5],Z[3],out=tr));hold.append(fnp.add(Z[5],Z[1],out=br));hold.append(fnp.add(Z[4],Z[6],out=Z[4]));hold.append(fnp.subtract(br,Z[4],out=bl))
        return C
    def __call__(self,a,b,L,P,old=None):
        M=self.M;fnp=self.fnp;m,k=int(a.shape[0]),int(a.shape[1]);n=int(b.shape[1])
        rank=0 if old is None else int(old[0].shape[1]);ar,ak,an=self.record['shape'];u=1<<L
        cm=math.ceil(m/ar/u)*u;ck=math.ceil((k+rank)/ak/u)*u;cn=math.ceil(n/an/u)*u
        self.family=family_for(L,cm,ck,cn);P=self.family[2]
        assert tuple(b.shape)==(k,n)
        assert M._sm40_owner is M._SCRATCH[0]
        assert self.family[0]in M._sm40_frames
        if hasattr(M,'_TR_LAST'):M._TR_LAST[0]=None
        self.cache.clear();self.stats=dict(temp_live=0,temp_peak=0,leaf_bytes=M._sm40_extra_leaf_bytes)
        aa,bb,av,bv,pad_dest=padding_views(M,m,k,n,rank,cm,ck,cn,ar,ak,an)
        for zero in pad_dest[4]:fnp.copyto(zero,0)
        fnp.copyto(pad_dest[0],a);fnp.copyto(pad_dest[1],b)
        if old is not None:
            fnp.copyto(pad_dest[2],old[0]);fnp.copyto(pad_dest[3],old[1])
        outerkey=(L,cm,ck,cn,P)
        if outerkey not in M._sm40_outer_cache:
            LA=M._sm40_outer['A'][:,:cm,:ck];RB=M._sm40_outer['B'][:,:ck,:cn];PC=M._sm40_outer['C'][:,:cm,:cn]
            pc=tuple(fnp.unstack(PC,axis=0))
            cuts=list(range(P,40,P))
            parts=tuple(zip(fnp.split(LA,cuts,axis=0),fnp.split(RB,cuts,axis=0),fnp.split(PC,cuts,axis=0)))
            M._sm40_outer_cache[outerkey]=(LA,RB,PC,pc,parts)
        LA,RB,PC,pc,parts=M._sm40_outer_cache[outerkey]
        linear_grouped(fnp,self.circuits['L'],av,(cm,ck),self.stats,out=LA,prepared=self.group_views('L',(cm,ck)))
        linear_grouped(fnp,self.circuits['R'],bv,(ck,cn),self.stats,out=RB,prepared=self.group_views('R',(ck,cn)))
        self.stats['packed_bytes']=M._sm40_outer_bytes
        try:
            for index,(A,B,C)in enumerate(parts):
                self.bb_batch(A,B,C,L,(outerkey,index))
        except BaseException:
            M._sm40_cache.clear()
            raise
        finally:
            self.cache.clear();M._BH_HOLD.clear()
            if hasattr(M,'_TR_LAST'):M._TR_LAST[0]=None
        packed_out=linear_grouped(fnp,self.circuits['P'],pc,(cm,cn),self.stats,prepared=self.group_views('P',(cm,cn)))
        out=fnp.reshape(fnp.transpose(fnp.reshape(packed_out,(ar,an,cm,cn)),(0,2,1,3)),(ar*cm,an*cn))
        self.stats['conservative_peak_bytes']=self.stats['leaf_bytes']+self.stats['packed_bytes']+self.stats['temp_peak']+int(out.shape[0])*int(out.shape[1])*4
        return fnp.copy(out[:m,:n])if tuple(out.shape)!=(m,n)else out
