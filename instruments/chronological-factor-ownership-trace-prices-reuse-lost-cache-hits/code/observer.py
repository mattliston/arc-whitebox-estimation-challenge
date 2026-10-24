"""Passive native source-call chronology. No numerical values or strong array registry."""
import functools
import json
import sys
import weakref

MAX_EVENTS = 100000
MAX_BYTES = 64000000
OWNERS = ('_mm','_ipt_product_ab','_cp_d6_shared','_ks_product','_d7_product',
          '_d7_product_kpad','_w5_port_product','_w5_port_product_body','_bb_product',
          '_push6_padded','_ipt_leaf_unwind_ab')
BLOCKING = set(OWNERS)-{'_mm'}
WARM = {'_ipt_leaf_unwind_ab','_ks_unwind','_ipt_forward_a_ab','_ipt_forward_b_ab',
        '_rs_load_a','_rs_load_b','_rs_out_quadrants','_ipt_bits'}
COLD = {'_ipt_side','_ipt_side_p','_ipt_view','_lvf_rs_level1','_rs_level1'}


class NativeTrace:
    def __init__(self, mod, budget_module, path):
        self.mod=mod; self.budget=budget_module; self.path=path
        self.events=0;self.bytes=0;self.seq=0;self.next_token=0
        self.refs={};self.stack=[];self.mm=[];self.roots={};self.saved={}
        self.ctx=None;self.file=None;self.restored=False
        self.source=mod.__file__;self.predict_codes=set()
        f=mod._predict_binary
        while True:
            if hasattr(f,'__code__') and f.__code__.co_name=='_predict_binary':
                self.predict_codes.add(f.__code__)
            if not hasattr(f,'__wrapped__'):break
            f=f.__wrapped__
        assert self.predict_codes

    def token(self,a):
        ident=id(a);old=self.refs.get(ident)
        if old is not None and old[0]() is a:return old[1]
        self.next_token+=1;t=self.next_token
        def gone(ref,key=ident):
            old=self.refs.get(key)
            if old is not None and old[0] is ref:self.refs.pop(key,None)
        self.refs[ident]=(weakref.ref(a,gone),t)
        return t

    def marker(self):
        x=self.mod._TR_LAST[0]
        return None if x is None else [self.token(x[0]),list(map(int,x[1]))]

    def descriptor(self,a):
        if not hasattr(a,'shape') or not hasattr(a,'dtype'):return None
        return dict(token=self.token(a),shape=list(map(int,a.shape)),dtype=str(a.dtype))

    def source_stack(self):
        f=sys._getframe(1);chain=[];layer=None
        try:
            while f is not None:
                if f.f_code in self.predict_codes:
                    layer=f.f_locals.get('l');break
                if f.f_code.co_filename==self.source:
                    chain.append([f.f_code.co_name,f.f_lineno])
                f=f.f_back
            return chain,layer
        finally:del f

    def emit(self,kind,**data):
        row=dict(event=self.events,kind=kind,**data)
        b=(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n').encode()
        if self.events>=MAX_EVENTS or self.bytes+len(b)>MAX_BYTES:raise RuntimeError('trace_cap')
        self.file.write(b);self.events+=1;self.bytes+=len(b)

    def wrap(self,name,original):
        @functools.wraps(original)
        def call(*args,**kwargs):
            self.seq+=1;cid=self.seq;chain,layer=self.source_stack()
            parent=self.stack[-1][0] if self.stack else None
            root=self.mm[0] if self.mm else None
            is_root=name=='_mm' and root is None
            if is_root:root=cid
            grant=self.mod._W5_PORT_SCOPE
            grant_active=bool(grant is not None and grant.get('grant') is not None)
            blocked=[x for x in chain if x[0] in BLOCKING]
            if is_root:self.roots[cid]=dict(warm=0,cold=0,other=0,products=0,unsupported_products=0)
            before=self.marker()
            self.emit('begin',call=cid,name=name,parent=parent,root=root,layer=layer,
                      is_root=is_root,grant=grant_active,blocked=blocked,
                      inputs=[self.descriptor(a) for a in args[:3]],marker=before)
            self.stack.append((cid,name))
            if name=='_mm':self.mm.append(cid)
            ok=False
            try:
                result=original(*args,**kwargs);ok=True;return result
            finally:
                self.emit('end',call=cid,name=name,root=root,ok=ok,marker=self.marker(),
                          charges=self.roots.get(cid) if is_root else None)
                assert self.stack.pop()==(cid,name)
                if name=='_mm':assert self.mm.pop()==cid
        return call

    def forward(self,original):
        @functools.wraps(original)
        def call(a,L,m,k):
            before=self.marker();frame=[int(L),int(m),int(k)];token=self.token(a)
            old=self.mod._TR_LAST[0]
            hit=old is not None and old[0] is a and old[1]==(L,m,k)
            del old
            stats=dict(self.mod._TR_STATS)
            paid=int(self.ctx._flops_used)
            root=self.mm[0] if self.mm else None
            warm_before=self.roots[root]['warm'] if root is not None else None
            self.emit('a_begin',root=root,call=self.stack[-1][0] if self.stack else None,
                      input=token,frame=frame,actual_hit=hit,marker=before)
            result=original(a,L,m,k)
            assert self.mod._TR_STATS['reused']-stats['reused']==int(hit)
            assert self.mod._TR_STATS['computed']-stats['computed']==int(not hit)
            after=self.marker();assert after==[token,frame]
            self.emit('a_end',root=root,input=token,frame=frame,actual_hit=hit,
                      marker=after,paid=int(self.ctx._flops_used)-paid,
                      warm=None if root is None else self.roots[root]['warm']-warm_before)
            return result
        return call

    def charge(self,original):
        def call(ctx,op_name,flop_cost,subscripts,shapes,**kwargs):
            before=ctx._flops_used
            result=original(ctx,op_name,flop_cost,subscripts,shapes,**kwargs)
            paid=int(ctx._flops_used-before)
            root=self.mm[0] if self.mm else None
            if root is None:return result
            chain,_=self.source_stack();names=[x[0] for x in chain]
            inner=names[0] if names else ''
            cold=any(n in COLD for n in names)
            warm=(inner in WARM or inner=='_ks_product' and op_name in ('matmul','add')) and not cold
            record=self.roots[root];record['cold' if cold else 'warm' if warm else 'other']+=paid
            if op_name in ('matmul','dot','einsum'):
                ss=[list(map(int,s)) for s in shapes];dd=list(map(str,kwargs.get('dtypes') or ()))
                supported=False;shape=None;depth=None
                if op_name=='matmul' and len(ss)==2 and dd and all(d=='float32' for d in dd):
                    a,b=ss;pre=a[:-2]
                    if pre and pre==b[:-2] and all(d==7 for d in pre) and inner in ('_ipt_leaf_unwind_ab','_ks_product'):
                        depth=len(pre);u=2**depth;shape=[a[-2]*u,a[-1]*u,b[-1]*u];supported=True
                record['products']+=1;record['unsupported_products']+=int(not supported)
                ks=next((cid for cid,n in reversed(self.stack) if n=='_ks_product'),None)
                self.emit('product',root=root,call=self.stack[-1][0] if self.stack else None,
                          ks=ks,op=op_name,shapes=ss,dtypes=dd,inner=inner,
                          supported=supported,shape=shape,depth=depth,paid=paid)
            return result
        return call

    def __enter__(self):
        self.file=self.path.open('xb')
        self.oldcharge=self.budget.BudgetContext._charge_op
        try:
            for name in OWNERS:
                original=getattr(self.mod,name);self.saved[name]=original
                setattr(self.mod,name,self.wrap(name,original))
            name='_ipt_forward_a_ab';original=getattr(self.mod,name);self.saved[name]=original
            setattr(self.mod,name,self.forward(original))
            self.budget.BudgetContext._charge_op=self.charge(self.oldcharge)
            self.emit('start',marker=self.marker(),max_events=MAX_EVENTS,max_bytes=MAX_BYTES)
            return self
        except BaseException:
            self.restore();raise

    def restore(self):
        for name,value in self.saved.items():setattr(self.mod,name,value)
        if hasattr(self,'oldcharge'):self.budget.BudgetContext._charge_op=self.oldcharge
        self.restored=all(getattr(self.mod,k) is v for k,v in self.saved.items())
        self.restored=self.restored and self.budget.BudgetContext._charge_op is self.oldcharge
        if self.file is not None:self.file.close();self.file=None

    def __exit__(self,typ,value,tb):
        try:
            if typ is None:
                assert not self.stack and not self.mm
                self.emit('finish',marker=self.marker(),roots=len(self.roots))
        finally:self.restore()
        return False

    def summary(self):
        return dict(events=self.events,bytes=self.bytes,root_calls=len(self.roots),
                    weak_live_tokens=len(self.refs),strong_array_registry=False,
                    patches_restored=self.restored,max_events=MAX_EVENTS,max_bytes=MAX_BYTES)
