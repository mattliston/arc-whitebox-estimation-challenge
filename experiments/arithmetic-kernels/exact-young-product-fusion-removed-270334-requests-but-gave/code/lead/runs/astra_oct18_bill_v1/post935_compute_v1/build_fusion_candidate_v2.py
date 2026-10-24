"""Immutable exact-real U/young fusion; no SDK import or numerical work."""
from pathlib import Path
import ast,hashlib,json
D=Path(__file__).resolve().parent;R=D.parents[1]
parent=R/'astra_oct18_accuracy_v1/setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py'
s=parent.read_text();sha=hashlib.sha256(parent.read_bytes()).hexdigest();assert sha=='95c8717670a5b4fbc2dcf57e289a5a0f5a8c6f9c656bee3a638655deb94b0cdf'
t=ast.parse(s);lines=s.splitlines()
def function(name):
 n=[n for n in t.body if isinstance(n,ast.FunctionDef)and n.name==name][-1]
 return '\n'.join(lines[n.lineno-1:n.end_lineno])
prices=json.loads((D/'FUSION_POLICY_PRICE_v2.json').read_text());rules={tuple(row[:4]):tuple(row[4:])for row in prices['rules']}
assert all(m<1024 and m-my in(0,64)and q>=40 and g>=-20000000 for(m,my,k,n),(g,q)in rules.items())
helper='''
# Public-shape tariff whitelist. No endpoint/rank values enter this policy.
_O18_FUSION_RULES = __RULES__
_O18_FUSION_CONTEXT = [None]
_O18_FUSION_STATE = dict(left=20000000, accepted=0, selected=[])

def _o18_fusion_prepare(layer, w, p, previous, newborn, basis):
    m, k = int(w.shape[1]), int(previous.shape[0])
    n, selected, base, t, ss = newborn
    n, q = int(n), int(base.shape[1])
    plan = dict(w=w, p=p, previous=previous, newborn=newborn, basis=basis,
                total=m, young_rows=m, negative=None, kept_count=m,
                ynt_context=None, rows=None, tail=None, U=None, eligible=False)
    ctx = _OTT_CONTEXT[0]
    if ctx is not None and 7 <= layer <= 14:
        # These are exactly the existing YNT eligibility checks/count operations.
        # Cache their result for its later invocation; no duplicate remote work.
        births = int(selected.shape[0])
        if (int(w.shape[0]) != 1024 or ctx['n'] != 1024
                or tuple(ctx['a'].shape) != (m,) or tuple(ctx['live'].shape) != (m,)
                or tuple(p.shape) != (k, n) or tuple(base.shape) != (1024, births)
                or tuple(t.shape) != (births,) or tuple(ss.shape) != (births,)
                or n + 2 * births > int(ctx['lam'].shape[0])):
            return None  # Unmodified parent retains its validation/error behavior.
        negative = (ctx['a'] < -2.5) & (~ctx['live'])
        kept = min(m, (int(fnp.nonzero(~negative)[0].shape[0]) + 63) // 64 * 64)
        plan.update(negative=negative, kept_count=kept, ynt_context=ctx,
                    young_rows=kept if m - kept >= 64 else m)
    my = plan['young_rows']
    key = (m, my, k, n + q)
    price = _O18_FUSION_RULES.get(key)
    if (price is None or tuple(basis.shape) != (k, 384)
            or _W5_PORT_SCOPE is not None or _FAST_RESCUE_ON
            or _UBC_CONTEXT[0] is not None or _UBX_CTX[0] is not None
            or _ASTRA_CLUSTER_CTX[0] is not None or _UBX_RC[0] is not None
            or _ASTRA_RLF_CONTEXT[0] is not None
            or (m, k, 384) in _SM12_RULE or (m, k, 384) in _O18_SHARED_RULES):
        return plan
    # Preserve every currently affordable compact10 route and its debit.
    if any(cost <= _O18_PEEL_STATE['left']
           for gain, cost, route in _O18_UNION_RULES.get((my, k, n + q), ())):
        return plan
    loss = max(0, -int(price[0]))
    if loss <= _O18_FUSION_STATE['left']:
        plan.update(eligible=True, key=key, price=price, loss=loss)
    return plan

_O18_FUSION_PARENT_NEWBORN = _astra_bcc_newborn_impl

def _astra_bcc_newborn_impl(w, p, previous, newborn):
    ctx = _O18_FUSION_CONTEXT[0]
    if (ctx is None or not ctx['eligible'] or p is not ctx['p']
            or previous is not ctx['previous'] or newborn is not ctx['newborn']):
        return _O18_FUSION_PARENT_NEWBORN(w, p, previous, newborn)
    n, selected, base, t, ss = newborn
    m, k, n, q = int(w.shape[1]), int(previous.shape[0]), int(n), int(base.shape[1])
    if (m != ctx['young_rows'] or tuple(ctx['basis'].shape) != (k, 384)
            or ctx['U'] is not None or int(p.shape[0]) != k
            or not _AB_ON or not _WINOGRAD or _SCRATCH is None
            or w.dtype != 'float32' or p.dtype != 'float32' or base.dtype != 'float32'):
        raise RuntimeError('U/young fusion ownership or kernel contract')
    if m != ctx['total'] and (ctx['rows'] is None or ctx['tail'] is None
                             or int(ctx['tail'].shape[0]) != 64):
        raise RuntimeError('U/young fusion requires exact captured YNT tail')
    hold = _E2Hold()
    wp = _xsg_rows(5154, w, previous).T
    prior = p[:, :n]
    common_rhs = _xsg_rows(5159, base, previous)
    rhs = fnp.concatenate((prior, common_rhs, ctx['basis']), axis=1)
    product = _mm(wp, rhs)
    # Copy before another kernel can touch scratch. No basis row is omitted.
    basis_kept = fnp.copy(product[:, n + q:])
    if m == ctx['total']:
        U = basis_kept
    else:
        wt = _xsg_cols(36501, ctx['w'], ctx['tail'])
        tail_left = _xsg_rows(36502, wt, previous).T
        basis_tail = _shared_old384_mm(tail_left, ctx['basis'])
        U = fnp.empty((ctx['total'], 384), dtype=fnp.float32)
        hold.append(fnp.put_along_axis(U, ctx['rows'][:, None], basis_kept, axis=0))
        hold.append(fnp.put_along_axis(U, ctx['tail'][:, None], basis_tail, axis=0))
    ctx['U'] = U
    inherited = product[:, :n]
    common = fnp.copy(product[:, n:n + q])
    _PHYS_HANDLES['newborn_common'] = common
    directions = _xsg_rows(5161, w, selected).T
    answer = fnp.empty((m, n + 2 * q), dtype=fnp.float32)
    hold.append(fnp.copyto(answer[:, :n], inherited))
    hold.append(fnp.add(common, directions * t, out=answer[:, n:n + q]))
    hold.append(fnp.add(common, directions * ss, out=answer[:, n + q:]))
    _ASTRA_BCC_STATS['newborn_elements'] += 2 * m * q
    _JUN_STATS['calls'] += 1
    _JUN_STATS['shapes'].append((m, k, n, q))
    _O18_FUSION_STATE['left'] -= ctx['loss']
    _O18_FUSION_STATE['accepted'] += 1
    _O18_FUSION_STATE['selected'].append((ctx['key'], ctx['price']))
    return answer
'''.replace('__RULES__',repr(rules))
# Materialize every product-dependent young output before the tail kernel.
# Thus safety does not rely on the old _mm result owning storage.
start = helper.index('    inherited = product[:, :n]')
end = helper.index("    _ASTRA_BCC_STATS['newborn_elements']", start)
young_materialization = helper[start:end]
helper = helper[:start] + helper[end:]
point = helper.index("    if m == ctx['total']:")
helper = helper[:point] + young_materialization + helper[point:]
ynt=function('_push_bank_shared_newborn')
old="    negative = (ctx['a'] < -2.5) & (~ctx['live'])\n    kept = min(m, (int(fnp.nonzero(~negative)[0].shape[0]) + 63) // 64 * 64)"
new="""    _fusion = _O18_FUSION_CONTEXT[0]
    if (_fusion is not None and _fusion['w'] is w and _fusion['p'] is p
            and _fusion['previous'] is previous and _fusion['newborn'] is newborn
            and _fusion['ynt_context'] is ctx):
        negative, kept = _fusion['negative'], _fusion['kept_count']
    else:
        negative = (ctx['a'] < -2.5) & (~ctx['live'])
        kept = min(m, (int(fnp.nonzero(~negative)[0].shape[0]) + 63) // 64 * 64)"""
assert ynt.count(old)==1;ynt=ynt.replace(old,new)
needle="    try:\n        pushed = _YNT_PARENT_PUSH"
replacement="""    try:
        if _fusion is not None and _fusion['eligible'] and _fusion['w'] is w:
            _fusion['rows'], _fusion['tail'] = rows, tail
        pushed = _YNT_PARENT_PUSH"""
assert ynt.count(needle)==1;ynt=ynt.replace(needle,replacement)
push=function('_astra_bcc_push_impl')
a=push.index('    U = _w5_port_call(');b=push.index('    answer = _astra_bcc_join(',a)
original=push[a:b]
Uline=original.splitlines()[0]
block='''    inherited_columns, selected, base, t, ss = newborn
    if old_columns > int(inherited_columns):
        raise RuntimeError('Shared-old384 attempted newborn confinement')
    young_newborn = (int(inherited_columns) - old_columns, selected, base, t, ss)
    young_p = p[:, old_columns - _cp_old0:]
    fusion = _o18_fusion_prepare(layer, wx, young_p, previous, young_newborn, active_basis)
    saved_fusion = _O18_FUSION_CONTEXT[0]
    if saved_fusion is not None:
        raise RuntimeError('nested U/young fusion context')
    _O18_FUSION_CONTEXT[0] = fusion
    try:
        if fusion is not None and fusion['eligible']:
            young = _astra_bcc_capture('young', lambda: _push_bank_shared_newborn(wx, young_p, previous, young_newborn))
            U = fusion['U']
            if U is None or tuple(U.shape) != (int(wx.shape[1]), rank):
                raise RuntimeError('U/young fusion failed to preserve full basis')
            old = _astra_bcc_capture('old', lambda: _slr_reconstruct(U, state))
            _shared_old384_finite(old, 'all-row reconstructed old bank')
            state['pre_basis'] = U
        else:
__ULINE__
            old = _astra_bcc_capture('old', lambda: _slr_reconstruct(U, state))
            _shared_old384_finite(old, 'all-row reconstructed old bank')
            state['pre_basis'] = U
            young = _astra_bcc_capture('young', lambda: _push_bank_shared_newborn(wx, young_p, previous, young_newborn))
    finally:
        _O18_FUSION_CONTEXT[0] = saved_fusion
'''.replace('__ULINE__','        '+Uline)
push=push[:a]+block+push[b:]
reset='''
_O18_FUSION_PARENT_PREDICT = Estimator.predict
def _o18_fusion_predict(self, *args, **kwargs):
    if _O18_FUSION_CONTEXT[0] is not None:
        raise RuntimeError('U/young context survived prediction')
    _O18_FUSION_STATE.update(left=20000000, accepted=0, selected=[])
    try:
        return _O18_FUSION_PARENT_PREDICT(self, *args, **kwargs)
    finally:
        _O18_FUSION_CONTEXT[0] = None
Estimator.predict = _o18_fusion_predict
'''
fragment='\n'+helper+'\n\n'+ynt+'\n\n'+push+'\n'+reset
candidate=s+fragment;ast.parse(candidate)
out=D/'fusion_candidate_v2';out.mkdir(exist_ok=False)
(out/'fragment_v2.py').write_text(fragment);p=out/'candidate_95c_fusion_HELD_v2.py';p.write_text(candidate)
receipt=dict(status='HELD_SOURCE_ONLY_REQUIRES_PARITY_AND_NATIVE',parent=str(parent),parent_sha256=sha,candidate=str(p),candidate_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rules=len(rules),policy_receipt=str(D/'FUSION_POLICY_PRICE_v2.json'),source_changes=['Exact95c source retained as prefix; appended current YNT/push/newborn implementations only.','Shared existing YNT count calculation; rejected routes retain U→old→young product order.','Accepted routes retain all384 basis columns and all basis rows with explicit64-row tail product.','Owned U is complete before old reconstruction; all young columns and newborn common/directions remain in original order.','Current affordable compact10 routes and all reuse contexts remain parent-owned; no setup/PB/layout/physicsparameter changes.'],limits=['FP32 product grouping changes, so exact-real identity is not bit-parity claim.','New productshapes may need coldview binding; measured request savings are warm metadata.','20MF bounded negativebill allowance uses conservative metadata credit; no scoregain banked.','Current hosted residual cushion is already insufficient; this alone is forecast to save~356calls, not restore allmargin.'])
(out/'BUILD_v2.json').open('x').write(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
