import ast
from common import *

def put(name,s):
    with (HERE/name).open('x') as f:f.write(s)
old=D/'current_p_batch_process_endpoint_v1'
for name in ('guard.py','retirement.py'):put(name,(old/name).read_text())
s=SOURCE.read_text();assert pin(SOURCE)['sha256']=='8260ddb1f4966cc6acc4c318a53835981fde1b839d946ac10604b18325d2a040'
t=ast.parse(s);edits=[]
for name,pairs in [
    ('nonlinear_scalar',[("aux = dict(sigma=s,alpha=a", "aux = dict(pre_mu=mu,sigma=s,alpha=a")]),
    ('_predict_learned_compact',[(
        "        delta, _ = gnn_step(l, x, aux['sigma'], w, state, P)\n        return fnp.asarray(delta, dtype=fnp.float64)",
        """        global _DUAL_PACKET
        delta, hplus = gnn_step(l, x, aux['sigma'], w, state, P)
        minus_host, minus_aux = nonlinear_scalar(-aux['pre_mu'], aux['pre_v'], -aux['pre_k3'], aux['pre_k4'])
        minus_w = -w
        minus_x = final_scalar_features(l, minus_host, minus_aux, minus_w)
        minus_delta, hminus = gnn_step(l, minus_x, aux['sigma'], minus_w, state, P)
        if _DUAL_PACKET is not None:
            raise RuntimeError('Unretired dual terminal packet')
        _DUAL_PACKET = (hplus,hminus,aux['sigma'],m,delta,minus_delta,minus_host,aux['pre_mu'])
        return fnp.asarray(delta, dtype=fnp.float64)""")])]:
    node=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name][-1]
    before=ast.get_source_segment(s,node);after=before
    for a,b in pairs:assert after.count(a)==1;after=after.replace(a,b,1)
    assert s.count(before)==1;s=s.replace(before,after,1);edits.append((before,after))
extra='\n# Research capture packet, always retired outside the completed bill.\n_DUAL_PACKET=None\n'
s+=extra;ast.parse(s);inverse=s[:-len(extra)]
for before,after in reversed(edits):assert inverse.count(after)==1;inverse=inverse.replace(after,before,1)
assert inverse==SOURCE.read_text()
put('capture_candidate.py',s)
write(HERE/'build.json',dict(completed=True,parent=pin(SOURCE),capture_candidate=pin(HERE/'capture_candidate.py'),
      exact_source_inverse=True,mathematics='Only one extra exact scalar sign-reversal and unchanged-GNN terminal evaluation.',
      changed_functions=['nonlinear_scalar','_predict_learned_compact'],source_edits=[dict(before=a,after=b) for a,b in edits]))

native=(old/'native.py').read_text()
a=native.index("    ap.add_argument('stage'");b=native.index('    out = HERE/name',a)
native=native[:a]+'''    ap.add_argument('role', choices=['parent','capture'])
    ap.add_argument('--rank',type=int)
    args = ap.parse_args()
    args.stage = 'information_capture'
    verify()
    meta = read(HERE/'inputs.json')
    if args.rank is None:
        assert args.role=='capture'
        ranks,name=[35,100,35],'qualification'
    else:
        assert args.rank in (PARENT_NEW if args.role=='parent' else TRAIN[1:])
        assert read(HERE/'qualification.json')['passed']
        ranks,name=[args.rank],args.role+'_'+str(args.rank)
    source=pin(SOURCE if args.role=='parent' else HERE/'capture_candidate.py')
    check(source)
'''+native[b:]
native=native.replace("assert len(variances) == 16 and len(selectors) == 45", "assert len(variances) == (17 if args.role=='capture' else 16) and len(selectors) == 45\n        if args.role=='capture':\n            assert variances[15].tobytes()==variances[16].tobytes()",1)
native=native.replace('for layer,v in enumerate(variances):','for layer,v in enumerate(variances[:16]):',1)
marker="        write(out/f'call_{index:02d}.json',record)"
addition='''        packet_refs=[]
        if args.role=='capture':
            packet=mod._DUAL_PACKET
            assert len(packet)==8
            packet_refs=[weakref.ref(x) for x in packet]
            fields={key:np.asarray(array).copy() for key,array in zip(PACKET,packet)}
            assert all(np.isfinite(x).all() for x in fields.values())
            assert fields['hplus'].shape==fields['hminus'].shape==(1024,64)
            assert fields['hplus'].dtype==fields['hminus'].dtype==np.float32
            assert all(fields[k].shape==(1024,) for k in PACKET[2:])
            discrepancy=float(np.max(np.abs(fields['host']-fields['minus_host']-fields['mu'])))
            assert discrepancy<=1e-12,discrepancy
            served=np.maximum(fields['host']+fields['delta'].astype(np.float64),0).astype(np.float32)
            assert np.array_equal(served,value[15])
            packet_path=out/f'call_{index:02d}_packet.npz'
            with packet_path.open('xb') as f:np.savez(f,**fields)
            record.update(packet=pin(packet_path),scalar_complement_max_abs=discrepancy,extra_gaussian_variance_exact=True)
            mod._DUAL_PACKET=None
            del packet,fields,served
'''
assert native.count(marker)==1;native=native.replace(marker,addition+marker,1)
native=native.replace('[weakref.ref(w) for w in ww]', '[weakref.ref(w) for w in ww] + packet_refs',1)
native=native.replace('tracked_objects=18,live_objects=alive','tracked_objects=len(tracked),live_objects=alive',1)
# The context bodies are deliberately unchanged; packet processing is after billing.
def scopes(source):
    return [ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(source))
            if isinstance(n,ast.With) and any('BudgetContext' in ast.unparse(q.context_expr) for q in n.items)]
assert scopes(native)==scopes((old/'native.py').read_text())
put('native.py',native)
span=(D/'terminal_representation_v1/information.py').read_text().split('\ndef main():',1)[0]
put('span.py',span)

fit=read(D/'current_full_lower_p_fit_v1/inputs.json')
meta=dict(train=TRAIN,parent_new=PARENT_NEW,weights={str(r):fit['weights'][str(r)] for r in TRAIN+[100]},
          targets={str(r):fit['targets'][str(r)] for r in TRAIN},baselines={},parent=pin(SOURCE),targets_decoded=False)
triple=D/'current_p_joint_moment_native_v1/native_parent/result.json'
q=read(triple);assert q['source']==pin(SOURCE) and [r['rank'] for r in q['records']]==[35,100,35]
meta['qualification_reference']=pin(triple)
meta['baselines']['35']=dict(record=q['records'][0],source=q['source'],receipt=pin(triple))
for rank in [40,41,43,44,45,46,47,49,50,51,52,53,54,55]:
    path=D/f'current_p_downstream_retirement_v1/panel_{rank}/result.json'
    q=read(path);assert q['source']['sha256']==pin(SOURCE)['sha256'] and len(q['records'])==1
    r=q['records'][0];assert r['rank']==rank and r['includes_initialization']
    for key in ('prediction','bill','weights','input_bill','input_reset','lifecycle'):check(r[key])
    meta['baselines'][str(rank)]=dict(record=r,source=q['source'],receipt=pin(path))
for p in list(meta['weights'].values())+list(meta['targets'].values()):check(p)
write(HERE/'inputs.json',meta)
external=[SOURCE,triple,old/'native.py',old/'guard.py',old/'retirement.py',A/'birth_direction_fit_v1/native_v2.py',
          A/'RELU_COMPLEMENT_PROBE_V1_CLOSED.md',A/'generate_relu_complement_probe_v1.py',
          D/'terminal_representation_v1/information.py',D/'terminal_representation_v1/NEGATIVE_RESULT.md']
for b in meta['baselines'].values():
    external+=[Path(b['receipt']['path']),Path(b['source']['path'])]
    external+=[Path(b['record'][k]['path']) for k in ('prediction','bill','input_reset','lifecycle')]
pins=[pin(p) for p in sorted(set(external+list(HERE.glob('*.py'))+list(HERE.glob('*.md'))+[HERE/'inputs.json',HERE/'build.json']))]
write(HERE/'sources_sealed.json',dict(pins=pins,budget_context_ASTs_exact=True,source_inverse=True,train=TRAIN,
      original_failures_preserved=True,targets_decoded=False))
print(dict(prepared=True,pins=len(pins),reused_cold_baselines=len(meta['baselines']),new_parent_calls=len(PARENT_NEW)),flush=True)
