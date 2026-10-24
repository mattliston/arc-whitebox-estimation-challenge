from pathlib import Path
import ast,copy,difflib,hashlib,json
D=Path(__file__).resolve().parent;H=D.parent;P=H/'hill_climb_pair_rr_inverse_v1';T=H/'hill_climb_star_consumer_basis32_v1'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def once(p,x):
 with Path(p).open('x') as f:f.write(x if isinstance(x,str) else json.dumps(x,indent=2,allow_nan=False)+'\n')
parent=pin(P/'candidate.py');assert parent['sha256']=='a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'
s=Path(parent['path']).read_text();lines=s.splitlines(True)
n=[n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='_shared_old384_push' and '_jn_rows' in ''.join(lines[n.lineno-1:n.end_lineno])][-1]
before=''.join(lines[n.lineno-1:n.end_lineno]);after=before
old=next(line for line in before.splitlines() if '_jn_rows = fnp.sort' in line)
new="""            if layer == 10:
                _jrt_order = fnp.argsort(-_jn_d.astype(fnp.float64), kind='stable')
                _jn_rows = fnp.concatenate([fnp.sort(_jrt_order[:rank]),
                                           fnp.sort(_jrt_order[rank:rank + 32])])
                del _jrt_order
            else:
"""+'    '+old
assert after.count(old)==1;after=after.replace(old,new,1)
old="        del K, unused_R, Y, Omega, _jn_rows"
new=old+"""
        if layer == 10:
            Qn, core = _jrt_reduce(Qn, core, rank)"""
assert after.count(old)==1;after=after.replace(old,new,1)
frag=(D/'fragment.py').read_text()
out=''.join(lines[:n.lineno-1])+after+''.join(lines[n.end_lineno:])+'\n\n'+frag
assert out[:-len('\n\n'+frag)].replace(after,before,1)==s
compile(out,str(D/'candidate.py'),'exec');once(D/'candidate.py',out)
once(D/'source.diff',''.join(difflib.unified_diff(s.splitlines(True),out.splitlines(True),fromfile=parent['path'],tofile=str(D/'candidate.py'))))
cfg=json.loads((T/'native_screen.json').read_text())
for role in ('base','parity_parent'):assert cfg['arms'][role]['source']==parent
cfg['arms']['candidate']=copy.deepcopy(cfg['arms']['base']);cfg['arms']['candidate']['source']=pin(D/'candidate.py')
for size in ('2','6'):
 b=cfg['arms']['candidate']['blocks'][size];b.pop('artifacts',None);b['tag']=D.name+'_train'+size+'_v1';b['guard']=str(D/('native'+size+'_guard.json'))
for size in ('2','8'):
 cfg['stages'][size]['result']=str(D/('native'+size+'_assessment.json'));cfg['stages'][size]['assessment_guard']=str(D/('native'+size+'_assess_guard.json'))
cfg['boundary_before8']['receipt']=str(D/'NATIVE8_NOT_AUTHORIZED.json');cfg['boundary_before8']['assessment_guard']=str(D/'NATIVE8_NOT_AUTHORIZED_guard.json')
protocol=dict(schema='join10-tail-ritz-v1',source_parent=parent,candidate=pin(D/'candidate.py'),scope='Only L10: keep current rank320 index-sorted landmarks first; append up to32 next-energy landmarks sorted by index. Current wider Y/QR/Nystrom-core construction; preserve first256 Q columns and choose largest-algebraic64 eigenvectors of96x96 tail core. Actual factors projected by unchanged code; ranks384/320/384 unchanged.',formula='T=blockdiag(I256,Vtail); Qnew=Qwide T; coreNew=T.T core T, including common/tail cross blocks. Stored core uses current FP32 core; tail eigh uses FP64; vectors narrow to FP32 for frame/cross products. Tail diagonal uses selected eigenvalues. No eigenvalue clipping.',caveat='Restricted Gram approximation, not full-tensor optimum. Prefix span preserved in real arithmetic by appended landmark ordering, not promised bitwise equal after widened native matrix products/QR. Current local common256 reads first256; tail orientation changes inherited difference factors and later joins.',screen='N2 raw<=-.1% and adjusted<=-.1%, at least1win,maxreg<=1%, finite16 and first10 layers byte exact. Then unchanged N8/resource/bank gates. No width/rank/layer/metric/scale sweep or rescue.',official_bank=pin(H/'official_bank_S35_v1.json'))
once(D/'protocol.json',protocol)
once(D/'build.json',dict(source_parent=parent,candidate=pin(D/'candidate.py'),fragment=pin(D/'fragment.py'),builder=pin(__file__),protocol=pin(D/'protocol.json'),whole_source_inverse=True,only_changed_existing_function='_shared_old384_push',new_helper='_jrt_reduce',ast_compile=True))
cfg['provenance']=[pin(D/'build.json'),pin(D/'protocol.json'),pin(P/'retained_component.json'),pin(P/'native8_assessment.json'),pin(P/'native8_assess_guard.json'),pin(P/'native_screen_continuation.json'),pin(H/'official_bank_S35_v1.json'),pin(T/'native_screen.json'),pin(H/'hill_climb_prefix128_edge_bias_hoist_v1/engine_inverse.json')]
cfg['interpretation']='One middle-join oversampled tail Ritz accuracy hypothesis vs RRPI. Supplemental raw<=-.1% and first10 exact gates required.'
once(D/'native_screen.json',cfg);cp=pin(D/'native_screen.json');b=cfg['arms']['candidate']['blocks']['2']
native=str(H.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python')
once(D/'commands_native2.json',dict(config=cp,commands=dict(native2=[native,'-B',b['driver']['path'],str(D/'candidate.py'),'35,100',b['tag']],native2_assessment=[native,'-B',cfg['engine']['path'],cp['path'],cp['sha256'],'2']),guards=dict(native2=b['guard'],native2_assessment=cfg['stages']['2']['assessment_guard']),limits=dict(native2_seconds=180,assessment_seconds=120,rss_bytes=7500000000,added_swap_bytes=0)))
print(json.dumps({n:pin(D/n) for n in ('candidate.py','native_screen.json','protocol.json','commands_native2.json')}))
