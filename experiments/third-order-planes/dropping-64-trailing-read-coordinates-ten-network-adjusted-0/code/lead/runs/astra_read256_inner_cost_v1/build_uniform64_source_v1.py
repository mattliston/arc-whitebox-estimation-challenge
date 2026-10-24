from pathlib import Path
import ast,hashlib,json,difflib
D=Path(__file__).resolve().parent;O=D.parents[2];P=O/'lead/runs/astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py';s=P.read_text();assert hashlib.sha256(s.encode()).hexdigest()=='2b10843a76fc033325ceee9dd30703eff29a1fa535cdc094c5feef72023a0c3c'
t=ast.parse(s);n=[n for n in t.body if isinstance(n,ast.FunctionDef)and n.name=='_uvp_inner'][-1];old=ast.get_source_segment(s,n);needle="    inner = _ubx_product(Dv, Fv2.T, uv, 'vt')\n";assert old.count(needle)==1
replacement='''    rank = int(Fv2.shape[0])
    layer = int(_UVP_CUR[0])
    cut = (_ASTRA_READ64_ON and 7 <= layer <= 14 and rank in (320, 384)
           and int(ru) == 256 and str(Fv2.dtype) == 'float32')
    if cut:
        keep = rank - 64
        reduced = _ubx_product(Dv, views(views(Fv2[:keep, :]).T), uv, 'vt')
        inner = fnp.empty((rows, rank), dtype=fnp.float32)
        top, tail = views(fnp.split(inner, [keep], axis=1))
        h_(fnp.copyto(top, reduced))
        h_(fnp.copyto(tail, 0))
        del top, tail, reduced
        views.clear()
        _ASTRA_READ64_STATS['events'].append((layer, rows, P, rank, keep, int(ru)))
    else:
        inner = _ubx_product(Dv, Fv2.T, uv, 'vt')
'''
new=old.replace(needle,replacement).replace('"""(L^2 lam) F^T with F in u/v form: rows x rank. = the parent\'s arithmetic; the two products through _ubx_product."""','"""Full-rank old read with a fixed prospective vt-only 64-coordinate cut."""')
lines=s.splitlines(keepends=True);lines[n.lineno-1:n.end_lineno]=[new+'\n'];body=''.join(lines)
fragment='''
# Fixed prospective read-only difference-tail cut. No stored factor is modified.
_ASTRA_READ64_ON = True
_ASTRA_READ64_STATS = dict(events=[], retired=True)
_ASTRA_READ64_PARENT_PREDICT = Estimator.predict
def _astra_read64_predict(self, *args, **kwargs):
    _ASTRA_READ64_STATS.clear()
    _ASTRA_READ64_STATS.update(events=[], retired=False)
    try:
        return _ASTRA_READ64_PARENT_PREDICT(self, *args, **kwargs)
    finally:
        _ASTRA_READ64_STATS['retired'] = True
Estimator.predict = _astra_read64_predict
'''
out=body+fragment;target=D/'candidate_uniform64_read_HELD_v1.py';compile(out,str(target),'exec');target.write_text(out)
back=body.splitlines(keepends=True);newn=[n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef)and n.name=='_uvp_inner'][-1];back[newn.lineno-1:newn.end_lineno]=[old+'\n'];assert ''.join(back)==s
(D/'uniform64_source_diff_v1.patch').write_text(''.join(difflib.unified_diff(s.splitlines(keepends=True),out.splitlines(keepends=True),fromfile=P.name,tofile=target.name)))
manifest=dict(status='HELD_SOURCE_NO_MODEL',candidate=str(target),candidate_sha256=hashlib.sha256(out.encode()).hexdigest(),parent=str(P),parent_sha256=hashlib.sha256(s.encode()).hexdigest(),scope='Only _uvp_inner vt outputwidth rank-64 for L7–14, rank320/384, common256. Zero-pad tofullrank then unchangedcommonadd. State/reconstruction/joins/propagation/Smirnov/tail/refinement unchanged. Appended Python-only per-predict event metadata.',inverse_function_and_wrapper_recovers_parent_byte_exact=True,first_unchanged_layers='layer indices0–6 because cut starts atsource layer7; actual native proof pending',expected_events_per_predict=10,expected_event_ranks=[384,384,384,320,320,320,320,384,320,384],numerical_state_persisted=False,prospective_cost='UNIFORM64_CURRENT24_WARM_TARIFF_v2.json; .66294454% warmbill forecast, no accuracyclaim')
(D/'UNIFORM64_SOURCE_HELD_v1.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
