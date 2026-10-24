"""Fixed first-L7-join residual-diagonal selector implementation contract."""
def cleanup(mod):
    return dict(active_clear=mod._JRS_ACTIVE==[False],on=mod._JRS_ON is True,ranks=mod._R320_RULE)
def gate(row):
    st=row['stats'];r=st['_JRS_STATS']
    assert r['retired'] is True and st['_JRS_ACTIVE']==[False]
    assert r['calls']==1 and len(r['records'])==1
    a=r['records'][0]
    assert a['layer']==7 and a['rank']==384 and 384<=a['m']<=1024
    assert 0<a['columns']<=6144 and a['columns']%2==0
    assert a['original_anchor'] is True and type(a['fallback'])is bool
    assert a['reason'] in ('nonpositive_or_nonfinite_common_energy','nonfinite_residual_score') if a['fallback'] else a['reason'] is None
    assert row['jrs_cleanup']==dict(active_clear=True,on=True,ranks={'7':384,'10':320,'13':384})

