"""Fixed same-rank first-join diagonal implementation contract."""
def cleanup(mod):
    return dict(active_clear=mod._JCDP_ACTIVE==[False],on=mod._JCDP_ON is True,ranks=mod._R320_RULE)
def gate(row):
    st=row['stats'];r=st['_JCDP_STATS']
    assert r['retired'] is True and st['_JCDP_ACTIVE']==[False]
    assert r['calls']==1 and r['layers']==[7] and len(r['factor_shapes'])==1
    rank,width=r['factor_shapes'][0];assert rank==384 and 0<width<=6144 and width%2==0
    assert row['jcdp_cleanup']==dict(active_clear=True,on=True,ranks={'7':384,'10':320,'13':384})
