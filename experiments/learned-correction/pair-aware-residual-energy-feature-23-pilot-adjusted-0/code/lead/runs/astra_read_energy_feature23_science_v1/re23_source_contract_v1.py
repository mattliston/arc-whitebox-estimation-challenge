"""Fixed read-energy feature23 implementation contract."""
def cleanup(mod):
    return dict(active_clear=mod._RE23_ACTIVE==[False],pending_clear=mod._RE23_PENDING==[None])
def gate(row):
    st=row['stats'];r=st['_RE23_STATS'];rr=st['_RTT_STATS']
    assert r['retired'] is True and st['_RE23_ACTIVE']==[False] and st['_RE23_PENDING']==[None]
    assert rr['retired'] is True and [x['layer']for x in rr['reads']]==list(range(1,15))
    changed=[x for x in rr['reads']if x['changed'] and 7<=x['layer']<=14]
    assert [x['layer']for x in r['reads']]==r['applied']==[x['layer']for x in changed]
    for a,b in zip(r['reads'],changed):
        assert a['m']==b['m'] and a['k']==b['k'] and a['tail']==b['c2']-b['c15']>0
        assert 1<=a['blocks']<=7 and b['old_inner_calls']==int(b['old_rank']>0)
    assert row['re23_cleanup']==dict(active_clear=True,pending_clear=True)
