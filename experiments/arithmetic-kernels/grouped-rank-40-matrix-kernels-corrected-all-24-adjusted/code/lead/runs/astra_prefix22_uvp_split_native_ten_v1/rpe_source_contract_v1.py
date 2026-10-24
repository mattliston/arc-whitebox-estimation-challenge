def cleanup(mod):
    return dict(pending_clear=mod._RE22_PENDING==[None],active_clear=mod._RE22_ACTIVE==[False])
def gate(row):
    st=row['stats']['_RE22_STATS']
    assert row['re22_cleanup']==dict(pending_clear=True,active_clear=True)
    assert row['stats']['_RE22_ACTIVE']==[False] and st['retired'] is True
    changed=[r for r in row['stats']['_RTT_STATS']['reads'] if r['changed']]
    assert changed
    assert [r['layer']for r in st['reads']]==[r['layer']for r in changed]
    assert st['feature_layers']==[r['layer']for r in changed]
    for r,b in zip(st['reads'],changed):
        assert 1<=r['layer']<=14 and r['m']==b['m']
        assert r['prefix']==b['c15'] and r['tail']==b['c2']-b['c15']>0
        assert r['qrank']==b['refine_rank_out'] and r['qrank']in (0,1,2)
