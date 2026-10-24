def cleanup(mod):
    return dict(delta_absent='f24_tail_delta' not in mod._E2_LAYER)
def gate(row):
    st=row['stats']['_F24_STATS']
    assert row['f24_cleanup']==dict(delta_absent=True)
    assert st['retired'] is True
    expected=[x for x in row['stats']['_RTT_STATS']['reads'] if x['changed'] and 2<=x['layer']<=14]
    assert [x['layer'] for x in st['layers']]==[x['layer'] for x in expected]
    for a,b in zip(st['layers'],expected):
        assert a['rows']==b['m'] and a['young']==b['k'] and a['tail']==b['c2']-b['c15']
        assert 0<=a['diagonal_rows']<=min(a['rows'],a['tail'])
