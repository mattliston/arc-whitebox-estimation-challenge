import math

def summarize(rows):
    assert len(rows)==40
    for r in rows:
        for role in ('parent','trained'):
            x=r['roles'][role]
            assert math.isfinite(x['final_raw']) and x['final_raw']>=0
            assert 0<x['flops']<=2**41 and x['multiplier']==max(.1,x['flops']/2**41)
            assert x['final_adjusted']==x['final_raw']*x['multiplier']
    means={role:{k:sum(r['roles'][role][k] for r in rows)/40
        for k in ('final_raw','final_adjusted','flops')} for role in ('parent','trained')}
    assert means['parent']['final_adjusted']>0
    ratio=means['trained']['final_adjusted']/means['parent']['final_adjusted']
    wins=sum(r['roles']['trained']['final_adjusted']<r['roles']['parent']['final_adjusted'] for r in rows)
    return dict(means=means,adjusted_ratio=ratio,raw_ratio=means['trained']['final_raw']/means['parent']['final_raw'],
        adjusted_wins=wins,passed=ratio<1 and wins>=24)
