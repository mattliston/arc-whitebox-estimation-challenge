import numpy as np
def stats(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    ea=float(np.mean(a*a));eb=float(np.mean(b*b));cross=float(np.mean(a*b));observed=.5*(ea+eb)
    dispersion=float(.5*np.mean((a-b)**2))
    assert abs(observed-cross-dispersion)<=2e-10*max(observed,1e-30)
    return dict(energy_A=ea,energy_B=eb,observed_energy=observed,cross_energy=cross,dispersion_energy=dispersion,
        dispersion_fraction=dispersion/observed if observed else None,cosine=cross/(ea*eb)**.5 if ea and eb else None)
def arm_stats(a,b,d):
    result=stats(a-d,b-d);base=stats(a,b)
    ga=base['energy_A']-result['energy_A'];gb=base['energy_B']-result['energy_B']
    gap=float(2*np.mean(d*(a-b)))
    assert abs((ga-gb)-gap)<=2e-10*max(base['observed_energy'],1e-30)
    result.update(ratio_A=result['energy_A']/base['energy_A'],ratio_B=result['energy_B']/base['energy_B'],
        cross_ratio=result['cross_energy']/base['cross_energy'] if base['cross_energy'] else None,
        gain_A=ga,gain_B=gb,gain_gap=gap,correction_energy=float(np.mean(d*d)))
    return result
def pool(items):
    out={k:sum(r[k] for r in items) for k in ('energy_A','energy_B','observed_energy','cross_energy','dispersion_energy')}
    out['dispersion_fraction']=out['dispersion_energy']/out['observed_energy'] if out['observed_energy'] else None
    out['cosine']=out['cross_energy']/(out['energy_A']*out['energy_B'])**.5 if out['energy_A'] and out['energy_B'] else None
    return out
