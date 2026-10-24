"""Exact-fringe peeling added to the same bounded arithmetic DP.

No estimator runs. Preserve the pad-only v1. Peeling computes the large
divisible core plus exact classical bottom/right/inner fringe products;
it is a partitioning identity, not sparsity or unbilled arithmetic.
"""
from pathlib import Path

here=Path(__file__).resolve().parent
source=(here/'price_mixed_fmm_envelope_v1.py').read_text()
source=source.replace("OUT=HERE/'mixed_fmm_envelope_v1.json'","OUT=HERE/'mixed_fmm_envelope_peel_v2.json'")
needle="                options.append((p['rank']*cost+linear,p['rank']*memory,p['name']+' / '+child))"
assert source.count(needle)==1
source=source.replace(needle,needle+'''
            # Alternative: leave exact thin fringes outside the fast core.
            a,b,c=(x//d for x,d in zip((m,k,n),p['shape']))
            if min(a,b,c)<8:continue
            m0,k0,n0=(x*d for x,d in zip((a,b,c),p['shape']))
            rm,rk,rn=m-m0,k-k0,n-n0
            if not (rm or rk or rn):continue
            fringe=2*m0*n0*rk
            memory_fringe=0
            if rk:memory_fringe=max(memory_fringe,4*(m0*rk+rk*n0+m0*n0))
            if rn:
                fringe+=2*m0*k*rn-m0*rn
                memory_fringe=max(memory_fringe,4*(m0*k+k*rn+m0*rn))
            if rm:
                fringe+=2*rm*k*n-rm*n
                memory_fringe=max(memory_fringe,4*(rm*k+k*n+rm*n))
            linear=ca*a*b+cb*b*c+cc*a*c
            for cost,memory,child in frontier(a,b,c,stages-1):
                options.append((p['rank']*cost+linear+fringe,max(p['rank']*memory,memory_fringe),p['name']+'[peel] / '+child))
''')
source=source.replace("qualification='Bounded optimistic arithmetic envelope:","qualification='Padding OR exact classical fringe peeling, including inner-fringe accumulation additions. Bounded optimistic arithmetic envelope:")
exec(compile(source,str(here/'price_mixed_fmm_envelope_v1.py'),'exec'),dict(__file__=str(here/'price_mixed_fmm_envelope_v1.py'),__name__='__main__'))
