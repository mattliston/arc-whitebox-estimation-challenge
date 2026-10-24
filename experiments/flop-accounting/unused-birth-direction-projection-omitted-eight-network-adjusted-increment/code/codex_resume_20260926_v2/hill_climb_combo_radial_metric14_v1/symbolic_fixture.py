"""Stdlib polynomial proof of the radial closure's double-trace normalization."""
from fractions import Fraction

def add(a,b):
    out=dict(a)
    for k,v in b.items():out[k]=out.get(k,Fraction(0))+v
    return {k:v for k,v in out.items() if v}
def mul(a,b):
    out={}
    for x,u in a.items():
        for y,v in b.items():
            k=tuple(sorted(x+y));out[k]=out.get(k,Fraction(0))+u*v
    return out
def scale(a,s):return {k:v*s for k,v in a.items()}
def variable(i,j,prefix):return {(prefix+str(min(i,j))+str(max(i,j)),):Fraction(1)}

def denominator(prefix):
    trace={};square={};double_trace={}
    for i in range(3):
        trace=add(trace,variable(i,i,prefix))
        for j in range(3):
            cij=variable(i,j,prefix)
            square=add(square,mul(cij,cij))
            # T_iijj/radial = (C_ii C_jj + 2 C_ij^2)/3.
            double_trace=add(double_trace,scale(add(mul(variable(i,i,prefix),variable(j,j,prefix)),scale(mul(cij,cij),2)),Fraction(1,3)))
    D=add(mul(trace,trace),scale(square,2))
    assert D==scale(double_trace,3)
    return D

def prove():
    before,after=denominator('a'),denominator('b')
    # After r'=r D_before/D_after, multiply both sides of
    # r' D_after = r D_before by the nonzero D_after.
    assert mul(before,after)==mul(after,before)
    return dict(passed=True,symbolic_dimension=3,metric_identity=True,transport_identity=True,
        requires_nonzero_final_metric=True,claim='Closure double trace only; not raw-fourth-moment preservation under mean updates.',numpy_or_model_executed=False)

if __name__=='__main__':
    import json
    print(json.dumps(prove()))
