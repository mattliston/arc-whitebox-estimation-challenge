"""Small stdlib scalar fixture; no array/model execution or backend-error claim."""
from pathlib import Path
import ast, hashlib, json, math, struct
D=Path(__file__).resolve().parent

def f32(x): return struct.unpack('<f',struct.pack('<f',x))[0]
def index(x): return int(f32(f32(min(abs(f32(x)),8.0)*2**20)+.5))
def main():
    protocol=json.loads((D/'protocol.json').read_text())
    source=(D/'fragment.py').read_bytes()
    assert hashlib.sha256(source).hexdigest()==protocol['fragment']['sha256']
    ast.parse(source)
    points=[0.,-0.,2**-149,2**-126,2**-22,2**-21,1.,2.,4.,8.,-8.,16.,-16.]
    for k in (0,1,2,511,1023,1048575,2097151,4194303,8388607):
        x=f32((k+.5)*2**-20);bits=struct.unpack('<I',struct.pack('<f',x))[0]
        for off in (-1,0,1):
            y=struct.unpack('<f',struct.pack('<I',bits+off))[0]
            points.extend((y,-y))
    for x in points:
        x=f32(x);i=index(x);assert 0<=i<=8388608
        clamp=min(abs(x),8.)
        assert abs(i*2**-20-clamp)<=2**-21+2**-45
        estimate=math.copysign(f32(math.tanh(i*2**-20)),x)
        assert abs(estimate-math.tanh(x))<=2**-21+2**-45+2**-25
        if x==0: assert math.copysign(1.,estimate)==math.copysign(1.,x)
    print(json.dumps(dict(passed=True,scalar_cases=len(points),arrays_executed=False,
      bound_conditional_on_table_primitive_rounding=True,backend_tanh_certified=False)))
if __name__=='__main__':main()
