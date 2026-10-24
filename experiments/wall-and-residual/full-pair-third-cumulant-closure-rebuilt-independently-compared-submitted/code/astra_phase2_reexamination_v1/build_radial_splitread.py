"""Versioned exact tiling of the final oversized CP read into D4 blocks."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
target=HERE/"candidate_radial_gnn_splitread_v1.py"
if target.exists():raise FileExistsError(target)
src=(HERE/"candidate_radial_gnn_v1.py").read_text()
old='''    if m%16 or k%16 or n%16 or k>14336:
        return a@b'''
new='''    if m%16 or k%16 or n%16:
        return a@b
    if k>14336:
        out=_mm(a[:,:14336],b[:14336,:])
        for start in range(14336,k,14336):
            end=min(start+14336,k)
            out=out+_mm(a[:,start:end],b[start:end,:])
        return out'''
assert src.count(old)==1
target.write_text(src.replace(old,new))
print(target)
