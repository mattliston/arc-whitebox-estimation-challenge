"""Versioned extension of the certified scalar-final builder for Sym(b,B).

Its active covariance diagonal is already available from the compact
covariance push. Cold diagonals contribute to variance, not to B. All
deployment numerical operations remain ordinary metered FlopScope calls.
"""
from pathlib import Path

HERE=Path(__file__).resolve().parent
path=HERE/"build_compact_scalar_final_v1.py"
source=path.read_text()
edits={
    "payload_compact_final_head_v1.npz":"payload_fullpair_final_head_v1.npz",
    'old = "        Cpre = _cov_compact(w32,C,previous,previous_alive) if cov32 else w.T @ C @ w"':
        'old = "        if cov32:"',
    "            v = _cov_compact_diagonal(w32,C,previous,previous_alive)":
        "            v,bv = _cov_compact_diagonal(w32,C,previous,previous_alive)",
    "+_star_compact_diagonal(w32,star,previous,previous_alive)":
        "+_star_compact_diagonal(w32,star,previous,previous_alive)+3*bv*(w.T@b)",
    "    v = fnp.sum(wa.astype(fnp.float64)*pushed.astype(fnp.float64),axis=0)":
        "    v = fnp.sum(wa.astype(fnp.float64)*pushed.astype(fnp.float64),axis=0)\n    active_v = v",
    "    return v\n\ndef _star_compact_diagonal":
        "    return v,active_v\n\ndef _star_compact_diagonal",
}
for old,new in edits.items():
    assert source.count(old)==1,old
    source=source.replace(old,new)
exec(compile(source,str(path),"exec"),dict(__file__=str(path),__name__="__main__"))
