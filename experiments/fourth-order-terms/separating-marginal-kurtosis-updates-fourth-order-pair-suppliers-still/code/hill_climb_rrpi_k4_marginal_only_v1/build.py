"""Source-only fixed K4 marginal intervention, preserving radial Q31 suppliers."""
from pathlib import Path
import hashlib,json
D=Path(__file__).resolve().parent
H=D.parent
P=H/'hill_climb_pair_rr_inverse_v1/candidate.py'
K=H/'hill_climb_rrpi_k4_rowdiag_v1'
PARENT_SHA='a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'
DONOR_SHA='ebd4c32f26a3bcfec8e76f587db611ab5b5bdc053cb7d927eb61b40381389bc7'


def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))


def source():
    parent=pin(P);donor=pin(K/'candidate.py');assert parent['sha256']==PARENT_SHA and donor['sha256']==DONOR_SHA
    original=P.read_text();base=(K/'candidate.py').read_text();delta=json.loads((K/'source_delta.json').read_text())
    assert delta['parent']==parent and delta['candidate']==donor and delta['body_inverse_whole_source_exact']
    assert pin(delta['fragment']['path'])==delta['fragment']
    suffix=delta['append_separator']+Path(delta['fragment']['path']).read_text();assert base.endswith(suffix)
    inverse=base[:-len(suffix)]
    for p in reversed(delta['replacements']):
        assert inverse.count(p['after'])==1;inverse=inverse.replace(p['after'],p['before'],1)
    assert inverse==original
    old_hidden="                k4 = k4 + _K4R_ALPHA * _k4r_delta(w32, _k4r_previous[1])"
    new_hidden="                _E2_LAYER['k4_q31_base'] = k4   # own current banded radial estimate, before marginal correction\n"+old_hidden
    old_pair=("    _r31 = _k31_row_scale(mu, k3, v)\n"
              "    if extra31 is None:\n"
              "        _e2_hold.append(fnp.multiply(_ps32(t4 * _r31 if _r31 is not None else t4)[:, None], rho, out=st['T31']))\n"
              "    else:\n"
              "        x31 = _e2_H(extra31 / _e2_H(_e2_H(v * s)[:, None])) / _e2_H(s[None, :])\n"
              "        fnp.fill_diagonal(x31, 0)\n"
              "        _e2_hold.append(fnp.add(_ps32(t4 * _r31 if _r31 is not None else t4)[:, None] * rho, _ps32(x31), out=st['T31']))")
    new_pair=("    _q31_k4 = L.get('k4_q31_base')\n"
              "    _q31_t4 = t4 if _q31_k4 is None else _q31_k4 / _e2_H(v ** 2)\n"+
              old_pair.replace('t4 * _r31 if _r31 is not None else t4','_q31_t4 * _r31 if _r31 is not None else _q31_t4'))
    old_gn="(_s19_k4u if _S19_GN_TORCH else k4), aux))"
    new_gn="(_s19_k4u if _S19_GN_TORCH else _E2_LAYER.get('k4_q31_base', k4)), aux))"
    edits=[dict(old=old_hidden,new=new_hidden),dict(old=old_pair,new=new_pair),dict(old=old_gn,new=new_gn)]
    changed=base
    for e in edits:
        assert changed.count(e['old'])==1 and e['new'] not in changed
        changed=changed.replace(e['old'],e['new'],1)
    inverse=changed
    for e in reversed(edits):
        assert inverse.count(e['new'])==1;inverse=inverse.replace(e['new'],e['old'],1)
    assert inverse==base
    assert changed.count('_K4R_ALPHA = 1.0')==1
    zero=changed.replace('_K4R_ALPHA = 1.0','_K4R_ALPHA = 0.0',1)
    compile(changed,str(D/'candidate.py'),'exec');compile(zero,str(D/'candidate_zero.py'),'exec')
    return changed,zero,edits


def main():
    changed,zero,edits=source()
    for name,text in [('candidate.py',changed),('candidate_zero.py',zero)]:
        with (D/name).open('x') as f:f.write(text)
    record=dict(parent=pin(P),donor=pin(K/'candidate.py'),candidate=pin(D/'candidate.py'),zero=pin(D/'candidate_zero.py'),
        donor_delta=pin(K/'source_delta.json'),build=pin(__file__),edits=edits,
        donor_whole_source_inverse_to_parent=True,whole_source_inverse_to_donor=True,compiled=True,
        alpha=1.0,layers=list(range(2,16)),fit_or_coefficient_search=False,
        semantics='Correct marginal k4 and its existing local consumers; keep pair and GN Q31 tied to current pre-add banded radial k4. K22 input unchanged. Terminal scalar retains the donor correction.',
        zero_identity_design='alpha0 bypasses donor correction/state arithmetic and never creates the context key; pair reuses original t4 object and GN original k4. Exact native byte/bill identity remains required.',
        lifetime='Context clears each layer before correction; saved k4 is a reference to the pre-add vector, not a mutable scratch alias. No cross-network state.',
        extra_arithmetic_above_donor_flops_upper_bound=26624,
        cost='Only a second length1024 square/divide at13 hidden layers; donor own measured cost remains applicable only as a forecast until native billing.',
        distinction='Earlier rowdiag physical, head coadapt and amplitude trials changed the radial-Q31 suppliers together with marginal k4; this fixed intervention separates those channels.',
        model_or_array_job_executed_by_builder=False)
    with (D/'source_delta.json').open('x') as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:record[k] for k in ('candidate','zero','build')}))


if __name__=='__main__':main()
