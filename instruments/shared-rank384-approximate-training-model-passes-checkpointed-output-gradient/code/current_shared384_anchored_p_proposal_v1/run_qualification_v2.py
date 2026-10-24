"""Byte-preserving owned-input plumbing repair; frozen numerical helper unchanged."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
OWNER=HERE/'run_qualification.py'
OWNER_SHA='0232c06ca22a0dea1926fef41e867d04a2997ba12b90b12b574a83d196de10c7'


def derive():
    raw=OWNER.read_bytes()
    if OWNER.is_symlink() or hashlib.sha256(raw).hexdigest()!=OWNER_SHA:
        raise RuntimeError('Preserved failed pre-model worker owner')
    original=raw.decode();source=original;changes=[]
    for before,after in (
        ("NAMES={'qualification.py','QUALIFICATION_PROTOCOL.md','run_qualification.py','supervise.py'}",
         "NAMES={'qualification.py','QUALIFICATION_PROTOCOL.md','run_qualification.py','supervise.py',"
         "'run_qualification_v2.py','supervise_v2.py','QUALIFICATION_PLUMBING_V2.md'}"),
        ("output=HERE/'qualification_v1'","output=HERE/'qualification_v2'"),
        ("        result['cost_certificate']=prepared['cost_certificate']",
         "        result['cost_certificate']=dict(prepared['cost_certificate'])\n"
         "        result['cost_certificate']['owned_ingress_extra_source_upper']=100_000_000\n"
         "        result['cost_certificate']['complete_source_upper']+=100_000_000\n"
         "        require(result['cost_certificate']['complete_source_upper']<8_000_000_000_000,'Full v2 cost cap')"),
        ("        weights=np.load(prepared['weights_path'],allow_pickle=False);remember(weights)",
         "        loaded=np.load(prepared['weights_path'],allow_pickle=False);remember(loaded)\n"
         "        require(type(loaded) is np.ndarray and loaded.shape==(16,1024,1024)\n"
         "            and loaded.dtype==np.float32 and loaded.flags.c_contiguous,'Original typed canonical C-layout ingress')\n"
         "        require(type(loaded.base) is np.ndarray and loaded.base.flags.owndata,'Actual owned loader base')\n"
         "        remember(loaded.base);loader_refs=[weakref.ref(loaded),weakref.ref(loaded.base)]\n"
         "        weights=loaded.copy(order='K');remember(weights)\n"
         "        require(weights.tobytes()==loaded.tobytes(),'Complete ingress byte parity, no dtype/layout change')\n"
         "        result['owned_ingress_copy_bytes']=weights.nbytes\n"
         "        del loaded;gc.collect()\n"
         "        require(all(ref() is None for ref in loader_refs),'Loader view/base retired before any model work')"),
        ("complete_source_upper=prepared['cost_certificate']['complete_source_upper'],",
         "complete_source_upper=prepared['cost_certificate']['complete_source_upper']+100_000_000,"),
    ):
        if source.count(before)!=1:raise RuntimeError('Exact owned-ingress-only worker splice')
        at=source.index(before);changes.append((at,before,after));source=source[:at]+after+source[at+len(before):]
    inverse=source
    for at,before,after in reversed(changes):
        if inverse[at:at+len(after)]!=after:raise RuntimeError('Exact owned-ingress worker inverse')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    if inverse!=original:raise RuntimeError('All model, gradient, replay, retirement and resource checks unchanged')
    return source


def main():
    env=dict(__file__=str(Path(__file__).resolve()),__name__='_qualification_ingress_v2')
    exec(compile(derive(),str(Path(__file__).resolve())+':owned-ingress-only','exec'),env)
    return env['main']()


if __name__=='__main__':raise SystemExit(main())
