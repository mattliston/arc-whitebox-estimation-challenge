"""Inert fixed current-d50 tile census; no alternate numerical kernel."""
import argparse
import ast
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
BD = NEW/'birth_direction_fit_v1'
STORE = NEW.parent/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
SOURCE = BD/'build_terminal_v1/estimator.py'
REFERENCE = BD/'panel_v2/terminal/result.json'
FIXED = {
    SOURCE: 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451',
    REFERENCE: '6ad5687a0baf3265e2c663b2d0ea453262f57fa6a84474fd2fbac9ad1211aade',
    BD/'native_v2.py': '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b',
    NEW/'packed_arena_generic.py': '4e4a3dea533c222f34ba3a08538245414e149dbc0f918fd097267b0f869fb27e',
    STORE/'manifest.json': '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50',
    STORE/'rows/observer_fit/0035/row_manifest.json': '5067b577620cb6d5dc97c61b62f10e8fb3f98880b99fe9320f1ef8a8ca567809',
    STORE/'rows/observer_fit/0035/weights.npy': '14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2',
}
EXTRA_COLD_BINDING_FLOPS = 561056000


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def price(m, k, n, depth):
    assert all(v % 2**depth == 0 for v in (m,k,n))
    q = Fraction(7,4)
    total = 7**depth*(m//2**depth)*(n//2**depth)*(2*k//2**depth-1)
    total += q*(m*k+k*n+m*n)*sum(q**j for j in range(depth))
    assert total.denominator == 1
    return int(total)


def check_additional_bindings():
    """Reproduce the extra reshape bill from pinned metadata, no arrays."""
    layout = load_module(NEW/'packed_arena_generic.py', 'd50_census_layout').layout
    total = 0
    for m,k,n in ((1024,1024,8192),(1024,8192,1024)):
        child_size,specs,_ = layout(m//2,k//2,n//2,5)
        top = 7*(m*k+k*n+m*n)//4+3*m*n//4
        assert 4*(top+child_size) <= 1168900096
        total += top+sum(stop-start for start,stop,_ in specs)
    assert total == EXTRA_COLD_BINDING_FLOPS


class Observer:
    def __init__(self, candidate):
        self.calls, self.ctx = [], None
        original = candidate._packed_product_cached

        def observed(a,b,out,views,levels):
            shape = [int(a.shape[-2]), int(a.shape[-1]), int(b.shape[-1])]
            before = int(self.ctx.flops_used)
            answer = original(a,b,out,views,levels)
            delta = int(self.ctx.flops_used)-before
            self.calls.append(dict(shape=shape, depth=int(levels), dtype=str(a.dtype), flops=delta))
            return answer
        candidate._packed_product_cached = observed

    def clear(self):
        self.calls.clear()
        self.ctx = None


def once(source, old, new):
    assert source.count(old) == 1, old[:80]
    return source.replace(old,new)


def run_native(pins, reference):
    path = BD/'native_v2.py'
    old = load_module(path, 'd50_census_preserved_native')
    text = path.read_text()
    fn = next(x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef) and x.name=='main')
    original = ''.join(text.splitlines(keepends=True)[fn.lineno-1:fn.end_lineno])
    a = original.index("    inputs={str(p):sha(p) for p in (candidate,HERE/'PROTOCOL.md',")
    b = original.index('    manifest=json.loads(',a)
    admission = ('    inputs=dict(CENSUS_PINS)\n'
                 '    for path,digest in PINS.items():\n'
                 '        assert sha(path)==digest,path\n'
                 '        inputs[str(path)]=digest\n'
                 '    ranks=(35,)\n')
    edits = [
        (original[a:b], admission),
        ('        estimator=None\n', '        observer=Observer(module)\n        estimator=None\n'),
        ('                started=time.monotonic()\n', '                observer.clear()\n                started=time.monotonic()\n'),
        ('                    if estimator is None: estimator=module.Estimator()\n',
         '                    observer.ctx=ctx\n                    if estimator is None: estimator=module.Estimator()\n'),
        ("                trial['peak_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)",
         "                census=args.out/f'call_{call:02d}_tiles.json'\n"
         "                write(census,observer.calls)\n"
         "                result['artifacts'][str(census)]=sha(census)\n"
         "                trial['tile_census']=str(census)\n"
         "                observer.clear()\n"
         "                trial['own_stored_bytes_equal']=values.tobytes()==np.load(COLD_REF['output'],allow_pickle=False).tobytes()\n"
         "                trial['own_stored_bill_equal']=trial['flops']==COLD_REF['flops']-(INIT_FLOPS if call else 0)\n"
         "                trial['peak_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)"),
        ('                guard(trial)\n',
         "                guard(trial)\n"
         "                assert trial['own_stored_bytes_equal'] and trial['own_stored_bill_equal'], 'Census changed original output or bill'\n"),
        ('        result.update(passed=True,hashes_unchanged=True,',
         "        result['schema']='current_d50_fixed35_cost_census_v1'\n"
         "        result['alternate_kernel_executed']=False\n"
         '        result.update(passed=True,hashes_unchanged=True,'),
    ]
    source = original
    for before, after in edits:
        source = once(source,before,after)
    inverse = source
    for before, after in reversed(edits):
        inverse = once(inverse,after,before)
    assert inverse == original
    env = dict(vars(old), __file__=str(Path(__file__).resolve()), PARENT=NEW,
        CENSUS_PINS=pins, COLD_REF=reference['trials'][0],
        INIT_FLOPS=reference['initialization_flops'], Observer=Observer)
    original_argv = sys.argv
    sys.argv = [str(__file__), '--source',str(SOURCE), '--split','development','--out',str(HERE/'native')]
    try:
        exec(compile(source,str(__file__)+':preserved_native','exec'),env)
        return env['main']()
    finally:
        sys.argv = original_argv


def counterfactual():
    native = json.loads((HERE/'native/result.json').read_text())
    assert native['passed'] and len(native['trials']) == 2
    for path,digest in native['source_bindings'].items():
        assert sha(path)==digest
    for path,digest in native['artifacts'].items():
        assert sha(path)==digest
    records = []
    for trial in native['trials']:
        tiles = json.loads(Path(trial['tile_census']).read_text())
        rectangles = [t for t in tiles if t['depth']==4 and max(t['shape'])>1024]
        prices = []
        for tile in rectangles:
            m,k,n = tile['shape']
            assert tile['dtype']=='float32' and tile['flops']==price(m,k,n,4)
            mp,kp,npad = ((v+63)//64*64 for v in (m,k,n))
            # Pinned flopscope.12: zeros allocates at0 FLOPs; copyto charges
            # the written subview, not the padded owner. Wall/RSS still matter.
            padding = (m*k if (mp,kp)!=(m,k) else 0)
            padding += k*n if (kp,npad)!=(k,n) else 0
            replacement = price(mp,kp,npad,6)+padding
            prices.append(dict(**tile,padded_shape=[mp,kp,npad],additional_input_padding_flops=padding,
                hypothetical_d6_flops=replacement,saved_flops=tile['flops']-replacement))
        saving = sum(p['saved_flops'] for p in prices)
        setup = EXTRA_COLD_BINDING_FLOPS if trial['includes_initialization'] else 0
        hypothetical = trial['flops']-saving+setup
        records.append(dict(call=trial['call'], current_complete_flops=trial['flops'],
            all_packed_calls=len(tiles), rectangular_calls=len(rectangles),
            measured_rectangular_kernel_flops=sum(t['flops'] for t in rectangles),
            hypothetical_complete_flops=hypothetical,additional_cold_binding_flops=setup,
            hypothetical_complete_cost_ratio=hypothetical/trial['flops'],tiles=prices))
    result = dict(passed=True,completed=True,truth_accessed=False,score_measured=False,
        schema='current_d50_fixed35_same_trajectory_price_v1',
        native_result_sha256=sha(HERE/'native/result.json'),records=records,
        scope='One fixed native trajectory; alternate-kernel numerical path and whole-card runtime remain unqualified.',
        advance_to_single_product_screen=all(r['hypothetical_complete_cost_ratio']<=.92 for r in records),
        milestone_claim=False,finished_utc=datetime.now(timezone.utc).isoformat())
    write(HERE/'price.json',result)
    print(json.dumps({**{k:v for k,v in result.items() if k!='records'},
        'records':[{k:v for k,v in r.items() if k!='tiles'} for r in records]}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--source-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    assert sha(__file__)==args.source_sha256
    pins = {str(p):sha(p) for p in (*FIXED,Path(__file__),HERE/'PROTOCOL.md')}
    for path,digest in FIXED.items():
        assert pins[str(path)]==digest,path
    check_additional_bindings()
    reference = json.loads(REFERENCE.read_text())
    assert reference['passed'] and reference['trials'][0]['rank']==35
    for path in (STORE/'manifest.json', STORE/'rows/observer_fit/0035/row_manifest.json',
                 STORE/'rows/observer_fit/0035/weights.npy'):
        assert reference['source_bindings'][str(path)]==FIXED[path]
    item = reference['trials'][0]
    assert sha(item['output'])==item['output_sha256']
    pins[item['output']]=item['output_sha256']
    os.chdir(NEW)
    code = run_native(pins,reference)
    if code == 0:
        counterfactual()
    return code


if __name__=='__main__':
    raise SystemExit(main())
