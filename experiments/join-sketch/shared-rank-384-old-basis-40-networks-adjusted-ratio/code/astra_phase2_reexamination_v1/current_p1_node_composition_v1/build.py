"""Inert stdlib-only literal composition of frozen NODE and P1; no refit."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
P1 = NEW/'current_d50_native_objective_p1_v1'
NODE = NEW/'birth_direction_node_head_v1'
BD = NEW/'birth_direction_fit_v1'
OUT = HERE/'build_v1'
P1_SHA = '0deb237350e753220b575796c0278f86884b8461bfeec4ab53eae9c15522c435'
NODE_SHA = 'f072ad28a70698c152fa9a9ce31ba78191d9aece4b6928087d9b99c49a8dd30a'
ZERO_SHA = '1a99b49fcdc5cbfdaa03a40e676c0a910c5cf4e2d001965bf9e15e18678f1f05'
BANK_SHA = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'
PINS = {
    P1/'build_v1/estimator.py': P1_SHA,
    P1/'build_v1/result.json': '877990bc84cb85431ff98f83e1e7012517c8b7914a1e0b39b29bf7d0b776b0c1',
    P1/'score2_v1/result.json': '4f60a8def29f1a9326d68d5c25bbe79b3f26289af8cf46de81609fb91cf055e5',
    P1/'panel_v1/score/result.json': '81ebbb54207e4f0060aaa1a91320c17de89b6fdd53e7d5326c17993829e6e526',
    NODE/'build_zero_v1/estimator.py': ZERO_SHA,
    NODE/'build_terminal_v1/estimator.py': NODE_SHA,
    NODE/'build_terminal_v1/build.json': '3efa491ce8c2aa4e7725346b87e33602a8f384e4ae9aa3fc9bb4246c0e193f9a',
    NODE/'terminal_score_v2/result.json': 'ffb2ed5a3386674d707a37918ce454c20c16f6ca7f906e111adfd335b9934930',
    BD/'build_terminal_v1/estimator.py': BANK_SHA,
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    path = Path(path).resolve()
    require(path.suffix in ('.py','.json','.md'), 'Only source and receipt metadata are read')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def span(source, name):
    nodes = [x for x in ast.parse(source).body if isinstance(x,ast.Assign)
             and len(x.targets) == 1 and isinstance(x.targets[0],ast.Name)
             and x.targets[0].id == name]
    require(len(nodes) == 1, 'One complete literal assignment: '+name)
    x = nodes[0]
    lines = source.splitlines(keepends=True)
    a, b = sum(map(len,lines[:x.lineno-1])), sum(map(len,lines[:x.end_lineno]))
    ast.literal_eval(x.value)
    return a,b,source[a:b]


def replace(source, name, replacement):
    a,b,before = span(source,name)
    result = source[:a]+replacement+source[b:]
    x,y,after = span(result,name)
    require(after == replacement and result[:x]+before+result[y:] == source,
            'Literal replacement exact whole-source inverse')
    return result


def admission(source_sha, protocol_sha):
    pins = {str(p):v for p,v in PINS.items()}
    pins.update({str(Path(__file__).resolve()):source_sha,
                 str(HERE/'PROTOCOL.md'):protocol_sha})
    require(all(isinstance(v,str) and len(v) == 64 and sha(p) == v for p,v in pins.items()),
            'Every frozen source/result and reviewed local source pin')
    for folder in (P1/'build_v1',P1/'score2_v1',P1/'panel_v1/score',NODE/'terminal_score_v2'):
        r,c = read(folder/'result.json'),read(folder/'completion.json')
        require(r['passed'] and r['completed'] and c['passed'] and c['completed']
                and c['result_sha256'] == pins[str(folder/'result.json')], 'Completed receipt seal')
        pins[str(folder/'completion.json')] = sha(folder/'completion.json')
    pscore = read(P1/'panel_v1/score/result.json')
    require(pscore['candidate_sha256'] == P1_SHA and pscore['retain_local_component']
            and pscore['qualification_passed']
            and all(pscore['group_strict_gain_gate'][g] for g in ('public8','virgin6')),
            'P1 is the separately retained local baseline')
    nscore = read(NODE/'terminal_score_v2/result.json')
    require(nscore['pilot_gate_passed'] is False and nscore['integrity_passed']
            and nscore['numerical_fidelity_passed'] and not nscore['refit_performed'],
            'NODE fixed pilot failure remains FAILED, not promoted')
    nb,nc = read(NODE/'build_terminal_v1/build.json'),read(NODE/'build_terminal_v1/completion.json')
    require(nb['passed'] and nb['completed'] and nb['candidate_sha256'] == NODE_SHA
            and nb['inverse_source_byte_exact'] and nb['only_A_literal_changed']
            and nc['passed'] and nc['completed']
            and nc['build_sha256'] == pins[str(NODE/'build_terminal_v1/build.json')],
            'Original exact frozen terminal NODE source')
    pins[str(NODE/'build_terminal_v1/completion.json')] = sha(NODE/'build_terminal_v1/completion.json')
    require(not any(x in sys.modules for x in ('numpy','torch','flopscope')),'Stdlib-only admission')
    return pins


def construct():
    parent = (P1/'build_v1/estimator.py').read_text()
    bank = (BD/'build_terminal_v1/estimator.py').read_text()
    zero = (NODE/'build_zero_v1/estimator.py').read_text()
    node = (NODE/'build_terminal_v1/estimator.py').read_text()
    plit, blit = span(parent,'_GNN_LITERAL')[2],span(bank,'_GNN_LITERAL')[2]
    require(replace(parent,'_GNN_LITERAL',blit) == bank,'P1 changes only its original P assignment')
    require(span(zero,'_GNN_LITERAL')[2] == span(node,'_GNN_LITERAL')[2] == blit,
            'NODE learned with the exact old P assignment')
    require(replace(node,'_BDNH_A_LITERAL',span(zero,'_BDNH_A_LITERAL')[2]) == zero,
            'Original NODE zero/terminal differ only at A')
    candidates = {'parent':parent,'zero':replace(zero,'_GNN_LITERAL',plit),
                  'terminal':replace(node,'_GNN_LITERAL',plit)}
    require(replace(candidates['terminal'],'_BDNH_A_LITERAL',
                    span(candidates['zero'],'_BDNH_A_LITERAL')[2]) == candidates['zero'],
            'New composition zero/terminal differ only at the same A')
    for role,source in candidates.items():
        require(span(source,'_GNN_LITERAL')[2] == plit
                and span(source,'_BD_COEFFICIENT_LITERAL')[2] == span(bank,'_BD_COEFFICIENT_LITERAL')[2],
                'Exact P1 and original cstar source bytes')
        ast.parse(source)
    return candidates


def verify_build(build_sha, candidate_sha, source_sha, protocol_sha):
    pins = admission(source_sha,protocol_sha)
    require(sha(OUT/'result.json') == build_sha,'Exact new build seal')
    result,end = read(OUT/'result.json'),read(OUT/'completion.json')
    require(result['passed'] and result['completed'] and end['passed'] and end['completed']
            and end['result_sha256'] == build_sha and result['source_bindings'] == pins,
            'Complete new build result and source frame')
    for role,text in construct().items():
        item = result['candidates'][role]
        require(Path(item['path']) == OUT/(role+'_estimator.py')
                and Path(item['path']).read_bytes() == text.encode()
                and sha(item['path']) == item['sha256'], 'Exact literal-composition replay')
    require(result['candidates']['terminal']['sha256'] == candidate_sha,'Only reviewed terminal composition')
    return dict(build=result,candidates=result['candidates'],source_bindings=pins,
                input_bindings={str(OUT/'result.json'):build_sha,
                                str(OUT/'completion.json'):sha(OUT/'completion.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',action='store_true')
    parser.add_argument('--source-sha256')
    parser.add_argument('--protocol-sha256')
    args = parser.parse_args()
    if not args.build:
        parser.print_help(); return 0
    require(sys.flags.optimize == 0,'Assertions enabled')
    require(not OUT.exists(),'New exclusive directory; never overwrite or resume')
    start = time.monotonic()
    def expired(_s,_f):
        raise TimeoutError('Literal-only build60s limit')
    signal.signal(signal.SIGALRM,expired);signal.alarm(60)
    pins = admission(args.source_sha256,args.protocol_sha256)
    candidates = construct()
    OUT.mkdir(exist_ok=False)
    result = dict(schema='current_p1_node_composition_build_v1',passed=False,completed=False,
        source_bindings=pins,candidates={},artifacts={},estimator_executed=False,
        truth_arrays_accessed=False,refit_performed=False,old_NODE_gate_still_failed=True,
        whole_source_literal_inverse=True,P1_payload_byte_exact=True,NODE_A_byte_exact=True,
        cstar_byte_exact=True,production_qualified=False,official_score=False,
        started_utc=datetime.now(timezone.utc).isoformat())
    try:
        for role,text in candidates.items():
            target = OUT/(role+'_estimator.py')
            with target.open('x') as stream:
                stream.write(text)
            result['candidates'][role] = dict(path=str(target),sha256=sha(target))
            result['artifacts'][str(target)] = sha(target)
        require(all(sha(p) == v for p,v in pins.items()),'Inputs unchanged during composition')
        require(time.monotonic()-start < 60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2e9,
                'Source-only60s/2GB bound')
        result.update(passed=True,completed=True,hashes_unchanged=True)
    except BaseException:
        result['error'] = traceback.format_exc()
    result.update(wall_s=time.monotonic()-start,
                  peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    write(OUT/'result.json',result)
    write(OUT/'completion.json',dict(passed=result['passed'],completed=result['completed'],
                                    result_sha256=sha(OUT/'result.json')))
    signal.alarm(0)
    print(json.dumps({k:result[k] for k in ('passed','candidates','wall_s','peak_rss_bytes')}),flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
