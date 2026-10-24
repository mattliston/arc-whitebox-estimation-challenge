"""Held source-only preparation; root activates only after GN3 full40 fixes parent."""
from pathlib import Path
import ast
import copy
import difflib
import hashlib
import json
import math
import sys

D = Path(__file__).resolve().parent
H = D.parent
G = H/'hill_climb_rrpi_native_joint3_recovery_v1'
BEFORE = "head_b=P['head_b'][l], gain=1 + 0.25 * fnp.tanh(P['gain'][l:l + 1]))"
AFTER = "head_b=P['head_b'][l], gain=(1 + 0.25 * fnp.tanh(P['gain'][l:l + 1])) * _UNIFORM_GNN_GAIN)"


def pin(path):
    p = Path(path).resolve(); b = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))


def checked(record):
    assert pin(record['path']) == record, record['path']
    return Path(record['path']).read_bytes()


def load(record):
    return json.loads(checked(record))


def once(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def edit_source(parent, multiplier):
    """One scalar multiplication while a layer cache entry is first constructed."""
    assert math.isfinite(multiplier) and .98 <= multiplier <= 1.02
    assert parent.count(BEFORE) == 1 and AFTER not in parent
    tree = ast.parse(parent)
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_e2_gnn_layer']
    assert len(definitions) == 1
    assert BEFORE in ast.get_source_segment(parent, definitions[0])
    suffix = '\n# Fixed uniform effective GNN gain; no cache mutation or parameter fit here.\n_UNIFORM_GNN_GAIN = '+repr(float(multiplier))+'\n'
    source = parent.replace(BEFORE, AFTER) + suffix
    compile(source, '<uniform-gnn-gain>', 'exec')
    assert source[:-len(suffix)].replace(AFTER, BEFORE) == parent
    new_tree = ast.parse(source)
    old_functions = {n.name:ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name:ast.dump(n, include_attributes=False) for n in new_tree.body if isinstance(n, ast.FunctionDef)}
    assert [k for k in old_functions if old_functions[k] != new_functions[k]] == ['_e2_gnn_layer']
    return source, dict(before=BEFORE, after=AFTER, suffix=suffix, exact_inverse=True)


def control(plan, assessment, role, ranks, parent):
    arm = copy.deepcopy(plan['arms'][role]); records = assessment['artifacts'][role]
    assert arm['source'] == parent and arm['ranks'] == ranks
    assert plan['ranks'] == ranks
    m = load(records['manifest']); g = load(records['guard'])
    assert m['candidate'] == parent['path'] and m['sha256'] == parent['sha256'] and m['ranks'] == ranks
    assert m['runtime'] == plan['runtime'] and m['numpy'] == '2.4.6'
    assert g['command'] == arm['command'] and g['completed'] and g['exit_code'] == 0 and g['error'] is None
    assert g['whole_group_retired'] and g['remaining_rows'] == [] and g['max_swap_growth_bytes'] == 0
    assert g['rss_trigger_bytes'] == 7500000000 and g['max_rss_bytes'] < 7500000000
    assert 0 < g['elapsed_s'] <= arm['limits']['seconds']
    checked(records['rows'])
    for record in records['predictions'].values(): checked(record)
    return dict(arm=arm, artifacts=records)


def main():
    assert len(sys.argv) == 4, 'PREPARATION_SHA {gn3|rrpi} GN3_FULL40_ASSESSMENT_SHA'
    prep_pin = pin(D/'preparation_plan.json'); assert prep_pin['sha256'] == sys.argv[1]
    p = load(prep_pin); assert p['builder'] == pin(__file__)
    for record in p['pins']: checked(record)
    role = sys.argv[2]; assert role in ('gn3', 'rrpi')
    broad_pin = pin(G/'broad_assessment.json'); assert broad_pin['sha256'] == sys.argv[3]
    broad = load(broad_pin)
    assert broad['receipt_checks_passed'] and broad['candidate'] == p['parents']['gn3']
    assert broad['stage'] == 'VALID40' and broad['baseline_submission'] == 333111
    assert role == ('gn3' if broad['component_passed'] else 'rrpi')
    guard_pin = pin(G/'broad_assess_guard.json'); guard = load(guard_pin)
    assert guard['command'] == [p['runtime'], '-B', broad['assessor']['path'], broad['config']['path'], broad['config']['sha256'], broad['admission']['sha256']]
    assert guard['completed'] and guard['exit_code'] == 0 and guard['error'] is None
    assert guard['whole_group_retired'] and guard['remaining_rows'] == [] and guard['max_swap_growth_bytes'] == 0
    assert guard['rss_trigger_bytes'] == 7500000000 and guard['max_rss_bytes'] < 7500000000 and 0 < guard['elapsed_s'] <= 120
    parent = p['parents'][role]; parent_text = checked(parent).decode()
    fp = load(p['fit4_plan']); fa = load(p['fit4_assessment'])
    cp = load(p['check8_plan']); ca = load(p['check8_assessment'])
    assert fa['passed_integrity'] and ca['passed_integrity'] and fa['admitted'] and ca['admitted']
    assert ca['fit4_admission'] == p['fit4_assessment']
    assert fp['driver'] == cp['driver'] == p['driver']
    arm_role = 'candidate' if role == 'gn3' else 'base'
    fit_ranks = [35,100,36,45]; check_ranks = [52,58,64,70,76,80,84,88]
    controls = dict(fit4=control(fp,fa,arm_role,fit_ranks,parent),check8=control(cp,ca,arm_role,check_ranks,parent))
    assert not set(fit_ranks) & set(check_ranks)
    arms = {}; inverse = {}
    for name,multiplier in (('minus',.98),('plus',1.02)):
        source, proof = edit_source(parent_text,multiplier)
        path = D/(name+'.py')
        with path.open('x') as f: f.write(source)
        (D/(name+'.diff')).write_text(''.join(difflib.unified_diff(parent_text.splitlines(True),source.splitlines(True),fromfile=parent['path'],tofile=str(path))))
        tag = D.name+'_'+name+'_fit4_v1'
        command = [p['runtime'],'-B',p['driver']['path'],str(path),','.join(map(str,fit_ranks)),tag]
        record = D/(name+'_guard.json')
        arms[name] = dict(source=pin(path),multiplier=multiplier,ranks=fit_ranks,tag=tag,output=str(H/tag),driver=p['driver'],guard=str(record),limits=dict(seconds=150,rss_bytes=7500000000,added_swap_bytes=0),command=command,guard_command=[p['light'],'-B',p['guard_source']['path'],'--record',str(record),'--seconds','150','--rss-limit','7500000000']+command,log=str(D/(name+'.log')))
        inverse[name] = dict(parent=parent,candidate=pin(path),**proof)
    protocol = dict(schema='uniform-effective-gnn-gain-native-fit4-v1',preparation=prep_pin,builder=pin(__file__),parent=parent,parent_role=role,parent_full40_assessment=broad_pin,parent_full40_assessment_guard=guard_pin,fit_ranks=fit_ranks,check_ranks=check_ranks,driver=p['driver'],guard_source=p['guard_source'],runtime=p['runtime'],light=p['light'],setup_counts=fp['setup_counts'],data=dict(fit4=fp['data'],check8=cp['data']),controls=controls,arms=arms,source_inverse=inverse,step_fraction=.02,ridge_diagonal_fraction=.1,box=[-1.,1.],parity_layers=[0],gates=p['gates'],one_endpoint_only=True,old_failures_preserved=True,no_new_dials_or_panels=True,no_numerical_execution=True,scope='Effective cached gain of existing GNN correction at all L1–15; L0 remains zero. Scale output gain, not raw tanh parameter. Layer cache is populated once without cumulative mutation.',solver='One-coordinate x=clip(-g/(1.1*G),-1,1); zeroG fixesx0. Final-layer residual weighted by saved baseline utilization; positive predicted gain only. Even remainder diagnostic, not exact derivative.',endpoint_multiplier='1+.02*x',bank_gain_pct_min=1.,resources_and_full40_still_required=True)
    protocol.update(gain_hook=dict(before=BEFORE,after=AFTER),scalar_name='_UNIFORM_GNN_GAIN',source_editor=pin(__file__),fit_source=pin(D/'fit.py'),endpoint_assessor=pin(D/'assess_endpoint.py'))
    once(D/'protocol.json',protocol)
    once(D/'acquisition_commands.json',dict(protocol=pin(D/'protocol.json'),parent=parent,thread_env=dict.fromkeys(('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'),'1'),sequential=True,minus=arms['minus']['guard_command'],minus_log=arms['minus']['log'],plus=arms['plus']['guard_command'],plus_log=arms['plus']['log'],fitting_and_endpoint_not_executed=True))
    print(json.dumps(dict(protocol=pin(D/'protocol.json'),commands=pin(D/'acquisition_commands.json'),parent=parent)))


if __name__ == '__main__': main()
