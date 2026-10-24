"""Source/bytecode and SDK scope audit; never import or execute an estimator."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
D = HERE.parent
P2 = D.parent
SOURCE = D/'current_p_release_v1/candidate.py'
SDK = P2/'data_gate/.venv/lib/python3.11/site-packages/whestbench'
REMOVE = {'_FINAL_LITERAL','_FINAL_PARAMS','final_input','final_delta','_predict_compact_final'}
FUNCTIONS = {'final_input','final_delta','_predict_compact_final'}


def pin(path):
    path = Path(path).resolve()
    b = path.read_bytes()
    return dict(path=str(path),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def write(path,value):
    with path.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False)
        f.write('\n')


def bound_names(node):
    if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
        return [node.name]
    if isinstance(node,ast.Assign):
        return [v.id for t in node.targets for v in ast.walk(t) if isinstance(v,ast.Name)]
    return []


def function(tree,name):
    result = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(result)==1
    return result[0]


def calls(node,name):
    return [n for n in ast.walk(node) if isinstance(n,ast.Call) and
        ((isinstance(n.func,ast.Name) and n.func.id==name) or
         (isinstance(n.func,ast.Attribute) and n.func.attr==name))]


def code_semantics(code):
    def constant(c):
        if isinstance(c,types.CodeType):
            return code_semantics(c)
        if isinstance(c,tuple):
            return tuple(constant(v) for v in c)
        return c
    return (code.co_name,code.co_argcount,code.co_posonlyargcount,code.co_kwonlyargcount,
            code.co_nlocals,code.co_stacksize,code.co_flags,code.co_code,code.co_names,
            code.co_varnames,code.co_freevars,code.co_cellvars,tuple(constant(c) for c in code.co_consts))


assert pin(SOURCE)['sha256']=='8260ddb1f4966cc6acc4c318a53835981fde1b839d946ac10604b18325d2a040'
source = SOURCE.read_text()
tree = ast.parse(source)
removed = [n for n in tree.body if REMOVE & set(bound_names(n))]
assert len(removed)==5 and {name for n in removed for name in bound_names(n)}==REMOVE
assert not any(calls(tree,name) for name in ['globals','locals','vars','eval','exec','getattr','setattr','dir'])
assert not any(isinstance(n,ast.Attribute) and n.attr in ['__dict__','__globals__'] for n in ast.walk(tree))
for node in removed:
    if isinstance(node,ast.FunctionDef):
        assert node.name in FUNCTIONS and not node.decorator_list
        assert not node.args.defaults and not node.args.kw_defaults and node.returns is None
        assert all(a.annotation is None for a in node.args.args)
    else:
        value = ast.literal_eval(node.value)
        assert value is None or isinstance(value,dict)
retained = [n for n in tree.body if n not in removed]
for node in retained:
    assert not any(isinstance(n,ast.Name) and n.id in REMOVE for n in ast.walk(node))
    assert not any(isinstance(n,(ast.Global,ast.Nonlocal)) and REMOVE & set(n.names) for n in ast.walk(node))
lines = source.splitlines(keepends=True)
omit = {line for node in removed for line in range(node.lineno-1,node.end_lineno)}
pruned = ''.join(line for i,line in enumerate(lines) if i not in omit)
filtered_tree = ast.Module(body=retained,type_ignores=tree.type_ignores)
assert ast.dump(ast.parse(pruned),include_attributes=False)==ast.dump(filtered_tree,include_attributes=False)
original_codes = [c for c in compile(source,str(SOURCE),'exec').co_consts if isinstance(c,types.CodeType) and c.co_name not in FUNCTIONS]
pruned_codes = [c for c in compile(pruned,'in-memory-only','exec').co_consts if isinstance(c,types.CodeType)]
assert [code_semantics(c) for c in original_codes]==[code_semantics(c) for c in pruned_codes]
entrypoints = [n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Estimator']
entry = entrypoints[-1]
predict = next(n for n in entry.body if isinstance(n,ast.FunctionDef) and n.name=='predict')
assert len(calls(predict,'_predict_learned_compact'))==1 and not calls(predict,'_predict_compact_final')
worker_tree = ast.parse((SDK/'subprocess_worker.py').read_text())
main = function(worker_tree,'main')
branches = {}
for node in ast.walk(main):
    if isinstance(node,ast.If) and isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Name):
        if node.test.left.id=='command' and len(node.test.comparators)==1 and isinstance(node.test.comparators[0],ast.Constant):
            branches[node.test.comparators[0].value] = ast.Module(body=node.body,type_ignores=[])
assert len(calls(branches['start'],'load_estimator_from_path'))==1
assert not calls(branches['start'],'BudgetContext')
assert len(calls(branches['predict'],'_handle_predict'))==1
assert not calls(branches['predict'],'load_estimator_from_path')
handler = function(worker_tree,'_handle_predict')
assert len(calls(handler,'BudgetContext'))==1 and not calls(handler,'load_estimator_from_path')
scopes = [n for n in ast.walk(handler) if isinstance(n,ast.With) and
    any(isinstance(i.context_expr,ast.Name) and i.context_expr.id=='budget_ctx' for i in n.items)]
assert len(scopes)==1 and len(calls(scopes[0],'predict'))==1
loader_tree = ast.parse((SDK/'loader.py').read_text())
assert len(calls(function(loader_tree,'_import_module_from_path'),'exec_module'))==1
assert len(calls(function(loader_tree,'load_estimator_from_path'),'_import_module_from_path'))==1
plain = D/'current_p_release_v1/plain_client.py'
plain_tree = ast.parse(plain.read_text())
plain_scopes = [n for n in ast.walk(plain_tree) if isinstance(n,ast.With) and
    any(isinstance(i.optional_vars,ast.Name) and i.optional_vars.id=='ctx' for i in n.items)]
assert len(plain_scopes)==1 and len(calls(plain_scopes[0],'exec_module'))==1
prediction_only = D/'readout_residual_runtime_v1/predict_only_probe_v1/client.py'
po_tree = ast.parse(prediction_only.read_text())
po_scopes = [n for n in ast.walk(po_tree) if isinstance(n,ast.With) and
    any(isinstance(i.optional_vars,ast.Name) and i.optional_vars.id=='ctx' for i in n.items)]
assert len(po_scopes)==1 and not calls(po_scopes[0],'exec_module')
assert len(calls(po_tree,'exec_module'))==1
assert not {'numpy','torch','flopscope','whestbench'} & set(sys.modules)
nodes = []
for n in removed:
    text = ''.join(lines[n.lineno-1:n.end_lineno]).encode()
    nodes.append(dict(names=bound_names(n),line=n.lineno,end_line=n.end_lineno,bytes=len(text),
        sha256=hashlib.sha256(text).hexdigest(),kind=type(n).__name__))
paths = [SOURCE,SDK/'subprocess_worker.py',SDK/'loader.py',SDK/'runner.py',plain,prediction_only,
         D/'full_lower_gaussian_gate_reuse_v1/RESULT.md',D/'full_lower_gaussian_gate_reuse_v1/outcome.json',Path(__file__)]
report = f'''# Unused final head is outside the installed prediction clock

The exact official331760 source contains an unused older final-head payload
and helpers: final_input,final_delta,_FINAL_LITERAL,_FINAL_PARAMS and
_predict_compact_final. These five top-level nodes occupy
{sum(n['bytes'] for n in nodes)} of{len(source.encode())} source bytes
({sum(n['bytes'] for n in nodes)/len(source.encode()):.3%}). The active final
Estimator calls _predict_learned_compact. All references to the five removed
names are internal to those nodes; there is no candidate namespace reflection.
The assignments contain only literal values and the removed definitions have
no decorators, defaults or annotations with execution side effects.

An in-memory deletion parses and leaves the retained source AST exact. All
remaining compiled function/class/comprehension code objects also match after
excluding source-location metadata. This is a source/bytecode observation,
not a measured native/client equivalence or runtime benefit. No transformed
candidate was written, imported or executed. No archive or active file changed.

The installed SDK loads and constructs the estimator in its separate start
command. Its predict command calls _handle_predict, whose BudgetContext wraps
predict and output materialization, not module import. The loader's
exec_module therefore runs before the prediction clock. Removing an unused
module payload does not remove an executed prediction statement or charged
array operation. Unmeasured indirect memory/cache effects are not a runtime
repair claim. This lead is not activated as a prediction-budget optimization.

The preserved current-P ordinary release probe includes cold module loading
inside its prediction context; it is a broader cold-call check. The existing
predict_only_probe_v1 places loading outside, matching this SDK distinction.
A parsing-only speedup in the broader probe would not establish a reduction
in SDK prediction residual. Any future runtime claim must use the intended
clock and independently preserve output/bill integrity. This audit does not
change or relax either existing probe or any current acceptance threshold.

The prior CDF-reuse study is a separate executed negative: prediction-only
residual ratio0.993664585825508 and3/5wins failed its fixed<=.98 and>=4 gates.
It remains closed. Neither a source-size reduction nor this audit reopens it.

Evidence is the pinned local installed SDK and saved source; no remote
implementation or runtime improvement is inferred. No estimator calls,
numeric imports, new target access, fitting or numerical experiment occurred.
The signed-message fit and both active bridges are unchanged. Their complete
native accuracy outcome remains the next decision.
'''
with (HERE/'RESULT.md').open('x') as f:
    f.write(report)
write(HERE/'audit.json',dict(completed=True,utc=datetime.now(timezone.utc).isoformat(),
    source=pin(SOURCE),unused_top_level_nodes=nodes,removed_bytes=sum(n['bytes'] for n in nodes),
    source_bytes=len(source.encode()),active_entrypoint='Estimator.predict -> _predict_learned_compact',
    remaining_source_AST_exact=True,remaining_compiled_code_objects_exact_excluding_source_locations=True,
    retained_top_level_code_objects=len(original_codes),estimator_module_executed=False,
    transformed_candidate_written=False,source_preview_only_in_memory=True,
    sdk_start_loads_module=True,sdk_predict_window_excludes_module_load=True,
    broad_release_probe_includes_module_load=True,existing_prediction_only_probe_excludes_module_load=True,
    charged_prediction_work_removed=0,runtime_gain_measured=False,remote_runtime_claim=False,
    disposition='do not activate parsing-only deletion as prediction-budget repair',
    cdf_reuse_negative_preserved=True,active_studies_changed=False,target_arrays_opened=False,
    numerical_imports=False,estimator_calls=0,report=pin(HERE/'RESULT.md'),pins=[pin(p) for p in paths]))
print(dict(completed=True,removed_source_bytes=sum(n['bytes'] for n in nodes),
    retained_code_objects_exact=len(original_codes),sdk_module_load_outside_prediction=True,
    candidate_written=False,estimator_calls=0,activated=False),flush=True)
