"""TAG scout-native check_sn3_parent_v1 (2026-10-12; sources by line ranges): may fragment_sn3_v1.py be appended to <parent.py>? SN3 copies some function
bodies from the S50 files; this compares the parent's LAST top-level definition of each with the S50 twin's (f04fa779), by source text,
and checks the globals SN3 rebinds exist. Exit 0 = all equal (append is safe); exit 1 = a difference (regenerate SN3 for this parent).
usage: check_sn3_parent_v1.py <parent.py> [<reference.py>]"""
import ast
import sys
from pathlib import Path

O = Path(__file__).resolve().parents[1]
REF = O / 'engineer/a62_modules/s49/jmS50_v1tw_x/candidate.py'
COPIED = ('_hz_leaf', '_hzd_load', '_uvp_inner', '_hz_ks_product', '_ipt_forward_b_ab', '_hzd_forward_a', '_hz_views', '_hz_groups',
          '_hz_side', '_ks_acc', '_ks_unwind', '_ipt_side', '_ubc_find', '_uvp_state', '_slr_reconstruct', '_shared_old384_mm')
NAMES = ('_HZD_PARENT_KS', '_UVP_ORIG_RECON', '_UBC_RUN', '_UBC_CONTEXT', '_UBC_PARENT_B', '_UBC_LIMIT', '_TR_LAST', '_BH_HOLD', '_HZ_STATS',
         '_UVP_CUR', '_predict_binary')


def defs(path):
    text = Path(path).read_text()
    lines = text.split('\n')
    tree = ast.parse(text)
    out, assigned = {}, set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            out.setdefault(node.name, []).append('\n'.join(lines[node.lineno - 1:node.end_lineno]))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                if isinstance(t, ast.Name):
                    assigned.add(t.id)
    return out, assigned


parent = sys.argv[1]
ref = sys.argv[2] if len(sys.argv) > 2 else str(REF)
pd, pa = defs(parent)
rd, _ra = defs(ref)
bad = 0
for name in COPIED:
    if name not in pd or name not in rd:
        print('MISSING', name, 'parent' if name not in pd else 'reference'); bad += 1
        continue
    same = pd[name] == rd[name]          # every definition, in order (later ones override earlier ones)
    print('SAME' if same else 'DIFF', name, '(%d definitions)' % len(pd[name]))
    bad += 0 if same else 1
for name in NAMES:
    ok = name in pa or name in pd
    print('OK' if ok else 'MISSING', name)
    bad += 0 if ok else 1
print('VERDICT:', 'append SN3' if not bad else 'DO NOT append; regenerate SN3 for this parent')
sys.exit(1 if bad else 0)
