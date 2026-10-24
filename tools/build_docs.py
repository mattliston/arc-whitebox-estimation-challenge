#!/usr/bin/env python3
"""Generate the front door and glossary; promote the reviewed estimator anatomy."""
import collections
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    g = json.loads((ROOT / 'graph/graph.json').read_text())
    manifest = json.loads((ROOT / 'graph/manifest.json').read_text())['copies']
    counts = collections.Counter(n['kind'] for n in g['nodes'])
    subs = {s['submission_id']: s for s in g['submissions']}
    best = min((s for s in g['submissions'] if s['status'] == 'GRADED' and not s['failed_networks']), key=lambda s: s['score_public'])
    text = (ROOT / 'tools/README.template.md').read_text().format(score337641=subs[337641]['score_public'], score337619=subs[337619]['score_public'], submissions=len(g['submissions']), archives=sum(bool(s.get('archive_path')) for s in g['submissions']))
    (ROOT / 'README.md').write_text(text)
    (ROOT / 'release/README.md').write_text('''# Designated release packages

Start with [run and reproduction instructions](../estimator/REPRODUCE.md): install the pinned runtime, rebuild the exact estimator files, run a seeded network, then evaluate on the public dataset.

| Submission | Inference source | Original archive | Recorded grade |
|---|---|---|---|
| #337641 | [Python](../estimator/submission_337641.py) | [archive](../submissions/337641/submission.tar.gz) | [record](../submissions/337641/README.md) |
| #337619 | [Python](../estimator/submission_337619.py) | [archive](../submissions/337619/submission.tar.gz) | [record](../submissions/337619/README.md) |

`common/parents/` and `common/readout/` contain the parent sources, fitted coefficients and original export programs used by `tools/reproduce_estimator.py`. The designated files embed all inference weights.

The [original packaging README](ARCHIVE_README.md), [dependency manifest](DEPENDENCY_MANIFEST.md) and subdirectory READMEs are byte-preserved historical records. Use the reproduction guide for the tested installation and run commands; it explains the later SDK versions and the distinction between local execution and hosted timing.
''')
    (ROOT / 'estimator').mkdir(exist_ok=True)
    shutil.copyfile(ROOT / '_build/ANATOMY_draft.md', ROOT / 'estimator/ANATOMY.md')
    glossary = json.loads((ROOT / '_build/GLOSSARY.json').read_text())
    (ROOT / 'graph/GLOSSARY.json').write_text(json.dumps(glossary, indent=1, ensure_ascii=False) + '\n')
    paths = {}
    for c in manifest:
        paths.setdefault(c['origin'], c['dest'])
    lines = ['# Glossary', '', 'Definitions are tied to the archived implementation and research records. Unstated acronym expansions remain explicit; inferred interpretations are labeled.', '']
    for term, entry in sorted(glossary.items(), key=lambda item: item[0].lower()):
        lines += ['## ' + term, '', entry.get('definition', ''), '']
        if entry.get('expansion'):
            lines += ['Expansion: ' + entry['expansion'] + '.', '']
        lines += ['Evidence status: ' + entry.get('confidence', 'unspecified') + '.', '']
        for source in entry.get('sources', []):
            origin, sep, fragment = source.partition('#')
            if origin in paths:
                target = '../' + paths[origin] + (sep + fragment if sep else '')
                lines.append(f'- [{Path(origin).name}{sep + fragment if sep else ""}]({target})')
            else:
                lines.append(f'- Frozen-source reference: `{source}` (outside the copied Phase 2 evidence set).')
        lines.append('')
    (ROOT / 'graph/GLOSSARY.md').write_text('\n'.join(lines))
    print(f'README, estimator anatomy, and {len(glossary)} glossary entries generated')

if __name__ == '__main__':
    main()
