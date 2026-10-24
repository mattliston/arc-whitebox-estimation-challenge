#!/usr/bin/env python3
"""Check generated documentation links and source-line references."""
import functools
import json
import re
from pathlib import Path
from urllib.parse import unquote
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
@functools.lru_cache(maxsize=None)
def lines(path):
    with path.open('rb') as f:return sum(1 for _ in f)
def main():
    graph=json.loads((ROOT/'graph/graph.json').read_text())
    docs={ROOT/p for p in ['README.md','release/README.md','estimator/ANATOMY.md','estimator/REPRODUCE.md','graph/schema.md','graph/GLOSSARY.md','data/DATA_INDEX.md','submissions/SUBMISSIONS.md','experiments/README.md','records/README.md','instruments/README.md']}
    docs.update(ROOT/n['directory']/n.get('document','README.md') for n in graph['nodes'])
    docs.update((ROOT/'experiments').glob('*/README.md'))
    errors=[];count=anchors=0
    for doc in sorted(docs):
        for match in re.finditer(r'\]\(([^\s]+?)\)',doc.read_text()):
            raw=match[1].strip('<>')
            if raw.startswith(('http:','https:','mailto:','#')):continue
            rel,_,fragment=raw.partition('#');target=(doc.parent/unquote(rel)).resolve();count+=1
            if not target.is_relative_to(ROOT) or not target.exists():
                errors.append(f'{doc.relative_to(ROOT)}: missing {raw}');continue
            m=re.fullmatch(r'L(\d+)(?:-L?(\d+))?',fragment)
            if m and target.is_file():
                anchors+=1
                if not 1<=int(m[1])<=int(m[2] or m[1])<=lines(target):errors.append(f'{doc.relative_to(ROOT)}: line bounds {raw}')
    for name in ['graph/score.svg','graph/dependencies.svg']:ET.parse(ROOT/name)
    report=dict(documents=len(docs),links=count,line_anchors=anchors,errors=errors,svg_xml_valid=True)
    (ROOT/'_build/document_validation.json').write_text(json.dumps(report,indent=1)+'\n')
    print(f'{len(docs)} documents, {count} links, {anchors} source anchors, {len(errors)} errors')
    for error in errors[:30]:print(error)
    raise SystemExit(bool(errors))
if __name__=='__main__':main()
