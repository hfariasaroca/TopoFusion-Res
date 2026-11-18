#!/usr/bin/env python3
"""Collect candidate source files for review; does NOT create a public code release.

Run on the author's server. This script never changes the source project.
WARNING: Review the resulting ZIP for sensitive data before sharing it.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import zipfile

# Archive only small, text-based source files. Exclude notebooks and configuration
# contents: the latter frequently contain internal storage locations or secrets.
INCLUDE_SUFFIXES = {'.py', '.sh'}
INCLUDE_FILENAMES = {'requirements.txt', 'pyproject.toml', 'setup.py', 'setup.cfg',
                     'environment.yml', 'environment.yaml'}
SKIP_DIRS = {'.git', '.venv', 'venv', 'env', '__pycache__', '.ipynb_checkpoints',
             'node_modules', 'wandb', 'runs', 'checkpoints', 'datasets', 'data',
             'prediction_cubes', 'archive_versions', 'legacy_not_for_final',
             'publication_figures_previous_versions_20260925',
             'publication_figures_previous_versions_20260926',
             'publication_figures_previous_versions_20260927'}
MAX_BYTES = 1_500_000
# Conservative screen only; cannot reliably detect every secret.
SUSPECT = re.compile(r'(?i)(-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|'
                     r'aws_secret_access_key\s*[=:]|'
                     r'(?:api[_-]?key|secret[_-]?key|password|access[_-]?token)\s*[=:]\s*[\"\']?\S{8,})')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.source.expanduser().resolve()
    dest=args.output.expanduser().resolve()
    if not root.is_dir():
        parser.error(f'Source directory does not exist: {root}')
    if dest.is_relative_to(root):
        parser.error('ZIP output must be outside the source tree')
    files=[]; excluded=[]
    for cur, dirs, names in os.walk(root, followlinks=False):
        dirs[:]=sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith('.'))
        for name in sorted(names):
            p=Path(cur)/name
            if p.is_symlink() or not p.is_file(): continue
            rel=p.relative_to(root).as_posix()
            if (p.suffix.lower() not in INCLUDE_SUFFIXES
                    and p.name.lower() not in INCLUDE_FILENAMES):
                continue
            size=p.stat().st_size
            if size>MAX_BYTES:
                excluded.append((rel, 'oversize',size)); continue
            try:
                raw=p.read_bytes(); body=raw.decode('utf-8')
            except Exception:
                excluded.append((rel, 'not UTF-8 text',size)); continue
            if SUSPECT.search(body):
                excluded.append((rel, 'possible secret; review locally',size)); continue
            imports=[]
            if p.suffix.lower()=='.py':
                try:
                    tree=ast.parse(body)
                    imports=sorted(set((n.module or '') if isinstance(n,ast.ImportFrom)
                        else a.name.split('.')[0]
                        for n in ast.walk(tree) if isinstance(n,(ast.ImportFrom,ast.Import))
                        for a in ([None] if isinstance(n,ast.ImportFrom) else n.names)))
                except SyntaxError:
                    imports=['<parse failed>']
            files.append((rel,raw,size,hashlib.sha256(raw).hexdigest(),';'.join(imports)))
    manifest=[dict(path=rel,bytes=size,sha256=h,imports=im) for rel,_,size,h,im in files]
    dest.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for rel,raw,*_ in files:
            z.writestr('source_candidates/'+rel,raw)
        z.writestr('INVENTORY.json',json.dumps({'source_project_basename':root.name,
                    'included':manifest,'excluded':[
                        {'path':p,'reason':why,'bytes':n} for p,why,n in excluded]},
                    indent=2,ensure_ascii=False))
        buf=io.StringIO(); writer=csv.DictWriter(buf,fieldnames=['path','bytes','sha256','imports'])
        writer.writeheader(); writer.writerows(manifest)
        z.writestr('INVENTORY.csv',buf.getvalue())
        z.writestr('REVIEW_BEFORE_SHARING.txt',
            'Candidate source collection ONLY. Review all included scripts for credentials, '
            'internal IPs, passwords, personal paths, intellectual-property rights and '
            'private dataset references. DO NOT publish this archive as-is.\n')
    print(f'Created: {dest}\nIncluded source files: {len(files)}; excluded candidates: {len(excluded)}')
    print('WARNING: This is a review archive. Inspect contents before sharing or publishing.')

if __name__=='__main__': main()
