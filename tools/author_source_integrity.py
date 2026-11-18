#!/usr/bin/env python3
"""Verify the unmodified author-source copies using SHA-256."""
from pathlib import Path
import hashlib,csv,sys
root=Path(__file__).resolve().parents[1]
with (root/'source_manifest.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
bad=[]
for row in rows:
    p=root/row['repo_path']
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
        bad.append(row['repo_path'])
print(f'Code-integrity check: {len(rows)-len(bad)}/{len(rows)} files match v5 source hashes')
if bad: print('Modified/missing:',*bad,sep='\n  ')
sys.exit(bool(bad))
