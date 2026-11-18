#!/usr/bin/env python3
"""Check byte identity of all shipped author source files against original audit."""
from pathlib import Path
import hashlib,csv,sys
ROOT=Path(__file__).resolve().parents[1]
manifest=ROOT/'source_manifest.csv'
errors=[];n=0
with manifest.open(newline='',encoding='utf8') as f:
 for row in csv.DictReader(f):
  n+=1
  p=ROOT/row['repo_path']
  if not p.is_file():errors.append('MISSING '+row['repo_path']);continue
  if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
   errors.append('HASH CHANGED '+row['repo_path'])
print('Audited original Python files:',n,'mismatches:',len(errors))
for e in errors:print(e)
sys.exit(bool(errors))
