#!/usr/bin/env python3
"""Verify that the packaged reference files match their recorded SHA-256 checksums."""
from pathlib import Path
import csv, hashlib, sys
ROOT=Path(__file__).resolve().parents[1]
manifest=ROOT/'reference_results'/'FILE_MANIFEST.csv'
if not manifest.is_file():
    raise SystemExit('No reference_results/FILE_MANIFEST.csv found')
problems=[]; count=0
with manifest.open(newline='',encoding='utf-8') as f:
    for row in csv.DictReader(f):
        path=ROOT/row['repo_path'];count+=1
        if not path.is_file():problems.append('MISSING '+row['repo_path']);continue
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=row['sha256']:problems.append('CHECKSUM MISMATCH '+row['repo_path'])
print(f'Checked {count} reference files; errors: {len(problems)}')
if problems:
    print('\n'.join(problems));sys.exit(1)
