#!/usr/bin/env python3
"""Read-only static source audit. Distinguishes source completeness from runtime portability."""
from pathlib import Path
import ast, sys, json
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT
required=[EXP/'models'/n for n in (
    'dataset_model_E.py','dataset_model_E_ablation.py',
    'topofusion_base.py','model_e2_residual_head.py'
)]+[EXP/'configs/config_model_E.json']
missing=[p for p in required if not p.is_file()]
print('Core source and configuration:',len(required)-len(missing),'/',len(required))
for p in missing:print('MISSING',p.relative_to(ROOT))
errors=[];abs_paths=[]; scripts=[]
for folder in (EXP/'scripts', EXP/'analysis'/'scripts',EXP/'models'):
    for p in sorted(folder.glob('*.py')):
        scripts.append(p)
        try: ast.parse(p.read_text(encoding='utf-8'))
        except Exception as e: errors.append((p,e))
        if '/workspace/SwinIR_HF/Normalizacion' in p.read_text(encoding='utf-8'):abs_paths.append(p)
for p,e in errors:print('SYNTAX ERROR',p.relative_to(ROOT),e)
print('Parsed:',len(scripts),'Python files; syntax errors:',len(errors))
print('Scripts with historical absolute workspace roots:',len(abs_paths))
for p in abs_paths:print('  PATH-REVIEW',p.relative_to(ROOT))
if not missing:
    cfg=json.loads((EXP/'configs/config_model_E.json').read_text(encoding='utf-8'))
    print('Legacy config training.phase1_epochs:',cfg.get('training',{}).get('phase1_epochs'))
    print('Final published scripts explicitly use phase-1 40 epochs, phase-2 12 epochs')
if missing or errors:
    print('FAIL: missing or syntactically invalid source files')
    sys.exit(1)
print('PASS: core sources are present and parse successfully.')
if abs_paths:print('PENDING: portability fixes and external inputs. No end-to-end reproducibility claim.')
