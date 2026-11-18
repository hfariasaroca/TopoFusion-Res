#!/usr/bin/env python3
"""Read-only report of expected climate data and final checkpoints in a local checkout."""
import json
from pathlib import Path
from argparse import ArgumentParser

parser = ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
args=parser.parse_args()
root=args.root.expanduser().resolve()
exp=root/'climate_swinir_experiments/model_E_cmip_swinunet_topography'
cfg=json.loads((exp/'configs/config_model_E.json').read_text(encoding='utf8'))
print('Repository root:',root)
print('Climate input paths in original config (may need relocation):')
for label, v in cfg['raw_data'].items():
    p=Path(v); p=(root/p if not p.is_absolute() else p)
    print('  ', 'OK' if p.is_file() else 'MISSING',label,':',p)
print('Final checkpoints:')
for rel in (
 'phase1_full_seed20260813_40epochs/model_E_phase1_best.pth',
 'phase1_no_fine_topo_seed20260813_40epochs/model_E_phase1_best.pth',
 'phase2_e2_full_seed20260813_12epochs/model_E2_v2_best.pth',
 'phase2_e2_no_fine_topo_seed20260813_12epochs/model_E2_v2_best.pth',
):
    p=exp/'checkpoints'/rel
    print('  ', 'OK' if p.is_file() else 'MISSING',rel)
print('U-Net reference file:')
p=root/'data_tomas/unet_outputs/tas_cmip_cmip_output.nc'
print('  ', 'OK' if p.is_file() else 'MISSING',p)
