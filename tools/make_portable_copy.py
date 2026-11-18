#!/usr/bin/env python3
"""Create a local runnable copy of author-source files, relocating filesystem paths only.

Only ROOT path constants and two first-stage checkpoint absolute paths are
changed. Source tracked by source_manifest.csv remains unmodified. Large climate inputs are never fabricated or copied; an included small derived
22×22 orography .npy is copied with the source tree. Other raw climate inputs
must be provided lawfully.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

ROOT_LITERAL = 'ROOT = Path("/workspace/SwinIR_HF/Normalizacion")'
CKPT_PREFIX = 'Path("/workspace/SwinIR_HF/Normalizacion/climate_swinir_experiments/'
EXPT = 'climate_swinir_experiments/model_E_cmip_swinunet_topography'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest',type=Path,default=Path('run'),help='New output directory (must not exist)')
    args=parser.parse_args()
    project=Path(__file__).resolve().parents[1]
    output=args.dest.expanduser().resolve()
    if output.exists():parser.error(f'Destination already exists: {output}')
    original=project/'climate_swinir_experiments'
    if not original.is_dir():parser.error('Original scientific scripts directory not found')
    (output/'climate_swinir_experiments').parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(original, output/'climate_swinir_experiments')
    count=0
    for path in (output/'climate_swinir_experiments').rglob('*.py'):
        text=path.read_text(encoding='utf8')
        if ROOT_LITERAL not in text and CKPT_PREFIX not in text:
            continue
        # Script locations differ in nesting depth; derive ROOT from location.
        relative=path.relative_to(output)
        depth=4 if 'paper_validation_final_seed20260813' in relative.parts else 3
        text2=text.replace(ROOT_LITERAL,f'ROOT = Path(__file__).resolve().parents[{depth}]')
        # Dedicated E2 checkpoint literals still use the historical path.
        # Keep this replacement restricted to the 2 specific source lines.
        for variant in ('full','no_fine_topo'):
            suffix=f'checkpoints/phase1_{variant}_seed20260813_40epochs/model_E_phase1_best.pth'
            before=f'Path("/workspace/SwinIR_HF/Normalizacion/{EXPT}/{suffix}")'
            after=f'ROOT / "{EXPT}/{suffix}"'
            text2=text2.replace(before,after)
        if '/workspace/SwinIR_HF/Normalizacion' in text2:
            raise RuntimeError(f'Unpatched workspace reference in {relative}')
        path.write_text(text2,encoding='utf8')
        count += 1
    config=output/EXPT/'configs/config_model_E.json'
    cfg=json.loads(config.read_text(encoding='utf8'))
    for field,old in cfg['raw_data'].items():
        cfg['raw_data'][field]=str(Path('inputs/raw')/Path(old).name)
    cfg['benchmarks']['unet_cmip_cmip_output_path']='data_tomas/unet_outputs/tas_cmip_cmip_output.nc'
    config.write_text(json.dumps(cfg,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    (output/'inputs/raw').mkdir(parents=True,exist_ok=True)
    (output/'climate_swinir_analysis/data').mkdir(parents=True,exist_ok=True)
    (output/'data_tomas/unet_outputs').mkdir(parents=True,exist_ok=True)
    (output/EXPT/'diagnostics').mkdir(parents=True,exist_ok=True)
    (output/'PATH_RELOCATION_NOTE.txt').write_text('Path-only working copy produced by tools/make_portable_copy.py.\n'
        f'Original author source remains in {project}.\n'
        'Raw NetCDF/GHG inputs must be provided separately; nothing is downloaded.\n'
        'Do not confuse path relocation with a verified independent reproduction.\n',encoding='utf8')
    print(f'Created working copy: {output}')
    print(f'Relocated ROOT constants in {count} Python source files; updated local JSON paths.')
    print('Original scientific sources and hashes in repository are unchanged.')
    print('Next: python tools/check_run_inputs.py --root run --mode prepare')

if __name__=='__main__':main()
