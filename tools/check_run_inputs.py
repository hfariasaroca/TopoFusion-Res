#!/usr/bin/env python3
"""Inspect mandatory files for prep, train, residual training or paper evaluation."""
from __future__ import annotations
import argparse,json
from pathlib import Path
EXPT='climate_swinir_experiments/model_E_cmip_swinunet_topography'
DATA='data/torch_dataset_model_E_physical_v1'

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--root',type=Path,default=Path('run'))
    a.add_argument('--mode',choices=('prepare','train','residual','evaluate'),required=True)
    args=a.parse_args()
    root=args.root.expanduser().resolve()
    exp=root/EXPT
    cfg=json.loads((exp/'configs/config_model_E.json').read_text())
    files={}
    def need(name,path):files[name]=path if path.is_absolute() else root/path
    if args.mode=='prepare':
        for key,value in cfg['raw_data'].items():
            if key=='training_ref_path':continue # present in config, unused by current preparation entry point
            need('raw:'+key,Path(value))
        need('fine orography',Path(cfg['paths']['orography_path']))
        need('coarse GCM orography 22x22',Path(EXPT)/'diagnostics/cmip_orog_22x22_first_valid.npy')
    else:
        for name in ('train_inputs.npy','train_targets.npy','time_features_train.npy','ghg_features_train.npy',
                     'val_inputs.npy','val_targets.npy','time_features_val.npy','ghg_features_val.npy',
                     'train_time.npy','val_time.npy','coarse_static_22x22.npy','static_features_64x64.npy',
                     'target_valid_mask_64x64.npy'):
            if args.mode in ('train','residual'):need('prepared:'+name,Path(EXPT)/DATA/name)
        if args.mode=='evaluate':
            for name in ('test_inputs_ssp370.npy','test_targets_ssp370.npy','time_features_test_ssp370.npy',
                        'ghg_features_test_ssp370.npy','test_time_ssp370.npy','coarse_static_22x22.npy',
                        'static_features_64x64.npy','target_valid_mask_64x64.npy'):
                need('prepared:'+name,Path(EXPT)/DATA/name)
        if args.mode=='residual':
            need('fine orography',Path(cfg['paths']['orography_path']))
            for tag in ('full','no_fine_topo'):
                need('phase1:'+tag,Path(EXPT)/f'checkpoints/phase1_{tag}_seed20260813_40epochs/model_E_phase1_best.pth')
        if args.mode=='evaluate':
            for stage,variant,filename in (
                ('phase1','full','model_E_phase1_best.pth'),('phase1','no_fine_topo','model_E_phase1_best.pth'),
                ('phase2_e2','full','model_E2_v2_best.pth'),('phase2_e2','no_fine_topo','model_E2_v2_best.pth')):
                epochs='40epochs' if stage=='phase1' else '12epochs'
                need(f'checkpoint:{stage}-{variant}',Path(EXPT)/f'checkpoints/{stage}_{variant}_seed20260813_{epochs}/{filename}')
            need('U-Net comparison NetCDF',Path('data_tomas/unet_outputs/tas_cmip_cmip_output.nc'))
    missing=0
    for name,p in files.items():
        ok=p.is_file()
        print(('OK      ' if ok else 'MISSING ')+name+' : '+str(p))
        missing += not ok
    print(f'\n{len(files)-missing}/{len(files)} required files found for {args.mode}.')
    if missing:
        print('Obtain inputs legally and place them at these paths before running the corresponding stage.')
        raise SystemExit(1)
if __name__=='__main__':main()
