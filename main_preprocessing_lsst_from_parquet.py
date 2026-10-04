"""
Seconde partie de main_preprocessing_lsst.py (machine avec torch/GPU).
Lit le parquet produit par make_parquet_lsst.py.

Usage: python main_preprocessing_lsst_from_parquet.py alerts_processed_<TAG>.parquet
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from prepare_data_lsst import prepare_data

PARQUET = sys.argv[1]
DIM = 4
TAG = os.path.basename(PARQUET)[:-len('.parquet')].replace('alerts_processed_', '', 1)

df_alerts = pd.read_parquet(PARQUET)
assert sorted(df_alerts['fid'].unique()) == [1, 2, 3, 4], \
    f"Bands present: {sorted(df_alerts['fid'].unique())} (expected g,r,i,z = 1..4)"

data_obj = prepare_data(df_alerts, dim=DIM, train_size=0.8, train_batch_size=8, convert_to_tensor=True,
                        custom_train_min_time=None, custom_train_max_time=None)

for split in ['train', 'val', 'test']:
    torch.save(data_obj[f'{split}_dataloader'], f'{split}_dataloader_{TAG}.pth')
    torch.save(data_obj[f'{split}_data_combined'], f'{split}_data_combined_{TAG}.pth')
    np.save(f'{split}_objIds_{TAG}.npy', data_obj[f'{split}_objIds'])
np.save(f'total_objIds_{TAG}.npy', data_obj['total_objIds'])
np.save(f'total_common_finkclasses_{TAG}.npy', data_obj['total_common_finkclasses'])
np.save(f'duration_lcs_{TAG}.npy', data_obj['duration_lcs'])
np.save(f'num_datapoints_lcs_{TAG}.npy', data_obj['seq_len_all'])
np.save(f'min_max_magdiffs_{TAG}.npy', data_obj['min_max_magdiffs'])
np.save(f'min_max_mags_{TAG}.npy', data_obj['min_max_mags'])
np.save(f'train_val_test_min_max_times_{TAG}.npy', data_obj['train_val_test_min_max_times'])
np.save(f'train_min_max_times_{TAG}.npy', data_obj['train_min_max_times'])
print('Done. Now run tan_unsupervised.py with --dim 4 '
      f'--train_val_test_min_max_times_filename train_val_test_min_max_times_{TAG}.npy')
